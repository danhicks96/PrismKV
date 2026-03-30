"""
kv_collector.py — Extract KV cache tensors from HuggingFace models via forward hooks.

Captures raw (key_states, value_states) per attention layer for a calibration
corpus.  One layer is collected at a time to bound memory usage.

Handles head_dim not divisible by 3 via zero-padding (e.g. GPT-2 head_dim=64 → 66).

Author: Dan Hicks (github.com/danhicks96)
"""

from __future__ import annotations

import gc
from typing import Dict, List, Optional, Tuple

import torch

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def _check_transformers() -> None:
    if not HAS_TRANSFORMERS:
        raise ImportError(
            "transformers is required for KVCollector. "
            "Install with: pip install prismkv[eval]"
        )


# ---------------------------------------------------------------------------
# Dim alignment helper
# ---------------------------------------------------------------------------

def pad_to_multiple_of_3(tensor: torch.Tensor) -> torch.Tensor:
    """
    Pad the last dimension to the nearest multiple of 3.

    GPT-2 head_dim=64 → padded to 66 (2 zero dims, 3% bit waste).
    If already divisible by 3, returns the tensor unchanged.
    """
    d = tensor.shape[-1]
    rem = d % 3
    if rem == 0:
        return tensor
    pad = 3 - rem
    return torch.nn.functional.pad(tensor, (0, pad))


def unpad_from_multiple_of_3(tensor: torch.Tensor, original_dim: int) -> torch.Tensor:
    """Strip the zero-padding added by pad_to_multiple_of_3."""
    return tensor[..., :original_dim]


# ---------------------------------------------------------------------------
# KVCollector
# ---------------------------------------------------------------------------

class KVCollector:
    """
    Collect key and value tensors from a HuggingFace causal LM.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, e.g. "gpt2" or "facebook/opt-125m".
    device : str
        Torch device string, e.g. "cpu" or "cuda:0".
    pad_dim : bool
        If True (default), pad head_dim to nearest multiple of 3 so the output
        is usable directly with StackedPlaneQuantizer.

    Usage
    -----
        collector = KVCollector("gpt2")
        result = collector.collect(text, layer_indices=[0, 1, 2])
        # result["layer_0"]["keys"]   shape: (n_tokens, head_dim_padded)
        # result["layer_0"]["values"] shape: (n_tokens, head_dim_padded)
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        pad_dim: bool = True,
    ) -> None:
        _check_transformers()
        self.model_name = model_name
        self.device = device
        self.pad_dim = pad_dim

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        ).to(device)
        self.model.eval()

        # Will be populated after first collect() call
        self.original_head_dim: Optional[int] = None
        self.padded_head_dim: Optional[int] = None
        self.n_layers: int = self.model.config.num_hidden_layers

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def collect(
        self,
        text: str,
        layer_indices: Optional[List[int]] = None,
        max_tokens: int = 4096,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Run the model on text and collect KV tensors per layer.

        Processes one layer at a time to avoid accumulating all layers
        simultaneously.

        Parameters
        ----------
        text         : calibration text
        layer_indices: which layers to collect (default: all)
        max_tokens   : truncate input to this many tokens

        Returns
        -------
        dict mapping "layer_{i}" → {"keys": Tensor(N_heads*T, head_dim),
                                     "values": Tensor(N_heads*T, head_dim)}
        where N_heads*T = number of (head, token) pairs collected.
        """
        if layer_indices is None:
            layer_indices = list(range(self.n_layers))

        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
        input_ids = tokens["input_ids"].to(self.device)

        results: Dict[str, Dict[str, torch.Tensor]] = {}

        for layer_idx in layer_indices:
            keys_buf: List[torch.Tensor] = []
            vals_buf: List[torch.Tensor] = []

            hook_handle = self._register_hook(layer_idx, keys_buf, vals_buf)
            try:
                with torch.no_grad():
                    self.model(input_ids)
            finally:
                hook_handle.remove()

            if not keys_buf:
                continue

            # Stack and reshape: (1, n_heads, seq_len, head_dim) → (n_heads*seq_len, head_dim)
            k = keys_buf[0].squeeze(0)   # (n_heads, seq_len, head_dim)
            v = vals_buf[0].squeeze(0)

            n_heads, seq_len, head_dim = k.shape
            self.original_head_dim = head_dim

            k_flat = k.reshape(-1, head_dim).cpu().float()   # (n_heads*seq_len, head_dim)
            v_flat = v.reshape(-1, head_dim).cpu().float()

            if self.pad_dim:
                k_flat = pad_to_multiple_of_3(k_flat)
                v_flat = pad_to_multiple_of_3(v_flat)
                self.padded_head_dim = k_flat.shape[-1]

            results[f"layer_{layer_idx}"] = {"keys": k_flat, "values": v_flat}

            # Free hook buffers before next layer
            del keys_buf, vals_buf
            gc.collect()

        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _register_hook(
        self,
        layer_idx: int,
        keys_buf: List[torch.Tensor],
        vals_buf: List[torch.Tensor],
    ):
        """
        Register a forward hook to capture key and value tensors.

        Transformers 5.x changed GPT-2 to no longer expose (key, value) in
        the attention output tuple.  We instead hook the QKV projection layer
        (c_attn for GPT-2) and split out K and V ourselves.  For models that
        do expose KV in the attention output, we fall back to scanning the
        output for 4-D tensors.
        """
        hook_module, hook_mode = self._get_hook_target(layer_idx)

        if hook_mode == "qkv_proj":
            # c_attn / q_kvproj outputs (batch, seq, 3*embed_dim) or (batch, seq, embed_dim)
            # We reconstruct K and V then reshape to (batch, n_heads, seq, head_dim).
            n_heads = self.model.config.n_head if hasattr(self.model.config, "n_head") else self.model.config.num_attention_heads

            def hook(module, inputs, outputs):
                out = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
                embed_dim = out.shape[-1] // 3
                _, k, v = out.split(embed_dim, dim=-1)      # each (batch, seq, embed_dim)
                head_dim = embed_dim // n_heads
                b, s, _ = k.shape
                k = k.view(b, s, n_heads, head_dim).transpose(1, 2)  # (b, n_heads, s, head_dim)
                v = v.view(b, s, n_heads, head_dim).transpose(1, 2)
                keys_buf.append(k.detach())
                vals_buf.append(v.detach())

        else:
            # Generic: scan output tuple for 4-D tensors (batch, heads, seq, head_dim)
            def hook(module, inputs, outputs):
                items = outputs if isinstance(outputs, (list, tuple)) else [outputs]
                for item in items:
                    if isinstance(item, torch.Tensor) and item.ndim == 4:
                        if not keys_buf:
                            keys_buf.append(item.detach())
                        elif not vals_buf:
                            vals_buf.append(item.detach())
                        if keys_buf and vals_buf:
                            return
                    elif isinstance(item, (list, tuple)):
                        for subitem in item:
                            if isinstance(subitem, torch.Tensor) and subitem.ndim == 4:
                                if not keys_buf:
                                    keys_buf.append(subitem.detach())
                                elif not vals_buf:
                                    vals_buf.append(subitem.detach())
                                if keys_buf and vals_buf:
                                    return

        return hook_module.register_forward_hook(hook)

    def _get_hook_target(self, layer_idx: int) -> Tuple[torch.nn.Module, str]:
        """
        Return (module_to_hook, mode) where mode is 'qkv_proj' or 'generic'.

        'qkv_proj' mode: hook the linear QKV projection and split K/V from it.
        'generic' mode:  hook the attention block and scan output for 4-D tensors.
        """
        model = self.model

        # GPT-2 style: combined QKV Conv1D layer
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            c_attn = model.transformer.h[layer_idx].attn.c_attn
            return c_attn, "qkv_proj"

        # OPT style
        if hasattr(model, "model") and hasattr(model.model, "decoder"):
            return model.model.decoder.layers[layer_idx].self_attn, "generic"

        # LLaMA / Mistral style
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers[layer_idx].self_attn, "generic"

        raise NotImplementedError(
            f"Cannot locate attention module for layer {layer_idx} in {self.model_name}. "
            "Add support in KVCollector._get_hook_target()."
        )

    def __repr__(self) -> str:
        return (
            f"KVCollector(model={self.model_name!r}, "
            f"n_layers={self.n_layers}, device={self.device!r})"
        )
