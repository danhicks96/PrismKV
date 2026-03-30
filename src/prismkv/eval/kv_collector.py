"""
kv_collector.py — Extract KV cache tensors from HuggingFace models via forward hooks.

Captures raw (key_states, value_states) per attention layer for a calibration
corpus.  One layer is collected at a time to bound memory usage.

Handles:
  - GPT-2 style combined QKV projections
  - LLaMA / Mistral separate k_proj / v_proj (hooks two sub-modules)
  - Grouped Query Attention (n_kv_heads < n_heads)
  - head_dim not divisible by 3 via zero-padding (GPT-2 64 → 66)

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

from prismkv.eval.model_arch import ModelArchRegistry, get_n_kv_heads


def _check_transformers() -> None:
    if not HAS_TRANSFORMERS:
        raise ImportError(
            "transformers is required for KVCollector. "
            "Install with: pip install prismkv[eval]"
        )


# ---------------------------------------------------------------------------
# Dim alignment helpers
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
    return torch.nn.functional.pad(tensor, (0, 3 - rem))


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
        HuggingFace model identifier, e.g. "gpt2", "meta-llama/Llama-3.2-1B".
    device : str
        Torch device string, e.g. "cpu" or "cuda:0".
    pad_dim : bool
        If True (default), pad head_dim to nearest multiple of 3.

    Notes
    -----
    GQA models (LLaMA-2-70B, Mistral-7B, …) produce KV tensors with shape
    (batch, n_kv_heads, seq, head_dim) where n_kv_heads < n_heads.
    ``self.n_kv_heads`` reflects this; collected vectors are labelled per KV head.
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

        self.original_head_dim: Optional[int] = None
        self.padded_head_dim: Optional[int] = None
        self.n_layers: int = self.model.config.num_hidden_layers
        self.n_kv_heads: int = get_n_kv_heads(self.model)

        self._arch_desc = ModelArchRegistry.detect(self.model)

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

        Returns
        -------
        dict mapping "layer_{i}" → {"keys":   Tensor(n_kv_heads*seq, head_dim),
                                     "values": Tensor(n_kv_heads*seq, head_dim)}
        """
        if layer_indices is None:
            layer_indices = list(range(self.n_layers))

        tokens = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_tokens
        )
        input_ids = tokens["input_ids"].to(self.device)

        results: Dict[str, Dict[str, torch.Tensor]] = {}

        for layer_idx in layer_indices:
            keys_buf: List[torch.Tensor] = []
            vals_buf: List[torch.Tensor] = []

            handles = self._register_hook(layer_idx, keys_buf, vals_buf)
            try:
                with torch.no_grad():
                    self.model(input_ids)
            finally:
                for h in handles:
                    h.remove()

            if not keys_buf or not vals_buf:
                continue

            k = keys_buf[0]   # (batch, n_kv_heads, seq, head_dim)
            v = vals_buf[0]

            if k.ndim == 3:
                k = k.unsqueeze(0)
            if v.ndim == 3:
                v = v.unsqueeze(0)

            _, n_kv_heads, seq_len, head_dim = k.shape
            self.original_head_dim = head_dim

            k_flat = k.squeeze(0).reshape(-1, head_dim).cpu().float()
            v_flat = v.squeeze(0).reshape(-1, head_dim).cpu().float()

            if self.pad_dim:
                k_flat = pad_to_multiple_of_3(k_flat)
                v_flat = pad_to_multiple_of_3(v_flat)
                self.padded_head_dim = k_flat.shape[-1]

            results[f"layer_{layer_idx}"] = {"keys": k_flat, "values": v_flat}

            del keys_buf, vals_buf
            gc.collect()

        return results

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _register_hook(
        self,
        layer_idx: int,
        keys_buf: List[torch.Tensor],
        vals_buf: List[torch.Tensor],
    ) -> List:
        """
        Register forward hook(s) and return a list of handles to remove later.

        Routing:
          qkv_proj    — single combined QKV projection (GPT-2, Falcon)
          kv_separate — separate k_proj and v_proj sub-modules (LLaMA family)
          generic     — hook the whole attention block, scan for 4-D tensors
        """
        desc = self._arch_desc
        mode = desc.hook_mode

        if mode == "qkv_proj":
            module = desc.get_attn_module(self.model, layer_idx)
            cfg = self.model.config

            def hook_qkv(mod, inputs, outputs):
                k, v = desc.split_kv(outputs, self.model, cfg)
                keys_buf.append(k.detach())
                vals_buf.append(v.detach())

            return [module.register_forward_hook(hook_qkv)]

        elif mode == "kv_separate":
            attn_module = desc.get_attn_module(self.model, layer_idx)
            cfg = self.model.config
            n_kv = self.n_kv_heads
            head_dim = cfg.hidden_size // cfg.num_attention_heads

            # Hook k_proj and v_proj directly inside the attention module
            if not hasattr(attn_module, "k_proj") or not hasattr(attn_module, "v_proj"):
                # Fallback to generic if sub-modules not found
                return self._register_generic_hook(attn_module, keys_buf, vals_buf)

            def hook_k(mod, inputs, output):
                b, s, _ = output.shape
                k = output.view(b, s, n_kv, head_dim).transpose(1, 2)
                keys_buf.append(k.detach())

            def hook_v(mod, inputs, output):
                b, s, _ = output.shape
                v = output.view(b, s, n_kv, head_dim).transpose(1, 2)
                vals_buf.append(v.detach())

            return [
                attn_module.k_proj.register_forward_hook(hook_k),
                attn_module.v_proj.register_forward_hook(hook_v),
            ]

        else:
            attn_module = desc.get_attn_module(self.model, layer_idx)
            return self._register_generic_hook(attn_module, keys_buf, vals_buf)

    def _register_generic_hook(
        self,
        module: torch.nn.Module,
        keys_buf: List[torch.Tensor],
        vals_buf: List[torch.Tensor],
    ) -> List:
        """Scan the output tuple for 4-D (b, heads, seq, head_dim) tensors."""
        def hook(mod, inputs, outputs):
            items = outputs if isinstance(outputs, (list, tuple)) else [outputs]
            for item in items:
                if isinstance(item, torch.Tensor) and item.ndim == 4:
                    if not keys_buf:
                        keys_buf.append(item.detach())
                    elif not vals_buf:
                        vals_buf.append(item.detach())
                    if keys_buf and vals_buf:
                        return

        return [module.register_forward_hook(hook)]

    def __repr__(self) -> str:
        return (
            f"KVCollector(model={self.model_name!r}, arch={self._arch_desc.arch.value!r}, "
            f"n_layers={self.n_layers}, n_kv_heads={self.n_kv_heads}, device={self.device!r})"
        )
