"""
model_arch.py — Architecture detection and per-arch KV hook factories.

Supports GPT-2, OPT, LLaMA, Mistral, Falcon, Qwen2, and Phi-2 out of the
box.  New architectures can be registered via ModelArchRegistry.register().

For each architecture we expose:
  - how to find the attention module for layer i
  - how to extract (keys, values) with correct shape (b, n_kv_heads, s, head_dim)
  - whether the model uses Grouped Query Attention (n_kv_heads < n_heads)

Author: Dan Hicks (github.com/danhicks96)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Architecture enum
# ---------------------------------------------------------------------------

class ModelArch(str, Enum):
    GPT2      = "gpt2"
    OPT       = "opt"
    LLAMA     = "llama"       # LLaMA 1/2/3, Mistral, Gemma, …
    FALCON    = "falcon"
    QWEN2     = "qwen2"
    PHI       = "phi"
    UNKNOWN   = "unknown"


# ---------------------------------------------------------------------------
# Architecture descriptor
# ---------------------------------------------------------------------------

@dataclass
class ArchDescriptor:
    """
    Describes how to interact with a specific model architecture.

    Fields
    ------
    arch            : ModelArch enum value
    hook_mode       : "qkv_proj" | "kv_separate" | "generic"
    get_attn_module : function (model, layer_idx) → nn.Module to hook
    split_kv        : function (module_output, model, config) → (k, v)
                      where k, v have shape (batch, n_kv_heads, seq, head_dim)
    n_kv_heads_attr : config attribute for num_key_value_heads
                      (None → equals num_attention_heads, i.e. no GQA)
    aliases         : additional model_type substrings that map to this descriptor
    """
    arch: ModelArch
    hook_mode: str
    get_attn_module: Callable
    split_kv: Callable
    n_kv_heads_attr: Optional[str] = None
    aliases: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Built-in split_kv factories
# ---------------------------------------------------------------------------

def _split_combined_qkv(n_heads_attr: str = "n_head"):
    """For models with a single fused QKV projection (e.g. GPT-2 c_attn)."""
    def _split(out: torch.Tensor, model, cfg) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(out, torch.Tensor):
            out = out[0]
        n_heads = getattr(cfg, n_heads_attr)
        embed_dim = out.shape[-1] // 3
        _, k, v = out.split(embed_dim, dim=-1)      # each (b, s, embed)
        head_dim = embed_dim // n_heads
        b, s, _ = k.shape
        k = k.view(b, s, n_heads, head_dim).transpose(1, 2)
        v = v.view(b, s, n_heads, head_dim).transpose(1, 2)
        return k, v
    return _split


def _split_falcon_qkv():
    """Falcon uses a combined QKV but with a different split ratio."""
    def _split(out: torch.Tensor, model, cfg) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(out, torch.Tensor):
            out = out[0]
        n_heads = cfg.num_attention_heads
        n_kv = getattr(cfg, "num_kv_heads", 1)
        head_dim = cfg.hidden_size // n_heads
        # Falcon: [q(n_heads), k(n_kv), v(n_kv)] concatenated along last dim
        q_dim = n_heads * head_dim
        k_dim = n_kv * head_dim
        q, k, v = out.split([q_dim, k_dim, k_dim], dim=-1)
        b, s, _ = k.shape
        k = k.view(b, s, n_kv, head_dim).transpose(1, 2)
        v = v.view(b, s, n_kv, head_dim).transpose(1, 2)
        return k, v
    return _split


def _split_separate_kv(k_proj_attr: str, v_proj_attr: str):
    """
    For models with separate k_proj and v_proj (LLaMA, Mistral, Qwen2, Phi).

    We hook the attention module itself and capture outputs of k_proj / v_proj
    via sub-hooks.  Returns a factory that produces an (attn_module, hook_fn)
    pair in a special two-hook mode — handled in KVCollector._register_hook().
    """
    def _split(out, model, cfg) -> Tuple[torch.Tensor, torch.Tensor]:
        # Not used in two-hook mode; placeholder for interface consistency.
        raise RuntimeError("_split_separate_kv should not be called directly")
    return _split


# ---------------------------------------------------------------------------
# ModelArchRegistry
# ---------------------------------------------------------------------------

class ModelArchRegistry:
    """
    Registry of known model architectures.

    Usage:
        desc = ModelArchRegistry.detect(model)
        print(desc.arch, desc.hook_mode)
    """

    _registry: List[ArchDescriptor] = []

    @classmethod
    def register(cls, descriptor: ArchDescriptor) -> None:
        cls._registry.insert(0, descriptor)   # newest first

    @classmethod
    def detect(cls, model) -> ArchDescriptor:
        cfg = model.config
        arch_name = getattr(cfg, "model_type", "").lower()
        for desc in cls._registry:
            patterns = [desc.arch.value] + (desc.aliases or [])
            if any(p in arch_name or arch_name in p for p in patterns):
                return desc
        return cls._unknown()

    @classmethod
    def _unknown(cls) -> ArchDescriptor:
        def _get(model, idx):
            raise NotImplementedError(
                "Unknown model architecture. Register it via ModelArchRegistry.register()."
            )
        return ArchDescriptor(
            arch=ModelArch.UNKNOWN,
            hook_mode="generic",
            get_attn_module=_get,
            split_kv=lambda out, m, c: (None, None),
        )


# ---------------------------------------------------------------------------
# Register built-in architectures
# ---------------------------------------------------------------------------

# GPT-2 family (gpt2, distilgpt2)
ModelArchRegistry.register(ArchDescriptor(
    arch=ModelArch.GPT2,
    hook_mode="qkv_proj",
    get_attn_module=lambda model, i: model.transformer.h[i].attn.c_attn,
    split_kv=_split_combined_qkv("n_head"),
    n_kv_heads_attr=None,   # GPT-2: no GQA
))

# OPT family
ModelArchRegistry.register(ArchDescriptor(
    arch=ModelArch.OPT,
    hook_mode="kv_separate",
    get_attn_module=lambda model, i: model.model.decoder.layers[i].self_attn,
    split_kv=_split_separate_kv("k_proj", "v_proj"),
    n_kv_heads_attr=None,
))

# LLaMA / Mistral / Gemma / CodeLlama (all use model.model.layers[i].self_attn)
ModelArchRegistry.register(ArchDescriptor(
    arch=ModelArch.LLAMA,
    hook_mode="kv_separate",
    get_attn_module=lambda model, i: model.model.layers[i].self_attn,
    split_kv=_split_separate_kv("k_proj", "v_proj"),
    n_kv_heads_attr="num_key_value_heads",   # GQA in LLaMA-2 70B, Mistral
    aliases=["mistral", "gemma", "codellama", "stablelm", "internlm"],
))

# Falcon
ModelArchRegistry.register(ArchDescriptor(
    arch=ModelArch.FALCON,
    hook_mode="qkv_proj",
    get_attn_module=lambda model, i: model.transformer.h[i].self_attention.query_key_value,
    split_kv=_split_falcon_qkv(),
    n_kv_heads_attr="num_kv_heads",
))

# Qwen2
ModelArchRegistry.register(ArchDescriptor(
    arch=ModelArch.QWEN2,
    hook_mode="kv_separate",
    get_attn_module=lambda model, i: model.model.layers[i].self_attn,
    split_kv=_split_separate_kv("k_proj", "v_proj"),
    n_kv_heads_attr="num_key_value_heads",
))

# Phi-2 / Phi-3
ModelArchRegistry.register(ArchDescriptor(
    arch=ModelArch.PHI,
    hook_mode="kv_separate",
    get_attn_module=lambda model, i: model.model.layers[i].self_attn,
    split_kv=_split_separate_kv("k_proj", "v_proj"),
    n_kv_heads_attr="num_key_value_heads",
))


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------

def get_n_kv_heads(model) -> int:
    """Return number of KV heads (< n_heads for GQA models, else n_heads)."""
    cfg = model.config
    desc = ModelArchRegistry.detect(model)
    if desc.n_kv_heads_attr:
        return getattr(cfg, desc.n_kv_heads_attr,
                       getattr(cfg, "num_attention_heads", 1))
    return getattr(cfg, "num_attention_heads",
                   getattr(cfg, "n_head", 1))
