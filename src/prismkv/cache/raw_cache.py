"""
raw_cache.py — Framework-agnostic compressed KV cache.

RawKVCache operates entirely on raw PyTorch tensors with no dependency on
HuggingFace, vLLM, or any other inference framework.  Use it in:

  - Custom autoregressive generation loops
  - Research / prototyping with any model
  - As the foundation that framework adapters (HF, vLLM, sidecar) build on

Quick start
-----------
    from prismkv.cache.raw_cache import RawKVCache
    from prismkv.cache.backend import PrismKVBackend
    from prismkv.cache.cache_config import PrismKVConfig

    backend = PrismKVBackend(PrismKVConfig(), head_dim=64)
    cache = RawKVCache(backend)

    # Custom generation loop — one layer shown:
    for step in range(max_new_tokens):
        k_new, v_new = model_attn_layer(x)           # (..., seq_len, head_dim)
        k_full, v_full = cache.update(0, k_new, v_new)   # full context, decompressed
        attn_out = scaled_dot_product_attention(q, k_full, v_full)

    print(cache.memory_footprint())   # {'codes_bytes': ..., 'compression': ...}

Multi-layer usage
-----------------
    backends = {i: PrismKVBackend(cfg, head_dim=64) for i in range(n_layers)}
    cache = RawKVCache(backends)   # dict of backends, one per layer

    # During forward pass:
    k_full, v_full = cache.update(layer_idx, k_new, v_new)

Per-layer bit configs
---------------------
    from prismkv.cache.cache_config import PrismKVConfig
    configs = [PrismKVConfig(bits_z=3, bits_r=3, bits_theta=3) if i < 6
               else PrismKVConfig() for i in range(12)]
    backends = {i: PrismKVBackend(configs[i], head_dim=64) for i in range(12)}
    cache = RawKVCache(backends)

Author: Dan Hicks (github.com/danhicks96)
"""

from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple, Union

import torch

from prismkv.cache.backend import CacheBackend, PrismKVBackend
from prismkv.cache.cache_config import PrismKVConfig


class RawKVCache:
    """
    Framework-agnostic compressed KV cache.

    Stores compressed int16 codes internally and returns decompressed float
    tensors on read.  No inheritance from any framework base class.

    Parameters
    ----------
    backend : CacheBackend | PrismKVBackend | Dict[int, CacheBackend]
        - A single backend: used for all layers (requires consistent head_dim).
        - A dict keyed by layer_idx: enables per-layer configs (e.g. M7
          adaptive bit allocation) or different head_dims per layer.
    """

    def __init__(
        self,
        backend: Union[CacheBackend, Dict[int, CacheBackend]],
    ) -> None:
        if isinstance(backend, dict):
            self._backends: Dict[int, CacheBackend] = backend
            self._default_backend: Optional[CacheBackend] = None
        else:
            self._backends = {}
            self._default_backend = backend

        # Compressed storage: layer_idx → (k_codes, v_codes)
        # k_codes shape: (batch * n_heads * seq_len, n_groups) int16
        self._k_codes: Dict[int, torch.Tensor] = {}
        self._v_codes: Dict[int, torch.Tensor] = {}

        # Cached shapes for memory accounting
        self._shapes: Dict[int, Tuple[int, int, int]] = {}  # (batch, n_heads, seq_len)

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress and append new K/V tokens; return full decompressed context.

        Parameters
        ----------
        layer_idx : int
        k         : (..., seq_len, head_dim) float — any leading dims accepted
        v         : (..., seq_len, head_dim) float

        Returns
        -------
        k_full : (..., total_seq_len, head_dim) float32  — full accumulated context
        v_full : (..., total_seq_len, head_dim) float32

        The leading shape and dtype of k/v are preserved in the output.
        """
        orig_shape = k.shape       # (..., seq_len, head_dim)
        head_dim = orig_shape[-1]
        seq_len = orig_shape[-2]
        leading = orig_shape[:-2]  # e.g. (batch, n_heads)
        N_new = 1
        for d in leading:
            N_new *= d
        N_new *= seq_len

        orig_dtype = k.dtype
        backend = self._get_backend(layer_idx, head_dim)

        k_flat = k.reshape(N_new, head_dim)
        v_flat = v.reshape(N_new, head_dim)

        k_codes_new, v_codes_new = backend.compress(k_flat, v_flat)

        with self._lock:
            if layer_idx not in self._k_codes:
                self._k_codes[layer_idx] = k_codes_new
                self._v_codes[layer_idx] = v_codes_new
                total_seq = seq_len
            else:
                self._k_codes[layer_idx] = torch.cat(
                    [self._k_codes[layer_idx], k_codes_new], dim=0
                )
                self._v_codes[layer_idx] = torch.cat(
                    [self._v_codes[layer_idx], v_codes_new], dim=0
                )
                prev_total = self._k_codes[layer_idx].shape[0]
                total_seq = prev_total // (N_new // seq_len)

            # Record leading shape for memory accounting
            if leading:
                self._shapes[layer_idx] = (*leading, seq_len)  # type: ignore[assignment]

        # Decompress full sequence
        k_all, v_all = backend.decompress(
            self._k_codes[layer_idx], self._v_codes[layer_idx]
        )

        # Reshape back to (..., total_seq_len, head_dim)
        total_n = self._k_codes[layer_idx].shape[0]
        total_seq_len = total_n // (N_new // seq_len)
        out_shape = (*leading, total_seq_len, head_dim)
        k_full = k_all.reshape(out_shape).to(orig_dtype)
        v_full = v_all.reshape(out_shape).to(orig_dtype)

        return k_full, v_full

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return full decompressed K/V for a layer without updating.

        Raises KeyError if layer_idx has not been updated yet.
        """
        if layer_idx not in self._k_codes:
            raise KeyError(f"Layer {layer_idx} has no cached data.")
        backend = self._get_backend(layer_idx)
        return backend.decompress(self._k_codes[layer_idx], self._v_codes[layer_idx])

    def get_codes(
        self, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return raw int16 codes (k_codes, v_codes) for a layer."""
        if layer_idx not in self._k_codes:
            raise KeyError(f"Layer {layer_idx} has no cached data.")
        return self._k_codes[layer_idx], self._v_codes[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """
        Return number of cached tokens for a layer.

        Works correctly for both single-head and multi-head shapes.
        """
        if layer_idx not in self._k_codes:
            return 0
        total_n = self._k_codes[layer_idx].shape[0]
        if layer_idx in self._shapes:
            shape = self._shapes[layer_idx]
            n_heads_x_batch = 1
            for d in shape[:-1]:
                n_heads_x_batch *= d
            return total_n // n_heads_x_batch
        return total_n  # single-head case

    def clear(self, layer_idx: Optional[int] = None) -> None:
        """
        Clear cached codes.

        Parameters
        ----------
        layer_idx : int or None — clear specific layer, or all layers if None.
        """
        with self._lock:
            if layer_idx is None:
                self._k_codes.clear()
                self._v_codes.clear()
                self._shapes.clear()
            else:
                self._k_codes.pop(layer_idx, None)
                self._v_codes.pop(layer_idx, None)
                self._shapes.pop(layer_idx, None)

    @property
    def cached_layers(self) -> list:
        """List of layer indices with cached data."""
        return sorted(self._k_codes.keys())

    # ------------------------------------------------------------------
    # Memory diagnostics
    # ------------------------------------------------------------------

    def memory_footprint(self) -> dict:
        """
        Return compressed vs FP16 memory breakdown.

        Returns
        -------
        dict:
            codes_bytes : bytes used by int16 codes (K + V, all layers)
            fp16_bytes  : equivalent FP16 storage
            compression : fp16_bytes / codes_bytes
            n_layers    : number of layers with data
        """
        codes_bytes = 0
        fp16_bytes = 0

        for layer_idx in self._k_codes:
            kc = self._k_codes[layer_idx]
            vc = self._v_codes[layer_idx]
            codes_bytes += kc.nbytes + vc.nbytes

            backend = self._get_backend(layer_idx)
            n_tokens = kc.shape[0]
            head_dim = backend.head_dim
            fp16_bytes += n_tokens * head_dim * 2 * 2  # K+V, 2 bytes each

        compression = fp16_bytes / codes_bytes if codes_bytes > 0 else float("nan")
        return {
            "codes_bytes": codes_bytes,
            "fp16_bytes": fp16_bytes,
            "compression": compression,
            "n_layers": len(self._k_codes),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_backend(
        self, layer_idx: int, head_dim: Optional[int] = None
    ) -> CacheBackend:
        if layer_idx in self._backends:
            return self._backends[layer_idx]
        if self._default_backend is not None:
            return self._default_backend
        # Auto-create from default config if head_dim is provided
        if head_dim is not None:
            backend = PrismKVBackend(PrismKVConfig(), head_dim=head_dim)
            self._backends[layer_idx] = backend
            return backend
        raise KeyError(
            f"No backend for layer {layer_idx}. Pass a backend dict or "
            "ensure the default backend is set."
        )

    def __repr__(self) -> str:
        fp = self.memory_footprint()
        return (
            f"RawKVCache(n_layers={fp['n_layers']}, "
            f"compression={fp['compression']:.1f}× vs FP16)"
        )
