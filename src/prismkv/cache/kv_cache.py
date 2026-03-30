"""
kv_cache.py — PrismKVCache: drop-in compressed KV cache for HuggingFace models.

Subclasses transformers.DynamicCache and intercepts update() calls to:
  1. Encode incoming key/value states with StackedPlaneQuantizer
  2. Store compressed int16 codes in _key_codes / _val_codes
  3. Decode back to FP and forward decoded tensors to the parent cache
     so that attention computation and all parent methods work unchanged

Memory savings come from _key_codes/_val_codes being ~3× smaller than FP16:
  FP16 head_dim=64: 2 bytes × 64 = 128 bytes/token/head
  PrismKV 4+4+4 bits, padded to 66: ceil(12/16) × 22 codes × 2 bytes = 44 bytes
  → ~2.9× compression

Usage (HuggingFace generate)
-----------------------------
    from prismkv.cache import PrismKVCache, PrismKVConfig

    cache = PrismKVCache(PrismKVConfig(bits_z=4, bits_r=4, bits_theta=4))
    output = model.generate(input_ids, past_key_values=cache, use_cache=True)

Author: Dan Hicks (github.com/danhicks96)
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Union

import torch

from prismkv.cache.cache_config import PrismKVConfig
from prismkv.cache.dim_aligner import DimAligner
from prismkv.quantizer.stacked_plane import StackedPlaneQuantizer

try:
    from transformers import DynamicCache
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    DynamicCache = object  # sentinel for class definition


def _require_transformers() -> None:
    if not HAS_TRANSFORMERS:
        raise ImportError(
            "transformers is required for PrismKVCache. "
            "Install with: pip install prismkv[cache]  or  pip install transformers"
        )


class PrismKVCache(DynamicCache):
    """
    Compressed KV cache using PrismKV 3-D stacked-plane quantization.

    Drop-in replacement for transformers.DynamicCache.  isinstance checks pass.
    All parent methods (crop, reorder_cache, get_seq_length, etc.) work on the
    decoded FP tensors stored in the parent's DynamicLayer objects.

    The _key_codes and _val_codes lists hold int16-packed PrismKV codes and
    represent the compressed representation.  Use memory_footprint() to compare
    compressed vs full FP16 size.

    Parameters
    ----------
    config : PrismKVConfig
        Quantization configuration.  Defaults to 4+4+4 bits uniform.
    configs : List[PrismKVConfig] | None
        Per-layer configs (length must equal model's num_hidden_layers).
        If provided, overrides `config`.  Used by M7 adaptive bit allocation.
    """

    def __init__(
        self,
        config: Optional[PrismKVConfig] = None,
        configs: Optional[List[PrismKVConfig]] = None,
    ) -> None:
        _require_transformers()
        super().__init__()

        self._default_config = config or PrismKVConfig()
        self._per_layer_configs = configs  # None or list

        # Compressed code storage: one entry per layer
        self._key_codes: List[Optional[torch.Tensor]] = []  # (N, m) int16
        self._val_codes: List[Optional[torch.Tensor]] = []

        # Per-layer quantizers and aligners, initialised lazily on first update()
        self._quantizers: Dict[int, StackedPlaneQuantizer] = {}
        self._aligners: Dict[int, DimAligner] = {}

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Core override
    # ------------------------------------------------------------------

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compress incoming K/V, store codes, decode for attention, update parent.

        key_states   : (batch, n_heads, new_seq_len, head_dim)
        value_states : (batch, n_heads, new_seq_len, head_dim)
        """
        batch, n_heads, new_seq_len, head_dim = key_states.shape
        orig_dtype = key_states.dtype

        q, aligner = self._get_quantizer_and_aligner(layer_idx, head_dim)

        # Flatten to (N, head_dim), pad to (N, padded_dim)
        N = batch * n_heads * new_seq_len
        k_flat = aligner.pad(key_states.reshape(N, head_dim).float())
        v_flat = aligner.pad(value_states.reshape(N, head_dim).float())

        # Encode
        k_codes = q.encode(k_flat)   # (N, m) int64
        v_codes = q.encode(v_flat)

        # Decode
        k_dec = aligner.unpad(q.decode(k_codes)).to(orig_dtype)  # (N, head_dim)
        v_dec = aligner.unpad(q.decode(v_codes)).to(orig_dtype)

        # Store compressed codes (int16 — 12-bit codes fit)
        k_codes_i16 = k_codes.to(torch.int16)
        v_codes_i16 = v_codes.to(torch.int16)

        with self._lock:
            while len(self._key_codes) <= layer_idx:
                self._key_codes.append(None)
                self._val_codes.append(None)

            if self._key_codes[layer_idx] is None:
                self._key_codes[layer_idx] = k_codes_i16
                self._val_codes[layer_idx] = v_codes_i16
            else:
                self._key_codes[layer_idx] = torch.cat(
                    [self._key_codes[layer_idx], k_codes_i16], dim=0
                )
                self._val_codes[layer_idx] = torch.cat(
                    [self._val_codes[layer_idx], v_codes_i16], dim=0
                )

        # Reshape decoded tensors back to (batch, n_heads, new_seq_len, head_dim)
        k_decoded = k_dec.reshape(batch, n_heads, new_seq_len, head_dim)
        v_decoded = v_dec.reshape(batch, n_heads, new_seq_len, head_dim)

        # Forward decoded tensors to parent (handles concatenation with past tokens)
        return super().update(k_decoded, v_decoded, layer_idx, *args, **kwargs)

    # ------------------------------------------------------------------
    # Memory diagnostics
    # ------------------------------------------------------------------

    def memory_footprint(self) -> dict:
        """
        Return memory usage of compressed codes vs equivalent FP16 storage.

        Returns
        -------
        dict with keys:
          'codes_bytes'    : bytes used by _key_codes + _val_codes (int16)
          'fp16_bytes'     : bytes the same data would occupy as FP16
          'compression'    : fp16_bytes / codes_bytes (higher = more compression)
          'n_layers'       : number of layers with cached data
        """
        codes_bytes = 0
        fp16_bytes = 0

        for layer_idx, (kc, vc) in enumerate(
            zip(self._key_codes, self._val_codes)
        ):
            if kc is None:
                continue
            codes_bytes += kc.nbytes + vc.nbytes

            # FP16 equivalent: same number of tokens × head_dim × 2 bytes × 2 (K+V)
            if layer_idx in self._aligners:
                head_dim = self._aligners[layer_idx].original_dim
                n_tokens = kc.shape[0]  # batch * n_heads * seq_len
                fp16_bytes += n_tokens * head_dim * 2 * 2

        compression = fp16_bytes / codes_bytes if codes_bytes > 0 else float("nan")

        return {
            "codes_bytes": codes_bytes,
            "fp16_bytes": fp16_bytes,
            "compression": compression,
            "n_layers": sum(1 for kc in self._key_codes if kc is not None),
        }

    def compression_ratio(self) -> float:
        """Shorthand: FP16 bytes / compressed bytes."""
        return self.memory_footprint()["compression"]

    # ------------------------------------------------------------------
    # Crop / reorder override (keep codes in sync with parent)
    # ------------------------------------------------------------------

    def crop(self, max_length: int) -> None:
        """Crop to max_length tokens — syncs both parent layers and _*_codes."""
        super().crop(max_length)
        # Crop our code lists to match (the parent crops per-layer key tensors)
        # We don't know n_heads/batch from here, so we truncate codes proportionally
        # based on what the parent layer now holds.
        for layer_idx in range(len(self._key_codes)):
            if self._key_codes[layer_idx] is None:
                continue
            if layer_idx < len(self.layers):
                layer = self.layers[layer_idx]
                if layer.is_initialized and layer.keys.shape[-2] < self._key_codes[layer_idx].shape[0]:
                    new_n = layer.keys.shape[-2] * layer.keys.shape[0] * layer.keys.shape[1]
                    self._key_codes[layer_idx] = self._key_codes[layer_idx][:new_n]
                    self._val_codes[layer_idx] = self._val_codes[layer_idx][:new_n]

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder cache for beam search — syncs codes with parent layer reorder."""
        super().reorder_cache(beam_idx)
        # Codes are stored flat; reorder by beam_idx in the batch dimension
        for layer_idx in range(len(self._key_codes)):
            kc = self._key_codes[layer_idx]
            vc = self._val_codes[layer_idx]
            if kc is None:
                continue
            if layer_idx in self._aligners and layer_idx < len(self.layers):
                layer = self.layers[layer_idx]
                if layer.is_initialized:
                    batch, n_heads, seq_len, _ = layer.keys.shape
                    # codes shape: (batch * n_heads * seq_len, m) → reorder batch dim
                    kc2 = kc.reshape(batch, n_heads, seq_len, -1)[beam_idx]
                    vc2 = vc.reshape(batch, n_heads, seq_len, -1)[beam_idx]
                    self._key_codes[layer_idx] = kc2.reshape(-1, kc.shape[-1])
                    self._val_codes[layer_idx] = vc2.reshape(-1, vc.shape[-1])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_config(self, layer_idx: int) -> PrismKVConfig:
        if self._per_layer_configs is not None and layer_idx < len(self._per_layer_configs):
            return self._per_layer_configs[layer_idx]
        return self._default_config

    def _get_quantizer_and_aligner(
        self, layer_idx: int, head_dim: int
    ) -> tuple[StackedPlaneQuantizer, DimAligner]:
        if layer_idx not in self._quantizers:
            cfg = self._get_config(layer_idx)
            aligner = DimAligner(head_dim)
            q = StackedPlaneQuantizer(
                dim=aligner.padded_dim,
                bits_z=cfg.bits_z,
                bits_r=cfg.bits_r,
                bits_theta=cfg.bits_theta,
                seed=cfg.rotation_seed,
            )
            if cfg.codebook_path is not None:
                try:
                    q.load_codebooks(cfg.codebook_path)
                except Exception as e:
                    if not cfg.fallback_to_uniform:
                        raise
                    import warnings
                    warnings.warn(
                        f"PrismKVCache: failed to load codebook '{cfg.codebook_path}': {e}. "
                        "Falling back to uniform polar quantization.",
                        stacklevel=3,
                    )
            self._quantizers[layer_idx] = q
            self._aligners[layer_idx] = aligner

        return self._quantizers[layer_idx], self._aligners[layer_idx]

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fp = self.memory_footprint()
        n_layers = fp["n_layers"]
        comp = fp["compression"]
        return (
            f"PrismKVCache({self._default_config!r}, "
            f"n_layers_cached={n_layers}, "
            f"compression={comp:.1f}× vs FP16)"
        )
