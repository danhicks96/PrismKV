"""
backend.py — Framework-agnostic CacheBackend protocol + PrismKVBackend implementation.

The CacheBackend protocol defines the minimal interface any inference engine needs
to interact with PrismKV compression.  The quantizer math lives entirely in
PrismKVBackend; all framework-specific wiring (HF, vLLM, llama-cpp, custom loops)
is in separate adapter modules.

Protocol
--------
    compress(k, v)           → (k_codes, v_codes)   raw tensors → int16 codes
    decompress(k_codes, v_codes) → (k, v)           int16 codes → float tensors

Shapes
------
    k, v        : (N, head_dim)          float32 — N = batch × heads × seq_len
    k_codes     : (N, head_dim // 3)     int16   — one code per triplet group
    v_codes     : (N, head_dim // 3)     int16

Usage
-----
    from prismkv.cache.backend import PrismKVBackend
    from prismkv.cache.cache_config import PrismKVConfig

    backend = PrismKVBackend(PrismKVConfig(bits_z=4, bits_r=4, bits_theta=4), head_dim=64)
    k_codes, v_codes = backend.compress(k, v)   # k, v: (N, 64)
    k_hat, v_hat    = backend.decompress(k_codes, v_codes)

Author: Dan Hicks (github.com/danhicks96)
"""

from __future__ import annotations

from typing import Tuple

import torch

from prismkv.cache.cache_config import PrismKVConfig
from prismkv.cache.dim_aligner import DimAligner
from prismkv.quantizer.stacked_plane import StackedPlaneQuantizer


# ---------------------------------------------------------------------------
# Protocol (structural subtyping — no ABC inheritance required)
# ---------------------------------------------------------------------------


class CacheBackend:
    """
    Protocol / base class for PrismKV cache backends.

    Concrete implementations only need to implement ``compress`` and
    ``decompress``.  Subclassing is optional — duck typing works too.

    Any inference framework adapter (HuggingFace, vLLM, llama-cpp, custom)
    wraps one or more ``CacheBackend`` instances and calls these two methods.

    Shape contract
    --------------
    Both methods operate on **2-D flat tensors**: all batch, head, and
    sequence dimensions are merged into the leading ``N`` axis.  Callers
    are responsible for reshaping before calling and after returning.

        k, v        : (N, head_dim)      float32
        k_codes     : (N, n_groups)      int16    where n_groups = padded_dim // 3
        v_codes     : (N, n_groups)      int16
    """

    def compress(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress a (key, value) pair to int16 codes.

        Parameters
        ----------
        k : (N, head_dim) float32
        v : (N, head_dim) float32

        Returns
        -------
        k_codes : (N, n_groups) int16
        v_codes : (N, n_groups) int16
        """
        raise NotImplementedError

    def decompress(
        self, k_codes: torch.Tensor, v_codes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompress int16 codes to float tensors.

        Parameters
        ----------
        k_codes : (N, n_groups) int16
        v_codes : (N, n_groups) int16

        Returns
        -------
        k : (N, head_dim) float32
        v : (N, head_dim) float32
        """
        raise NotImplementedError

    @property
    def config(self) -> PrismKVConfig:
        """Return the quantization configuration."""
        raise NotImplementedError

    @property
    def head_dim(self) -> int:
        """Return the original (unpadded) head dimension."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------


class PrismKVBackend(CacheBackend):
    """
    Concrete CacheBackend backed by StackedPlaneQuantizer.

    This is the single implementation of the compression math.  All
    framework adapters (HF, vLLM, sidecar, custom loops) instantiate this
    and call compress / decompress.

    Parameters
    ----------
    config   : PrismKVConfig — bits_z, bits_r, bits_theta, codebook_path, etc.
    head_dim : int           — original model head dimension (before padding)
    device   : str           — "cpu" or "cuda" (default "cpu")

    Examples
    --------
    Standalone (no framework):

        backend = PrismKVBackend(PrismKVConfig(), head_dim=64)
        k_codes, v_codes = backend.compress(k, v)   # k: (N, 64) float32
        k_hat, v_hat    = backend.decompress(k_codes, v_codes)

    Layer-by-layer (custom autoregressive loop):

        backends = [PrismKVBackend(cfg, head_dim=64) for _ in range(n_layers)]
        k_codes, v_codes = backends[layer_idx].compress(k.reshape(-1, 64), v.reshape(-1, 64))
    """

    def __init__(
        self,
        config: PrismKVConfig,
        head_dim: int,
        device: str = "cpu",
    ) -> None:
        self._config = config
        self._device = device
        self._aligner = DimAligner(head_dim)

        self._quantizer = StackedPlaneQuantizer(
            dim=self._aligner.padded_dim,
            bits_z=config.bits_z,
            bits_r=config.bits_r,
            bits_theta=config.bits_theta,
            seed=config.rotation_seed,
        )

        if config.codebook_path is not None:
            try:
                self._quantizer.load_codebooks(config.codebook_path)
            except Exception as e:
                if not config.fallback_to_uniform:
                    raise
                import warnings
                warnings.warn(
                    f"PrismKVBackend: failed to load codebook '{config.codebook_path}': {e}. "
                    "Falling back to uniform polar quantization.",
                    stacklevel=2,
                )

    # ------------------------------------------------------------------
    # CacheBackend interface
    # ------------------------------------------------------------------

    def compress(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress (N, head_dim) float32 tensors to (N, n_groups) int16 codes.

        Pads head_dim to nearest multiple of 3 internally.
        """
        k_f = self._aligner.pad(k.float())
        v_f = self._aligner.pad(v.float())
        k_codes = self._quantizer.encode(k_f).to(torch.int16)
        v_codes = self._quantizer.encode(v_f).to(torch.int16)
        return k_codes, v_codes

    def decompress(
        self, k_codes: torch.Tensor, v_codes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompress (N, n_groups) int16 codes to (N, head_dim) float32.

        Strips padding back to original head_dim.
        """
        k = self._aligner.unpad(self._quantizer.decode(k_codes.to(torch.int64)))
        v = self._aligner.unpad(self._quantizer.decode(v_codes.to(torch.int64)))
        return k, v

    @property
    def config(self) -> PrismKVConfig:
        return self._config

    @property
    def head_dim(self) -> int:
        return self._aligner.original_dim

    @property
    def padded_dim(self) -> int:
        return self._aligner.padded_dim

    @property
    def n_groups(self) -> int:
        """Number of triplet groups per token (= padded_dim // 3)."""
        return self._aligner.padded_dim // 3

    def __repr__(self) -> str:
        return (
            f"PrismKVBackend(head_dim={self.head_dim}, "
            f"padded={self.padded_dim}, "
            f"{self._config.bits_per_dim:.1f}bits/dim, "
            f"{self._config.compression_vs_fp16:.1f}× vs FP16)"
        )
