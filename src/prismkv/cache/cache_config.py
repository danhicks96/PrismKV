"""
cache_config.py — Configuration dataclass for PrismKVCache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class PrismKVConfig:
    """
    Configuration for PrismKVCache.

    Parameters
    ----------
    bits_z     : bits for coarse z quantization (default 4 → 16 bins)
    bits_r     : bits for (x,y) radius (default 4 → 16 bins)
    bits_theta : bits for (x,y) angle (default 4 → 16 bins)
    codebook_path : path to a .npz codebook file for the learned v2 path
                    (None = uniform polar fallback)
    fallback_to_uniform : if codebook_path load fails, silently fall back to
                          uniform rather than raising (default True)
    rotation_seed : RNG seed for the global rotation matrix (default 42)
    """
    bits_z: int = 4
    bits_r: int = 4
    bits_theta: int = 4
    codebook_path: Optional[str] = None
    fallback_to_uniform: bool = True
    rotation_seed: int = 42

    @property
    def bits_per_dim(self) -> float:
        return (self.bits_z + self.bits_r + self.bits_theta) / 3

    @property
    def compression_vs_fp16(self) -> float:
        """Theoretical memory reduction vs FP16 storage."""
        return 16 / self.bits_per_dim

    def __repr__(self) -> str:
        return (
            f"PrismKVConfig(bits_z={self.bits_z}, bits_r={self.bits_r}, "
            f"bits_theta={self.bits_theta}, {self.bits_per_dim:.1f} bits/dim, "
            f"{self.compression_vs_fp16:.1f}× vs FP16)"
        )
