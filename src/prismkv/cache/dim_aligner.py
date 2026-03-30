"""
dim_aligner.py — Head-dimension padding for models where head_dim % 3 ≠ 0.

GPT-2 uses head_dim=64, which is not divisible by 3.  DimAligner pads it to
66 (the nearest multiple of 3), then strips the padding after dequantization.

Waste: 2 extra dims / 66 total = 3% bit waste.  Documented as a known
limitation; the CUDA kernel (future) can handle this more efficiently.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class DimAligner:
    """
    Zero-pad / unpad a tensor's last dimension to the nearest multiple of 3.

    Parameters
    ----------
    original_dim : int — the actual head_dim of the model
    """

    def __init__(self, original_dim: int) -> None:
        self.original_dim = original_dim
        self.pad_width = (3 - original_dim % 3) % 3
        self.padded_dim = original_dim + self.pad_width

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        """Append self.pad_width zero columns to the last dimension."""
        if self.pad_width == 0:
            return x
        return F.pad(x, (0, self.pad_width))

    def unpad(self, x: torch.Tensor) -> torch.Tensor:
        """Strip the zero columns added by pad()."""
        if self.pad_width == 0:
            return x
        return x[..., : self.original_dim]

    def __repr__(self) -> str:
        return (
            f"DimAligner(original_dim={self.original_dim}, "
            f"padded_dim={self.padded_dim}, "
            f"waste={self.pad_width / self.padded_dim * 100:.1f}%)"
        )
