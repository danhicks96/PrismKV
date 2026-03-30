"""
bias_correction.py — Empirical per-z-bin bias correction for StackedPlaneQuantizer.

For each z-bin, the quantizer introduces a systematic bias:
    E[v_hat | i_z] − E[v | i_z]

This is estimated on a holdout split of calibration data and stored as a
(bins_z, 3) float32 table (one 3-vector per z-bin: Δz, Δx, Δy).

At decode time: apply  v_hat_corrected = v_hat − bias_table[i_z]  to each
triplet.  Correction is clipped to ±delta_z/2 to prevent overcorrection on
out-of-distribution inputs.

Limitation vs QJL
-----------------
This corrects vector-level mean bias only.  True QJL-style bias correction
for attention scores  q·k  requires Q at decode time (unavailable in cache).
That is v2.4+ work.  This module provides the best achievable correction from
the quantizer side alone.

Author: Dan Hicks (github.com/danhicks96)
"""

from __future__ import annotations

import math
from typing import Optional

import torch


class BiasTable:
    """
    Per-z-bin mean reconstruction bias in triplet (z, x, y) space.

    Shape: (bins_z, 3) float32.  Row i stores  E[v_hat | i_z=i] − E[v | i_z=i]
    for the three triplet coordinates.

    Parameters
    ----------
    table  : Tensor shape (bins_z, 3) — bias vectors
    delta_z: float — z bin width (used to clip corrections to ±delta_z/2)
    """

    def __init__(self, table: torch.Tensor, delta_z: float) -> None:
        assert table.ndim == 2 and table.shape[1] == 3, (
            f"Expected shape (bins_z, 3), got {tuple(table.shape)}"
        )
        self.table = table.float()           # (bins_z, 3)
        self.bins_z = table.shape[0]
        self.delta_z = delta_z
        self._clip = delta_z / 2.0

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def apply(
        self,
        rotated_q: torch.Tensor,
        i_z: torch.Tensor,
        z_idx: torch.Tensor,
        x_idx: torch.Tensor,
        y_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Subtract per-z-bin bias from a reconstructed rotated tensor.

        Parameters
        ----------
        rotated_q : Tensor shape (N, dim) — decoded vectors in rotation space
        i_z       : Tensor shape (N, m) long — z-bin index per group
        z_idx, x_idx, y_idx : Tensor shape (m,) — group index tensors

        Returns
        -------
        corrected : Tensor shape (N, dim) — bias-subtracted rotated vectors
        """
        N, dim = rotated_q.shape
        corrected = rotated_q.clone()

        # Lookup bias for each (vector, group) pair: (N, m, 3)
        bias = self.table[i_z]                              # (N, m, 3)
        bias = bias.clamp(-self._clip, self._clip)

        # Apply: subtract bias from each triplet coordinate
        corrected[:, z_idx] -= bias[..., 0]
        corrected[:, x_idx] -= bias[..., 1]
        corrected[:, y_idx] -= bias[..., 2]

        return corrected

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def max_abs_bias_per_dim(self) -> float:
        """Maximum absolute bias value across all z-bins and dimensions."""
        return self.table.abs().max().item()

    def mean_abs_bias(self) -> float:
        return self.table.abs().mean().item()

    def __repr__(self) -> str:
        return (
            f"BiasTable(bins_z={self.bins_z}, "
            f"max_abs={self.max_abs_bias_per_dim():.4f}, "
            f"mean_abs={self.mean_abs_bias():.4f})"
        )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_bias(
    quantizer,                    # StackedPlaneQuantizer
    vectors: torch.Tensor,        # (N, dim) calibration vectors
    holdout_fraction: float = 0.2,
    seed: int = 0,
) -> BiasTable:
    """
    Estimate per-z-bin mean reconstruction bias using a holdout split.

    Parameters
    ----------
    quantizer        : StackedPlaneQuantizer — must be calibrated before calling
    vectors          : Tensor shape (N, dim) — raw (unrotated) calibration vectors
    holdout_fraction : fraction withheld for bias estimation (default 0.2)
    seed             : RNG seed for train/holdout split

    Returns
    -------
    BiasTable
    """
    N = vectors.shape[0]
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=gen)

    n_holdout = max(1, int(N * holdout_fraction))
    holdout_idx = perm[:n_holdout]
    holdout = vectors[holdout_idx]                             # (H, dim)

    H = holdout.shape[0]
    bins_z = quantizer.bins_z

    # Rotate holdout
    rotated = holdout @ quantizer.R.T                         # (H, dim)

    # Get z-bin assignments for each group
    z = rotated[:, quantizer.z_idx].float()                   # (H, m)
    delta_z = (quantizer.z_max - quantizer.z_min) / bins_z
    i_z = ((z - quantizer.z_min) / delta_z).floor().long().clamp(0, bins_z - 1)  # (H, m)

    # Encode and decode holdout (using current quantizer, which may have learned CBs)
    codes = quantizer.encode(holdout)                          # (H, m)
    recon = quantizer.decode(codes)                            # (H, dim)

    # Bias = mean(recon − truth) per (z-bin, coordinate) in rotated space
    rotated_recon = recon @ quantizer.R.T                     # (H, dim)
    diff_rot = rotated_recon - rotated                        # (H, dim)

    # Accumulate per-z-bin mean bias in triplet space
    bias_table = torch.zeros(bins_z, 3)
    counts = torch.zeros(bins_z)

    # Flatten (H, m) → (H*m,) for scatter
    i_z_flat = i_z.reshape(-1)                                # (H*m,)
    dz_flat  = diff_rot[:, quantizer.z_idx].reshape(-1)       # (H*m,)
    dx_flat  = diff_rot[:, quantizer.x_idx].reshape(-1)
    dy_flat  = diff_rot[:, quantizer.y_idx].reshape(-1)

    counts.scatter_add_(0, i_z_flat, torch.ones(H * quantizer.m))
    bias_table[:, 0].scatter_add_(0, i_z_flat, dz_flat)
    bias_table[:, 1].scatter_add_(0, i_z_flat, dx_flat)
    bias_table[:, 2].scatter_add_(0, i_z_flat, dy_flat)

    nonempty = counts > 0
    bias_table[nonempty] /= counts[nonempty].unsqueeze(1)

    return BiasTable(bias_table, delta_z=delta_z)
