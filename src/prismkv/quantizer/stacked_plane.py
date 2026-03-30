"""
stacked_plane.py — 3-D Stacked-Plane KV Cache Quantizer (PrismKV)

Extension of the 2-D polar baseline that captures cross-plane relationships
by grouping KV vector dimensions into triplets and conditioning the 2-D polar
quantization of (x, y) on a coarse quantization of a third coordinate z.

Triplet layout (after global rotation):
    group k  →  z at index 3k,  x at index 3k+1,  y at index 3k+2

For each group:
  1. Quantize z coarsely into B_z bins  → index i_z
  2. Quantize (x, y) in polar form      → indices (i_r, i_theta)
  3. Pack all three into one int64 code per group

At the same bits-per-dimension budget as the 2-D baseline (e.g., B_z=4,
B_r=4, B_theta=4 → 12 bits / 3 dims = 4.0 bits/dim), the conditional
structure allows per-z-slice codebook adaptation in v2 (learned tables).
v1 uses uniform tables; the architecture is established here.

Author: Dan Hicks (github.com/danhicks96)
Repo:   https://github.com/danhicks96/PrismKV
First published: 2026-03-30
"""

import math
import torch

from prismkv.utils import make_rotation


class StackedPlaneQuantizer:
    """
    Training-free 3-D conditional stacked-plane quantizer for KV cache vectors.

    Parameters
    ----------
    dim      : int   — vector dimension; must satisfy dim % 3 == 0
    bits_z   : int   — bits for coarse z quantization (default 4 → 16 bins)
    bits_r   : int   — bits for (x,y) radius (default 4 → 16 bins)
    bits_theta : int — bits for (x,y) azimuth angle (default 4 → 16 bins)
    seed     : int   — rotation matrix seed; use the same value as the
                       PolarQuantizer2D for a fair head-to-head comparison
    z_min, z_max : float | None
                       Conservative defaults ±sqrt(dim). Call calibrate() to
                       tighten from real data.
    r_max    : float | None
                       Conservative default sqrt(dim). Call calibrate() to
                       tighten from real data.
    """

    def __init__(
        self,
        dim: int,
        bits_z: int = 4,
        bits_r: int = 4,
        bits_theta: int = 4,
        seed: int = 42,
        z_min: float | None = None,
        z_max: float | None = None,
        r_max: float | None = None,
    ):
        if dim % 3 != 0:
            raise ValueError(
                f"dim must be divisible by 3 for StackedPlaneQuantizer, got {dim}. "
                f"Hint: use dim=192, 384, 768, etc."
            )

        self.dim = dim
        self.bits_z = bits_z
        self.bits_r = bits_r
        self.bits_theta = bits_theta
        self.m = dim // 3  # number of triplet groups

        self.bins_z = 2 ** bits_z
        self.bins_r = 2 ** bits_r
        self.bins_theta = 2 ** bits_theta

        # Rotation matrix (same construction as baseline_2d for fair comparison)
        self.R = make_rotation(dim, seed)
        self.R_inv = self.R.T  # orthogonal → inverse = transpose

        # Static index tensors — computed once, reused every encode/decode call
        k = torch.arange(self.m)
        self.z_idx = 3 * k          # shape (m,)
        self.x_idx = 3 * k + 1
        self.y_idx = 3 * k + 2

        # Quantization ranges
        conservative = math.sqrt(dim)
        self.z_min = z_min if z_min is not None else -conservative
        self.z_max = z_max if z_max is not None else conservative
        self.r_max = r_max if r_max is not None else conservative

    # ------------------------------------------------------------------
    # Optional calibration
    # ------------------------------------------------------------------

    def calibrate(self, vectors: torch.Tensor) -> None:
        """
        Tighten quantization ranges using a representative sample of vectors.

        This is optional — the conservative sqrt(dim) defaults work without
        calibration, but tighter bounds reduce quantization error.

        Parameters
        ----------
        vectors : Tensor shape (N, dim)  — raw (unrotated) KV vectors
        """
        rotated = vectors @ self.R.T  # (N, dim)

        z_vals = rotated[:, self.z_idx]                      # (N, m)
        self.z_min = z_vals.min().item()
        self.z_max = z_vals.max().item()

        x = rotated[:, self.x_idx]
        y = rotated[:, self.y_idx]
        radii = torch.sqrt(x ** 2 + y ** 2)                  # (N, m)
        self.r_max = radii.max().item()

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Quantize a batch of KV vectors into compact integer codes.

        Parameters
        ----------
        vectors : Tensor shape (N, dim)

        Returns
        -------
        codes : Tensor shape (N, m) dtype int64
            Each code packs (i_z, i_r, i_theta):
            code = (i_z << (bits_r + bits_theta)) | (i_r << bits_theta) | i_theta
        """
        rotated = vectors @ self.R.T                          # (N, dim)

        # Extract the three coordinate streams
        z = rotated[:, self.z_idx]                            # (N, m)
        x = rotated[:, self.x_idx]                            # (N, m)
        y = rotated[:, self.y_idx]                            # (N, m)

        # --- Quantize z (uniform, left-edge floor) ---
        delta_z = (self.z_max - self.z_min) / self.bins_z
        i_z = ((z - self.z_min) / delta_z).floor().long()
        i_z = i_z.clamp(0, self.bins_z - 1)                  # (N, m)

        # --- Quantize (x, y) in polar form ---
        r = torch.sqrt(x ** 2 + y ** 2).clamp(0.0, self.r_max)
        theta = torch.atan2(y, x)                             # range (-pi, pi]

        i_r = (r / self.r_max * (self.bins_r - 1)).round().long()
        i_r = i_r.clamp(0, self.bins_r - 1)

        i_theta = ((theta + math.pi) / (2 * math.pi) * (self.bins_theta - 1)).round().long()
        i_theta = i_theta.clamp(0, self.bins_theta - 1)

        # --- Pack into one int64 per group ---
        codes = (i_z << (self.bits_r + self.bits_theta)) | (i_r << self.bits_theta) | i_theta
        return codes  # (N, m), int64

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct approximate KV vectors from codes.

        Parameters
        ----------
        codes : Tensor shape (N, m) dtype int64

        Returns
        -------
        vectors : Tensor shape (N, dim)
        """
        mask_theta = (1 << self.bits_theta) - 1
        mask_r = (1 << self.bits_r) - 1

        i_theta = (codes & mask_theta).float()                # (N, m)
        i_r = ((codes >> self.bits_theta) & mask_r).float()
        i_z = (codes >> (self.bits_r + self.bits_theta)).float()

        # --- Recover z using bin-center convention (minimises bias) ---
        delta_z = (self.z_max - self.z_min) / self.bins_z
        z_q = self.z_min + (i_z + 0.5) * delta_z             # (N, m)

        # --- Recover (x, y) from polar ---
        r_q = i_r / (self.bins_r - 1) * self.r_max
        theta_q = i_theta / (self.bins_theta - 1) * 2 * math.pi - math.pi

        x_q = r_q * torch.cos(theta_q)                       # (N, m)
        y_q = r_q * torch.sin(theta_q)

        # --- Scatter back into a (N, dim) rotated tensor ---
        N = codes.shape[0]
        rotated_q = torch.zeros(N, self.dim, dtype=codes.float().dtype)
        rotated_q[:, self.z_idx] = z_q
        rotated_q[:, self.x_idx] = x_q
        rotated_q[:, self.y_idx] = y_q

        # --- Un-rotate ---
        return rotated_q @ self.R_inv.T                       # (N, dim)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def bits_per_dim(self) -> float:
        return (self.bits_z + self.bits_r + self.bits_theta) / 3

    def compression_vs_fp32(self) -> float:
        return 32 / self.bits_per_dim()

    def error_bound(self) -> float:
        """
        Theoretical per-triplet Euclidean error bound (design doc §3.5).

        Assumes worst-case uniform quantization error for each of the three
        components: radius, angle, and z-coordinate.

        Returns
        -------
        bound : float  — ||v_triplet - v_hat_triplet|| ≤ bound
        """
        delta_r = self.r_max / max(self.bins_r - 1, 1)
        delta_theta = 2 * math.pi / max(self.bins_theta - 1, 1)
        delta_z = (self.z_max - self.z_min) / self.bins_z
        return math.sqrt(
            (delta_r / 2) ** 2
            + (self.r_max * delta_theta / 2) ** 2
            + (delta_z / 2) ** 2
        )

    def __repr__(self) -> str:
        return (
            f"StackedPlaneQuantizer(dim={self.dim}, "
            f"bits_z={self.bits_z}, bits_r={self.bits_r}, bits_theta={self.bits_theta}, "
            f"{self.bits_per_dim():.1f} bits/dim, "
            f"{self.compression_vs_fp32():.1f}x vs FP32)"
        )
