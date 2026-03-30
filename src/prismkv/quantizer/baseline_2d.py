"""
baseline_2d.py — 2-D Polar Quantizer (TurboQuant-style baseline)

Groups a d-dimensional vector into d/2 pairs, converts each pair to polar
coordinates (r, θ), and quantizes r and θ independently with uniform bins.

This is the baseline that PrismKV's StackedPlaneQuantizer is compared against.
Both use the same random-rotation pre-processing step so the comparison is fair.
"""

import math
import torch


class PolarQuantizer2D:
    """
    Training-free 2-D polar quantizer for KV cache vectors.

    Parameters
    ----------
    dim : int
        Vector dimension. Must be even.
    bits_r : int
        Bits allocated to the radius (default 4 → 16 bins).
    bits_theta : int
        Bits allocated to the angle (default 4 → 16 bins).
    seed : int
        Seed for the random rotation matrix so encoder and decoder share it.
    """

    def __init__(self, dim: int, bits_r: int = 4, bits_theta: int = 4, seed: int = 42):
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")

        self.dim = dim
        self.bits_r = bits_r
        self.bits_theta = bits_theta
        self.n_pairs = dim // 2

        self.bins_r = 2 ** bits_r        # e.g. 16
        self.bins_theta = 2 ** bits_theta  # e.g. 16

        # Pre-compute rotation matrix R (orthogonal, from QR decomp of Gaussian)
        torch.manual_seed(seed)
        G = torch.randn(dim, dim)
        Q, _ = torch.linalg.qr(G)
        self.R = Q          # (dim, dim)
        self.R_inv = Q.T    # orthogonal → inverse = transpose

        # Quantization ranges (set during calibration or fixed conservatively)
        # r is always ≥ 0; after rotation, empirically bounded well by ~3σ of N(0,1)
        self.r_max = math.sqrt(dim)   # conservative upper bound pre-calibration

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Quantize a batch of KV vectors.

        Parameters
        ----------
        vectors : Tensor shape (N, dim)

        Returns
        -------
        codes : Tensor shape (N, n_pairs) dtype int32
            Each code packs (i_r, i_theta) into a single integer:
            code = (i_r << bits_theta) | i_theta
        """
        N = vectors.shape[0]
        rotated = vectors @ self.R.T   # (N, dim)
        pairs = rotated.view(N, self.n_pairs, 2)   # (N, n_pairs, 2)

        x = pairs[..., 0]   # (N, n_pairs)
        y = pairs[..., 1]

        r = torch.sqrt(x ** 2 + y ** 2).clamp(min=0.0, max=self.r_max)
        theta = torch.atan2(y, x)    # range (-π, π]

        # Uniform quantization
        i_r = (r / self.r_max * (self.bins_r - 1)).round().long().clamp(0, self.bins_r - 1)
        i_theta = ((theta + math.pi) / (2 * math.pi) * (self.bins_theta - 1)).round().long().clamp(0, self.bins_theta - 1)

        codes = (i_r << self.bits_theta) | i_theta   # (N, n_pairs)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct approximate KV vectors from codes.

        Parameters
        ----------
        codes : Tensor shape (N, n_pairs) dtype int32

        Returns
        -------
        vectors : Tensor shape (N, dim)
        """
        mask_theta = (1 << self.bits_theta) - 1

        i_r = (codes >> self.bits_theta).float()
        i_theta = (codes & mask_theta).float()

        r_q = i_r / (self.bins_r - 1) * self.r_max
        theta_q = i_theta / (self.bins_theta - 1) * 2 * math.pi - math.pi

        x_q = r_q * torch.cos(theta_q)
        y_q = r_q * torch.sin(theta_q)

        # Re-assemble pairs → flat vector → un-rotate
        N = codes.shape[0]
        pairs_q = torch.stack([x_q, y_q], dim=-1)   # (N, n_pairs, 2)
        rotated_q = pairs_q.view(N, self.dim)
        vectors_q = rotated_q @ self.R_inv.T         # (N, dim)
        return vectors_q

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def bits_per_dim(self) -> float:
        return (self.bits_r + self.bits_theta) / 2

    def compression_vs_fp32(self) -> float:
        return 32 / self.bits_per_dim()

    def __repr__(self):
        return (
            f"PolarQuantizer2D(dim={self.dim}, bits_r={self.bits_r}, "
            f"bits_theta={self.bits_theta}, "
            f"{self.bits_per_dim():.1f} bits/dim, "
            f"{self.compression_vs_fp32():.1f}x vs FP32)"
        )
