"""
lloyd_max.py — Lloyd-Max optimal 1-D scalar quantizer for PrismKV

Implements iterative Lloyd-Max quantization (k-means on 1-D scalars) to minimize
MSE for non-uniform distributions. Applied to the z-axis component of the 3-D
stacked-plane quantizer, where GPT-2 z values span [-30, +27] — a heavily
non-uniform distribution that uniform binning handles poorly.

Algorithm (Lloyd-Max):
    Alternate until convergence (or max_iter):
      1. Boundary update: boundary[k] = (centroid[k-1] + centroid[k]) / 2
      2. Centroid update: centroid[k] = E[z | boundary[k] < z ≤ boundary[k+1]]
    where E[z | ...] is the empirical mean within the interval.

References:
    Lloyd (1982) — "Least Squares Quantization in PCM", IEEE Trans. IT.
    Max (1960)  — "Quantizing for Minimum Distortion", IRE Trans. IT.

Author: Dan Hicks (github.com/danhicks96)
First published: 2026-03-30
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional


class LloydMaxQuantizer1D:
    """
    Optimal 1-D scalar quantizer via iterative Lloyd-Max algorithm.

    Minimizes MSE for an empirical distribution represented by a sample.

    Parameters
    ----------
    K : int
        Number of quantization bins (must be ≥ 2).
    max_iter : int
        Maximum iterations of the boundary/centroid alternation.
    tol : float
        Convergence tolerance on maximum centroid shift.
    """

    def __init__(self, K: int, max_iter: int = 100, tol: float = 1e-6):
        if K < 2:
            raise ValueError(f"K must be ≥ 2, got {K}")
        self.K = K
        self.max_iter = max_iter
        self.tol = tol

        # Set after fit()
        self.boundaries: Optional[torch.Tensor] = None  # shape (K+1,)
        self.centroids: Optional[torch.Tensor] = None   # shape (K,)
        self.n_iters: int = 0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, z: torch.Tensor) -> "LloydMaxQuantizer1D":
        """
        Fit the quantizer to an empirical sample.

        Parameters
        ----------
        z : Tensor shape (N,) — sample of scalar values

        Returns
        -------
        self (for chaining)
        """
        if z.ndim != 1:
            raise ValueError(f"z must be 1-D, got shape {z.shape}")
        if len(z) < self.K:
            raise ValueError(f"Need at least K={self.K} samples, got {len(z)}")

        z_sorted = z.float().sort().values  # (N,) ascending

        # --- Initialise centroids via quantiles ---
        # Use K uniform quantile positions to spread centroids over the distribution
        quant_pts = torch.linspace(0.0, 1.0, self.K + 2)[1:-1]  # K interior points
        indices = (quant_pts * (len(z_sorted) - 1)).long().clamp(0, len(z_sorted) - 1)
        centroids = z_sorted[indices].float().clone()             # (K,)

        boundaries = self._boundaries_from_centroids(centroids, z_sorted)

        # --- Iterative Lloyd-Max ---
        for i in range(self.max_iter):
            new_centroids = self._centroids_from_boundaries(z_sorted, boundaries)
            shift = (new_centroids - centroids).abs().max().item()
            centroids = new_centroids
            boundaries = self._boundaries_from_centroids(centroids, z_sorted)
            self.n_iters = i + 1
            if shift < self.tol:
                break

        self.centroids = centroids
        self.boundaries = boundaries
        return self

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Quantize scalar values to bin indices.

        Parameters
        ----------
        z : Tensor shape (...) float

        Returns
        -------
        indices : Tensor shape (...) long  in [0, K-1]
        """
        if self.boundaries is None:
            raise RuntimeError("Call fit() before encode()")
        # torch.bucketize: returns index of first boundary > z
        # boundaries[0] = -inf, boundaries[K] = +inf
        # Result is in [0, K] — clamp to [0, K-1]
        idx = torch.bucketize(z.float(), self.boundaries[1:-1])  # (...)
        return idx.long().clamp(0, self.K - 1)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Map bin indices back to centroid values.

        Parameters
        ----------
        indices : Tensor shape (...) long  in [0, K-1]

        Returns
        -------
        z_q : Tensor shape (...) float
        """
        if self.centroids is None:
            raise RuntimeError("Call fit() before decode()")
        return self.centroids[indices.long().clamp(0, self.K - 1)]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def mse(self, z: torch.Tensor) -> float:
        """Mean-squared quantization error on sample z."""
        idx = self.encode(z)
        z_q = self.decode(idx)
        return ((z.float() - z_q) ** 2).mean().item()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save boundaries and centroids into an existing or new .npz file.

        Keys: ``z_boundaries`` (K+1,), ``z_centroids`` (K,).
        If the file already exists and contains other keys, those are preserved.
        """
        path = Path(path)
        existing = {}
        if path.exists():
            data = np.load(path)
            existing = dict(data)
        existing["z_boundaries"] = self.boundaries.numpy()
        existing["z_centroids"] = self.centroids.numpy()
        np.savez(path, **existing)

    @classmethod
    def load(cls, path: str | Path) -> "LloydMaxQuantizer1D":
        """
        Load from a .npz file that contains ``z_boundaries`` and ``z_centroids``.
        """
        data = np.load(path)
        boundaries = torch.from_numpy(data["z_boundaries"].astype(np.float32))
        centroids = torch.from_numpy(data["z_centroids"].astype(np.float32))
        K = len(centroids)
        obj = cls(K=K)
        obj.boundaries = boundaries
        obj.centroids = centroids
        return obj

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _boundaries_from_centroids(
        centroids: torch.Tensor, z_sorted: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute K+1 boundaries from K centroids.

        boundary[0] = z_sorted[0] - 1  (open lower bound, always below data)
        boundary[k] = (centroid[k-1] + centroid[k]) / 2  for k in 1..K-1
        boundary[K] = z_sorted[-1] + 1 (open upper bound, always above data)
        """
        K = len(centroids)
        mid = (centroids[:-1] + centroids[1:]) / 2.0  # (K-1,)
        lower = z_sorted[0:1] - 1.0
        upper = z_sorted[-1:] + 1.0
        return torch.cat([lower, mid, upper])  # (K+1,)

    @staticmethod
    def _centroids_from_boundaries(
        z_sorted: torch.Tensor, boundaries: torch.Tensor
    ) -> torch.Tensor:
        """
        Update centroids as mean of data falling in each bin.

        If a bin is empty (can happen early in iteration), keep the boundary
        midpoint so we don't collapse bins.
        """
        K = len(boundaries) - 1
        centroids = torch.empty(K, dtype=torch.float32)
        for k in range(K):
            lo = boundaries[k].item()
            hi = boundaries[k + 1].item()
            mask = (z_sorted > lo) & (z_sorted <= hi)
            if mask.any():
                centroids[k] = z_sorted[mask].mean()
            else:
                centroids[k] = (boundaries[k] + boundaries[k + 1]) / 2.0
        return centroids
