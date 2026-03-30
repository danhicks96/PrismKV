"""
utils.py — Shared utilities for PrismKV quantizers.
"""

import math
import torch


def make_rotation(dim: int, seed: int = 42) -> torch.Tensor:
    """
    Generate a random orthogonal rotation matrix via QR decomposition.

    The same (dim, seed) pair always produces the same matrix, so encoder
    and decoder stay in sync without storing the matrix explicitly.

    Parameters
    ----------
    dim  : int  — square matrix dimension
    seed : int  — random seed (default 42)

    Returns
    -------
    Q : Tensor shape (dim, dim), orthogonal: Q @ Q.T == I
    """
    torch.manual_seed(seed)
    G = torch.randn(dim, dim)
    Q, _ = torch.linalg.qr(G)
    return Q


def calibrate_r_max(rotated_vectors: torch.Tensor, quantile: float = 0.9999) -> float:
    """
    Compute an empirical r_max from a batch of already-rotated KV vectors.

    Pairs consecutive dimensions (0,1), (2,3), ... and computes the
    radius of each pair, then returns the given quantile across all pairs
    and all vectors. This gives a tighter bound than the conservative
    sqrt(dim) default.

    Parameters
    ----------
    rotated_vectors : Tensor shape (N, dim), dim must be even
    quantile        : float in (0, 1), default 0.9999

    Returns
    -------
    r_max : float
    """
    N, dim = rotated_vectors.shape
    if dim % 2 != 0:
        raise ValueError(f"dim must be even for pair-wise radius, got {dim}")
    pairs = rotated_vectors.view(N, dim // 2, 2)
    radii = torch.sqrt(pairs[..., 0] ** 2 + pairs[..., 1] ** 2)  # (N, dim//2)
    return torch.quantile(radii.flatten(), quantile).item()
