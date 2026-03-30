"""
learned_codebook.py — Per-z-slice k-means codebooks for StackedPlaneQuantizer

Each z-bin gets its own set of K=2^(bits_r+bits_theta) Cartesian (x,y) centroids,
trained via Lloyd's algorithm on a calibration corpus. At encode time, the nearest
centroid replaces the uniform polar quantization of (x, y).

The codebook is shared across all triplet groups (since the random rotation makes
all groups statistically similar). Training pools all N×m (x,y) pairs from all
groups, splits by i_z bin, and runs k-means per bin.

This is the v2 scientific contribution: by conditioning the (x,y) codebook on the
coarse z-index, the quantizer adapts to the actual (x,y) distribution within each
z-slice rather than forcing all slices to share one set of uniform polar bins.

Author: Dan Hicks (github.com/danhicks96)
Repo:   https://github.com/danhicks96/PrismKV
"""

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Internal: pure-torch Lloyd's k-means
# ---------------------------------------------------------------------------

def _kmeans(
    pts: torch.Tensor,
    K: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 0,
) -> torch.Tensor:
    """
    Lloyd's algorithm for 2-D points.

    Parameters
    ----------
    pts     : Tensor shape (N, 2)
    K       : int — number of centroids
    max_iter: int — maximum iterations (default 100)
    tol     : float — stop when centroid shift < tol (default 1e-4)
    seed    : int — random seed for initial centroid selection

    Returns
    -------
    centroids : Tensor shape (K, 2)
    """
    N = pts.shape[0]
    device = pts.device

    if N == 0:
        return torch.zeros(K, 2, device=device)

    # If too few points to fill K centroids, tile + jitter
    if N < K:
        repeats = (K // N) + 1
        pts = pts.repeat(repeats, 1)[:K]
        pts = pts + torch.randn_like(pts) * 1e-6

    # Initialise from a random subset (k-means++ would be better but adds deps)
    gen = torch.Generator(device=device).manual_seed(seed)
    perm = torch.randperm(pts.shape[0], generator=gen, device=device)[:K]
    centroids = pts[perm].clone().float()
    pts = pts.float()

    for _ in range(max_iter):
        # --- Assignment: (N, K) pairwise distances ---
        dists = torch.cdist(pts, centroids)           # (N, K)
        assignments = dists.argmin(dim=1)             # (N,)

        # --- Update: vectorised scatter_add ---
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(K, dtype=torch.long, device=device)

        counts.scatter_add_(
            0, assignments,
            torch.ones(pts.shape[0], dtype=torch.long, device=device)
        )
        new_centroids.scatter_add_(
            0,
            assignments.unsqueeze(1).expand(-1, 2),
            pts,
        )

        nonempty = counts > 0
        new_centroids[nonempty] /= counts[nonempty].float().unsqueeze(1)

        # Empty cluster: reinitialise at a random data point + tiny noise
        empty = ~nonempty
        n_empty = int(empty.sum().item())
        if n_empty:
            rand_idx = torch.randperm(pts.shape[0], generator=gen, device=device)[:n_empty]
            new_centroids[empty] = pts[rand_idx] + torch.randn(
                n_empty, 2, generator=gen, device=device
            ) * 1e-4

        shift = (new_centroids - centroids).norm().item()
        centroids = new_centroids
        if shift < tol:
            break

    return centroids


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class LearnedSliceCodebook:
    """
    Per-z-slice k-means codebooks for the 3-D stacked-plane quantizer.

    Each z-bin `b` (0 .. bins_z-1) stores K centroids in Cartesian (x, y)
    space.  Encoding finds the nearest centroid; decoding returns its
    coordinates directly — no cos/sin needed.

    The codebook is shared across all triplet groups within a quantizer
    instance.  This is justified because the global random rotation makes
    all groups statistically identically distributed before quantization.

    Parameters
    ----------
    codebooks : Tensor shape (bins_z, K, 2)
    bins_z    : int
    K         : int  — number of centroids (= 2^(bits_r + bits_theta))
    z_min, z_max, r_max : float — ranges stored alongside the codebook so
                                   encoder and decoder always agree
    """

    FILE_VERSION = "prismkv-cb-v1"

    def __init__(
        self,
        codebooks: torch.Tensor,
        bins_z: int,
        K: int,
        z_min: float,
        z_max: float,
        r_max: float,
    ) -> None:
        assert codebooks.shape == (bins_z, K, 2), (
            f"Expected codebooks shape ({bins_z}, {K}, 2), got {codebooks.shape}"
        )
        self.codebooks = codebooks.float()   # (bins_z, K, 2)
        self.bins_z = bins_z
        self.K = K
        self.z_min = z_min
        self.z_max = z_max
        self.r_max = r_max

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        rotated_vectors: torch.Tensor,
        z_idx: torch.Tensor,
        x_idx: torch.Tensor,
        y_idx: torch.Tensor,
        z_min: float,
        z_max: float,
        r_max: float,
        bins_z: int,
        K: int,
        max_iter: int = 100,
        tol: float = 1e-4,
        seed: int = 0,
    ) -> "LearnedSliceCodebook":
        """
        Train one codebook per z-bin by pooling all triplet groups.

        Parameters
        ----------
        rotated_vectors : Tensor shape (N, dim)  — already rotated by R
        z_idx, x_idx, y_idx : Tensor shape (m,)  — group index tensors
        z_min, z_max, r_max : float              — quantization ranges
        bins_z : int                             — number of z-bins
        K      : int                             — centroids per bin
        """
        N = rotated_vectors.shape[0]

        # Extract all (z, x, y) triplets — (N, m) each
        z = rotated_vectors[:, z_idx].float()
        x = rotated_vectors[:, x_idx].float()
        y = rotated_vectors[:, y_idx].float()

        # Compute i_z for every (vector, group) pair — (N, m)
        delta_z = (z_max - z_min) / bins_z
        i_z_all = ((z - z_min) / delta_z).floor().long().clamp(0, bins_z - 1)

        # Flatten to (N*m,)
        i_z_flat = i_z_all.reshape(-1)
        xy_flat = torch.stack([x, y], dim=-1).reshape(-1, 2)  # (N*m, 2)

        codebooks = torch.zeros(bins_z, K, 2)
        for b in range(bins_z):
            mask = (i_z_flat == b)
            pts = xy_flat[mask]          # all (x,y) pairs in this z-slice
            centroids = _kmeans(pts, K, max_iter=max_iter, tol=tol, seed=seed + b)
            codebooks[b] = centroids

        return cls(codebooks, bins_z=bins_z, K=K,
                   z_min=z_min, z_max=z_max, r_max=r_max)

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode_xy(
        self,
        xy: torch.Tensor,
        i_z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Find nearest centroid index for each (x, y) pair.

        Parameters
        ----------
        xy  : Tensor shape (N, m, 2)
        i_z : Tensor shape (N, m)  dtype long — z-bin index

        Returns
        -------
        i_flat : Tensor shape (N, m)  dtype long — centroid index in [0, K)
        """
        N, m, _ = xy.shape
        xy_flat = xy.reshape(N * m, 2).float()
        i_z_flat = i_z.reshape(N * m)
        i_flat = torch.zeros(N * m, dtype=torch.long)

        for b in range(self.bins_z):
            mask = (i_z_flat == b)
            if not mask.any():
                continue
            pts = xy_flat[mask]                          # (P, 2)
            centroids = self.codebooks[b]                # (K, 2)
            dists = torch.cdist(pts.float(), centroids)  # (P, K)
            i_flat[mask] = dists.argmin(dim=1)

        return i_flat.reshape(N, m)

    def decode_xy(
        self,
        i_flat: torch.Tensor,
        i_z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Look up centroid Cartesian coordinates.

        Parameters
        ----------
        i_flat : Tensor shape (N, m)  dtype long — centroid index
        i_z    : Tensor shape (N, m)  dtype long — z-bin index

        Returns
        -------
        x_q, y_q : Tensor shape (N, m) each
        """
        # Advanced indexing: codebooks[i_z, i_flat] → (N, m, 2)
        xy = self.codebooks[i_z, i_flat]   # (N, m, 2)
        return xy[..., 0], xy[..., 1]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path, metadata: Optional[dict] = None) -> None:
        """
        Save codebook to a .npz file.

        The file contains:
          - version    : str
          - bins_z     : int
          - K          : int
          - z_min/max, r_max : float
          - codebooks  : float32 array (bins_z, K, 2)
          - any extra keys from `metadata`
        """
        arrays: dict = {
            "codebooks": self.codebooks.numpy(),
            "z_min": np.float32(self.z_min),
            "z_max": np.float32(self.z_max),
            "r_max": np.float32(self.r_max),
            "bins_z": np.int32(self.bins_z),
            "K": np.int32(self.K),
            "version": np.bytes_(self.FILE_VERSION),
        }
        if metadata:
            for k, v in metadata.items():
                arrays[f"meta_{k}"] = np.array(v)
        np.savez(str(path), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "LearnedSliceCodebook":
        """Load from a .npz file produced by save()."""
        data = np.load(str(path), allow_pickle=False)

        version = data["version"].item()
        if hasattr(version, "decode"):
            version = version.decode()
        if version != cls.FILE_VERSION:
            raise ValueError(
                f"Unsupported codebook version '{version}'. "
                f"Expected '{cls.FILE_VERSION}'."
            )

        codebooks = torch.from_numpy(data["codebooks"].copy())
        return cls(
            codebooks=codebooks,
            bins_z=int(data["bins_z"]),
            K=int(data["K"]),
            z_min=float(data["z_min"]),
            z_max=float(data["z_max"]),
            r_max=float(data["r_max"]),
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def centroid_radii(self) -> torch.Tensor:
        """Return radius of each centroid per z-bin. Shape: (bins_z, K)."""
        x = self.codebooks[..., 0]
        y = self.codebooks[..., 1]
        return torch.sqrt(x ** 2 + y ** 2)

    def __repr__(self) -> str:
        return (
            f"LearnedSliceCodebook(bins_z={self.bins_z}, K={self.K}, "
            f"z=[{self.z_min:.2f}, {self.z_max:.2f}], r_max={self.r_max:.2f})"
        )
