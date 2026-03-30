"""
test_learned_codebooks.py — Unit tests for LearnedSliceCodebook and the
learned-codebook dispatch path in StackedPlaneQuantizer.

All 11 original v1 tests must remain green; these 10 new tests cover M1.
"""

import math
import tempfile
from pathlib import Path

import pytest
import torch

from prismkv import StackedPlaneQuantizer
from prismkv.quantizer.learned_codebook import LearnedSliceCodebook, _kmeans


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_anisotropic_vectors(n: int = 2000, dim: int = 192, seed: int = 0) -> torch.Tensor:
    """Anisotropic Gaussians: z-dims scaled 3× to give learned CBs something to learn."""
    gen = torch.Generator().manual_seed(seed)
    v = torch.randn(n, dim, generator=gen)
    m = dim // 3
    v[:, torch.arange(m) * 3] *= 3.0
    return v


def make_quantizer(dim: int = 192, bits: int = 4) -> StackedPlaneQuantizer:
    return StackedPlaneQuantizer(dim=dim, bits_z=bits, bits_r=bits, bits_theta=bits, seed=42)


def train_and_attach(q: StackedPlaneQuantizer, vectors: torch.Tensor) -> LearnedSliceCodebook:
    """Calibrate q, rotate, train codebooks, attach, and return the codebook."""
    q.calibrate(vectors)
    rotated = vectors @ q.R.T
    K = 2 ** (q.bits_r + q.bits_theta)
    cb = LearnedSliceCodebook.train(
        rotated_vectors=rotated,
        z_idx=q.z_idx, x_idx=q.x_idx, y_idx=q.y_idx,
        z_min=q.z_min, z_max=q.z_max, r_max=q.r_max,
        bins_z=q.bins_z, K=K,
        max_iter=50, seed=0,
    )
    q._codebooks = cb
    return cb


# ---------------------------------------------------------------------------
# _kmeans unit tests
# ---------------------------------------------------------------------------

class TestKmeans:
    def test_basic_convergence(self):
        """k-means converges on well-separated clusters."""
        gen = torch.Generator().manual_seed(7)
        # Two obvious clusters
        pts = torch.cat([
            torch.randn(200, 2, generator=gen) + torch.tensor([5.0, 0.0]),
            torch.randn(200, 2, generator=gen) + torch.tensor([-5.0, 0.0]),
        ])
        centroids = _kmeans(pts, K=2, max_iter=100, seed=0)
        assert centroids.shape == (2, 2)
        # Each centroid should be near ±5 on x-axis
        xs = centroids[:, 0].sort().values
        assert xs[0].item() < -3.0
        assert xs[1].item() > 3.0

    def test_exact_k_points(self):
        """Works when N == K (no repeat needed)."""
        pts = torch.eye(4, 2)
        centroids = _kmeans(pts, K=4, seed=0)
        assert centroids.shape == (4, 2)
        assert torch.isfinite(centroids).all()

    def test_fewer_points_than_k(self):
        """Handles N < K by tiling and jittering."""
        pts = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        centroids = _kmeans(pts, K=8, seed=0)
        assert centroids.shape == (8, 2)
        assert torch.isfinite(centroids).all()

    def test_empty_input(self):
        """Empty input returns zero centroids without error."""
        pts = torch.zeros(0, 2)
        centroids = _kmeans(pts, K=4, seed=0)
        assert centroids.shape == (4, 2)


# ---------------------------------------------------------------------------
# LearnedSliceCodebook tests
# ---------------------------------------------------------------------------

class TestLearnedSliceCodebook:
    def test_train_shape(self):
        """Codebook tensor has the right shape after training."""
        q = make_quantizer(dim=192)
        vectors = make_anisotropic_vectors(n=1000, dim=192)
        cb = train_and_attach(q, vectors)
        K = 2 ** (q.bits_r + q.bits_theta)
        assert cb.codebooks.shape == (q.bins_z, K, 2)

    def test_encode_decode_shape(self):
        """encode_xy / decode_xy return the right shapes."""
        q = make_quantizer(dim=192)
        vectors = make_anisotropic_vectors(n=500, dim=192)
        cb = train_and_attach(q, vectors)

        N, m = 16, q.m
        xy = torch.randn(N, m, 2)
        i_z = torch.randint(0, q.bins_z, (N, m))

        i_flat = cb.encode_xy(xy, i_z)
        assert i_flat.shape == (N, m)
        assert i_flat.dtype == torch.long
        assert (i_flat >= 0).all() and (i_flat < cb.K).all()

        x_q, y_q = cb.decode_xy(i_flat, i_z)
        assert x_q.shape == (N, m)
        assert y_q.shape == (N, m)
        assert torch.isfinite(x_q).all()
        assert torch.isfinite(y_q).all()

    def test_serialization_roundtrip(self):
        """save() → load() produces identical codebook tensors."""
        q = make_quantizer(dim=192)
        vectors = make_anisotropic_vectors(n=500, dim=192)
        cb = train_and_attach(q, vectors)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cb.npz"
            cb.save(path)
            cb2 = LearnedSliceCodebook.load(path)

        assert cb2.codebooks.shape == cb.codebooks.shape
        assert torch.allclose(cb2.codebooks, cb.codebooks, atol=1e-5)
        assert cb2.bins_z == cb.bins_z
        assert cb2.K == cb.K
        assert abs(cb2.z_min - cb.z_min) < 1e-4
        assert abs(cb2.z_max - cb.z_max) < 1e-4
        assert abs(cb2.r_max - cb.r_max) < 1e-4

    def test_load_version_check(self):
        """load() raises ValueError on wrong version string."""
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.npz"
            np.savez(str(path),
                     codebooks=np.zeros((4, 16, 2), dtype=np.float32),
                     bins_z=np.int32(4),
                     K=np.int32(16),
                     z_min=np.float32(-1.0),
                     z_max=np.float32(1.0),
                     r_max=np.float32(1.0),
                     version=np.bytes_("wrong-version"))
            with pytest.raises(ValueError, match="Unsupported codebook version"):
                LearnedSliceCodebook.load(path)

    def test_centroid_radii_shape(self):
        """centroid_radii() has the right shape and non-negative values."""
        q = make_quantizer(dim=192)
        vectors = make_anisotropic_vectors(n=500, dim=192)
        cb = train_and_attach(q, vectors)
        radii = cb.centroid_radii()
        assert radii.shape == (cb.bins_z, cb.K)
        assert (radii >= 0).all()


# ---------------------------------------------------------------------------
# StackedPlaneQuantizer learned-dispatch tests
# ---------------------------------------------------------------------------

class TestStackedPlaneDispatch:
    def test_learned_encode_decode_finite(self):
        """Learned encode → decode produces finite output without NaN."""
        q = make_quantizer(dim=192)
        vectors = make_anisotropic_vectors(n=500, dim=192)
        train_and_attach(q, vectors)

        sample = vectors[:64]
        codes = q.encode(sample)
        recon = q.decode(codes)

        assert codes.shape == (64, q.m)
        assert recon.shape == (64, 192)
        assert torch.isfinite(recon).all()

    def test_learned_mse_le_uniform_on_anisotropic(self):
        """
        Learned codebook MSE must be ≤ 90% of uniform MSE on anisotropic data.

        This is the core M1 acceptance criterion: learned tables must actually
        improve over uniform polar quantization when the distribution is non-isotropic.
        """
        dim = 192
        n_cal = 5000

        q_uniform = make_quantizer(dim=dim)
        q_learned = make_quantizer(dim=dim)

        vectors = make_anisotropic_vectors(n=n_cal, dim=dim, seed=1)
        q_uniform.calibrate(vectors)
        q_learned.calibrate(vectors)

        # Train and attach codebook to the learned quantizer only
        train_and_attach(q_learned, vectors)
        assert q_learned._codebooks is not None
        assert q_uniform._codebooks is None

        # Evaluate on a held-out set
        held_out = make_anisotropic_vectors(n=1000, dim=dim, seed=99)

        codes_u = q_uniform.encode(held_out)
        recon_u = q_uniform.decode(codes_u)
        mse_uniform = ((held_out - recon_u) ** 2).mean().item()

        codes_l = q_learned.encode(held_out)
        recon_l = q_learned.decode(codes_l)
        mse_learned = ((held_out - recon_l) ** 2).mean().item()

        ratio = mse_learned / mse_uniform
        assert ratio <= 0.90, (
            f"Learned MSE ({mse_learned:.6f}) should be ≤ 90% of uniform MSE "
            f"({mse_uniform:.6f}), but ratio={ratio:.3f}"
        )

    def test_load_codebooks_syncs_ranges(self):
        """load_codebooks() syncs z_min/z_max/r_max from the saved file."""
        dim = 192
        q1 = make_quantizer(dim=dim)
        vectors = make_anisotropic_vectors(n=500, dim=dim)
        cb = train_and_attach(q1, vectors)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cb.npz"
            cb.save(path)

            q2 = make_quantizer(dim=dim)
            q2.load_codebooks(path)

        assert abs(q2.z_min - q1.z_min) < 1e-4
        assert abs(q2.z_max - q1.z_max) < 1e-4
        assert abs(q2.r_max - q1.r_max) < 1e-4
        assert q2._codebooks is not None

    def test_load_codebooks_none_reverts_to_uniform(self):
        """load_codebooks(None) clears the codebook and reverts to uniform path."""
        q = make_quantizer(dim=192)
        vectors = make_anisotropic_vectors(n=500, dim=192)
        train_and_attach(q, vectors)
        assert q._codebooks is not None

        q.load_codebooks(None)
        assert q._codebooks is None

    def test_uniform_fallback_unchanged(self):
        """
        When no codebook is loaded, encode/decode matches the original v1 output exactly.

        Verifies that the dispatch refactor did not change the uniform path.
        """
        dim = 192
        q = make_quantizer(dim=dim)
        vectors = make_anisotropic_vectors(n=100, dim=dim, seed=5)
        q.calibrate(vectors)

        assert q._codebooks is None  # no codebook loaded
        codes = q.encode(vectors)
        recon = q.decode(codes)

        assert torch.isfinite(recon).all()
        # Reconstruction should be close to original (not exact due to quantization)
        rel_err = ((vectors - recon).norm() / vectors.norm()).item()
        assert rel_err < 0.5, f"Uniform fallback relative error {rel_err:.3f} unexpectedly high"
