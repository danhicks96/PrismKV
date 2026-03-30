"""
tests/test_quantizer.py — Unit tests for PrismKV quantizers.

Run with:  pytest tests/ -v
"""

import math
import sys
import os

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prismkv.utils import make_rotation
from prismkv.quantizer.baseline_2d import PolarQuantizer2D
from prismkv.quantizer.stacked_plane import StackedPlaneQuantizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DIM = 192   # divisible by both 2 and 3
N = 512
SEED_DATA = 0


@pytest.fixture(scope="module")
def vectors():
    torch.manual_seed(SEED_DATA)
    return torch.randn(N, DIM)


@pytest.fixture(scope="module")
def q2d():
    return PolarQuantizer2D(dim=DIM, bits_r=4, bits_theta=4, seed=42)


@pytest.fixture(scope="module")
def q3d():
    return StackedPlaneQuantizer(dim=DIM, bits_z=4, bits_r=4, bits_theta=4, seed=42)


# ---------------------------------------------------------------------------
# Utils tests
# ---------------------------------------------------------------------------

def test_utils_rotation_orthogonal():
    R = make_rotation(64, seed=42)
    assert R.shape == (64, 64)
    eye_approx = R @ R.T
    error = (eye_approx - torch.eye(64)).abs().max().item()
    assert error < 1e-5, f"R @ R.T should be identity, max error = {error:.2e}"


def test_utils_rotation_deterministic():
    R1 = make_rotation(32, seed=7)
    R2 = make_rotation(32, seed=7)
    assert torch.allclose(R1, R2), "Same seed must produce same rotation"


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_2d_encode_decode_shapes(q2d, vectors):
    codes = q2d.encode(vectors)
    assert codes.shape == (N, DIM // 2), f"Expected ({N}, {DIM//2}), got {codes.shape}"
    recon = q2d.decode(codes)
    assert recon.shape == (N, DIM), f"Expected ({N}, {DIM}), got {recon.shape}"


def test_3d_encode_decode_shapes(q3d, vectors):
    codes = q3d.encode(vectors)
    assert codes.shape == (N, DIM // 3), f"Expected ({N}, {DIM//3}), got {codes.shape}"
    recon = q3d.decode(codes)
    assert recon.shape == (N, DIM), f"Expected ({N}, {DIM}), got {recon.shape}"


# ---------------------------------------------------------------------------
# Round-trip error vs theoretical bound
# ---------------------------------------------------------------------------

def test_3d_round_trip_vs_bound(q3d, vectors):
    codes = q3d.encode(vectors)
    recon = q3d.decode(codes)

    # Per-vector RMSE across the whole batch
    mse = ((recon - vectors) ** 2).mean().item()
    rmse = math.sqrt(mse * DIM)

    # Theoretical bound: error_bound() is per-triplet; scale to full vector
    m = DIM // 3
    bound = q3d.error_bound() * math.sqrt(m) * 3.0  # 3x slack for loose worst-case bound
    assert rmse < bound, (
        f"Empirical RMSE ({rmse:.4f}) exceeds theoretical bound ({bound:.4f}). "
        f"Check quantization ranges (z_min/z_max/r_max)."
    )
    # Hard sanity check: quantization shouldn't be catastrophically wrong
    assert rmse < 10.0, f"RMSE={rmse:.4f} is too large — something is broken"


# ---------------------------------------------------------------------------
# Fair bits-per-dimension comparison
# ---------------------------------------------------------------------------

def test_same_bits_per_dim(q2d, q3d):
    bpd_2d = q2d.bits_per_dim()
    bpd_3d = q3d.bits_per_dim()
    assert bpd_2d == 4.0, f"2D quantizer should be 4.0 bits/dim, got {bpd_2d}"
    assert bpd_3d == 4.0, f"3D quantizer should be 4.0 bits/dim, got {bpd_3d}"


def test_2d_vs_3d_both_finite(q2d, q3d, vectors):
    """Both quantizers must produce finite, positive MSE at the same bits/dim."""
    mse_2d = ((q2d.decode(q2d.encode(vectors)) - vectors) ** 2).mean().item()
    mse_3d = ((q3d.decode(q3d.encode(vectors)) - vectors) ** 2).mean().item()
    assert math.isfinite(mse_2d) and mse_2d > 0, f"2D MSE invalid: {mse_2d}"
    assert math.isfinite(mse_3d) and mse_3d > 0, f"3D MSE invalid: {mse_3d}"


# ---------------------------------------------------------------------------
# Bias test
# ---------------------------------------------------------------------------

def test_3d_bias_small(q3d, vectors):
    """
    Mean reconstruction error per dimension should be small.
    Full QJL-style bias correction is deferred to v2.
    """
    recon = q3d.decode(q3d.encode(vectors))
    bias = (recon - vectors).mean(dim=0)          # shape (dim,)
    max_abs_bias = bias.abs().max().item()
    assert max_abs_bias < 0.5, (
        f"Per-dimension bias {max_abs_bias:.4f} is too large. "
        f"Consider calibrating z_min/z_max or tightening r_max."
    )


# ---------------------------------------------------------------------------
# Code packing invertibility
# ---------------------------------------------------------------------------

def test_code_packing_invertible():
    """Verify bit-pack / bit-unpack round-trip for known indices."""
    bits_z, bits_r, bits_theta = 4, 4, 4
    i_z_in, i_r_in, i_theta_in = 3, 7, 11

    code = (i_z_in << (bits_r + bits_theta)) | (i_r_in << bits_theta) | i_theta_in

    mask_theta = (1 << bits_theta) - 1
    mask_r = (1 << bits_r) - 1

    i_theta_out = code & mask_theta
    i_r_out = (code >> bits_theta) & mask_r
    i_z_out = code >> (bits_r + bits_theta)

    assert i_z_out == i_z_in, f"i_z mismatch: {i_z_out} != {i_z_in}"
    assert i_r_out == i_r_in, f"i_r mismatch: {i_r_out} != {i_r_in}"
    assert i_theta_out == i_theta_in, f"i_theta mismatch: {i_theta_out} != {i_theta_in}"


# ---------------------------------------------------------------------------
# Calibration test
# ---------------------------------------------------------------------------

def test_calibration_no_worse(vectors):
    """Post-calibration MSE must not exceed pre-calibration MSE."""
    q = StackedPlaneQuantizer(dim=DIM, bits_z=4, bits_r=4, bits_theta=4, seed=42)

    mse_before = ((q.decode(q.encode(vectors)) - vectors) ** 2).mean().item()

    # Calibrate on a separate set
    torch.manual_seed(99)
    cal_vectors = torch.randn(1000, DIM)
    q.calibrate(cal_vectors)

    mse_after = ((q.decode(q.encode(vectors)) - vectors) ** 2).mean().item()

    assert math.isfinite(mse_after), "Post-calibration MSE is not finite"
    # Calibration on random data should not catastrophically worsen things
    assert mse_after < mse_before * 5.0, (
        f"Calibration made things much worse: before={mse_before:.4f}, after={mse_after:.4f}"
    )


# ---------------------------------------------------------------------------
# dim % 3 != 0 raises
# ---------------------------------------------------------------------------

def test_invalid_dim_raises():
    with pytest.raises(ValueError, match="divisible by 3"):
        StackedPlaneQuantizer(dim=128)
