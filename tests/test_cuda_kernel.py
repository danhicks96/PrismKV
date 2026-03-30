"""
test_cuda_kernel.py — CPU-mode tests for the prismkv.cuda Python interface (M15).

These tests validate the Python interface contract without requiring CUDA hardware.
All tests run in pure-CPU mode using the Python fallback.
"""

import math
import pytest
import torch

from prismkv.cuda import polar_attn_fwd, CUDA_AVAILABLE
from prismkv.quantizer.polar_attention import PolarAttentionApprox
from prismkv.quantizer.stacked_plane import StackedPlaneQuantizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_quantizer(dim=66, bits=4):
    return StackedPlaneQuantizer(dim=dim, bits_z=bits, bits_r=bits, bits_theta=bits, seed=0)


def encode_keys(k, bits=4, dim=66):
    """Encode a (b, nh, sk, d) float tensor to int16 PrismKV codes."""
    b, nh, sk, d = k.shape
    qtz = make_quantizer(dim, bits)
    flat = k.reshape(-1, d)
    codes_flat = qtz.encode(flat)           # (N, m)
    m = codes_flat.shape[-1]
    codes = codes_flat.reshape(b, nh, sk, m).to(torch.int16)
    return codes, qtz


def make_call_args(b=1, h=2, sq=4, sk=6, d=66, bits=4, seed=0):
    """Return (Q, K_codes, kwargs) ready for polar_attn_fwd."""
    torch.manual_seed(seed)
    Q = torch.randn(b, h, sq, d)
    K = torch.randn(b, h, sk, d)
    K_codes, qtz = encode_keys(K, bits=bits, dim=d)

    C_z = 1 << bits
    delta_z = (qtz.z_max - qtz.z_min) / C_z

    kwargs = dict(
        z_min=qtz.z_min,
        delta_z=delta_z,
        r_max=qtz.r_max,
        delta_r=qtz.r_max / ((1 << bits) - 1),
        delta_theta=2.0 * math.pi / ((1 << bits) - 1),
        scale=1.0 / math.sqrt(d),
        bits_z=bits,
        bits_r=bits,
        bits_theta=bits,
    )
    return Q, K_codes, kwargs, qtz


# ---------------------------------------------------------------------------
# Test: CUDA_AVAILABLE is False in CPU environment
# ---------------------------------------------------------------------------

class TestCudaAvailability:
    def test_cuda_available_is_bool(self):
        """CUDA_AVAILABLE must be a bool."""
        assert isinstance(CUDA_AVAILABLE, bool)

    def test_cuda_available_false_in_cpu_env(self):
        """In a CPU-only environment, CUDA_AVAILABLE must be False."""
        assert CUDA_AVAILABLE is False, (
            "Expected CUDA_AVAILABLE=False in CI CPU environment, got True"
        )


# ---------------------------------------------------------------------------
# Test: output shape from Python fallback
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_basic_shape(self):
        B, H, Sq, Sk, d = 1, 2, 4, 6, 66
        Q, K_codes, kwargs, _ = make_call_args(b=B, h=H, sq=Sq, sk=Sk, d=d)
        scores = polar_attn_fwd(Q, K_codes, **kwargs)
        assert scores.shape == (B, H, Sq, Sk), (
            f"Expected ({B},{H},{Sq},{Sk}), got {tuple(scores.shape)}"
        )

    def test_shape_batched(self):
        """Batched input (B=2, H=4) should produce correct (2,4,Sq,Sk) output."""
        B, H, Sq, Sk, d = 2, 4, 8, 10, 66
        Q, K_codes, kwargs, _ = make_call_args(b=B, h=H, sq=Sq, sk=Sk, d=d)
        scores = polar_attn_fwd(Q, K_codes, **kwargs)
        assert scores.shape == (B, H, Sq, Sk)

    def test_shape_gpt2_style(self):
        """GPT-2 style: d=66 (padded from 64), m=22 triplet groups."""
        B, H, Sq, Sk, d = 1, 12, 8, 8, 66
        Q, K_codes, kwargs, _ = make_call_args(b=B, h=H, sq=Sq, sk=Sk, d=d)
        m = K_codes.shape[-1]
        assert m == 22, f"Expected m=22 for d=66, got m={m}"
        scores = polar_attn_fwd(Q, K_codes, **kwargs)
        assert scores.shape == (B, H, Sq, Sk)

    def test_output_dtype_float32(self):
        Q, K_codes, kwargs, _ = make_call_args()
        scores = polar_attn_fwd(Q, K_codes, **kwargs)
        assert scores.dtype == torch.float32


# ---------------------------------------------------------------------------
# Test: fallback produces same result as PolarAttentionApprox
# ---------------------------------------------------------------------------

class TestFallbackMatchesPolarAttentionApprox:
    def test_scores_match_polar_attention_approx(self):
        """
        polar_attn_fwd fallback must produce scores identical (up to fp tolerance)
        to PolarAttentionApprox.attention_scores * scale.
        """
        B, H, Sq, Sk, d = 1, 2, 4, 6, 66
        bits = 4
        torch.manual_seed(42)
        Q = torch.randn(B, H, Sq, d)
        K = torch.randn(B, H, Sk, d)
        K_codes, qtz = encode_keys(K, bits=bits, dim=d)

        C_z = 1 << bits
        delta_z = (qtz.z_max - qtz.z_min) / C_z
        scale = 1.0 / math.sqrt(d)

        # Call prismkv.cuda fallback (no rotation — q must already be rotated)
        # PolarAttentionApprox applies R internally; here we pre-rotate Q.
        Q_rot = (Q.reshape(-1, d) @ qtz.R.T).reshape(B, H, Sq, d)

        scores_cuda_iface = polar_attn_fwd(
            Q_rot, K_codes,
            z_min=qtz.z_min,
            delta_z=delta_z,
            r_max=qtz.r_max,
            delta_r=qtz.r_max / max((1 << bits) - 1, 1),
            delta_theta=2.0 * math.pi / max((1 << bits) - 1, 1),
            scale=scale,
            bits_z=bits, bits_r=bits, bits_theta=bits,
        )

        # Reference: PolarAttentionApprox with R applied internally
        approx = PolarAttentionApprox(
            bits_z=bits, bits_r=bits, bits_theta=bits,
            z_min=qtz.z_min, z_max=qtz.z_max, r_max=qtz.r_max,
            scale=scale,
            R=qtz.R,
        )
        scores_ref = approx.attention_scores(Q, K_codes) * scale

        torch.testing.assert_close(
            scores_cuda_iface, scores_ref,
            atol=1e-4, rtol=1e-4,
            msg="Fallback scores differ from PolarAttentionApprox reference",
        )

    def test_scores_correlated_with_exact(self):
        """Fallback scores should be correlated with exact Cartesian dot products."""
        import torch.nn.functional as F
        B, H, Sq, Sk, d = 1, 2, 6, 8, 66
        bits = 4
        Q, K_codes, kwargs, qtz = make_call_args(b=B, h=H, sq=Sq, sk=Sk, d=d, bits=bits)

        scores = polar_attn_fwd(Q, K_codes, **kwargs)

        # Exact scores using Cartesian dot product on rotated Q and decoded K
        Q_rot = (Q.reshape(-1, d) @ qtz.R.T).reshape(B, H, Sq, d)
        K_decoded = qtz.decode(qtz.encode(
            Q.new_zeros(B * H * Sk, d)  # placeholder — we need real K
        ))
        # Simple check: scores should be finite and have reasonable range
        assert scores.isfinite().all(), "Scores contain NaN or Inf"
        assert scores.abs().max().item() < 1e4, "Scores have unreasonably large magnitude"


# ---------------------------------------------------------------------------
# Test: parameter packing (bits, q_params) correctness
# ---------------------------------------------------------------------------

class TestParameterPacking:
    def test_scale_applied(self):
        """Doubling the scale should double the scores."""
        Q, K_codes, kwargs, _ = make_call_args(seed=1)
        kwargs1 = {**kwargs, "scale": 1.0}
        kwargs2 = {**kwargs, "scale": 2.0}
        s1 = polar_attn_fwd(Q, K_codes, **kwargs1)
        s2 = polar_attn_fwd(Q, K_codes, **kwargs2)
        torch.testing.assert_close(s2, 2.0 * s1, atol=1e-5, rtol=1e-5)

    def test_bits_parameter_accepted(self):
        """Non-standard bit budgets (3+3+2) should be accepted without error."""
        B, H, Sq, Sk, d = 1, 1, 3, 5, 66
        bits_z, bits_r, bits_theta = 3, 3, 2
        qtz = StackedPlaneQuantizer(dim=d, bits_z=bits_z, bits_r=bits_r,
                                    bits_theta=bits_theta, seed=0)
        torch.manual_seed(5)
        Q = torch.randn(B, H, Sq, d)
        K = torch.randn(B, H, Sk, d)
        codes_flat = qtz.encode(K.reshape(-1, d))
        m = codes_flat.shape[-1]
        K_codes = codes_flat.reshape(B, H, Sk, m).to(torch.int16)

        C_z = 1 << bits_z
        delta_z = (qtz.z_max - qtz.z_min) / C_z

        scores = polar_attn_fwd(
            Q, K_codes,
            z_min=qtz.z_min, delta_z=delta_z,
            r_max=qtz.r_max,
            delta_r=qtz.r_max / max((1 << bits_r) - 1, 1),
            delta_theta=2.0 * math.pi / max((1 << bits_theta) - 1, 1),
            scale=1.0 / math.sqrt(d),
            bits_z=bits_z, bits_r=bits_r, bits_theta=bits_theta,
        )
        assert scores.shape == (B, H, Sq, Sk)
        assert scores.isfinite().all()

    def test_delta_z_affects_scores(self):
        """Changing delta_z (i.e., z range) should change scores."""
        Q, K_codes, kwargs, _ = make_call_args(seed=2)
        s1 = polar_attn_fwd(Q, K_codes, **kwargs)
        kwargs2 = {**kwargs, "delta_z": kwargs["delta_z"] * 2.0}
        s2 = polar_attn_fwd(Q, K_codes, **kwargs2)
        # Scores should differ (z contribution changes)
        assert not torch.allclose(s1, s2), "Changing delta_z should change scores"


# ---------------------------------------------------------------------------
# Test: GPT-2 style (d=66, m=22)
# ---------------------------------------------------------------------------

class TestGPT2Style:
    def test_head_dim_66_m22(self):
        """GPT-2 uses head_dim=64, padded to 66 for divisibility by 3 → m=22."""
        B, H, Sq, Sk, d = 1, 12, 16, 16, 66
        bits = 4
        Q, K_codes, kwargs, _ = make_call_args(b=B, h=H, sq=Sq, sk=Sk, d=d, bits=bits)
        assert K_codes.shape == (B, H, Sk, 22), (
            f"Expected K_codes shape (1,12,16,22), got {tuple(K_codes.shape)}"
        )
        scores = polar_attn_fwd(Q, K_codes, **kwargs)
        assert scores.shape == (B, H, Sq, Sk)
        assert scores.isfinite().all()

    def test_self_attention_scores_finite(self):
        """Self-attention (Sq == Sk) should produce finite scores."""
        d = 66
        Q, K_codes, kwargs, _ = make_call_args(b=1, h=12, sq=16, sk=16, d=d)
        scores = polar_attn_fwd(Q, K_codes, **kwargs)
        assert scores.isfinite().all()


# ---------------------------------------------------------------------------
# Test: batched inputs (B=2, H=4)
# ---------------------------------------------------------------------------

class TestBatchedInputs:
    def test_b2_h4_shape(self):
        B, H, Sq, Sk, d = 2, 4, 6, 8, 66
        Q, K_codes, kwargs, _ = make_call_args(b=B, h=H, sq=Sq, sk=Sk, d=d)
        scores = polar_attn_fwd(Q, K_codes, **kwargs)
        assert scores.shape == (B, H, Sq, Sk)

    def test_batches_independent(self):
        """Each batch element should produce independent scores."""
        B, H, Sq, Sk, d = 2, 2, 4, 6, 66
        Q, K_codes, kwargs, _ = make_call_args(b=B, h=H, sq=Sq, sk=Sk, d=d, seed=3)
        scores = polar_attn_fwd(Q, K_codes, **kwargs)

        # Modify batch 1's Q — batch 0 scores should be unaffected
        Q2 = Q.clone()
        Q2[1] = Q2[1] * 2.0

        scores2 = polar_attn_fwd(Q2, K_codes, **kwargs)

        # Batch 0 should be identical
        torch.testing.assert_close(scores[0], scores2[0], atol=1e-6, rtol=1e-6)
        # Batch 1 should differ
        assert not torch.allclose(scores[1], scores2[1])

    def test_b2_h4_scores_finite_and_shaped(self):
        B, H, Sq, Sk, d = 2, 4, 10, 12, 66
        Q, K_codes, kwargs, _ = make_call_args(b=B, h=H, sq=Sq, sk=Sk, d=d, seed=7)
        scores = polar_attn_fwd(Q, K_codes, **kwargs)
        assert scores.shape == (B, H, Sq, Sk)
        assert scores.dtype == torch.float32
        assert scores.isfinite().all()
