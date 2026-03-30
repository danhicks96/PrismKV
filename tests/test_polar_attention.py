"""
test_polar_attention.py — Tests for polar-space attention approximation (M9).
"""

import math
import pytest
import torch
import torch.nn.functional as F

from prismkv.quantizer.polar_attention import (
    polar_dot_product,
    polar_dot_product_from_codes,
    PolarAttentionApprox,
    measure_polar_approx_error,
)
from prismkv.quantizer.stacked_plane import StackedPlaneQuantizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_quantizer(dim=66, bits=4):
    return StackedPlaneQuantizer(dim=dim, bits_z=bits, bits_r=bits, bits_theta=bits, seed=0)


def encode_keys(q, bits=4, dim=66):
    """Encode a (b, nh, sk, d) tensor; return (codes int16, quantizer)."""
    b, nh, sk, d = q.shape
    flat = q.reshape(-1, d)
    qtz = make_quantizer(dim, bits)
    codes_flat = qtz.encode(flat)             # (N, m)
    m = codes_flat.shape[-1]
    codes = codes_flat.reshape(b, nh, sk, m).to(torch.int16)
    return codes, qtz


# ---------------------------------------------------------------------------
# polar_dot_product identity
# ---------------------------------------------------------------------------

class TestPolarDotProduct:
    def test_identity_with_decoded_vectors(self):
        """
        polar_dot_product should exactly reproduce Cartesian dot product
        when k_z, k_r, k_theta come from the corresponding Cartesian k vector.
        """
        torch.manual_seed(42)
        m = 10
        q = torch.randn(3, m * 3)
        k = torch.randn(3, m * 3)

        k_r3 = k.reshape(3, m, 3)
        k_z_val = k_r3[:, :, 0]
        k_x_val = k_r3[:, :, 1]
        k_y_val = k_r3[:, :, 2]

        k_r = torch.sqrt(k_x_val ** 2 + k_y_val ** 2)
        k_theta = torch.atan2(k_y_val, k_x_val)

        approx = polar_dot_product(q, k_z_val, k_r, k_theta)
        exact = (q * k).sum(dim=-1)

        torch.testing.assert_close(approx, exact, atol=1e-5, rtol=1e-4)

    def test_zero_query_gives_near_zero(self):
        m = 4
        q = torch.zeros(2, m * 3)
        k_z = torch.randn(2, m)
        k_r = torch.abs(torch.randn(2, m))
        k_theta = torch.randn(2, m)
        result = polar_dot_product(q, k_z, k_r, k_theta)
        assert result.abs().max().item() < 1e-4  # eps=1e-12 in sqrt gives ~2e-6

    def test_output_shape(self):
        m = 8
        q = torch.randn(5, m * 3)
        k_z = torch.randn(5, m)
        k_r = torch.abs(torch.randn(5, m))
        k_theta = torch.randn(5, m)
        out = polar_dot_product(q, k_z, k_r, k_theta)
        assert out.shape == (5,)


# ---------------------------------------------------------------------------
# polar_dot_product_from_codes
# ---------------------------------------------------------------------------

class TestPolarDotProductFromCodes:
    def setup_method(self):
        torch.manual_seed(0)
        self.b, self.nh, self.sq, self.sk, self.d = 1, 2, 4, 6, 66
        self.q = torch.randn(self.b, self.nh, self.sq, self.d)
        self.k = torch.randn(self.b, self.nh, self.sk, self.d)
        self.codes, self.qtz = encode_keys(self.k, bits=4, dim=self.d)
        self.bits = 4

    def test_output_shape(self):
        scores = polar_dot_product_from_codes(
            self.q, self.codes, self.bits, self.bits, self.bits,
            self.qtz.z_min, self.qtz.z_max, self.qtz.r_max, R=self.qtz.R,
        )
        assert scores.shape == (self.b, self.nh, self.sq, self.sk)

    def test_reasonable_approximation(self):
        """
        Approx scores should be correlated with exact scores.
        Cosine similarity > 0.5 for 4-bit encoding.
        """
        scores_approx = polar_dot_product_from_codes(
            self.q, self.codes, self.bits, self.bits, self.bits,
            self.qtz.z_min, self.qtz.z_max, self.qtz.r_max, R=self.qtz.R,
        )
        scores_exact = torch.matmul(self.q, self.k.transpose(-2, -1))

        cos = F.cosine_similarity(
            scores_approx.flatten(),
            scores_exact.flatten(),
            dim=0,
        ).item()
        assert cos > 0.5, f"cosine_sim={cos:.3f}, expected > 0.5"

    def test_higher_bits_more_accurate(self):
        """More bits → better approximation."""
        def approx_error(bits):
            codes, qtz = encode_keys(self.k, bits=bits, dim=self.d)
            scores_approx = polar_dot_product_from_codes(
                self.q, codes, bits, bits, bits,
                qtz.z_min, qtz.z_max, qtz.r_max, R=qtz.R,
            )
            scores_exact = torch.matmul(self.q, self.k.transpose(-2, -1))
            return (scores_approx - scores_exact).abs().mean().item()

        err_3 = approx_error(3)
        err_5 = approx_error(5)
        assert err_5 < err_3, f"err_3={err_3:.4f}, err_5={err_5:.4f}"

    def test_dtype_float32(self):
        scores = polar_dot_product_from_codes(
            self.q, self.codes, self.bits, self.bits, self.bits,
            self.qtz.z_min, self.qtz.z_max, self.qtz.r_max, R=self.qtz.R,
        )
        assert scores.dtype == torch.float32


# ---------------------------------------------------------------------------
# PolarAttentionApprox
# ---------------------------------------------------------------------------

class TestPolarAttentionApprox:
    def setup_method(self):
        torch.manual_seed(7)
        self.b, self.nh = 1, 2
        self.sq, self.sk = 5, 8
        self.d = 66
        self.q = torch.randn(self.b, self.nh, self.sq, self.d)
        self.k = torch.randn(self.b, self.nh, self.sk, self.d)
        self.v = torch.randn(self.b, self.nh, self.sk, self.d)
        self.codes, self.qtz = encode_keys(self.k, bits=4, dim=self.d)
        self.approx = PolarAttentionApprox(
            bits_z=4, bits_r=4, bits_theta=4,
            z_min=self.qtz.z_min, z_max=self.qtz.z_max, r_max=self.qtz.r_max,
            R=self.qtz.R,
        )

    def test_attention_scores_shape(self):
        scores = self.approx.attention_scores(self.q, self.codes)
        assert scores.shape == (self.b, self.nh, self.sq, self.sk)

    def test_forward_output_shape(self):
        out, weights = self.approx.forward(self.q, self.codes, self.v)
        assert out.shape == (self.b, self.nh, self.sq, self.d)
        assert weights.shape == (self.b, self.nh, self.sq, self.sk)

    def test_weights_sum_to_one(self):
        _, weights = self.approx.forward(self.q, self.codes, self.v)
        row_sums = weights.sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5)

    def test_causal_mask_applied(self):
        """With a causal mask, future positions get zero weight."""
        causal_mask = torch.full((self.sq, self.sk), float("-inf"))
        for i in range(self.sq):
            causal_mask[i, : i + 1] = 0.0
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        _, weights = self.approx.forward(self.q, self.codes, self.v, attn_mask=causal_mask)
        for i in range(self.sq):
            if i + 1 < self.sk:
                future_weight = weights[0, 0, i, i + 1:].abs().max().item()
                assert future_weight < 1e-6

    def test_scale_does_not_affect_raw_scores(self):
        """scale is only applied in forward(), not attention_scores()."""
        a1 = PolarAttentionApprox(bits_z=4, bits_r=4, bits_theta=4,
                                   z_min=self.qtz.z_min, z_max=self.qtz.z_max,
                                   r_max=self.qtz.r_max, scale=1.0)
        a2 = PolarAttentionApprox(bits_z=4, bits_r=4, bits_theta=4,
                                   z_min=self.qtz.z_min, z_max=self.qtz.z_max,
                                   r_max=self.qtz.r_max, scale=0.1)
        s1 = a1.attention_scores(self.q, self.codes)
        s2 = a2.attention_scores(self.q, self.codes)
        torch.testing.assert_close(s1, s2)


# ---------------------------------------------------------------------------
# measure_polar_approx_error
# ---------------------------------------------------------------------------

class TestMeasurePolarApproxError:
    def test_returns_expected_keys(self):
        torch.manual_seed(1)
        b, nh, sq, sk, d = 1, 1, 3, 5, 66
        q = torch.randn(b, nh, sq, d)
        k = torch.randn(b, nh, sk, d)
        codes, qtz = encode_keys(k, bits=4, dim=d)
        result = measure_polar_approx_error(
            q, k, codes, 4, 4, 4,
            qtz.z_min, qtz.z_max, qtz.r_max, R=qtz.R,
        )
        for key in ("mean_abs_error", "max_abs_error", "relative_error", "cosine_sim"):
            assert key in result

    def test_cosine_sim_in_range(self):
        torch.manual_seed(3)
        b, nh, sq, sk, d = 1, 1, 4, 4, 66
        q = torch.randn(b, nh, sq, d)
        k = torch.randn(b, nh, sk, d)
        codes, qtz = encode_keys(k, bits=4, dim=d)
        result = measure_polar_approx_error(
            q, k, codes, 4, 4, 4,
            qtz.z_min, qtz.z_max, qtz.r_max, R=qtz.R,
        )
        assert -1.01 <= result["cosine_sim"] <= 1.01

    def test_decoded_k_matches_polar_approx(self):
        """
        If we use the decoded k (round-tripped through quantize→decode),
        the polar-code dot product should closely match the Cartesian dot product.
        """
        torch.manual_seed(5)
        b, nh, sq, sk, d = 1, 1, 2, 3, 66
        q = torch.randn(b, nh, sq, d)
        k_raw = torch.randn(b, nh, sk, d)

        qtz = make_quantizer(d, 4)
        k_flat = k_raw.reshape(-1, d)
        codes_flat = qtz.encode(k_flat)
        k_dec = qtz.decode(codes_flat).reshape(b, nh, sk, d)
        codes = codes_flat.reshape(b, nh, sk, -1).to(torch.int16)

        result = measure_polar_approx_error(
            q, k_dec, codes, 4, 4, 4,
            qtz.z_min, qtz.z_max, qtz.r_max, R=qtz.R,
        )
        assert result["mean_abs_error"] < 0.1, (
            f"mean_abs_error={result['mean_abs_error']:.4f}"
        )
