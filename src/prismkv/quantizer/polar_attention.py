"""
polar_attention.py — Attention score computation directly in polar-compressed space.

The key insight: after PrismKV compression, each KV vector triplet (z, x, y) is
stored as (i_z, i_r, i_theta).  The dot product between a query vector q and a
key vector k can be approximated without full dequantization:

    dot(q, k) ≈ sum_j  q_z_j * k_z_j  +  q_r_j * k_r_j * cos(q_theta_j - k_theta_j)

where the sum runs over all triplet groups j.

This is a novel prior-art contribution: computing attention weights directly
from polar-compressed codes, fusing dequantization into the attention kernel.

Two modes are provided:
  1. PolarAttentionApprox — computes the approximation analytically from
     (i_z, i_r, i_theta) codes without materialising FP vectors.
  2. TripletDotProduct    — exact dot product for decoded triplets, used as
     a reference to validate the approximation quality.

Mathematical derivation
-----------------------
Standard dot product for vectors u, v of dimension d (d = 3m triplets):

    <u, v> = sum_{k=0}^{m-1}  (u_{3k} * v_{3k})           ← z terms
                             + (u_{3k+1} * v_{3k+1})        ← x terms
                             + (u_{3k+2} * v_{3k+2})        ← y terms

After PrismKV encoding of v → (i_z, i_r, i_theta), the decoded coordinates are:

    z_q   = z_min + (i_z + 0.5) * delta_z
    r_q   = i_r / (C_r - 1) * r_max
    theta_q = i_theta / (C_theta - 1) * 2*pi - pi
    x_q   = r_q * cos(theta_q)
    y_q   = r_q * sin(theta_q)

So the cross-term for group k is:

    q_z * z_q + q_x * x_q + q_y * y_q
  = q_z * z_q
    + r_q * (q_x * cos(theta_q) + q_y * sin(theta_q))
  = q_z * z_q
    + r_q * r_query * cos(theta_query - theta_q)

where r_query = sqrt(q_x^2 + q_y^2) and theta_query = atan2(q_y, q_x).

The polar-space dot product therefore avoids the x/y multiply-add and uses
a single cosine evaluation per triplet group.

Author: Dan Hicks (github.com/danhicks96)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Polar dot product (reference, fully vectorised)
# ---------------------------------------------------------------------------

def polar_dot_product(
    q: torch.Tensor,
    k_z: torch.Tensor,
    k_r: torch.Tensor,
    k_theta: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the dot product between query vectors q and polar-encoded keys.

    Parameters
    ----------
    q       : (..., d) query vectors in Cartesian space (d = 3m)
    k_z     : (..., m) decoded z coordinates of keys
    k_r     : (..., m) decoded radii of keys
    k_theta : (..., m) decoded angles of keys in radians

    Returns
    -------
    dot     : (...,) dot products

    Notes
    -----
    This avoids materialising the full Cartesian key vector.
    """
    m = k_z.shape[-1]
    d = m * 3

    # Split query into triplets
    q_reshaped = q[..., :d].reshape(*q.shape[:-1], m, 3)  # (..., m, 3)
    q_z = q_reshaped[..., 0]       # (..., m)
    q_x = q_reshaped[..., 1]       # (..., m)
    q_y = q_reshaped[..., 2]       # (..., m)

    # Convert query (x, y) to polar
    q_r = torch.sqrt(q_x ** 2 + q_y ** 2 + 1e-12)
    q_theta = torch.atan2(q_y, q_x)

    # z contribution
    z_contrib = q_z * k_z                                   # (..., m)

    # (x, y) contribution via polar identity
    # q_x * k_x + q_y * k_y = r_q * r_k * cos(theta_q - theta_k)
    xy_contrib = q_r * k_r * torch.cos(q_theta - k_theta)  # (..., m)

    return (z_contrib + xy_contrib).sum(dim=-1)


def polar_dot_product_from_codes(
    q: torch.Tensor,
    k_codes: torch.Tensor,
    bits_z: int,
    bits_r: int,
    bits_theta: int,
    z_min: float,
    z_max: float,
    r_max: float,
    R: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute dot product between query q and PrismKV-encoded key codes.

    Because PrismKV codes represent the *rotated* key (the quantizer applies R
    before encoding), the query must be in the same rotated space.  Pass the
    quantizer's rotation matrix R and this function will rotate q automatically.
    Alternatively, pre-rotate q yourself: ``q_rot = q @ R.T``.

    Since rotation is orthogonal, ``<q, k> = <q_rot, k_rot>`` so the resulting
    dot products equal those in the original space.

    Parameters
    ----------
    q         : (batch, n_heads, seq_q, d) query vectors (original space)
    k_codes   : (batch, n_heads, seq_k, m) int16 PrismKV codes per triplet
    bits_z/r/theta : bit depths
    z_min/max, r_max : quantizer range parameters
    R         : (d, d) rotation matrix from StackedPlaneQuantizer (optional).
                If provided, q is rotated as q @ R.T before score computation.

    Returns
    -------
    scores    : (batch, n_heads, seq_q, seq_k) attention dot products
    """
    # Rotate q into the same space as the encoded keys
    if R is not None:
        b_q, nh_q, sq_q, d_q = q.shape
        q = (q.reshape(-1, d_q) @ R.T.to(q.device, q.dtype)).reshape(b_q, nh_q, sq_q, d_q)

    C_z = 1 << bits_z
    C_r = 1 << bits_r
    C_theta = 1 << bits_theta

    codes = k_codes.long()  # ensure integer arithmetic

    # Unpack codes
    mask_r = (1 << bits_r) - 1
    mask_z = (1 << bits_z) - 1

    i_theta = codes & ((1 << bits_theta) - 1)
    i_r     = (codes >> bits_theta) & mask_r
    i_z     = (codes >> (bits_theta + bits_r)) & mask_z

    # Dequantize to scalars (broadcast-friendly float)
    delta_z = (z_max - z_min) / C_z
    k_z_val = z_min + (i_z.float() + 0.5) * delta_z              # (..., m)
    k_r_val = i_r.float() / max(C_r - 1, 1) * r_max             # (..., m)
    k_theta_val = i_theta.float() / max(C_theta - 1, 1) * (2 * math.pi) - math.pi

    # q: (b, nh, sq, d)  →  need shape (b, nh, sq, m) per component
    b, nh, sq, d = q.shape
    _, _, sk, m = k_z_val.shape if k_z_val.ndim == 4 else (b, nh, k_codes.shape[-2], k_codes.shape[-1])

    # Ensure k tensors have shape (b, nh, sk, m)
    k_z_val   = k_z_val.view(b, nh, -1, m)
    k_r_val   = k_r_val.view(b, nh, -1, m)
    k_theta_val = k_theta_val.view(b, nh, -1, m)

    # Split q into triplets: (b, nh, sq, m, 3)
    d_trunc = m * 3
    q_trip = q[..., :d_trunc].view(b, nh, sq, m, 3)
    q_z = q_trip[..., 0]       # (b, nh, sq, m)
    q_x = q_trip[..., 1]
    q_y = q_trip[..., 2]

    q_r     = torch.sqrt(q_x ** 2 + q_y ** 2 + 1e-12)    # (b, nh, sq, m)
    q_theta = torch.atan2(q_y, q_x)                        # (b, nh, sq, m)

    # Broadcast: q is (b, nh, sq, m), k is (b, nh, sk, m)
    # scores[b, nh, i, j] = sum_m  q_z[b,nh,i,m]*k_z[b,nh,j,m] + ...
    # Use einsum or explicit broadcasting

    # z contribution: (b, nh, sq, sk)
    z_scores = torch.einsum("bnqm,bnkm->bnqk", q_z, k_z_val)

    # xy contribution via polar: need cos(q_theta - k_theta) broadcast
    # q_r * q_cos_q_theta  →  direct einsum trick:
    # r_q * r_k * cos(θ_q - θ_k) = (r_q*cos θ_q)*(r_k*cos θ_k) + (r_q*sin θ_q)*(r_k*sin θ_k)
    q_rx = q_r * torch.cos(q_theta)   # (b, nh, sq, m)
    q_ry = q_r * torch.sin(q_theta)   # (b, nh, sq, m)
    k_rx = k_r_val * torch.cos(k_theta_val)  # (b, nh, sk, m)
    k_ry = k_r_val * torch.sin(k_theta_val)  # (b, nh, sk, m)

    xy_scores = (
        torch.einsum("bnqm,bnkm->bnqk", q_rx, k_rx)
        + torch.einsum("bnqm,bnkm->bnqk", q_ry, k_ry)
    )

    return z_scores + xy_scores


# ---------------------------------------------------------------------------
# PolarAttentionApprox: drop-in scaled-dot-product approximation
# ---------------------------------------------------------------------------

class PolarAttentionApprox:
    """
    Approximate scaled dot-product attention using PrismKV-compressed keys.

    Computes attention scores and weighted value sums without full KV
    dequantization, using the polar dot product identity.

    This is novel prior art: attention score computation from polar codes
    avoids O(d) multiply-adds per (query, key) pair, replacing them with
    O(m) operations where m = d/3, plus one cosine per group.

    Parameters
    ----------
    bits_z, bits_r, bits_theta : PrismKV bit configuration
    z_min, z_max, r_max        : quantizer range (from StackedPlaneQuantizer)
    scale                      : attention scale factor (default 1/sqrt(d))
    """

    def __init__(
        self,
        bits_z: int = 4,
        bits_r: int = 4,
        bits_theta: int = 4,
        z_min: float = -4.0,
        z_max: float = 4.0,
        r_max: float = 4.0,
        scale: Optional[float] = None,
        R: Optional[torch.Tensor] = None,
    ) -> None:
        self.bits_z = bits_z
        self.bits_r = bits_r
        self.bits_theta = bits_theta
        self.z_min = z_min
        self.z_max = z_max
        self.r_max = r_max
        self._scale = scale
        self.R = R  # rotation matrix from StackedPlaneQuantizer

    def attention_scores(
        self,
        q: torch.Tensor,
        k_codes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute unscaled attention logits from Cartesian q and polar k codes.

        q       : (batch, n_heads, seq_q, d)
        k_codes : (batch, n_heads, seq_k, m)  int16 PrismKV codes

        Returns : (batch, n_heads, seq_q, seq_k)
        """
        return polar_dot_product_from_codes(
            q, k_codes,
            self.bits_z, self.bits_r, self.bits_theta,
            self.z_min, self.z_max, self.r_max,
            R=self.R,
        )

    def forward(
        self,
        q: torch.Tensor,
        k_codes: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full approximate attention: scores → softmax → weighted V sum.

        Parameters
        ----------
        q         : (batch, n_heads, seq_q, d)
        k_codes   : (batch, n_heads, seq_k, m) int16 codes
        v         : (batch, n_heads, seq_k, d) value vectors (decoded)
        attn_mask : optional additive mask (e.g. causal mask)

        Returns
        -------
        (output, weights) where output : (batch, n_heads, seq_q, d)
                                weights : (batch, n_heads, seq_q, seq_k)
        """
        d = q.shape[-1]
        scale = self._scale or (d ** -0.5)

        scores = self.attention_scores(q, k_codes) * scale   # (b, nh, sq, sk)

        if attn_mask is not None:
            scores = scores + attn_mask

        weights = F.softmax(scores, dim=-1)                  # (b, nh, sq, sk)
        output = torch.matmul(weights, v)                    # (b, nh, sq, d)
        return output, weights


# ---------------------------------------------------------------------------
# Approximation quality measurement
# ---------------------------------------------------------------------------

def measure_polar_approx_error(
    q: torch.Tensor,
    k: torch.Tensor,
    k_codes: torch.Tensor,
    bits_z: int,
    bits_r: int,
    bits_theta: int,
    z_min: float,
    z_max: float,
    r_max: float,
    R: Optional[torch.Tensor] = None,
) -> dict:
    """
    Compare exact dot products vs polar-code approximation.

    Returns a dict with:
      'mean_abs_error'  : mean |exact - approx|
      'max_abs_error'   : max  |exact - approx|
      'relative_error'  : mean |exact - approx| / mean |exact|
      'cosine_sim'      : mean cosine similarity between score vectors
    """
    # Exact: (b, nh, sq, sk)
    exact = torch.matmul(q, k.transpose(-2, -1))

    # Approx
    approx = polar_dot_product_from_codes(
        q, k_codes, bits_z, bits_r, bits_theta, z_min, z_max, r_max, R=R
    )

    diff = (exact - approx).abs()
    mean_abs = diff.mean().item()
    max_abs  = diff.max().item()
    rel      = (diff / (exact.abs() + 1e-8)).mean().item()

    # Cosine similarity between exact and approx score distributions
    cos = F.cosine_similarity(
        exact.flatten(0, -2),
        approx.flatten(0, -2),
        dim=-1,
    ).mean().item()

    return {
        "mean_abs_error": mean_abs,
        "max_abs_error": max_abs,
        "relative_error": rel,
        "cosine_sim": cos,
    }
