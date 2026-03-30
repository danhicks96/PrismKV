"""
prismkv.cuda — CUDA-accelerated fused dequantize + polar attention kernel.

Requires CUDA >= 11.8 and PyTorch with CUDA support.
Falls back silently to the Python polar_attention.py implementation when CUDA is not available.

Build:
    python setup_cuda.py build_ext --inplace

"""
from __future__ import annotations
import torch
from typing import Optional, Tuple

CUDA_AVAILABLE = False
try:
    from prismkv_cuda import polar_attn_fwd as _cuda_fwd
    CUDA_AVAILABLE = True
except ImportError:
    pass


def polar_attn_fwd(
    Q: torch.Tensor,
    K_codes: torch.Tensor,
    z_min: float,
    delta_z: float,
    r_max: float,
    delta_r: float,
    delta_theta: float,
    scale: float,
    bits_z: int,
    bits_r: int,
    bits_theta: int,
) -> torch.Tensor:
    """
    Fused dequantize + polar attention forward pass.

    Computes attention scores S[b,h,i,j] = scale * <Q[b,h,i], K_j>
    directly from int16 PrismKV codes without materialising FP16 key tensors.

    Parameters
    ----------
    Q         : (B, H, Sq, d) float32 — query vectors, Cartesian space (rotated by R)
    K_codes   : (B, H, Sk, m) int16  — packed PrismKV codes, m = d/3
    z_min, delta_z, r_max, delta_r, delta_theta : quantizer parameters
    scale     : attention scale (typically 1/sqrt(d))
    bits_z, bits_r, bits_theta : bit budgets

    Returns
    -------
    scores : (B, H, Sq, Sk) float32 — unmasked attention logits
    """
    if CUDA_AVAILABLE and Q.is_cuda:
        q_params = torch.tensor(
            [z_min, delta_z, r_max, delta_r, delta_theta, scale],
            dtype=torch.float32, device=Q.device
        )
        bits = torch.tensor([bits_z, bits_r, bits_theta], dtype=torch.int32, device=Q.device)
        return _cuda_fwd(Q, K_codes, q_params, bits)
    return _python_fallback(Q, K_codes, z_min, delta_z, r_max, delta_r, delta_theta, scale, bits_z, bits_r, bits_theta)


def _python_fallback(Q, K_codes, z_min, delta_z, r_max, delta_r, delta_theta, scale, bits_z, bits_r, bits_theta):
    """Pure-Python fallback using polar_dot_product_from_codes.

    The upstream polar_dot_product_from_codes signature uses z_max rather than
    delta_z, so we reconstruct z_max = z_min + delta_z * (2**bits_z) here.
    The scale factor is applied after the dot-product computation.
    """
    import math
    from prismkv.quantizer.polar_attention import polar_dot_product_from_codes
    B, H, Sq, d = Q.shape
    B2, H2, Sk, m = K_codes.shape
    assert B == B2 and H == H2, "Batch and head dimensions must match"

    # Reconstruct z_max from z_min and delta_z
    C_z = 1 << bits_z
    z_max = z_min + delta_z * C_z

    scores = polar_dot_product_from_codes(
        q=Q,
        k_codes=K_codes,
        bits_z=bits_z,
        bits_r=bits_r,
        bits_theta=bits_theta,
        z_min=z_min,
        z_max=z_max,
        r_max=r_max,
        R=None,
    )
    return scores * scale
