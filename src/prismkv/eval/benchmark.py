"""
benchmark.py — Compare PrismKV quantization schemes on real or synthetic KV vectors.

Metrics per scheme:
  - Per-vector RMSE
  - Cosine similarity (mean, min)
  - Relative error  ||v - v_hat|| / ||v||
  - Simulated KV memory at 4096-token context (MB)
  - Encode+decode throughput (vectors/sec)

Author: Dan Hicks (github.com/danhicks96)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch

from prismkv.quantizer.baseline_2d import PolarQuantizer2D
from prismkv.quantizer.stacked_plane import StackedPlaneQuantizer
from prismkv.quantizer.learned_codebook import LearnedSliceCodebook


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SchemeResult:
    name: str
    bits_per_dim: float
    rmse: float
    cosine_sim_mean: float
    cosine_sim_min: float
    relative_error_mean: float
    memory_mb_4k: float         # simulated KV memory at 4096 tokens
    throughput_vps: float       # encode+decode vectors per second
    n_vectors: int


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_scheme(
    vectors: torch.Tensor,
    encode_fn,
    decode_fn,
    bits_per_dim: float,
    name: str,
    n_heads: int = 1,
    original_dim: Optional[int] = None,
) -> SchemeResult:
    """
    Evaluate one quantization scheme on a batch of vectors.

    Parameters
    ----------
    vectors      : (N, dim) float32
    encode_fn    : callable(Tensor) → codes
    decode_fn    : callable(codes)  → Tensor (N, dim)
    bits_per_dim : bits consumed per original (unpadded) dimension
    name         : display name for the scheme
    n_heads      : number of attention heads (for memory calculation)
    original_dim : unpadded head dim (for memory; defaults to vectors.shape[1])
    """
    N, dim = vectors.shape
    d = original_dim if original_dim is not None else dim

    # Throughput measurement
    t0 = time.perf_counter()
    codes = encode_fn(vectors)
    recon = decode_fn(codes)
    elapsed = time.perf_counter() - t0
    throughput = N / elapsed if elapsed > 0 else float("inf")

    # Trim padding if present (compare on original dims only)
    v_cmp = vectors[..., :d]
    r_cmp = recon[..., :d]

    # RMSE
    diff = v_cmp - r_cmp
    rmse = diff.pow(2).mean(dim=1).sqrt().mean().item()

    # Cosine similarity
    v_norm = torch.nn.functional.normalize(v_cmp, dim=1)
    r_norm = torch.nn.functional.normalize(r_cmp, dim=1)
    cos = (v_norm * r_norm).sum(dim=1)
    cos_mean = cos.mean().item()
    cos_min = cos.min().item()

    # Relative error
    v_norms = v_cmp.norm(dim=1)
    rel_err = (diff.norm(dim=1) / v_norms.clamp(min=1e-8)).mean().item()

    # Simulated memory at 4096-token context
    # FP32 baseline: 4096 * d * 4 bytes * 2 (K+V) * n_heads
    # Quantized: 4096 * d * bits_per_dim/8 bytes * 2 * n_heads
    tokens = 4096
    memory_mb_4k = (tokens * d * bits_per_dim / 8 * 2 * n_heads) / (1024 ** 2)

    return SchemeResult(
        name=name,
        bits_per_dim=bits_per_dim,
        rmse=rmse,
        cosine_sim_mean=cos_mean,
        cosine_sim_min=cos_min,
        relative_error_mean=rel_err,
        memory_mb_4k=memory_mb_4k,
        throughput_vps=throughput,
        n_vectors=N,
    )


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    vectors: torch.Tensor,
    dim_2d: Optional[int] = None,
    dim_3d: Optional[int] = None,
    codebook_path: Optional[str] = None,
    bits: int = 4,
    n_heads: int = 1,
    original_dim: Optional[int] = None,
    label: str = "",
) -> List[SchemeResult]:
    """
    Compare 2D polar, 3D uniform, and (optionally) 3D learned on one batch.

    Parameters
    ----------
    vectors       : (N, dim) calibration/evaluation vectors (already padded if needed)
    dim_2d        : dim for 2D quantizer (default: vectors.shape[1])
    dim_3d        : dim for 3D quantizer (default: vectors.shape[1])
    codebook_path : .npz file for learned codebook (optional)
    bits          : bits per component (applied to all schemes)
    n_heads       : attention heads (for memory calculation)
    original_dim  : unpadded head dim (for memory/error on actual dims)
    label         : descriptive label printed in the table header
    """
    N, dim = vectors.shape
    d2 = dim_2d if dim_2d is not None else dim
    d3 = dim_3d if dim_3d is not None else dim

    # --- 2D polar baseline ---
    if d2 % 2 != 0:
        d2 += 1  # pad to even
    q2 = PolarQuantizer2D(dim=d2, bits_r=bits, bits_theta=bits, seed=42)
    result_2d = evaluate_scheme(
        vectors[:, :d2],
        encode_fn=q2.encode,
        decode_fn=q2.decode,
        bits_per_dim=q2.bits_per_dim(),
        name="2D Polar (uniform)",
        n_heads=n_heads,
        original_dim=original_dim,
    )

    # --- 3D uniform ---
    if d3 % 3 != 0:
        d3 = d3 + (3 - d3 % 3) % 3
    q3u = StackedPlaneQuantizer(dim=d3, bits_z=bits, bits_r=bits, bits_theta=bits, seed=42)
    q3u.calibrate(vectors[:, :d3])
    result_3d_uniform = evaluate_scheme(
        vectors[:, :d3],
        encode_fn=q3u.encode,
        decode_fn=q3u.decode,
        bits_per_dim=q3u.bits_per_dim(),
        name="3D Stacked-Plane (uniform)",
        n_heads=n_heads,
        original_dim=original_dim,
    )

    results = [result_2d, result_3d_uniform]

    # --- 3D learned (optional) ---
    if codebook_path is not None:
        q3l = StackedPlaneQuantizer(dim=d3, bits_z=bits, bits_r=bits, bits_theta=bits, seed=42)
        q3l.load_codebooks(codebook_path)
        result_3d_learned = evaluate_scheme(
            vectors[:, :d3],
            encode_fn=q3l.encode,
            decode_fn=q3l.decode,
            bits_per_dim=q3l.bits_per_dim(),
            name="3D Stacked-Plane (learned)",
            n_heads=n_heads,
            original_dim=original_dim,
        )
        results.append(result_3d_learned)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(results: List[SchemeResult], title: str = "") -> None:
    """Print a formatted comparison table to stdout."""
    if title:
        print(f"\n{'═' * 78}")
        print(f"  {title}")
        print(f"{'═' * 78}")

    header = (
        f"{'Scheme':<32} {'bits/dim':>8} {'RMSE':>10} {'CosSim':>8} "
        f"{'RelErr':>8} {'Mem4K(MB)':>10} {'Vec/s':>10}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r.name:<32} {r.bits_per_dim:>8.1f} {r.rmse:>10.6f} "
            f"{r.cosine_sim_mean:>8.4f} {r.relative_error_mean:>8.4f} "
            f"{r.memory_mb_4k:>10.3f} {r.throughput_vps:>10.0f}"
        )

    if len(results) >= 2:
        r0 = results[0]
        for r in results[1:]:
            pct = (r.rmse - r0.rmse) / r0.rmse * 100
            sign = "+" if pct > 0 else ""
            print(f"  {r.name} vs {r0.name}: RMSE {sign}{pct:.1f}%")

    print()


def save_results(results: List[SchemeResult], path: str | Path) -> None:
    """Save benchmark results as JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
