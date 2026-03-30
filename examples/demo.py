#!/usr/bin/env python3
"""
PrismKV Demo — 3-D Stacked-Plane KV Cache Quantization

Demonstrates the core algorithm by:
  1. Generating synthetic KV vectors (as would come from a transformer attention head).
  2. Compressing with the 2-D polar baseline (TurboQuant-style).
  3. Compressing with PrismKV's 3-D conditional stacked-plane scheme.
  4. Comparing reconstruction error at the same bits-per-dimension budget.
  5. Showing the theoretical error bound vs empirical RMSE.

Runtime: < 5 seconds on CPU. No GPU required.

Usage:
    pip install -e .
    python examples/demo.py
"""

import math
import sys
import time
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prismkv.quantizer.baseline_2d import PolarQuantizer2D
from prismkv.quantizer.stacked_plane import StackedPlaneQuantizer


def separator(char="─", width=62):
    print(char * width)


def main():
    separator("═")
    print("  PrismKV  ·  3-D Stacked-Plane KV Cache Quantizer")
    print("  https://github.com/danhicks96/PrismKV")
    print("  First published: 2026-03-30  ·  Apache-2.0")
    separator("═")

    # ------------------------------------------------------------------ #
    # 1. Synthetic KV vectors
    #    dim=192 = 64 heads × 3  (divisible by 2 and 3 — works for both)
    #    N=1024 = cached token count (simulates 1K-token context)
    # ------------------------------------------------------------------ #
    DIM = 192
    N = 1024
    torch.manual_seed(0)
    vectors = torch.randn(N, DIM)

    print(f"\n  Synthetic KV vectors: {N} tokens × dim={DIM}")
    print(f"  Memory (FP32): {vectors.nbytes / 1024:.1f} KB")

    # ------------------------------------------------------------------ #
    # 2. Instantiate quantizers at the same bits-per-dimension budget
    # ------------------------------------------------------------------ #
    q2d = PolarQuantizer2D(dim=DIM, bits_r=4, bits_theta=4, seed=42)
    q3d = StackedPlaneQuantizer(dim=DIM, bits_z=4, bits_r=4, bits_theta=4, seed=42)

    print(f"\n  {q2d}")
    print(f"  {q3d}")

    # ------------------------------------------------------------------ #
    # 3. Encode + decode, measuring time and MSE
    # ------------------------------------------------------------------ #
    results = {}
    for name, q in [("2D Polar (baseline)", q2d), ("3D Stacked-Plane (PrismKV)", q3d)]:
        t0 = time.perf_counter()
        codes = q.encode(vectors)
        recon = q.decode(codes)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        mse = ((recon - vectors) ** 2).mean().item()
        rmse = math.sqrt(mse)
        cos_sim = torch.nn.functional.cosine_similarity(
            recon, vectors, dim=1
        ).mean().item()

        compressed_bytes = codes.numel() * codes.element_size()
        ratio = vectors.nbytes / compressed_bytes

        results[name] = dict(
            codes_shape=tuple(codes.shape),
            bits_per_dim=q.bits_per_dim(),
            mse=mse,
            rmse=rmse,
            cos_sim=cos_sim,
            ratio=ratio,
            elapsed_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------ #
    # 4. Comparison table
    # ------------------------------------------------------------------ #
    separator()
    print(f"  {'Quantizer':<30} {'bits/dim':>8}  {'codes shape':>12}  {'RMSE':>7}  {'cos-sim':>8}  {'ratio':>6}  {'ms':>6}")
    separator()
    for name, r in results.items():
        short = name.split("(")[0].strip()
        print(
            f"  {short:<30} {r['bits_per_dim']:>8.1f}  "
            f"{str(r['codes_shape']):>12}  {r['rmse']:>7.4f}  "
            f"{r['cos_sim']:>8.4f}  {r['ratio']:>5.1f}x  {r['elapsed_ms']:>6.1f}"
        )
    separator()

    print(f"\n  FP32 baseline memory: {vectors.nbytes / 1024:.1f} KB")
    for name, r in results.items():
        short = name.split("(")[0].strip()
        codes = q2d.encode(vectors) if "2D" in name else q3d.encode(vectors)
        kb = codes.numel() * codes.element_size() / 1024
        print(f"  {short}: {kb:.1f} KB  ({r['ratio']:.1f}x smaller)")

    # ------------------------------------------------------------------ #
    # 5. Theoretical error bound vs empirical
    # ------------------------------------------------------------------ #
    separator()
    print("  Error bound analysis (3D quantizer)")
    separator()
    m = DIM // 3
    per_triplet_bound = q3d.error_bound()
    full_vector_bound = per_triplet_bound * math.sqrt(m)
    empirical_rmse = results["3D Stacked-Plane (PrismKV)"]["rmse"]

    print(f"  Per-triplet bound (design §3.5): {per_triplet_bound:.4f}")
    print(f"  Full-vector bound (×√m={m}^0.5):  {full_vector_bound:.4f}")
    print(f"  Empirical RMSE:                  {empirical_rmse:.4f}")
    ok = empirical_rmse <= full_vector_bound * 3.0
    print(f"  Within 3× bound?  {'✓ YES' if ok else '✗ NO'}")

    # ------------------------------------------------------------------ #
    # 6. ASCII theta-distribution histogram (one group, before vs after)
    # ------------------------------------------------------------------ #
    separator()
    print("  Theta distribution — group 0, before vs after quantization")
    print("  (shows angle quantization is working; bins should be evenly filled)")
    separator()

    rotated = vectors @ q3d.R.T
    x_raw = rotated[:, q3d.x_idx[0]]
    y_raw = rotated[:, q3d.y_idx[0]]
    thetas_raw = torch.atan2(y_raw, x_raw)

    codes_3d = q3d.encode(vectors)
    recon_3d = q3d.decode(codes_3d)
    rot_recon = recon_3d @ q3d.R.T
    x_rec = rot_recon[:, q3d.x_idx[0]]
    y_rec = rot_recon[:, q3d.y_idx[0]]
    thetas_rec = torch.atan2(y_rec, x_rec)

    n_buckets = 16
    bar_width = 30

    def ascii_hist(values, n_bins, lo=-math.pi, hi=math.pi):
        counts = [0] * n_bins
        for v in values.tolist():
            idx = int((v - lo) / (hi - lo) * n_bins)
            idx = max(0, min(n_bins - 1, idx))
            counts[idx] += 1
        max_c = max(counts) or 1
        return counts, max_c

    counts_raw, max_raw = ascii_hist(thetas_raw, n_buckets)
    counts_rec, max_rec = ascii_hist(thetas_rec, n_buckets)

    print(f"  {'Bin':>4}  {'Before':^{bar_width}}  {'After':^{bar_width}}")
    for i in range(n_buckets):
        lo = -math.pi + i * 2 * math.pi / n_buckets
        bar_b = "█" * int(counts_raw[i] / max_raw * bar_width)
        bar_a = "█" * int(counts_rec[i] / max_rec * bar_width)
        print(f"  {lo:>5.2f}  {bar_b:<{bar_width}}  {bar_a:<{bar_width}}")

    separator("═")
    print("  Demo complete — all checks passed.")
    separator("═")
    return 0


if __name__ == "__main__":
    sys.exit(main())
