#!/usr/bin/env python3
"""
find_optimal_bit_split.py — Grid search over (bits_z, bits_r, bits_theta) splits.

Enumerates all valid 3-way splits of a fixed bits-per-triplet budget and
evaluates RMSE + CosSim on real GPT-2 KV data.  Also benchmarks 2D polar and
the Lloyd-Max z upgrade for comparison.

Usage
-----
python scripts/find_optimal_bit_split.py \
    --kv-file kv_data/gpt2_all_layers_keys.pt \
    --codebook kv_data/codebook_gpt2_all_layers.npz \
    --bits-per-dim 4 \
    --save-json results/bit_split_search.json

Arguments
---------
--kv-file       Path to a .pt file containing a (N, dim) float32 tensor of KV vectors.
--codebook      Path to a .npz codebook (optional; used for the learned baseline).
--bits-per-dim  Target bits/dim (default: 4).  bits-per-triplet = 3 * bits-per-dim.
--n-samples     Max vectors to use (default: 10000; -1 = all).
--save-json     Where to write the ranking table.  Default: results/bit_split_search.json.
--percentile-clip  Percentile clip for calibration (default: 0.005).
"""

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import torch

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prismkv.quantizer.stacked_plane import StackedPlaneQuantizer
from prismkv.quantizer.baseline_2d import PolarQuantizer2D


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rmse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    return ((original - reconstructed) ** 2).mean().sqrt().item()


def cos_sim(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    orig_n = original / (original.norm(dim=-1, keepdim=True) + 1e-8)
    recon_n = reconstructed / (reconstructed.norm(dim=-1, keepdim=True) + 1e-8)
    return (orig_n * recon_n).sum(dim=-1).mean().item()


def z_mse(quantizer: StackedPlaneQuantizer, vectors: torch.Tensor) -> float:
    """MSE on the z component only."""
    rotated = vectors @ quantizer.R.T
    z = rotated[:, quantizer.z_idx].reshape(-1)
    codes = quantizer.encode(vectors)
    bits_flat = quantizer.bits_r + quantizer.bits_theta
    i_z = (codes >> bits_flat).reshape(-1)
    if quantizer._lloyd_max_z is not None:
        z_q = quantizer._lloyd_max_z.decode(i_z)
    else:
        delta_z = (quantizer.z_max - quantizer.z_min) / quantizer.bins_z
        z_q = quantizer.z_min + (i_z.float() + 0.5) * delta_z
    return ((z.float() - z_q) ** 2).mean().item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search over (bz, br, bt) bit splits.")
    parser.add_argument("--kv-file", required=True, help=".pt file with (N, dim) float32 tensor")
    parser.add_argument("--codebook", default=None, help=".npz codebook (optional)")
    parser.add_argument("--bits-per-dim", type=float, default=4.0)
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--save-json", default="results/bit_split_search.json")
    parser.add_argument("--percentile-clip", type=float, default=0.005)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading KV data from {args.kv_file} ...", flush=True)
    vectors: torch.Tensor = torch.load(args.kv_file, map_location="cpu", weights_only=True)
    if vectors.dtype != torch.float32:
        vectors = vectors.float()
    if vectors.ndim != 2:
        raise ValueError(f"Expected (N, dim) tensor, got {vectors.shape}")

    N, dim = vectors.shape
    print(f"  Loaded {N:,} vectors of dim={dim}")

    if args.n_samples > 0 and N > args.n_samples:
        idx = torch.randperm(N)[:args.n_samples]
        vectors = vectors[idx]
        N = args.n_samples
        print(f"  Sub-sampled to {N:,} vectors")

    bits_per_triplet = int(round(args.bits_per_dim * 3))
    print(f"  Budget: {args.bits_per_dim} bits/dim → {bits_per_triplet} bits/triplet")

    results = []

    # ------------------------------------------------------------------
    # 2D Polar baseline
    # ------------------------------------------------------------------
    print("\n[2D Polar baseline]")
    bits_2d = bits_per_triplet // 2   # split evenly across r, theta for 2 dims
    # 2D groups vectors into pairs; dim must be even; pad if needed
    dim_2d = dim if dim % 2 == 0 else dim + 1
    vecs_2d = vectors if dim == dim_2d else torch.nn.functional.pad(vectors, (0, 1))
    q2d = PolarQuantizer2D(dim=dim_2d, bits_r=bits_2d, bits_theta=bits_2d, seed=42)
    # Calibrate r_max from data (PolarQuantizer2D has no calibrate(); set directly)
    rotated_2d = vecs_2d @ q2d.R.T
    x2d = rotated_2d[:, 0::2]
    y2d = rotated_2d[:, 1::2]
    q2d.r_max = torch.sqrt(x2d ** 2 + y2d ** 2).max().item()
    codes_2d = q2d.encode(vecs_2d)
    recon_2d = q2d.decode(codes_2d)[:, :dim]
    r2d = rmse(vectors, recon_2d)
    c2d = cos_sim(vectors, recon_2d)
    print(f"  2D Polar ({bits_2d}+{bits_2d}): RMSE={r2d:.4f}  CosSim={c2d:.4f}")
    results.append({
        "config": f"2D Polar (br={bits_2d}, bt={bits_2d})",
        "bits_per_dim": args.bits_per_dim,
        "bz": None, "br": bits_2d, "bt": bits_2d,
        "rmse": round(r2d, 6),
        "cossim": round(c2d, 6),
        "lloyd_max_z": False,
        "learned_codebook": False,
        "rank_score": r2d,
    })

    # ------------------------------------------------------------------
    # 3D grid search
    # ------------------------------------------------------------------
    print(f"\n[3D grid search over (bz, br, bt) with sum={bits_per_triplet}, each in [1,8]]")
    triples = [
        (bz, br, bt)
        for bz in range(1, 9)
        for br in range(1, 9)
        for bt in range(1, 9)
        if bz + br + bt == bits_per_triplet
    ]
    print(f"  {len(triples)} configurations to evaluate")

    for bz, br, bt in triples:
        q = StackedPlaneQuantizer(dim=dim, bits_z=bz, bits_r=br, bits_theta=bt, seed=42)
        q.calibrate(vectors, percentile_clip=args.percentile_clip)
        codes = q.encode(vectors)
        recon = q.decode(codes)
        r = rmse(vectors, recon)
        c = cos_sim(vectors, recon)
        results.append({
            "config": f"3D Uniform (bz={bz}, br={br}, bt={bt})",
            "bits_per_dim": (bz + br + bt) / 3,
            "bz": bz, "br": br, "bt": bt,
            "rmse": round(r, 6),
            "cossim": round(c, 6),
            "lloyd_max_z": False,
            "learned_codebook": False,
            "rank_score": r,
        })

    # Sort 3D uniform results to find best split
    uniform_3d = [x for x in results if x["bz"] is not None and not x["lloyd_max_z"]]
    best_uniform = min(uniform_3d, key=lambda x: x["rmse"])
    print(f"  Best 3D uniform split: bz={best_uniform['bz']}, br={best_uniform['br']}, "
          f"bt={best_uniform['bt']}  RMSE={best_uniform['rmse']:.4f}")

    # ------------------------------------------------------------------
    # 3D uniform equal-split (4+4+4) — explicit baseline
    # ------------------------------------------------------------------
    bz_eq = br_eq = bt_eq = bits_per_triplet // 3
    if (bz_eq, br_eq, bt_eq) not in [(x["bz"], x["br"], x["bt"]) for x in uniform_3d]:
        # Add if not already included
        q_eq = StackedPlaneQuantizer(dim=dim, bits_z=bz_eq, bits_r=br_eq, bits_theta=bt_eq, seed=42)
        q_eq.calibrate(vectors, percentile_clip=args.percentile_clip)
        codes_eq = q_eq.encode(vectors)
        recon_eq = q_eq.decode(codes_eq)
        r_eq = rmse(vectors, recon_eq)
        c_eq = cos_sim(vectors, recon_eq)
        print(f"\n[3D uniform equal split ({bz_eq}+{br_eq}+{bt_eq})]")
        print(f"  RMSE={r_eq:.4f}  CosSim={c_eq:.4f}")
    else:
        # Find it in results
        eq_entry = next(
            x for x in uniform_3d
            if x["bz"] == bz_eq and x["br"] == br_eq and x["bt"] == bt_eq
        )
        r_eq = eq_entry["rmse"]
        print(f"\n[3D uniform equal split ({bz_eq}+{br_eq}+{bt_eq})]")
        print(f"  RMSE={r_eq:.4f}  (from grid)")

    # ------------------------------------------------------------------
    # Best split + Lloyd-Max z
    # ------------------------------------------------------------------
    bz_b, br_b, bt_b = best_uniform["bz"], best_uniform["br"], best_uniform["bt"]
    print(f"\n[Best split + Lloyd-Max z (bz={bz_b}, br={br_b}, bt={bt_b})]")
    q_lm = StackedPlaneQuantizer(dim=dim, bits_z=bz_b, bits_r=br_b, bits_theta=bt_b, seed=42)
    q_lm.calibrate(vectors, percentile_clip=args.percentile_clip)
    q_lm.calibrate_lloyd_max_z(vectors)
    codes_lm = q_lm.encode(vectors)
    recon_lm = q_lm.decode(codes_lm)
    r_lm = rmse(vectors, recon_lm)
    c_lm = cos_sim(vectors, recon_lm)
    z_mse_lm = z_mse(q_lm, vectors)
    print(f"  RMSE={r_lm:.4f}  CosSim={c_lm:.4f}  z-MSE={z_mse_lm:.6f}")

    # Uniform z MSE for same split (for comparison)
    q_uni_b = StackedPlaneQuantizer(dim=dim, bits_z=bz_b, bits_r=br_b, bits_theta=bt_b, seed=42)
    q_uni_b.calibrate(vectors, percentile_clip=args.percentile_clip)
    z_mse_uni = z_mse(q_uni_b, vectors)
    z_improvement = (z_mse_uni - z_mse_lm) / z_mse_uni * 100
    print(f"  z-MSE uniform={z_mse_uni:.6f}  → Lloyd-Max improvement: {z_improvement:.1f}%")

    results.append({
        "config": f"3D Best-split+LloydMax (bz={bz_b}, br={br_b}, bt={bt_b})",
        "bits_per_dim": (bz_b + br_b + bt_b) / 3,
        "bz": bz_b, "br": br_b, "bt": bt_b,
        "rmse": round(r_lm, 6),
        "cossim": round(c_lm, 6),
        "lloyd_max_z": True,
        "learned_codebook": False,
        "z_mse_lloyd": round(z_mse_lm, 8),
        "z_mse_uniform": round(z_mse_uni, 8),
        "z_mse_improvement_pct": round(z_improvement, 2),
        "rank_score": r_lm,
    })

    # ------------------------------------------------------------------
    # Equal split (4+4+4) + Lloyd-Max z
    # ------------------------------------------------------------------
    print(f"\n[Equal split ({bz_eq}+{br_eq}+{bt_eq}) + Lloyd-Max z]")
    q_lm_eq = StackedPlaneQuantizer(dim=dim, bits_z=bz_eq, bits_r=br_eq, bits_theta=bt_eq, seed=42)
    q_lm_eq.calibrate(vectors, percentile_clip=args.percentile_clip)
    q_lm_eq.calibrate_lloyd_max_z(vectors)
    codes_lm_eq = q_lm_eq.encode(vectors)
    recon_lm_eq = q_lm_eq.decode(codes_lm_eq)
    r_lm_eq = rmse(vectors, recon_lm_eq)
    c_lm_eq = cos_sim(vectors, recon_lm_eq)
    z_mse_lm_eq = z_mse(q_lm_eq, vectors)
    z_mse_uni_eq_entry = next(
        (x for x in uniform_3d if x["bz"] == bz_eq and x["br"] == br_eq and x["bt"] == bt_eq),
        None
    )
    print(f"  RMSE={r_lm_eq:.4f}  CosSim={c_lm_eq:.4f}  z-MSE={z_mse_lm_eq:.6f}")

    results.append({
        "config": f"3D Equal-split+LloydMax (bz={bz_eq}, br={br_eq}, bt={bt_eq})",
        "bits_per_dim": (bz_eq + br_eq + bt_eq) / 3,
        "bz": bz_eq, "br": br_eq, "bt": bt_eq,
        "rmse": round(r_lm_eq, 6),
        "cossim": round(c_lm_eq, 6),
        "lloyd_max_z": True,
        "learned_codebook": False,
        "z_mse_lloyd": round(z_mse_lm_eq, 8),
        "rank_score": r_lm_eq,
    })

    # ------------------------------------------------------------------
    # Learned codebook baseline (if provided)
    # ------------------------------------------------------------------
    if args.codebook and Path(args.codebook).exists():
        print(f"\n[3D Learned codebook (from {args.codebook})]")
        q_learned = StackedPlaneQuantizer(dim=dim, bits_z=bz_eq, bits_r=br_eq, bits_theta=bt_eq, seed=42)
        try:
            q_learned.load_codebooks(args.codebook)
            codes_learned = q_learned.encode(vectors)
            recon_learned = q_learned.decode(codes_learned)
            r_learned = rmse(vectors, recon_learned)
            c_learned = cos_sim(vectors, recon_learned)
            print(f"  RMSE={r_learned:.4f}  CosSim={c_learned:.4f}")
            results.append({
                "config": f"3D Learned codebook (bz={bz_eq}, br={br_eq}, bt={bt_eq})",
                "bits_per_dim": (bz_eq + br_eq + bt_eq) / 3,
                "bz": bz_eq, "br": br_eq, "bt": bt_eq,
                "rmse": round(r_learned, 6),
                "cossim": round(c_learned, 6),
                "lloyd_max_z": False,
                "learned_codebook": True,
                "rank_score": r_learned,
            })
        except Exception as e:
            print(f"  WARNING: Could not evaluate learned codebook: {e}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    # Sort by RMSE (ascending)
    ranked = sorted(results, key=lambda x: x["rank_score"])

    print("\n" + "=" * 70)
    print(f"{'Rank':<5} {'Config':<45} {'RMSE':>8} {'CosSim':>8}")
    print("-" * 70)
    for rank, entry in enumerate(ranked[:10], 1):
        print(f"#{rank:<4} {entry['config']:<45} {entry['rmse']:>8.4f} {entry['cossim']:>8.4f}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Z-MSE improvement check
    # ------------------------------------------------------------------
    lm_entries = [x for x in results if x.get("lloyd_max_z") and "z_mse_improvement_pct" in x]
    if lm_entries:
        best_lm = max(lm_entries, key=lambda x: x["z_mse_improvement_pct"])
        pct = best_lm["z_mse_improvement_pct"]
        passed = pct >= 15.0
        print(f"\nLloyd-Max z MSE improvement: {pct:.1f}%  {'[PASS ≥15%]' if passed else '[WARN <15%]'}")

    # Improvement vs equal-split uniform
    eq_uniform_entry = next(
        (x for x in results
         if x.get("bz") == bz_eq and x.get("br") == br_eq and x.get("bt") == bt_eq
         and not x["lloyd_max_z"] and not x["learned_codebook"]),
        None
    )
    if eq_uniform_entry:
        # Compare best 3D Lloyd-Max vs 3D uniform equal-split (same algorithm family)
        best_3d_lm = next(
            (x for x in ranked if x.get("lloyd_max_z") and x.get("bz") is not None),
            ranked[0]
        )
        improvement_vs_eq = (eq_uniform_entry["rmse"] - best_3d_lm["rmse"]) / eq_uniform_entry["rmse"] * 100
        passed_split = improvement_vs_eq >= 5.0
        print(f"Best 3D+LloydMax vs 3D equal-split uniform: {improvement_vs_eq:.1f}% RMSE improvement  "
              f"{'[PASS ≥5%]' if passed_split else '[WARN <5%]'}")

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    out = {
        "metadata": {
            "kv_file": args.kv_file,
            "n_vectors": N,
            "dim": dim,
            "bits_per_dim": args.bits_per_dim,
            "bits_per_triplet": bits_per_triplet,
            "percentile_clip": args.percentile_clip,
            "date": "2026-03-30",
        },
        "ranking": ranked,
        "best_config": ranked[0],
        "equal_split_uniform": eq_uniform_entry,
        "equal_split_lloyd_max": next(
            (x for x in results
             if x.get("bz") == bz_eq and x.get("lloyd_max_z")),
            None
        ),
    }

    save_path = Path(args.save_json)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
