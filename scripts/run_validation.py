#!/usr/bin/env python
"""
run_validation.py — One-shot full validation pipeline for PrismKV.

Runs:
  1. KV collection (skipped if kv_data/ already populated)
  2. Codebook training (skipped if kv_data/codebook_gpt2_all_layers.npz exists)
  3. Per-layer benchmark (all 12 GPT-2 layers)
  4. Adaptive allocation E2E
  5. Bias correction quality check
  6. Pseudo-perplexity (optional, --pseudo-ppl flag)

Writes results/validation_report.json.

Usage::

    # Full run (requires transformers, ~15 min on CPU)
    python scripts/run_validation.py

    # Skip pseudo-perplexity (faster, ~5 min)
    python scripts/run_validation.py --no-pseudo-ppl

    # Use pre-collected KV data
    python scripts/run_validation.py --kv-dir kv_data

    # Limit pseudo-ppl tokens (CI mode)
    python scripts/run_validation.py --ppl-tokens 64

Author: Dan Hicks (github.com/danhicks96)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path

# Make sure we can import prismkv from the source tree
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import torch


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PrismKV full validation pipeline")
    p.add_argument(
        "--kv-dir", default="kv_data",
        help="Directory containing pre-collected KV tensors (default: kv_data/)",
    )
    p.add_argument(
        "--output", default="results/validation_report.json",
        help="Path to write JSON report (default: results/validation_report.json)",
    )
    p.add_argument(
        "--model", default="gpt2",
        help="HuggingFace model name for collection + ppl (default: gpt2)",
    )
    p.add_argument(
        "--pseudo-ppl", dest="pseudo_ppl", action="store_true", default=True,
        help="Run pseudo-perplexity evaluation (requires transformers)",
    )
    p.add_argument(
        "--no-pseudo-ppl", dest="pseudo_ppl", action="store_false",
        help="Skip pseudo-perplexity evaluation",
    )
    p.add_argument(
        "--ppl-tokens", type=int, default=256,
        help="Number of tokens for pseudo-perplexity (default: 256; use 64 for CI)",
    )
    p.add_argument(
        "--bits", type=int, default=4,
        help="Bit budget to evaluate at (default: 4)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------


def _collect_kv_data(kv_dir: Path, model_name: str) -> None:
    """Collect GPT-2 KV data for all 12 layers if not already present."""
    print(f"[collect] Collecting KV data to {kv_dir}/")
    kv_dir.mkdir(exist_ok=True)

    from prismkv.eval.kv_collector import KVCollector

    text = (
        "Call me Ishmael. Some years ago—never mind how long precisely—having "
        "little money in my purse, and nothing particular to interest me on "
        "shore, I thought I would sail about a little and see the watery part "
        "of the world. "
    ) * 50   # ~500 tokens × 50 = enough calibration data

    collector = KVCollector(model_name, device="cpu", pad_dim=True)
    layer_indices = list(range(collector.n_layers))
    collected = collector.collect(text, layer_indices=layer_indices)

    all_keys = []
    for l in layer_indices:
        keys = collected[f"layer_{l}"]["keys"]
        vals = collected[f"layer_{l}"]["values"]
        torch.save(keys, kv_dir / f"gpt2_layer_{l}_keys.pt")
        torch.save(vals, kv_dir / f"gpt2_layer_{l}_values.pt")
        all_keys.append(keys)
        print(f"  layer {l:2d}: {keys.shape[0]} vectors")

    all_keys_tensor = torch.cat(all_keys, dim=0)
    torch.save(all_keys_tensor, kv_dir / "gpt2_all_layers_keys.pt")
    print(f"[collect] Done. Total KV vectors: {all_keys_tensor.shape[0]}")


def _train_codebook(kv_dir: Path, bits: int) -> Path:
    """Train multi-layer learned codebook; returns path to .npz file."""
    cb_path = kv_dir / "codebook_gpt2_all_layers.npz"
    if cb_path.exists():
        print(f"[codebook] Using existing codebook: {cb_path}")
        return cb_path

    print("[codebook] Training multi-layer learned codebook...")
    from prismkv import StackedPlaneQuantizer
    from prismkv.quantizer.learned_codebook import LearnedSliceCodebook

    all_keys = torch.load(kv_dir / "gpt2_all_layers_keys.pt", weights_only=True).float()
    dim = all_keys.shape[1]

    q = StackedPlaneQuantizer(dim=dim, bits_z=bits, bits_r=bits, bits_theta=bits, seed=42)
    q.calibrate(all_keys)
    rotated = all_keys @ q.R.T
    K = 2 ** (q.bits_r + q.bits_theta)

    cb = LearnedSliceCodebook.train(
        rotated_vectors=rotated,
        z_idx=q.z_idx, x_idx=q.x_idx, y_idx=q.y_idx,
        z_min=q.z_min, z_max=q.z_max, r_max=q.r_max,
        bins_z=q.bins_z, K=K, max_iter=30, seed=0,
    )
    cb.save(cb_path)
    print(f"[codebook] Saved to {cb_path}")
    return cb_path


# ---------------------------------------------------------------------------
# Per-layer benchmark
# ---------------------------------------------------------------------------


def _run_per_layer_benchmark(
    kv_dir: Path, cb_path: Path, bits: int
) -> dict:
    """Run run_benchmark on all 12 GPT-2 layers. Returns per-layer results."""
    from prismkv.eval.benchmark import run_benchmark
    from prismkv.eval.kv_collector import pad_to_multiple_of_3

    print("[benchmark] Running per-layer benchmark (all 12 layers)...")
    layer_results = {}
    for layer_idx in range(12):
        keys_file = kv_dir / f"gpt2_layer_{layer_idx}_keys.pt"
        if not keys_file.exists():
            continue
        keys = torch.load(keys_file, weights_only=True).float()
        keys_padded = pad_to_multiple_of_3(keys)

        results = run_benchmark(
            keys_padded, bits=bits, codebook_path=str(cb_path), original_dim=64
        )
        layer_results[f"layer_{layer_idx}"] = [
            {"name": r.name, "rmse": r.rmse, "cosine_sim_mean": r.cosine_sim_mean,
             "bits_per_dim": r.bits_per_dim}
            for r in results
        ]
        r_uniform = next((r for r in results if "uniform" in r.name and "3D" in r.name), None)
        r_learned = next((r for r in results if "learned" in r.name), None)
        rmse_u = r_uniform.rmse if r_uniform else float("nan")
        rmse_l = r_learned.rmse if r_learned else float("nan")
        ratio = rmse_l / rmse_u if rmse_u > 0 else float("nan")
        print(f"  layer {layer_idx:2d}: uniform RMSE={rmse_u:.4f}  learned RMSE={rmse_l:.4f}  ratio={ratio:.3f}")

    return layer_results


# ---------------------------------------------------------------------------
# Bias correction
# ---------------------------------------------------------------------------


def _run_bias_correction_check(kv_dir: Path) -> dict:
    """Check bias correction effectiveness on layer-0 keys."""
    from prismkv.quantizer.stacked_plane import StackedPlaneQuantizer
    from prismkv.eval.kv_collector import pad_to_multiple_of_3

    print("[bias] Checking bias correction on layer-0 keys...")
    keys = torch.load(kv_dir / "gpt2_layer_0_keys.pt", weights_only=True).float()
    keys_padded = pad_to_multiple_of_3(keys)

    q = StackedPlaneQuantizer(dim=keys_padded.shape[1], bits_z=4, bits_r=4, bits_theta=4, seed=42)
    q.calibrate(keys_padded)

    codes = q.encode(keys_padded)
    recon_before = q.decode(codes)
    mae_before = (recon_before - keys_padded).abs().mean().item()

    q.calibrate_bias(keys_padded)
    codes = q.encode(keys_padded)
    recon_after = q.decode(codes)
    mae_after = (recon_after - keys_padded).abs().mean().item()

    bias_per_dim = (recon_after - keys_padded).mean(dim=0)
    max_abs_bias = bias_per_dim.abs().max().item()

    result = {
        "mae_before": mae_before,
        "mae_after": mae_after,
        "max_abs_bias_per_dim": max_abs_bias,
        "passes": max_abs_bias < 0.1,
    }
    print(f"  MAE before={mae_before:.4f}  after={mae_after:.4f}  max_abs_bias={max_abs_bias:.4f}  {'PASS' if result['passes'] else 'FAIL'}")
    return result


# ---------------------------------------------------------------------------
# Adaptive allocation
# ---------------------------------------------------------------------------


def _run_adaptive_allocation(kv_dir: Path, bits: int) -> dict:
    """Run adaptive allocation E2E on layer-0 keys with synthetic entropy."""
    from prismkv.quantizer.bit_alloc import BitAllocator
    from prismkv.quantizer.stacked_plane import StackedPlaneQuantizer
    from prismkv.eval.kv_collector import pad_to_multiple_of_3

    print("[adaptive] Running adaptive allocation E2E...")
    torch.manual_seed(42)
    entropy = torch.rand(12, 12) * 2.5 + 0.5  # synthetic heterogeneous

    alloc = BitAllocator(entropy, target_avg_bits_per_dim=float(bits))
    alloc.compute()
    mean_bits = alloc.mean_bits_per_dim
    target_delta = abs(mean_bits - bits)

    layer0_cfg = alloc.to_prism_configs()[0]
    keys = torch.load(kv_dir / "gpt2_layer_0_keys.pt", weights_only=True).float()
    keys_padded = pad_to_multiple_of_3(keys)

    q = StackedPlaneQuantizer(
        dim=keys_padded.shape[1],
        bits_z=layer0_cfg.bits_z,
        bits_r=layer0_cfg.bits_r,
        bits_theta=layer0_cfg.bits_theta,
        seed=42,
    )
    q.calibrate(keys_padded)
    recon = q.decode(q.encode(keys_padded))
    rmse = (recon - keys_padded).pow(2).mean(dim=1).sqrt().mean().item()

    result = {
        "target_bits": bits,
        "mean_bits_actual": mean_bits,
        "target_delta": target_delta,
        "layer0_config": {"bits_z": layer0_cfg.bits_z, "bits_r": layer0_cfg.bits_r, "bits_theta": layer0_cfg.bits_theta},
        "layer0_rmse": rmse,
        "passes": target_delta < 0.05,
    }
    print(f"  mean_bits={mean_bits:.4f}  delta={target_delta:.4f}  layer0_rmse={rmse:.4f}  {'PASS' if result['passes'] else 'FAIL'}")
    return result


# ---------------------------------------------------------------------------
# Pseudo-perplexity
# ---------------------------------------------------------------------------


def _run_pseudo_ppl(model_name: str, n_tokens: int, bits: int) -> dict:
    """Measure pseudo-perplexity delta at specified bit budget."""
    print(f"[ppl] Measuring pseudo-perplexity ({n_tokens} tokens, {bits} bits)...")
    from prismkv.eval.e2e_benchmark import measure_pseudo_perplexity

    scores = measure_pseudo_perplexity(
        model_name=model_name,
        n_tokens=n_tokens,
        bits_configs=[bits],
    )
    fp16_ppl = scores.get("fp16", float("nan"))
    prism_ppl = scores.get(f"prismkv_{bits}bit", float("nan"))
    delta = prism_ppl - fp16_ppl if math.isfinite(fp16_ppl) and math.isfinite(prism_ppl) else float("nan")

    result = {
        "fp16_nats": fp16_ppl,
        f"prismkv_{bits}bit_nats": prism_ppl,
        "delta_nats": delta,
        "passes": math.isfinite(delta) and delta < 1.5,
    }
    print(f"  fp16={fp16_ppl:.4f}  prismkv_{bits}bit={prism_ppl:.4f}  delta={delta:.4f}  {'PASS' if result['passes'] else 'FAIL'}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    kv_dir = Path(args.kv_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)

    t_start = time.time()
    report: dict = {
        "model": args.model,
        "bits": args.bits,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Step 1: Collect KV data (or skip if already present)
    if not kv_dir.exists() or not (kv_dir / "gpt2_all_layers_keys.pt").exists():
        _collect_kv_data(kv_dir, args.model)
    else:
        print(f"[collect] Using existing KV data in {kv_dir}/")

    # Step 2: Train codebook
    cb_path = _train_codebook(kv_dir, args.bits)

    # Step 3: Per-layer benchmark
    report["per_layer_benchmark"] = _run_per_layer_benchmark(kv_dir, cb_path, args.bits)

    # Step 4: Adaptive allocation
    report["adaptive_allocation"] = _run_adaptive_allocation(kv_dir, args.bits)

    # Step 5: Bias correction
    report["bias_correction"] = _run_bias_correction_check(kv_dir)

    # Step 6: Pseudo-perplexity (optional)
    if args.pseudo_ppl:
        try:
            report["pseudo_perplexity"] = _run_pseudo_ppl(
                args.model, args.ppl_tokens, args.bits
            )
        except ImportError as e:
            print(f"[ppl] Skipped — {e}")
            report["pseudo_perplexity"] = {"skipped": str(e)}

    # Summary
    elapsed = time.time() - t_start
    report["elapsed_seconds"] = round(elapsed, 1)

    passes = {
        "adaptive_allocation": report["adaptive_allocation"].get("passes", False),
        "bias_correction": report["bias_correction"].get("passes", False),
    }
    if "pseudo_perplexity" in report and "passes" in report["pseudo_perplexity"]:
        passes["pseudo_perplexity"] = report["pseudo_perplexity"]["passes"]

    report["summary"] = {
        "all_checks_pass": all(passes.values()),
        "checks": passes,
    }

    output_path.write_text(json.dumps(report, indent=2))
    print(f"\n[done] Report written to {output_path} ({elapsed:.0f}s)")
    print(f"       All checks pass: {report['summary']['all_checks_pass']}")


if __name__ == "__main__":
    main()
