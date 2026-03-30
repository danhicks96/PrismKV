#!/usr/bin/env python
"""
run_e2e_benchmark.py — CLI for the PrismKV end-to-end benchmark.

Usage examples
--------------
# Synthetic data, default config (GPT-2 style: 12 layers, 12 heads, d=64)
python scripts/run_e2e_benchmark.py

# Use real KV vectors from a GPT-2 calibration run
python scripts/run_e2e_benchmark.py --kv-file results/kv_vectors.pt

# Custom config
python scripts/run_e2e_benchmark.py --head-dim 128 --n-heads 16 --n-layers 32 \\
    --bits 3 4 5 --context-lengths 4096 16384 32768

# Also measure pseudo-perplexity (requires transformers, downloads GPT-2)
python scripts/run_e2e_benchmark.py --pseudo-ppl
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prismkv.eval.e2e_benchmark import (
    run_e2e_benchmark,
    print_e2e_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PrismKV end-to-end benchmark: memory footprint + quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--kv-file", type=str, default=None,
                        help="Path to (N, head_dim) float32 tensor (.pt) for quality eval. "
                             "If omitted, synthetic anisotropic Gaussians are used.")
    parser.add_argument("--head-dim", type=int, default=64,
                        help="Attention head dimension (default: 64)")
    parser.add_argument("--n-heads", type=int, default=12,
                        help="Number of KV attention heads (default: 12)")
    parser.add_argument("--n-layers", type=int, default=12,
                        help="Number of transformer layers (default: 12)")
    parser.add_argument("--n-synthetic", type=int, default=10000,
                        help="Number of synthetic vectors if --kv-file not given (default: 10000)")
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4, 5],
                        help="Bit budgets to test (default: 3 4 5)")
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[1024, 4096, 16384],
                        help="Context lengths for memory table (default: 1024 4096 16384)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-json", type=str, default=None,
                        help="Save quality results to JSON file")
    parser.add_argument("--pseudo-ppl", action="store_true",
                        help="Also measure pseudo-perplexity (requires transformers + GPT-2 download)")
    parser.add_argument("--ppl-model", type=str, default="gpt2",
                        help="HuggingFace model name for pseudo-perplexity (default: gpt2)")
    parser.add_argument("--ppl-tokens", type=int, default=256,
                        help="Number of tokens to evaluate for pseudo-perplexity (default: 256)")
    args = parser.parse_args()

    kv_vectors = None
    if args.kv_file is not None:
        kv_vectors = torch.load(args.kv_file, map_location="cpu").float()
        print(f"Loaded KV vectors: {kv_vectors.shape} from {args.kv_file}")

    report = run_e2e_benchmark(
        kv_vectors=kv_vectors,
        head_dim=args.head_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_synthetic=args.n_synthetic,
        context_lengths=args.context_lengths,
        bits_configs=args.bits,
        seed=args.seed,
    )

    print_e2e_table(report)

    if args.save_json:
        import dataclasses, json as _json
        out = {
            "memory_profiles": [dataclasses.asdict(p) for p in report.memory_profiles],
            "quality_results": [dataclasses.asdict(r) for r in report.quality_results],
            "config": {
                "head_dim": report.head_dim,
                "n_heads": report.n_heads,
                "n_layers": report.n_layers,
            },
        }
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            _json.dump(out, f, indent=2)
        print(f"Results saved to {args.save_json}")

    if args.pseudo_ppl:
        print("Measuring pseudo-perplexity (this may take several minutes)...")
        from prismkv.eval.e2e_benchmark import measure_pseudo_perplexity
        ppl = measure_pseudo_perplexity(
            model_name=args.ppl_model,
            n_tokens=args.ppl_tokens,
            bits_configs=args.bits,
            seed=args.seed,
        )
        print("\nPseudo-perplexity (avg cross-entropy, nats/token):")
        fp16_loss = ppl.get("fp16", float("nan"))
        for name, loss in sorted(ppl.items()):
            marker = " ← baseline" if name == "fp16" else f"  Δ={loss - fp16_loss:+.4f}"
            print(f"  {name:<20} {loss:.4f}{marker}")


if __name__ == "__main__":
    main()
