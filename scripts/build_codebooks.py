"""
build_codebooks.py — Train and save per-z-slice k-means codebooks for PrismKV.

Usage
-----
# Synthetic anisotropic Gaussian data (no model required, <30s on 16 cores):
python3 scripts/build_codebooks.py --source synthetic --dim 192 --out cb_synthetic_192.npz

# GPT-2 real KV distributions (requires transformers + a saved calibration .pt file):
python3 scripts/build_codebooks.py --source file --kv-path kv_gpt2.pt --dim 64 --out cb_gpt2.npz

The output .npz can be loaded into StackedPlaneQuantizer via:
    q = StackedPlaneQuantizer(dim=192)
    q.load_codebooks("cb_synthetic_192.npz")
"""

import argparse
import math
import time
from pathlib import Path

import torch

from prismkv import StackedPlaneQuantizer
from prismkv.quantizer.learned_codebook import LearnedSliceCodebook


# ---------------------------------------------------------------------------
# Synthetic calibration data generator
# ---------------------------------------------------------------------------

def make_synthetic_vectors(
    n: int,
    dim: int,
    seed: int = 0,
    anisotropy: float = 3.0,
) -> torch.Tensor:
    """
    Generate N anisotropic Gaussian vectors of shape (N, dim).

    Dimensions alternate between two scales (1.0 and anisotropy) so that
    the z-coordinate has a different typical magnitude than (x, y).
    This mimics the non-isotropic structure of real KV distributions and
    gives learned codebooks a meaningful advantage to exploit.

    Parameters
    ----------
    n         : number of vectors
    dim       : vector dimension (must be divisible by 3)
    seed      : random seed for reproducibility
    anisotropy: scale factor applied to every third dim (z-dims)
    """
    if dim % 3 != 0:
        raise ValueError(f"dim must be divisible by 3, got {dim}")

    gen = torch.Generator().manual_seed(seed)
    vecs = torch.randn(n, dim, generator=gen)

    # Scale z-dims up to create inter-group correlation exploitable by learned CBs
    m = dim // 3
    z_cols = torch.arange(m) * 3          # indices 0, 3, 6, ...
    vecs[:, z_cols] *= anisotropy

    return vecs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PrismKV per-z-slice k-means codebooks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source", choices=["synthetic", "file"], default="synthetic",
        help="Data source: 'synthetic' generates anisotropic Gaussians; "
             "'file' loads a pre-saved .pt tensor produced by collect_kv_calibration.py",
    )
    parser.add_argument(
        "--kv-path", type=str, default=None,
        help="Path to .pt file containing a (N, dim) float32 tensor (required for --source file)",
    )
    parser.add_argument("--dim",        type=int,   default=192,   help="Vector dimension (must be divisible by 3)")
    parser.add_argument("--n-vecs",     type=int,   default=50_000, help="Calibration vectors (synthetic source only)")
    parser.add_argument("--bits-z",     type=int,   default=4,     help="Bits for z quantization")
    parser.add_argument("--bits-r",     type=int,   default=4,     help="Bits for radius quantization")
    parser.add_argument("--bits-theta", type=int,   default=4,     help="Bits for angle quantization")
    parser.add_argument("--max-iter",   type=int,   default=100,   help="k-means max iterations per bin")
    parser.add_argument("--seed",       type=int,   default=42,    help="Random seed")
    parser.add_argument("--out",        type=str,   default="codebook.npz", help="Output .npz path")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load / generate calibration vectors
    # ------------------------------------------------------------------
    print(f"[build_codebooks] source={args.source}, dim={args.dim}")

    if args.source == "synthetic":
        print(f"  generating {args.n_vecs:,} anisotropic Gaussian vectors …")
        t0 = time.perf_counter()
        vectors = make_synthetic_vectors(args.n_vecs, args.dim, seed=args.seed)
        print(f"  generated in {time.perf_counter() - t0:.1f}s")

    else:  # file
        if args.kv_path is None:
            parser.error("--kv-path is required when --source file")
        kv_path = Path(args.kv_path)
        if not kv_path.exists():
            raise FileNotFoundError(f"KV file not found: {kv_path}")
        print(f"  loading vectors from {kv_path} …")
        vectors = torch.load(str(kv_path), weights_only=True)
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2-D tensor (N, dim), got shape {tuple(vectors.shape)}")
        if vectors.shape[1] != args.dim:
            raise ValueError(
                f"Tensor dim {vectors.shape[1]} does not match --dim {args.dim}. "
                "Pass the correct --dim or re-export the calibration file."
            )
        print(f"  loaded {vectors.shape[0]:,} vectors")

    # ------------------------------------------------------------------
    # Build quantizer and calibrate ranges
    # ------------------------------------------------------------------
    print(f"  calibrating quantization ranges …")
    q = StackedPlaneQuantizer(
        dim=args.dim,
        bits_z=args.bits_z,
        bits_r=args.bits_r,
        bits_theta=args.bits_theta,
        seed=args.seed,
    )
    q.calibrate(vectors)
    print(f"  z ∈ [{q.z_min:.3f}, {q.z_max:.3f}],  r_max={q.r_max:.3f}")

    # ------------------------------------------------------------------
    # Apply global rotation (codebooks are trained in rotated space)
    # ------------------------------------------------------------------
    rotated = vectors @ q.R.T                              # (N, dim)

    # ------------------------------------------------------------------
    # Train codebooks
    # ------------------------------------------------------------------
    K = 2 ** (args.bits_r + args.bits_theta)
    bins_z = q.bins_z
    print(f"  training {bins_z} codebooks, K={K} centroids each …")
    t0 = time.perf_counter()

    cb = LearnedSliceCodebook.train(
        rotated_vectors=rotated,
        z_idx=q.z_idx,
        x_idx=q.x_idx,
        y_idx=q.y_idx,
        z_min=q.z_min,
        z_max=q.z_max,
        r_max=q.r_max,
        bins_z=bins_z,
        K=K,
        max_iter=args.max_iter,
        seed=args.seed,
    )

    elapsed = time.perf_counter() - t0
    print(f"  training complete in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Quick quality check: learned MSE vs uniform MSE
    # ------------------------------------------------------------------
    print("  running quality check (first 2000 vectors) …")
    sample = vectors[:2000]

    # Uniform encode/decode (no codebook)
    codes_u = q.encode(sample)
    recon_u = q.decode(codes_u)
    mse_uniform = ((sample - recon_u) ** 2).mean().item()

    # Learned encode/decode
    q._codebooks = cb
    codes_l = q.encode(sample)
    recon_l = q.decode(codes_l)
    mse_learned = ((sample - recon_l) ** 2).mean().item()

    ratio = mse_learned / mse_uniform if mse_uniform > 0 else float("nan")
    status = "PASS" if ratio <= 0.95 else "WARN (>0.95)"
    print(f"  MSE uniform={mse_uniform:.6f}  learned={mse_learned:.6f}  ratio={ratio:.3f}  [{status}]")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cb.save(out_path, metadata={
        "dim": args.dim,
        "bits_z": args.bits_z,
        "bits_r": args.bits_r,
        "bits_theta": args.bits_theta,
        "source": args.source,
        "n_calibration": vectors.shape[0],
        "seed": args.seed,
    })

    size_kb = out_path.stat().st_size / 1024
    print(f"  saved → {out_path}  ({size_kb:.1f} KB)")
    print(f"[build_codebooks] done.")


if __name__ == "__main__":
    main()
