"""
e2e_benchmark.py — End-to-end benchmark: memory footprint and reconstruction quality.

Produces two outputs:

1. **Memory table** — KV cache memory at context lengths {1K, 4K, 16K} for:
   - FP16 baseline (DynamicCache)
   - PrismKV at 3, 4, and 5 bits/dim

2. **Quality table** — RMSE, cosine similarity, and relative error at each bit budget,
   evaluated on either real KV vectors (from KVCollector) or synthetic data.

The ``measure_pseudo_perplexity`` function additionally computes cross-entropy-based
pseudo-perplexity with GPT-2; it requires the ``transformers`` optional dep group
(``pip install prismkv[eval]``) and downloads GPT-2 weights on first call (~500 MB).

Quick start (no model download)::

    from prismkv.eval.e2e_benchmark import run_e2e_benchmark, print_e2e_table
    results = run_e2e_benchmark(head_dim=64, n_heads=12, n_layers=12)
    print_e2e_table(results)

Author: Dan Hicks (github.com/danhicks96)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import torch

from prismkv.quantizer.stacked_plane import StackedPlaneQuantizer
from prismkv.quantizer.baseline_2d import PolarQuantizer2D
from prismkv.cache.dim_aligner import DimAligner


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MemoryProfile:
    """Memory footprint of one configuration at one context length."""
    scheme: str              # "FP16", "3bit", "4bit", "5bit"
    bits_per_dim: float      # effective bits per original dimension
    context_len: int         # number of tokens
    n_layers: int
    n_heads: int
    head_dim: int
    memory_mb: float         # total KV cache memory in MB


@dataclass
class QualityResult:
    """Reconstruction quality for one bit budget."""
    scheme: str
    bits_per_dim: float
    rmse: float
    cosine_sim_mean: float
    relative_error_mean: float
    throughput_vps: float    # encode+decode vectors per second
    n_vectors: int


@dataclass
class E2EReport:
    """Combined memory + quality report."""
    memory_profiles: List[MemoryProfile]
    quality_results: List[QualityResult]
    head_dim: int
    n_heads: int
    n_layers: int
    context_lengths: List[int]
    bits_configs: List[int]
    adaptive_result: Optional["QualityResult"] = None  # set when adaptive_allocation=True


# ---------------------------------------------------------------------------
# Memory table
# ---------------------------------------------------------------------------


def compute_memory_table(
    n_layers: int,
    n_heads: int,
    head_dim: int,
    context_lengths: Optional[List[int]] = None,
    bits_configs: Optional[List[int]] = None,
) -> List[MemoryProfile]:
    """
    Compute theoretical KV cache memory footprints.

    Memory formula
    --------------
    FP16:  n_layers × 2 (K+V) × n_heads × context_len × head_dim × 2 bytes
    Nbit:  n_layers × 2 (K+V) × n_heads × context_len × head_dim × (bits/8) bytes

    PrismKV uses int16 codes of shape (batch, n_heads, seq, head_dim/3) where
    each int16 stores 12 bits.  Effective bits per original dimension = 12/3 = 4.
    For other bit budgets the formula scales accordingly.

    Parameters
    ----------
    n_layers        : number of transformer layers
    n_heads         : number of (KV) attention heads
    head_dim        : dimension per head in the original (unpadded) model
    context_lengths : token counts to evaluate (default: [1024, 4096, 16384])
    bits_configs    : bit budgets to evaluate (default: [3, 4, 5])

    Returns
    -------
    List of MemoryProfile, one per (scheme × context_length) combination.
    """
    if context_lengths is None:
        context_lengths = [1024, 4096, 16384]
    if bits_configs is None:
        bits_configs = [3, 4, 5]

    profiles: List[MemoryProfile] = []
    for ctx in context_lengths:
        # FP16 baseline
        fp16_bytes = n_layers * 2 * n_heads * ctx * head_dim * 2
        profiles.append(MemoryProfile(
            scheme="FP16",
            bits_per_dim=16.0,
            context_len=ctx,
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
            memory_mb=fp16_bytes / (1024 ** 2),
        ))
        # Compressed configs
        for bits in bits_configs:
            compressed_bytes = n_layers * 2 * n_heads * ctx * head_dim * (bits / 8)
            profiles.append(MemoryProfile(
                scheme=f"{bits}bit",
                bits_per_dim=float(bits),
                context_len=ctx,
                n_layers=n_layers,
                n_heads=n_heads,
                head_dim=head_dim,
                memory_mb=compressed_bytes / (1024 ** 2),
            ))
    return profiles


def compression_ratio(bits: float) -> float:
    """FP16 → compressed compression ratio (larger = more compression)."""
    return 16.0 / bits


# ---------------------------------------------------------------------------
# Quality evaluation
# ---------------------------------------------------------------------------


def _make_synthetic_vectors(n: int, head_dim: int, seed: int = 42) -> torch.Tensor:
    """Generate anisotropic Gaussian KV-like vectors for testing."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    base = torch.randn(n, head_dim, generator=rng)
    # Mild anisotropy — first third of dims have smaller variance (like real K distributions)
    scale = torch.ones(head_dim)
    scale[: head_dim // 3] *= 0.5
    return base * scale


def evaluate_quality(
    vectors: torch.Tensor,
    bits_configs: Optional[List[int]] = None,
    seed: int = 42,
) -> List[QualityResult]:
    """
    Evaluate reconstruction quality at multiple bit budgets.

    Compares 2D polar (baseline) vs 3D PrismKV at each bit budget.

    Parameters
    ----------
    vectors      : (N, head_dim) float32 KV vectors — padded to multiple of 3
    bits_configs : bit budgets to compare (default: [3, 4, 5])
    seed         : random seed for quantizer rotation matrix

    Returns
    -------
    List of QualityResult, one per (scheme × bits) combination.
    """
    if bits_configs is None:
        bits_configs = [3, 4, 5]

    N, dim = vectors.shape
    results: List[QualityResult] = []

    for bits in bits_configs:
        # 3D PrismKV
        d3 = dim + (3 - dim % 3) % 3  # pad to multiple of 3
        v3 = torch.nn.functional.pad(vectors, (0, d3 - dim)) if d3 != dim else vectors
        q3 = StackedPlaneQuantizer(
            dim=d3, bits_z=bits, bits_r=bits, bits_theta=bits, seed=seed
        )
        q3.calibrate(v3)

        t0 = time.perf_counter()
        codes = q3.encode(v3)
        recon = q3.decode(codes)
        elapsed = time.perf_counter() - t0

        diff = vectors - recon[:, :dim]
        rmse = diff.pow(2).mean(dim=1).sqrt().mean().item()
        v_norm = torch.nn.functional.normalize(vectors, dim=1)
        r_norm = torch.nn.functional.normalize(recon[:, :dim], dim=1)
        cos_mean = (v_norm * r_norm).sum(dim=1).mean().item()
        rel_err = (
            diff.norm(dim=1) / vectors.norm(dim=1).clamp(min=1e-8)
        ).mean().item()
        tps = N / elapsed if elapsed > 0 else float("inf")

        results.append(QualityResult(
            scheme=f"PrismKV-{bits}bit",
            bits_per_dim=float(bits),
            rmse=rmse,
            cosine_sim_mean=cos_mean,
            relative_error_mean=rel_err,
            throughput_vps=tps,
            n_vectors=N,
        ))

        # 2D polar baseline (same bits budget, nearest even dim)
        d2 = dim + dim % 2
        v2 = torch.nn.functional.pad(vectors, (0, d2 - dim)) if d2 != dim else vectors
        q2 = PolarQuantizer2D(dim=d2, bits_r=bits, bits_theta=bits, seed=seed)

        t0 = time.perf_counter()
        codes2 = q2.encode(v2)
        recon2 = q2.decode(codes2)
        elapsed2 = time.perf_counter() - t0

        diff2 = vectors - recon2[:, :dim]
        rmse2 = diff2.pow(2).mean(dim=1).sqrt().mean().item()
        v_norm2 = torch.nn.functional.normalize(vectors, dim=1)
        r_norm2 = torch.nn.functional.normalize(recon2[:, :dim], dim=1)
        cos2 = (v_norm2 * r_norm2).sum(dim=1).mean().item()
        rel2 = (
            diff2.norm(dim=1) / vectors.norm(dim=1).clamp(min=1e-8)
        ).mean().item()
        tps2 = N / elapsed2 if elapsed2 > 0 else float("inf")

        results.append(QualityResult(
            scheme=f"2DPolar-{bits}bit",
            bits_per_dim=float(bits),
            rmse=rmse2,
            cosine_sim_mean=cos2,
            relative_error_mean=rel2,
            throughput_vps=tps2,
            n_vectors=N,
        ))

    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_e2e_benchmark(
    kv_vectors: Optional[torch.Tensor] = None,
    head_dim: int = 64,
    n_heads: int = 12,
    n_layers: int = 12,
    n_synthetic: int = 5000,
    context_lengths: Optional[List[int]] = None,
    bits_configs: Optional[List[int]] = None,
    seed: int = 42,
    adaptive_allocation: bool = False,
    entropy: Optional[torch.Tensor] = None,
) -> E2EReport:
    """
    Run the full end-to-end benchmark.

    Parameters
    ----------
    kv_vectors         : optional (N, head_dim) float32 real KV vectors for quality eval.
                         If None, synthetic anisotropic Gaussians are used.
    head_dim           : attention head dimension (default 64, GPT-2 style)
    n_heads            : number of attention heads (default 12)
    n_layers           : number of transformer layers (default 12)
    n_synthetic        : number of synthetic vectors to generate if kv_vectors is None
    context_lengths    : token counts for memory table (default [1024, 4096, 16384])
    bits_configs       : bit budgets (default [3, 4, 5])
    seed               : random seed
    adaptive_allocation: if True, run an additional adaptive quality pass using
                         entropy water-filling (BitAllocator).  Requires entropy
                         to be provided or will use synthetic uniform entropy.
    entropy            : optional (n_layers, n_heads) entropy tensor for adaptive
                         allocation; ignored when adaptive_allocation=False.

    Returns
    -------
    E2EReport with memory_profiles and quality_results.
    When adaptive_allocation=True, E2EReport.adaptive_result holds the
    additional QualityResult for the "3D Adaptive (entropy water-fill)" scheme.
    """
    if context_lengths is None:
        context_lengths = [1024, 4096, 16384]
    if bits_configs is None:
        bits_configs = [3, 4, 5]

    # Memory table (no model required)
    memory = compute_memory_table(
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        context_lengths=context_lengths,
        bits_configs=bits_configs,
    )

    # Quality evaluation
    if kv_vectors is None:
        vectors = _make_synthetic_vectors(n_synthetic, head_dim, seed=seed)
    else:
        vectors = kv_vectors.float()

    quality = evaluate_quality(vectors, bits_configs=bits_configs, seed=seed)

    # Adaptive allocation quality pass
    adaptive_result: Optional[QualityResult] = None
    if adaptive_allocation:
        adaptive_result = _evaluate_adaptive(
            vectors=vectors,
            n_layers=n_layers,
            n_heads=n_heads,
            entropy=entropy,
            target_bits=4,
            seed=seed,
        )

    return E2EReport(
        memory_profiles=memory,
        quality_results=quality,
        head_dim=head_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        context_lengths=context_lengths,
        bits_configs=bits_configs,
        adaptive_result=adaptive_result,
    )


def _evaluate_adaptive(
    vectors: torch.Tensor,
    n_layers: int,
    n_heads: int,
    entropy: Optional[torch.Tensor],
    target_bits: int = 4,
    seed: int = 42,
) -> QualityResult:
    """
    Evaluate reconstruction quality using entropy-driven adaptive bit allocation.

    Uses BitAllocator to compute per-layer configs, then evaluates the
    layer-averaged config on the provided vectors.

    Parameters
    ----------
    vectors    : (N, head_dim) float32 KV vectors
    n_layers   : model layer count
    n_heads    : model head count
    entropy    : (n_layers, n_heads) entropy tensor; if None, synthetic entropy used
    target_bits: mean bits/dim target
    seed       : random seed

    Returns
    -------
    QualityResult with scheme="3D Adaptive (entropy water-fill)"
    """
    from prismkv.quantizer.bit_alloc import BitAllocator

    if entropy is None:
        # Synthetic heterogeneous entropy for testing without a model
        torch.manual_seed(seed)
        entropy = torch.rand(n_layers, n_heads) * 2.5 + 0.5

    alloc = BitAllocator(entropy, target_avg_bits_per_dim=float(target_bits))
    alloc.compute()

    # Use the layer-averaged config from layer 0 as representative
    layer_configs = alloc.to_prism_configs(per_head=False)
    avg_cfg = layer_configs[0]

    N, dim = vectors.shape
    d3 = dim + (3 - dim % 3) % 3
    v3 = torch.nn.functional.pad(vectors, (0, d3 - dim)) if d3 != dim else vectors

    q = StackedPlaneQuantizer(
        dim=d3,
        bits_z=avg_cfg.bits_z,
        bits_r=avg_cfg.bits_r,
        bits_theta=avg_cfg.bits_theta,
        seed=seed,
    )
    q.calibrate(v3)

    import time
    t0 = time.perf_counter()
    codes = q.encode(v3)
    recon = q.decode(codes)
    elapsed = time.perf_counter() - t0

    diff = vectors - recon[:, :dim]
    rmse = diff.pow(2).mean(dim=1).sqrt().mean().item()
    v_norm = torch.nn.functional.normalize(vectors, dim=1)
    r_norm = torch.nn.functional.normalize(recon[:, :dim], dim=1)
    cos_mean = (v_norm * r_norm).sum(dim=1).mean().item()
    rel_err = (
        diff.norm(dim=1) / vectors.norm(dim=1).clamp(min=1e-8)
    ).mean().item()
    tps = N / elapsed if elapsed > 0 else float("inf")

    return QualityResult(
        scheme="3D Adaptive (entropy water-fill)",
        bits_per_dim=alloc.mean_bits_per_dim,
        rmse=rmse,
        cosine_sim_mean=cos_mean,
        relative_error_mean=rel_err,
        throughput_vps=tps,
        n_vectors=N,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_memory_table(report: E2EReport) -> None:
    """Print memory footprint comparison table."""
    print(f"\n{'═' * 72}")
    print("  KV Cache Memory Footprint")
    print(f"  Model: {report.n_layers}L × {report.n_heads}H × d={report.head_dim}")
    print(f"{'═' * 72}")

    schemes = ["FP16"] + [f"{b}bit" for b in report.bits_configs]
    header = f"{'Context':>10}  " + "  ".join(f"{s:>12}" for s in schemes)
    print(header)
    print("-" * len(header))

    for ctx in report.context_lengths:
        row_profiles = {
            p.scheme: p
            for p in report.memory_profiles
            if p.context_len == ctx
        }
        fp16_mb = row_profiles.get("FP16", None)
        row = f"{ctx:>10,}  "
        for scheme in schemes:
            p = row_profiles.get(scheme)
            if p is None:
                row += f"{'N/A':>12}  "
            elif scheme == "FP16":
                row += f"{p.memory_mb:>10.1f}MB  "
            else:
                ratio = fp16_mb.memory_mb / p.memory_mb if fp16_mb else 0.0
                row += f"{p.memory_mb:>7.1f}MB({ratio:.1f}×)  "
        print(row)
    print()


def print_quality_table(report: E2EReport) -> None:
    """Print reconstruction quality comparison table."""
    print(f"\n{'═' * 78}")
    print("  Reconstruction Quality (RMSE / CosSim / RelErr)")
    print(f"  Vectors: {report.quality_results[0].n_vectors if report.quality_results else 0:,}  "
          f"head_dim={report.head_dim}")
    print(f"{'═' * 78}")

    header = (
        f"{'Scheme':<24} {'bits/dim':>8} {'RMSE':>10} "
        f"{'CosSim':>8} {'RelErr':>8} {'Vec/s':>10}"
    )
    print(header)
    print("-" * len(header))

    for r in report.quality_results:
        print(
            f"{r.scheme:<24} {r.bits_per_dim:>8.1f} {r.rmse:>10.6f} "
            f"{r.cosine_sim_mean:>8.4f} {r.relative_error_mean:>8.4f} "
            f"{r.throughput_vps:>10.0f}"
        )
    print()

    # PrismKV vs 2D Polar improvement summary
    print("  PrismKV vs 2D Polar baseline:")
    for bits in report.bits_configs:
        prism = next((r for r in report.quality_results if r.scheme == f"PrismKV-{bits}bit"), None)
        polar = next((r for r in report.quality_results if r.scheme == f"2DPolar-{bits}bit"), None)
        if prism and polar and polar.rmse > 0:
            pct = (prism.rmse - polar.rmse) / polar.rmse * 100
            sign = "+" if pct > 0 else ""
            print(f"  {bits}bit: RMSE {sign}{pct:.1f}%  CosSim Δ={prism.cosine_sim_mean - polar.cosine_sim_mean:+.4f}")
    print()


def print_e2e_table(report: E2EReport) -> None:
    """Print both memory and quality tables."""
    print_memory_table(report)
    print_quality_table(report)


# ---------------------------------------------------------------------------
# Optional: pseudo-perplexity (requires transformers)
# ---------------------------------------------------------------------------


def measure_pseudo_perplexity(
    model_name: str = "gpt2",
    corpus: Optional[str] = None,
    n_tokens: int = 512,
    bits_configs: Optional[List[int]] = None,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Estimate pseudo-perplexity: cross-entropy of the model with compressed vs
    uncompressed KV cache.

    This function requires ``transformers`` (``pip install prismkv[eval]``) and
    will download model weights on first call (~500 MB for GPT-2).

    The metric is computed by running teacher-forced next-token prediction on a
    fixed text corpus and measuring average cross-entropy loss.  Lower is better.

    Parameters
    ----------
    model_name   : HuggingFace model name (default "gpt2")
    corpus       : text to evaluate; defaults to the opening of Moby Dick (~500 tokens)
    n_tokens     : number of tokens to evaluate over (default 512)
    bits_configs : bit budgets to test (default [3, 4, 5])
    seed         : random seed for PrismKVConfig

    Returns
    -------
    Dict mapping scheme name to average cross-entropy loss (nats per token).
    E.g.: {"fp16": 3.21, "prismkv_3bit": 3.28, "prismkv_4bit": 3.22, "prismkv_5bit": 3.21}
    """
    try:
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError("pip install prismkv[eval] to use measure_pseudo_perplexity") from e

    from prismkv.cache import PrismKVCache, PrismKVConfig

    if bits_configs is None:
        bits_configs = [3, 4, 5]

    if corpus is None:
        corpus = (
            "Call me Ishmael. Some years ago—never mind how long precisely—having little "
            "money in my purse, and nothing particular to interest me on shore, I thought "
            "I would sail about a little and see the watery part of the world. It is a way "
            "I have of driving off the spleen and regulating the circulation. Whenever I "
            "find myself growing grim about the mouth; whenever it is a damp, drizzly "
            "November in my soul; whenever I find myself involuntarily pausing before "
            "coffin warehouses, and bringing up the rear of every funeral I meet; and "
            "especially whenever my hypos get such an upper hand of me, that it requires "
            "a strong moral principle to prevent me from deliberately stepping into the "
            "street, and methodically knocking people's hats off—then, I account it high "
            "time to get to sea as soon as I can. This is my substitute for pistol and "
            "ball. With a philosophical flourish Cato throws himself upon his sword; I "
            "quietly take to the ship."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    input_ids = tokenizer.encode(corpus, return_tensors="pt")
    input_ids = input_ids[:, :n_tokens]
    n = input_ids.shape[1]

    results: Dict[str, float] = {}

    import torch.nn.functional as F

    def _eval_loss(past_kv_class, config=None) -> float:
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for i in range(1, n):
                prefix = input_ids[:, :i]
                target = input_ids[:, i]
                if config is not None:
                    past = past_kv_class(config=config)
                else:
                    past = past_kv_class()
                out = model(prefix, past_key_values=past, use_cache=True)
                logits = out.logits[:, -1, :]  # (1, vocab)
                loss = F.cross_entropy(logits, target)
                total_loss += loss.item()
                count += 1
        return total_loss / count if count > 0 else float("nan")

    # FP16 baseline (DynamicCache)
    from transformers import DynamicCache
    results["fp16"] = _eval_loss(DynamicCache, config=None)

    # PrismKV at each bit budget
    for bits in bits_configs:
        cfg = PrismKVConfig(bits_z=bits, bits_r=bits, bits_theta=bits)
        results[f"prismkv_{bits}bit"] = _eval_loss(PrismKVCache, config=cfg)

    return results
