"""
adaptive_demo.py — Adaptive bit allocation demo.

Shows how BitAllocator produces per-layer PrismKVConfig allocations from
synthetic attention entropy data, then validates with PrismKVCache.

Usage:
    python3 examples/adaptive_demo.py
"""

import torch
from prismkv.quantizer.bit_alloc import BitAllocator
from prismkv.eval.attention_entropy import attention_entropy_from_weights

try:
    from prismkv.cache import PrismKVCache
    HAS_CACHE = True
except ImportError:
    HAS_CACHE = False


def simulate_attention_entropy(n_layers: int = 12, n_heads: int = 12, seed: int = 0) -> torch.Tensor:
    """
    Simulate realistic per-head attention entropy.

    Real models typically show:
    - Early layers: moderate/high entropy (diffuse, syntactic)
    - Middle layers: mixed
    - Late layers: lower entropy (sharp, semantic content)
    """
    gen = torch.Generator().manual_seed(seed)
    base = torch.rand(n_layers, n_heads, generator=gen) * 2.0 + 0.5

    # Make late layers sharper (lower entropy)
    for l in range(n_layers):
        decay = 1.0 - 0.3 * (l / n_layers)
        base[l] *= decay

    return base.clamp(min=0.1, max=5.0)


def main():
    print("══════════════════════════════════════════════════════════")
    print("  PrismKV Adaptive Bit Allocation Demo")
    print("══════════════════════════════════════════════════════════\n")

    # Simulate GPT-2 style (12 layers, 12 heads)
    n_layers, n_heads = 12, 12
    target_bits = 4.0

    entropy = simulate_attention_entropy(n_layers, n_heads)
    print(f"Simulated entropy shape: {tuple(entropy.shape)}")
    print(f"Mean entropy: {entropy.mean():.3f}, Std: {entropy.std():.3f}")
    print(f"Min: {entropy.min():.3f}, Max: {entropy.max():.3f}\n")

    # Run bit allocator
    allocator = BitAllocator(
        entropy,
        target_avg_bits_per_dim=target_bits,
        alpha=1.0,
    ).compute()

    print(allocator.summary())

    # Show per-layer allocation table
    print("\n── Per-layer PrismKVConfig allocation ──")
    layer_configs = allocator.to_prism_configs(per_head=False)
    for i, cfg in enumerate(layer_configs):
        entropy_mean = entropy[i].mean().item()
        print(f"  Layer {i:2d}: entropy_mean={entropy_mean:.3f}  →  {cfg}")

    print(f"\nActual mean bits/dim: {allocator.mean_bits_per_dim:.4f} (target: {target_bits})")
    assert abs(allocator.mean_bits_per_dim - target_bits) < 0.11, "Mean should be close to target"

    # Validate with PrismKVCache
    if HAS_CACHE:
        print("\n── PrismKVCache with adaptive configs ──")
        cache = PrismKVCache(configs=layer_configs)
        for layer_idx in range(n_layers):
            k = torch.randn(1, n_heads, 16, 64)  # GPT-2 head_dim=64
            v = torch.randn(1, n_heads, 16, 64)
            cache.update(k, v, layer_idx=layer_idx)
        fp = cache.memory_footprint()
        print(f"  Compressed codes: {fp['codes_bytes'] / 1024:.1f} KB")
        print(f"  FP16 equivalent:  {fp['fp16_bytes'] / 1024:.1f} KB")
        print(f"  Compression:      {fp['compression']:.1f}× vs FP16")
    else:
        print("\n(Install transformers for cache integration demo)")

    print("\n══════════════════════════════════════════════════════════")
    print("  Adaptive demo complete")
    print("══════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
