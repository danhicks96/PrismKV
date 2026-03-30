"""
test_bit_alloc.py — Tests for BitAllocator and attention entropy utilities.
"""

import math
import pytest
import torch

from prismkv.quantizer.bit_alloc import BitAllocator, enumerate_valid_configs, nearest_config
from prismkv.eval.attention_entropy import attention_entropy_from_weights


# ---------------------------------------------------------------------------
# enumerate_valid_configs
# ---------------------------------------------------------------------------

class TestEnumerateConfigs:
    def test_sorted_by_total_bits(self):
        configs = enumerate_valid_configs(min_bits=2, max_bits=4)
        totals = [sum(c) for c in configs]
        assert totals == sorted(totals)

    def test_all_in_range(self):
        configs = enumerate_valid_configs(min_bits=3, max_bits=5)
        for bz, br, bt in configs:
            assert 3 <= bz <= 5
            assert 3 <= br <= 5
            assert 3 <= bt <= 5

    def test_count(self):
        configs = enumerate_valid_configs(min_bits=2, max_bits=3)
        assert len(configs) == 2 ** 3   # 2×2×2 = 8


class TestNearestConfig:
    def test_exact_match(self):
        bz, br, bt = nearest_config(4.0)
        assert (bz + br + bt) / 3 == 4.0

    def test_nearest_rounds_to_closest(self):
        cfg = nearest_config(2.67)  # 3+3+2 = 8/3 = 2.67
        assert abs(sum(cfg) / 3 - 2.67) < 0.05


# ---------------------------------------------------------------------------
# attention_entropy_from_weights
# ---------------------------------------------------------------------------

class TestAttentionEntropyFromWeights:
    def test_uniform_is_max_entropy(self):
        seq = 8
        uniform = torch.ones(1, 1, seq, seq) / seq
        ent = attention_entropy_from_weights(uniform)
        assert abs(ent.item() - math.log(seq)) < 0.01

    def test_sharp_is_low_entropy(self):
        seq = 8
        sharp = torch.zeros(1, 1, seq, seq)
        sharp[:, :, :, 0] = 1.0   # all attention on first token
        ent = attention_entropy_from_weights(sharp)
        assert ent.item() < 0.01

    def test_output_shape(self):
        """entropy shape: (batch, n_heads)."""
        attn = torch.softmax(torch.randn(2, 4, 10, 10), dim=-1)
        ent = attention_entropy_from_weights(attn)
        assert ent.shape == (2, 4)


# ---------------------------------------------------------------------------
# BitAllocator
# ---------------------------------------------------------------------------

class TestBitAllocator:
    def make_entropy(self, n_layers=4, n_heads=8, pattern="mixed"):
        """Create synthetic entropy tensors."""
        if pattern == "uniform":
            return torch.ones(n_layers, n_heads) * 2.0
        elif pattern == "mixed":
            gen = torch.Generator().manual_seed(42)
            return torch.rand(n_layers, n_heads, generator=gen) * 3.0 + 0.5
        elif pattern == "sharp":
            return torch.ones(n_layers, n_heads) * 0.1
        return torch.ones(n_layers, n_heads)

    def test_compute_returns_self(self):
        entropy = self.make_entropy()
        allocator = BitAllocator(entropy, target_avg_bits_per_dim=4.0)
        result = allocator.compute()
        assert result is allocator

    def test_mean_bits_within_tolerance(self):
        """Mean bits/dim after rounding should be within 0.1 of target."""
        entropy = self.make_entropy(pattern="mixed")
        for target in [3.0, 4.0, 5.0]:
            allocator = BitAllocator(entropy, target_avg_bits_per_dim=target).compute()
            assert abs(allocator.mean_bits_per_dim - target) < 0.1, (
                f"target={target}, actual={allocator.mean_bits_per_dim:.3f}"
            )

    def test_allocations_shape(self):
        entropy = self.make_entropy(n_layers=3, n_heads=6)
        allocator = BitAllocator(entropy, target_avg_bits_per_dim=4.0).compute()
        assert allocator.allocations.shape == (3, 6)

    def test_high_sensitivity_gets_more_bits(self):
        """
        Low-entropy (sharp) heads should receive more bits than high-entropy heads.
        """
        n_layers, n_heads = 1, 2
        # Head 0: very sharp (entropy=0.1) → high sensitivity → more bits
        # Head 1: diffuse (entropy=4.0)  → low sensitivity → fewer bits
        entropy = torch.tensor([[0.1, 4.0]])
        allocator = BitAllocator(entropy, target_avg_bits_per_dim=4.0, alpha=1.0).compute()
        alloc = allocator.allocations
        assert alloc[0, 0] > alloc[0, 1], (
            f"Sharp head should get more bits: {alloc[0,0]:.2f} vs {alloc[0,1]:.2f}"
        )

    def test_to_prism_configs_length(self):
        entropy = self.make_entropy(n_layers=4, n_heads=8)
        allocator = BitAllocator(entropy, target_avg_bits_per_dim=4.0).compute()

        # Per-layer: length = n_layers
        per_layer = allocator.to_prism_configs(per_head=False)
        assert len(per_layer) == 4

        # Per-head: length = n_layers * n_heads
        per_head = allocator.to_prism_configs(per_head=True)
        assert len(per_head) == 4 * 8

    def test_to_prism_configs_valid_bits(self):
        from prismkv.cache.cache_config import PrismKVConfig
        entropy = self.make_entropy(pattern="mixed")
        allocator = BitAllocator(entropy, target_avg_bits_per_dim=4.0,
                                 min_bits=2, max_bits=6).compute()
        configs = allocator.to_prism_configs(per_head=False)
        for cfg in configs:
            assert isinstance(cfg, PrismKVConfig)
            assert 2 <= cfg.bits_z <= 6
            assert 2 <= cfg.bits_r <= 6
            assert 2 <= cfg.bits_theta <= 6

    def test_summary_string(self):
        entropy = self.make_entropy()
        allocator = BitAllocator(entropy, target_avg_bits_per_dim=4.0).compute()
        summary = allocator.summary()
        assert "BitAllocator" in summary
        assert "target=" in summary

    def test_not_computed_raises(self):
        entropy = self.make_entropy()
        allocator = BitAllocator(entropy)
        with pytest.raises(RuntimeError, match="compute"):
            _ = allocator.mean_bits_per_dim

    def test_uniform_entropy_uniform_allocation(self):
        """Uniform entropy across all heads → all heads get the same bits."""
        entropy = torch.ones(4, 8) * 2.0
        allocator = BitAllocator(entropy, target_avg_bits_per_dim=4.0, alpha=1.0).compute()
        alloc = allocator.allocations
        # All values should be close to target
        assert alloc.std().item() < 0.01


# ---------------------------------------------------------------------------
# Integration: BitAllocator → PrismKVCache
# ---------------------------------------------------------------------------

try:
    from transformers import DynamicCache
    from prismkv.cache import PrismKVCache
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestBitAllocatorCacheIntegration:
    def test_per_layer_cache_accepts_alloc_configs(self):
        """
        PrismKVCache with per-layer configs from BitAllocator runs without error.
        """
        entropy = torch.rand(12, 12) * 2.0 + 0.5  # GPT-2: 12 layers, 12 heads
        allocator = BitAllocator(entropy, target_avg_bits_per_dim=4.0).compute()
        layer_configs = allocator.to_prism_configs(per_head=False)  # 12 configs

        cache = PrismKVCache(configs=layer_configs)
        assert isinstance(cache, DynamicCache)

        # Run a few update() calls (simulating 12-layer GPT-2)
        for layer_idx in range(12):
            k = torch.randn(1, 12, 8, 64)
            v = torch.randn(1, 12, 8, 64)
            cache.update(k, v, layer_idx=layer_idx)

        fp = cache.memory_footprint()
        assert fp["n_layers"] == 12
        assert fp["compression"] >= 1.5
