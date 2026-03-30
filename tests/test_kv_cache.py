"""
test_kv_cache.py — Tests for PrismKVCache, PrismKVConfig, DimAligner.

Core tests run without transformers (using a mock DynamicCache).
Integration tests require transformers and are gated by the has_transformers marker.
"""

import pytest
import torch

from prismkv.cache.cache_config import PrismKVConfig
from prismkv.cache.dim_aligner import DimAligner


# ---------------------------------------------------------------------------
# DimAligner
# ---------------------------------------------------------------------------

class TestDimAligner:
    def test_no_padding_multiple_of_3(self):
        aligner = DimAligner(192)
        assert aligner.pad_width == 0
        assert aligner.padded_dim == 192
        t = torch.randn(10, 192)
        assert aligner.pad(t) is t

    def test_padding_64_to_66(self):
        aligner = DimAligner(64)
        assert aligner.pad_width == 2
        assert aligner.padded_dim == 66
        t = torch.randn(10, 64)
        padded = aligner.pad(t)
        assert padded.shape == (10, 66)
        assert (padded[:, 64:] == 0).all()

    def test_roundtrip(self):
        for dim in [64, 65, 66, 100, 127, 128]:
            aligner = DimAligner(dim)
            t = torch.randn(8, dim)
            restored = aligner.unpad(aligner.pad(t))
            assert restored.shape == (8, dim)
            assert torch.equal(restored, t), f"roundtrip failed for dim={dim}"


# ---------------------------------------------------------------------------
# PrismKVConfig
# ---------------------------------------------------------------------------

class TestPrismKVConfig:
    def test_defaults(self):
        cfg = PrismKVConfig()
        assert cfg.bits_z == 4
        assert cfg.bits_r == 4
        assert cfg.bits_theta == 4
        assert cfg.codebook_path is None
        assert cfg.fallback_to_uniform is True

    def test_bits_per_dim(self):
        cfg = PrismKVConfig(bits_z=3, bits_r=3, bits_theta=2)
        assert abs(cfg.bits_per_dim - 8 / 3) < 1e-9

    def test_compression_vs_fp16(self):
        cfg = PrismKVConfig(bits_z=4, bits_r=4, bits_theta=4)  # 4.0 bits/dim
        assert abs(cfg.compression_vs_fp16 - 4.0) < 1e-9


# ---------------------------------------------------------------------------
# PrismKVCache (requires transformers)
# ---------------------------------------------------------------------------

try:
    from transformers import DynamicCache
    from prismkv.cache import PrismKVCache
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestPrismKVCache:
    def _make_kv(self, batch=1, n_heads=4, seq_len=8, head_dim=66):
        """Helper: create random K/V tensors in the expected shape."""
        k = torch.randn(batch, n_heads, seq_len, head_dim)
        v = torch.randn(batch, n_heads, seq_len, head_dim)
        return k, v

    def test_isinstance_dynamic_cache(self):
        """PrismKVCache passes isinstance(DynamicCache) check."""
        cache = PrismKVCache()
        assert isinstance(cache, DynamicCache)

    def test_update_returns_tuple(self):
        """update() returns (key_states, value_states)."""
        cache = PrismKVCache()
        k, v = self._make_kv()
        result = cache.update(k, v, layer_idx=0)
        assert isinstance(result, tuple) and len(result) == 2
        rk, rv = result
        assert rk.shape == k.shape
        assert rv.shape == v.shape

    def test_update_output_finite(self):
        """Decoded K/V from update() are finite (no NaN/inf)."""
        cache = PrismKVCache()
        k, v = self._make_kv(head_dim=66)
        rk, rv = cache.update(k, v, layer_idx=0)
        assert torch.isfinite(rk).all()
        assert torch.isfinite(rv).all()

    def test_codes_stored_after_update(self):
        """After update(), _key_codes and _val_codes are populated."""
        cache = PrismKVCache()
        k, v = self._make_kv(batch=1, n_heads=4, seq_len=8, head_dim=66)
        cache.update(k, v, layer_idx=0)
        assert len(cache._key_codes) > 0
        assert cache._key_codes[0] is not None
        assert cache._val_codes[0] is not None
        assert cache._key_codes[0].dtype == torch.int16

    def test_codes_accumulate_across_calls(self):
        """Codes for the same layer accumulate across multiple update() calls."""
        cache = PrismKVCache()
        k, v = self._make_kv(batch=1, n_heads=2, seq_len=5, head_dim=66)
        cache.update(k, v, layer_idx=0)
        n_after_first = cache._key_codes[0].shape[0]

        k2, v2 = self._make_kv(batch=1, n_heads=2, seq_len=3, head_dim=66)
        cache.update(k2, v2, layer_idx=0)
        n_after_second = cache._key_codes[0].shape[0]

        assert n_after_second == n_after_first + 2 * 3  # 1*2*3 new vectors

    def test_multi_layer_update(self):
        """Each layer has its own code buffer."""
        cache = PrismKVCache()
        for layer_idx in range(3):
            k, v = self._make_kv(head_dim=66)
            cache.update(k, v, layer_idx=layer_idx)

        assert len(cache._key_codes) == 3
        assert all(kc is not None for kc in cache._key_codes)

    def test_memory_footprint_compression(self):
        """Compression ratio vs FP16 should be ≥ 2× for 4-bit scheme."""
        cache = PrismKVCache(PrismKVConfig(bits_z=4, bits_r=4, bits_theta=4))
        for layer_idx in range(4):
            k, v = self._make_kv(batch=1, n_heads=8, seq_len=32, head_dim=66)
            cache.update(k, v, layer_idx=layer_idx)

        fp = cache.memory_footprint()
        assert fp["n_layers"] == 4
        assert fp["codes_bytes"] > 0
        assert fp["fp16_bytes"] > 0
        assert fp["compression"] >= 2.0, (
            f"Expected ≥2× compression vs FP16, got {fp['compression']:.2f}×"
        )

    def test_head_dim_64_padded(self):
        """GPT-2 head_dim=64 (not divisible by 3) is handled transparently."""
        cache = PrismKVCache()
        k = torch.randn(1, 12, 10, 64)
        v = torch.randn(1, 12, 10, 64)
        rk, rv = cache.update(k, v, layer_idx=0)
        assert rk.shape == k.shape
        assert rv.shape == v.shape
        assert torch.isfinite(rk).all()

    def test_per_layer_configs(self):
        """Per-layer configs are applied independently."""
        configs = [
            PrismKVConfig(bits_z=3, bits_r=3, bits_theta=2),  # 2.67 bits/dim
            PrismKVConfig(bits_z=4, bits_r=4, bits_theta=4),  # 4.0 bits/dim
        ]
        cache = PrismKVCache(configs=configs)
        for layer_idx in range(2):
            k, v = self._make_kv(head_dim=66)
            cache.update(k, v, layer_idx=layer_idx)

        # Layer 0 should have used 2.67-bit config (fewer code bits → different m)
        q0 = cache._quantizers[0]
        q1 = cache._quantizers[1]
        assert q0.bits_z == 3 and q0.bits_theta == 2
        assert q1.bits_z == 4 and q1.bits_theta == 4

    def test_fallback_on_bad_codebook_path(self):
        """With fallback_to_uniform=True, a bad codebook path warns but doesn't crash."""
        import warnings
        cfg = PrismKVConfig(
            codebook_path="/nonexistent/cb.npz",
            fallback_to_uniform=True,
        )
        cache = PrismKVCache(cfg)
        k, v = self._make_kv(head_dim=66)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rk, rv = cache.update(k, v, layer_idx=0)
        assert any("fallback" in str(x.message).lower() or "failed" in str(x.message).lower()
                   for x in w), "Expected a warning about failed codebook load"
        assert torch.isfinite(rk).all()

    def test_output_dtype_preserved(self):
        """Decoded output dtype matches input dtype."""
        cache = PrismKVCache()
        for dtype in [torch.float32, torch.float16]:
            c = PrismKVCache()
            k = torch.randn(1, 2, 4, 66, dtype=dtype)
            v = torch.randn(1, 2, 4, 66, dtype=dtype)
            rk, rv = c.update(k, v, layer_idx=0)
            assert rk.dtype == dtype, f"Expected {dtype}, got {rk.dtype}"


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestPrismKVCacheGPT2Integration:
    def test_gpt2_generate_with_prismkv_cache(self):
        """
        GPT-2 generate() with PrismKVCache produces coherent output and no errors.

        This test verifies the full integration path: the cache is injected into
        generate(), used for all 12 layers, and produce a token sequence.
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from prismkv.cache import PrismKVCache, PrismKVConfig

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32)
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        input_ids = tokenizer("The quick brown fox", return_tensors="pt")["input_ids"]
        cache = PrismKVCache(PrismKVConfig(bits_z=4, bits_r=4, bits_theta=4))

        with torch.no_grad():
            output = model.generate(
                input_ids,
                past_key_values=cache,
                use_cache=True,
                max_new_tokens=10,
                do_sample=False,
            )

        # Output should be longer than input
        assert output.shape[1] > input_ids.shape[1]

        # Cache should have data for all 12 GPT-2 layers
        fp = cache.memory_footprint()
        assert fp["n_layers"] == 12

        # Compression should be meaningful vs FP16
        assert fp["compression"] >= 1.5, (
            f"Expected ≥1.5× compression, got {fp['compression']:.2f}×"
        )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        assert len(decoded) > 10, "Expected non-trivial generated text"
