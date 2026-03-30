"""
test_kv_collector.py — Tests for KVCollector and benchmark utilities.

These tests use synthetic tensors (no model download required) to validate
the evaluation infrastructure.  Integration tests that require transformers
are gated behind the has_transformers marker.
"""

import pytest
import torch

from prismkv.eval.benchmark import evaluate_scheme, run_benchmark, SchemeResult
from prismkv.eval.kv_collector import pad_to_multiple_of_3, unpad_from_multiple_of_3


# ---------------------------------------------------------------------------
# Dim alignment helpers
# ---------------------------------------------------------------------------

class TestDimAlignment:
    def test_pad_divisible_unchanged(self):
        """Tensors already divisible by 3 pass through unchanged."""
        t = torch.randn(10, 192)
        assert pad_to_multiple_of_3(t) is t  # same object, no copy

    def test_pad_64_to_66(self):
        """GPT-2 head_dim=64 → padded to 66."""
        t = torch.randn(100, 64)
        padded = pad_to_multiple_of_3(t)
        assert padded.shape == (100, 66)
        assert (padded[:, 64:] == 0).all()    # zero padding

    def test_pad_65_to_66(self):
        t = torch.randn(5, 65)
        padded = pad_to_multiple_of_3(t)
        assert padded.shape == (5, 66)

    def test_unpad_roundtrip(self):
        """unpad restores original values exactly."""
        t = torch.randn(20, 64)
        padded = pad_to_multiple_of_3(t)
        restored = unpad_from_multiple_of_3(padded, original_dim=64)
        assert restored.shape == (20, 64)
        assert torch.equal(restored, t)

    def test_higher_rank_tensors(self):
        """Works on (batch, seq, head_dim) shaped tensors too."""
        t = torch.randn(2, 50, 64)
        padded = pad_to_multiple_of_3(t)
        assert padded.shape == (2, 50, 66)


# ---------------------------------------------------------------------------
# SchemeResult / evaluate_scheme
# ---------------------------------------------------------------------------

class TestEvaluateScheme:
    def _make_trivial_quantizer(self, noise_scale: float = 0.01):
        """Returns encode/decode pair that adds Gaussian noise (simulates quantization)."""
        def encode(v):
            return v + torch.randn_like(v) * noise_scale

        def decode(codes):
            return codes  # codes already include noise

        return encode, decode

    def test_evaluate_scheme_fields(self):
        """evaluate_scheme returns a SchemeResult with all required fields."""
        N, dim = 100, 192
        vectors = torch.randn(N, dim)
        encode_fn, decode_fn = self._make_trivial_quantizer(noise_scale=0.0)  # perfect
        result = evaluate_scheme(vectors, encode_fn, decode_fn,
                                 bits_per_dim=4.0, name="test")
        assert isinstance(result, SchemeResult)
        assert result.name == "test"
        assert result.n_vectors == N
        assert result.bits_per_dim == 4.0
        assert result.rmse >= 0
        assert 0.0 <= result.cosine_sim_mean <= 1.0 + 1e-6
        assert result.relative_error_mean >= 0
        assert result.memory_mb_4k > 0
        assert result.throughput_vps > 0

    def test_evaluate_scheme_perfect_reconstruction(self):
        """Identity encode/decode → near-zero RMSE and cosine_sim ≈ 1."""
        N, dim = 200, 192
        vectors = torch.randn(N, dim)
        result = evaluate_scheme(
            vectors,
            encode_fn=lambda v: v.clone(),
            decode_fn=lambda c: c.clone(),
            bits_per_dim=32.0,
            name="identity",
        )
        assert result.rmse < 1e-5
        assert result.cosine_sim_mean > 0.9999

    def test_memory_calculation(self):
        """Memory formula: 4096 * dim * bits/8 * 2 (K+V) / 1MB."""
        N, dim = 10, 192
        vectors = torch.randn(N, dim)
        result = evaluate_scheme(
            vectors,
            encode_fn=lambda v: v,
            decode_fn=lambda c: c,
            bits_per_dim=4.0,
            name="mem_test",
            n_heads=1,
        )
        expected = 4096 * 192 * 4.0 / 8 * 2 / (1024 ** 2)
        assert abs(result.memory_mb_4k - expected) < 1e-6


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------

class TestRunBenchmark:
    def test_returns_two_results_no_codebook(self):
        """Without a codebook, run_benchmark returns 2-D and 3-D uniform."""
        vectors = torch.randn(200, 192)
        results = run_benchmark(vectors, bits=4)
        assert len(results) == 2
        names = {r.name for r in results}
        assert any("2D" in n for n in names)
        assert any("3D" in n for n in names)

    def test_all_results_finite(self):
        """All metrics should be finite (no NaN/inf)."""
        import math
        vectors = torch.randn(200, 192)
        for r in run_benchmark(vectors, bits=4):
            assert math.isfinite(r.rmse),             f"RMSE not finite for {r.name}"
            assert math.isfinite(r.cosine_sim_mean),  f"cosine not finite for {r.name}"
            assert math.isfinite(r.relative_error_mean), f"rel_err not finite for {r.name}"

    def test_dim_not_divisible_by_3(self):
        """
        Benchmark handles head_dim=64 (not divisible by 3) gracefully.
        Mirrors the real GPT-2 use case.
        """
        vectors = pad_to_multiple_of_3(torch.randn(200, 64))   # → (200, 66)
        results = run_benchmark(vectors, bits=4, original_dim=64)
        assert len(results) >= 2
        for r in results:
            assert r.rmse >= 0

    def test_bits_per_dim_equal_across_schemes(self):
        """2D and 3D schemes at the same bit setting should have equal bits/dim."""
        vectors = torch.randn(200, 192)
        results = run_benchmark(vectors, bits=4)
        bits = [r.bits_per_dim for r in results]
        assert all(b == bits[0] for b in bits), f"bits/dim differ: {bits}"


# ---------------------------------------------------------------------------
# Transformers integration (skipped if not installed)
# ---------------------------------------------------------------------------

try:
    import transformers  # noqa: F401
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestKVCollectorIntegration:
    def test_gpt2_collect_layer0(self):
        """
        KVCollector on GPT-2 returns KV tensors with correct dimensions.
        GPT-2 n_heads=12, head_dim=64 → padded to 66.
        """
        from prismkv.eval.kv_collector import KVCollector

        collector = KVCollector("gpt2", device="cpu", pad_dim=True)
        assert collector.n_layers == 12

        short_text = "The quick brown fox jumps over the lazy dog."
        results = collector.collect(short_text, layer_indices=[0])

        assert "layer_0" in results
        keys = results["layer_0"]["keys"]
        vals = results["layer_0"]["values"]

        # GPT-2: n_heads=12, head_dim=64 padded to 66
        assert keys.ndim == 2
        assert keys.shape[1] == 66, f"Expected padded dim=66, got {keys.shape[1]}"
        assert keys.shape[0] > 0
        assert keys.shape == vals.shape
        assert torch.isfinite(keys).all()
        assert torch.isfinite(vals).all()

        # Padded dims should be zero
        assert (keys[:, 64:] == 0).all()

    # ------------------------------------------------------------------
    # Class-level KV collection shared across all multi-layer tests
    # (GPT-2 forward pass is fast; collect once per test class)
    # ------------------------------------------------------------------

    _all_layer_keys: dict = {}   # populated by setup_class

    @classmethod
    def setup_class(cls):
        """Collect all 12 GPT-2 layers once; store in cls._all_layer_keys."""
        from prismkv.eval.kv_collector import KVCollector

        text = "The quick brown fox jumps over the lazy dog. " * 100  # ~1200 tokens
        collector = KVCollector("gpt2", device="cpu", pad_dim=True)
        layer_indices = list(range(12))
        collected = collector.collect(text, layer_indices=layer_indices)
        cls._all_layer_keys = {
            l: collected[f"layer_{l}"]["keys"]
            for l in layer_indices
        }

    def test_gpt2_benchmark_all_layers(self):
        """
        Full benchmark on real GPT-2 KV distributions across all 12 layers.
        Learned scheme should be ≤95% RMSE of uniform on ≥10/12 layers.
        """
        import tempfile
        from pathlib import Path
        from prismkv.eval.benchmark import run_benchmark, print_table
        from prismkv.quantizer.learned_codebook import LearnedSliceCodebook
        from prismkv import StackedPlaneQuantizer

        all_keys = self._all_layer_keys
        assert len(all_keys) == 12, "Expected keys for all 12 GPT-2 layers"

        layers_learned_wins = 0
        layer_rmse_uniform = []
        layer_rmse_learned = []

        for layer_idx, keys in all_keys.items():
            dim = keys.shape[1]  # 66 (padded)
            q = StackedPlaneQuantizer(dim=dim, bits_z=4, bits_r=4, bits_theta=4, seed=42)
            q.calibrate(keys)
            rotated = keys @ q.R.T
            K = 2 ** (q.bits_r + q.bits_theta)
            cb = LearnedSliceCodebook.train(
                rotated_vectors=rotated,
                z_idx=q.z_idx, x_idx=q.x_idx, y_idx=q.y_idx,
                z_min=q.z_min, z_max=q.z_max, r_max=q.r_max,
                bins_z=q.bins_z, K=K, max_iter=30, seed=0,
            )

            with tempfile.TemporaryDirectory() as tmp:
                cb_path = Path(tmp) / f"gpt2_cb_layer{layer_idx}.npz"
                cb.save(cb_path)
                bench_results = run_benchmark(
                    keys, bits=4, codebook_path=str(cb_path), original_dim=64
                )

            r_uniform = next(r for r in bench_results if "uniform" in r.name and "3D" in r.name)
            r_learned = next(r for r in bench_results if "learned" in r.name)
            layer_rmse_uniform.append(r_uniform.rmse)
            layer_rmse_learned.append(r_learned.rmse)

            ratio = r_learned.rmse / r_uniform.rmse
            if ratio <= 0.95:
                layers_learned_wins += 1

        # Per-layer heterogeneity: RMSE should vary across layers
        rmse_tensor = torch.tensor(layer_rmse_uniform)
        rmse_std = rmse_tensor.std().item()
        assert rmse_std > 0, "Per-layer RMSE should vary across GPT-2 layers"

        # Learned codebook wins on ≥10/12 layers
        assert layers_learned_wins >= 10, (
            f"Learned codebook should outperform uniform on ≥10/12 layers, "
            f"only won on {layers_learned_wins}/12"
        )

    def test_gpt2_per_layer_entropy_range(self):
        """
        Per-head attention entropy should span a meaningful range across layers,
        demonstrating that GPT-2 has heterogeneous attention patterns.
        """
        from prismkv.eval.attention_entropy import collect_attention_entropy
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager")
        model.eval()

        text = "The quick brown fox jumps over the lazy dog. " * 20
        entropy = collect_attention_entropy(model, text, tokenizer, device="cpu")

        assert entropy.shape == (12, 12), \
            f"Expected (12 layers, 12 heads) entropy, got {entropy.shape}"

        # Entropy values should be positive
        assert (entropy > 0).all(), "All entropy values should be positive"

        # There should be meaningful range in entropy (heterogeneous heads)
        entropy_range = entropy.max().item() - entropy.min().item()
        assert entropy_range > 0.5, \
            f"Entropy range {entropy_range:.3f} too small — expected heterogeneous heads"
