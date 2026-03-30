"""
tests/test_e2e_benchmark.py — Tests for the end-to-end benchmark module.

All tests use synthetic data; no model downloads required.
"""

import math
import pytest
import torch

from prismkv.eval.e2e_benchmark import (
    compute_memory_table,
    compression_ratio,
    evaluate_quality,
    run_e2e_benchmark,
    print_e2e_table,
    E2EReport,
    MemoryProfile,
    QualityResult,
    _make_synthetic_vectors,
)


# ---------------------------------------------------------------------------
# Memory table tests
# ---------------------------------------------------------------------------


class TestComputeMemoryTable:
    def test_fp16_formula(self):
        """FP16 memory = n_layers × 2 × n_heads × ctx × head_dim × 2 bytes."""
        profiles = compute_memory_table(
            n_layers=12, n_heads=12, head_dim=64,
            context_lengths=[4096], bits_configs=[]
        )
        fp16 = [p for p in profiles if p.scheme == "FP16"]
        assert len(fp16) == 1
        expected_bytes = 12 * 2 * 12 * 4096 * 64 * 2
        expected_mb = expected_bytes / (1024 ** 2)
        assert abs(fp16[0].memory_mb - expected_mb) < 1e-6

    def test_compressed_lt_fp16(self):
        """Compressed memory < FP16 for bits < 16."""
        profiles = compute_memory_table(
            n_layers=12, n_heads=12, head_dim=64,
            context_lengths=[4096], bits_configs=[3, 4, 5]
        )
        fp16_mb = next(p.memory_mb for p in profiles if p.scheme == "FP16")
        for bits in [3, 4, 5]:
            compressed_mb = next(p.memory_mb for p in profiles if p.scheme == f"{bits}bit")
            assert compressed_mb < fp16_mb, f"{bits}bit should be smaller than FP16"

    def test_more_bits_more_memory(self):
        """Higher bit budget → more memory."""
        profiles = compute_memory_table(
            n_layers=12, n_heads=12, head_dim=64,
            context_lengths=[4096], bits_configs=[3, 4, 5]
        )
        mem = {p.scheme: p.memory_mb for p in profiles if p.context_len == 4096}
        assert mem["3bit"] < mem["4bit"] < mem["5bit"]

    def test_longer_context_more_memory(self):
        """Longer context → more memory."""
        profiles = compute_memory_table(
            n_layers=12, n_heads=12, head_dim=64,
            context_lengths=[1024, 4096], bits_configs=[]
        )
        fp16_1k = next(p.memory_mb for p in profiles if p.scheme == "FP16" and p.context_len == 1024)
        fp16_4k = next(p.memory_mb for p in profiles if p.scheme == "FP16" and p.context_len == 4096)
        assert fp16_4k == pytest.approx(fp16_1k * 4, rel=1e-6)

    def test_returns_correct_types(self):
        profiles = compute_memory_table(
            n_layers=4, n_heads=8, head_dim=64,
            context_lengths=[1024, 4096], bits_configs=[4]
        )
        for p in profiles:
            assert isinstance(p, MemoryProfile)
            assert p.memory_mb > 0
            assert p.n_layers == 4

    def test_default_context_and_bits(self):
        profiles = compute_memory_table(n_layers=12, n_heads=12, head_dim=64)
        contexts = {p.context_len for p in profiles}
        assert {1024, 4096, 16384} == contexts


class TestCompressionRatio:
    def test_fp16_ratio(self):
        assert compression_ratio(16.0) == pytest.approx(1.0)

    def test_4bit_ratio(self):
        assert compression_ratio(4.0) == pytest.approx(4.0)

    def test_3bit_ratio(self):
        assert compression_ratio(3.0) == pytest.approx(16 / 3)


# ---------------------------------------------------------------------------
# Quality evaluation tests
# ---------------------------------------------------------------------------


class TestEvaluateQuality:
    def setup_method(self):
        self.vectors = _make_synthetic_vectors(500, 66, seed=1)  # 66 = multiple of 3

    def test_returns_quality_results(self):
        results = evaluate_quality(self.vectors, bits_configs=[4])
        assert len(results) == 2  # PrismKV-4bit + 2DPolar-4bit
        for r in results:
            assert isinstance(r, QualityResult)

    def test_rmse_nonnegative(self):
        results = evaluate_quality(self.vectors, bits_configs=[3, 4])
        for r in results:
            assert r.rmse >= 0.0, f"Negative RMSE for {r.scheme}"

    def test_cosine_sim_bounded(self):
        results = evaluate_quality(self.vectors, bits_configs=[4])
        for r in results:
            assert -1.0 <= r.cosine_sim_mean <= 1.0 + 1e-5, f"CosSim out of range for {r.scheme}"

    def test_more_bits_lower_rmse(self):
        """PrismKV should have lower RMSE at higher bit budget."""
        results = evaluate_quality(self.vectors, bits_configs=[3, 5])
        rmse = {r.scheme: r.rmse for r in results}
        assert rmse["PrismKV-5bit"] <= rmse["PrismKV-3bit"]

    def test_throughput_positive(self):
        results = evaluate_quality(self.vectors, bits_configs=[4])
        for r in results:
            assert r.throughput_vps > 0

    def test_n_vectors_correct(self):
        results = evaluate_quality(self.vectors, bits_configs=[4])
        for r in results:
            assert r.n_vectors == 500


# ---------------------------------------------------------------------------
# run_e2e_benchmark
# ---------------------------------------------------------------------------


class TestRunE2EBenchmark:
    def test_synthetic_returns_report(self):
        report = run_e2e_benchmark(
            head_dim=66, n_heads=4, n_layers=4,
            n_synthetic=200, context_lengths=[1024], bits_configs=[4],
            seed=7
        )
        assert isinstance(report, E2EReport)

    def test_memory_profiles_populated(self):
        report = run_e2e_benchmark(
            head_dim=66, n_heads=4, n_layers=4,
            n_synthetic=100, context_lengths=[1024, 4096], bits_configs=[3, 4],
        )
        # FP16 + 3bit + 4bit × 2 context lengths = 6 profiles
        assert len(report.memory_profiles) == 6

    def test_quality_results_populated(self):
        report = run_e2e_benchmark(
            head_dim=66, n_heads=4, n_layers=4,
            n_synthetic=100, bits_configs=[3, 4],
        )
        # PrismKV + 2DPolar at each bit budget = 4
        assert len(report.quality_results) == 4

    def test_custom_kv_vectors(self):
        torch.manual_seed(0)
        kv = torch.randn(300, 64)
        report = run_e2e_benchmark(
            kv_vectors=kv, head_dim=64, n_heads=4, n_layers=4,
            bits_configs=[4],
        )
        assert report.quality_results[0].n_vectors == 300

    def test_report_head_dim_matches(self):
        report = run_e2e_benchmark(
            head_dim=99, n_heads=4, n_layers=4, n_synthetic=100, bits_configs=[4]
        )
        assert report.head_dim == 99

    def test_print_e2e_table_does_not_crash(self, capsys):
        report = run_e2e_benchmark(
            head_dim=66, n_heads=4, n_layers=4,
            n_synthetic=100, context_lengths=[1024], bits_configs=[4],
        )
        print_e2e_table(report)
        captured = capsys.readouterr()
        assert "FP16" in captured.out
        assert "PrismKV-4bit" in captured.out
