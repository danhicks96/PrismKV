"""
tests/test_m12_framework_agnostic.py — Tests for M12 framework-agnostic layer.

Covers:
  - CacheBackend protocol / PrismKVBackend
  - RawKVCache (single backend, dict backends, multi-layer)
  - VLLMSwapCompressor (no vLLM install required — tests the compress/decompress
    math, not the engine attachment)
  - PrismKVSidecar HTTP service (starts real server on dynamic port)
"""

import math
import socket
import threading
import time

import pytest
import torch

from prismkv.cache.backend import CacheBackend, PrismKVBackend
from prismkv.cache.cache_config import PrismKVConfig
from prismkv.cache.raw_cache import RawKVCache
from prismkv.cache.vllm_adapter import VLLMSwapCompressor
from prismkv.sidecar import PrismKVSidecar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_kv(n: int = 100, head_dim: int = 66, seed: int = 0):
    g = torch.Generator()
    g.manual_seed(seed)
    k = torch.randn(n, head_dim, generator=g)
    v = torch.randn(n, head_dim, generator=g)
    return k, v


# ---------------------------------------------------------------------------
# PrismKVBackend
# ---------------------------------------------------------------------------

class TestPrismKVBackend:
    def setup_method(self):
        self.cfg = PrismKVConfig(bits_z=4, bits_r=4, bits_theta=4)
        self.backend = PrismKVBackend(self.cfg, head_dim=66)

    def test_compress_returns_int16(self):
        k, v = _make_kv(50, 66)
        k_codes, v_codes = self.backend.compress(k, v)
        assert k_codes.dtype == torch.int16
        assert v_codes.dtype == torch.int16

    def test_codes_shape(self):
        k, v = _make_kv(50, 66)
        k_codes, _ = self.backend.compress(k, v)
        # padded_dim=66, n_groups=22
        assert k_codes.shape == (50, 22)

    def test_decompress_shape(self):
        k, v = _make_kv(50, 66)
        k_codes, v_codes = self.backend.compress(k, v)
        k_hat, v_hat = self.backend.decompress(k_codes, v_codes)
        assert k_hat.shape == (50, 66)
        assert v_hat.shape == (50, 66)

    def test_decompress_close_to_original(self):
        k, v = _make_kv(200, 66, seed=7)
        k_codes, v_codes = self.backend.compress(k, v)
        k_hat, v_hat = self.backend.decompress(k_codes, v_codes)
        rmse = (k - k_hat).pow(2).mean().sqrt().item()
        assert rmse < 1.0, f"RMSE too high: {rmse}"

    def test_head_dim_property(self):
        assert self.backend.head_dim == 66

    def test_n_groups_property(self):
        assert self.backend.n_groups == 22  # 66 // 3

    def test_non_multiple_of_3_padded(self):
        """head_dim=64 is padded to 66 internally."""
        b = PrismKVBackend(self.cfg, head_dim=64)
        assert b.padded_dim == 66
        assert b.head_dim == 64
        k, v = _make_kv(20, 64)
        k_codes, v_codes = b.compress(k, v)
        k_hat, v_hat = b.decompress(k_codes, v_codes)
        assert k_hat.shape == (20, 64)

    def test_backend_repr(self):
        r = repr(self.backend)
        assert "PrismKVBackend" in r
        assert "4.0bits/dim" in r

    def test_implements_cache_backend_interface(self):
        """Duck-typing: PrismKVBackend satisfies CacheBackend."""
        assert hasattr(self.backend, "compress")
        assert hasattr(self.backend, "decompress")
        assert hasattr(self.backend, "config")
        assert hasattr(self.backend, "head_dim")


# ---------------------------------------------------------------------------
# RawKVCache
# ---------------------------------------------------------------------------

class TestRawKVCache:
    def setup_method(self):
        self.cfg = PrismKVConfig(bits_z=4, bits_r=4, bits_theta=4)
        self.backend = PrismKVBackend(self.cfg, head_dim=64)
        self.cache = RawKVCache(self.backend)

    def test_update_returns_full_sequence(self):
        k, v = _make_kv(10, 64)
        k_full, v_full = self.cache.update(0, k, v)
        assert k_full.shape == (10, 64)

    def test_update_accumulates(self):
        k1, v1 = _make_kv(5, 64, seed=1)
        k2, v2 = _make_kv(5, 64, seed=2)
        self.cache.update(0, k1, v1)
        k_full, v_full = self.cache.update(0, k2, v2)
        assert k_full.shape[0] == 10  # 5 + 5 tokens

    def test_multidim_input(self):
        """(batch, n_heads, seq_len, head_dim) input shape."""
        k = torch.randn(2, 4, 6, 64)
        v = torch.randn(2, 4, 6, 64)
        k_full, v_full = self.cache.update(0, k, v)
        assert k_full.shape == (2, 4, 6, 64)

    def test_multidim_accumulates(self):
        k1 = torch.randn(1, 2, 3, 64)
        v1 = torch.randn(1, 2, 3, 64)
        k2 = torch.randn(1, 2, 5, 64)
        v2 = torch.randn(1, 2, 5, 64)
        self.cache.update(0, k1, v1)
        k_full, v_full = self.cache.update(0, k2, v2)
        assert k_full.shape == (1, 2, 8, 64)  # 3 + 5 seq tokens

    def test_multi_layer(self):
        for layer_idx in range(3):
            k, v = _make_kv(10, 64, seed=layer_idx)
            self.cache.update(layer_idx, k, v)
        assert set(self.cache.cached_layers) == {0, 1, 2}

    def test_get_after_update(self):
        k, v = _make_kv(10, 64)
        self.cache.update(0, k, v)
        k_got, v_got = self.cache.get(0)
        assert k_got.shape == (10, 64)

    def test_get_before_update_raises(self):
        with pytest.raises(KeyError):
            self.cache.get(99)

    def test_get_codes(self):
        k, v = _make_kv(10, 64)
        self.cache.update(0, k, v)
        kc, vc = self.cache.get_codes(0)
        assert kc.dtype == torch.int16

    def test_get_seq_length(self):
        k, v = _make_kv(12, 64)
        self.cache.update(0, k, v)
        assert self.cache.get_seq_length(0) == 12

    def test_clear_specific_layer(self):
        for i in range(3):
            k, v = _make_kv(5, 64)
            self.cache.update(i, k, v)
        self.cache.clear(layer_idx=1)
        assert 1 not in self.cache.cached_layers
        assert 0 in self.cache.cached_layers
        assert 2 in self.cache.cached_layers

    def test_clear_all(self):
        for i in range(3):
            k, v = _make_kv(5, 64)
            self.cache.update(i, k, v)
        self.cache.clear()
        assert self.cache.cached_layers == []

    def test_memory_footprint(self):
        k, v = _make_kv(100, 64)
        self.cache.update(0, k, v)
        fp = self.cache.memory_footprint()
        assert fp["compression"] > 1.0
        assert fp["codes_bytes"] < fp["fp16_bytes"]
        assert fp["n_layers"] == 1

    def test_dict_backends(self):
        """Dict of backends — one per layer with different configs."""
        backends = {
            0: PrismKVBackend(PrismKVConfig(bits_z=3, bits_r=3, bits_theta=3), head_dim=64),
            1: PrismKVBackend(PrismKVConfig(bits_z=5, bits_r=5, bits_theta=5), head_dim=64),
        }
        cache = RawKVCache(backends)
        for i in [0, 1]:
            k, v = _make_kv(10, 64, seed=i)
            cache.update(i, k, v)
        assert len(cache.cached_layers) == 2

    def test_repr(self):
        k, v = _make_kv(10, 64)
        self.cache.update(0, k, v)
        r = repr(self.cache)
        assert "RawKVCache" in r


# ---------------------------------------------------------------------------
# VLLMSwapCompressor (tests the math; no vLLM install required)
# ---------------------------------------------------------------------------

class TestVLLMSwapCompressor:
    def setup_method(self):
        self.compressor = VLLMSwapCompressor(
            config=PrismKVConfig(bits_z=4, bits_r=4, bits_theta=4),
            head_dim=64,
            n_layers=4,
        )

    def _make_block(self, n_heads=8, block_size=16, head_dim=64):
        return torch.randn(n_heads, block_size, head_dim)

    def test_compress_block_returns_int16(self):
        k_block = self._make_block()
        v_block = self._make_block()
        k_codes, v_codes = self.compressor.compress_block(0, 42, k_block, v_block)
        assert k_codes.dtype == torch.int16
        assert v_codes.dtype == torch.int16

    def test_compress_block_shape(self):
        n_heads, block_size, head_dim = 8, 16, 64
        k_block = self._make_block(n_heads, block_size, head_dim)
        v_block = self._make_block(n_heads, block_size, head_dim)
        k_codes, _ = self.compressor.compress_block(0, 1, k_block, v_block)
        # N = n_heads * block_size, n_groups = padded_dim // 3 = 66//3 = 22
        assert k_codes.shape == (n_heads * block_size, 22)

    def test_round_trip(self):
        n_heads, block_size = 4, 8
        k_block = self._make_block(n_heads, block_size)
        v_block = self._make_block(n_heads, block_size)
        self.compressor.compress_block(0, 7, k_block, v_block)
        k_hat, v_hat = self.compressor.decompress_block(
            0, 7, n_heads=n_heads, block_size=block_size, dtype=torch.float32, device="cpu"
        )
        assert k_hat.shape == (n_heads, block_size, 64)
        rmse = (k_block - k_hat).pow(2).mean().sqrt().item()
        assert rmse < 1.0

    def test_decompress_missing_block_raises(self):
        with pytest.raises(KeyError):
            self.compressor.decompress_block(0, 999, n_heads=4, block_size=8)

    def test_evict_block(self):
        k_block = self._make_block(4, 8)
        v_block = self._make_block(4, 8)
        self.compressor.compress_block(1, 5, k_block, v_block)
        self.compressor.evict_block(1, 5)
        with pytest.raises(KeyError):
            self.compressor.decompress_block(1, 5, n_heads=4, block_size=8)

    def test_repr(self):
        assert "VLLMSwapCompressor" in repr(self.compressor)


# ---------------------------------------------------------------------------
# PrismKVSidecar HTTP service
# ---------------------------------------------------------------------------

class TestPrismKVSidecar:
    def setup_method(self):
        port = _free_port()
        self.server = PrismKVSidecar(host="127.0.0.1", port=port)
        self.server.start_background()
        self.base_url = f"http://127.0.0.1:{port}"
        # Wait for server to be ready
        for _ in range(30):
            try:
                import urllib.request
                urllib.request.urlopen(f"{self.base_url}/health", timeout=1)
                break
            except Exception:
                time.sleep(0.05)

    def teardown_method(self):
        self.server.stop()

    def _post(self, endpoint: str, payload: dict) -> dict:
        import json as _json
        import urllib.request
        body = _json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return _json.loads(resp.read())

    def _get(self, endpoint: str) -> dict:
        import json as _json
        import urllib.request
        with urllib.request.urlopen(f"{self.base_url}{endpoint}", timeout=5) as resp:
            return _json.loads(resp.read())

    def test_health_endpoint(self):
        resp = self._get("/health")
        assert resp["status"] == "ok"

    def test_compress_returns_codes(self):
        k = torch.randn(5, 66).tolist()
        v = torch.randn(5, 66).tolist()
        resp = self._post("/compress", {"k": k, "v": v, "head_dim": 66})
        assert "k_codes" in resp
        assert "v_codes" in resp
        assert resp["head_dim"] == 66

    def test_compress_decompress_round_trip(self):
        k_orig = torch.randn(8, 66)
        v_orig = torch.randn(8, 66)
        comp = self._post("/compress", {"k": k_orig.tolist(), "v": v_orig.tolist()})
        decomp = self._post("/decompress", {
            "k_codes": comp["k_codes"],
            "v_codes": comp["v_codes"],
            "head_dim": 66,
        })
        k_hat = torch.tensor(decomp["k"])
        rmse = (k_orig - k_hat).pow(2).mean().sqrt().item()
        assert rmse < 1.0

    def test_compress_non_multiple_of_3(self):
        """head_dim=64 should work (padded to 66 internally)."""
        k = torch.randn(5, 64).tolist()
        v = torch.randn(5, 64).tolist()
        resp = self._post("/compress", {"k": k, "v": v, "head_dim": 64})
        assert "k_codes" in resp

    def test_compress_bad_input(self):
        import json as _json
        import urllib.request
        import urllib.error
        body = _json.dumps({"k": "not_a_list", "v": [[1.0, 2.0]]}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/compress",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=5)
        assert exc.value.code == 400

    def test_unknown_endpoint(self):
        import urllib.request
        import urllib.error
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"{self.base_url}/unknown", timeout=5)
        assert exc.value.code == 404

    def test_custom_bit_config(self):
        k = torch.randn(4, 66).tolist()
        v = torch.randn(4, 66).tolist()
        resp = self._post("/compress", {
            "k": k, "v": v, "bits_z": 3, "bits_r": 3, "bits_theta": 3
        })
        assert resp["bits_per_dim"] == pytest.approx(3.0)
