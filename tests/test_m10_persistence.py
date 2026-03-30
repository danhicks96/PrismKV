"""
test_m10_persistence.py — Tests for cache persistence and APIAdapter (M10).
"""

import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

from prismkv.rag.adapters import APIAdapter
from prismkv.rag.schema import Chunk


# ---------------------------------------------------------------------------
# APIAdapter — mock HTTP server
# ---------------------------------------------------------------------------

import socket
import time


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_mock_server(handler_cls, port, ready_event):
    server = HTTPServer(("127.0.0.1", port), handler_cls)
    server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ready_event.set()
    server.handle_request()   # handle exactly one request then stop


class TestAPIAdapter:
    def _serve_once(self, response_body: bytes, content_type: str = "application/json"):
        """Start a temporary HTTP server on a free port; serve one response."""
        port = _free_port()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.end_headers()
                self.wfile.write(response_body)

            def log_message(self, *args):
                pass

        ready = threading.Event()
        t = threading.Thread(target=_run_mock_server, args=(Handler, port, ready), daemon=True)
        t.start()
        ready.wait(timeout=2)
        return port

    def test_get_list_of_strings(self):
        payload = json.dumps(["hello world", "foo bar"]).encode()
        port = self._serve_once(payload)
        adapter = APIAdapter(f"http://127.0.0.1:{port}/", source_id="test")
        chunks = adapter.chunks()
        assert len(chunks) == 2
        assert any(c.text == "hello world" for c in chunks)

    def test_get_list_of_dicts_with_text_field(self):
        payload = json.dumps([
            {"body": "The quick brown fox", "id": 1},
            {"body": "jumps over the lazy dog", "id": 2},
        ]).encode()
        port = self._serve_once(payload)
        adapter = APIAdapter(
            f"http://127.0.0.1:{port}/",
            text_field="body",
            source_id="docs",
        )
        chunks = adapter.chunks()
        assert len(chunks) == 2
        texts = [c.text for c in chunks]
        assert "The quick brown fox" in texts

    def test_get_single_dict(self):
        payload = json.dumps({"title": "Hello", "content": "World"}).encode()
        port = self._serve_once(payload)
        adapter = APIAdapter(f"http://127.0.0.1:{port}/", source_id="single")
        chunks = adapter.chunks()
        assert len(chunks) >= 1

    def test_source_id_stored_in_metadata(self):
        payload = json.dumps(["some text"]).encode()
        port = self._serve_once(payload)
        adapter = APIAdapter(f"http://127.0.0.1:{port}/", source_id="my_api")
        chunks = adapter.chunks()
        assert all(c.source_id == "my_api" for c in chunks)

    def test_network_error_raises_runtime_error(self):
        adapter = APIAdapter("http://127.0.0.1:19999/nonexistent", timeout=1)
        with pytest.raises(RuntimeError, match="failed"):
            adapter.chunks()

    def test_base_metadata_applied(self):
        payload = json.dumps(["text chunk"]).encode()
        port = self._serve_once(payload)
        adapter = APIAdapter(
            f"http://127.0.0.1:{port}/",
            metadata={"dataset": "wiki"},
        )
        chunks = adapter.chunks()
        assert all(c.metadata.get("dataset") == "wiki" for c in chunks)

    def test_chunks_are_chunk_objects(self):
        payload = json.dumps(["foo", "bar"]).encode()
        port = self._serve_once(payload)
        adapter = APIAdapter(f"http://127.0.0.1:{port}/")
        for chunk in adapter.chunks():
            assert isinstance(chunk, Chunk)
            assert chunk.id
            assert chunk.text


# ---------------------------------------------------------------------------
# Cache persistence (requires transformers)
# ---------------------------------------------------------------------------

try:
    from prismkv.cache import PrismKVCache, PrismKVConfig, save_cache, load_cache
    HAS_CACHE = True
except ImportError:
    HAS_CACHE = False


@pytest.mark.skipif(not HAS_CACHE, reason="transformers not installed")
class TestCachePersistence:
    def _make_cache(self, n_layers=3, n_heads=4, seq=8, head_dim=64):
        cache = PrismKVCache(PrismKVConfig(bits_z=4, bits_r=4, bits_theta=4))
        for i in range(n_layers):
            k = torch.randn(1, n_heads, seq, head_dim)
            v = torch.randn(1, n_heads, seq, head_dim)
            cache.update(k, v, layer_idx=i)
        return cache

    def test_save_creates_file(self):
        cache = self._make_cache()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cache.npz"
            save_cache(cache, path)
            assert path.exists()
            assert path.stat().st_size > 0

    def test_save_load_roundtrip_n_layers(self):
        cache = self._make_cache(n_layers=4)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cache.npz"
            save_cache(cache, path)
            loaded = load_cache(path, device="cpu")
        # Same number of layers with codes
        saved_layers = sum(1 for kc in cache._key_codes if kc is not None)
        loaded_layers = sum(1 for kc in loaded._key_codes if kc is not None)
        assert loaded_layers == saved_layers

    def test_save_load_codes_identical(self):
        cache = self._make_cache(n_layers=2)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cache.npz"
            save_cache(cache, path)
            loaded = load_cache(path, device="cpu")

        for i in range(2):
            orig_k = cache._key_codes[i]
            load_k = loaded._key_codes[i]
            if orig_k is not None and load_k is not None:
                assert orig_k.shape == load_k.shape
                assert (orig_k == load_k).all()

    def test_save_preserves_config(self):
        cfg = PrismKVConfig(bits_z=3, bits_r=5, bits_theta=4)
        cache = PrismKVCache(config=cfg)
        k = torch.randn(1, 2, 4, 64)
        v = torch.randn(1, 2, 4, 64)
        cache.update(k, v, layer_idx=0)

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cache.npz"
            save_cache(cache, path)
            loaded = load_cache(path)

        assert loaded._default_config.bits_z == 3
        assert loaded._default_config.bits_r == 5
        assert loaded._default_config.bits_theta == 4

    def test_save_load_subdirectory_created(self):
        cache = self._make_cache(n_layers=1)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "subdir" / "deep" / "cache.npz"
            save_cache(cache, path)
            assert path.exists()

    def test_memory_footprint_after_load(self):
        cache = self._make_cache(n_layers=3)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cache.npz"
            save_cache(cache, path)
            loaded = load_cache(path)
        fp = loaded.memory_footprint()
        assert fp["n_layers"] == 3
        assert fp["codes_bytes"] > 0
