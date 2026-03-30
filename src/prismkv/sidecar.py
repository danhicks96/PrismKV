"""
sidecar.py — Lightweight HTTP compression service for framework-agnostic integration.

Runs as a standalone process.  Any inference engine that can make HTTP calls
(Python, Go, Rust, C++, Node.js, …) can offload KV compression to this service
without importing any PrismKV Python code.

This is the integration path for engines that do not expose raw KV tensors to
Python (e.g. Ollama / llama.cpp): intercept tensors at the application layer
(before they would be passed to the engine, or captured via model hooks), send
them to the sidecar for compression, store the codes, and decompress on demand.

API
---
POST /compress
    Request body (JSON):
        {
          "k": [[float, ...], ...],     # (N, head_dim) float32
          "v": [[float, ...], ...],     # (N, head_dim) float32
          "head_dim": int,              # optional — inferred from k if omitted
          "bits_z": int,                # optional — default 4
          "bits_r": int,                # optional — default 4
          "bits_theta": int             # optional — default 4
        }
    Response body (JSON):
        {
          "k_codes": [[int, ...], ...], # (N, n_groups) int16
          "v_codes": [[int, ...], ...],
          "head_dim": int,
          "n_groups": int,
          "bits_per_dim": float
        }

POST /decompress
    Request body (JSON):
        {
          "k_codes": [[int, ...], ...], # (N, n_groups) int16
          "v_codes": [[int, ...], ...],
          "head_dim": int,              # original (unpadded) head_dim
          "bits_z": int,                # must match compress call
          "bits_r": int,
          "bits_theta": int
        }
    Response body (JSON):
        {
          "k": [[float, ...], ...],     # (N, head_dim) float32
          "v": [[float, ...], ...]
        }

GET /health
    Returns: {"status": "ok", "version": "1.0.0"}

Running
-------
    # From the command line:
    python -m prismkv.sidecar --port 8765 --host 127.0.0.1

    # From Python:
    from prismkv.sidecar import PrismKVSidecar
    server = PrismKVSidecar(port=8765)
    server.start()          # blocking
    # or
    server.start_background()  # non-blocking (thread)
    server.stop()

Client example (Python requests)
----------------------------------
    import requests
    import numpy as np

    k = np.random.randn(10, 64).astype(np.float32)
    v = np.random.randn(10, 64).astype(np.float32)

    resp = requests.post("http://localhost:8765/compress", json={
        "k": k.tolist(), "v": v.tolist(), "head_dim": 64
    })
    codes = resp.json()

    resp2 = requests.post("http://localhost:8765/decompress", json={
        "k_codes": codes["k_codes"], "v_codes": codes["v_codes"],
        "head_dim": 64,
    })
    result = resp2.json()
    k_hat = np.array(result["k"])   # reconstructed

Author: Dan Hicks (github.com/danhicks96)
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional

import torch

from prismkv.cache.backend import PrismKVBackend
from prismkv.cache.cache_config import PrismKVConfig

_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    """HTTP request handler for the PrismKV sidecar service."""

    # Set by PrismKVSidecar before serving
    backend_cache: Dict[str, PrismKVBackend] = {}
    backend_lock = threading.Lock()

    def log_message(self, fmt: str, *args: Any) -> None:  # silence access logs
        pass

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "version": _VERSION})
        else:
            self._send_json(404, {"error": f"Unknown endpoint: {self.path}"})

    def do_POST(self) -> None:
        body = self._read_body()
        if body is None:
            return

        if self.path == "/compress":
            self._handle_compress(body)
        elif self.path == "/decompress":
            self._handle_decompress(body)
        else:
            self._send_json(404, {"error": f"Unknown endpoint: {self.path}"})

    # ------------------------------------------------------------------
    # Endpoint implementations
    # ------------------------------------------------------------------

    def _handle_compress(self, body: dict) -> None:
        try:
            k = torch.tensor(body["k"], dtype=torch.float32)
            v = torch.tensor(body["v"], dtype=torch.float32)
        except (KeyError, ValueError, TypeError, RuntimeError) as e:
            self._send_json(400, {"error": f"Invalid k/v: {e}"})
            return

        if k.dim() != 2 or v.dim() != 2:
            self._send_json(400, {"error": "k and v must be 2-D arrays (N, head_dim)"})
            return

        head_dim = k.shape[1]
        bits_z = int(body.get("bits_z", 4))
        bits_r = int(body.get("bits_r", 4))
        bits_theta = int(body.get("bits_theta", 4))

        backend = self._get_backend(head_dim, bits_z, bits_r, bits_theta)
        k_codes, v_codes = backend.compress(k, v)

        self._send_json(200, {
            "k_codes": k_codes.tolist(),
            "v_codes": v_codes.tolist(),
            "head_dim": head_dim,
            "n_groups": backend.n_groups,
            "bits_per_dim": (bits_z + bits_r + bits_theta) / 3,
        })

    def _handle_decompress(self, body: dict) -> None:
        try:
            k_codes = torch.tensor(body["k_codes"], dtype=torch.int16)
            v_codes = torch.tensor(body["v_codes"], dtype=torch.int16)
            head_dim = int(body["head_dim"])
        except (KeyError, ValueError, TypeError) as e:
            self._send_json(400, {"error": f"Invalid codes or head_dim: {e}"})
            return

        if k_codes.dim() != 2 or v_codes.dim() != 2:
            self._send_json(400, {"error": "k_codes and v_codes must be 2-D arrays (N, n_groups)"})
            return

        bits_z = int(body.get("bits_z", 4))
        bits_r = int(body.get("bits_r", 4))
        bits_theta = int(body.get("bits_theta", 4))

        backend = self._get_backend(head_dim, bits_z, bits_r, bits_theta)

        try:
            k, v = backend.decompress(k_codes, v_codes)
        except Exception as e:
            self._send_json(500, {"error": f"Decompress failed: {e}"})
            return

        self._send_json(200, {
            "k": k.tolist(),
            "v": v.tolist(),
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_backend(
        self, head_dim: int, bits_z: int, bits_r: int, bits_theta: int
    ) -> PrismKVBackend:
        cache_key = f"{head_dim}:{bits_z}:{bits_r}:{bits_theta}"
        with self.__class__.backend_lock:
            if cache_key not in self.__class__.backend_cache:
                cfg = PrismKVConfig(bits_z=bits_z, bits_r=bits_r, bits_theta=bits_theta)
                self.__class__.backend_cache[cache_key] = PrismKVBackend(cfg, head_dim=head_dim)
            return self.__class__.backend_cache[cache_key]

    def _read_body(self) -> Optional[dict]:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self._send_json(400, {"error": "Empty request body"})
            return None
        try:
            raw = self.rfile.read(length)
            return json.loads(raw)
        except (json.JSONDecodeError, Exception) as e:
            self._send_json(400, {"error": f"JSON parse error: {e}"})
            return None

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Server class
# ---------------------------------------------------------------------------


class PrismKVSidecar:
    """
    Lightweight HTTP sidecar for framework-agnostic KV compression.

    Parameters
    ----------
    host : str — bind address (default "127.0.0.1"; use "0.0.0.0" for LAN)
    port : int — TCP port (default 8765)

    Usage
    -----
        server = PrismKVSidecar(port=8765)
        server.start()   # blocks — Ctrl+C to stop

    Background mode (for testing or embedding in another process):
        server = PrismKVSidecar(port=8765)
        server.start_background()
        # ... do work ...
        server.stop()
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start serving — blocks until KeyboardInterrupt or stop()."""
        self._server = HTTPServer((self.host, self.port), _Handler)
        print(f"PrismKV sidecar running at http://{self.host}:{self.port}")
        print("Endpoints: POST /compress  POST /decompress  GET /health")
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self._server.server_close()

    def start_background(self) -> None:
        """Start serving in a background thread (non-blocking)."""
        self._server = HTTPServer((self.host, self.port), _Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background server."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="PrismKV sidecar — HTTP KV compression service"
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address (default 127.0.0.1; use 0.0.0.0 for LAN)")
    parser.add_argument("--port", type=int, default=8765,
                        help="TCP port (default 8765)")
    args = parser.parse_args()

    PrismKVSidecar(host=args.host, port=args.port).start()


if __name__ == "__main__":
    _main()
