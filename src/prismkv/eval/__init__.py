"""
prismkv.eval — KV distribution collection and benchmarking utilities.

Optional dependency group. Install with:  pip install prismkv[eval]
"""

from prismkv.eval.kv_collector import KVCollector
from prismkv.eval.benchmark import run_benchmark

__all__ = ["KVCollector", "run_benchmark"]
