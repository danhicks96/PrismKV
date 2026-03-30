"""
prismkv.eval — KV distribution collection and benchmarking utilities.

Optional dependency group. Install with:  pip install prismkv[eval]
"""

from prismkv.eval.kv_collector import KVCollector
from prismkv.eval.benchmark import run_benchmark
from prismkv.eval.model_arch import ModelArch, ModelArchRegistry, get_n_kv_heads

__all__ = ["KVCollector", "run_benchmark", "ModelArch", "ModelArchRegistry", "get_n_kv_heads"]
