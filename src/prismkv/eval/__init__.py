"""
prismkv.eval — KV distribution collection and benchmarking utilities.

Optional dependency group. Install with:  pip install prismkv[eval]
"""

from prismkv.eval.kv_collector import KVCollector
from prismkv.eval.benchmark import run_benchmark
from prismkv.eval.model_arch import ModelArch, ModelArchRegistry, get_n_kv_heads
from prismkv.eval.e2e_benchmark import run_e2e_benchmark, print_e2e_table, E2EReport

__all__ = [
    "KVCollector", "run_benchmark",
    "ModelArch", "ModelArchRegistry", "get_n_kv_heads",
    "run_e2e_benchmark", "print_e2e_table", "E2EReport",
]
