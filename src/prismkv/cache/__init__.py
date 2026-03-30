"""
prismkv.cache — Compressed KV cache: framework-agnostic core + HuggingFace adapter.

Framework-agnostic (no extra deps):
    from prismkv.cache import PrismKVConfig, PrismKVBackend, RawKVCache

HuggingFace drop-in (requires transformers):
    from prismkv.cache import PrismKVCache          # pip install prismkv[cache]

vLLM swap-compressor (requires vllm):
    from prismkv.cache import VLLMSwapCompressor    # pip install prismkv[vllm]

HTTP sidecar (stdlib only):
    from prismkv.sidecar import PrismKVSidecar
    python -m prismkv.sidecar --port 8765
"""

from prismkv.cache.cache_config import PrismKVConfig
from prismkv.cache.dim_aligner import DimAligner
from prismkv.cache.backend import CacheBackend, PrismKVBackend
from prismkv.cache.raw_cache import RawKVCache
from prismkv.cache.kv_cache import PrismKVCache
from prismkv.cache.cache_store import save_cache, load_cache
from prismkv.cache.vllm_adapter import VLLMSwapCompressor

__all__ = [
    # Config
    "PrismKVConfig", "DimAligner",
    # Framework-agnostic
    "CacheBackend", "PrismKVBackend", "RawKVCache",
    # HuggingFace adapter
    "PrismKVCache",
    # Persistence
    "save_cache", "load_cache",
    # vLLM adapter
    "VLLMSwapCompressor",
]
