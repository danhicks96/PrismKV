"""
prismkv.cache — Drop-in compressed KV cache for HuggingFace models.

Install transformers to use:  pip install prismkv[cache]
"""

from prismkv.cache.cache_config import PrismKVConfig
from prismkv.cache.dim_aligner import DimAligner
from prismkv.cache.kv_cache import PrismKVCache

__all__ = ["PrismKVConfig", "DimAligner", "PrismKVCache"]
