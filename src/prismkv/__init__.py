"""
PrismKV — 3-D Stacked-Plane KV Cache Quantizer

A training-free KV cache compression algorithm that extends the 2-D polar
quantization of TurboQuant into a conditional 3-D structure. Instead of
grouping KV vector dimensions into independent 2-D pairs, PrismKV groups
them into triplets and uses a coarse z-index to condition the fine 2-D polar
quantization of the (x, y) plane — capturing cross-plane relationships that
flat 2-D schemes miss.

Author: Dan Hicks (github.com/danhicks96)
Repo:   https://github.com/danhicks96/PrismKV
"""

from prismkv.quantizer.baseline_2d import PolarQuantizer2D
from prismkv.quantizer.stacked_plane import StackedPlaneQuantizer
from prismkv.quantizer.learned_codebook import LearnedSliceCodebook

__all__ = ["PolarQuantizer2D", "StackedPlaneQuantizer", "LearnedSliceCodebook"]
__version__ = "1.3.0"

# Cache classes available when transformers is installed
try:
    from prismkv.cache import PrismKVCache, PrismKVConfig, DimAligner
    __all__ += ["PrismKVCache", "PrismKVConfig", "DimAligner"]
except ImportError:
    pass
