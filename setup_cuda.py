#!/usr/bin/env python
"""
Build the PrismKV CUDA extension.

Requires:
  - CUDA >= 11.8
  - PyTorch with CUDA support (pip install torch --index-url .../cu118)
  - nvcc on PATH

Usage:
  python setup_cuda.py build_ext --inplace
  # Then: from prismkv.cuda import polar_attn_fwd, CUDA_AVAILABLE
"""
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="prismkv_cuda",
    ext_modules=[
        CUDAExtension(
            name="prismkv_cuda",
            sources=[
                "src/prismkv/cuda/prismkv_cuda.cpp",
                "src/prismkv/cuda/polar_attn_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-arch=sm_80",          # Ampere (A100); sm_70 for V100
                    "--use_fast_math",
                    "-std=c++17",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
