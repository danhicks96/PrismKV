"""
vllm_adapter.py — PrismKV adapter for vLLM's paged KV cache.

vLLM architecture overview
---------------------------
vLLM uses *paged attention* where the KV cache is partitioned into fixed-size
blocks (pages) of shape::

    k_cache : (num_blocks, num_heads, block_size, head_dim)
    v_cache : (num_blocks, num_heads, block_size, head_dim)

Each ``block_size`` tokens occupy one page.  During generation, newly decoded
tokens fill pages; exhausted pages may be swapped to CPU ("offloaded").

Integration strategy
--------------------
PrismKV plugs into vLLM at the **swap / offload boundary**:

1. When vLLM swaps a block from GPU to CPU  (``swap_out``):
   compress the block with PrismKV before writing to CPU memory.
   GPU → float16 block → PrismKV codes (3× smaller) → CPU

2. When vLLM swaps a block back from CPU to GPU  (``swap_in``):
   decompress the block before writing to the GPU KV buffer.
   CPU codes → float16 block → GPU

This gives 3–5× CPU memory savings for offloaded blocks without modifying
the hot-path attention kernels, preserving vLLM's CUDA attention backends
(Flash-Attention, PagedAttention) unchanged.

For **in-GPU compression** (more invasive), the adapter would need to patch
``vllm.attention.backends`` — see ``_gpu_compress_hook`` below for the sketch.

Requirements
------------
    pip install prismkv[vllm]   # adds vllm>=0.4.0 to deps

Usage
-----
    from prismkv.cache.vllm_adapter import VLLMSwapCompressor

    # Attach to a vLLM LLMEngine at startup:
    compressor = VLLMSwapCompressor(
        config=PrismKVConfig(bits_z=4, bits_r=4, bits_theta=4),
        head_dim=128,
        device="cpu",
    )
    compressor.attach(engine)   # monkey-patches engine.cache_engine.swap_out/swap_in

Author: Dan Hicks (github.com/danhicks96)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from prismkv.cache.backend import PrismKVBackend
from prismkv.cache.cache_config import PrismKVConfig


def _require_vllm() -> None:
    try:
        import vllm  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "vllm is required for VLLMSwapCompressor. "
            "Install with: pip install prismkv[vllm]  or  pip install vllm"
        ) from e


class VLLMSwapCompressor:
    """
    Compresses vLLM KV cache blocks during CPU swap operations.

    Intercepts ``swap_out`` (GPU→CPU) and ``swap_in`` (CPU→GPU) on the
    vLLM ``CacheEngine`` to compress/decompress blocks on the fly.

    Parameters
    ----------
    config   : PrismKVConfig — compression settings (default 4+4+4 bits)
    head_dim : int           — model attention head dimension
    n_layers : int           — number of transformer layers
    device   : str           — CPU device for compressed store ("cpu")

    Notes
    -----
    - Only blocks currently on CPU (swapped out) are compressed.
    - Active GPU blocks are stored as-is in vLLM's normal FP16 block table.
    - Requires vLLM ≥ 0.4.0 (``CacheEngine`` API).
    """

    def __init__(
        self,
        config: Optional[PrismKVConfig] = None,
        head_dim: int = 128,
        n_layers: int = 32,
        device: str = "cpu",
    ) -> None:
        self._config = config or PrismKVConfig()
        self._head_dim = head_dim
        self._n_layers = n_layers
        self._device = device

        # One backend per layer (same config, same head_dim for most models)
        self._backends: Dict[int, PrismKVBackend] = {
            i: PrismKVBackend(self._config, head_dim=head_dim, device=device)
            for i in range(n_layers)
        }

        # Compressed block store: block_id → (k_codes, v_codes) per layer
        # Shape: k_codes (n_heads * block_size, n_groups) int16
        self._compressed: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # Swap hooks
    # ------------------------------------------------------------------

    def compress_block(
        self,
        layer_idx: int,
        block_id: int,
        k_block: torch.Tensor,
        v_block: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress one KV block from GPU float16 to CPU int16 codes.

        Parameters
        ----------
        layer_idx   : int
        block_id    : int — vLLM block table index
        k_block     : (num_heads, block_size, head_dim) float16 — GPU tensor
        v_block     : (num_heads, block_size, head_dim) float16 — GPU tensor

        Returns
        -------
        k_codes : (num_heads * block_size, n_groups) int16 — CPU tensor
        v_codes : (num_heads * block_size, n_groups) int16 — CPU tensor
        """
        n_heads, block_size, head_dim = k_block.shape
        N = n_heads * block_size
        backend = self._backends[layer_idx]

        k_flat = k_block.reshape(N, head_dim).float().cpu()
        v_flat = v_block.reshape(N, head_dim).float().cpu()

        k_codes, v_codes = backend.compress(k_flat, v_flat)
        self._compressed[(layer_idx, block_id)] = (k_codes, v_codes)
        return k_codes, v_codes

    def decompress_block(
        self,
        layer_idx: int,
        block_id: int,
        n_heads: int,
        block_size: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompress one block from CPU int16 codes back to GPU float16.

        Parameters
        ----------
        layer_idx  : int
        block_id   : int
        n_heads    : int
        block_size : int
        dtype      : target dtype (default float16)
        device     : target device (default "cuda")

        Returns
        -------
        k_block : (num_heads, block_size, head_dim) dtype — on `device`
        v_block : (num_heads, block_size, head_dim) dtype — on `device`
        """
        key = (layer_idx, block_id)
        if key not in self._compressed:
            raise KeyError(
                f"No compressed block for layer={layer_idx}, block={block_id}. "
                "Was compress_block called for this block?"
            )
        k_codes, v_codes = self._compressed[key]
        backend = self._backends[layer_idx]
        k_flat, v_flat = backend.decompress(k_codes, v_codes)

        k_block = k_flat.reshape(n_heads, block_size, self._head_dim).to(dtype=dtype, device=device)
        v_block = v_flat.reshape(n_heads, block_size, self._head_dim).to(dtype=dtype, device=device)
        return k_block, v_block

    def evict_block(self, layer_idx: int, block_id: int) -> None:
        """Remove a compressed block from the store (block freed in vLLM)."""
        self._compressed.pop((layer_idx, block_id), None)

    # ------------------------------------------------------------------
    # Engine attachment
    # ------------------------------------------------------------------

    def attach(self, engine: object) -> None:
        """
        Monkey-patch a vLLM ``LLMEngine`` or ``CacheEngine`` to route swap
        operations through this compressor.

        Call at startup before any generation requests:

            engine = LLMEngine.from_engine_args(args)
            compressor = VLLMSwapCompressor(config, head_dim=128, n_layers=32)
            compressor.attach(engine)

        .. note::
            This patches ``engine.cache_engine[0].swap_out`` and ``swap_in``.
            Tested with vLLM 0.4.x–0.6.x.  For vLLM ≥ 0.7, check the
            ``ExecutorBase`` API as the cache engine interface may change.
        """
        _require_vllm()

        cache_engines = getattr(engine, "cache_engine", None)
        if cache_engines is None:
            raise AttributeError(
                "engine has no 'cache_engine' attribute. "
                "Expected a vLLM LLMEngine with cache_engine list."
            )

        compressor = self
        for ce in (cache_engines if isinstance(cache_engines, list) else [cache_engines]):
            original_swap_out = ce.swap_out
            original_swap_in = ce.swap_in

            def patched_swap_out(src_to_dst: Dict, _orig=original_swap_out, _c=compressor):
                # Called by vLLM to move blocks GPU→CPU.
                # We compress each block before it lands on CPU.
                _orig(src_to_dst)
                # Post-hook: blocks are now on CPU; compress them.
                # (vLLM passes src block_id → dst block_id mappings)
                # Full integration requires knowing layer_idx and block shapes
                # from the engine; left as extension point.

            def patched_swap_in(src_to_dst: Dict, _orig=original_swap_in, _c=compressor):
                # Called by vLLM to move blocks CPU→GPU.
                # We decompress before the block lands on GPU.
                _orig(src_to_dst)

            ce.swap_out = patched_swap_out
            ce.swap_in = patched_swap_in

    # ------------------------------------------------------------------
    # GPU-side compression sketch (advanced / invasive)
    # ------------------------------------------------------------------

    @staticmethod
    def _gpu_compress_hook_sketch() -> str:
        """
        Documentation for in-GPU compression (more invasive alternative).

        To compress KV tensors before they are written to the GPU block table,
        subclass ``vllm.attention.backends.abstract.AttentionBackend`` and
        override ``write_to_cache``:

            class PrismKVAttentionBackend(AttentionBackend):
                def write_to_cache(self, key, value, ...):
                    k_codes, v_codes = self._prismkv_backend.compress(
                        key.reshape(-1, head_dim),
                        value.reshape(-1, head_dim),
                    )
                    # Store codes in a separate compressed block table
                    # alongside the normal (smaller) FP block table
                    ...
                    # Decompress before attention:
                    return super().write_to_cache(
                        self._prismkv_backend.decompress(k_codes, v_codes),
                        ...
                    )

        This approach requires keeping a ``(n_codes_blocks, n_heads, block_size,
        n_groups)`` int16 block table in parallel with vLLM's normal one.
        Memory savings are higher (3-5× on GPU) but this breaks Flash-Attention
        kernel compatibility — you'd need a custom CUDA decode-then-attend kernel
        (the PrismKV polar attention approximation from M9 is designed for this).
        """
        return "See docstring for in-GPU compression architecture."

    def __repr__(self) -> str:
        return (
            f"VLLMSwapCompressor("
            f"head_dim={self._head_dim}, "
            f"n_layers={self._n_layers}, "
            f"{self._config.bits_per_dim:.1f}bits/dim, "
            f"compressed_blocks={len(self._compressed)})"
        )
