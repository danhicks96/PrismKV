"""
cache_store.py — Persistent serialization for PrismKVCache.

Saves and loads compressed int16 codes + quantizer configs to/from disk,
enabling cross-session KV cache reuse (e.g., for static prompt prefixes or
RAG context windows that don't change between requests).

File format (NPZ):
  - "codes_key_{i}"  : int16 array (N, m) for layer i
  - "codes_val_{i}"  : int16 array (N, m) for layer i
  - "config_bz"      : scalar int (default bits_z)
  - "config_br"      : scalar int (default bits_r)
  - "config_bt"      : scalar int (default bits_theta)
  - "n_layers"       : scalar int

Author: Dan Hicks (github.com/danhicks96)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
import torch

from prismkv.cache.cache_config import PrismKVConfig

if TYPE_CHECKING:
    from prismkv.cache.kv_cache import PrismKVCache


def save_cache(cache: "PrismKVCache", path: Union[str, Path]) -> None:
    """
    Serialize a PrismKVCache's compressed codes to a .npz file.

    Only the compressed int16 codes are saved (not the decoded FP tensors
    stored in the parent DynamicCache layers).  On reload, the FP tensors
    are reconstructed from the codes.

    Parameters
    ----------
    cache : PrismKVCache — must have at least one layer cached
    path  : destination file path (will be created/overwritten)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict = {}
    n_layers = len(cache._key_codes)

    for i, (kc, vc) in enumerate(zip(cache._key_codes, cache._val_codes)):
        if kc is not None:
            arrays[f"codes_key_{i}"] = kc.cpu().numpy()
            arrays[f"codes_val_{i}"] = vc.cpu().numpy()

    cfg = cache._default_config
    arrays["config_bz"] = np.int32(cfg.bits_z)
    arrays["config_br"] = np.int32(cfg.bits_r)
    arrays["config_bt"] = np.int32(cfg.bits_theta)
    arrays["n_layers"]  = np.int32(n_layers)

    np.savez_compressed(str(path), **arrays)


def load_cache(path: Union[str, Path], device: str = "cpu") -> "PrismKVCache":
    """
    Deserialize a PrismKVCache from a .npz file.

    The decoded FP tensors are reconstructed from codes using the saved config,
    and forwarded to the parent DynamicCache so all parent methods work.

    Parameters
    ----------
    path   : path to .npz file written by save_cache()
    device : torch device for tensors

    Returns
    -------
    PrismKVCache with codes and parent layers populated
    """
    from prismkv.cache.kv_cache import PrismKVCache

    data = np.load(str(path))

    cfg = PrismKVConfig(
        bits_z=int(data["config_bz"]),
        bits_r=int(data["config_br"]),
        bits_theta=int(data["config_bt"]),
    )
    n_layers = int(data["n_layers"])

    cache = PrismKVCache(config=cfg)

    for i in range(n_layers):
        key = f"codes_key_{i}"
        val = f"codes_val_{i}"
        if key not in data:
            continue

        kc = torch.from_numpy(data[key]).to(device)
        vc = torch.from_numpy(data[val]).to(device)

        # Extend internal lists
        while len(cache._key_codes) <= i:
            cache._key_codes.append(None)
            cache._val_codes.append(None)
        cache._key_codes[i] = kc
        cache._val_codes[i] = vc

        # Reconstruct FP tensors and seed the parent DynamicCache layer
        # codes shape: (N_flat, m) where N_flat = batch * n_heads * seq_len
        # We store as batch=1, n_heads=1 (shape unknown at reload time),
        # so we use a flat sequence representation: (1, 1, N_flat, head_dim)
        q, aligner = cache._get_quantizer_and_aligner(i, _infer_head_dim(kc, cfg))
        k_dec = aligner.unpad(q.decode(kc.long())).unsqueeze(0).unsqueeze(0)
        v_dec = aligner.unpad(q.decode(vc.long())).unsqueeze(0).unsqueeze(0)
        # Update parent layers (flat sequence view)
        from transformers import DynamicCache
        DynamicCache.update(cache, k_dec, v_dec, layer_idx=i)

    return cache


def _infer_head_dim(codes: torch.Tensor, cfg: PrismKVConfig) -> int:
    """
    Infer original head_dim from code shape.

    codes shape: (N, m) where m = padded_dim / 3
    head_dim ≥ m * 3 (padded) → try multiples of 3 near m*3
    """
    m = codes.shape[-1]
    padded_dim = m * 3
    # Padded dim is the smallest multiple of 3 ≥ original head_dim.
    # Common head dims: 64→66, 128→129, 256→258, 96→96, 192→192
    for candidate in (padded_dim, padded_dim - 1, padded_dim - 2):
        if candidate > 0:
            return candidate
    return padded_dim
