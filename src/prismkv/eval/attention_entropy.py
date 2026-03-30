"""
attention_entropy.py — Per-head attention entropy measurement for adaptive bit allocation.

Attention entropy H(l, h) = -Σ a_ij log a_ij measures how "sharp" each head's
attention distribution is.  Low entropy = sharp attention = high sensitivity =
more bits needed.  High entropy = diffuse attention = tolerant head = fewer bits.

Author: Dan Hicks (github.com/danhicks96)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch


def attention_entropy_from_weights(
    attn_weights: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Compute Shannon entropy of an attention weight matrix.

    Parameters
    ----------
    attn_weights : Tensor shape (..., seq_len, seq_len) — softmax probabilities
    eps          : numerical stability floor

    Returns
    -------
    entropy : Tensor shape (...,) — one entropy value per (batch, head) pair
    """
    p = attn_weights.clamp(min=eps)
    return -(p * p.log()).sum(dim=-1).mean(dim=-1)   # mean over query positions


def collect_attention_entropy(
    model,
    text: str,
    tokenizer,
    device: str = "cpu",
    max_tokens: int = 512,
) -> torch.Tensor:
    """
    Run a model forward pass with output_attentions=True and collect per-head entropy.

    Requires transformers.

    Parameters
    ----------
    model     : HuggingFace CausalLM
    text      : calibration text
    tokenizer : HuggingFace tokenizer
    device    : torch device string
    max_tokens: truncation limit

    Returns
    -------
    entropy : Tensor shape (n_layers, n_heads)
    """
    ids = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=max_tokens)["input_ids"].to(device)

    model.eval()
    with torch.no_grad():
        out = model(ids, output_attentions=True)

    # out.attentions: tuple of (batch, n_heads, seq, seq) tensors, one per layer
    if not hasattr(out, "attentions") or out.attentions is None:
        raise RuntimeError(
            "Model did not return attention weights. "
            "Ensure model.config._attn_implementation == 'eager'."
        )

    entropies = []
    for attn in out.attentions:           # (batch, n_heads, seq, seq)
        h = attention_entropy_from_weights(attn.squeeze(0))  # (n_heads,)
        entropies.append(h)

    return torch.stack(entropies)         # (n_layers, n_heads)
