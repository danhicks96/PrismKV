"""
bit_alloc.py — Water-filling adaptive bit allocation across attention heads.

Per-head sensitivity = 1 / H(l, h) where H is attention entropy.
Low-entropy (sharp) heads get more bits; high-entropy (diffuse) heads get fewer.

The allocation is:
    sensitivity[l, h] = 1 / H[l, h]
    normalized = sensitivity / mean(sensitivity)
    bits_float[l, h] = target_avg * normalized   (rescaled to preserve mean)

Each float value is rounded to the nearest valid (bits_z, bits_r, bits_theta)
configuration from a pre-enumerated grid, subject to the constraint that the
mean total bits/dim matches the target within a tolerance.

Author: Dan Hicks (github.com/danhicks96)
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Tuple

import torch

from prismkv.cache.cache_config import PrismKVConfig


# ---------------------------------------------------------------------------
# Valid configuration grid
# ---------------------------------------------------------------------------

def enumerate_valid_configs(
    min_bits: int = 2,
    max_bits: int = 6,
) -> List[Tuple[int, int, int]]:
    """
    Enumerate all (bits_z, bits_r, bits_theta) triples in [min_bits, max_bits]^3.

    Returns a list of tuples sorted by total bits.
    """
    configs = list(itertools.product(
        range(min_bits, max_bits + 1),
        range(min_bits, max_bits + 1),
        range(min_bits, max_bits + 1),
    ))
    return sorted(configs, key=lambda c: sum(c))


_DEFAULT_GRID = enumerate_valid_configs(min_bits=2, max_bits=6)


def nearest_config(
    target_bits_per_dim: float,
    grid: Optional[List[Tuple[int, int, int]]] = None,
) -> Tuple[int, int, int]:
    """
    Find the (bits_z, bits_r, bits_theta) configuration whose bits/dim is
    closest to target_bits_per_dim.
    """
    g = grid or _DEFAULT_GRID
    return min(g, key=lambda c: abs(sum(c) / 3 - target_bits_per_dim))


# ---------------------------------------------------------------------------
# BitAllocator
# ---------------------------------------------------------------------------

class BitAllocator:
    """
    Compute per-layer, per-head bit allocations from attention entropy.

    Parameters
    ----------
    entropy  : Tensor shape (n_layers, n_heads)
    target_avg_bits_per_dim : float — target mean bits/dim across all heads
    alpha    : float — scaling factor for sensitivity spread (default 1.0)
               Higher alpha → more aggressive redistribution
    min_bits, max_bits : valid bit range per component
    eps      : entropy floor to avoid division by zero for very sharp heads
    """

    def __init__(
        self,
        entropy: torch.Tensor,
        target_avg_bits_per_dim: float = 4.0,
        alpha: float = 1.0,
        min_bits: int = 2,
        max_bits: int = 6,
        eps: float = 1e-3,
    ) -> None:
        self.entropy = entropy                   # (n_layers, n_heads)
        self.target = target_avg_bits_per_dim
        self.alpha = alpha
        self.grid = enumerate_valid_configs(min_bits, max_bits)
        self.eps = eps

        n_layers, n_heads = entropy.shape
        self.n_layers = n_layers
        self.n_heads = n_heads

        self._alloc: Optional[torch.Tensor] = None  # (n_layers, n_heads) float
        self._configs: Optional[List[List[Tuple[int, int, int]]]] = None

    def compute(self) -> "BitAllocator":
        """
        Run the water-filling computation.  Results stored in self.allocations
        and self.configs.

        Returns self for chaining.
        """
        # Sensitivity: inverse entropy (high sensitivity = more bits)
        sensitivity = 1.0 / (self.entropy.clamp(min=self.eps))      # (L, H)
        mean_sens = sensitivity.mean()
        normalized = sensitivity / mean_sens                          # mean ≈ 1.0

        # Float bits/dim: target * (1 + alpha * (normalized - 1))
        # This preserves mean = target while scaling by sensitivity
        alloc_float = self.target * (1 + self.alpha * (normalized - 1.0))
        alloc_float = alloc_float.clamp(
            min=self.grid[0][0] / 3,   # min bits/dim from grid
            max=self.grid[-1][0],      # max bits/dim from grid
        )

        # Rescale so mean exactly equals target
        alloc_float = alloc_float * (self.target / alloc_float.mean())

        self._alloc = alloc_float

        # Round each head to nearest valid config
        configs = []
        for l in range(self.n_layers):
            layer_configs = []
            for h in range(self.n_heads):
                cfg_tuple = nearest_config(alloc_float[l, h].item(), self.grid)
                layer_configs.append(cfg_tuple)
            configs.append(layer_configs)
        self._configs = configs

        # Post-rounding correction: greedily adjust to bring mean within tolerance
        self._correct_mean(alloc_float)

        return self

    def _correct_mean(self, alloc_float: torch.Tensor) -> None:
        """
        Greedy post-rounding correction.  After independent per-head rounding,
        the mean can drift from target by up to half a grid step.  This pass
        repeatedly upgrades/downgrades the head whose current allocation deviates
        most from alloc_float until the residual is below one half-step.
        """
        n = self.n_layers * self.n_heads

        # Group grid entries by total bits (bz + br + bt)
        grid_by_total: dict = {}
        for cfg in self.grid:
            t = sum(cfg)
            grid_by_total.setdefault(t, []).append(cfg)
        sorted_totals = sorted(grid_by_total)
        min_t, max_t = sorted_totals[0], sorted_totals[-1]

        half_step = 1.0 / (6.0 * n)  # half the minimum mean-change per swap

        for _ in range(n * 6):
            actual_mean = (
                sum(sum(self._configs[l][h]) / 3
                    for l in range(self.n_layers)
                    for h in range(self.n_heads))
                / n
            )
            err = actual_mean - self.target
            if abs(err) <= half_step:
                break

            if err > 0:
                # Downgrade the head whose bits/dim most exceeds its float target
                best_l, best_h = max(
                    ((l, h)
                     for l in range(self.n_layers)
                     for h in range(self.n_heads)
                     if sum(self._configs[l][h]) > min_t),
                    key=lambda lh: (
                        sum(self._configs[lh[0]][lh[1]]) / 3
                        - alloc_float[lh[0], lh[1]].item()
                    ),
                )
                cur_t = sum(self._configs[best_l][best_h])
                new_t = max(t for t in sorted_totals if t < cur_t)
                self._configs[best_l][best_h] = min(
                    grid_by_total[new_t],
                    key=lambda c: abs(sum(c) / 3 - alloc_float[best_l, best_h].item()),
                )
            else:
                # Upgrade the head whose bits/dim most falls short of its float target
                best_l, best_h = max(
                    ((l, h)
                     for l in range(self.n_layers)
                     for h in range(self.n_heads)
                     if sum(self._configs[l][h]) < max_t),
                    key=lambda lh: (
                        alloc_float[lh[0], lh[1]].item()
                        - sum(self._configs[lh[0]][lh[1]]) / 3
                    ),
                )
                cur_t = sum(self._configs[best_l][best_h])
                new_t = min(t for t in sorted_totals if t > cur_t)
                self._configs[best_l][best_h] = min(
                    grid_by_total[new_t],
                    key=lambda c: abs(sum(c) / 3 - alloc_float[best_l, best_h].item()),
                )

    @property
    def mean_bits_per_dim(self) -> float:
        """Actual mean bits/dim after rounding to valid configs."""
        if self._configs is None:
            raise RuntimeError("Call compute() first.")
        total = sum(sum(c) / 3 for row in self._configs for c in row)
        return total / (self.n_layers * self.n_heads)

    @property
    def allocations(self) -> torch.Tensor:
        """Float bits/dim allocation before rounding. Shape (n_layers, n_heads)."""
        if self._alloc is None:
            raise RuntimeError("Call compute() first.")
        return self._alloc

    def to_prism_configs(self, per_head: bool = False) -> List[PrismKVConfig]:
        """
        Convert rounded allocations to a list of PrismKVConfig objects.

        Parameters
        ----------
        per_head : if False (default), average across heads per layer and
                   return one PrismKVConfig per layer.  If True, return one
                   per (layer, head) in row-major order.

        Returns
        -------
        List of PrismKVConfig, length = n_layers (per_head=False) or
        n_layers * n_heads (per_head=True).
        """
        if self._configs is None:
            raise RuntimeError("Call compute() first.")

        results = []
        for l in range(self.n_layers):
            if per_head:
                for h in range(self.n_heads):
                    bz, br, bt = self._configs[l][h]
                    results.append(PrismKVConfig(bits_z=bz, bits_r=br, bits_theta=bt))
            else:
                # Average bits per component across heads for this layer
                total_bz = sum(c[0] for c in self._configs[l])
                total_br = sum(c[1] for c in self._configs[l])
                total_bt = sum(c[2] for c in self._configs[l])
                n = self.n_heads
                bz = round(total_bz / n)
                br = round(total_br / n)
                bt = round(total_bt / n)
                results.append(PrismKVConfig(bits_z=bz, bits_r=br, bits_theta=bt))

        return results

    def summary(self) -> str:
        """Human-readable allocation summary."""
        if self._configs is None:
            return "BitAllocator(not computed)"
        lines = [
            f"BitAllocator(target={self.target:.2f} bits/dim, "
            f"actual={self.mean_bits_per_dim:.3f} bits/dim, "
            f"n_layers={self.n_layers}, n_heads={self.n_heads})"
        ]
        for l in range(self.n_layers):
            bpd = [sum(c) / 3 for c in self._configs[l]]
            lines.append(
                f"  layer {l:2d}: "
                f"min={min(bpd):.2f} max={max(bpd):.2f} mean={sum(bpd)/len(bpd):.2f} bits/dim"
            )
        return "\n".join(lines)
