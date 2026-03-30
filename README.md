# PrismKV: 3-D Stacked-Plane KV Cache Quantization

**First published:** 2026-03-30
**Author:** Dan Hicks · [github.com/danhicks96](https://github.com/danhicks96)
**License:** Apache-2.0
**Status:** Defensive prior-art publication. All ideas herein are released under Apache-2.0.

---

## The Idea

Large language models cache key (`K`) and value (`V`) tensors for every previously seen token — the "KV cache." At long context lengths this cache dominates GPU memory. Recent work (Google's TurboQuant, ICLR 2026) showed that quantizing KV vectors to 3–4 bits using 2-D polar coordinates after a random rotation achieves near-lossless compression at 6× memory reduction.

**PrismKV extends this to 3-D.**

TurboQuant groups each `d`-dimensional KV vector into `d/2` independent pairs `(x, y)` and quantizes each pair in polar form `(r, θ)`. Each pair is quantized *without context from its neighbors*. This is optimal for isotropic Gaussian data but misses cross-dimensional correlations that real KV distributions exhibit.

PrismKV introduces a **conditional stacked-plane structure**:

- Group dimensions into triplets `(z, x, y)` instead of pairs
- Coarsely quantize the `z` coordinate into `B_z` bins → index `i_z`
- Use `i_z` to *condition* the 2-D polar quantization of `(x, y)` — selecting a per-z-slice codebook
- This creates a **3-D quantization cell**: a wedge of polar space at a specific `z` level

The result is a hierarchical encoding that captures relationships between the three coordinates. At the same bits-per-dimension budget (e.g., `B_z=4, B_r=4, B_θ=4` → 4.0 bits/dim), the conditional structure allows per-slice codebook adaptation that flat 2-D schemes cannot express.

---

## The Math

### Notation

```
v ∈ R^d         — a rotated KV vector (after global rotation R)
d = 3 * m       — dim must be divisible by 3; m = number of triplet groups
B_z, B_r, B_θ  — bits allocated to z, radius, and angle
C_z = 2^B_z     — number of z-bins
C_r = 2^B_r     — number of radius bins
C_θ = 2^B_θ     — number of angle bins
```

### Step 0 — Global Rotation (same as TurboQuant)

```
v_rot = R @ v
```

`R` is a `(d, d)` random orthogonal matrix (QR decomposition of a seeded Gaussian draw). This spreads energy uniformly across dimensions, making coordinates approximately independent — a prerequisite for efficient scalar quantization.

### Step 1 — Triplet Extraction

After rotation, index the `d` dimensions as:

```
z-dim for group k:  index 3k       (k = 0, 1, ..., m-1)
x-dim for group k:  index 3k + 1
y-dim for group k:  index 3k + 2
```

No dimension is shared between groups (no overlapping). Each group `k` gives a triplet `(z_k, x_k, y_k)`.

### Step 2 — Coarse z Quantization

```
Δ_z = (z_max - z_min) / C_z
i_z = floor((z - z_min) / Δ_z)  ∈ {0, ..., C_z - 1}
```

`z_min`, `z_max` are set conservatively to `±sqrt(d)` (or tightened via `calibrate()`).

### Step 3 — Conditional 2-D Polar Quantization

Convert `(x, y)` to polar form:

```
r     = sqrt(x^2 + y^2)
θ     = atan2(y, x)          ∈ (-π, π]
```

Quantize uniformly (v1 uses the same table for all z-slices; per-slice learned tables are v2):

```
i_r     = round(r / r_max * (C_r - 1))          ∈ {0, ..., C_r - 1}
i_θ     = round((θ + π) / (2π) * (C_θ - 1))    ∈ {0, ..., C_θ - 1}
```

### Step 4 — Packing

```
code = (i_z << (B_r + B_θ)) | (i_r << B_θ) | i_θ
```

Total bits per triplet: `B_z + B_r + B_θ`.
Bits per dimension: `(B_z + B_r + B_θ) / 3`.

### Dequantization

```
z_q   = z_min + (i_z + 0.5) * Δ_z          ← bin-center (unbiased)
r_q   = i_r / (C_r - 1) * r_max
θ_q   = i_θ / (C_θ - 1) * 2π - π

x_q   = r_q * cos(θ_q)
y_q   = r_q * sin(θ_q)

v_hat = R^T @ reassembled(z_q, x_q, y_q)
```

### Error Bound

Worst-case per-triplet Euclidean reconstruction error (design doc §3.5):

```
‖(z, x, y) - (z_q, x_q, y_q)‖
  ≤ sqrt( (Δ_r/2)^2 + (r_max · Δ_θ/2)^2 + (Δ_z/2)^2 )
```

where `Δ_r = r_max / (C_r - 1)` and `Δ_θ = 2π / (C_θ - 1)`.

---

## Bit Budget

| Scheme                        | Bits per KV vector | Bits/dim | vs FP32 |
|-------------------------------|--------------------|----------|---------|
| FP32 (no compression)         | 32d                | 32.0     | 1×      |
| FP16 (no compression)         | 16d                | 16.0     | 2×      |
| 2-D polar, 4+4 bits           | 8 × (d/2) = 4d    | 4.0      | 8×      |
| **3-D stacked-plane, 4+4+4**  | **12 × (d/3) = 4d** | **4.0** | **8×** |
| 3-D stacked-plane, 3+3+2 bits | 8 × (d/3) = 2.67d | 2.67     | 12×     |

The 3-D scheme at `B_z=3, B_r=3, B_θ=2` (2.67 bits/dim) has no 2-D equivalent — you cannot reach 2.67 bits/dim with integer-bit 2-D polar. This is one regime where 3-D strictly enables smaller codebooks.

---

## Comparison to Related Work

| Method | Training required | Conditioning | Unbiased attention scores |
|--------|------------------|--------------|--------------------------|
| TurboQuant (2026) | None | None (independent 2-D pairs) | Yes (QJL) |
| **PrismKV v1** | **None** | **z-conditioned 2-D polar** | Empirically small bias |
| KIVI | Calibration data | None | No |
| SnapKV | Fine-tuning | None | No |
| Product Quantization | Dataset training | None | No |

**What is new in PrismKV:**
1. The triplet partition `(z, x, y)` with no overlapping coordinates
2. Using the coarsely-quantized `z` index to *select* per-slice codebooks for `(x, y)` — a conditional product quantizer in 3-D
3. The architecture for per-z-slice learned codebooks (v2), not possible in any 2-D scheme without a separate full-dimensional index

**What is not claimed:**
- PrismKV v1 does not yet beat TurboQuant on empirical benchmarks. The per-slice tables are uniform (same as baseline). The advantage materializes with learned tables (v2) on real, non-Gaussian KV distributions.
- Full QJL-style attention score unbiasedness is v2 work.

---

## Quick Start

```bash
git clone https://github.com/danhicks96/PrismKV
cd PrismKV
pip install -e .
python3 examples/demo.py
```

Expected output (CPU, <5 seconds):

```
══════════════════════════════════════════════════════════════
  PrismKV  ·  3-D Stacked-Plane KV Cache Quantizer
  ...
  2D Polar (baseline)            4.0  (1024, 96)   ...
  3D Stacked-Plane (PrismKV)     4.0  (1024, 64)   ...
══════════════════════════════════════════════════════════════
```

### Run tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

All 9 tests should pass.

---

## Repository Layout

```
PrismKV/
├── src/
│   └── prismkv/
│       ├── __init__.py                  — package entry point
│       ├── utils.py                     — make_rotation(), calibrate_r_max()
│       └── quantizer/
│           ├── __init__.py
│           ├── baseline_2d.py           — 2-D polar quantizer (TurboQuant-style)
│           └── stacked_plane.py         — 3-D conditional quantizer (PrismKV)
├── tests/
│   └── test_quantizer.py                — 9 unit tests
├── examples/
│   └── demo.py                          — runnable comparison demo
├── design.md                            — full architecture & math specification
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Roadmap

### v2 (planned)
- **Learned per-z-slice codebooks** — k-means on real KV distributions per z-bin; this is where the theoretical advantage of conditioning is realized
- **QJL-style bias correction** — unbiased attention score estimation for the z-conditioned scheme
- **KVCacheWrapper** — drop-in replacement for raw PyTorch KV tensor caches in HuggingFace-compatible models
- **CUDA kernel** — on-the-fly dequantization fused with attention computation

### v3 (planned)
- **RAG framework** — Adapters (text, embedding, API), IngestionEngine (graph-indexed chunks), RAG Engine (retrieval + context assembly + generation hook) using PrismKV for the KV cache layer
- **Adaptive bit allocation** — per-layer or per-head bit budgets based on attention entropy

---

## Citation / Prior Art

This repository was publicly released on **2026-03-30** as a defensive publication. If you build on these ideas, a citation is appreciated but not required under the Apache-2.0 license:

```
@misc{hicks2026prismkv,
  author = {Dan Hicks},
  title  = {PrismKV: 3-D Stacked-Plane KV Cache Quantization},
  year   = {2026},
  url    = {https://github.com/danhicks96/PrismKV}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
