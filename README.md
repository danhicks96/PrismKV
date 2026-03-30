# PrismKV: 3-D Stacked-Plane KV Cache Quantization

[![CI](https://github.com/danhicks96/PrismKV/actions/workflows/ci.yml/badge.svg)](https://github.com/danhicks96/PrismKV/actions/workflows/ci.yml)

**First published:** 2026-03-30
**Author:** Dan Hicks · [github.com/danhicks96](https://github.com/danhicks96)
**License:** Apache-2.0
**Version:** 0.6.0
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

| Method | Training required | Conditioning | Bias correction | Adaptive bits |
|--------|------------------|--------------|-----------------|---------------|
| TurboQuant (2026) | None | None (independent 2-D pairs) | Yes (QJL) | No |
| **PrismKV v1** | **None** | **z-conditioned 2-D polar** | No | No |
| **PrismKV v2** | **K-means calibration** | **Per-z-bin learned codebooks** | **Yes (BiasTable)** | **Yes (entropy water-filling)** |
| KIVI | Calibration data | None | No | No |
| SnapKV | Fine-tuning | None | No | No |
| Product Quantization | Dataset training | None | No | No |

**What is new in PrismKV:**
1. The triplet partition `(z, x, y)` with no overlapping coordinates
2. Using the coarsely-quantized `z` index to *select* per-slice codebooks for `(x, y)` — a conditional product quantizer in 3-D
3. Per-z-slice learned codebooks trained via pure-torch k-means — not possible in any 2-D scheme without a separate full-dimensional index
4. Per-z-bin bias correction table (QJL-style, no training required beyond calibration)
5. Water-filling adaptive bit allocation from per-head attention entropy

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

All 131 tests pass (36 core, 95 eval+cache+RAG).

For the full suite including RAG and cache tests:

```bash
pip install -e ".[dev,eval,cache,rag]"
pytest tests/ -v
```

---

## Repository Layout

```
PrismKV/
├── src/prismkv/
│   ├── quantizer/
│   │   ├── stacked_plane.py      — 3-D conditional quantizer (core prior art)
│   │   ├── baseline_2d.py        — 2-D polar baseline (TurboQuant-style)
│   │   ├── learned_codebook.py   — per-z-bin k-means codebooks (M1)
│   │   ├── bias_correction.py    — QJL-style per-z-bin bias table (M4)
│   │   └── bit_alloc.py          — water-filling adaptive bit allocation (M7)
│   ├── eval/
│   │   ├── kv_collector.py       — transformers 5.x KV hook collector (M2)
│   │   ├── benchmark.py          — RMSE / cosine / throughput benchmarks (M2)
│   │   └── attention_entropy.py  — per-head Shannon entropy (M7)
│   ├── cache/
│   │   ├── kv_cache.py           — PrismKVCache(DynamicCache) drop-in (M3)
│   │   ├── cache_config.py       — PrismKVConfig dataclass
│   │   └── dim_aligner.py        — pad head_dim to multiple of 3
│   └── rag/
│       ├── rag_engine.py         — RAGEngine public API (M6)
│       ├── vector_store.py       — SQLite + pure-torch cosine store
│       ├── graph_index.py        — NetworkX DiGraph + BFS expansion
│       ├── ingestion.py          — IngestionEngine with deduplication
│       ├── retriever.py          — hybrid vector + graph retrieval
│       ├── context_assembler.py  — token-budget-aware context builder
│       ├── adapters.py           — TextAdapter, DictAdapter, FileAdapter
│       └── schema.py             — Chunk, Node, RetrievalResult
├── tests/                        — 131 tests across all modules
├── examples/
│   ├── demo.py                   — 2-D vs 3-D quantizer comparison
│   ├── hf_integration.py         — GPT-2 with PrismKVCache
│   ├── rag_demo.py               — CPU-only RAG pipeline demo
│   ├── usurper_rag_demo.py       — 50-dict game-state ingestion
│   └── adaptive_demo.py          — BitAllocator → PrismKVCache
├── scripts/
│   ├── build_codebooks.py        — CLI: train learned codebooks
│   └── collect_kv_calibration.py — extract KV tensors from GPT-2
├── design.md                     — full architecture & math specification
└── pyproject.toml
```

---

## What's Shipped

| Milestone | Version | Description |
|-----------|---------|-------------|
| M1 | 0.2.0 | Learned per-z-slice codebooks — k-means on real KV distributions |
| M2 | 0.2.0 | KV benchmarking eval layer — RMSE, cosine sim, throughput |
| M3 | 0.2.0 | `PrismKVCache(DynamicCache)` — drop-in HuggingFace cache replacement |
| M4 | 0.3.0 | QJL-style bias correction — per-z-bin `BiasTable` |
| M5 | 0.4.0 | CI/CD — GitHub Actions + PyPI OIDC trusted publishing |
| M6 | 0.5.0 | RAG framework — vector store, graph index, adapters, RAGEngine |
| M7 | 0.6.0 | Adaptive bit allocation — water-filling from attention entropy |

### Still planned
- **CUDA kernel** — on-the-fly dequantization fused with attention computation
- **ONNX export** — quantized cache for inference engines

## RAG Framework (M6)

PrismKV ships a complete RAG pipeline that uses the compressed KV cache internally:

```python
from prismkv.rag import RAGEngine
from prismkv.rag.adapters import DictAdapter

engine = RAGEngine(db_path=":memory:", embedder=my_embed_fn)

# Ingest — any adapter: text file, dict list, plain string
engine.ingest(DictAdapter(game_states, entity_key="name"))

# Query
results = engine.retrieve("throne room conflict", top_k=5)

# Generate
response = engine.generate("What happened at the throne room?", generation_fn=my_llm)
```

Hybrid retrieval: cosine vector search + NetworkX graph BFS expansion. SHA-256 content deduplication. Token-budget-aware context assembly.

## Adaptive Bit Allocation (M7)

Per-head bit budgets derived from attention entropy — sharp heads (low entropy) get more bits:

```python
from prismkv.quantizer.bit_alloc import BitAllocator
from prismkv.cache import PrismKVCache

allocator = BitAllocator(entropy, target_avg_bits_per_dim=4.0).compute()
configs = allocator.to_prism_configs(per_head=False)  # one PrismKVConfig per layer

cache = PrismKVCache(configs=configs)
```

The allocator uses water-filling (`sensitivity = 1/H(l,h)`) with a greedy post-rounding correction that guarantees the mean bits/dim is within `1/(6n)` of target after discretisation.

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
