# PrismKV: 3-D Stacked-Plane KV Cache Quantization

[![CI](https://github.com/danhicks96/PrismKV/actions/workflows/ci.yml/badge.svg)](https://github.com/danhicks96/PrismKV/actions/workflows/ci.yml)

**First published:** 2026-03-30
**Author:** Dan Hicks · [github.com/danhicks96](https://github.com/danhicks96)
**License:** Apache-2.0
**Version:** 1.1.0
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

## Benchmark Results (Real GPT-2 Data)

Measured on 73,728 KV vectors from GPT-2 (12 layers × 12 heads × 512 tokens, Moby Dick Ch. 1).
Head dim 64 padded to 66. Codebooks trained in-distribution (all 12 layers). CPU-only.

| Scheme                      | bits/dim | RMSE   | CosSim | Mem/4K |
|-----------------------------|----------|--------|--------|--------|
| FP32 (no compression)       | 32.0     | 0      | 1.000  | 24 MB  |
| FP16 (no compression)       | 16.0     | ~0     | ≈1.000 | 12 MB  |
| 2D Polar (uniform)          | 4.0      | 0.336  | 0.988  | 3.0 MB |
| 3D Stacked-Plane (uniform)  | 4.0      | 0.778  | 0.883  | 3.0 MB |
| **3D Stacked-Plane (learned)** | **4.0** | **0.612** | **0.926** | **3.0 MB** |
| 3D Stacked-Plane 3+3+2      | 2.67     | 1.895  | 0.575  | 2.0 MB |

**Learned codebook improvement over 3D uniform: 21.4% lower RMSE** (validates the ≥5% target).

**Memory savings vs FP16: 4× at 4 bits/dim.** The 3+3+2 config reaches 2.67 bits/dim (6× vs FP16) — a
ratio unreachable by integer-bit 2D polar, which requires even bits/dim (2.0, 4.0, 6.0, …).

Full results and methodology: [`results/benchmark_gpt2.json`](results/benchmark_gpt2.json) and
[`results/benchmark_gpt2_notes.md`](results/benchmark_gpt2_notes.md).

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
# Core tests — no model downloads, <2 min
pip install -e ".[dev]"
pytest tests/test_quantizer.py tests/test_learned_codebooks.py \
       tests/test_bias_correction.py tests/test_bit_alloc.py \
       tests/test_polar_attention.py tests/test_e2e_benchmark.py \
       tests/test_m12_framework_agnostic.py -v

# Full suite — requires transformers + networkx, ~5 min
pip install -e ".[dev,eval,cache,rag]"
pytest tests/ -v
```

234 tests across all modules (126 core, 108 eval+cache+RAG).

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
│   │   ├── attention_entropy.py  — per-head Shannon entropy (M7)
│   │   ├── model_arch.py         — ModelArchRegistry + GQA support (M8)
│   │   └── e2e_benchmark.py      — memory table + quality report (M11)
│   ├── cache/
│   │   ├── backend.py            — CacheBackend protocol + PrismKVBackend (M12)
│   │   ├── raw_cache.py          — RawKVCache: framework-agnostic (M12)
│   │   ├── vllm_adapter.py       — VLLMSwapCompressor (M12)
│   │   ├── kv_cache.py           — PrismKVCache(DynamicCache) drop-in (M3)
│   │   ├── cache_config.py       — PrismKVConfig dataclass
│   │   ├── dim_aligner.py        — pad head_dim to multiple of 3
│   │   └── cache_store.py        — save_cache / load_cache NPZ (M10)
│   └── rag/
│       ├── rag_engine.py         — RAGEngine public API (M6)
│       ├── vector_store.py       — SQLite + pure-torch cosine store
│       ├── graph_index.py        — NetworkX DiGraph + BFS expansion
│       ├── ingestion.py          — IngestionEngine with deduplication
│       ├── retriever.py          — hybrid vector + graph retrieval
│       ├── context_assembler.py  — token-budget-aware context builder
│       ├── adapters.py           — TextAdapter, DictAdapter, FileAdapter, APIAdapter
│       └── schema.py             — Chunk, Node, RetrievalResult
├── src/prismkv/sidecar.py            — HTTP compression service (M12)
├── tests/                        — 234 tests across all modules
├── examples/
│   ├── demo.py                   — 2-D vs 3-D quantizer comparison
│   ├── hf_integration.py         — GPT-2 with PrismKVCache
│   ├── rag_demo.py               — CPU-only RAG pipeline demo
│   ├── usurper_rag_demo.py       — 50-dict game-state ingestion
│   └── adaptive_demo.py          — BitAllocator → PrismKVCache
├── scripts/
│   ├── build_codebooks.py        — CLI: train learned codebooks
│   ├── collect_kv_calibration.py — extract KV tensors from GPT-2
│   ├── run_e2e_benchmark.py      — CLI: memory + quality benchmark (M11)
│   └── run_sidecar.py            — CLI: start HTTP sidecar (M12)
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
| M8 | 0.7.0 | Multi-model support — `ModelArchRegistry`, GQA-aware `KVCollector` |
| M9 | 0.8.0 | Polar-space attention approximation — novel prior-art contribution |
| M10 | 0.9.0 | Cache persistence (`save_cache`/`load_cache`) + `APIAdapter` |
| M11 | 1.0.0 | End-to-end benchmark — memory table + quality comparison |
| M12 | 1.1.0 | Framework-agnostic layer — `RawKVCache`, vLLM adapter, HTTP sidecar |

### Future work
- **CUDA kernel** — the fused dequantize + polar attention kernel is written and compilable (`src/prismkv/cuda/polar_attn_kernel.cu`, requires CUDA >= 11.8); runtime integration into the attention path is a future step. Build with `python setup_cuda.py build_ext --inplace`.

## RAG Framework (M6)

PrismKV ships a complete RAG pipeline that uses the compressed KV cache internally:

```python
from prismkv.rag import RAGEngine
from prismkv.rag.adapters import DictAdapter

engine = RAGEngine(db_path=":memory:", embedder=my_embed_fn)

# Ingest — any adapter: text file, dict list, plain string, REST endpoint
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

## Multi-Model Support (M8)

Auto-detect transformer architecture and collect real KV vectors from any supported model:

```python
from prismkv.eval.model_arch import ModelArchRegistry
from prismkv.eval.kv_collector import KVCollector

# Supports GPT-2, OPT, LLaMA/Mistral/Gemma/CodeLlama, Falcon, Qwen2, Phi
arch = ModelArchRegistry.detect(model)

collector = KVCollector(model, device="cpu")
kv_data = collector.collect(input_ids, layer_idx=0)  # {0: {"keys": ..., "values": ...}}
```

GQA-aware: reads `num_key_value_heads` from config for LLaMA-2-70B, Mistral-7B, etc.

## Polar-Space Attention Approximation (M9)

Approximate attention scores directly from compressed PrismKV codes — no full dequantization:

```python
from prismkv.quantizer.polar_attention import PolarAttentionApprox, measure_polar_approx_error

# Drop-in scaled dot-product approximation
approx = PolarAttentionApprox(
    bits_z=4, bits_r=4, bits_theta=4,
    z_min=qtz.z_min, z_max=qtz.z_max, r_max=qtz.r_max,
    scale=1/math.sqrt(head_dim), R=qtz.R,
)
output, weights = approx.forward(q, k_codes, v)  # (b, nh, sq, d), (b, nh, sq, sk)

# Measure approximation error vs exact Cartesian dot product
err = measure_polar_approx_error(q, k, k_codes, ..., R=qtz.R)
# {'mean_abs_error': ..., 'max_abs_error': ..., 'cosine_sim': ...}
```

The identity `<q, k> = Σ_j q_z·k_z + r_q·r_k·cos(θ_q − θ_k)` per triplet group enables
computing attention scores from codes without materialising full FP16 key tensors.

## Cache Persistence (M10)

Save and load compressed KV caches to disk:

```python
from prismkv.cache.cache_store import save_cache, load_cache

# Serialize compressed codes + config to NPZ
save_cache(cache, "checkpoint.npz")

# Reconstruct — returns PrismKVCache with fully seeded DynamicCache layers
cache = load_cache("checkpoint.npz", device="cpu")
```

REST API ingestion for the RAG engine:

```python
from prismkv.rag.adapters import APIAdapter

engine.ingest(APIAdapter(
    "https://api.example.com/articles",
    text_field="body",
    source_id="api_articles",
))
```

## End-to-End Benchmark (M11)

Memory footprint and reconstruction quality comparison — no model download required:

```python
from prismkv.eval.e2e_benchmark import run_e2e_benchmark, print_e2e_table

report = run_e2e_benchmark(head_dim=64, n_heads=12, n_layers=12)
print_e2e_table(report)
```

```
KV Cache Memory Footprint  (12L × 12H × d=64)
Context     FP16       3bit       4bit       5bit
  1,024    18.0MB   3.4MB(5.3×)  4.5MB(4.0×)  5.6MB(3.2×)
  4,096    72.0MB  13.5MB(5.3×) 18.0MB(4.0×) 22.5MB(3.2×)
 16,384   288.0MB  54.0MB(5.3×) 72.0MB(4.0×) 90.0MB(3.2×)
```

For pseudo-perplexity measurement (requires GPT-2 download):

```bash
python scripts/run_e2e_benchmark.py --pseudo-ppl
```

## Framework-Agnostic Integration (M12)

PrismKV v1.1.0 works with any inference engine — not just HuggingFace.

### Custom autoregressive loop (no framework)

```python
from prismkv.cache import PrismKVBackend, RawKVCache, PrismKVConfig

backend = PrismKVBackend(PrismKVConfig(), head_dim=64)
cache = RawKVCache(backend)

# Inside your generation loop:
for step in range(max_new_tokens):
    k_new, v_new = my_model_attention(x)           # (..., seq_len, head_dim)
    k_ctx, v_ctx = cache.update(layer_idx, k_new, v_new)  # full context
    attn_out = scaled_dot_product_attention(q, k_ctx, v_ctx)

print(cache.memory_footprint())  # {'compression': 3.2, 'codes_bytes': ...}
```

Per-layer bit budgets:

```python
from prismkv.cache import PrismKVBackend, RawKVCache, PrismKVConfig

backends = {
    i: PrismKVBackend(PrismKVConfig(bits_z=3, bits_r=3, bits_theta=3), head_dim=64)
    if i < 6
    else PrismKVBackend(PrismKVConfig(), head_dim=64)
    for i in range(12)
}
cache = RawKVCache(backends)  # 3 bits for layers 0-5, 4 bits for 6-11
```

### vLLM — compress at CPU swap boundary

```python
from prismkv.cache import VLLMSwapCompressor, PrismKVConfig

compressor = VLLMSwapCompressor(
    config=PrismKVConfig(bits_z=4, bits_r=4, bits_theta=4),
    head_dim=128,
    n_layers=32,
)
compressor.attach(engine)   # patches engine.cache_engine swap_out/swap_in
```

KV blocks evicted to CPU are compressed 3–5× before leaving GPU. Active GPU blocks and attention kernels are untouched.

### HTTP sidecar — any language, any engine

```bash
# Start the sidecar (stdlib only, no extra deps):
python -m prismkv.sidecar --port 8765
```

```python
import requests, numpy as np

k = np.random.randn(10, 64).astype(np.float32)
v = np.random.randn(10, 64).astype(np.float32)

# Compress
codes = requests.post("http://localhost:8765/compress",
    json={"k": k.tolist(), "v": v.tolist()}).json()

# Decompress
result = requests.post("http://localhost:8765/decompress",
    json={"k_codes": codes["k_codes"], "v_codes": codes["v_codes"],
          "head_dim": 64}).json()
k_hat = np.array(result["k"])
```

The sidecar is the integration path for engines that manage their KV cache in C++ (llama.cpp, Ollama) — intercept tensors at the application layer before they reach the engine.

| Engine | Integration |
|---|---|
| HuggingFace | `PrismKVCache(DynamicCache)` — drop-in, unchanged |
| Custom PyTorch | `RawKVCache(PrismKVBackend(...))` — no framework needed |
| vLLM | `VLLMSwapCompressor.attach(engine)` |
| llama.cpp / Ollama | HTTP sidecar via `python -m prismkv.sidecar` |
| Any language | HTTP POST to sidecar |

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
