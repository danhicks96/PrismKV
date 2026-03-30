# Changelog

All notable changes are documented here. Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.8.0] — 2026-03-30

### Added
- **M9 Polar-space attention approximation** — novel prior-art contribution:
  attention scores computed directly from PrismKV codes without full dequantization
- `polar_dot_product(q, k_z, k_r, k_theta)` — exact polar identity:
  `<q, k> = Σ q_z·k_z + r_q·r_k·cos(θ_q - θ_k)` per triplet group
- `polar_dot_product_from_codes(q, k_codes, ..., R=)` — unpacks int16 codes,
  dequantizes in-place, applies rotation, returns (b, nh, sq, sk) score matrix
- `PolarAttentionApprox` — drop-in scaled-dot-product approximation; accepts R
  for automatic query rotation; supports causal and additive masks
- `measure_polar_approx_error` — diagnostic: mean/max abs error + cosine similarity
  vs exact Cartesian dot product
- 15 new tests in `test_polar_attention.py`

---

## [0.7.0] — 2026-03-30

### Added
- **M8 Multi-model support**: `ModelArchRegistry` — extensible architecture registry
  with built-in support for GPT-2, OPT, LLaMA/Mistral/Gemma/CodeLlama, Falcon, Qwen2, Phi
- `ArchDescriptor` — per-arch hook mode, module locator, KV split function, and aliases list
- **GQA support**: `get_n_kv_heads()` reads `num_key_value_heads` from config for
  Grouped Query Attention models (LLaMA-2-70B, Mistral-7B, …)
- `KVCollector` now uses `ModelArchRegistry.detect()` for automatic hook routing;
  new `kv_separate` mode hooks `k_proj`/`v_proj` sub-modules directly for LLaMA family
- `ModelArchRegistry.register()` — runtime registration of custom architectures
- 18 new tests in `test_model_arch.py`

---

## [0.6.0] — 2026-03-30

### Added
- **M7 Adaptive Bit Allocation**: `BitAllocator` — water-filling allocation from
  per-head attention entropy; `sensitivity = 1 / H(l, h)` drives more bits to
  sharp (low-entropy) heads
- `prismkv.eval.attention_entropy` — `attention_entropy_from_weights()`,
  `collect_attention_entropy()` utilities
- `PrismKVCache` now accepts `configs: List[PrismKVConfig]` for per-layer bit budgets
- `scripts/build_bit_alloc.py` — CLI to compute and save bit allocations from
  a calibration run
- `examples/adaptive_demo.py` — GPT-2 style adaptive allocation demo
- 18 new tests in `test_bit_alloc.py`
- Post-rounding greedy correction in `BitAllocator` ensures `mean_bits_per_dim`
  matches target within `1/(6n)` after discrete config rounding

---

## [0.5.0] — 2026-03-30

### Added
- **M6 RAG Framework**: full Retrieval-Augmented Generation pipeline
- `prismkv.rag.VectorStore` — SQLite-backed cosine-similarity store (pure torch)
- `prismkv.rag.GraphIndex` — NetworkX DiGraph with SQLite-persisted adjacency;
  BFS expansion at retrieval depth=2
- `prismkv.rag.RAGEngine` — public API: `ingest()`, `retrieve()`, `generate()`
- `TextAdapter`, `DictAdapter`, `FileAdapter` — ingestion adapters; `DictAdapter`
  converts game-state dicts to natural-language sentences per field
- `ContextAssembler` — token-budget-aware context window builder
- SHA-256 content deduplication; `timestamp` ordering for narrative coherence
- `examples/rag_demo.py`, `examples/usurper_rag_demo.py` — CPU-only demos
  (no model downloads; swap in ollama for real use)
- 50+ new tests across `test_rag_adapters.py`, `test_rag_ingestion.py`,
  `test_rag_engine.py`

---

## [0.4.0] — 2026-03-30

### Added
- **CI/CD**: GitHub Actions `ci.yml` — two-job matrix (core tests; eval+cache tests)
- **PyPI publishing**: `publish.yml` — OIDC trusted publishing on `v*.*.*` tags
- **PEP 561**: `py.typed` marker for mypy compatibility
- `CHANGELOG.md`, `CONTRIBUTING.md`

---

## [0.3.0] — 2026-03-30

### Added
- **M4 Bias Correction**: `BiasTable` + `calibrate_bias()` in `prismkv.quantizer.bias_correction`
- `StackedPlaneQuantizer.calibrate_bias()` — stores per-z-bin mean error table
- `decode()` now applies bias correction when `_bias` is set
- 11 new tests in `test_bias_correction.py`

---

## [0.2.0+v2.2] — 2026-03-30

### Added
- **M3 KVCacheWrapper**: `PrismKVCache(DynamicCache)` drop-in compressed cache
- `PrismKVConfig` dataclass — per-layer or global quantization configuration
- `DimAligner` — pads head_dim to nearest multiple of 3 (GPT-2 64→66)
- `examples/hf_integration.py` — one-liner GPT-2 generation demo with memory report
- `prismkv.cache` module with `[cache]` optional dep group in pyproject.toml
- 18 new tests in `test_kv_cache.py`

---

## [0.2.0] — 2026-03-30

### Added
- **M1 Learned Codebooks**: `LearnedSliceCodebook` — per-z-bin k-means codebooks
  trained in Cartesian (x,y) space via pure-torch Lloyd's algorithm
- `StackedPlaneQuantizer.load_codebooks()` — dispatch to learned centroids
- `scripts/build_codebooks.py` — CLI codebook training (synthetic + file sources)
- **M2 KV Benchmarking**: `prismkv.eval` — `KVCollector`, `run_benchmark`
- `scripts/collect_kv_calibration.py` — KV extraction using Moby Dick ch.1 corpus
- `KVCollector` compatible with transformers 5.x (hooks c_attn QKV projection)
- `numpy` promoted from optional to required dependency
- 28 new tests (14 M1 + 14 M2)

---

## [0.1.0] — 2026-03-30

### Added
- Initial defensive prior-art publication
- `StackedPlaneQuantizer` — 3-D stacked-plane KV cache quantizer
- `PolarQuantizer2D` — 2-D polar TurboQuant-style baseline
- `make_rotation()` — seeded random orthogonal rotation matrix
- `examples/demo.py` — CPU demo comparing 2-D vs 3-D at equal bits/dim
- `design.md` — full architecture and math specification
- 11 unit tests, Apache-2.0 license
