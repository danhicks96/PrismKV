# Changelog

All notable changes are documented here. Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
