# Contributing to PrismKV

Thank you for your interest in contributing!

## Quick Start

```bash
git clone https://github.com/danhicks96/PrismKV
cd PrismKV
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest tests/test_quantizer.py tests/test_learned_codebooks.py tests/test_bias_correction.py -v
```

## Running All Tests

```bash
# Core tests (no transformers required, <2 min)
pip install -e ".[dev]"
pytest tests/test_quantizer.py tests/test_learned_codebooks.py \
       tests/test_bias_correction.py tests/test_bit_alloc.py \
       tests/test_polar_attention.py tests/test_e2e_benchmark.py -v

# Full suite (requires transformers + networkx, ~5 min)
pip install -e ".[dev,eval,cache,rag]"
pytest tests/ -v
```

198 tests total across all modules.

## Contribution Guidelines

1. **All existing tests must stay green** — run the full suite before opening a PR.
2. **New features need tests.** Aim for ≥85% coverage on `src/prismkv/quantizer/`.
3. **`dim % 3 == 0` everywhere.** Use `DimAligner` for models where head_dim is not divisible by 3 (e.g. GPT-2 d=64 → padded to 66).
4. **No new hard dependencies** without discussion. The core quantizer (`torch`, `numpy`) must stay lean.
5. **Prior-art scope.** Novel ideas should be documented in `design.md` and committed promptly.
6. **Feature branches + PRs.** `main` is protected — no direct pushes. Open a PR; CI must pass before merging.

## Project Layout

```
src/prismkv/
├── quantizer/    — StackedPlaneQuantizer, PolarQuantizer2D, codebooks, bias, bit_alloc
├── cache/        — PrismKVCache (HuggingFace DynamicCache subclass), cache_store
├── eval/         — KVCollector, benchmark, ModelArchRegistry, e2e_benchmark
└── rag/          — RAG framework: RAGEngine, VectorStore, GraphIndex, adapters
```

## PyPI Publishing

Releases are published to PyPI automatically via GitHub Actions OIDC trusted publishing
when a `v*.*.*` tag is pushed to `main`.

**One-time setup** (repo owner only):
1. Register `prismkv` on PyPI (create the project).
2. Go to PyPI → Account settings → Publishing → Add a new publisher:
   - Owner: `danhicks96`
   - Repository: `PrismKV`
   - Workflow: `publish.yml`
   - Environment: (leave blank)
3. Push a `v*.*.*` tag — `publish.yml` will build and upload automatically.

## Reporting Issues

Open an issue at https://github.com/danhicks96/PrismKV/issues.
