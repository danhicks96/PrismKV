# Contributing to PrismKV

Thank you for your interest in contributing!

## Quick Start

```bash
git clone https://github.com/danhicks96/PrismKV
cd PrismKV
pip install -e ".[dev]"
pytest tests/test_quantizer.py tests/test_learned_codebooks.py tests/test_bias_correction.py -v
```

## Running All Tests

```bash
pip install -e ".[dev,eval,cache]"
pytest tests/ -v
```

## Contribution Guidelines

1. **All 68+ existing tests must remain green.**
2. **New features need tests.** Aim for ≥85% coverage on `src/prismkv/`.
3. **`dim % 3 == 0` everywhere.** Use `DimAligner` for models where head_dim is not divisible by 3.
4. **No new hard dependencies** without discussion. The core quantizer (`torch`, `numpy`) must stay lean.
5. **Prior-art scope.** This repository is a defensive publication. Novel ideas should be documented in `design.md` and committed promptly.

## Project Layout

```
src/prismkv/
├── quantizer/    — StackedPlaneQuantizer, PolarQuantizer2D, codebooks, bias
├── cache/        — PrismKVCache (HuggingFace DynamicCache subclass)
├── eval/         — KVCollector, benchmark utilities
└── rag/          — RAG framework (v3.0, in progress)
```

## Reporting Issues

Open an issue at https://github.com/danhicks96/PrismKV/issues.
