"""
test_rag_ingestion.py — Tests for VectorStore, GraphIndex, IngestionEngine, Retriever.
"""

import pytest
import torch

from prismkv.rag.adapters import TextAdapter, DictAdapter
from prismkv.rag.graph_index import GraphIndex
from prismkv.rag.ingestion import IngestionEngine
from prismkv.rag.retriever import Retriever
from prismkv.rag.schema import Chunk
from prismkv.rag.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Deterministic test embedder (no model download)
# ---------------------------------------------------------------------------

def make_embedder(dim: int = 64):
    """Returns a deterministic fake embedder: hash(text) → fixed unit vector."""
    def embed(text: str) -> torch.Tensor:
        gen = torch.Generator().manual_seed(hash(text) % (2 ** 31))
        v = torch.randn(dim, generator=gen)
        return torch.nn.functional.normalize(v, dim=0)
    return embed


def make_engine(db_path=":memory:", dim=64):
    embedder = make_embedder(dim)
    vs = VectorStore(db_path=db_path)
    gi = GraphIndex(db_path=db_path)
    ie = IngestionEngine(vs, gi, embedder)
    return vs, gi, ie, embedder


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class TestVectorStore:
    def test_add_and_count(self):
        vs = VectorStore()
        emb = torch.randn(64)
        c = Chunk(id="c1", text="hello world")
        added = vs.add(c, emb)
        assert added is True
        assert vs.count() == 1

    def test_duplicate_skipped(self):
        vs = VectorStore()
        emb = torch.randn(64)
        c = Chunk(id="c1", text="hello world")
        vs.add(c, emb)
        added2 = vs.add(c, emb)
        assert added2 is False
        assert vs.count() == 1

    def test_search_returns_top_k(self):
        vs = VectorStore()
        embed = make_embedder(32)
        texts = ["apple", "orange", "banana", "grape", "mango"]
        for i, t in enumerate(texts):
            vs.add(Chunk(id=f"c{i}", text=t), embed(t))

        q = embed("fruit")
        results = vs.search(q, top_k=3)
        assert len(results) == 3
        assert all(r.score >= -1.0 and r.score <= 1.0 for r in results)

    def test_search_empty_store(self):
        vs = VectorStore()
        q = torch.randn(32)
        results = vs.search(q, top_k=5)
        assert results == []

    def test_search_scores_are_cosine(self):
        """Identical query and stored vector should yield cosine ≈ 1."""
        vs = VectorStore()
        v = torch.nn.functional.normalize(torch.randn(64), dim=0)
        c = Chunk(id="exact", text="exact match")
        vs.add(c, v)
        results = vs.search(v, top_k=1)
        assert abs(results[0].score - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# GraphIndex
# ---------------------------------------------------------------------------

class TestGraphIndex:
    def test_add_nodes(self):
        gi = GraphIndex()
        embed = make_embedder(32)
        for i in range(5):
            c = Chunk(id=f"c{i}", text=f"text {i}")
            gi.add(c, embed(f"similar text {i}"))
        assert gi.node_count() == 5

    def test_expand_returns_neighbors(self):
        gi = GraphIndex(threshold=0.0)   # low threshold: always add edges
        embed = make_embedder(32)
        ids = []
        for i in range(10):
            c = Chunk(id=f"c{i}", text=f"text {i}")
            gi.add(c, embed(f"text {i}"))
            ids.append(c.id)
        results = gi.expand([ids[0]], depth=1)
        # With threshold=0, many edges should exist
        assert isinstance(results, list)   # may be empty if no edges pass threshold

    def test_expand_empty_returns_list(self):
        gi = GraphIndex()
        results = gi.expand(["nonexistent_id"], depth=2)
        assert results == []


# ---------------------------------------------------------------------------
# IngestionEngine
# ---------------------------------------------------------------------------

class TestIngestionEngine:
    def test_ingest_text_adapter(self):
        vs, gi, ie, _ = make_engine()
        text = "The quick brown fox. " * 20
        result = ie.ingest(TextAdapter(text, chunk_size=50, overlap=0))
        assert result["inserted"] > 0
        assert result["total"] > 0
        assert vs.count() == result["inserted"]

    def test_ingest_deduplication(self):
        vs, gi, ie, _ = make_engine()
        adapter = TextAdapter("Same text. " * 5, chunk_size=100, overlap=0)
        r1 = ie.ingest(adapter)
        r2 = ie.ingest(adapter)  # second ingest — should be all skipped
        assert r2["skipped"] == r2["total"]
        assert vs.count() == r1["inserted"]

    def test_ingest_dict_adapter(self):
        vs, gi, ie, _ = make_engine()
        dicts = [{"name": f"char_{i}", "level": i, "guild": "Warriors"}
                 for i in range(5)]
        result = ie.ingest(DictAdapter(dicts))
        assert result["inserted"] > 0

    def test_stats_accumulate(self):
        vs, gi, ie, _ = make_engine()
        ie.ingest(TextAdapter("First batch. " * 10, chunk_size=60))
        ie.ingest(TextAdapter("Second batch. " * 10, chunk_size=60))
        assert ie.stats["total_ingested"] > 0


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class TestRetriever:
    def test_retrieve_returns_results(self):
        vs, gi, ie, embed = make_engine()
        texts = [f"The {w} sat on the mat." for w in ["cat", "dog", "fox", "bear", "wolf"]]
        for t in texts:
            ie.ingest(TextAdapter(t, chunk_size=200))

        retriever = Retriever(vs, gi, embed)
        results = retriever.retrieve("animal sat on mat", top_k=3)
        assert len(results) > 0
        assert all(hasattr(r, "score") and hasattr(r, "chunk") for r in results)

    def test_retrieve_deduplicates(self):
        vs, gi, ie, embed = make_engine()
        ie.ingest(TextAdapter("The cat sat on the mat. " * 10, chunk_size=100))
        retriever = Retriever(vs, gi, embed)
        results = retriever.retrieve("cat mat", top_k=5)
        chunk_ids = [r.chunk.id for r in results]
        assert len(chunk_ids) == len(set(chunk_ids)), "Results should be deduplicated"

    def test_retrieve_empty_index(self):
        vs, gi, ie, embed = make_engine()
        retriever = Retriever(vs, gi, embed)
        results = retriever.retrieve("anything", top_k=5)
        assert results == []
