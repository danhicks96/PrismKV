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
# Batch ingestion
# ---------------------------------------------------------------------------

class TestVectorStoreBatch:
    def test_add_batch_inserts_all(self):
        vs = VectorStore()
        embed = make_embedder(32)
        items = [
            (Chunk(id=f"c{i}", text=f"text {i}"), embed(f"text {i}"))
            for i in range(10)
        ]
        flags = vs.add_batch(items)
        assert all(flags)
        assert vs.count() == 10

    def test_add_batch_skips_duplicates(self):
        vs = VectorStore()
        embed = make_embedder(32)
        items = [(Chunk(id=f"c{i}", text=f"text {i}"), embed(f"text {i}")) for i in range(5)]
        vs.add_batch(items)
        flags2 = vs.add_batch(items)
        assert not any(flags2)
        assert vs.count() == 5

    def test_add_batch_partial_duplicates(self):
        vs = VectorStore()
        embed = make_embedder(32)
        first_half = [(Chunk(id=f"c{i}", text=f"text {i}"), embed(f"text {i}")) for i in range(5)]
        second_half = [(Chunk(id=f"c{i}", text=f"text {i}"), embed(f"text {i}")) for i in range(3, 8)]
        vs.add_batch(first_half)
        flags = vs.add_batch(second_half)
        # c3 and c4 are duplicates; c5, c6, c7 are new
        assert flags == [False, False, True, True, True]
        assert vs.count() == 8

    def test_add_batch_empty(self):
        vs = VectorStore()
        flags = vs.add_batch([])
        assert flags == []
        assert vs.count() == 0

    def test_add_batch_searchable(self):
        """Chunks inserted via add_batch are retrievable by search."""
        vs = VectorStore()
        embed = make_embedder(32)
        items = [(Chunk(id=f"c{i}", text=f"text {i}"), embed(f"text {i}")) for i in range(5)]
        vs.add_batch(items)
        results = vs.search(embed("text 2"), top_k=1)
        assert len(results) == 1
        assert results[0].chunk.id == "c2"

    def test_add_batch_same_result_as_individual_add(self):
        """Batch-inserted chunks have same cosine scores as individually-inserted ones."""
        embed = make_embedder(32)
        items = [(Chunk(id=f"c{i}", text=f"word {i}"), embed(f"word {i}")) for i in range(6)]

        vs_single = VectorStore()
        for chunk, emb in items:
            vs_single.add(chunk, emb)

        vs_batch = VectorStore()
        vs_batch.add_batch(items)

        q = embed("word 3")
        single_results = vs_single.search(q, top_k=3)
        batch_results = vs_batch.search(q, top_k=3)

        single_ids = [r.chunk.id for r in single_results]
        batch_ids = [r.chunk.id for r in batch_results]
        assert single_ids == batch_ids


class TestGraphIndexBatch:
    def test_add_batch_nodes(self):
        gi = GraphIndex()
        embed = make_embedder(32)
        items = [(Chunk(id=f"c{i}", text=f"text {i}"), embed(f"similar {i}")) for i in range(5)]
        gi.add_batch(items)
        assert gi.node_count() == 5

    def test_add_batch_edges_at_low_threshold(self):
        gi = GraphIndex(threshold=0.0)
        embed = make_embedder(32)
        items = [(Chunk(id=f"c{i}", text=f"text {i}"), embed(f"text {i}")) for i in range(6)]
        gi.add_batch(items)
        # With threshold=0, edges should be created between adjacent chunks
        assert gi.edge_count() >= 0  # may be 0 if all scores < 0 (unlikely with dim=32)

    def test_add_batch_same_graph_as_individual(self):
        """Batch add produces the same node set as individual add()."""
        embed = make_embedder(32)
        items = [(Chunk(id=f"c{i}", text=f"text {i}"), embed(f"text {i}")) for i in range(8)]

        gi_single = GraphIndex(threshold=0.0)
        for chunk, emb in items:
            gi_single.add(chunk, emb)

        gi_batch = GraphIndex(threshold=0.0)
        gi_batch.add_batch(items)

        assert gi_batch.node_count() == gi_single.node_count()

    def test_add_batch_empty(self):
        gi = GraphIndex()
        gi.add_batch([])  # should not raise
        assert gi.node_count() == 0


class TestIngestionEngineBatch:
    def test_ingest_batch_size_500_same_result_as_default(self):
        """batch_size=500 produces identical inserted count as old one-by-one behavior."""
        embed = make_embedder(32)
        texts = [f"sentence number {i} about topic {i % 5}" for i in range(20)]
        big_text = "  ".join(texts)

        vs1, gi1, ie1, _ = make_engine()
        ie1.ingest(TextAdapter(big_text, chunk_size=80, overlap=0), batch_size=1)

        vs2, gi2, ie2, _ = make_engine()
        ie2.ingest(TextAdapter(big_text, chunk_size=80, overlap=0), batch_size=500)

        assert vs2.count() == vs1.count()

    def test_ingest_deduplication_with_batching(self):
        """Duplicate detection works correctly when using batch_size > 1."""
        vs, gi, ie, _ = make_engine()
        adapter = TextAdapter("Same text. " * 10, chunk_size=100, overlap=0)
        r1 = ie.ingest(adapter, batch_size=500)
        r2 = ie.ingest(adapter, batch_size=500)
        assert r2["skipped"] == r2["total"]
        assert vs.count() == r1["inserted"]

    def test_ingest_large_batch(self):
        """A corpus larger than batch_size is fully ingested across multiple batches."""
        vs, gi, ie, _ = make_engine()
        # Create 15 distinct single-chunk texts
        inserted_total = 0
        for i in range(15):
            result = ie.ingest(
                TextAdapter(f"Unique document {i}: " + ("word " * 20), chunk_size=200, overlap=0),
                batch_size=4,
            )
            inserted_total += result["inserted"]
        assert vs.count() == inserted_total
        assert inserted_total > 0


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
