"""
test_rag_engine.py — Tests for RAGEngine (the public API).
"""

import pytest
import torch

from prismkv.rag import RAGEngine, TextAdapter, DictAdapter


# ---------------------------------------------------------------------------
# Deterministic fake embedder
# ---------------------------------------------------------------------------

def make_embedder(dim: int = 64):
    def embed(text: str) -> torch.Tensor:
        gen = torch.Generator().manual_seed(hash(text) % (2 ** 31))
        v = torch.randn(dim, generator=gen)
        return torch.nn.functional.normalize(v, dim=0)
    return embed


def make_engine(**kwargs):
    return RAGEngine(db_path=":memory:", embedder=make_embedder(), **kwargs)


# ---------------------------------------------------------------------------
# Basic engine tests
# ---------------------------------------------------------------------------

class TestRAGEngine:
    def test_requires_embedder(self):
        with pytest.raises(ValueError, match="embedder"):
            RAGEngine(db_path=":memory:")

    def test_ingest_and_query(self):
        engine = make_engine()
        engine.ingest(TextAdapter("The ancient dragon guards the golden hoard. " * 5))
        context = engine.query("dragon hoard")
        assert isinstance(context, str)
        assert len(context) > 0

    def test_query_empty_returns_empty(self):
        engine = make_engine()
        context = engine.query("anything")
        assert context == ""

    def test_generate_calls_fn(self):
        engine = make_engine()
        engine.ingest(TextAdapter("Magic spells protect the tower. " * 5))
        called = []
        def gen_fn(prompt):
            called.append(prompt)
            return "The tower is protected."
        result = engine.generate("What protects the tower?", generation_fn=gen_fn)
        assert len(called) == 1
        assert "tower" in result.lower() or "protected" in result.lower()

    def test_generate_no_context_uses_prompt_directly(self):
        """When no relevant context is found, prompt is passed directly to gen_fn."""
        engine = make_engine()
        received = []
        def gen_fn(prompt):
            received.append(prompt)
            return prompt[:50]
        engine.generate("test query", generation_fn=gen_fn)
        assert received[0] == "test query"

    def test_ingest_returns_stats(self):
        engine = make_engine()
        result = engine.ingest(TextAdapter("Hello world. " * 20, chunk_size=60))
        assert "inserted" in result
        assert "skipped" in result
        assert "total" in result
        assert result["inserted"] + result["skipped"] == result["total"]

    def test_duplicate_detection(self):
        engine = make_engine()
        adapter = TextAdapter("Same content. " * 10, chunk_size=80)
        r1 = engine.ingest(adapter)
        r2 = engine.ingest(adapter)
        assert r1["inserted"] > 0
        assert r2["inserted"] == 0
        assert r2["skipped"] == r2["total"]

    def test_stats_property(self):
        engine = make_engine()
        engine.ingest(TextAdapter("Test data. " * 20, chunk_size=60))
        s = engine.stats
        assert "vector_store_size" in s
        assert "graph_nodes" in s
        assert s["vector_store_size"] > 0

    def test_top_k_override(self):
        engine = make_engine(top_k=10)
        texts = [f"Document about topic {i}. " * 5 for i in range(20)]
        for t in texts:
            engine.ingest(TextAdapter(t, chunk_size=200))
        results = engine.retrieve("topic", top_k=3)
        assert len(results) <= 5   # top_k=3 vector + possible graph hits ≤ 5 here


class TestRAGEngineUsurperPattern:
    """
    Tests the DictAdapter game-state ingestion pattern for usurper-successor.
    """

    CHARACTER_STATES = [
        {"name": "Elowen", "level": 12, "guild": "Iron Wolves",
         "last_action": "killed an ancient dragon", "location": "throne room"},
        {"name": "Torvik", "level": 8, "guild": "Merchant League",
         "last_action": "negotiated a trade deal", "location": "market"},
        {"name": "Syla", "level": 15, "guild": "Shadow Court",
         "last_action": "stole the crown jewels", "location": "vault"},
    ]

    def test_ingest_game_state_dicts(self):
        engine = make_engine()
        result = engine.ingest(DictAdapter(self.CHARACTER_STATES))
        assert result["inserted"] > 0

    def test_retrieve_character_by_name(self):
        engine = make_engine()
        engine.ingest(DictAdapter(self.CHARACTER_STATES))
        context = engine.query("What has Elowen done recently?")
        assert isinstance(context, str)

    def test_mixed_ingestion(self):
        engine = make_engine()
        engine.ingest(DictAdapter(self.CHARACTER_STATES))
        world_log = (
            "The Iron Wolves have declared war on the Shadow Court. "
            "Elowen has been promoted to Guild Champion. "
            "The throne room is sealed. "
        )
        engine.ingest(TextAdapter(world_log))
        context = engine.query("throne room situation", top_k=5)
        assert len(context) > 0

    def test_query_returns_top_3_context(self):
        engine = make_engine()
        engine.ingest(DictAdapter(self.CHARACTER_STATES))
        results = engine.retrieve("guild activities", top_k=3)
        assert 1 <= len(results) <= 6   # top_k=3 vector + possible graph expansion

    def test_generate_narrative(self):
        engine = make_engine()
        engine.ingest(DictAdapter(self.CHARACTER_STATES))
        responses = []
        def fake_llm(prompt):
            responses.append(prompt)
            return prompt[:100]
        engine.generate("Elowen enters the throne room", generation_fn=fake_llm)
        # Context should have been included in the prompt
        assert len(responses[0]) > len("Elowen enters the throne room")
