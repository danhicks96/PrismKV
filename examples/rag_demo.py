"""
rag_demo.py — CPU-only RAG demo with no model downloads.

Uses a deterministic fake embedder.  Demonstrates:
- TextAdapter ingestion
- DictAdapter ingestion (game-state dict pattern)
- query() context retrieval
- generate() with a trivial pass-through LLM

Usage:
    python3 examples/rag_demo.py
"""

import torch
from prismkv.rag import RAGEngine, TextAdapter, DictAdapter


# ---------------------------------------------------------------------------
# Fake embedder: deterministic, no downloads needed
# ---------------------------------------------------------------------------

def fake_embed(text: str, dim: int = 128) -> torch.Tensor:
    """Reproducible unit-norm embedding from text hash."""
    gen = torch.Generator().manual_seed(abs(hash(text)) % (2 ** 31))
    v = torch.randn(dim, generator=gen)
    return torch.nn.functional.normalize(v, dim=0)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

WORLD_LOG = """
The Iron Wolves guild declared war on the Shadow Court after the theft of the
Starfire Amulet.  Three assassins were dispatched to eliminate the guild leader
Elowen.  The battle of Thornwall Keep left 200 soldiers dead.  The Dragon Queen
has remained neutral but is watching closely.  Ancient ruins beneath the city
hold secrets that could shift the balance of power.
"""

CHARACTER_STATES = [
    {"name": "Elowen", "level": 12, "guild": "Iron Wolves",
     "last_action": "defeated three assassins", "location": "Thornwall Keep",
     "hp": 89, "mana": 144},
    {"name": "Varek", "level": 18, "guild": "Shadow Court",
     "last_action": "ordered the theft of the Starfire Amulet", "location": "Shadow Citadel"},
    {"name": "Mira", "level": 6, "guild": "Merchant League",
     "last_action": "sold maps of the ancient ruins", "location": "Market District"},
    {"name": "Dorn", "level": 14, "guild": "Iron Wolves",
     "last_action": "scouted Thornwall Keep perimeter", "location": "eastern watchtower"},
]


def main():
    print("══════════════════════════════════════════════════════")
    print("  PrismKV RAG Demo — CPU-only, no model downloads")
    print("══════════════════════════════════════════════════════\n")

    engine = RAGEngine(
        db_path=":memory:",
        embedder=fake_embed,
        top_k=3,
    )

    # Ingest world narrative
    r1 = engine.ingest(TextAdapter(WORLD_LOG, chunk_size=300, overlap=30))
    print(f"World log ingested: {r1}")

    # Ingest character states as structured dicts
    r2 = engine.ingest(DictAdapter(CHARACTER_STATES, entity_key="name"))
    print(f"Character states ingested: {r2}")

    print(f"\nEngine: {engine}\n")

    # Queries
    queries = [
        "What happened at Thornwall Keep?",
        "What is Elowen's current situation?",
        "Tell me about the Shadow Court's recent activities.",
        "Who is selling maps of the ruins?",
    ]

    for q in queries:
        context = engine.query(q, top_k=3)
        print(f"Query: {q!r}")
        # Show first 200 chars of context
        print(f"Context: {context[:200]}{'...' if len(context) > 200 else ''}\n")

    # Full generate call
    print("──── generate() demo ────")
    response = engine.generate(
        "Elowen arrives at the ancient ruins. What does she find?",
        generation_fn=lambda p: f"[Prompt assembled, len={len(p)}] Based on context, "
                                 "Elowen discovers ancient stone tablets.",
        top_k=5,
    )
    print(f"Response: {response}\n")

    print("══════════════════════════════════════════════════════")
    print("  Demo complete (<30s CPU-only)")
    print("══════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
