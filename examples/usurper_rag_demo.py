"""
usurper_rag_demo.py — RAG integration pattern for usurper-successor BBS game.

Demonstrates the exact API pattern the game project should use:
- Ingest 50 character-state dicts
- Ingest world event log
- Query for context
- Generate narrative (using a stub LLM — swap in ollama for real use)

Usage:
    python3 examples/usurper_rag_demo.py

Real integration (in usurper-successor):
    from prismkv.rag import RAGEngine, DictAdapter, TextAdapter
    from prismkv.cache import PrismKVCache, PrismKVConfig

    # Build engine once at game startup
    engine = RAGEngine(
        db_path="./game_memory.sqlite",
        embedder=lambda text: my_sentence_transformer.encode(text, convert_to_tensor=True),
    )

    # Inject game state on each turn
    engine.ingest(DictAdapter(game.get_all_character_states()))
    engine.ingest(TextAdapter(game.get_event_log_since_last_turn()))

    # Generate narrative
    response = engine.generate(
        prompt=f"{player_name} enters {current_location}. What happens?",
        generation_fn=lambda p: ollama.generate("llama3.2:1b", p)["response"]
    )
"""

import random
import torch

from prismkv.rag import RAGEngine, DictAdapter, TextAdapter


# ---------------------------------------------------------------------------
# Synthetic game state generator
# ---------------------------------------------------------------------------

GUILDS = ["Iron Wolves", "Shadow Court", "Merchant League", "Dragon Guard", "Arcane Circle"]
LOCATIONS = ["throne room", "market", "dungeon", "forest", "vault", "cathedral", "ruins"]
ACTIONS = [
    "defeated a rival", "stole an artifact", "brokered a peace treaty",
    "discovered ancient magic", "escaped imprisonment", "won a tournament",
    "gathered intelligence", "sabotaged a shipment",
]


def generate_game_state(n_chars: int = 50, seed: int = 42) -> list:
    rng = random.Random(seed)
    chars = []
    for i in range(n_chars):
        chars.append({
            "name": f"Character_{i:03d}",
            "level": rng.randint(1, 20),
            "guild": rng.choice(GUILDS),
            "last_action": rng.choice(ACTIONS),
            "location": rng.choice(LOCATIONS),
            "hp": rng.randint(20, 200),
            "gold": rng.randint(0, 10000),
            "is_alive": rng.random() > 0.1,
        })
    return chars


EVENT_LOG = """
The annual Tournament of Blades concluded with Character_007 claiming the
championship.  Three guild wars are now active: Iron Wolves vs Shadow Court,
Merchant League vs Arcane Circle.  The Dragon Guard has declared neutrality.
An ancient artifact — the Soulstone of Karath — was stolen from the royal
vault.  The king has offered 10,000 gold for its recovery.  Several characters
near the throne room reported seeing mysterious hooded figures.  The dungeons
are rumored to contain a portal to the ruins.
"""


# ---------------------------------------------------------------------------
# Fake embedder (no downloads)
# ---------------------------------------------------------------------------

def fake_embed(text: str, dim: int = 128) -> torch.Tensor:
    gen = torch.Generator().manual_seed(abs(hash(text)) % (2 ** 31))
    v = torch.randn(dim, generator=gen)
    return torch.nn.functional.normalize(v, dim=0)


# ---------------------------------------------------------------------------
# Stub LLM (swap in ollama for real use)
# ---------------------------------------------------------------------------

def stub_llm(prompt: str) -> str:
    """Stand-in for a real LLM. Returns a fixed response for demo purposes."""
    return (
        "[Stub LLM response — replace with: "
        "ollama.generate('llama3.2:1b', prompt)['response']]\n"
        f"Prompt length received: {len(prompt)} chars."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("══════════════════════════════════════════════════════")
    print("  usurper-successor RAG Integration Demo")
    print("══════════════════════════════════════════════════════\n")

    # Build engine
    engine = RAGEngine(
        db_path=":memory:",
        embedder=fake_embed,
        top_k=5,
        graph_threshold=0.60,
    )

    # 1. Ingest 50 character states
    states = generate_game_state(n_chars=50)
    r1 = engine.ingest(DictAdapter(states, entity_key="name", source_id="character_state"))
    print(f"Character states: {r1['inserted']} chunks inserted, {r1['skipped']} skipped")

    # 2. Ingest world event log
    r2 = engine.ingest(TextAdapter(EVENT_LOG, chunk_size=250, overlap=30, source_id="event_log"))
    print(f"Event log: {r2['inserted']} chunks inserted")

    print(f"\n{engine}\n")

    # 3. Sample queries
    queries = [
        ("throne room situation", 3),
        ("who is in the dungeon", 3),
        ("guild conflicts", 5),
    ]

    for query, k in queries:
        results = engine.retrieve(query, top_k=k)
        print(f"Query: {query!r}  →  {len(results)} result(s)")
        for r in results[:3]:
            snippet = r.chunk.text[:80].replace("\n", " ")
            print(f"  [{r.score:.3f}] {snippet}")
        print()

    # 4. Generate narrative
    print("──── Narrative generation ────")
    for scenario in [
        "Character_007 enters the throne room. What do they discover?",
        "The Soulstone of Karath has been found in the dungeon. What happens next?",
    ]:
        print(f"\nScenario: {scenario}")
        response = engine.generate(
            scenario,
            generation_fn=stub_llm,
            context_prefix="Game State Context:\n",
        )
        print(f"Response: {response}")

    print("\n══════════════════════════════════════════════════════")
    print("  Integration test complete")
    print("  Swap stub_llm() with your ollama/OpenAI call to go live")
    print("══════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
