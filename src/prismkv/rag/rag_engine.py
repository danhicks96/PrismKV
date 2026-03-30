"""
rag_engine.py — RAGEngine: unified interface for ingest + query + generate.

This is the primary public API for the PrismKV RAG framework.  The
``usurper-successor`` game project and other consumers should use this class
directly rather than composing the lower-level components themselves.

Usage
-----
    from prismkv.rag import RAGEngine, TextAdapter, DictAdapter

    engine = RAGEngine(db_path="./game_memory.sqlite", embedder=my_embed_fn)
    engine.ingest(DictAdapter(character_state_dicts))
    engine.ingest(TextAdapter(world_event_log))

    context = engine.query("Elowen enters the throne room", top_k=5)
    response = engine.generate(
        "Elowen enters the throne room. What happens next?",
        generation_fn=lambda p: ollama.generate("llama3.2:1b", p)["response"]
    )

Author: Dan Hicks (github.com/danhicks96)
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch

from prismkv.rag.adapters import BaseAdapter
from prismkv.rag.context_assembler import ContextAssembler
from prismkv.rag.graph_index import GraphIndex
from prismkv.rag.ingestion import IngestionEngine
from prismkv.rag.retriever import Retriever
from prismkv.rag.schema import RetrievalResult
from prismkv.rag.vector_store import VectorStore


GenerationFn = Callable[[str], str]
EmbedFn = Callable[[str], torch.Tensor]


class RAGEngine:
    """
    Unified Retrieval-Augmented Generation engine.

    Parameters
    ----------
    db_path     : path to the SQLite file (":memory:" for ephemeral / testing)
    embedder    : callable(text: str) → Tensor shape (d,)
                  For tests: use a deterministic fake.
                  For production: use a SentenceTransformer or similar.
    graph_threshold : cosine similarity threshold for graph edges (default 0.70)
    context_max_tokens : token budget for assembled context string (default 1024)
    top_k       : default number of vector search results (default 5)
    graph_depth : BFS depth for graph expansion (default 2)
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        embedder: Optional[EmbedFn] = None,
        graph_threshold: float = 0.70,
        context_max_tokens: int = 1024,
        top_k: int = 5,
        graph_depth: int = 2,
    ) -> None:
        if embedder is None:
            raise ValueError(
                "RAGEngine requires an embedder function.  "
                "Pass embedder=lambda text: my_model.encode(text) or "
                "embedder=lambda text: torch.randn(128) for testing."
            )

        self._db_path = db_path
        self._embedder = embedder
        self._top_k = top_k

        self._vector_store = VectorStore(db_path=db_path)
        self._graph_index = GraphIndex(db_path=db_path, threshold=graph_threshold)
        self._ingestion = IngestionEngine(
            self._vector_store, self._graph_index, embedder
        )
        self._retriever = Retriever(
            self._vector_store, self._graph_index, embedder,
            graph_depth=graph_depth,
        )
        self._assembler = ContextAssembler(max_tokens=context_max_tokens)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, adapter: BaseAdapter) -> dict:
        """
        Ingest all chunks from an adapter into the engine's index.

        Parameters
        ----------
        adapter : any BaseAdapter subclass (TextAdapter, DictAdapter, FileAdapter)

        Returns
        -------
        dict with 'inserted', 'skipped', 'total'
        """
        return self._ingestion.ingest(adapter)

    def query(self, text: str, top_k: Optional[int] = None) -> str:
        """
        Retrieve relevant context for a query and return it as a string.

        Parameters
        ----------
        text  : query text
        top_k : number of results (default: engine's top_k)

        Returns
        -------
        Assembled context string ready to prepend to a prompt.
        """
        k = top_k if top_k is not None else self._top_k
        results = self._retriever.retrieve(text, top_k=k)
        return self._assembler.assemble(results)

    def retrieve(self, text: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Return raw RetrievalResult objects (for inspection or custom assembly).
        """
        k = top_k if top_k is not None else self._top_k
        return self._retriever.retrieve(text, top_k=k)

    def generate(
        self,
        prompt: str,
        generation_fn: GenerationFn,
        top_k: Optional[int] = None,
        context_prefix: str = "Context:\n",
        prompt_prefix: str = "\n\nQuestion: ",
    ) -> str:
        """
        Retrieve context, assemble a prompt, and call generation_fn.

        Parameters
        ----------
        prompt        : user query or instruction
        generation_fn : callable(full_prompt: str) → str
                        e.g. lambda p: ollama.generate("llama3.2:1b", p)["response"]
        top_k         : override default retrieval count
        context_prefix: header prepended to context block
        prompt_prefix : separator between context and prompt

        Returns
        -------
        Generated text string
        """
        context = self.query(prompt, top_k=top_k)
        if context:
            full_prompt = f"{context_prefix}{context}{prompt_prefix}{prompt}"
        else:
            full_prompt = prompt
        return generation_fn(full_prompt)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        return self._ingestion.stats

    def __repr__(self) -> str:
        s = self.stats
        return (
            f"RAGEngine(db={self._db_path!r}, "
            f"chunks={s['vector_store_size']}, "
            f"graph_nodes={s['graph_nodes']}, "
            f"graph_edges={s['graph_edges']})"
        )
