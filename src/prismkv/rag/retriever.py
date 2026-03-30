"""
retriever.py — Hybrid vector + graph retrieval.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch

from prismkv.rag.schema import Chunk, RetrievalResult
from prismkv.rag.vector_store import VectorStore
from prismkv.rag.graph_index import GraphIndex


EmbedFn = Callable[[str], torch.Tensor]


class Retriever:
    """
    Combines vector search (top-K cosine) with graph BFS expansion.

    Parameters
    ----------
    vector_store   : VectorStore
    graph_index    : GraphIndex
    embed_fn       : callable(text) → embedding Tensor
    graph_depth    : BFS depth for graph expansion (default 2)
    graph_results  : max graph hits to return alongside vector hits
    """

    def __init__(
        self,
        vector_store: VectorStore,
        graph_index: GraphIndex,
        embed_fn: EmbedFn,
        graph_depth: int = 2,
        graph_results: int = 10,
    ) -> None:
        self.vector_store = vector_store
        self.graph_index = graph_index
        self.embed_fn = embed_fn
        self.graph_depth = graph_depth
        self.graph_results = graph_results

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Retrieve top-k chunks for a query using vector + graph search.

        Parameters
        ----------
        query : natural language query
        top_k : number of vector results (graph results may add more)

        Returns
        -------
        List of RetrievalResult, deduplicated, sorted by score descending.
        """
        q_emb = self.embed_fn(query)

        # Vector search
        vector_hits = self.vector_store.search(q_emb, top_k=top_k)
        seed_ids = [r.chunk.id for r in vector_hits]

        # Graph expansion from vector hits
        graph_hits = self.graph_index.expand(
            seed_ids,
            depth=self.graph_depth,
            max_results=self.graph_results,
        )

        # Resolve graph hits into full Chunk objects from vector store
        resolved_graph = []
        if graph_hits:
            graph_ids = [r.chunk.id for r in graph_hits]
            # Fetch from vector store DB
            full_chunks = self.vector_store._fetch_chunks(graph_ids)
            id_to_chunk = {c.id: c for c in full_chunks}
            for r in graph_hits:
                if r.chunk.id in id_to_chunk:
                    resolved_graph.append(RetrievalResult(
                        chunk=id_to_chunk[r.chunk.id],
                        score=r.score,
                        retrieval_source="graph",
                    ))

        # Merge and deduplicate
        seen: set = set(seed_ids)
        merged = list(vector_hits)
        for r in resolved_graph:
            if r.chunk.id not in seen:
                merged.append(r)
                seen.add(r.chunk.id)

        merged.sort(key=lambda r: (-r.score, r.chunk.timestamp))
        return merged
