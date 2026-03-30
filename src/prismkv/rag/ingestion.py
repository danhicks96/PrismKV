"""
ingestion.py — IngestionEngine: drives adapters → embedder → vector store + graph index.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch

from prismkv.rag.adapters import BaseAdapter
from prismkv.rag.schema import Chunk
from prismkv.rag.vector_store import VectorStore
from prismkv.rag.graph_index import GraphIndex


EmbedFn = Callable[[str], torch.Tensor]


class IngestionEngine:
    """
    Coordinates chunk production, embedding, deduplication, and indexing.

    Parameters
    ----------
    vector_store  : VectorStore instance
    graph_index   : GraphIndex instance
    embed_fn      : callable(text: str) → Tensor shape (d,)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        graph_index: GraphIndex,
        embed_fn: EmbedFn,
    ) -> None:
        self.vector_store = vector_store
        self.graph_index = graph_index
        self.embed_fn = embed_fn
        self._ingested = 0
        self._skipped = 0

    def ingest(self, adapter: BaseAdapter) -> dict:
        """
        Ingest all chunks from an adapter.

        Returns
        -------
        dict with keys 'inserted', 'skipped', 'total'
        """
        chunks = adapter.chunks()
        inserted = 0
        skipped = 0

        for i, chunk in enumerate(chunks):
            chunk.timestamp = self._ingested + i
            embedding = self.embed_fn(chunk.text)

            added = self.vector_store.add(chunk, embedding)
            if added:
                self.graph_index.add(chunk, embedding)
                inserted += 1
            else:
                skipped += 1

        self._ingested += inserted
        self._skipped += skipped
        return {"inserted": inserted, "skipped": skipped, "total": len(chunks)}

    @property
    def stats(self) -> dict:
        return {
            "total_ingested": self._ingested,
            "total_skipped": self._skipped,
            "vector_store_size": self.vector_store.count(),
            "graph_nodes": self.graph_index.node_count(),
            "graph_edges": self.graph_index.edge_count(),
        }
