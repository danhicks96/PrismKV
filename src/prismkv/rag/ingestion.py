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

    def ingest(self, adapter: BaseAdapter, batch_size: int = 500) -> dict:
        """
        Ingest all chunks from an adapter.

        Parameters
        ----------
        adapter    : any BaseAdapter subclass
        batch_size : number of chunks per SQLite transaction (default 500).
                     Larger values reduce commit overhead during bulk ingestion.
                     Set to 1 to use the legacy one-chunk-per-transaction path.

        Returns
        -------
        dict with keys 'inserted', 'skipped', 'total'
        """
        chunks = list(adapter.chunks())
        total_inserted = 0
        total_skipped = 0

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            for i, chunk in enumerate(batch):
                chunk.timestamp = self._ingested + start + i

            # Embed each chunk (per-chunk; batched embedding APIs can wrap embed_fn externally)
            pairs = [(chunk, self.embed_fn(chunk.text)) for chunk in batch]

            # One transaction for the whole sub-batch
            insert_flags = self.vector_store.add_batch(pairs)
            batch_inserted = sum(insert_flags)
            batch_skipped = len(pairs) - batch_inserted

            # Pass only newly-inserted chunks to the graph
            new_pairs = [p for p, flag in zip(pairs, insert_flags) if flag]
            if new_pairs:
                self.graph_index.add_batch(new_pairs)

            total_inserted += batch_inserted
            total_skipped += batch_skipped

        self._ingested += total_inserted
        self._skipped += total_skipped
        return {
            "inserted": total_inserted,
            "skipped": total_skipped,
            "total": len(chunks),
        }

    @property
    def stats(self) -> dict:
        return {
            "total_ingested": self._ingested,
            "total_skipped": self._skipped,
            "vector_store_size": self.vector_store.count(),
            "graph_nodes": self.graph_index.node_count(),
            "graph_edges": self.graph_index.edge_count(),
        }
