"""
graph_index.py — NetworkX directed graph index with SQLite persistence.

Nodes = chunk IDs.  Directed edges where cosine similarity > threshold.
During ingestion, new chunks are compared against the last N_NEIGHBOR_WINDOW
chunks only to avoid quadratic cost.

Retrieval: BFS from top-K vector hits, depth=2, returns additional context nodes.

Thread safety: threading.Lock on all writes.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Dict, List, Optional, Set

import torch

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

from prismkv.rag.schema import Chunk, RetrievalResult


_CREATE_GRAPH_EDGES = """
CREATE TABLE IF NOT EXISTS graph_edges (
    src    TEXT NOT NULL,
    dst    TEXT NOT NULL,
    weight REAL NOT NULL,
    PRIMARY KEY (src, dst)
)
"""

EDGE_THRESHOLD = 0.70     # cosine similarity threshold for adding an edge
N_NEIGHBOR_WINDOW = 50    # compare each new chunk against only the last N
MAX_GRAPH_NODES = 500_000 # above this, fall back to pure vector search


class GraphIndex:
    """
    Directed similarity graph over chunk embeddings.

    Parameters
    ----------
    db_path   : SQLite file path (":memory:" for in-memory)
    threshold : cosine similarity threshold for edge creation (default 0.70)
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        threshold: float = EDGE_THRESHOLD,
    ) -> None:
        if not HAS_NX:
            raise ImportError(
                "networkx is required for GraphIndex. "
                "Install with: pip install networkx"
            )
        self.threshold = threshold
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_GRAPH_EDGES)
        self._conn.commit()

        # In-memory graph (lazily synced from DB)
        self._G: nx.DiGraph = nx.DiGraph()
        self._loaded = False

        # Ring buffer of recent (chunk_id, embedding) for window-based edge creation
        self._recent: List[tuple] = []   # [(chunk_id, Tensor), ...]

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add(self, chunk: Chunk, embedding: torch.Tensor) -> None:
        """
        Add a node for chunk and create edges to semantically similar recent chunks.

        Only compares against the last N_NEIGHBOR_WINDOW chunks to bound cost.
        """
        with self._lock:
            self._G.add_node(chunk.id)
            self._loaded = True  # mark as modified

            q_norm = torch.nn.functional.normalize(embedding.float().reshape(1, -1), dim=1)

            new_edges = []
            if self._recent:
                recent_ids = [r[0] for r in self._recent]
                recent_embs = torch.stack([r[1] for r in self._recent])   # (W, d)
                recent_norms = torch.nn.functional.normalize(recent_embs, dim=1)
                scores = (recent_norms @ q_norm.T).squeeze(1)              # (W,)

                for i, score in enumerate(scores.tolist()):
                    if score >= self.threshold:
                        src = chunk.id
                        dst = recent_ids[i]
                        self._G.add_edge(src, dst, weight=score)
                        self._G.add_edge(dst, src, weight=score)
                        new_edges.append((src, dst, score))
                        new_edges.append((dst, src, score))

            if new_edges:
                with self._conn as conn:
                    conn.executemany(
                        "INSERT OR REPLACE INTO graph_edges (src, dst, weight) VALUES (?, ?, ?)",
                        new_edges,
                    )

            # Update ring buffer
            self._recent.append((chunk.id, embedding.float()))
            if len(self._recent) > N_NEIGHBOR_WINDOW:
                self._recent.pop(0)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def expand(
        self,
        seed_chunk_ids: List[str],
        depth: int = 2,
        max_results: int = 20,
    ) -> List[RetrievalResult]:
        """
        BFS from seed nodes to depth, collecting neighbor chunk IDs.

        Parameters
        ----------
        seed_chunk_ids : list of chunk IDs to expand from (from vector search)
        depth          : BFS depth (default 2)
        max_results    : cap on total results returned

        Returns
        -------
        list of RetrievalResult with retrieval_source='graph'
        """
        if len(self._G) > MAX_GRAPH_NODES:
            return []   # graceful degradation at scale

        visited: Set[str] = set(seed_chunk_ids)
        frontier: Set[str] = set(seed_chunk_ids)
        graph_hits: Dict[str, float] = {}

        for _ in range(depth):
            next_frontier: Set[str] = set()
            for node in frontier:
                if node not in self._G:
                    continue
                for neighbor, data in self._G[node].items():
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
                        w = data.get("weight", 0.0)
                        graph_hits[neighbor] = max(graph_hits.get(neighbor, 0.0), w)
            frontier = next_frontier
            if not frontier:
                break

        # Return top hits by edge weight, excluding seeds
        sorted_hits = sorted(graph_hits.items(), key=lambda x: -x[1])[:max_results]
        results = []
        for chunk_id, score in sorted_hits:
            results.append(RetrievalResult(
                chunk=Chunk(id=chunk_id, text="", content_hash=chunk_id),
                score=score,
                retrieval_source="graph",
            ))
        return results

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def node_count(self) -> int:
        return len(self._G)

    def edge_count(self) -> int:
        return self._G.number_of_edges()

    def __repr__(self) -> str:
        return (
            f"GraphIndex(nodes={self.node_count()}, edges={self.edge_count()}, "
            f"threshold={self.threshold})"
        )
