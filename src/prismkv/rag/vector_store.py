"""
vector_store.py — SQLite-backed vector store with pure-torch cosine search.

Stores embeddings as float32 numpy blobs in SQLite.  On startup (or after
new ingestion), loads all embeddings into a (N, d) tensor for fast CPU cosine
search via torch.mm.

Handles N=1M chunks within 62GB RAM (384-dim: ~1.5 GB).

Thread safety: write lock on INSERT; reads are lock-free (snapshot semantics).
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from prismkv.rag.schema import Chunk, RetrievalResult


# SQL schema
_CREATE_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    text         TEXT NOT NULL,
    metadata     TEXT NOT NULL DEFAULT '{}',
    source_id    TEXT NOT NULL DEFAULT '',
    content_hash TEXT UNIQUE NOT NULL,
    timestamp    INTEGER NOT NULL DEFAULT 0
)
"""

_CREATE_EMBEDDINGS = """
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id TEXT PRIMARY KEY,
    vector   BLOB NOT NULL
)
"""

_CREATE_METADATA_TABLE = """
CREATE TABLE IF NOT EXISTS store_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
)
"""

SCHEMA_VERSION = "prismkv-vs-v1"


class VectorStore:
    """
    SQLite + pure-torch cosine similarity vector store.

    Parameters
    ----------
    db_path   : path to the SQLite database file (":memory:" for in-memory)
    emb_dim   : embedding dimension (inferred from first insert if None)
    """

    def __init__(self, db_path: str = ":memory:", emb_dim: Optional[int] = None) -> None:
        self.db_path = db_path
        self.emb_dim = emb_dim
        self._lock = threading.Lock()

        # In-memory cache of (chunk_ids, embedding matrix)
        self._ids: List[str] = []
        self._emb_matrix: Optional[torch.Tensor] = None   # (N, d) float32
        self._dirty = True  # needs reload from DB

        self._conn = self._make_connection()
        self._init_schema()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _make_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        with self._conn as conn:
            conn.execute(_CREATE_CHUNKS)
            conn.execute(_CREATE_EMBEDDINGS)
            conn.execute(_CREATE_METADATA_TABLE)
            conn.execute(
                "INSERT OR IGNORE INTO store_metadata (key, value) VALUES (?, ?)",
                ("version", SCHEMA_VERSION),
            )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, chunk: Chunk, embedding: torch.Tensor) -> bool:
        """
        Insert a chunk and its embedding.  Skips exact duplicates (by content_hash).

        Returns True if inserted, False if already present.
        """
        import json
        vec = embedding.float().numpy().tobytes()

        with self._lock:
            try:
                with self._conn as conn:
                    conn.execute(
                        "INSERT INTO chunks (id, text, metadata, source_id, content_hash, timestamp) "
                        "VALUES (?, ?, ?, ?, ?, "
                        "(SELECT COALESCE(MAX(timestamp), 0) + 1 FROM chunks))",
                        (chunk.id, chunk.text, json.dumps(chunk.metadata),
                         chunk.source_id, chunk.content_hash),
                    )
                    conn.execute(
                        "INSERT INTO embeddings (chunk_id, vector) VALUES (?, ?)",
                        (chunk.id, vec),
                    )
                self._dirty = True
                return True
            except sqlite3.IntegrityError:
                # content_hash UNIQUE violation → duplicate
                return False

    def add_batch(
        self, items: List[Tuple[Chunk, torch.Tensor]]
    ) -> List[bool]:
        """
        Insert a batch of (Chunk, embedding) pairs in a single SQLite transaction.

        Deduplicates by content_hash (same semantics as add()).  Setting _dirty
        only once per batch avoids reloading the embedding matrix N times during
        large ingestion runs.

        Returns
        -------
        list[bool] parallel to *items* — True if the chunk was newly inserted,
        False if it was already present (duplicate skipped).
        """
        import json

        if not items:
            return []

        with self._lock:
            # One SELECT to find which content_hashes already exist in the DB.
            hashes = [c.content_hash for c, _ in items]
            placeholders = ",".join("?" * len(hashes))
            existing = {
                row[0]
                for row in self._conn.execute(
                    f"SELECT content_hash FROM chunks WHERE content_hash IN ({placeholders})",
                    hashes,
                ).fetchall()
            }

            # Deduplicate within the batch itself (keep first occurrence per hash).
            seen_in_batch: set = set()
            new_items = []
            flags = []
            for c, e in items:
                if c.content_hash in existing or c.content_hash in seen_in_batch:
                    flags.append(False)
                else:
                    seen_in_batch.add(c.content_hash)
                    new_items.append((c, e))
                    flags.append(True)

            if new_items:
                max_ts = self._conn.execute(
                    "SELECT COALESCE(MAX(timestamp), 0) FROM chunks"
                ).fetchone()[0]

                chunk_rows = [
                    (
                        c.id, c.text, json.dumps(c.metadata),
                        c.source_id, c.content_hash, max_ts + i + 1,
                    )
                    for i, (c, _) in enumerate(new_items)
                ]
                emb_rows = [
                    (c.id, e.float().numpy().tobytes())
                    for c, e in new_items
                ]

                with self._conn as conn:
                    conn.executemany(
                        "INSERT INTO chunks "
                        "(id, text, metadata, source_id, content_hash, timestamp) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        chunk_rows,
                    )
                    conn.executemany(
                        "INSERT INTO embeddings (chunk_id, vector) VALUES (?, ?)",
                        emb_rows,
                    )
                self._dirty = True  # mark once for the whole batch

        return flags

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_embedding: torch.Tensor, top_k: int = 5) -> List[RetrievalResult]:
        """
        Return top-k chunks by cosine similarity to query_embedding.

        Parameters
        ----------
        query_embedding : Tensor shape (d,) or (1, d)
        top_k           : number of results to return
        """
        self._ensure_matrix_loaded()

        if self._emb_matrix is None or len(self._ids) == 0:
            return []

        q = query_embedding.float().reshape(1, -1)
        q = torch.nn.functional.normalize(q, dim=1)
        mat = torch.nn.functional.normalize(self._emb_matrix, dim=1)

        scores = (mat @ q.T).squeeze(1)                         # (N,)
        k = min(top_k, len(self._ids))
        top_scores, top_indices = scores.topk(k)

        chunks = self._fetch_chunks([self._ids[i] for i in top_indices.tolist()])
        return [
            RetrievalResult(chunk=c, score=float(s), retrieval_source="vector")
            for c, s in zip(chunks, top_scores.tolist())
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_matrix_loaded(self) -> None:
        if not self._dirty:
            return
        rows = self._conn.execute(
            "SELECT e.chunk_id, e.vector FROM embeddings e"
        ).fetchall()
        if not rows:
            self._ids = []
            self._emb_matrix = None
            self._dirty = False
            return

        ids = []
        vecs = []
        for chunk_id, blob in rows:
            ids.append(chunk_id)
            arr = np.frombuffer(blob, dtype=np.float32)
            vecs.append(arr)
            if self.emb_dim is None:
                self.emb_dim = len(arr)

        self._ids = ids
        self._emb_matrix = torch.tensor(np.stack(vecs), dtype=torch.float32)
        self._dirty = False

    def _fetch_chunks(self, chunk_ids: List[str]) -> List[Chunk]:
        import json
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._conn.execute(
            f"SELECT id, text, metadata, source_id, content_hash, timestamp "
            f"FROM chunks WHERE id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        # Preserve order matching chunk_ids
        by_id = {}
        for row in rows:
            meta = json.loads(row[2])
            c = Chunk(id=row[0], text=row[1], metadata=meta,
                      source_id=row[3], content_hash=row[4], timestamp=row[5])
            by_id[c.id] = c
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def __repr__(self) -> str:
        return f"VectorStore(db={self.db_path!r}, n_chunks={self.count()})"
