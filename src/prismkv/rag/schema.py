"""
schema.py — Core data structures for the PrismKV RAG framework.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    """
    A single unit of text to be indexed and retrieved.

    Attributes
    ----------
    id          : unique identifier (UUID or deterministic hash)
    text        : raw text content
    metadata    : arbitrary key-value pairs (type, field, entity, source, etc.)
    source_id   : identifier of the source document / adapter
    content_hash: SHA-256 of text for deduplication
    timestamp   : ingestion order counter (used for narrative coherence sorting)
    """
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_id: str = ""
    content_hash: str = ""
    timestamp: int = 0

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.text.encode()).hexdigest()
        if not self.id:
            self.id = self.content_hash[:16]


@dataclass
class Node:
    """
    A node in the graph index, corresponding to one Chunk.

    Attributes
    ----------
    chunk_id : id of the associated Chunk
    edges    : list of (neighbor_chunk_id, cosine_similarity) pairs
    """
    chunk_id: str
    edges: List[tuple] = field(default_factory=list)   # [(chunk_id, weight), ...]


@dataclass
class RetrievalResult:
    """
    A single retrieved chunk with its retrieval score.

    Attributes
    ----------
    chunk      : the retrieved Chunk
    score      : cosine similarity or BFS-weighted score
    retrieval_source : 'vector' or 'graph'
    """
    chunk: Chunk
    score: float
    retrieval_source: str = "vector"
