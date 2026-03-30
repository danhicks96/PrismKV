"""
prismkv.rag — Retrieval-Augmented Generation framework.

Quick start:
    from prismkv.rag import RAGEngine, TextAdapter, DictAdapter, FileAdapter

    engine = RAGEngine(db_path="memory.sqlite", embedder=my_embed_fn)
    engine.ingest(TextAdapter("Long text to index..."))
    engine.ingest(DictAdapter([{"name": "Elowen", "level": 12, "guild": "Iron Wolves"}]))
    context = engine.query("What do we know about Elowen?")
    response = engine.generate("Tell me about Elowen.", generation_fn=my_llm)
"""

from prismkv.rag.rag_engine import RAGEngine
from prismkv.rag.adapters import TextAdapter, DictAdapter, FileAdapter
from prismkv.rag.schema import Chunk, RetrievalResult
from prismkv.rag.vector_store import VectorStore
from prismkv.rag.graph_index import GraphIndex
from prismkv.rag.ingestion import IngestionEngine
from prismkv.rag.retriever import Retriever
from prismkv.rag.context_assembler import ContextAssembler

__all__ = [
    "RAGEngine",
    "TextAdapter", "DictAdapter", "FileAdapter",
    "Chunk", "RetrievalResult",
    "VectorStore", "GraphIndex",
    "IngestionEngine", "Retriever", "ContextAssembler",
]
