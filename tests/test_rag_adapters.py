"""
test_rag_adapters.py — Tests for TextAdapter, DictAdapter, FileAdapter.
"""

import tempfile
import pytest
from prismkv.rag.adapters import TextAdapter, DictAdapter, FileAdapter
from prismkv.rag.schema import Chunk


class TestTextAdapter:
    def test_produces_chunks(self):
        adapter = TextAdapter("Hello world " * 50, chunk_size=100, overlap=20)
        chunks = adapter.chunks()
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_size_respected(self):
        adapter = TextAdapter("A" * 1000, chunk_size=200, overlap=0)
        chunks = adapter.chunks()
        for c in chunks[:-1]:   # last chunk may be shorter
            assert len(c.text) <= 200

    def test_empty_text_no_chunks(self):
        assert TextAdapter("").chunks() == []
        assert TextAdapter("   \n  ").chunks() == []

    def test_chunk_ids_unique(self):
        adapter = TextAdapter("Some text here. " * 100, chunk_size=50, overlap=0)
        chunks = adapter.chunks()
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_metadata_propagated(self):
        adapter = TextAdapter("Test text.", metadata={"doc_type": "test"}, source_id="src1")
        chunks = adapter.chunks()
        for c in chunks:
            assert c.metadata.get("doc_type") == "test"
            assert c.source_id == "src1"

    def test_content_hash_set(self):
        adapter = TextAdapter("Hello world.")
        chunks = adapter.chunks()
        assert all(len(c.content_hash) == 64 for c in chunks)


class TestDictAdapter:
    def test_produces_chunks_for_each_field(self):
        dicts = [{"name": "Alice", "level": 5, "guild": "Mages"}]
        chunks = DictAdapter(dicts).chunks()
        texts = [c.text for c in chunks]
        assert any("Alice" in t for t in texts)
        assert any("level" in t.lower() or "5" in t for t in texts)
        assert any("Mages" in t for t in texts)

    def test_entity_key_skipped(self):
        dicts = [{"name": "Bob", "hp": 100}]
        chunks = DictAdapter(dicts, entity_key="name").chunks()
        # "name" itself should not appear as a key in the sentences
        assert not any("name is Bob" in c.text.lower() for c in chunks)

    def test_metadata_contains_entity_and_field(self):
        dicts = [{"name": "Eve", "rank": "captain"}]
        chunks = DictAdapter(dicts).chunks()
        for c in chunks:
            assert c.metadata.get("entity") == "Eve"
            assert "field" in c.metadata

    def test_empty_list_no_chunks(self):
        assert DictAdapter([]).chunks() == []

    def test_multiple_dicts(self):
        dicts = [{"name": f"char_{i}", "level": i} for i in range(5)]
        chunks = DictAdapter(dicts).chunks()
        assert len(chunks) == 5   # 1 non-name field per dict

    def test_boolean_values(self):
        dicts = [{"name": "Hero", "is_alive": True, "is_cursed": False}]
        chunks = DictAdapter(dicts).chunks()
        texts = " ".join(c.text for c in chunks)
        assert "is" in texts.lower()

    def test_list_values(self):
        dicts = [{"name": "Party", "members": ["Alice", "Bob", "Charlie"]}]
        chunks = DictAdapter(dicts).chunks()
        assert len(chunks) == 1
        assert "Alice" in chunks[0].text

    def test_chunk_ids_unique_across_dicts(self):
        dicts = [{"name": f"c{i}", "val": i} for i in range(10)]
        chunks = DictAdapter(dicts).chunks()
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))


class TestFileAdapter:
    def test_reads_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test document. " * 30)
            path = f.name

        chunks = FileAdapter(path, chunk_size=100).chunks()
        assert len(chunks) > 0
        assert all("test" in c.text.lower() for c in chunks)

    def test_source_id_is_path(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Sample content.")
            path = f.name

        chunks = FileAdapter(path).chunks()
        assert all(c.source_id == path for c in chunks)
