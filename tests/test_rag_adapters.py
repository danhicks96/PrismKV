"""
test_rag_adapters.py — Tests for TextAdapter, DictAdapter, FileAdapter, ChatGPTExportAdapter.
"""

import json
import tempfile
import pytest
from prismkv.rag.adapters import TextAdapter, DictAdapter, FileAdapter, ChatGPTExportAdapter
from prismkv.rag.schema import Chunk


def _make_chatgpt_export(conversations: list) -> str:
    """Write a ChatGPT export fixture to a temp file; return path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(conversations, f)
    f.close()
    return f.name


def _simple_conv(title: str, turns: list[tuple[str, str]], create_time: int = 0) -> dict:
    """Build a minimal ChatGPT export conversation dict from (user, assistant) pairs."""
    mapping = {}
    prev = None
    for i, (role, text) in enumerate(turns):
        nid = f"node_{i}"
        mapping[nid] = {
            "id": nid,
            "parent": prev,
            "children": [f"node_{i+1}"] if i + 1 < len(turns) else [],
            "message": {
                "author": {"role": role},
                "content": {"parts": [text]},
            },
        }
        prev = nid
    return {"title": title, "create_time": create_time, "mapping": mapping}


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


class TestChatGPTExportAdapter:
    def test_one_turn_pair_produces_one_chunk(self):
        conv = _simple_conv("Test convo", [("user", "Hello"), ("assistant", "Hi there")])
        path = _make_chatgpt_export([conv])
        chunks = ChatGPTExportAdapter(path).chunks()
        assert len(chunks) == 1
        assert "Hello" in chunks[0].text
        assert "Hi there" in chunks[0].text

    def test_multiple_turns_produce_multiple_chunks(self):
        conv = _simple_conv("Multi-turn", [
            ("user", "Q1"), ("assistant", "A1"),
            ("user", "Q2"), ("assistant", "A2"),
            ("user", "Q3"), ("assistant", "A3"),
        ])
        path = _make_chatgpt_export([conv])
        chunks = ChatGPTExportAdapter(path).chunks()
        assert len(chunks) == 3

    def test_title_in_chunk_text(self):
        conv = _simple_conv("My Conversation Title", [("user", "question"), ("assistant", "answer")])
        path = _make_chatgpt_export([conv])
        chunks = ChatGPTExportAdapter(path).chunks()
        assert "My Conversation Title" in chunks[0].text

    def test_metadata_contains_create_time(self):
        conv = _simple_conv("Timed", [("user", "q"), ("assistant", "a")], create_time=1234567890)
        path = _make_chatgpt_export([conv])
        chunks = ChatGPTExportAdapter(path).chunks()
        assert chunks[0].metadata["create_time"] == 1234567890

    def test_multiple_conversations(self):
        convs = [
            _simple_conv(f"Conv {i}", [("user", f"q{i}"), ("assistant", f"a{i}")])
            for i in range(5)
        ]
        path = _make_chatgpt_export(convs)
        chunks = ChatGPTExportAdapter(path).chunks()
        assert len(chunks) == 5

    def test_chunk_ids_unique(self):
        convs = [
            _simple_conv(f"Conv {i}", [("user", f"unique question {i}"), ("assistant", f"answer {i}")])
            for i in range(10)
        ]
        path = _make_chatgpt_export(convs)
        chunks = ChatGPTExportAdapter(path).chunks()
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_unpaired_user_turn_skipped(self):
        """A trailing user message without an assistant reply is not emitted."""
        conv = _simple_conv("Incomplete", [
            ("user", "Q1"), ("assistant", "A1"),
            ("user", "dangling question"),
        ])
        path = _make_chatgpt_export([conv])
        chunks = ChatGPTExportAdapter(path).chunks()
        assert len(chunks) == 1

    def test_empty_export_no_chunks(self):
        path = _make_chatgpt_export([])
        chunks = ChatGPTExportAdapter(path).chunks()
        assert chunks == []

    def test_extra_metadata_propagated(self):
        conv = _simple_conv("Meta test", [("user", "q"), ("assistant", "a")])
        path = _make_chatgpt_export([conv])
        chunks = ChatGPTExportAdapter(path, metadata={"project": "test_proj"}).chunks()
        assert chunks[0].metadata["project"] == "test_proj"
