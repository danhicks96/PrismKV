"""
adapters.py — Data source adapters that produce Chunk objects for ingestion.

Adapters
--------
TextAdapter  : sliding-window chunking of plain text
DictAdapter  : flattens Python dicts (game state, structured data) into Chunks
FileAdapter  : reads a file and delegates to TextAdapter
APIAdapter   : fetches JSON from a REST endpoint and converts to Chunks
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Dict, Iterable, Iterator, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from prismkv.rag.schema import Chunk


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseAdapter:
    """Common interface — all adapters implement chunks()."""

    def chunks(self) -> List[Chunk]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# TextAdapter
# ---------------------------------------------------------------------------

class TextAdapter(BaseAdapter):
    """
    Split plain text into overlapping chunks of fixed character size.

    Parameters
    ----------
    text       : the source text
    chunk_size : target chunk length in characters (default 400)
    overlap    : overlap between consecutive chunks in characters (default 50)
    source_id  : optional provenance label stored in each Chunk.source_id
    metadata   : extra metadata applied to all chunks from this source
    """

    def __init__(
        self,
        text: str,
        chunk_size: int = 400,
        overlap: int = 50,
        source_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.text = text
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.source_id = source_id
        self._metadata = metadata or {}

    def chunks(self) -> List[Chunk]:
        text = self.text.strip()
        if not text:
            return []

        result = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            snippet = text[start:end].strip()
            if snippet:
                chunk_id = hashlib.sha256(
                    f"{self.source_id}:{idx}:{snippet}".encode()
                ).hexdigest()[:16]
                result.append(Chunk(
                    id=chunk_id,
                    text=snippet,
                    metadata={**self._metadata, "chunk_index": idx},
                    source_id=self.source_id,
                ))
            start += self.chunk_size - self.overlap
            idx += 1
        return result


# ---------------------------------------------------------------------------
# DictAdapter
# ---------------------------------------------------------------------------

class DictAdapter(BaseAdapter):
    """
    Convert a list of Python dicts into natural-language Chunks.

    Each key-value pair in each dict becomes a separate chunk sentence,
    enabling targeted retrieval by entity, field, or fact type.

    This is the primary adapter for the ``usurper-successor`` game use case:
    character state, world events, and faction data can be injected directly
    as dicts without any text pre-processing.

    Parameters
    ----------
    dicts      : list of dicts to ingest
    entity_key : key whose value is used as the entity name in sentences
                 (default "name"; if absent, "item_{i}" is used)
    source_id  : optional provenance label
    metadata   : base metadata applied to all chunks
    """

    def __init__(
        self,
        dicts: List[Dict[str, Any]],
        entity_key: str = "name",
        source_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.dicts = dicts
        self.entity_key = entity_key
        self.source_id = source_id
        self._metadata = metadata or {}

    def chunks(self) -> List[Chunk]:
        result = []
        for i, d in enumerate(self.dicts):
            entity = str(d.get(self.entity_key, f"item_{i}"))
            for key, value in d.items():
                if key == self.entity_key:
                    continue
                sentence = self._to_sentence(entity, key, value)
                if not sentence:
                    continue
                chunk_id = hashlib.sha256(
                    f"{self.source_id}:{entity}:{key}:{value}".encode()
                ).hexdigest()[:16]
                result.append(Chunk(
                    id=chunk_id,
                    text=sentence,
                    metadata={
                        **self._metadata,
                        "type": self.source_id or "dict",
                        "field": key,
                        "entity": entity,
                    },
                    source_id=self.source_id,
                ))
        return result

    @staticmethod
    def _to_sentence(entity: str, key: str, value: Any) -> str:
        """Convert a (entity, key, value) triple to a natural language sentence."""
        key_human = key.replace("_", " ")
        if isinstance(value, bool):
            pred = "is" if value else "is not"
            return f"{entity} {pred} {key_human}."
        if isinstance(value, (int, float)):
            return f"{entity} has {key_human} of {value}."
        if isinstance(value, list):
            if not value:
                return ""
            items = ", ".join(str(v) for v in value[:5])
            return f"{entity} {key_human}: {items}."
        val = str(value).strip()
        if not val:
            return ""
        return f"{entity} {key_human} is {val}."


# ---------------------------------------------------------------------------
# FileAdapter
# ---------------------------------------------------------------------------

class FileAdapter(BaseAdapter):
    """
    Read a text file and delegate to TextAdapter.

    Parameters
    ----------
    path       : path to the file
    encoding   : file encoding (default utf-8)
    chunk_size, overlap, metadata : passed through to TextAdapter
    """

    def __init__(
        self,
        path: str,
        encoding: str = "utf-8",
        chunk_size: int = 400,
        overlap: int = 50,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.path = path
        self.encoding = encoding
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._metadata = metadata or {}

    def chunks(self) -> List[Chunk]:
        with open(self.path, encoding=self.encoding) as f:
            text = f.read()
        return TextAdapter(
            text,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            source_id=self.path,
            metadata=self._metadata,
        ).chunks()


# ---------------------------------------------------------------------------
# APIAdapter
# ---------------------------------------------------------------------------

class APIAdapter(BaseAdapter):
    """
    Fetch data from a REST endpoint and ingest the JSON response as Chunks.

    Supports GET and POST requests.  The response must be valid JSON.
    The adapter traverses the JSON tree to extract string values as chunks.
    For list-of-dicts responses, set ``text_field`` to extract a specific
    field from each item as the chunk text.

    Parameters
    ----------
    url        : endpoint URL
    method     : "GET" (default) or "POST"
    payload    : dict payload for POST requests (JSON-encoded automatically)
    headers    : additional HTTP headers (merged with Content-Type/Accept)
    text_field : for list-of-dicts: field name to use as chunk text
    timeout    : request timeout in seconds (default 10)
    source_id  : optional provenance label stored in each Chunk
    metadata   : base metadata applied to all chunks

    Examples
    --------
    # Ingest paginated blog posts
    adapter = APIAdapter(
        "https://api.example.com/posts",
        text_field="body",
        source_id="blog",
    )
    engine.ingest(adapter)
    """

    def __init__(
        self,
        url: str,
        method: str = "GET",
        payload: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        text_field: Optional[str] = None,
        timeout: int = 10,
        source_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.url = url
        self.method = method.upper()
        self.payload = payload
        self.headers = headers or {}
        self.text_field = text_field
        self.timeout = timeout
        self.source_id = source_id or url
        self._metadata = metadata or {}

    def chunks(self) -> List[Chunk]:
        data = self._fetch()
        return list(self._parse(data))

    def _fetch(self) -> Any:
        """Perform the HTTP request and return parsed JSON."""
        body = json.dumps(self.payload).encode() if self.payload else None
        hdrs = {"Content-Type": "application/json", "Accept": "application/json"}
        hdrs.update(self.headers)

        req = Request(self.url, data=body, headers=hdrs, method=self.method)
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except (URLError, HTTPError) as exc:
            raise RuntimeError(
                f"APIAdapter: request to {self.url!r} failed: {exc}"
            ) from exc

    def _parse(self, data: Any, _depth: int = 0) -> Iterator[Chunk]:
        """Extract text chunks from arbitrary JSON structure (max depth 4)."""
        if _depth > 4:
            return

        if isinstance(data, list):
            for i, item in enumerate(data):
                if (isinstance(item, dict)
                        and self.text_field
                        and self.text_field in item):
                    text = str(item[self.text_field])
                    meta = {k: v for k, v in item.items() if k != self.text_field
                            and not isinstance(v, (dict, list))}
                    meta.update(self._metadata)
                    meta["source_id"] = self.source_id
                    meta["item_index"] = i
                    chunk_id = hashlib.sha256(
                        f"{self.source_id}:{i}:{text}".encode()
                    ).hexdigest()[:16]
                    yield Chunk(
                        id=chunk_id,
                        text=text,
                        metadata=meta,
                        source_id=self.source_id,
                    )
                elif isinstance(item, str) and item.strip():
                    chunk_id = hashlib.sha256(
                        f"{self.source_id}:{i}:{item}".encode()
                    ).hexdigest()[:16]
                    yield Chunk(
                        id=chunk_id,
                        text=item,
                        metadata={**self._metadata, "source_id": self.source_id,
                                  "item_index": i},
                        source_id=self.source_id,
                    )
                elif isinstance(item, (dict, list)):
                    yield from self._parse(item, _depth + 1)

        elif isinstance(data, dict):
            for path, value in self._flatten_dict(data):
                if isinstance(value, str) and value.strip():
                    chunk_id = hashlib.sha256(
                        f"{self.source_id}:{path}:{value}".encode()
                    ).hexdigest()[:16]
                    yield Chunk(
                        id=chunk_id,
                        text=value,
                        metadata={**self._metadata, "source_id": self.source_id,
                                  "path": path},
                        source_id=self.source_id,
                    )

        elif isinstance(data, str) and data.strip():
            chunk_id = hashlib.sha256(
                f"{self.source_id}:{data}".encode()
            ).hexdigest()[:16]
            yield Chunk(
                id=chunk_id,
                text=data,
                metadata={**self._metadata, "source_id": self.source_id},
                source_id=self.source_id,
            )

    def _flatten_dict(
        self,
        d: Dict[str, Any],
        prefix: str = "",
    ) -> Iterator[tuple]:
        """Yield (dotted_path, value) for all scalar leaves of a dict."""
        for key, value in d.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (dict, list)):
                yield from self._flatten_dict(value if isinstance(value, dict)
                                               else {str(i): v for i, v in enumerate(value)},
                                               path)
            else:
                yield path, value
