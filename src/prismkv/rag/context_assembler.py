"""
context_assembler.py — Assemble retrieved chunks into a prompt context string.
"""

from __future__ import annotations

from typing import List, Optional

from prismkv.rag.schema import RetrievalResult


class ContextAssembler:
    """
    Assembles a list of RetrievalResult objects into a formatted context string.

    Parameters
    ----------
    max_tokens      : approximate token budget (characters / 4) for context
    separator       : string placed between chunks in the output
    include_scores  : if True, prepend relevance scores to each chunk
    sort_by_timestamp : if True (default), sort chunks by ingestion order
                        for narrative coherence; otherwise sort by score
    """

    def __init__(
        self,
        max_tokens: int = 1024,
        separator: str = "\n\n---\n\n",
        include_scores: bool = False,
        sort_by_timestamp: bool = True,
    ) -> None:
        self.max_tokens = max_tokens
        self.separator = separator
        self.include_scores = include_scores
        self.sort_by_timestamp = sort_by_timestamp

    def assemble(self, results: List[RetrievalResult]) -> str:
        """
        Produce a context string from a list of retrieval results.

        Chunks are trimmed to fit within max_tokens (approx).
        """
        if not results:
            return ""

        if self.sort_by_timestamp:
            sorted_results = sorted(results, key=lambda r: r.chunk.timestamp)
        else:
            sorted_results = sorted(results, key=lambda r: -r.score)

        parts = []
        char_budget = self.max_tokens * 4  # ~4 chars per token
        used = 0

        for r in sorted_results:
            text = r.chunk.text
            if self.include_scores:
                text = f"[{r.score:.3f}] {text}"
            chunk_chars = len(text) + len(self.separator)
            if used + chunk_chars > char_budget:
                # Try truncated version
                remaining = char_budget - used - len(self.separator)
                if remaining > 40:
                    parts.append(text[:remaining] + "…")
                break
            parts.append(text)
            used += chunk_chars

        return self.separator.join(parts)
