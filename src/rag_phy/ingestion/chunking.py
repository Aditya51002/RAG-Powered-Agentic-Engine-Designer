"""Deterministic text chunking and normalized content-hash deduplication."""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass, replace
from typing import Callable, Iterable

from rag_phy.config import ChunkingConfig
from rag_phy.ingestion.documents import SourceDocument


@dataclass(frozen=True)
class DocumentChunk:
    """A retrieval unit retaining all source references for duplicate passages."""

    chunk_id: str
    text: str
    source_refs: tuple[str, ...]
    ordinal: int


def normalized_content_hash(text: str) -> str:
    """Return SHA-256 for Unicode-normalized, case-folded, whitespace-collapsed text."""
    canonical = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class TextChunker:
    """Split parsed text into configured overlapping character windows."""

    def __init__(
        self,
        config: ChunkingConfig,
        token_counter: Callable[[str], int] | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Create configured character windows with an optional model-token ceiling."""
        if (token_counter is None) != (max_tokens is None):
            raise ValueError("A token counter and max_tokens must be provided together")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self._config = config
        self._token_counter = token_counter
        self._max_tokens = max_tokens

    def chunk(self, documents: Iterable[SourceDocument]) -> list[DocumentChunk]:
        """Split source documents and deduplicate normalized chunk contents.

        Args:
            documents: Parsed source documents in stable processing order.

        Returns:
            Unique chunks with duplicate provenance merged into source references.
        """
        chunks, _ = self.chunk_with_count(documents)
        return chunks

    def chunk_with_count(
        self, documents: Iterable[SourceDocument]
    ) -> tuple[list[DocumentChunk], int]:
        """Return unique chunks and the number of pre-deduplication windows."""
        unique_chunks: dict[str, DocumentChunk] = {}
        raw_chunk_count = 0
        for document in documents:
            text = document.text.strip()
            ordinal = 0
            start = 0
            while start < len(text):
                end = min(start + self._config.chunk_size_characters, len(text))
                if self._token_counter is not None and self._max_tokens is not None:
                    end = self._fit_token_window(text, start, end)
                chunk_text = text[start:end].strip()
                if chunk_text:
                    raw_chunk_count += 1
                    chunk_id = normalized_content_hash(chunk_text)
                    existing = unique_chunks.get(chunk_id)
                    if existing is None:
                        unique_chunks[chunk_id] = DocumentChunk(
                            chunk_id=chunk_id,
                            text=chunk_text,
                            source_refs=(document.source_ref,),
                            ordinal=ordinal,
                        )
                    elif document.source_ref not in existing.source_refs:
                        unique_chunks[chunk_id] = replace(
                            existing,
                            source_refs=existing.source_refs + (document.source_ref,),
                        )
                ordinal += 1
                if end >= len(text):
                    break
                overlap = min(self._config.overlap_characters, end - start - 1)
                start = max(start + 1, end - overlap)
        return list(unique_chunks.values()), raw_chunk_count

    def _fit_token_window(self, text: str, start: int, end: int) -> int:
        """Find the largest character window under the configured tokenizer limit."""
        if self._token_counter is None or self._max_tokens is None:
            return end
        if self._token_counter(text[start:end]) <= self._max_tokens:
            return end
        low = start + 1
        high = end
        while low < high:
            middle = (low + high + 1) // 2
            if self._token_counter(text[start:middle]) <= self._max_tokens:
                low = middle
            else:
                high = middle - 1
        if low == start or self._token_counter(text[start:low]) > self._max_tokens:
            raise ValueError("A single character exceeds the configured model token limit")
        return low


__all__ = ["DocumentChunk", "TextChunker", "normalized_content_hash"]
