"""Deterministic text chunking and normalized content-hash deduplication."""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass, replace
from typing import Iterable

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

    def __init__(self, config: ChunkingConfig) -> None:
        """Create a chunker from typed character-window settings."""
        self._config = config

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
        step = self._config.chunk_size_characters - self._config.overlap_characters
        raw_chunk_count = 0
        for document in documents:
            text = document.text.strip()
            ordinal = 0
            start = 0
            while start < len(text):
                chunk_text = text[start : start + self._config.chunk_size_characters].strip()
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
                if start + self._config.chunk_size_characters >= len(text):
                    break
                start += step
        return list(unique_chunks.values()), raw_chunk_count


__all__ = ["DocumentChunk", "TextChunker", "normalized_content_hash"]
