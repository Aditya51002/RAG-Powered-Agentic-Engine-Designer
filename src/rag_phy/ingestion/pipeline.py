"""Document ingestion and retrieval orchestration with injected adapters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from rag_phy.config import ModelsConfig
from rag_phy.ingestion.chunking import TextChunker
from rag_phy.ingestion.documents import DocumentLoader
from rag_phy.ingestion.embeddings import Embedder

if TYPE_CHECKING:
    from rag_phy.knowledge.vector_store import RetrievalResult, VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionReport:
    """Summary of one document indexing operation."""

    source_documents_loaded: int
    unique_chunks_indexed: int
    duplicate_chunks_removed: int


class IngestionPipeline:
    """Load, chunk, deduplicate, embed, and write sourced documents."""

    def __init__(
        self,
        config: ModelsConfig,
        loader: DocumentLoader,
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """Bind typed settings and swappable document, embedding, and vector adapters."""
        self._config = config
        self._loader = loader
        self._embedder = embedder
        self._vector_store = vector_store
        self._chunker = TextChunker(config.chunking)

    def ingest(self, paths: Sequence[str | Path]) -> IngestionReport:
        """Parse and index source files, deduplicating before embedding.

        Args:
            paths: Ordered source file paths to ingest.

        Returns:
            Counts for loaded source units, indexed chunks, and duplicates skipped.
        """
        if not paths:
            raise ValueError("At least one document source path is required")
        documents = [document for path in paths for document in self._loader.load(path)]
        chunks, raw_chunk_count = self._chunker.chunk_with_count(documents)
        embeddings = self._embedder.embed_documents([chunk.text for chunk in chunks])
        self._vector_store.upsert(chunks, embeddings)
        report = IngestionReport(
            source_documents_loaded=len(documents),
            unique_chunks_indexed=len(chunks),
            duplicate_chunks_removed=raw_chunk_count - len(chunks),
        )
        logger.info(
            "Document ingestion completed",
            extra={
                "source_documents_loaded": report.source_documents_loaded,
                "unique_chunks_indexed": report.unique_chunks_indexed,
                "duplicate_chunks_removed": report.duplicate_chunks_removed,
            },
        )
        return report

    def retrieve(self, query: str, limit: int | None = None) -> list[RetrievalResult]:
        """Embed a query and return nearest passages from the configured vector store."""
        if not query.strip():
            raise ValueError("Retrieval query must not be empty")
        result_limit = limit if limit is not None else self._config.vector_store.retrieval_candidates
        if result_limit <= 0:
            raise ValueError("Retrieval limit must be positive")
        query_embedding = self._embedder.embed_query(query)
        return self._vector_store.search(query_embedding, result_limit)

__all__ = ["IngestionPipeline", "IngestionReport"]
