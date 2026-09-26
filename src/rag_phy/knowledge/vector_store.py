"""Persistent Chroma vector store and typed retrieval records."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from rag_phy.ingestion.chunking import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    """Raised when vector indexing or retrieval fails."""


@dataclass(frozen=True)
class RetrievalResult:
    """Retrieved passage with cosine distance and all known source references."""

    chunk_id: str
    text: str
    distance: float
    source_refs: tuple[str, ...]


class VectorStore(Protocol):
    """Interface for vector indexing and nearest-neighbor retrieval."""

    def upsert(
        self, chunks: Sequence[DocumentChunk], embeddings: Sequence[Sequence[float]]
    ) -> None:
        """Insert or update chunks with their corresponding embeddings."""

    def search(self, query_embedding: Sequence[float], limit: int) -> list[RetrievalResult]:
        """Return nearest passages ordered by ascending cosine distance."""


class ChromaVectorStore:
    """Persistent Chroma collection using cosine-distance HNSW indexing."""

    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str,
        client: Any | None = None,
    ) -> None:
        """Open a persistent collection or accept a Chroma-compatible client.

        Args:
            persist_directory: Configured on-disk Chroma directory.
            collection_name: Collection to create or open.
            client: Optional injected Chroma client, useful for isolated test stores.

        Raises:
            VectorStoreError: If Chroma cannot initialize the requested collection.
        """
        try:
            if client is None:
                import chromadb

                directory = Path(persist_directory)
                directory.mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(path=str(directory))
            self._collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            logger.exception(
                "Chroma collection initialization failed",
                extra={"collection": collection_name},
            )
            raise VectorStoreError(f"Could not initialize Chroma collection {collection_name}") from exc

    def upsert(
        self, chunks: Sequence[DocumentChunk], embeddings: Sequence[Sequence[float]]
    ) -> None:
        """Persist chunk text, content hashes, source references, and vectors."""
        if len(chunks) != len(embeddings):
            raise ValueError("Each document chunk must have exactly one embedding")
        if not chunks:
            return
        embedding_dimensions = {len(vector) for vector in embeddings}
        if len(embedding_dimensions) != 1 or 0 in embedding_dimensions:
            raise ValueError("All indexed embeddings must have one consistent non-zero dimension")
        try:
            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                embeddings=[list(vector) for vector in embeddings],
                metadatas=[
                    {
                        "source_refs_json": json.dumps(chunk.source_refs),
                        "ordinal": chunk.ordinal,
                    }
                    for chunk in chunks
                ],
            )
        except Exception as exc:
            logger.exception("Chroma upsert failed", extra={"chunk_count": len(chunks)})
            raise VectorStoreError("Could not write document embeddings to Chroma") from exc

    def search(self, query_embedding: Sequence[float], limit: int) -> list[RetrievalResult]:
        """Query Chroma's approximate nearest-neighbor index and decode sources."""
        if limit <= 0:
            raise ValueError("Retrieval limit must be positive")
        if not query_embedding or not all(math.isfinite(float(value)) for value in query_embedding):
            raise ValueError("Query embedding must contain finite values")
        try:
            if self._collection.count() == 0:
                return []
            result = self._collection.query(
                query_embeddings=[list(query_embedding)],
                n_results=limit,
                include=["documents", "distances", "metadatas"],
            )
        except Exception as exc:
            logger.exception("Chroma vector query failed")
            raise VectorStoreError("Could not query Chroma collection") from exc

        ids = result["ids"][0]
        documents = result["documents"][0]
        distances = result["distances"][0]
        metadata_rows = result["metadatas"][0]
        retrieved: list[RetrievalResult] = []
        for chunk_id, text, distance, metadata in zip(
            ids, documents, distances, metadata_rows, strict=True
        ):
            try:
                source_refs = tuple(json.loads(metadata["source_refs_json"]))
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise VectorStoreError(
                    f"Chroma record {chunk_id} has invalid source-reference metadata"
                ) from exc
            retrieved.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=text,
                    distance=float(distance),
                    source_refs=source_refs,
                )
            )
        return retrieved


__all__ = ["ChromaVectorStore", "RetrievalResult", "VectorStore", "VectorStoreError"]
