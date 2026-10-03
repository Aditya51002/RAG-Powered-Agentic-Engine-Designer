"""Lazy configured knowledge retrieval adapter for the public HTTP API."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

from rag_phy.config import EmbeddingConfig, ModelsConfig, load_models_config
from rag_phy.ingestion import (
    DocumentLoader,
    Embedder,
    IngestionPipeline,
    SentenceTransformerEmbedder,
)
from rag_phy.knowledge import ChromaVectorStore, VectorStore

if TYPE_CHECKING:
    from rag_phy.api.service import KnowledgeSearchResult


class ConfiguredKnowledgeSearch:
    """Lazily compose the configured embedder, Chroma store, and shared retrieval pipeline."""

    def __init__(
        self,
        models_config_path: str | Path = "config/models.yaml",
        *,
        embedder_factory: Callable[[EmbeddingConfig], Embedder] = SentenceTransformerEmbedder,
        vector_store_factory: Callable[[Path, str], VectorStore] = ChromaVectorStore,
    ) -> None:
        self._models_config_path = Path(models_config_path)
        self._embedder_factory = embedder_factory
        self._vector_store_factory = vector_store_factory
        self._pipeline: IngestionPipeline | None = None
        self._initialization_lock = Lock()

    def __call__(self, query: str) -> Sequence[KnowledgeSearchResult]:
        """Retrieve configured top-k passages with their exact source references."""
        return self._get_pipeline().retrieve(query)

    def _get_pipeline(self) -> IngestionPipeline:
        if self._pipeline is not None:
            return self._pipeline
        with self._initialization_lock:
            if self._pipeline is None:
                config = load_models_config(self._models_config_path)
                embedder = self._create_embedder(config)
                vector_store = self._create_vector_store(config)
                self._pipeline = IngestionPipeline(
                    config,
                    DocumentLoader(),
                    embedder,
                    vector_store,
                )
            return self._pipeline

    def _create_embedder(self, config: ModelsConfig) -> Embedder:
        if config.embedding.provider != "sentence-transformers":
            raise ValueError(
                f"Knowledge search does not support embedder {config.embedding.provider!r}"
            )
        return self._embedder_factory(config.embedding)

    def _create_vector_store(self, config: ModelsConfig) -> VectorStore:
        vector_config = config.vector_store
        if vector_config.provider != "chroma":
            raise ValueError(
                f"Knowledge search does not support vector store {vector_config.provider!r}"
            )
        return self._vector_store_factory(
            vector_config.persist_directory,
            vector_config.collection_name,
        )


__all__ = ["ConfiguredKnowledgeSearch"]
