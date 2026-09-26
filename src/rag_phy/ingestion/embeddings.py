"""Dependency-injected BGE sentence embeddings with strict output validation."""

from __future__ import annotations

import logging
import math
from typing import Any, Protocol, Sequence

from rag_phy.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider cannot produce valid vectors."""


class Embedder(Protocol):
    """Interface for document and query embedding providers."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one fixed-dimensional vector per source text."""

    def embed_query(self, text: str) -> list[float]:
        """Return a fixed-dimensional vector for one retrieval query."""


class SentenceTransformerEmbedder:
    """BGE Sentence-Transformers adapter; model loading is injectable for offline tests."""

    def __init__(self, config: EmbeddingConfig, model: Any | None = None) -> None:
        """Load or accept a model matching the configured embedding dimension."""
        self._config = config
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(config.model_name)
            except Exception as exc:
                logger.exception(
                    "Embedding model initialization failed",
                    extra={"model_name": config.model_name},
                )
                raise EmbeddingError(
                    f"Could not load sentence embedding model {config.model_name}"
                ) from exc
        self._model = model
        try:
            dimension_reader = getattr(model, "get_embedding_dimension", None)
            if dimension_reader is None:
                dimension_reader = model.get_sentence_embedding_dimension
            actual_dimension = dimension_reader()
            model_max_tokens = model.max_seq_length
            if config.max_sequence_tokens > model_max_tokens:
                raise EmbeddingError(
                    f"Configured token limit {config.max_sequence_tokens} exceeds model limit "
                    f"{model_max_tokens}"
                )
        except EmbeddingError:
            raise
        except Exception as exc:
            logger.exception(
                "Embedding model metadata lookup failed",
                extra={"model_name": config.model_name},
            )
            raise EmbeddingError("Could not inspect embedding model metadata") from exc
        if actual_dimension != config.expected_dimension:
            raise EmbeddingError(
                f"Model {config.model_name} produces {actual_dimension} dimensions; "
                f"configuration requires {config.expected_dimension}"
            )

    def _encode(self, texts: Sequence[str]) -> Any:
        """Call the model and wrap inference failures with model context."""
        try:
            for text in texts:
                token_count = len(
                    self._model.tokenizer.encode(
                        text, add_special_tokens=True, verbose=False
                    )
                )
                if token_count > self._config.max_sequence_tokens:
                    raise EmbeddingError(
                        f"Input has {token_count} tokens, exceeding configured model limit "
                        f"{self._config.max_sequence_tokens}"
                    )
            return self._model.encode(
                list(texts),
                batch_size=self._config.batch_size,
                show_progress_bar=False,
                normalize_embeddings=self._config.normalize_embeddings,
            )
        except EmbeddingError:
            raise
        except Exception as exc:
            logger.exception("Embedding inference failed", extra={"model_name": self._config.model_name})
            raise EmbeddingError(f"Embedding inference failed for {self._config.model_name}") from exc

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode documents without query instructions and validate vector shape."""
        if not texts:
            return []
        encoded = self._encode(texts)
        return self._validate_vectors(encoded, len(texts))

    def count_tokens(self, text: str) -> int:
        """Return tokenizer length including special tokens for chunk-boundary checks."""
        return len(
            self._model.tokenizer.encode(text, add_special_tokens=True, verbose=False)
        )

    def embed_query(self, text: str) -> list[float]:
        """Encode a query using the configured model-specific retrieval instruction."""
        query = f"{self._config.query_instruction}{text}"
        vectors = self._encode([query])
        return self._validate_vectors(vectors, 1)[0]

    def _validate_vectors(self, raw_vectors: Any, expected_count: int) -> list[list[float]]:
        """Validate provider output count, dimension, and finite numeric values."""
        if hasattr(raw_vectors, "tolist"):
            raw_vectors = raw_vectors.tolist()
        if expected_count == 1 and raw_vectors and isinstance(raw_vectors[0], (int, float)):
            raw_vectors = [raw_vectors]
        try:
            vectors = [[float(value) for value in vector] for vector in raw_vectors]
        except (TypeError, ValueError) as exc:
            raise EmbeddingError("Embedding provider returned a malformed vector batch") from exc
        if len(vectors) != expected_count:
            raise EmbeddingError(
                f"Embedding provider returned {len(vectors)} vectors for {expected_count} texts"
            )
        for vector in vectors:
            if len(vector) != self._config.expected_dimension:
                raise EmbeddingError(
                    f"Expected {self._config.expected_dimension}-dimensional embedding; "
                    f"received {len(vector)} dimensions"
                )
            if not all(math.isfinite(value) for value in vector):
                raise EmbeddingError("Embedding provider returned a non-finite vector value")
        return vectors


__all__ = ["Embedder", "EmbeddingError", "SentenceTransformerEmbedder"]
