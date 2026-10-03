"""Configuration and vector-store readiness probe for the HTTP service."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from rag_phy.config import load_config, load_models_config
from rag_phy.knowledge import ChromaVectorStore


class _ReadinessStore(Protocol):
    def check_ready(self) -> None:
        """Perform an operation that confirms the configured store is reachable."""


def make_readiness_check(
    *,
    app_config_path: str | Path = "config/app.yaml",
    models_config_path: str | Path = "config/models.yaml",
    vector_store_factory: Callable[[Path, str], _ReadinessStore] = ChromaVectorStore,
) -> Callable[[], bool]:
    """Build a repeatable probe over validated config and the configured Chroma collection."""

    def check() -> bool:
        load_config(app_config_path)
        models_config = load_models_config(models_config_path)
        vector_config = models_config.vector_store
        if vector_config.provider != "chroma":
            raise ValueError(
                f"Readiness probe does not support vector store {vector_config.provider!r}"
            )
        store = vector_store_factory(
            vector_config.persist_directory,
            vector_config.collection_name,
        )
        store.check_ready()
        return True

    return check


__all__ = ["make_readiness_check"]
