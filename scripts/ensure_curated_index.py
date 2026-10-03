"""Build the configured corpus index only when its persistent collection is empty."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

try:
    from .build_curated_index import main as build_index
except ImportError:
    from build_curated_index import main as build_index

from rag_phy.config import load_models_config
from rag_phy.knowledge import ChromaVectorStore


class _CountableStore(Protocol):
    def count(self) -> int:
        """Return the current number of indexed records."""


def ensure_curated_index(
    *,
    build: Callable[[], int] = build_index,
    store_factory: Callable[[Path, str], _CountableStore] | None = None,
) -> bool:
    """Return whether indexing was performed; propagate failures instead of masking them."""
    config = load_models_config("config/models.yaml")
    vector = config.vector_store
    make_store: Callable[[Path, str], _CountableStore] = store_factory or ChromaVectorStore
    store = make_store(vector.persist_directory, vector.collection_name)
    if store.count() > 0:
        return False

    if build() != 0:
        raise RuntimeError("Could not build the configured curated corpus index")
    if make_store(vector.persist_directory, vector.collection_name).count() == 0:
        raise RuntimeError("Index build completed but the configured collection is still empty")
    return True


if __name__ == "__main__":
    indexed = ensure_curated_index()
    print("Built the configured corpus index." if indexed else "Using the existing corpus index.")
    sys.exit(0)
