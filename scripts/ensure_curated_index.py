"""Build the configured corpus index unless a matching completed index is recorded."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

try:
    from .build_curated_index import main as build_index
except ImportError:
    from build_curated_index import main as build_index

from rag_phy.config import ModelsConfig, load_models_config
from rag_phy.knowledge import ChromaVectorStore

ROOT = Path(__file__).resolve().parents[1]
CORPUS_MANIFEST = ROOT / "data/curated/corpus/manifest.json"


class _CountableStore(Protocol):
    def count(self) -> int:
        """Return the current number of indexed records."""


def ensure_curated_index(
    *,
    build: Callable[[], int] = build_index,
    store_factory: Callable[[Path, str], _CountableStore] | None = None,
    marker_path: Path | None = None,
) -> bool:
    """Build or reuse an index only when its corpus and embedding identity is known."""
    config = load_models_config(ROOT / "config/models.yaml")
    vector = config.vector_store
    make_store: Callable[[Path, str], _CountableStore] = store_factory or ChromaVectorStore
    persist_directory = vector.persist_directory
    if not persist_directory.is_absolute():
        persist_directory = ROOT / persist_directory
    current_marker = marker_path or persist_directory / "rag_phy_index.json"
    identity = _index_identity(config)
    store = make_store(persist_directory, vector.collection_name)
    marker_exists = current_marker.exists()
    if marker_exists:
        try:
            recorded_identity = json.loads(current_marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read vector index marker {current_marker}") from exc
        if recorded_identity != identity:
            raise RuntimeError(
                "The persistent vector index belongs to a different corpus or indexing "
                "configuration; "
                "recreate the vector_data volume before starting this image"
            )
        if store.count() > 0:
            return False

    if build() != 0:
        raise RuntimeError("Could not build the configured curated corpus index")
    if make_store(persist_directory, vector.collection_name).count() == 0:
        raise RuntimeError("Index build completed but the configured collection is still empty")
    current_marker.parent.mkdir(parents=True, exist_ok=True)
    temporary_marker = current_marker.with_suffix(f"{current_marker.suffix}.tmp")
    temporary_marker.write_text(
        json.dumps(identity, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_marker.replace(current_marker)
    return True


def _index_identity(config: ModelsConfig) -> dict[str, object]:
    manifest = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    for document in manifest["documents"]:
        source_path = ROOT / document["path"]
        actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if actual_hash != document["sha256"]:
            raise RuntimeError(f"Corpus checksum mismatch for {document['path']}")
    embedding = config.embedding
    vector = config.vector_store
    return {
        "corpus_version": manifest["corpus_version"],
        "documents": manifest["documents"],
        "chunking": config.chunking.model_dump(mode="json"),
        "embedding": {
            "provider": embedding.provider,
            "model_name": embedding.model_name,
            "expected_dimension": embedding.expected_dimension,
            "max_sequence_tokens": embedding.max_sequence_tokens,
            "normalize_embeddings": embedding.normalize_embeddings,
            "query_instruction": embedding.query_instruction,
        },
        "vector_store": {
            "provider": vector.provider,
            "collection_name": vector.collection_name,
        },
    }


if __name__ == "__main__":
    import os

    os.chdir(ROOT)
    indexed = ensure_curated_index()
    print("Built the configured corpus index." if indexed else "Using the existing corpus index.")
    sys.exit(0)
