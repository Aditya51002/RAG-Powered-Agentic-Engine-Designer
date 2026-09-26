#!/usr/bin/env python3
"""Build the configured persistent vector index from the checksummed corpus manifest."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from validate_qa_dataset import validate_qa_dataset

from rag_phy.config import load_models_config
from rag_phy.ingestion import DocumentLoader, IngestionPipeline, SentenceTransformerEmbedder
from rag_phy.knowledge.vector_store import ChromaVectorStore

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data/curated/corpus/manifest.json"


def main() -> int:
    errors = validate_qa_dataset()
    if errors:
        print("Cannot build index from an invalid QA/corpus manifest:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    os.chdir(ROOT)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    config = load_models_config(ROOT / "config/models.yaml")
    embedder = SentenceTransformerEmbedder(config.embedding)
    vector_store = ChromaVectorStore(
        config.vector_store.persist_directory,
        config.vector_store.collection_name,
    )
    pipeline = IngestionPipeline(config, DocumentLoader(), embedder, vector_store)
    report = pipeline.ingest([Path(document["path"]) for document in manifest["documents"]])
    print(
        f"Indexed corpus {manifest['corpus_version']}: "
        f"{report.source_documents_loaded} source pages, "
        f"{report.unique_chunks_indexed} chunks, "
        f"{report.duplicate_chunks_removed} duplicate chunks removed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
