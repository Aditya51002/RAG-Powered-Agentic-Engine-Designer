#!/usr/bin/env python3
"""Measure source-ID retrieval against the draft QA set without generating answers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from rag_phy.config import load_evaluation_config, load_models_config
from rag_phy.evaluation.dataset import load_qa_dataset
from rag_phy.ingestion import SentenceTransformerEmbedder
from rag_phy.knowledge.vector_store import ChromaVectorStore

try:
    from scripts.validate_qa_dataset import validate_qa_dataset
except ModuleNotFoundError:
    from validate_qa_dataset import validate_qa_dataset


ROOT = Path(__file__).resolve().parents[1]


def calculate_source_metrics(
    answerability: Sequence[bool],
    relevant_sources: Sequence[set[str]],
    retrieved_sources: Sequence[set[str]],
) -> dict[str, float]:
    """Calculate macro source-ID precision/recall and empty retrieval on unanswerables."""
    if not answerability or not (
        len(answerability) == len(relevant_sources) == len(retrieved_sources)
    ):
        raise ValueError("Evaluation inputs must have equal, non-zero case counts")
    answerable_count = sum(answerability)
    if not answerable_count:
        raise ValueError("At least one answerable QA case is required")

    precision_sum = 0.0
    recall_sum = 0.0
    unanswerable_count = 0
    unanswerable_empty = 0
    for answerable, relevant, retrieved in zip(
        answerability, relevant_sources, retrieved_sources, strict=True
    ):
        overlap = len(relevant & retrieved)
        if answerable:
            precision_sum += overlap / len(retrieved) if retrieved else 0.0
            recall_sum += overlap / len(relevant)
        else:
            unanswerable_count += 1
            unanswerable_empty += int(not retrieved)
            precision_sum += float(not retrieved)
    return {
        "retrieval_precision_macro": precision_sum / len(answerability),
        "retrieval_recall_macro_answerable": recall_sum / answerable_count,
        "unanswerable_retrieval_empty_rate": (
            unanswerable_empty / unanswerable_count if unanswerable_count else 1.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/evaluation/retrieval_draft.json")
    args = parser.parse_args()

    errors = validate_qa_dataset()
    if errors:
        print("Cannot evaluate retrieval against an invalid QA/corpus manifest:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    os.chdir(ROOT)
    dataset_path = load_evaluation_config(ROOT / "config/evaluation.yaml").qa_dataset_path
    dataset = load_qa_dataset(dataset_path)
    config = load_models_config(ROOT / "config/models.yaml")
    embedder = SentenceTransformerEmbedder(config.embedding)
    vector_store = ChromaVectorStore(
        config.vector_store.persist_directory,
        config.vector_store.collection_name,
    )
    answerability: list[bool] = []
    relevant_sources: list[set[str]] = []
    retrieved_sources: list[set[str]] = []
    retrieved_case_count = 0
    for case in dataset.cases:
        results = vector_store.search(
            embedder.embed_query(case.question),
            config.vector_store.retrieval_candidates,
        )
        source_ids = {
            source_id for result in results for source_id in result.source_refs if source_id.strip()
        }
        retrieved_case_count += bool(source_ids)
        answerability.append(case.answerable)
        relevant_sources.append(set(case.relevant_source_ids))
        retrieved_sources.append(source_ids)
    if not retrieved_case_count:
        print("The configured vector collection returned no corpus source IDs.", file=sys.stderr)
        return 1

    manifest = json.loads((ROOT / "data/curated/corpus/manifest.json").read_text(encoding="utf-8"))
    source_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    report: Mapping[str, object] = {
        "evidence_type": "retrieval_only_no_generation_no_ragas",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_version": dataset.dataset_version,
        "dataset_sha256": source_hash,
        "label_review_status": dataset.label_review_status,
        "source_corpus_version": manifest["corpus_version"],
        "embedding_model": config.embedding.model_name,
        "vector_collection": config.vector_store.collection_name,
        "retrieval_candidates": config.vector_store.retrieval_candidates,
        "case_count": len(dataset.cases),
        "answerable_case_count": sum(answerability),
        "unanswerable_case_count": len(answerability) - sum(answerability),
        "cases_with_retrieved_sources": retrieved_case_count,
        **calculate_source_metrics(answerability, relevant_sources, retrieved_sources),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
