#!/usr/bin/env python3
"""Export a blind, source-addressable QA review packet without exposing draft labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from rag_phy.evaluation.dataset import LabeledQADataset, load_qa_dataset

try:
    from scripts.validate_qa_dataset import validate_qa_dataset
except ModuleNotFoundError:
    from validate_qa_dataset import validate_qa_dataset

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data/evaluation/qa_set.json"
MANIFEST_PATH = ROOT / "data/curated/corpus/manifest.json"
DEFAULT_OUTPUT = ROOT / "data/evaluation/review/qa_review_packet.csv"
REVIEW_FIELDS = (
    "reviewed_answerable",
    "reviewed_reference_answer",
    "reviewed_relevant_source_ids",
    "searched_full_corpus",
    "reviewer_notes",
    "reviewer_name",
    "review_date_utc",
)
PACKET_FIELDS = (
    "dataset_version",
    "dataset_sha256",
    "source_corpus_version",
    "case_id",
    "question",
    "corpus_documents",
    *REVIEW_FIELDS,
)


def build_review_rows(
    dataset: LabeledQADataset,
    manifest: dict[str, Any],
    dataset_sha256: str,
) -> list[dict[str, str]]:
    """Build blinded rows with the corpus document references and empty review fields."""
    corpus_version = manifest.get("corpus_version")
    if not isinstance(corpus_version, str) or dataset.source_corpus_version != corpus_version:
        raise ValueError("QA dataset and corpus manifest versions do not match")
    documents = manifest.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("Corpus manifest must contain source documents")
    document_refs: list[dict[str, str]] = []
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("Corpus manifest contains a malformed document record")
        path = document.get("path")
        url = document.get("url")
        if not isinstance(path, str) or not isinstance(url, str):
            raise ValueError("Corpus documents require a repository path and public URL")
        document_refs.append({"repository_path": path, "official_url": url})
    sources_json = json.dumps(document_refs, ensure_ascii=False, sort_keys=True)
    rows = []
    for case in dataset.cases:
        row = {
            "dataset_version": dataset.dataset_version,
            "dataset_sha256": dataset_sha256,
            "source_corpus_version": corpus_version,
            "case_id": case.id,
            "question": case.question,
            "corpus_documents": sources_json,
        }
        row.update({field: "" for field in REVIEW_FIELDS})
        rows.append(row)
    return rows


def export_qa_review_packet(
    dataset_path: Path = DATASET_PATH,
    manifest_path: Path = MANIFEST_PATH,
    output_path: Path = DEFAULT_OUTPUT,
    root: Path = ROOT,
) -> Path:
    """Validate the draft set against the PDF, then write a reviewer-blinded CSV."""
    errors = validate_qa_dataset(dataset_path, manifest_path, root)
    if errors:
        raise ValueError("Cannot export QA review packet: " + "; ".join(errors))
    dataset = load_qa_dataset(dataset_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    rows = build_review_rows(dataset, manifest, digest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PACKET_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        output = export_qa_review_packet(args.dataset, args.manifest, args.output, ROOT)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"QA review packet export failed: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote blind review packet for {len(load_qa_dataset(args.dataset).cases)} cases: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
