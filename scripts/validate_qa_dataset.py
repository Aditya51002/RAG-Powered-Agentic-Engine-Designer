#!/usr/bin/env python3
"""Validate QA-set provenance against the checksummed, page-addressable corpus."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from rag_phy.evaluation.dataset import load_qa_dataset

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data/evaluation/qa_set.json"
MANIFEST_PATH = ROOT / "data/curated/corpus/manifest.json"


def validate_qa_dataset(
    dataset_path: Path = DATASET_PATH,
    manifest_path: Path = MANIFEST_PATH,
    root: Path = ROOT,
) -> list[str]:
    """Return schema, corpus checksum, source-page, and benchmark-size errors."""
    errors: list[str] = []
    try:
        dataset = load_qa_dataset(dataset_path)
        manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError, ValidationError) as exc:
        return [f"Could not load QA dataset or corpus manifest: {exc}"]

    if not 15 <= len(dataset.cases) <= 25:
        errors.append(f"Expected 15-25 QA cases, found {len(dataset.cases)}")
    unanswerable_count = sum(not case.answerable for case in dataset.cases)
    if unanswerable_count < 3:
        errors.append(f"Expected at least 3 unanswerable cases, found {unanswerable_count}")
    if not dataset.source_corpus_version or dataset.source_corpus_version != manifest.get(
        "corpus_version"
    ):
        errors.append("QA source_corpus_version does not match the corpus manifest")

    page_counts: dict[str, int] = {}
    manifest_docs = manifest.get("documents")
    if not isinstance(manifest_docs, list) or not manifest_docs:
        return errors + ["Corpus manifest must contain at least one document"]
    try:
        import pymupdf
    except ImportError:
        return errors + ["PDF page validation requires the optional PyMuPDF dependency"]

    for document in manifest_docs:
        relative_path = document.get("path", "")
        expected_hash = document.get("sha256", "")
        prefix = document.get("source_id_prefix", "")
        path = root / relative_path
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected_hash:
                errors.append(f"Corpus checksum mismatch for {relative_path}")
            with pymupdf.open(path) as pdf:
                page_counts[prefix] = len(pdf)
        except (OSError, ValueError) as exc:
            errors.append(f"Cannot validate corpus document {relative_path}: {exc}")

    for case in dataset.cases:
        for source_id in case.relevant_source_ids:
            matching_prefix = next(
                (prefix for prefix in page_counts if source_id.startswith(f"{prefix}#page=")),
                None,
            )
            if matching_prefix is None:
                errors.append(f"Case {case.id}: source ID does not resolve to the corpus: {source_id}")
                continue
            try:
                page_number = int(source_id.removeprefix(f"{matching_prefix}#page="))
                if not 1 <= page_number <= page_counts[matching_prefix]:
                    raise ValueError
            except ValueError:
                errors.append(f"Case {case.id}: source page is outside the corpus: {source_id}")
    return errors


def main() -> int:
    errors = validate_qa_dataset()
    if errors:
        print("QA dataset validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    dataset = load_qa_dataset(DATASET_PATH)
    suffix = " Independent label review is still pending." if dataset.label_review_status == "draft" else ""
    print(
        f"Validated {len(dataset.cases)} QA cases against the checksummed corpus; "
        f"{sum(not case.answerable for case in dataset.cases)} are unanswerable."
        f"{suffix}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
