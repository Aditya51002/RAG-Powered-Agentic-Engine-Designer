#!/usr/bin/env python3
"""Validate a completed blind QA review packet and report draft-label differences."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from rag_phy.evaluation.dataset import LabeledQACase, load_qa_dataset

try:
    from scripts.export_qa_review_packet import (
        PACKET_FIELDS,
        build_review_rows,
    )
    from scripts.validate_qa_dataset import (
        validate_qa_dataset,
        validate_source_references,
    )
except ModuleNotFoundError:
    from export_qa_review_packet import PACKET_FIELDS, build_review_rows
    from validate_qa_dataset import validate_qa_dataset, validate_source_references

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data/evaluation/qa_set.json"
MANIFEST_PATH = ROOT / "data/curated/corpus/manifest.json"
DEFAULT_PACKET = ROOT / "data/evaluation/review/qa_review_packet.csv"
DEFAULT_OUTPUT = ROOT / "reports/evaluation/qa_review_comparison.json"


def _normalized_answer(value: str) -> str:
    return " ".join(value.split()).casefold()


def _is_utc_iso8601(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    offset = parsed.utcoffset()
    return offset is not None and offset.total_seconds() == 0


def compare_qa_review(
    packet_path: Path = DEFAULT_PACKET,
    dataset_path: Path = DATASET_PATH,
    manifest_path: Path = MANIFEST_PATH,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Validate a completed packet and return a comparison report, without mutating labels."""
    errors = validate_qa_dataset(dataset_path, manifest_path, root)
    if errors:
        raise ValueError("Draft QA dataset is invalid: " + "; ".join(errors))
    dataset = load_qa_dataset(dataset_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    expected_rows = build_review_rows(dataset, manifest, dataset_digest)
    with packet_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != PACKET_FIELDS:
            raise ValueError("Review packet headers do not match the exported packet schema")
        rows = list(reader)
    if len(rows) != len(expected_rows):
        raise ValueError(f"Expected {len(expected_rows)} review rows, found {len(rows)}")

    expected_by_id = {row["case_id"]: row for row in expected_rows}
    reviewed_cases: list[LabeledQACase] = []
    reviewers: set[str] = set()
    review_dates: set[str] = set()
    for line, row in enumerate(rows, start=2):
        case_id = row.get("case_id", "")
        if case_id not in expected_by_id:
            raise ValueError(f"Row {line}: unexpected case ID {case_id!r}")
        if any(existing.id == case_id for existing in reviewed_cases):
            raise ValueError(f"Row {line}: duplicate case ID {case_id!r}")
        expected = expected_by_id[case_id]
        for field in PACKET_FIELDS[:6]:
            if row.get(field) != expected[field]:
                raise ValueError(f"Row {line}: packet metadata changed in {field}")

        answerable_text = row.get("reviewed_answerable", "").strip().casefold()
        if answerable_text not in {"true", "false"}:
            raise ValueError(f"Row {line}: reviewed_answerable must be true or false")
        answerable = answerable_text == "true"
        full_corpus = row.get("searched_full_corpus", "").strip().casefold()
        if full_corpus not in {"true", "false"}:
            raise ValueError(f"Row {line}: searched_full_corpus must be true or false")
        if not answerable and full_corpus != "true":
            raise ValueError(f"Row {line}: unanswerable judgments require a full-corpus search")

        raw_sources = row.get("reviewed_relevant_source_ids", "")
        sources = tuple(part.strip() for part in raw_sources.split(";") if part.strip())
        if raw_sources.strip() and len(sources) != len(raw_sources.split(";")):
            raise ValueError(f"Row {line}: source IDs must be separated by semicolons")
        reviewer = row.get("reviewer_name", "").strip()
        review_date = row.get("review_date_utc", "").strip()
        if not reviewer:
            raise ValueError(f"Row {line}: reviewer_name is required")
        if not _is_utc_iso8601(review_date):
            raise ValueError(f"Row {line}: review_date_utc must be an ISO 8601 UTC timestamp")
        reviewers.add(reviewer)
        review_dates.add(review_date)
        try:
            reviewed_cases.append(
                LabeledQACase(
                    id=case_id,
                    question=row["question"],
                    answerable=answerable,
                    reference_answer=row.get("reviewed_reference_answer", "").strip(),
                    relevant_source_ids=sources,
                )
            )
        except ValidationError as exc:
            raise ValueError(f"Row {line}: invalid reviewed label: {exc}") from exc

    reviewed_by_id = {case.id: case for case in reviewed_cases}
    if set(reviewed_by_id) != set(expected_by_id):
        raise ValueError("Review packet case IDs do not match the draft dataset")
    source_errors = validate_source_references(
        ((case.id, case.relevant_source_ids) for case in reviewed_cases),
        manifest,
        root,
    )
    if source_errors:
        raise ValueError("Reviewed source references are invalid: " + "; ".join(source_errors))

    draft_by_id = {case.id: case for case in dataset.cases}
    disagreements = []
    for case_id in (case.id for case in dataset.cases):
        draft = draft_by_id[case_id]
        reviewed = reviewed_by_id[case_id]
        differences = []
        if draft.answerable != reviewed.answerable:
            differences.append("answerability")
        if _normalized_answer(draft.reference_answer) != _normalized_answer(reviewed.reference_answer):
            differences.append("reference_answer")
        if set(draft.relevant_source_ids) != set(reviewed.relevant_source_ids):
            differences.append("relevant_source_ids")
        if differences:
            disagreements.append({"case_id": case_id, "differences": differences})

    unanswerable_count = sum(not case.answerable for case in reviewed_cases)
    review_complete = len(reviewed_cases) == len(dataset.cases)
    return {
        "report_version": "1.0",
        "dataset_version": dataset.dataset_version,
        "dataset_sha256": dataset_digest,
        "source_corpus_version": dataset.source_corpus_version,
        "review_complete": review_complete,
        "reviewers": sorted(reviewers),
        "review_dates_utc": sorted(review_dates),
        "case_count": len(reviewed_cases),
        "unanswerable_count": unanswerable_count,
        "benchmark_coverage_met": unanswerable_count >= 3,
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
        "requires_adjudication": bool(disagreements),
        "draft_label_status_after_comparison": dataset.label_review_status,
        "next_step": (
            "Adjudicate listed differences, update the dataset revision, then validate and obtain signoff."
            if disagreements
            else "Human signoff is still required before updating the dataset review status."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        report = compare_qa_review(args.packet, args.dataset, args.manifest, ROOT)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"QA review comparison failed: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote QA review comparison report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
