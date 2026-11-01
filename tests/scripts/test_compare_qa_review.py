"""Tests for validating and comparing independent QA review packets."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from rag_phy.evaluation.dataset import load_qa_dataset
from scripts.compare_qa_review import compare_qa_review
from scripts.export_qa_review_packet import PACKET_FIELDS, build_review_rows


def _completed_packet(tmp_path: Path, *, disagree: bool = False) -> tuple[Path, Path]:
    root = Path(__file__).parents[2]
    dataset_path = root / "data/evaluation/qa_set.json"
    dataset = load_qa_dataset(dataset_path)
    manifest = json.loads((root / "data/curated/corpus/manifest.json").read_text())
    rows = build_review_rows(
        dataset,
        manifest,
        hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
    )
    for row, case in zip(rows, dataset.cases, strict=True):
        row["reviewed_answerable"] = str(case.answerable).lower()
        row["reviewed_reference_answer"] = case.reference_answer
        row["reviewed_relevant_source_ids"] = ";".join(case.relevant_source_ids)
        row["searched_full_corpus"] = "true" if not case.answerable else "false"
        row["reviewer_name"] = "Independent Reviewer"
        row["review_date_utc"] = "2026-11-01T10:00:00Z"
    if disagree:
        rows[0]["reviewed_reference_answer"] += " Reviewer clarification."
    packet = tmp_path / "completed.csv"
    with packet.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PACKET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return packet, dataset_path


def test_matching_review_reports_complete_without_mutating_draft(tmp_path: Path) -> None:
    packet, dataset_path = _completed_packet(tmp_path)
    before = hashlib.sha256(dataset_path.read_bytes()).hexdigest()

    report = compare_qa_review(packet)

    assert report["review_complete"] is True
    assert report["disagreement_count"] == 0
    assert report["draft_label_status_after_comparison"] == "draft"
    assert report["benchmark_coverage_met"] is True
    assert hashlib.sha256(dataset_path.read_bytes()).hexdigest() == before


def test_review_differences_are_reported_for_adjudication(tmp_path: Path) -> None:
    packet, _ = _completed_packet(tmp_path, disagree=True)

    report = compare_qa_review(packet)

    assert report["requires_adjudication"] is True
    assert report["disagreements"] == [
        {"case_id": "nasa-cmc-001", "differences": ["reference_answer"]}
    ]
    assert report["draft_label_status_after_comparison"] == "draft"


def test_changed_packet_question_is_rejected(tmp_path: Path) -> None:
    packet, _ = _completed_packet(tmp_path)
    with packet.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["question"] += " changed"
    with packet.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PACKET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="metadata changed in question"):
        compare_qa_review(packet)


def test_unanswerable_judgment_requires_full_corpus_search(tmp_path: Path) -> None:
    packet, _ = _completed_packet(tmp_path)
    with packet.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    row = next(row for row in rows if row["reviewed_answerable"] == "false")
    row["searched_full_corpus"] = "false"
    with packet.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PACKET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="full-corpus search"):
        compare_qa_review(packet)
