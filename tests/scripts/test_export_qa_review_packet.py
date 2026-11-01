"""Checks that the QA reviewer packet preserves provenance without revealing draft labels."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from rag_phy.evaluation.dataset import load_qa_dataset
from scripts.export_qa_review_packet import (
    PACKET_FIELDS,
    REVIEW_FIELDS,
    build_review_rows,
    export_qa_review_packet,
)


def test_review_rows_are_blinded_and_tied_to_dataset_and_corpus() -> None:
    root = Path(__file__).parents[2]
    dataset_path = root / "data/evaluation/qa_set.json"
    manifest = json.loads((root / "data/curated/corpus/manifest.json").read_text())
    dataset = load_qa_dataset(dataset_path)
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()

    rows = build_review_rows(dataset, manifest, digest)

    assert len(rows) == 18
    assert tuple(rows[0]) == PACKET_FIELDS
    assert rows[0]["dataset_sha256"] == digest
    assert rows[0]["source_corpus_version"] == manifest["corpus_version"]
    assert "answerable" not in rows[0]
    assert "reference_answer" not in rows[0]
    assert "relevant_source_ids" not in rows[0]
    assert all(rows[0][field] == "" for field in REVIEW_FIELDS)
    assert "official_url" in rows[0]["corpus_documents"]


def test_exported_packet_has_one_blind_row_per_case(tmp_path: Path) -> None:
    output = export_qa_review_packet(output_path=tmp_path / "packet.csv")

    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 18
    assert all(row["reviewed_answerable"] == "" for row in rows)
    assert all(row["reviewed_relevant_source_ids"] == "" for row in rows)
    assert all(row["searched_full_corpus"] == "" for row in rows)
    assert rows[0]["corpus_documents"]
