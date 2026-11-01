"""Offline tests for auditable, per-case retrieval evidence."""

from __future__ import annotations

import pytest

from rag_phy.evaluation.dataset import LabeledQACase
from rag_phy.knowledge.vector_store import RetrievalResult
from scripts.evaluate_retrieval import serialize_case_results


def test_serialized_retrieval_records_keep_source_ids_and_distances() -> None:
    cases = (
        LabeledQACase(
            id="answerable",
            question="Where is the reported value?",
            answerable=True,
            reference_answer="It is reported.",
            relevant_source_ids=("report#page=1",),
        ),
        LabeledQACase(
            id="unanswerable",
            question="What is not in the report?",
            answerable=False,
            reference_answer="Unanswerable",
        ),
    )
    retrieved = (
        (
            RetrievalResult("chunk-1", "matching passage", 0.12, ("report#page=1",)),
            RetrievalResult("chunk-2", "other passage", 0.31, ("report#page=2",)),
        ),
        (RetrievalResult("chunk-3", "unrelated passage", 0.44, ("report#page=3",)),),
    )

    records = serialize_case_results(cases, retrieved)

    assert records[0]["source_precision"] == 0.5
    assert records[0]["source_recall"] == 1.0
    assert records[0]["retrieved_source_ids"] == ["report#page=1", "report#page=2"]
    assert records[0]["retrieved_chunks"][0] == {
        "chunk_id": "chunk-1",
        "source_ids": ["report#page=1"],
        "cosine_distance": 0.12,
    }
    assert records[1]["answerable"] is False
    assert records[1]["retrieval_empty"] is False
    assert records[1]["source_precision"] is None
    assert records[1]["source_recall"] is None


def test_serialization_rejects_mismatched_case_and_result_counts() -> None:
    case = LabeledQACase(
        id="answerable",
        question="Where is the reported value?",
        answerable=True,
        reference_answer="It is reported.",
        relevant_source_ids=("report#page=1",),
    )

    with pytest.raises(ValueError, match="exactly one retrieval result row"):
        serialize_case_results((case,), ())
