"""Versioned, hand-labeled evaluation cases and answer-system contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class LabeledQACase(BaseModel):
    """One question with a reference answer and source IDs judged relevant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    relevant_source_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("relevant_source_ids")
    @classmethod
    def validate_source_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values) or len(set(values)) != len(values):
            raise ValueError("relevant_source_ids must be non-empty and unique")
        return values


class LabeledQADataset(BaseModel):
    """Immutable evaluation dataset revision; empty revisions are setup scaffolds only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_version: str = Field(min_length=1)
    source_corpus_version: str | None = None
    cases: tuple[LabeledQACase, ...]

    @field_validator("cases")
    @classmethod
    def validate_case_ids(cls, cases: tuple[LabeledQACase, ...]) -> tuple[LabeledQACase, ...]:
        ids = [case.id for case in cases]
        if len(set(ids)) != len(ids):
            raise ValueError("case IDs must be unique within a dataset")
        return cases


class RetrievedContext(BaseModel):
    """One retrieved passage and its stable corpus source identifier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class AnswerWithContexts(BaseModel):
    """System output required to evaluate answer quality and retrieval provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: str = Field(min_length=1)
    retrieved_contexts: tuple[RetrievedContext, ...]


class RAGEvaluationTarget(Protocol):
    """Application adapter that answers and returns exactly the contexts it retrieved."""

    def answer(self, question: str) -> AnswerWithContexts:
        """Return a generated answer and its retrieved source passages."""


def load_qa_dataset(path: str | Path) -> LabeledQADataset:
    """Load and validate a versioned QA artifact from JSON."""
    dataset_path = Path(path)
    try:
        with dataset_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        return LabeledQADataset.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"Could not load valid QA dataset at {dataset_path}: {exc}") from exc


__all__ = [
    "AnswerWithContexts",
    "LabeledQACase",
    "LabeledQADataset",
    "RAGEvaluationTarget",
    "RetrievedContext",
    "load_qa_dataset",
]
