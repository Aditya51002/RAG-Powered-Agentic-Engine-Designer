"""RAGAS-backed answer evaluation with exact source-ID retrieval metrics."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Protocol

from pydantic import BaseModel, ConfigDict, Field

from rag_phy.evaluation.dataset import LabeledQADataset, RAGEvaluationTarget

logger = logging.getLogger(__name__)


class RagasBackend(Protocol):
    """Metric adapter; injectable so orchestration/tests need no provider credentials."""

    def evaluate(self, samples: list[dict[str, Any]]) -> Mapping[str, float]:
        """Evaluate standard RAGAS sample dictionaries and return aggregate scores."""


class EvaluationReport(BaseModel):
    """Aggregate retrieval and answer metric report tied to a dataset revision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_version: str
    source_corpus_version: str | None
    label_review_status: str
    case_count: int = Field(gt=0)
    answerable_case_count: int = Field(gt=0)
    unanswerable_retrieval_empty_rate: float = Field(ge=0, le=1)
    unanswerable_abstention_rate: float = Field(ge=0, le=1)
    retrieval_precision: float = Field(ge=0, le=1)
    retrieval_recall: float = Field(ge=0, le=1)
    ragas_scores: dict[str, float]


class ConfiguredRagasBackend:
    """Optional RAGAS adapter using configured collections metrics and injected evaluator LLM."""

    _METRICS: ClassVar[dict[str, str]] = {
        "context_precision": "ContextPrecision",
        "context_recall": "ContextRecall",
        "faithfulness": "Faithfulness",
    }

    def __init__(self, metric_names: Sequence[str], llm: Any = None, embeddings: Any = None) -> None:
        unknown = set(metric_names) - self._METRICS.keys()
        if unknown:
            raise ValueError(f"Unsupported RAGAS metric names: {sorted(unknown)}")
        if not metric_names:
            raise ValueError("At least one RAGAS metric is required")
        self._metric_names = tuple(metric_names)
        self._llm = llm
        self._embeddings = embeddings

    def evaluate(self, samples: list[dict[str, Any]]) -> Mapping[str, float]:
        """Invoke RAGAS lazily, keeping its optional dependency out of core installs."""
        try:
            from ragas import EvaluationDataset, evaluate
            from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness
        except ImportError as exc:
            raise RuntimeError("Install rag-phy[evaluation] to run RAGAS metrics") from exc

        metric_types = {
            "context_precision": ContextPrecision,
            "context_recall": ContextRecall,
            "faithfulness": Faithfulness,
        }
        metrics = [metric_types[name](llm=self._llm) for name in self._metric_names]
        dataset = EvaluationDataset.from_list(samples)
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self._llm,
            embeddings=self._embeddings,
            raise_exceptions=True,
            show_progress=False,
        )
        frame = result.to_pandas()
        scores = frame.mean(numeric_only=True).to_dict()
        return {str(name): float(value) for name, value in scores.items()}


def evaluate_dataset(
    dataset: LabeledQADataset,
    target: RAGEvaluationTarget,
    backend: RagasBackend,
) -> EvaluationReport:
    """Run reference-based RAGAS metrics and macro source-ID precision/recall."""
    if not dataset.cases:
        raise ValueError("Cannot evaluate an empty QA dataset; add reviewed labeled cases first")
    samples: list[dict[str, Any]] = []
    precision_sum = 0.0
    recall_sum = 0.0
    unanswerable_empty_count = 0
    unanswerable_abstention_count = 0
    unanswerable_count = 0
    for case in dataset.cases:
        output = target.answer(case.question)
        if case.answerable:
            samples.append(
                {
                    "user_input": case.question,
                    "retrieved_contexts": [context.text for context in output.retrieved_contexts],
                    "response": output.answer,
                    "reference": case.reference_answer,
                }
            )
        relevant = set(case.relevant_source_ids)
        retrieved = {context.source_id for context in output.retrieved_contexts}
        if case.answerable:
            precision_sum += len(relevant & retrieved) / len(retrieved) if retrieved else 0.0
            recall_sum += len(relevant & retrieved) / len(relevant)
        else:
            precision_sum += float(not retrieved)
            unanswerable_count += 1
            unanswerable_empty_count += int(not retrieved)
            unanswerable_abstention_count += int(output.abstained)

    if not samples:
        raise ValueError("RAGAS scoring requires at least one answerable case")

    raw_scores = backend.evaluate(samples)
    scores: dict[str, float] = {}
    for name, raw_score in raw_scores.items():
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError(f"RAGAS metric {name!r} returned a non-finite score")
        scores[name] = score
    count = len(dataset.cases)
    answerable_count = sum(case.answerable for case in dataset.cases)
    return EvaluationReport(
        dataset_version=dataset.dataset_version,
        source_corpus_version=dataset.source_corpus_version,
        label_review_status=dataset.label_review_status,
        case_count=count,
        answerable_case_count=answerable_count,
        unanswerable_retrieval_empty_rate=(
            unanswerable_empty_count / unanswerable_count if unanswerable_count else 1.0
        ),
        unanswerable_abstention_rate=(
            unanswerable_abstention_count / unanswerable_count if unanswerable_count else 1.0
        ),
        retrieval_precision=precision_sum / count,
        retrieval_recall=recall_sum / answerable_count,
        ragas_scores=scores,
    )


def save_evaluation_report(report: EvaluationReport, path: str | Path) -> None:
    """Write a stable JSON evaluation artifact, creating its parent directory."""
    report_path = Path(path)
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError:
        logger.exception("Failed to write evaluation report", extra={"path": str(report_path)})
        raise


__all__ = [
    "ConfiguredRagasBackend",
    "EvaluationReport",
    "RagasBackend",
    "evaluate_dataset",
    "save_evaluation_report",
]
