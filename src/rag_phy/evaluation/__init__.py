"""RAG answer evaluation, retrieval metrics, and optimization validity analysis."""

from rag_phy.evaluation.dataset import (
    AnswerWithContexts,
    LabeledQACase,
    LabeledQADataset,
    RetrievedContext,
    load_qa_dataset,
)
from rag_phy.evaluation.metrics import (
    ValidityObservation,
    ValidityRatePoint,
    cumulative_validity_rate,
    save_validity_chart,
    validity_observations_from_study,
)
from rag_phy.evaluation.ragas import (
    ConfiguredRagasBackend,
    EvaluationReport,
    evaluate_dataset,
    save_evaluation_report,
)

__all__ = [
    "AnswerWithContexts",
    "ConfiguredRagasBackend",
    "EvaluationReport",
    "LabeledQACase",
    "LabeledQADataset",
    "RetrievedContext",
    "ValidityObservation",
    "ValidityRatePoint",
    "cumulative_validity_rate",
    "evaluate_dataset",
    "load_qa_dataset",
    "save_evaluation_report",
    "save_validity_chart",
    "validity_observations_from_study",
]
