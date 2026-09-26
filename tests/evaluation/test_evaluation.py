"""Synthetic-only tests for evaluation contracts and metric calculations."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_phy.evaluation import (
    AnswerWithContexts,
    LabeledQACase,
    LabeledQADataset,
    RetrievedContext,
    ValidityObservation,
    cumulative_validity_rate,
    evaluate_dataset,
    load_qa_dataset,
    save_evaluation_report,
    save_validity_chart,
    validity_observations_from_study,
)
from rag_phy.evaluation.metrics import ValidityRatePoint
from rag_phy.evaluation.ragas import ConfiguredRagasBackend


class SyntheticTarget:
    def answer(self, question: str) -> AnswerWithContexts:
        assert question == "Synthetic question?"
        return AnswerWithContexts(
            answer="Synthetic reference answer.",
            retrieved_contexts=(
                RetrievedContext(source_id="test:relevant", text="Synthetic supporting passage."),
                RetrievedContext(source_id="test:extra", text="Synthetic distractor passage."),
            ),
        )


class SyntheticRagasBackend:
    def evaluate(self, samples):
        assert samples == [
            {
                "user_input": "Synthetic question?",
                "retrieved_contexts": [
                    "Synthetic supporting passage.",
                    "Synthetic distractor passage.",
                ],
                "response": "Synthetic reference answer.",
                "reference": "Synthetic reference answer.",
            }
        ]
        return {"faithfulness": 0.9}


def _dataset() -> LabeledQADataset:
    return LabeledQADataset(
        dataset_version="test-v1",
        source_corpus_version="synthetic-corpus-v1",
        cases=(
            LabeledQACase(
                id="test-case-1",
                question="Synthetic question?",
                reference_answer="Synthetic reference answer.",
                relevant_source_ids=("test:relevant",),
            ),
        ),
    )


def test_qa_artifact_loads_as_empty_scaffold_but_cannot_be_scored() -> None:
    dataset = load_qa_dataset(Path(__file__).parents[2] / "data/evaluation/qa_set.json")

    assert dataset.dataset_version == "0.2.0-draft"
    assert len(dataset.cases) == 18
    assert dataset.label_review_status == "draft"


def test_evaluation_reports_macro_source_id_precision_recall(tmp_path: Path) -> None:
    report = evaluate_dataset(_dataset(), SyntheticTarget(), SyntheticRagasBackend())

    assert report.case_count == 1
    assert report.retrieval_precision == pytest.approx(0.5)
    assert report.retrieval_recall == 1.0
    assert report.ragas_scores == {"faithfulness": 0.9}
    assert report.answerable_case_count == 1
    assert report.unanswerable_retrieval_empty_rate == 1.0
    output = tmp_path / "nested" / "report.json"
    save_evaluation_report(report, output)
    assert '"dataset_version": "test-v1"' in output.read_text(encoding="utf-8")


def test_evaluation_separates_unanswerable_retrieval_and_abstention() -> None:
    dataset = LabeledQADataset(
        dataset_version="test-v2",
        cases=(
            LabeledQACase(
                id="answerable",
                question="Synthetic question?",
                reference_answer="Synthetic reference answer.",
                relevant_source_ids=("test:relevant",),
            ),
            LabeledQACase(
                id="unanswerable",
                question="Out-of-corpus question?",
                answerable=False,
                reference_answer="Unanswerable",
            ),
        ),
    )

    class Target:
        def answer(self, question: str) -> AnswerWithContexts:
            if question == "Synthetic question?":
                return SyntheticTarget().answer(question)
            return AnswerWithContexts(
                answer="Not answerable from the indexed corpus.",
                retrieved_contexts=(),
                abstained=True,
            )

    report = evaluate_dataset(dataset, Target(), SyntheticRagasBackend())

    assert report.case_count == 2
    assert report.answerable_case_count == 1
    assert report.unanswerable_retrieval_empty_rate == 1.0
    assert report.unanswerable_abstention_rate == 1.0


def test_dataset_rejects_duplicate_case_ids_and_empty_answerable_sources() -> None:
    case = LabeledQACase(
        id="same",
        question="Question?",
        reference_answer="Answer.",
        relevant_source_ids=("source:one",),
    )
    with pytest.raises(ValueError, match="unique"):
        LabeledQADataset(dataset_version="v1", cases=(case, case))
    with pytest.raises(ValueError, match="answerable cases require"):
        LabeledQACase(
            id="empty-source",
            question="Question?",
            reference_answer="Answer.",
        )


def test_unanswerable_cases_must_have_explicit_abstention_and_no_sources() -> None:
    with pytest.raises(ValueError, match="explicit abstention"):
        LabeledQACase(
            id="bad-unanswerable",
            question="Out of corpus?",
            answerable=False,
            reference_answer="A made-up answer.",
        )
    with pytest.raises(ValueError, match="must not name"):
        LabeledQACase(
            id="bad-source",
            question="Out of corpus?",
            answerable=False,
            reference_answer="Unanswerable",
            relevant_source_ids=("source:one",),
        )


def test_cumulative_validity_rate_matches_synthetic_optimizer_history() -> None:
    points = cumulative_validity_rate(
        [
            ValidityObservation(iteration=3, valid=True),
            ValidityObservation(iteration=1, valid=False),
            ValidityObservation(iteration=2, valid=True),
        ]
    )

    assert [point.iteration for point in points] == [1, 2, 3]
    assert [point.valid_count for point in points] == [0, 1, 2]
    assert [point.trial_count for point in points] == [1, 2, 3]
    assert [point.validity_rate for point in points] == [0.0, 0.5, pytest.approx(2 / 3)]


def test_validity_rate_rejects_duplicate_iterations() -> None:
    with pytest.raises(ValueError, match="unique positive"):
        cumulative_validity_rate(
            [ValidityObservation(iteration=1, valid=True), ValidityObservation(iteration=1, valid=False)]
        )


def test_validity_observations_read_completed_optuna_trial_attributes() -> None:
    optuna = pytest.importorskip("optuna")
    study = optuna.create_study(direction="maximize")

    def objective(trial):
        trial.set_user_attr("valid", trial.number == 1)
        return float(trial.number)

    study.optimize(objective, n_trials=3)

    assert validity_observations_from_study(study) == (
        ValidityObservation(iteration=1, valid=False),
        ValidityObservation(iteration=2, valid=True),
        ValidityObservation(iteration=3, valid=False),
    )


def test_validity_chart_renders_when_evaluation_extra_is_installed(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    output = tmp_path / "charts" / "validity.png"

    save_validity_chart(
        (
            ValidityRatePoint(1, 0, 1, 0.0),
            ValidityRatePoint(2, 1, 2, 0.5),
            ValidityRatePoint(3, 2, 3, 2 / 3),
        ),
        output,
    )

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_ragas_backend_rejects_unknown_configured_metric() -> None:
    with pytest.raises(ValueError, match="Unsupported RAGAS metric"):
        ConfiguredRagasBackend(["unknown_metric"])
