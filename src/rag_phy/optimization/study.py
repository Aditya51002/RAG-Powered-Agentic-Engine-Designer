"""Optuna study runner connected to the Phase 5 workflow by injection."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Protocol

import optuna

from rag_phy.agents import CritiqueResult, DesignCandidate
from rag_phy.config import OptimizationConfig
from rag_phy.optimization.objective import DesignObjective, ObjectiveResult
from rag_phy.optimization.pareto import ParetoFrontier, ParetoPoint
from rag_phy.orchestration import DesignWorkflow
from rag_phy.physics.models import CycleResult

logger = logging.getLogger(__name__)


class CandidateSampler(Protocol):
    """Generate a validated design from configured Optuna search distributions."""

    def __call__(self, trial: optuna.trial.Trial) -> DesignCandidate:
        """Sample a candidate; production bounds must come from verified project config."""


class EngineWeightEstimator(Protocol):
    """Supply engine weight because the current cycle model does not calculate engine mass."""

    def __call__(self, candidate: DesignCandidate, performance: CycleResult) -> float:
        """Return engine weight in newtons from a sourced or caller-owned model."""


class OptimizationRunError(RuntimeError):
    """Raised when a trial cannot produce performance and a typed critique result."""


@dataclass(frozen=True)
class OptimizationRun:
    """Completed Optuna study and its independently maintained Pareto frontier."""

    study: Any
    pareto_frontier: tuple[ParetoPoint, ...]


class OptunaOptimizer:
    """Run a scalar Optuna study and incrementally retain valid Pareto-optimal designs."""

    def __init__(
        self,
        config: OptimizationConfig,
        workflow: DesignWorkflow,
        candidate_sampler: CandidateSampler,
        weight_estimator: EngineWeightEstimator,
    ) -> None:
        """Bind the configured objective to an injected graph, sampler, and weight model."""
        self._config = config
        self._workflow = workflow
        self._candidate_sampler = candidate_sampler
        self._weight_estimator = weight_estimator
        self._objective = DesignObjective(config.objective)
        self._frontier = ParetoFrontier(config.objective.signature_decimal_places)
        self._evaluated_candidates: dict[
            str, tuple[ObjectiveResult, CritiqueResult]
        ] = {}

    def run(self, design_goal: str) -> OptimizationRun:
        """Execute the configured number of actual Optuna trials for one design goal.

        Each sampled design is passed through the Phase 5 graph in single-candidate mode.
        Invalid critiques receive a severity-shaped score penalty but never enter the valid
        Pareto frontier. Search bounds remain owned by the injected candidate sampler.
        """
        if not design_goal.strip():
            raise ValueError("Design goal must not be empty")
        self._frontier = ParetoFrontier(self._config.objective.signature_decimal_places)
        self._evaluated_candidates = {}
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self._config.random_seed),
        )
        try:
            study.optimize(
                lambda trial: self._evaluate_trial(trial, design_goal.strip()),
                n_trials=self._config.trial_count,
                n_jobs=1,
            )
        except Exception:
            logger.exception("Optuna study failed", extra={"trial_count": len(study.trials)})
            raise
        logger.info(
            "Optuna study completed",
            extra={
                "trial_count": len(study.trials),
                "pareto_frontier_size": len(self._frontier.points),
                "best_score": study.best_value,
            },
        )
        return OptimizationRun(study=study, pareto_frontier=self._frontier.points)

    def _evaluate_trial(self, trial: optuna.trial.Trial, design_goal: str) -> float:
        """Sample, run, score, and attach auditable metrics to one Optuna trial."""
        candidate = self._candidate_sampler(trial)
        trial.set_user_attr("candidate", candidate.model_dump(mode="json"))
        signature = candidate.parameter_signature(
            self._config.objective.signature_decimal_places
        )
        cached = self._evaluated_candidates.get(signature)
        if cached is not None:
            objective_result, critique = cached
            trial.set_user_attr("duplicate_evaluation_reused", True)
            self._set_trial_metrics(trial, objective_result, candidate, critique)
            logger.info(
                "Reused prior evaluation for canonical duplicate design",
                extra={"trial": trial.number, "candidate_signature": signature},
            )
            return objective_result.score

        state = self._workflow.invoke(
            design_goal,
            initial_candidate=candidate,
            single_candidate=True,
        )
        performance = state.get("performance_result")
        critique = state.get("critique_result")
        if performance is None or critique is None:
            raise OptimizationRunError(
                "Workflow ended without both cycle performance and critique results"
            )
        try:
            weight_n = float(self._weight_estimator(candidate, performance))
        except Exception as exc:
            logger.exception("Engine weight estimation failed", extra={"trial": trial.number})
            raise OptimizationRunError("Engine weight estimator failed") from exc
        objective_result = self._objective.evaluate(
            performance=performance,
            critique=critique,
            engine_weight_n=weight_n,
        )
        self._evaluated_candidates[signature] = (objective_result, critique)
        self._set_trial_metrics(trial, objective_result, candidate, critique)
        if objective_result.valid:
            self._frontier.add(
                ParetoPoint(
                    candidate=candidate,
                    thrust_to_weight_ratio=objective_result.thrust_to_weight_ratio,
                    specific_fuel_consumption_kg_per_n_s=(
                        objective_result.specific_fuel_consumption_kg_per_n_s
                    ),
                    cited_sources=critique.cited_sources,
                )
            )
        logger.info(
            "Optuna trial evaluated",
            extra={
                "trial": trial.number,
                "score": objective_result.score,
                "valid": objective_result.valid,
                "penalty": objective_result.penalty,
            },
        )
        return objective_result.score

    def _set_trial_metrics(
        self,
        trial: optuna.trial.Trial,
        result: ObjectiveResult,
        candidate: DesignCandidate,
        critique: CritiqueResult,
    ) -> None:
        """Attach JSON-compatible objective decomposition, validity, and citations."""
        metrics = {
            "score": result.score,
            "valid": result.valid,
            "penalty": result.penalty,
            "thrust_to_weight_ratio": result.thrust_to_weight_ratio,
            "specific_fuel_consumption_kg_per_n_s": (
                result.specific_fuel_consumption_kg_per_n_s
            ),
            "normalized_violation_severity": critique.normalized_violation_severity,
            "candidate_signature": candidate.parameter_signature(
                self._config.objective.signature_decimal_places
            ),
            "cited_sources": list(critique.cited_sources),
            "constraint_violations": list(critique.constraint_violations),
        }
        if not all(math.isfinite(value) for value in (result.score, result.penalty)):
            raise OptimizationRunError("Objective metrics must be finite before study logging")
        for name, value in metrics.items():
            trial.set_user_attr(name, value)


__all__ = [
    "CandidateSampler",
    "EngineWeightEstimator",
    "OptunaOptimizer",
    "OptimizationRun",
    "OptimizationRunError",
]
