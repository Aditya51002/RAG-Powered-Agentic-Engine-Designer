"""Objective, incremental Pareto, and real Optuna mocked-pipeline tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_phy.agents import CritiqueResult, DesignCandidate
from rag_phy.config import (
    load_optimization_config,
    load_orchestration_config,
    load_physics_config,
)
from rag_phy.optimization import (
    DesignObjective,
    ObjectiveInputError,
    OptunaOptimizer,
    ParetoFrontier,
    ParetoPoint,
)
from rag_phy.orchestration import DesignWorkflow
from rag_phy.physics.models import CycleInput, CycleResult


def _candidate(pressure_ratio: float, tit_k: float = 1000.0) -> DesignCandidate:
    """Create a synthetic candidate; pressures Pa, temperatures K, flow kg/s."""
    return DesignCandidate(
        ambient_temperature_k=288.15,
        ambient_pressure_pa=101325.0,
        flight_speed_m_per_s=0.0,
        air_mass_flow_kg_per_s=10.0,
        compressor_pressure_ratio=pressure_ratio,
        turbine_inlet_temperature_k=tit_k,
        hot_section_material_name="SYNTHETIC_TEST_ALLOY",
    )


def _performance(thrust_n: float = 20000.0, sfc: float = 0.0000332) -> CycleResult:
    """Return synthetic cycle output; thrust N and SFC kg/(N s)."""
    return CycleResult(
        thrust_n=thrust_n,
        specific_fuel_consumption_kg_per_n_s=sfc,
        thermal_efficiency=0.35,
        fuel_air_ratio=0.02,
        stations=(),
    )


def _critique(
    valid: bool = True,
    severity: float = 0.0,
    citations: tuple[str, ...] = ("synthetic:test-source",),
) -> CritiqueResult:
    """Return a synthetic critique result, never engineering evidence."""
    return CritiqueResult(
        valid=valid,
        reasoning="Synthetic test critique.",
        cited_sources=citations,
        constraint_violations=() if valid else ("synthetic constraint violation",),
        normalized_violation_severity=severity,
    )


def test_objective_uses_known_weighted_score_and_severity_penalty() -> None:
    root = Path(__file__).parents[2]
    config = load_optimization_config(root / "config" / "optimization.yaml")
    objective = DesignObjective(config.objective)
    candidate = _candidate(6.0)

    valid = objective.evaluate(_performance(), _critique(), engine_weight_n=10000.0)
    invalid_mild = objective.evaluate(
        _performance(), _critique(valid=False, severity=0.1), engine_weight_n=10000.0
    )
    invalid_severe = objective.evaluate(
        _performance(), _critique(valid=False, severity=0.3), engine_weight_n=10000.0
    )

    assert candidate.compressor_pressure_ratio == 6.0
    assert valid.thrust_to_weight_ratio == pytest.approx(2.0)
    assert valid.score == pytest.approx(1.0)
    assert valid.penalty == 0.0
    assert invalid_mild.score == pytest.approx(-0.1)
    assert invalid_mild.penalty == pytest.approx(1.1)
    assert invalid_severe.score == pytest.approx(-0.3)
    assert invalid_severe.score < invalid_mild.score


def test_valid_without_citations_is_penalized_as_invalid() -> None:
    config = load_optimization_config("config/optimization.yaml")
    result = DesignObjective(config.objective).evaluate(
        _performance(), _critique(citations=()), engine_weight_n=10000.0
    )

    assert not result.valid
    assert result.penalty == config.objective.invalid_base_penalty


def test_objective_rejects_nonphysical_metrics() -> None:
    config = load_optimization_config("config/optimization.yaml")

    with pytest.raises(ObjectiveInputError, match="finite positive"):
        DesignObjective(config.objective).evaluate(
            _performance(thrust_n=float("nan")), _critique(), engine_weight_n=10000.0
        )


def test_pareto_frontier_incrementally_discards_dominated_points() -> None:
    frontier = ParetoFrontier(signature_decimal_places=6)
    point_a = ParetoPoint(_candidate(5), 2.0, 0.00004, ("synthetic:a",))
    point_b = ParetoPoint(_candidate(6), 1.5, 0.00003, ("synthetic:b",))
    dominated = ParetoPoint(_candidate(7), 1.0, 0.00005, ("synthetic:c",))
    dominates_both = ParetoPoint(_candidate(8), 3.0, 0.00002, ("synthetic:d",))

    assert frontier.add(point_a)
    assert frontier.add(point_b)
    assert not frontier.add(dominated)
    assert len(frontier.points) == 2
    assert frontier.add(dominates_both)
    assert frontier.points == (dominates_both,)
    assert not frontier.add(dominates_both)


class NoCallDesignAgent:
    """Assert the optimizer-seeded candidate path bypasses model generation."""

    def propose(self, goal: str, rejection_history=()) -> DesignCandidate:
        """Fail if seeded optimization unexpectedly calls the design model."""
        raise AssertionError("Seeded optimizer candidate should bypass model proposal")


class AlwaysValidCritique:
    """Return a source-bearing synthetic critique for the integration fixture."""

    def critique(self, candidate: DesignCandidate, performance: CycleResult) -> CritiqueResult:
        """Mark the test candidate valid with only synthetic evidence provenance."""
        return _critique()


def test_short_optuna_study_runs_phase5_mocked_pipeline() -> None:
    root = Path(__file__).parents[2]
    optimization_config = load_optimization_config(root / "config" / "optimization.yaml")
    physics_config = load_physics_config(root / "config" / "physics_bounds.yaml")
    orchestration_config = load_orchestration_config(
        root / "config" / "orchestration.yaml"
    ).model_copy(update={"max_iterations": 1, "max_invalid_revisions": 1})

    def candidate_sampler(trial) -> DesignCandidate:
        """Sample deliberately synthetic test ranges, not aircraft design bounds."""
        pressure_ratio = trial.suggest_float("synthetic_pressure_ratio", 4.0, 8.0)
        tit_k = trial.suggest_float("synthetic_tit_k", 900.0, 1100.0)
        return _candidate(pressure_ratio, tit_k)

    def simulator(inputs: CycleInput, config) -> CycleResult:
        """Map test inputs to deterministic, explicitly synthetic cycle metrics."""
        thrust_n = inputs.compressor_pressure_ratio * inputs.turbine_inlet_temperature_k
        sfc = 0.0000332 * inputs.turbine_inlet_temperature_k / (
            1000.0 * inputs.compressor_pressure_ratio / 6.0
        )
        return _performance(thrust_n=thrust_n, sfc=sfc)

    workflow = DesignWorkflow(
        design_agent=NoCallDesignAgent(),
        physics_config=physics_config,
        critique_agent=AlwaysValidCritique(),
        scorer=lambda candidate, performance: performance.thrust_n,
        orchestration_config=orchestration_config,
        simulator=simulator,
    )
    optimizer = OptunaOptimizer(
        config=optimization_config,
        workflow=workflow,
        candidate_sampler=candidate_sampler,
        weight_estimator=lambda candidate, performance: 10000.0,
    )

    progress = []
    result = optimizer.run("synthetic integration test", progress_callback=progress.append)

    assert len(result.study.trials) == optimization_config.trial_count
    assert result.study.best_trial.state.name == "COMPLETE"
    assert all(trial.user_attrs["valid"] for trial in result.study.trials)
    assert result.pareto_frontier
    assert all(point.cited_sources for point in result.pareto_frontier)
    assert [item.iteration for item in progress] == list(
        range(1, optimization_config.trial_count + 1)
    )
    assert progress[-1].best_score == pytest.approx(result.study.best_value)
    assert progress[-1].critique.cited_sources == ("synthetic:test-source",)


def test_duplicate_optuna_candidates_reuse_evaluation() -> None:
    root = Path(__file__).parents[2]
    optimization_config = load_optimization_config(
        root / "config" / "optimization.yaml"
    ).model_copy(update={"trial_count": 3})
    physics_config = load_physics_config(root / "config" / "physics_bounds.yaml")
    orchestration_config = load_orchestration_config(
        root / "config" / "orchestration.yaml"
    ).model_copy(update={"max_iterations": 1})
    simulation_calls = 0

    def simulator(inputs: CycleInput, config) -> CycleResult:
        nonlocal simulation_calls
        simulation_calls += 1
        return _performance()

    workflow = DesignWorkflow(
        design_agent=NoCallDesignAgent(),
        physics_config=physics_config,
        critique_agent=AlwaysValidCritique(),
        scorer=lambda candidate, performance: performance.thrust_n,
        orchestration_config=orchestration_config,
        simulator=simulator,
    )
    optimizer = OptunaOptimizer(
        config=optimization_config,
        workflow=workflow,
        candidate_sampler=lambda trial: _candidate(6.0),
        weight_estimator=lambda candidate, performance: 10000.0,
    )

    progress = []
    result = optimizer.run("duplicate sample test", progress_callback=progress.append)

    assert len(result.study.trials) == 3
    assert simulation_calls == 1
    assert sum(
        bool(trial.user_attrs.get("duplicate_evaluation_reused"))
        for trial in result.study.trials
    ) == 2
    assert len(progress) == 3
    assert all(event.best_valid for event in progress)
