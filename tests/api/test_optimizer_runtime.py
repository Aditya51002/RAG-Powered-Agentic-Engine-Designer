"""Tests for explicit optimizer-to-API mapping with a fake optimizer."""

from __future__ import annotations

from pathlib import Path

from rag_phy.agents import CritiqueResult, DesignCandidate
from rag_phy.api.runtime import OptimizationDesignRunExecutor
from rag_phy.optimization import OptimizationProgress, OptimizationRun, ParetoPoint
from rag_phy.physics.models import CycleResult


def _candidate() -> DesignCandidate:
    return DesignCandidate(
        ambient_temperature_k=290,
        ambient_pressure_pa=100000,
        flight_speed_m_per_s=0,
        air_mass_flow_kg_per_s=5,
        compressor_pressure_ratio=5,
        turbine_inlet_temperature_k=1000,
        hot_section_material_name="CMSX-4",
    )


def _performance() -> CycleResult:
    return CycleResult(12000, 0.000031, 0.34, 0.018, ())


def _critique() -> CritiqueResult:
    return CritiqueResult(
        valid=True,
        reasoning="screened by deterministic checks",
        cited_sources=("material:source#page=3",),
        constraint_violations=(),
        normalized_violation_severity=0,
    )


class _FakeOptimizer:
    def __init__(self) -> None:
        self.kwargs = None

    def run(self, design_goal, **kwargs):
        self.kwargs = kwargs
        candidate = _candidate()
        performance = _performance()
        critique = _critique()
        kwargs["progress_callback"](
            OptimizationProgress(
                iteration=1,
                total_iterations=kwargs["trial_count"],
                status="valid",
                candidate=candidate,
                performance=performance,
                critique=critique,
                score=0.8,
                best_candidate=candidate,
                best_performance=performance,
                best_score=0.8,
                best_valid=True,
            )
        )
        point = ParetoPoint(candidate, 2.1, 0.000031, critique.cited_sources)
        return OptimizationRun(
            study=object(),
            pareto_frontier=(point,),
            best_valid_candidate=candidate,
            best_valid_performance=performance,
            best_valid_critique=critique,
            best_valid_score=0.8,
        )


def test_optimizer_runtime_maps_valid_result_progress_and_trace_context(tmp_path: Path) -> None:
    optimizer = _FakeOptimizer()
    observed_trace_paths = []
    executor = OptimizationDesignRunExecutor(
        lambda path: (observed_trace_paths.append(path), optimizer)[1]
    )
    progress = []
    trace_path = tmp_path / "run.jsonl"

    result = executor.run("design goal", 23, progress.append, trace_path, "request-123")

    assert observed_trace_paths == [trace_path]
    assert optimizer.kwargs["trace_id"] == "request-123"
    assert optimizer.kwargs["trial_count"] == 23
    assert progress[0].total_iterations == 23
    assert result.accepted_candidate is not None
    assert result.accepted_candidate.hot_section_material_name == "CMSX-4"
    assert result.performance is not None
    assert result.performance.thrust_n == 12000
    assert result.cited_sources == ("material:source#page=3",)
    assert result.pareto_frontier[0].thrust_to_weight_ratio == 2.1


def test_optimizer_runtime_maps_no_valid_design_as_no_acceptance(tmp_path: Path) -> None:
    class NoValidOptimizer:
        def run(self, design_goal, **kwargs):
            return OptimizationRun(study=object(), pareto_frontier=())

    executor = OptimizationDesignRunExecutor(lambda path: NoValidOptimizer())

    result = executor.run("no solution", 1, lambda progress: None, tmp_path / "trace", "trace-id")

    assert result.accepted_candidate is None
    assert result.performance is None
    assert not result.cited_sources
    assert not result.pareto_frontier
