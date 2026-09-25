"""Offline integration tests for the bounded LangGraph design workflow."""

from __future__ import annotations

from pathlib import Path

from rag_phy.agents import CritiqueResult, DesignCandidate
from rag_phy.config import load_orchestration_config, load_physics_config
from rag_phy.orchestration.workflow import DesignWorkflow
from rag_phy.physics.models import CycleResult


def _candidate(pressure_ratio: float) -> DesignCandidate:
    """Create a synthetic candidate; pressures Pa, temperatures K, flow kg/s."""
    return DesignCandidate(
        ambient_temperature_k=288.15,
        ambient_pressure_pa=101325.0,
        flight_speed_m_per_s=0.0,
        air_mass_flow_kg_per_s=10.0,
        compressor_pressure_ratio=pressure_ratio,
        turbine_inlet_temperature_k=1100.0,
        hot_section_material_name="TEST_ALLOY_ALPHA",
    )


def _performance() -> CycleResult:
    """Provide deterministic synthetic cycle results; thrust N and SFC kg/(N s)."""
    return CycleResult(
        thrust_n=10000.0,
        specific_fuel_consumption_kg_per_n_s=0.00002,
        thermal_efficiency=0.35,
        fuel_air_ratio=0.02,
        stations=(),
    )


class SequenceDesignAgent:
    """Return preset proposals to emulate a deterministic offline model."""

    def __init__(self, candidates: list[DesignCandidate]) -> None:
        self.candidates = candidates
        self.calls = 0
        self.received_histories = []

    def propose(self, goal: str, rejection_history=()) -> DesignCandidate:
        """Return a queued candidate and capture the rejection history received."""
        self.received_histories.append(list(rejection_history))
        index = min(self.calls, len(self.candidates) - 1)
        self.calls += 1
        return self.candidates[index]


class SequenceCritiqueAgent:
    """Return a fixed typed critique for every cycle result."""

    def __init__(self, valid: bool) -> None:
        self.valid = valid
        self.calls = 0

    def critique(self, candidate: DesignCandidate, performance: CycleResult) -> CritiqueResult:
        """Return a synthetic critique with citations only when marked valid."""
        self.calls += 1
        return CritiqueResult(
            valid=self.valid,
            reasoning="Synthetic test result.",
            cited_sources=("synthetic:test-source",) if self.valid else (),
            constraint_violations=() if self.valid else ("synthetic rejected condition",),
        )


def _workflow(
    design_agent: SequenceDesignAgent,
    critique_agent: SequenceCritiqueAgent,
    *,
    max_iterations: int = 6,
    max_invalid_revisions: int = 2,
    tolerance: float = 0.001,
) -> DesignWorkflow:
    """Build the production graph around isolated synthetic nodes and configs."""
    root = Path(__file__).parents[2]
    return DesignWorkflow(
        design_agent=design_agent,
        physics_config=load_physics_config(root / "config" / "physics_bounds.yaml"),
        critique_agent=critique_agent,
        scorer=lambda candidate, performance: candidate.compressor_pressure_ratio,
        orchestration_config=load_orchestration_config(
            root / "config" / "orchestration.yaml"
        ).model_copy(
            update={
                "max_iterations": max_iterations,
                "max_invalid_revisions": max_invalid_revisions,
                "convergence_tolerance": tolerance,
            }
        ),
        simulator=lambda inputs, config: _performance(),
    )


def test_rejected_candidate_revises_and_stops_at_configured_limit() -> None:
    design = SequenceDesignAgent([_candidate(5.0)])
    critique = SequenceCritiqueAgent(valid=False)
    workflow = _workflow(design, critique, max_invalid_revisions=2)

    result = workflow.invoke("find a preliminary design")

    assert result["status"] == "max_invalid_revisions"
    assert result["iteration_count"] == 2
    assert result["invalid_revision_count"] == 2
    assert len(result["rejection_history"]) == 2
    assert len(result["transition_history"]) >= 9
    assert design.received_histories[1][0].reason == "synthetic rejected condition"


def test_improving_valid_candidates_converge_and_retain_best() -> None:
    candidates = [_candidate(5.0), _candidate(6.0), _candidate(6.0005)]
    design = SequenceDesignAgent(candidates)
    critique = SequenceCritiqueAgent(valid=True)
    workflow = _workflow(design, critique, tolerance=0.001)

    result = workflow.invoke("maximize a test objective")

    assert result["status"] == "converged"
    assert result["iteration_count"] == 3
    assert result["best_candidate"].compressor_pressure_ratio == 6.0005
    assert result["best_score"] == 6.0005
    assert len(result["trial_history"]) == 3
    assert result["best_critique"].cited_sources == ("synthetic:test-source",)


def test_max_iterations_terminates_even_while_score_improves() -> None:
    design = SequenceDesignAgent([_candidate(5.0), _candidate(6.0), _candidate(7.0)])
    workflow = _workflow(
        design,
        SequenceCritiqueAgent(valid=True),
        max_iterations=2,
        tolerance=0.0,
    )

    result = workflow.invoke("continue improving")

    assert result["status"] == "max_iterations"
    assert result["iteration_count"] == 2
    assert result["best_score"] == 6.0


def test_graph_has_explicit_workflow_nodes() -> None:
    workflow = _workflow(
        SequenceDesignAgent([_candidate(5.0)]), SequenceCritiqueAgent(valid=True)
    )

    node_names = set(workflow.graph.get_graph().nodes)
    assert {"propose", "simulate", "critique", "revise", "score", "optimizer_step"} <= node_names
