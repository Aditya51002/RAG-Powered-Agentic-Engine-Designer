"""Dashboard adapter tests against a synthetic optimizer event."""

from __future__ import annotations

from types import SimpleNamespace

from rag_phy.agents import CritiqueResult, DesignCandidate
from rag_phy.app import OptimizationDashboardService
from rag_phy.optimization import OptimizationProgress
from rag_phy.physics.models import CycleResult


def _candidate() -> DesignCandidate:
    return DesignCandidate(
        ambient_temperature_k=288.15,
        ambient_pressure_pa=101325.0,
        flight_speed_m_per_s=0.0,
        air_mass_flow_kg_per_s=10.0,
        compressor_pressure_ratio=5.0,
        turbine_inlet_temperature_k=1000.0,
        hot_section_material_name="SYNTHETIC_TEST_MATERIAL",
    )


def test_dashboard_service_forwards_requirements_and_evidence() -> None:
    candidate = _candidate()
    performance = CycleResult(
        thrust_n=10000.0,
        specific_fuel_consumption_kg_per_n_s=0.00002,
        thermal_efficiency=0.35,
        fuel_air_ratio=0.02,
        stations=(),
    )
    critique = CritiqueResult(
        valid=True,
        reasoning="Synthetic test explanation.",
        cited_sources=("synthetic:test-source",),
        constraint_violations=(),
    )
    event = OptimizationProgress(
        iteration=1,
        total_iterations=1,
        status="valid",
        candidate=candidate,
        performance=performance,
        critique=critique,
        score=1.0,
        best_candidate=candidate,
        best_performance=performance,
        best_score=1.0,
        best_valid=True,
    )

    class OptimizerStub:
        def run(self, goal, progress_callback):
            assert goal == (
                "Optimize the synthetic case\nAdditional design requirements:\n"
                "Synthetic test requirement."
            )
            progress_callback(event)
            return SimpleNamespace(pareto_frontier=(object(),))

    seen = []
    result = OptimizationDashboardService(OptimizerStub()).run_optimization(
        "Optimize the synthetic case", "Synthetic test requirement.", seen.append
    )

    assert result.status == "completed"
    assert result.updates == tuple(seen)
    assert seen[0].best_candidate == candidate
    assert seen[0].critique.cited_sources == ("synthetic:test-source",)
