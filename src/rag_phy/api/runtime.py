"""Explicit adapter from the domain optimizer into stable public API response models."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rag_phy.api.schemas import (
    CandidateResponse,
    DesignProgressResponse,
    DesignRunResultResponse,
    ParetoDesignResponse,
    PerformanceResponse,
)
from rag_phy.optimization import OptimizationProgress, OptunaOptimizer


class OptimizationDesignRunExecutor:
    """Adapt a deployment-configured optimizer factory to the API's executor protocol.

    The factory receives the per-run trace path so it can bind a JSONL trace sink to that run.
    Sourced search data, provider credentials, and retriever setup remain deployment-owned.
    """

    def __init__(self, optimizer_factory: Callable[[Path], OptunaOptimizer]) -> None:
        self._optimizer_factory = optimizer_factory

    def run(
        self,
        design_goal: str,
        trial_count: int,
        on_progress: Callable[[DesignProgressResponse], None],
        trace_path: Path,
        request_id: str,
    ) -> DesignRunResultResponse:
        """Run Optuna, stream stable progress schemas, and map valid results and citations."""
        optimizer = self._optimizer_factory(trace_path)

        def forward(progress: OptimizationProgress) -> None:
            on_progress(
                DesignProgressResponse(
                    iteration=progress.iteration,
                    total_iterations=progress.total_iterations,
                    status=progress.status,
                    best_candidate=CandidateResponse.model_validate(
                        progress.best_candidate.model_dump(mode="python")
                    ),
                    best_score=progress.best_score,
                    best_valid=progress.best_valid,
                )
            )

        run = optimizer.run(
            design_goal,
            progress_callback=forward,
            trace_id=request_id,
            trial_count=trial_count,
        )
        accepted = run.best_valid_candidate
        performance = run.best_valid_performance
        critique = run.best_valid_critique
        if accepted is None or performance is None or critique is None:
            accepted_response = None
            performance_response = None
            cited_sources: tuple[str, ...] = ()
        else:
            accepted_response = CandidateResponse.model_validate(
                accepted.model_dump(mode="python")
            )
            performance_response = PerformanceResponse(
                thrust_n=performance.thrust_n,
                specific_fuel_consumption_kg_per_n_s=(
                    performance.specific_fuel_consumption_kg_per_n_s
                ),
                thermal_efficiency=performance.thermal_efficiency,
                fuel_air_ratio=performance.fuel_air_ratio,
            )
            cited_sources = tuple(critique.cited_sources)

        frontier = tuple(
            ParetoDesignResponse(
                candidate=CandidateResponse.model_validate(
                    point.candidate.model_dump(mode="python")
                ),
                thrust_to_weight_ratio=point.thrust_to_weight_ratio,
                specific_fuel_consumption_kg_per_n_s=(
                    point.specific_fuel_consumption_kg_per_n_s
                ),
                cited_sources=tuple(point.cited_sources),
            )
            for point in run.pareto_frontier
        )
        return DesignRunResultResponse(
            accepted_candidate=accepted_response,
            performance=performance_response,
            pareto_frontier=frontier,
            cited_sources=cited_sources,
        )


__all__ = ["OptimizationDesignRunExecutor"]
