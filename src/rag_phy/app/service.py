"""Adapter from the configured Optuna optimizer to the dashboard service contract."""

from __future__ import annotations

from collections.abc import Callable

from rag_phy.app.dashboard import DashboardProgress, DashboardRunResult
from rag_phy.optimization import OptunaOptimizer


class OptimizationDashboardService:
    """Run the domain optimizer and translate its trial events into UI view models."""

    def __init__(self, optimizer: OptunaOptimizer) -> None:
        """Bind an optimizer whose sampler, workflow, and weight model are already configured."""
        self._optimizer = optimizer

    def run_optimization(
        self,
        design_goal: str,
        additional_constraints: str,
        on_progress: Callable[[DashboardProgress], None],
    ) -> DashboardRunResult:
        """Execute Optuna while forwarding each trial and accumulated best-so-far state."""
        if not design_goal.strip():
            raise ValueError("Design goal must not be empty")
        combined_goal = design_goal.strip()
        if additional_constraints.strip():
            combined_goal += "\nAdditional design requirements:\n" + additional_constraints.strip()

        updates: list[DashboardProgress] = []

        def forward(progress) -> None:
            update = DashboardProgress(
                iteration=progress.iteration,
                total_iterations=progress.total_iterations,
                status=progress.status,
                candidate=progress.candidate,
                performance=progress.performance,
                critique=progress.critique,
                score=progress.score,
                best_candidate=progress.best_candidate,
                best_performance=progress.best_performance,
                best_score=progress.best_score,
                best_valid=progress.best_valid,
            )
            updates.append(update)
            on_progress(update)

        run = self._optimizer.run(combined_goal, progress_callback=forward)
        status = "completed" if run.pareto_frontier else "completed_without_valid_candidates"
        return DashboardRunResult(status=status, updates=tuple(updates))


__all__ = ["OptimizationDashboardService"]
