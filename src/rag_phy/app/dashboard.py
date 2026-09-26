"""Thin Streamlit dashboard over an injected design-optimization service."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from rag_phy.agents import CritiqueResult, DesignCandidate
from rag_phy.physics.models import CycleResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DashboardProgress:
    """One live optimizer update, including evidence and the current best candidate."""

    iteration: int
    total_iterations: int
    status: str
    candidate: DesignCandidate | None = None
    performance: CycleResult | None = None
    critique: CritiqueResult | None = None
    score: float | None = None
    best_candidate: DesignCandidate | None = None
    best_performance: CycleResult | None = None
    best_score: float | None = None
    best_valid: bool | None = None


@dataclass(frozen=True)
class DashboardRunResult:
    """Terminal service status and final optimizer history for one UI submission."""

    status: str
    updates: tuple[DashboardProgress, ...]


class DashboardService(Protocol):
    """Configured optimization facade; provider and engineering adapters live outside UI."""

    def run_optimization(
        self,
        design_goal: str,
        additional_constraints: str,
        on_progress: Callable[[DashboardProgress], None],
    ) -> DashboardRunResult:
        """Run a configured optimization and report iterations as they complete."""


def render_dashboard(service: DashboardService) -> None:
    """Render input, live optimization progress, best-so-far, and cited critique details."""
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("Install rag-phy[app] to run the Streamlit dashboard") from exc

    st.title("RAG Phy")
    st.caption("Preliminary design-space exploration")
    st.caption(
        "Natural-language requirements guide proposals; only configured code-level checks are enforced."
    )

    with st.form("optimization_request"):
        design_goal = st.text_area("Design goal", placeholder="Describe the optimization objective")
        additional_constraints = st.text_area(
            "Additional constraints",
            placeholder="State operating requirements for the configured design service",
        )
        submitted = st.form_submit_button("Run optimization", type="primary")

    if not submitted:
        _render_previous_run(st)
        return
    if not design_goal.strip():
        st.error("Enter a design goal before starting an optimization.")
        return

    progress_bar = st.progress(0.0)
    progress_label = st.empty()
    best_panel = st.container()
    detail_panel = st.container()
    updates: list[DashboardProgress] = []

    def show_progress(update: DashboardProgress) -> None:
        updates.append(update)
        fraction = update.iteration / update.total_iterations if update.total_iterations else 0.0
        progress_bar.progress(min(max(fraction, 0.0), 1.0))
        progress_label.write(
            f"Iteration {update.iteration} of {update.total_iterations} | {update.status}"
        )
        with best_panel:
            _render_best(st, update)
        with detail_panel:
            _render_update_detail(st, update)

    try:
        with st.spinner("Optimization in progress"):
            result = service.run_optimization(
                design_goal.strip(), additional_constraints.strip(), show_progress
            )
    except Exception as exc:
        logger.exception("Dashboard optimization request failed")
        st.error(f"Optimization failed: {exc}")
        return

    if not result.updates and not updates:
        st.warning("The configured service completed without reporting any trial updates.")
    stored_updates = result.updates or tuple(updates)
    st.session_state["rag_phy_last_run"] = DashboardRunResult(result.status, stored_updates)
    st.success(f"Optimization finished: {result.status}")


def _render_previous_run(st: Any) -> None:
    """Show the most recent completed run when the page reruns after widget interaction."""
    result = st.session_state.get("rag_phy_last_run")
    if result is None or not result.updates:
        return
    latest = result.updates[-1]
    _render_best(st, latest)
    _render_update_detail(st, latest)
    with st.expander("Trial history", expanded=False):
        for update in result.updates:
            st.write(
                {
                    "iteration": update.iteration,
                    "status": update.status,
                    "score": update.score,
                    "valid": update.critique.valid if update.critique else None,
                }
            )


def _render_best(st: Any, update: DashboardProgress) -> None:
    """Display best-so-far candidate and key cycle outputs without recomputation."""
    st.subheader("Best-scoring candidate")
    if update.best_candidate is None:
        st.info("No candidate has passed validation yet.")
        return
    if update.best_valid:
        st.success("The best-scoring candidate passed configured checks.")
    else:
        st.warning("The best-scoring candidate is invalid and must not be used as a design.")
    left, middle, right = st.columns(3)
    left.metric("Objective score", _format_optional(update.best_score))
    right.metric(
        "Thrust",
        _format_optional(
            update.best_performance.thrust_n if update.best_performance else None,
            suffix=" N",
        ),
    )
    middle.metric(
        "Specific fuel consumption",
        _format_optional(
            update.best_performance.specific_fuel_consumption_kg_per_n_s
            if update.best_performance else None,
            suffix=" kg/(N s)",
        ),
    )
    st.json(update.best_candidate.model_dump(mode="json"))


def _render_update_detail(st: Any, update: DashboardProgress) -> None:
    """Show the iteration's critique, source citations, and hard-constraint failures."""
    if update.critique is None:
        return
    with st.expander(f"Iteration {update.iteration} reasoning and evidence", expanded=True):
        if update.candidate is not None:
            st.markdown("**Candidate**")
            st.json(update.candidate.model_dump(mode="json"))
        if update.score is not None:
            st.write({"objective_score": update.score})
        if update.performance is not None:
            st.write("Cycle performance")
            st.json(update.performance.model_dump(mode="json"))
        st.write(update.critique.reasoning)
        if update.critique.cited_sources:
            st.markdown("**Citations**")
            for source in update.critique.cited_sources:
                st.write(f"- {source}")
        else:
            st.warning("No source citations were attached to this critique.")
        if update.critique.constraint_violations:
            st.markdown("**Constraint violations**")
            for violation in update.critique.constraint_violations:
                st.write(f"- {violation}")


def _format_optional(value: float | None, suffix: str = "") -> str:
    """Format optional metrics for display without manufacturing a default value."""
    return "Not available" if value is None else f"{value:.5g}{suffix}"


__all__ = [
    "DashboardProgress",
    "DashboardRunResult",
    "DashboardService",
    "render_dashboard",
]
