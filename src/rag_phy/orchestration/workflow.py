"""Explicit LangGraph design-propose-simulate-critique workflow."""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from rag_phy.agents import (
    CandidateRejection,
    CritiqueAgent,
    CritiqueResult,
    DesignAgent,
    DesignCandidate,
    ProposalError,
)
from rag_phy.config import OrchestrationConfig, PhysicsConfig
from rag_phy.physics.cycle import simulate_cycle
from rag_phy.physics.models import CycleInput, CycleResult
from rag_phy.orchestration.tracing import WorkflowTraceSink, new_trace_event

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkflowEvent:
    """One audit event emitted by a graph node or state transition."""

    node: str
    iteration: int
    message: str


@dataclass(frozen=True)
class ScoredTrial:
    """One valid, cited design observation and its caller-defined scalar score."""

    candidate: DesignCandidate
    performance: CycleResult
    critique: CritiqueResult
    score: float


class WorkflowState(TypedDict, total=False):
    """Typed, serializable-in-shape state shared by the LangGraph nodes."""

    design_goal: str
    seed_candidate: DesignCandidate
    seed_consumed: bool
    single_candidate: bool
    current_candidate: DesignCandidate
    performance_result: CycleResult
    critique_result: CritiqueResult
    iteration_count: int
    invalid_revision_count: int
    rejection_history: list[CandidateRejection]
    current_score: float
    trial_history: list[ScoredTrial]
    best_candidate: DesignCandidate
    best_performance: CycleResult
    best_critique: CritiqueResult
    best_score: float
    status: str
    stop_reason: str
    transition_history: list[WorkflowEvent]
    trace_id: str
    last_span_id: str


class CycleSimulator(Protocol):
    """Callable interface for a deterministic cycle simulator."""

    def __call__(self, inputs: CycleInput, config: PhysicsConfig) -> CycleResult:
        """Simulate one candidate; physical inputs and outputs use SI units."""


class CandidateScorer(Protocol):
    """Caller-supplied scalar objective for ranking valid candidates."""

    def __call__(self, candidate: DesignCandidate, performance: CycleResult) -> float:
        """Return a finite score where larger values are preferred."""


class DesignWorkflow:
    """Compile and invoke a bounded LangGraph proposal, simulation, and critique loop."""

    def __init__(
        self,
        design_agent: DesignAgent,
        physics_config: PhysicsConfig,
        critique_agent: CritiqueAgent,
        scorer: CandidateScorer,
        orchestration_config: OrchestrationConfig,
        simulator: CycleSimulator = simulate_cycle,
        trace_sink: WorkflowTraceSink | None = None,
    ) -> None:
        """Bind injectable components and compile the named state graph."""
        self._design_agent = design_agent
        self._physics_config = physics_config
        self._critique_agent = critique_agent
        self._scorer = scorer
        self._config = orchestration_config
        self._simulator = simulator
        self._trace_sink = trace_sink
        self._graph = self._compile_graph()

    @property
    def graph(self) -> Any:
        """Return the compiled LangGraph for streaming, visualization, or direct inspection."""
        return self._graph

    def invoke(
        self,
        design_goal: str,
        initial_candidate: DesignCandidate | None = None,
        single_candidate: bool = False,
    ) -> WorkflowState:
        """Execute the graph, optionally starting from an optimizer-sampled candidate."""
        if not design_goal.strip():
            raise ValueError("Design goal must not be empty")
        initial_state: WorkflowState = {
            "design_goal": design_goal.strip(),
            "iteration_count": 0,
            "invalid_revision_count": 0,
            "rejection_history": [],
            "trial_history": [],
            "transition_history": [],
            "status": "running",
            "single_candidate": single_candidate,
            "trace_id": uuid.uuid4().hex,
        }
        if initial_candidate is not None:
            initial_state["seed_candidate"] = initial_candidate
            initial_state["seed_consumed"] = False
        recursion_limit = (
            self._config.max_iterations * 5
            + self._config.max_invalid_revisions
            + 5
        )
        try:
            return self._graph.invoke(
                initial_state,
                config={"recursion_limit": recursion_limit},
            )
        except Exception:
            logger.exception("Design workflow graph execution failed")
            raise

    def _compile_graph(self) -> Any:
        """Define the workflow as explicit nodes, directed edges, and conditional routes."""
        builder = StateGraph(WorkflowState)
        builder.add_node("propose", self._propose)
        builder.add_node("simulate", self._simulate)
        builder.add_node("critique", self._critique)
        builder.add_node("revise", self._revise)
        builder.add_node("score", self._score)
        builder.add_node("optimizer_step", self._optimizer_step)
        builder.add_node("converge", self._converge)
        builder.add_edge(START, "propose")
        builder.add_conditional_edges(
            "propose",
            lambda state: "simulate" if state.get("status") == "running" else END,
            {"simulate": "simulate", END: END},
        )
        builder.add_edge("simulate", "critique")
        builder.add_conditional_edges(
            "critique",
            self._route_after_critique,
            {"score": "score", "revise": "revise"},
        )
        builder.add_conditional_edges(
            "revise",
            self._route_after_revision,
            {"propose": "propose", "converge": "converge"},
        )
        builder.add_edge("score", "optimizer_step")
        builder.add_conditional_edges(
            "optimizer_step",
            self._route_after_optimizer_step,
            {"propose": "propose", "converge": "converge"},
        )
        builder.add_edge("converge", END)
        return builder.compile()

    def _propose(self, state: WorkflowState) -> WorkflowState:
        """Request a candidate with all bounded downstream rejection context."""
        use_seed = "seed_candidate" in state and not state.get("seed_consumed", False)
        if use_seed:
            candidate = state["seed_candidate"]
        else:
            try:
                candidate = self._design_agent.propose(
                    state["design_goal"], state.get("rejection_history", [])
                )
            except ProposalError as exc:
                return self._update(
                    state,
                    "propose",
                    f"Proposal stopped: {exc}",
                    status="proposal_exhausted",
                    stop_reason=str(exc),
                )
        iteration = state.get("iteration_count", 0) + 1
        return self._update(
            state,
            "propose",
            "Candidate proposal accepted by schema and duplicate checks",
            current_candidate=candidate,
            iteration_count=iteration,
            **({"seed_consumed": True} if use_seed else {}),
        )

    def _simulate(self, state: WorkflowState) -> WorkflowState:
        """Convert the candidate to the physics input schema and run the pure cycle."""
        candidate = state["current_candidate"]
        inputs = CycleInput.model_validate(
            candidate.model_dump(exclude={"hot_section_material_name"})
        )
        performance = self._simulator(inputs, self._physics_config)
        return self._update(
            state,
            "simulate",
            "Physics simulation completed",
            performance_result=performance,
        )

    def _critique(self, state: WorkflowState) -> WorkflowState:
        """Attach code-determined constraint decisions and retrieved evidence."""
        critique = self._critique_agent.critique(
            state["current_candidate"], state["performance_result"]
        )
        event = (
            "Critique approved with citations"
            if critique.valid
            else "Critique rejected candidate"
        )
        return self._update(state, "critique", event, critique_result=critique)

    def _revise(self, state: WorkflowState) -> WorkflowState:
        """Record rejection reason and advance the bounded revision counter."""
        critique = state["critique_result"]
        reason = "; ".join(critique.constraint_violations) or (
            "Critique did not approve the candidate with cited evidence"
        )
        rejection = CandidateRejection(candidate=state["current_candidate"], reason=reason)
        history = [*state.get("rejection_history", []), rejection]
        count = state.get("invalid_revision_count", 0) + 1
        return self._update(
            state,
            "revise",
            f"Recorded rejection {count}: {reason}",
            rejection_history=history,
            invalid_revision_count=count,
        )

    def _score(self, state: WorkflowState) -> WorkflowState:
        """Score a valid cited candidate using the injected objective function."""
        candidate = state["current_candidate"]
        performance = state["performance_result"]
        critique = state["critique_result"]
        if not critique.valid or not critique.cited_sources:
            raise ValueError("Only valid candidates with citations may be scored")
        score = float(self._scorer(candidate, performance))
        if not math.isfinite(score):
            raise ValueError("Candidate scorer must return a finite score")
        trial = ScoredTrial(candidate, performance, critique, score)
        history = [*state.get("trial_history", []), trial]
        return self._update(
            state,
            "score",
            f"Scored cited candidate with objective {score:g}",
            current_score=score,
            trial_history=history,
        )

    def _optimizer_step(self, state: WorkflowState) -> WorkflowState:
        """Update best-so-far state and apply configured plateau/iteration convergence."""
        score = state["current_score"]
        previous_best = state.get("best_score")
        improved = previous_best is None or score > previous_best
        if improved:
            best_values = {
                "best_candidate": state["current_candidate"],
                "best_performance": state["performance_result"],
                "best_critique": state["critique_result"],
                "best_score": score,
            }
        else:
            best_values = {}

        if state.get("single_candidate", False):
            return self._update(
                state,
                "optimizer_step",
                "Completed one optimizer-sampled candidate evaluation",
                **best_values,
                status="candidate_evaluated",
                stop_reason="single-candidate evaluation completed",
            )

        if previous_best is not None:
            improvement = max(0.0, score - previous_best)
            if improvement <= self._config.convergence_tolerance:
                return self._update(
                    state,
                    "optimizer_step",
                    f"Converged: improvement {improvement:g} is within tolerance",
                    **best_values,
                    status="converged",
                    stop_reason="score improvement is within configured tolerance",
                )
        if state["iteration_count"] >= self._config.max_iterations:
            return self._update(
                state,
                "optimizer_step",
                "Stopped at configured maximum iterations",
                **best_values,
                status="max_iterations",
                stop_reason="configured maximum iterations reached",
            )
        return self._update(
            state,
            "optimizer_step",
            "Continuing search; best-so-far state updated",
            **best_values,
        )

    def _route_after_critique(self, state: WorkflowState) -> str:
        """Route only cited passing critiques to scoring."""
        critique = state["critique_result"]
        return "score" if critique.valid and critique.cited_sources else "revise"

    def _route_after_revision(self, state: WorkflowState) -> str:
        """Route revision exhaustion through an explicit terminal graph node."""
        if state.get("single_candidate", False):
            return "converge"
        if state["invalid_revision_count"] >= self._config.max_invalid_revisions:
            return "converge"
        if state["iteration_count"] >= self._config.max_iterations:
            return "converge"
        return "propose"

    def _route_after_optimizer_step(self, state: WorkflowState) -> str:
        """Continue only while best-so-far bookkeeping remains in running status."""
        if state.get("status", "running") != "running":
            return "converge"
        return "propose"

    def _converge(self, state: WorkflowState) -> WorkflowState:
        """Set terminal status after an exhausted revision branch or score convergence."""
        if state.get("status") == "candidate_evaluated":
            status = "candidate_evaluated"
            reason = state.get("stop_reason", "single-candidate evaluation completed")
        elif state.get("status") not in {"converged", "max_iterations"}:
            if state.get("invalid_revision_count", 0) >= self._config.max_invalid_revisions:
                status = "max_invalid_revisions"
                reason = "invalid revision limit reached"
            else:
                status = "max_iterations"
                reason = "configured maximum iterations reached"
        else:
            status = state["status"]
            reason = state.get("stop_reason", "configured convergence condition reached")
        return self._update(
            state,
            "converge",
            reason,
            status=status,
            stop_reason=reason,
        )

    def _update(
        self,
        state: WorkflowState,
        node: str,
        message: str,
        **values: object,
    ) -> WorkflowState:
        """Return a partial state update and append one structured audit event."""
        update: WorkflowState = dict(values)
        update["transition_history"] = [
            *state.get("transition_history", []),
            WorkflowEvent(
                node,
                int(values.get("iteration_count", state.get("iteration_count", 0))),
                message,
            ),
        ]
        if self._trace_sink is not None:
            trace_event = new_trace_event(
                state.get("trace_id", ""),
                state.get("last_span_id"),
                node,
                int(values.get("iteration_count", state.get("iteration_count", 0))),
                message,
            )
            self._trace_sink.emit(trace_event)
            update["last_span_id"] = trace_event.span_id
        logger.info(
            "Workflow transition",
            extra={
                "node": node,
                "iteration": state.get("iteration_count", 0),
                "message": message,
            },
        )
        return update

__all__ = [
    "CandidateScorer",
    "CycleSimulator",
    "DesignWorkflow",
    "ScoredTrial",
    "WorkflowEvent",
    "WorkflowState",
]
