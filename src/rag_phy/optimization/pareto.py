"""Incremental non-dominated frontier for thrust-to-weight and SFC."""

from __future__ import annotations

import math
from dataclasses import dataclass

from rag_phy.agents import DesignCandidate


@dataclass(frozen=True)
class ParetoPoint:
    """Valid design metrics; T/W is maximized and SFC (kg/(N s)) minimized."""

    candidate: DesignCandidate
    thrust_to_weight_ratio: float
    specific_fuel_consumption_kg_per_n_s: float
    cited_sources: tuple[str, ...]


class ParetoFrontier:
    """Maintain a deduplicated non-dominated set with O(frontier size) updates."""

    def __init__(self, signature_decimal_places: int) -> None:
        """Configure canonical candidate deduplication precision."""
        if signature_decimal_places < 0:
            raise ValueError("Signature precision must be non-negative")
        self._precision = signature_decimal_places
        self._points: list[ParetoPoint] = []
        self._signatures: set[str] = set()

    @property
    def points(self) -> tuple[ParetoPoint, ...]:
        """Return a stable snapshot of the current non-dominated designs."""
        return tuple(self._points)

    def add(self, point: ParetoPoint) -> bool:
        """Insert a candidate unless duplicated or dominated by the current frontier.

        Returns:
            True if the point entered the frontier; otherwise False.
        """
        if not point.cited_sources:
            raise ValueError("Pareto points must retain at least one source citation")
        if not all(
            math.isfinite(metric) and metric > 0
            for metric in (
                point.thrust_to_weight_ratio,
                point.specific_fuel_consumption_kg_per_n_s,
            )
        ):
            raise ValueError("Pareto metrics must be positive")
        signature = point.candidate.parameter_signature(self._precision)
        if signature in self._signatures:
            return False
        if any(self._dominates(current, point) for current in self._points):
            return False

        survivors: list[ParetoPoint] = []
        for current in self._points:
            current_signature = current.candidate.parameter_signature(self._precision)
            if self._dominates(point, current):
                self._signatures.discard(current_signature)
            else:
                survivors.append(current)
        survivors.append(point)
        self._points = survivors
        self._signatures.add(signature)
        return True

    @staticmethod
    def _dominates(left: ParetoPoint, right: ParetoPoint) -> bool:
        """Apply standard strict Pareto dominance for maximize T/W, minimize SFC."""
        no_worse = (
            left.thrust_to_weight_ratio >= right.thrust_to_weight_ratio
            and left.specific_fuel_consumption_kg_per_n_s
            <= right.specific_fuel_consumption_kg_per_n_s
        )
        strictly_better = (
            left.thrust_to_weight_ratio > right.thrust_to_weight_ratio
            or left.specific_fuel_consumption_kg_per_n_s
            < right.specific_fuel_consumption_kg_per_n_s
        )
        return no_worse and strictly_better


__all__ = ["ParetoFrontier", "ParetoPoint"]
