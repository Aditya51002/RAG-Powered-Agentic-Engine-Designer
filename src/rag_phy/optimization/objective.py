"""Normalized scalar design objective and explicit infeasibility penalty."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from rag_phy.agents import CritiqueResult
from rag_phy.config import ObjectiveConfig
from rag_phy.physics.models import CycleResult

logger = logging.getLogger(__name__)


class ObjectiveInputError(ValueError):
    """Raised when objective metrics contain invalid physical values."""


@dataclass(frozen=True)
class ObjectiveResult:
    """Normalized objective decomposition; T/W and score are dimensionless."""

    score: float
    thrust_to_weight_ratio: float
    specific_fuel_consumption_kg_per_n_s: float
    valid: bool
    penalty: float


class DesignObjective:
    """Compute weighted thrust-to-weight/SFC utility and penalize invalid designs."""

    def __init__(self, config: ObjectiveConfig) -> None:
        """Bind normalized objective weights, references, and penalty coefficients."""
        self._config = config

    def evaluate(
        self,
        performance: CycleResult,
        critique: CritiqueResult,
        engine_weight_n: float,
    ) -> ObjectiveResult:
        """Calculate score from thrust (N), engine weight (N), and SFC (kg/(N s)).

        The valid score is `w_ttw * (T/W / T/W_ref) - w_sfc * (SFC / SFC_ref)`.
        Invalid designs subtract `base_penalty + severity_weight * normalized_severity`;
        the severity is supplied by deterministic structured checks, never by an LLM.
        """
        values = (
            performance.thrust_n,
            performance.specific_fuel_consumption_kg_per_n_s,
            engine_weight_n,
        )
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ObjectiveInputError(
                "Thrust, SFC, and estimated engine weight must be finite positive values"
            )
        ratio = performance.thrust_n / engine_weight_n
        normalized_ttw = ratio / self._config.thrust_to_weight_reference
        normalized_sfc = (
            performance.specific_fuel_consumption_kg_per_n_s
            / self._config.specific_fuel_consumption_reference_kg_per_n_s
        )
        valid = critique.valid and bool(critique.cited_sources)
        penalty = 0.0
        if not valid:
            penalty = (
                self._config.invalid_base_penalty
                + self._config.invalid_severity_penalty_weight
                * critique.normalized_violation_severity
            )
        score = (
            self._config.thrust_to_weight_weight * normalized_ttw
            - self._config.specific_fuel_consumption_weight * normalized_sfc
            - penalty
        )
        if not math.isfinite(score):
            raise ObjectiveInputError("Objective score is not finite")
        result = ObjectiveResult(
            score=score,
            thrust_to_weight_ratio=ratio,
            specific_fuel_consumption_kg_per_n_s=(
                performance.specific_fuel_consumption_kg_per_n_s
            ),
            valid=valid,
            penalty=penalty,
        )
        logger.info(
            "Candidate objective evaluated",
            extra={
                "score": result.score,
                "valid": result.valid,
                "penalty": result.penalty,
                "thrust_to_weight_ratio": result.thrust_to_weight_ratio,
            },
        )
        return result


__all__ = ["DesignObjective", "ObjectiveInputError", "ObjectiveResult"]
