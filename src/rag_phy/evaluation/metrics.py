"""Cumulative design-validity metrics and chart export."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidityObservation:
    """Boolean validity outcome for one completed optimizer trial."""

    iteration: int
    valid: bool


@dataclass(frozen=True)
class ValidityRatePoint:
    """Cumulative valid-trial fraction after a particular completed iteration."""

    iteration: int
    valid_count: int
    trial_count: int
    validity_rate: float


def cumulative_validity_rate(
    observations: Iterable[ValidityObservation],
) -> tuple[ValidityRatePoint, ...]:
    """Compute running validity rate in strictly increasing iteration order."""
    ordered = sorted(observations, key=lambda row: row.iteration)
    if not ordered:
        raise ValueError("At least one validity observation is required")
    iterations = [row.iteration for row in ordered]
    if any(iteration < 1 for iteration in iterations) or len(set(iterations)) != len(iterations):
        raise ValueError("Observation iterations must be unique positive integers")
    valid_count = 0
    points: list[ValidityRatePoint] = []
    for trial_count, row in enumerate(ordered, start=1):
        valid_count += int(row.valid)
        points.append(
            ValidityRatePoint(
                iteration=row.iteration,
                valid_count=valid_count,
                trial_count=trial_count,
                validity_rate=valid_count / trial_count,
            )
        )
    return tuple(points)


def validity_observations_from_study(study: object) -> tuple[ValidityObservation, ...]:
    """Extract completed Optuna outcomes; fail if a completed trial lacks validity metadata."""
    try:
        from optuna.trial import TrialState
    except ImportError as exc:
        raise RuntimeError("Install rag-phy[optimization] to read Optuna study history") from exc
    observations = []
    for trial in study.trials:
        if trial.state is not TrialState.COMPLETE:
            continue
        if "valid" not in trial.user_attrs:
            raise ValueError(f"Completed trial {trial.number} has no 'valid' user attribute")
        observations.append(ValidityObservation(trial.number + 1, bool(trial.user_attrs["valid"])))
    return tuple(observations)


def save_validity_chart(points: Iterable[ValidityRatePoint], path: str | Path) -> None:
    """Render a cumulative validity-rate PNG without requiring matplotlib at import time."""
    rows = tuple(points)
    if not rows:
        raise ValueError("At least one validity rate point is required")
    if any(
        not math.isfinite(row.validity_rate) or not 0 <= row.validity_rate <= 1
        for row in rows
    ):
        raise ValueError("Validity rates must be finite and within [0, 1]")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install rag-phy[evaluation] to render validity charts") from exc

    chart_path = Path(path)
    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    try:
        axis.plot(
            [row.iteration for row in rows],
            [row.validity_rate for row in rows],
            marker="o",
            linewidth=1.8,
        )
        axis.set(xlabel="Optimizer iteration", ylabel="Cumulative validity rate", ylim=(0, 1))
        axis.grid(True, alpha=0.25)
        chart_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(chart_path, dpi=160)
    except OSError:
        logger.exception("Failed to write validity chart", extra={"path": str(chart_path)})
        raise
    finally:
        plt.close(figure)


__all__ = [
    "ValidityObservation",
    "ValidityRatePoint",
    "cumulative_validity_rate",
    "save_validity_chart",
    "validity_observations_from_study",
]
