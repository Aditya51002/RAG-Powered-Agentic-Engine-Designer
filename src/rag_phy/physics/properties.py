"""CoolProp adapter for deterministic real-fluid property queries."""

from __future__ import annotations

import logging
from typing import Protocol

from CoolProp.CoolProp import PropsSI

logger = logging.getLogger(__name__)


class PropertyBackend(Protocol):
    """Interface for thermodynamic property evaluation."""

    def evaluate(
        self,
        output: str,
        first_key: str,
        first_value: float,
        second_key: str,
        second_value: float,
        fluid: str,
    ) -> float:
        """Return a CoolProp property in its SI unit system."""


class CoolPropBackend:
    """Evaluate thermodynamic properties using CoolProp's SI PropsSI interface."""

    def evaluate(
        self,
        output: str,
        first_key: str,
        first_value: float,
        second_key: str,
        second_value: float,
        fluid: str,
    ) -> float:
        """Evaluate one property and report property-library failures explicitly."""
        try:
            result = PropsSI(
                output,
                first_key,
                first_value,
                second_key,
                second_value,
                fluid,
            )
        except Exception as exc:
            logger.exception(
                "CoolProp query failed",
                extra={
                    "property": output,
                    "fluid": fluid,
                    "first_pair": (first_key, first_value),
                    "second_pair": (second_key, second_value),
                },
            )
            raise ValueError(f"CoolProp could not evaluate {output} for {fluid}") from exc
        return float(result)


__all__ = ["CoolPropBackend", "PropertyBackend"]
