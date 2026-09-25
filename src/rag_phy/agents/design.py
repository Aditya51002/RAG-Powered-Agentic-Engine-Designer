"""Structured, dependency-injected design proposal agent."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rag_phy.config import DesignAgentConfig

logger = logging.getLogger(__name__)


class DesignCandidate(BaseModel):
    """One proposed turbojet cycle point in SI units."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    ambient_temperature_k: float = Field(gt=0)
    ambient_pressure_pa: float = Field(gt=0)
    flight_speed_m_per_s: float = Field(ge=0)
    air_mass_flow_kg_per_s: float = Field(gt=0)
    compressor_pressure_ratio: float = Field(gt=1)
    turbine_inlet_temperature_k: float = Field(gt=0)
    hot_section_material_name: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_temperature_order(self) -> DesignCandidate:
        """Require turbine inlet temperature (K) to exceed ambient temperature (K)."""
        if self.turbine_inlet_temperature_k <= self.ambient_temperature_k:
            raise ValueError("Turbine inlet temperature must exceed ambient temperature")
        return self

    def parameter_signature(self, decimal_places: int) -> str:
        """Hash a canonicalized set of cycle parameters; decimals are config-controlled."""
        if decimal_places < 0:
            raise ValueError("Signature decimal places must be non-negative")
        canonical = {
            name: round(value, decimal_places) if isinstance(value, float) else value
            for name, value in self.model_dump(mode="python").items()
        }
        serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class CandidateRejection(BaseModel):
    """A prior candidate and the reason it was rejected by downstream validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: DesignCandidate
    reason: str = Field(min_length=1)


class LLMClient(Protocol):
    """Minimal interface for an injected text-generation client."""

    def complete(self, prompt: str) -> str:
        """Return one generated response for the supplied prompt."""


class ProposalError(RuntimeError):
    """Base error for design proposal failures."""


class LLMClientError(ProposalError):
    """Raised when the injected text-generation service fails."""


class ProposalExhaustedError(ProposalError):
    """Raised when all configured structured-output attempts fail."""


class DuplicateCandidateError(ProposalError):
    """Raised when the model only proposes candidates already rejected upstream."""


class DesignAgent:
    """Generate schema-valid, non-rejected turbojet design proposals."""

    def __init__(self, client: LLMClient, config: DesignAgentConfig) -> None:
        """Bind an injected generation client and validated behavior settings."""
        self._client = client
        self._config = config

    def propose(
        self,
        goal: str,
        rejection_history: Sequence[CandidateRejection] = (),
    ) -> DesignCandidate:
        """Propose a candidate while avoiding a bounded, deduplicated rejection history.

        Args:
            goal: Natural-language design objective supplied by the caller.
            rejection_history: Prior candidates and their downstream rejection reasons.

        Returns:
            A strictly validated cycle candidate with temperatures K, pressures Pa,
            speed m/s, and mass flow kg/s.

        Raises:
            ValueError: If the goal is empty.
            LLMClientError: If the injected generation client fails.
            ProposalExhaustedError: If no unique schema-valid candidate is generated.
            DuplicateCandidateError: If every bounded attempt repeats a rejection.
        """
        if not goal.strip():
            raise ValueError("Design goal must not be empty")
        history = self._bounded_history(rejection_history)
        rejected_signatures = {
            item.candidate.parameter_signature(self._config.signature_decimal_places)
            for item in history
        }
        feedback = "none"
        last_error = "No valid candidate returned"
        saw_duplicate = False

        for attempt in range(1, self._config.max_attempts + 1):
            prompt = self._render_prompt(goal.strip(), history, feedback)
            try:
                raw_response = self._client.complete(prompt)
            except Exception as exc:
                logger.exception("Design proposal generation failed", extra={"attempt": attempt})
                raise LLMClientError("Design proposal client failed") from exc

            try:
                candidate = DesignCandidate.model_validate_json(raw_response)
            except (ValidationError, ValueError) as exc:
                last_error = f"Response did not match the required schema: {exc}"
                feedback = last_error
                logger.warning(
                    "Rejected malformed design proposal",
                    extra={"attempt": attempt, "error": str(exc)},
                )
                continue

            signature = candidate.parameter_signature(self._config.signature_decimal_places)
            if signature in rejected_signatures:
                saw_duplicate = True
                last_error = "Candidate repeats a previously rejected parameter set"
                feedback = last_error
                logger.warning(
                    "Rejected duplicate design proposal",
                    extra={"attempt": attempt, "signature": signature},
                )
                continue

            logger.info("Accepted structured design proposal", extra={"attempt": attempt})
            return candidate

        if saw_duplicate and last_error == "Candidate repeats a previously rejected parameter set":
            raise DuplicateCandidateError(
                f"No unique candidate after {self._config.max_attempts} attempts"
            )
        raise ProposalExhaustedError(
            f"No valid candidate after {self._config.max_attempts} attempts: {last_error}"
        )

    def _bounded_history(
        self, history: Sequence[CandidateRejection]
    ) -> tuple[CandidateRejection, ...]:
        """Deduplicate by canonical signature and retain the most recent configured items."""
        unique: dict[str, CandidateRejection] = {}
        for item in history:
            signature = item.candidate.parameter_signature(
                self._config.signature_decimal_places
            )
            unique[signature] = item
        limit = self._config.rejection_history_limit
        if limit == 0:
            return ()
        return tuple(list(unique.values())[-limit:])

    def _render_prompt(
        self,
        goal: str,
        history: Sequence[CandidateRejection],
        feedback: str,
    ) -> str:
        """Render the configured prompt with a schema and bounded rejected designs."""
        rejection_payload = [
            {
                "candidate": item.candidate.model_dump(mode="json"),
                "reason": item.reason,
            }
            for item in history
        ]
        return self._config.prompt_template.format(
            schema=json.dumps(DesignCandidate.model_json_schema(), sort_keys=True),
            goal=goal,
            rejections=json.dumps(rejection_payload, sort_keys=True),
            feedback=feedback,
        )


__all__ = [
    "CandidateRejection",
    "DesignAgent",
    "DesignCandidate",
    "DuplicateCandidateError",
    "LLMClient",
    "LLMClientError",
    "ProposalError",
    "ProposalExhaustedError",
]
