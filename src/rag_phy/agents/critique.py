"""Source-grounded design critique with code-owned validity decisions."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from rag_phy.agents.design import DesignCandidate
from rag_phy.config import CritiqueAgentConfig, ModelsConfig
from rag_phy.ingestion.embeddings import Embedder
from rag_phy.knowledge.constraints import (
    ConstraintNotFoundError,
    MaterialConstraint,
    MaterialConstraintStore,
)
from rag_phy.knowledge.vector_store import RetrievalResult, VectorStore
from rag_phy.physics.models import CycleResult

logger = logging.getLogger(__name__)


class CritiqueResult(BaseModel):
    """Code-determined validity, explanation, source citations, and constraint failures."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    valid: bool
    reasoning: str
    cited_sources: tuple[str, ...]
    constraint_violations: tuple[str, ...]
    normalized_violation_severity: float = Field(default=0.0, ge=0, allow_inf_nan=False)


class CritiqueExplainer(Protocol):
    """Injected text generator used only to explain evidence and computed checks."""

    def complete(self, prompt: str) -> str:
        """Return an explanation for the supplied evidence-only critique prompt."""


class CritiqueExplanationError(RuntimeError):
    """Raised when evidence-based explanation generation fails or returns no text."""


class CritiqueAgent:
    """Compare selected material limits in code and explain using retrieved evidence."""

    def __init__(
        self,
        constraint_store: MaterialConstraintStore,
        vector_store: VectorStore,
        embedder: Embedder,
        explainer: CritiqueExplainer,
        models_config: ModelsConfig,
        config: CritiqueAgentConfig,
    ) -> None:
        """Bind structured/vector knowledge and injected embedding/explanation adapters."""
        self._constraint_store = constraint_store
        self._vector_store = vector_store
        self._embedder = embedder
        self._explainer = explainer
        self._models_config = models_config
        self._config = config

    def critique(self, candidate: DesignCandidate, performance: CycleResult) -> CritiqueResult:
        """Return validity from direct constraint comparisons plus cited retrieved evidence.

        Assumption: until a blade-metal thermal model exists, turbine inlet gas temperature
        is compared directly with the selected material's maximum service temperature as a
        screening proxy. It is not a metal-temperature prediction and cannot establish
        thermal safety; it may reject feasible cooled turbine designs.

        Args:
            candidate: Candidate cycle point; temperatures K, pressure Pa, flow kg/s.
            performance: Deterministic cycle result; thrust N and SFC kg/(N s).

        Returns:
            Typed critique with validity decided without the explainer, plus citation IDs.

        Raises:
            CritiqueExplanationError: If the injected explanation service fails or is empty.
        """
        violations: list[str] = []
        normalized_severity = 0.0
        source_ids: list[str] = []
        material = None
        try:
            material = self._constraint_store.get_material(
                candidate.hot_section_material_name
            )
            source_ids.append(material.source_id)
            limit_k = material.max_service_temperature_k
            if candidate.turbine_inlet_temperature_k > limit_k:
                normalized_severity = (
                    candidate.turbine_inlet_temperature_k - limit_k
                ) / limit_k
                violations.append(
                    f"Turbine inlet temperature {candidate.turbine_inlet_temperature_k:g} K "
                    f"exceeds {material.material_name} maximum service temperature "
                    f"{limit_k:g} K (source: {material.source_id})"
                )
        except ConstraintNotFoundError:
            violations.append(
                "No curated material constraint exists for "
                f"{candidate.hot_section_material_name!r}"
            )

        retrieval_query = self._retrieval_query(candidate, performance)
        query_embedding = self._embedder.embed_query(retrieval_query)
        retrieved = self._vector_store.search(
            query_embedding, self._models_config.vector_store.retrieval_candidates
        )
        document_sources = [
            source for result in retrieved for source in result.source_refs if source.strip()
        ]
        source_ids.extend(document_sources)
        if not document_sources:
            violations.append("No source-backed literature passage was retrieved")

        cited_sources = tuple(dict.fromkeys(source_ids))
        numeric_checks_pass = not violations
        if numeric_checks_pass and not cited_sources:
            logger.warning("Critique cannot approve a design without cited sources")
            violations.append("A passing design requires at least one cited source")
        valid = numeric_checks_pass and bool(cited_sources)

        prompt = self._render_prompt(candidate, performance, material, retrieved, violations)
        try:
            reasoning = self._explainer.complete(prompt).strip()
        except Exception as exc:
            logger.exception("Critique explanation generation failed")
            raise CritiqueExplanationError("Could not generate critique explanation") from exc
        if not reasoning:
            raise CritiqueExplanationError("Critique explainer returned an empty explanation")

        result = CritiqueResult(
            valid=valid,
            reasoning=reasoning,
            cited_sources=cited_sources,
            constraint_violations=tuple(violations),
            normalized_violation_severity=normalized_severity,
        )
        logger.info(
            "Design critique completed",
            extra={
                "valid": result.valid,
                "citation_count": len(result.cited_sources),
                "constraint_violation_count": len(result.constraint_violations),
            },
        )
        return result

    def _retrieval_query(self, candidate: DesignCandidate, performance: CycleResult) -> str:
        """Build a reproducible query covering proposed operating point and performance."""
        return (
            f"Turbojet turbine hot section material {candidate.hot_section_material_name}; "
            f"compressor pressure ratio {candidate.compressor_pressure_ratio:g}; "
            f"turbine inlet temperature {candidate.turbine_inlet_temperature_k:g} K; "
            f"thrust {performance.thrust_n:g} N; specific fuel consumption "
            f"{performance.specific_fuel_consumption_kg_per_n_s:g} kg/(N s)"
        )

    def _render_prompt(
        self,
        candidate: DesignCandidate,
        performance: CycleResult,
        material: MaterialConstraint | None,
        retrieved: Sequence[RetrievalResult],
        violations: Sequence[str],
    ) -> str:
        """Render configured explanation prompt using only code-computed checks and sources."""
        constraints = {
            "material_record": asdict(material) if material is not None else None,
            "violations": list(violations),
            "validity_is_determined_by_code": True,
        }
        passages = [
            {
                "text": result.text,
                "source_refs": result.source_refs,
            }
            for result in retrieved
        ]
        return self._config.prompt_template.format(
            candidate=json.dumps(candidate.model_dump(mode="json"), sort_keys=True),
            performance=json.dumps(asdict(performance), sort_keys=True),
            constraints=json.dumps(constraints, sort_keys=True),
            retrieved_passages=json.dumps(passages, sort_keys=True),
        )


__all__ = [
    "CritiqueAgent",
    "CritiqueExplanationError",
    "CritiqueExplainer",
    "CritiqueResult",
]
