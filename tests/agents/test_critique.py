"""Offline tests proving critique validity is decided by structured constraints."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_phy.agents import CritiqueAgent, DesignCandidate
from rag_phy.config import load_agent_prompts_config, load_models_config
from rag_phy.knowledge import (
    MaterialConstraintStore,
    RetrievalResult,
)
from rag_phy.physics.models import CycleResult


class FixedEmbedder:
    """Injected deterministic test embedder."""

    def embed_query(self, text: str) -> list[float]:
        """Return a fixed-dimensional vector without loading a model."""
        return [1.0] * 384


class FixedVectorStore:
    """Injected source-bearing retrieval fixture."""

    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.queries: list[tuple[list[float], int]] = []

    def search(self, query_embedding, limit: int) -> list[RetrievalResult]:
        """Record query parameters and return configured passages."""
        self.queries.append((list(query_embedding), limit))
        return self.results


class FixedExplainer:
    """Capturing explanation stub that cannot affect validity."""

    def __init__(self, text: str = "Explanation based on supplied evidence.") -> None:
        self.text = text
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        """Capture evidence supplied to the explanation model."""
        self.prompts.append(prompt)
        return self.text


def _candidate(material: str = "TEST_ALLOY_ALPHA", tit_k: float = 1100.0) -> DesignCandidate:
    """Build an explicitly synthetic candidate with temperatures in K and pressures in Pa."""
    return DesignCandidate(
        ambient_temperature_k=288.15,
        ambient_pressure_pa=101325.0,
        flight_speed_m_per_s=0.0,
        air_mass_flow_kg_per_s=10.0,
        compressor_pressure_ratio=8.0,
        turbine_inlet_temperature_k=tit_k,
        hot_section_material_name=material,
    )


def _performance() -> CycleResult:
    """Return synthetic performance fixture; thrust N and SFC kg/(N s)."""
    return CycleResult(
        thrust_n=10000.0,
        specific_fuel_consumption_kg_per_n_s=0.00002,
        thermal_efficiency=0.35,
        fuel_air_ratio=0.02,
        stations=(),
    )


def _agent(results: list[RetrievalResult], explainer: FixedExplainer | None = None) -> CritiqueAgent:
    """Wire production typed settings to synthetic, injected Phase 4 adapters."""
    root = Path(__file__).parents[2]
    constraints = MaterialConstraintStore.from_csv(
        root / "tests" / "fixtures" / "material_constraints_synthetic.csv"
    )
    prompts = load_agent_prompts_config(root / "config" / "agent_prompts.yaml")
    return CritiqueAgent(
        constraint_store=constraints,
        vector_store=FixedVectorStore(results),
        embedder=FixedEmbedder(),
        explainer=explainer or FixedExplainer(),
        models_config=load_models_config(root / "config" / "models.yaml"),
        config=prompts.critique_agent,
    )


def _retrieved() -> list[RetrievalResult]:
    """Return a clearly synthetic source-bearing passage fixture."""
    return [
        RetrievalResult(
            chunk_id="synthetic-chunk-1",
            text="Synthetic passage about turbine temperature screening.",
            distance=0.1,
            source_refs=("synthetic-literature:test-source-1",),
        )
    ]


def test_excess_temperature_is_rejected_and_constraint_is_cited() -> None:
    explainer = FixedExplainer("This design is valid and safe.")
    critique = _agent(_retrieved(), explainer).critique(_candidate(tit_k=1300), _performance())

    assert not critique.valid
    assert any("exceeds TEST_ALLOY_ALPHA maximum service temperature 1200 K" in item
               for item in critique.constraint_violations)
    assert "synthetic-test-record-alpha" in critique.cited_sources
    assert critique.normalized_violation_severity == pytest.approx(100 / 1200)
    assert "validity_is_determined_by_code" in explainer.prompts[0]
    assert critique.reasoning == "This design is valid and safe."


def test_within_limit_requires_retrieved_citation_to_be_valid() -> None:
    critique = _agent(_retrieved()).critique(_candidate(tit_k=1100), _performance())

    assert critique.valid
    assert critique.cited_sources == (
        "synthetic-test-record-alpha",
        "synthetic-literature:test-source-1",
    )
    assert critique.constraint_violations == ()


def test_missing_material_fails_closed_even_with_literature() -> None:
    critique = _agent(_retrieved()).critique(
        _candidate(material="UNKNOWN_TEST_MATERIAL"), _performance()
    )

    assert not critique.valid
    assert critique.cited_sources == ("synthetic-literature:test-source-1",)
    assert any("No curated material constraint" in item for item in critique.constraint_violations)


def test_no_retrieved_sources_cannot_produce_valid_verdict() -> None:
    critique = _agent([]).critique(_candidate(tit_k=1100), _performance())

    assert not critique.valid
    assert "synthetic-test-record-alpha" in critique.cited_sources
    assert any("No source-backed literature" in item for item in critique.constraint_violations)
