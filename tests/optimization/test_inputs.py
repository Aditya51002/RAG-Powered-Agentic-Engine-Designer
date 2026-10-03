"""Pure logic tests for sourced optimizer inputs; all values here are synthetic."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_phy.agents import DesignCandidate
from rag_phy.config import (
    CandidateSearchConfig,
    EngineWeightLookupConfig,
    load_candidate_search_config,
    load_engine_weight_lookup_config,
)
from rag_phy.optimization import ConfiguredCandidateSampler, SourcedEngineWeightEstimator
from rag_phy.physics.models import CycleResult


def _search_config() -> CandidateSearchConfig:
    source = ("SYNTHETIC-TEST",)
    return CandidateSearchConfig.model_validate(
        {
            "ambient_temperature_k": {
                "minimum": 280,
                "maximum": 300,
                "unit": "K",
                "strategy": "uniform",
                "source_ids": source,
            },
            "ambient_pressure_pa": {
                "minimum": 90000,
                "maximum": 110000,
                "unit": "Pa",
                "strategy": "log_uniform",
                "source_ids": source,
            },
            "flight_speed_m_per_s": {
                "minimum": 0,
                "maximum": 300,
                "unit": "m/s",
                "strategy": "uniform",
                "source_ids": source,
            },
            "air_mass_flow_kg_per_s": {
                "minimum": 1,
                "maximum": 20,
                "unit": "kg/s",
                "strategy": "log_uniform",
                "source_ids": source,
            },
            "compressor_pressure_ratio": {
                "minimum": 2,
                "maximum": 20,
                "unit": "1",
                "strategy": "log_uniform",
                "source_ids": source,
            },
            "turbine_inlet_temperature_k": {
                "minimum": 900,
                "maximum": 1400,
                "unit": "K",
                "strategy": "uniform",
                "source_ids": source,
            },
            "hot_section_materials": [{"name": "SYNTHETIC-ALLOY", "source_id": source[0]}],
        }
    )


class _Trial:
    def __init__(self) -> None:
        self.values: dict[str, float | str] = {}
        self.log_flags: dict[str, bool] = {}

    def suggest_float(self, name, low, high, log=False):
        self.log_flags[name] = log
        value = (low * high) ** 0.5 if log else (low + high) / 2
        self.values[name] = value
        return value

    def suggest_categorical(self, name, choices):
        self.values[name] = choices[0]
        return choices[0]


def _cycle(thrust_n: float) -> CycleResult:
    return CycleResult(thrust_n, 0.00003, 0.3, 0.02, ())


def test_configured_sampler_uses_configured_distributions() -> None:
    trial = _Trial()
    candidate = ConfiguredCandidateSampler(_search_config())(trial)

    assert isinstance(candidate, DesignCandidate)
    assert trial.log_flags["compressor_pressure_ratio"]
    assert trial.log_flags["air_mass_flow_kg_per_s"]
    assert not trial.log_flags["flight_speed_m_per_s"]
    assert candidate.hot_section_material_name == "SYNTHETIC-ALLOY"


def test_search_bounds_require_resolved_source_ledger_ids(tmp_path: Path) -> None:
    import yaml

    search_path = tmp_path / "search.yaml"
    search_path.write_text(yaml.safe_dump(_search_config().model_dump(mode="json")), encoding="utf-8")
    ledger = tmp_path / "sources.md"
    ledger.write_text("## OTHER-SOURCE\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SYNTHETIC-TEST"):
        load_candidate_search_config(search_path, ledger)


def test_engine_weight_lookup_interpolates_and_refuses_extrapolation() -> None:
    lookup = EngineWeightLookupConfig.model_validate(
        {
            "anchors": [
                {"rated_thrust_n": 100, "dry_mass_kg": 40, "source_id": "SYNTHETIC"},
                {"rated_thrust_n": 500, "dry_mass_kg": 200, "source_id": "SYNTHETIC"},
            ],
            "standard_gravity_m_per_s2": 10,
            "standard_gravity_source_id": "SYNTHETIC",
        }
    )
    estimator = SourcedEngineWeightEstimator(lookup)
    candidate = DesignCandidate(
        ambient_temperature_k=290,
        ambient_pressure_pa=100000,
        flight_speed_m_per_s=0,
        air_mass_flow_kg_per_s=5,
        compressor_pressure_ratio=5,
        turbine_inlet_temperature_k=1000,
        hot_section_material_name="SYNTHETIC-ALLOY",
    )

    assert estimator(candidate, _cycle(300)) == pytest.approx(1200)
    assert estimator(candidate, _cycle(100)) == 400
    with pytest.raises(ValueError, match="extrapolation is disabled"):
        estimator(candidate, _cycle(600))


def test_nasa_engine_weight_dataset_loads_with_ledger_provenance() -> None:
    root = Path(__file__).parents[2]
    config = load_engine_weight_lookup_config(
        root / "config" / "engine_weight.yaml",
        root / "data" / "curated" / "sources.md",
    )
    estimator = SourcedEngineWeightEstimator(config)
    candidate = DesignCandidate(
        ambient_temperature_k=290,
        ambient_pressure_pa=100000,
        flight_speed_m_per_s=0,
        air_mass_flow_kg_per_s=5,
        compressor_pressure_ratio=5,
        turbine_inlet_temperature_k=1000,
        hot_section_material_name="SYNTHETIC-ALLOY",
    )

    assert len(config.anchors) == 5
    assert [
        (anchor.rated_thrust_n, anchor.dry_mass_kg) for anchor in config.anchors
    ] == [
        (12677, 181),
        (13789, 185),
        (49817, 1270),
        (60048, 1920),
        (70278, 2277),
    ]
    assert estimator(candidate, _cycle(12677)) == pytest.approx(181 * 9.80665)
    assert estimator(candidate, _cycle(70278)) == pytest.approx(2277 * 9.80665)
