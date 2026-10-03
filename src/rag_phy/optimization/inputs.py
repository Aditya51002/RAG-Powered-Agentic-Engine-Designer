"""Provenance-required candidate sampling and engine-weight lookup logic."""

from __future__ import annotations

import optuna

from rag_phy.agents import DesignCandidate
from rag_phy.config import CandidateSearchConfig, EngineWeightLookupConfig
from rag_phy.physics.models import CycleResult


class ConfiguredCandidateSampler:
    """Sample candidates only from externally supplied, source-validated bounds.

    Numeric dimensions use their configured uniform or log-uniform distribution. Log-uniform
    sampling is appropriate when a sourced range spans multiplicative scales; uniform sampling
    is appropriate when equal absolute increments are intended. The sampler contains no default
    engineering bounds and records the selected material category as a trial parameter.
    """

    def __init__(self, config: CandidateSearchConfig) -> None:
        self._config = config

    def __call__(self, trial: optuna.trial.Trial) -> DesignCandidate:
        sampled = {
            name: self._suggest_range(trial, name, getattr(self._config, name))
            for name in CandidateSearchConfig.model_fields
            if name != "hot_section_materials"
        }
        material_names = [material.name for material in self._config.hot_section_materials]
        sampled["hot_section_material_name"] = trial.suggest_categorical(
            "hot_section_material_name", material_names
        )
        return DesignCandidate.model_validate(sampled)

    @staticmethod
    def _suggest_range(trial: optuna.trial.Trial, name: str, bounds) -> float:
        """Delegate the distribution choice to validated configuration."""
        return trial.suggest_float(
            name,
            bounds.minimum,
            bounds.maximum,
            log=bounds.strategy == "log_uniform",
        )


class SourcedEngineWeightEstimator:
    """Interpolate published dry engine weight against rated thrust, without extrapolation.

    The estimate is a preliminary lookup interpolation, not structural mass analysis. Cycle
    thrust is treated as comparable to published rated thrust, and interpolation between anchors
    assumes a linear trend; both limitations must be reviewed for a target engine class. Anchors
    must be externally sourced and checked into a source-ledger-backed configuration.
    """

    def __init__(self, config: EngineWeightLookupConfig) -> None:
        self._anchors = tuple(sorted(config.anchors, key=lambda anchor: anchor.rated_thrust_n))
        self._standard_gravity_m_per_s2 = config.standard_gravity_m_per_s2

    def __call__(self, candidate: DesignCandidate, performance: CycleResult) -> float:
        del candidate
        thrust = performance.thrust_n
        lower = self._anchors[0]
        upper = self._anchors[-1]
        if thrust < lower.rated_thrust_n or thrust > upper.rated_thrust_n:
            raise ValueError(
                "Cycle thrust lies outside sourced engine-weight anchors; extrapolation is disabled"
            )

        for anchor in self._anchors:
            if thrust == anchor.rated_thrust_n:
                return anchor.dry_mass_kg * self._standard_gravity_m_per_s2

        for lower, upper in zip(self._anchors, self._anchors[1:]):
            if lower.rated_thrust_n < thrust < upper.rated_thrust_n:
                fraction = (thrust - lower.rated_thrust_n) / (
                    upper.rated_thrust_n - lower.rated_thrust_n
                )
                dry_mass_kg = lower.dry_mass_kg + fraction * (
                    upper.dry_mass_kg - lower.dry_mass_kg
                )
                return dry_mass_kg * self._standard_gravity_m_per_s2
        raise ValueError("No sourced engine-weight interval covers cycle thrust")


__all__ = ["ConfiguredCandidateSampler", "SourcedEngineWeightEstimator"]
