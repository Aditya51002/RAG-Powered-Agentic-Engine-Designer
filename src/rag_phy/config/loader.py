"""Load and validate application configuration from YAML and environment."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class LoggingConfig(BaseModel):
    """Logging output settings."""

    model_config = ConfigDict(extra="forbid")

    level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    json_output: bool = Field(default=True, alias="json")


class DashboardConfig(BaseModel):
    """Optional import path for the deployment-owned, fully configured app service."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    service_factory: str | None = None

    @model_validator(mode="after")
    def validate_service_factory(self) -> DashboardConfig:
        if self.service_factory is not None and not self.service_factory.strip():
            raise ValueError("dashboard.service_factory must be a non-empty import path")
        return self


class LLMConfig(BaseModel):
    """Anthropic request policy, pricing metadata, and environment-injected secret."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["anthropic"]
    api_key: str | None = Field(default=None, repr=False, exclude=True)
    model_name: str = Field(min_length=1)
    max_tokens: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0, allow_inf_nan=False)
    transient_retry_attempts: int = Field(ge=0)
    retry_initial_delay_seconds: float = Field(ge=0, allow_inf_nan=False)
    retry_max_delay_seconds: float = Field(gt=0, allow_inf_nan=False)
    input_cost_usd_per_million_tokens: float = Field(ge=0, allow_inf_nan=False)
    output_cost_usd_per_million_tokens: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_retry_delay_order(self) -> LLMConfig:
        if self.retry_initial_delay_seconds > self.retry_max_delay_seconds:
            raise ValueError("Initial retry delay cannot exceed retry maximum delay")
        return self


class AppConfig(BaseModel):
    """Validated settings shared across the application."""

    model_config = ConfigDict(extra="forbid")

    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    llm: LLMConfig | None = None


class ComponentEfficiencies(BaseModel):
    """Isentropic or component efficiencies, dimensionless and in (0, 1]."""

    model_config = ConfigDict(extra="forbid")

    intake: float = Field(gt=0, le=1)
    compressor: float = Field(gt=0, le=1)
    combustor: float = Field(gt=0, le=1)
    turbine: float = Field(gt=0, le=1)
    mechanical: float = Field(gt=0, le=1)
    nozzle: float = Field(gt=0, le=1)


class NumericalConfig(BaseModel):
    """Numerical solver controls; tolerance has pressure units (Pa)."""

    model_config = ConfigDict(extra="forbid")

    root_pressure_tolerance_pa: float = Field(gt=0)
    root_max_iterations: int = Field(gt=0)


class PhysicsConfig(BaseModel):
    """Fluid model, sourced fuel data, and station-model configuration."""

    model_config = ConfigDict(extra="forbid")

    working_fluid: str = Field(min_length=1)
    fuel_lower_heating_value_j_per_kg: float = Field(gt=0)
    temperature_min_k: float = Field(gt=0)
    temperature_max_k: float = Field(gt=0)
    pressure_max_pa: float = Field(gt=0)
    combustor_pressure_loss_fraction: float = Field(ge=0, lt=1)
    component_efficiencies: ComponentEfficiencies
    numerics: NumericalConfig

    @model_validator(mode="after")
    def validate_property_bounds(self) -> PhysicsConfig:
        """Require the configured EOS temperature interval to be ordered."""
        if self.temperature_min_k >= self.temperature_max_k:
            raise ValueError("temperature_min_k must be below temperature_max_k")
        return self


class ChunkingConfig(BaseModel):
    """Text chunk sizing controls measured in Unicode characters."""

    model_config = ConfigDict(extra="forbid")

    chunk_size_characters: int = Field(gt=0)
    overlap_characters: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> ChunkingConfig:
        """Require overlap to be smaller than the chunk to guarantee forward progress."""
        if self.overlap_characters >= self.chunk_size_characters:
            raise ValueError("overlap_characters must be smaller than chunk_size_characters")
        return self


class EmbeddingConfig(BaseModel):
    """Sentence embedding provider and dimensionality settings."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    expected_dimension: int = Field(gt=0)
    max_sequence_tokens: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    normalize_embeddings: bool
    query_instruction: str


class VectorStoreConfig(BaseModel):
    """Vector store persistence and retrieval settings."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1)
    persist_directory: Path
    collection_name: str = Field(min_length=1)
    retrieval_candidates: int = Field(gt=0)


class ModelsConfig(BaseModel):
    """Typed settings for document chunking, embedding, and vector storage."""

    model_config = ConfigDict(extra="forbid")

    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig


class KnowledgeConfig(BaseModel):
    """Path to the curated CSV containing structured material constraints."""

    model_config = ConfigDict(extra="forbid")

    material_constraints_csv_path: Path


class OrchestrationConfig(BaseModel):
    """Bounded graph execution and scalar score convergence settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_iterations: int = Field(gt=0)
    max_invalid_revisions: int = Field(ge=0)
    convergence_tolerance: float = Field(ge=0, allow_inf_nan=False)


class ObjectiveConfig(BaseModel):
    """Normalized thrust-to-weight/SFC tradeoff and invalid-design penalty settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    thrust_to_weight_weight: float = Field(ge=0, allow_inf_nan=False)
    specific_fuel_consumption_weight: float = Field(ge=0, allow_inf_nan=False)
    thrust_to_weight_reference: float = Field(gt=0, allow_inf_nan=False)
    specific_fuel_consumption_reference_kg_per_n_s: float = Field(
        gt=0, allow_inf_nan=False
    )
    invalid_base_penalty: float = Field(gt=0, allow_inf_nan=False)
    invalid_severity_penalty_weight: float = Field(gt=0, allow_inf_nan=False)
    signature_decimal_places: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_objective_weights(self) -> ObjectiveConfig:
        """Require at least one objective to contribute to the valid-design score."""
        if self.thrust_to_weight_weight + self.specific_fuel_consumption_weight <= 0:
            raise ValueError("At least one objective weight must be positive")
        return self


class OptimizationConfig(BaseModel):
    """Optuna study controls and scalar engineering objective configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trial_count: int = Field(gt=0)
    random_seed: int
    objective: ObjectiveConfig


class SourcedRange(BaseModel):
    """Search interval with units, sampling strategy, and auditable source IDs."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    minimum: float
    maximum: float
    unit: str = Field(min_length=1)
    strategy: Literal["uniform", "log_uniform"]
    source_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_interval(self) -> SourcedRange:
        if self.minimum >= self.maximum:
            raise ValueError("Search range minimum must be below maximum")
        if self.strategy == "log_uniform" and self.minimum <= 0:
            raise ValueError("Log-uniform search ranges must be positive")
        return self


class SourcedMaterialOption(BaseModel):
    """Material category tied to the source supporting its constraint record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    source_id: str = Field(min_length=1)


class CandidateSearchConfig(BaseModel):
    """Fully sourced bounds for every continuous and categorical candidate field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ambient_temperature_k: SourcedRange
    ambient_pressure_pa: SourcedRange
    flight_speed_m_per_s: SourcedRange
    air_mass_flow_kg_per_s: SourcedRange
    compressor_pressure_ratio: SourcedRange
    turbine_inlet_temperature_k: SourcedRange
    hot_section_materials: tuple[SourcedMaterialOption, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_physical_order(self) -> CandidateSearchConfig:
        expected_units = {
            "ambient_temperature_k": "K",
            "ambient_pressure_pa": "Pa",
            "flight_speed_m_per_s": "m/s",
            "air_mass_flow_kg_per_s": "kg/s",
            "compressor_pressure_ratio": "1",
            "turbine_inlet_temperature_k": "K",
        }
        for field_name, unit in expected_units.items():
            if getattr(self, field_name).unit != unit:
                raise ValueError(f"{field_name} bounds must use {unit}")
        if self.ambient_temperature_k.minimum <= 0:
            raise ValueError("Ambient temperature bounds must be positive")
        if self.ambient_pressure_pa.minimum <= 0:
            raise ValueError("Ambient pressure bounds must be positive")
        if self.flight_speed_m_per_s.minimum < 0:
            raise ValueError("Flight speed cannot be negative")
        if self.air_mass_flow_kg_per_s.minimum <= 0:
            raise ValueError("Air mass flow bounds must be positive")
        if self.compressor_pressure_ratio.minimum <= 1:
            raise ValueError("Compressor pressure ratio must exceed one")
        if self.turbine_inlet_temperature_k.minimum <= self.ambient_temperature_k.maximum:
            raise ValueError("Sampled turbine inlet temperature must exceed all ambient bounds")
        return self


class EngineWeightAnchor(BaseModel):
    """Published sea-level-static engine thrust and dry mass pair."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    rated_thrust_n: float = Field(gt=0)
    dry_mass_kg: float = Field(gt=0)
    source_id: str = Field(min_length=1)


class EngineWeightLookupConfig(BaseModel):
    """Sourced thrust/mass anchors and documented gravity conversion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    anchors: tuple[EngineWeightAnchor, ...] = Field(min_length=2)
    standard_gravity_m_per_s2: float = Field(gt=0, allow_inf_nan=False)
    standard_gravity_source_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_anchor_order(self) -> EngineWeightLookupConfig:
        thrust_values = [anchor.rated_thrust_n for anchor in self.anchors]
        if len(set(thrust_values)) != len(thrust_values):
            raise ValueError("Engine-weight anchors must have unique rated thrust values")
        return self


class EvaluationConfig(BaseModel):
    """Evaluation dataset, reports, and persistent workflow trace paths."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    qa_dataset_path: Path
    report_directory: Path
    trace_jsonl_path: Path
    ragas_metric_names: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_metric_names(self) -> EvaluationConfig:
        if any(not name.strip() for name in self.ragas_metric_names):
            raise ValueError("ragas_metric_names must contain non-empty names")
        if len(set(self.ragas_metric_names)) != len(self.ragas_metric_names):
            raise ValueError("ragas_metric_names must be unique")
        return self


class DesignAgentConfig(BaseModel):
    """Prompt and bounded proposal controls for the design agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_template: str = Field(min_length=1)
    max_attempts: int = Field(gt=0)
    rejection_history_limit: int = Field(ge=0)
    signature_decimal_places: int = Field(ge=0)


class CritiqueAgentConfig(BaseModel):
    """Prompt settings for evidence-based critique explanations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_template: str = Field(min_length=1)


class AgentPromptsConfig(BaseModel):
    """Typed prompt and behavior settings for configured agents."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    design_agent: DesignAgentConfig
    critique_agent: CritiqueAgentConfig


def _parse_environment_value(value: str) -> Any:
    """Parse a scalar environment value using YAML's scalar rules."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML environment value: {value!r}") from exc


def _apply_environment_overrides(data: dict[str, Any], environ: Mapping[str, str]) -> None:
    """Apply RAG_PHY_* nested overrides to the YAML mapping in place."""
    prefix = "RAG_PHY_"
    for key, raw_value in environ.items():
        if not key.startswith(prefix):
            continue
        segments = key[len(prefix) :].lower().split("__")
        if any(not segment for segment in segments):
            raise ValueError(f"Invalid configuration environment variable name: {key}")
        target = data
        for segment in segments[:-1]:
            nested = target.setdefault(segment, {})
            if not isinstance(nested, dict):
                raise ValueError(f"Environment override {key} conflicts with a scalar setting")
            target = nested
        target[segments[-1]] = _parse_environment_value(raw_value)


def load_config(path: str | Path, environ: Mapping[str, str] | None = None) -> AppConfig:
    """Load typed settings from a YAML file and RAG_PHY_* environment overrides.

    Args:
        path: YAML configuration file path.
        environ: Optional environment mapping; defaults to the process environment.

    Returns:
        Validated application configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If YAML is malformed or its top level is not a mapping.
        pydantic.ValidationError: If settings do not match the typed schema.
    """
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration in {config_path}") from exc

    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration at {config_path} must contain a YAML mapping")

    _apply_environment_overrides(loaded, os.environ if environ is None else environ)
    return AppConfig.model_validate(loaded)


def load_physics_config(path: str | Path) -> PhysicsConfig:
    """Load validated cycle settings from a YAML file.

    Args:
        path: YAML file containing physics-model settings.

    Returns:
        Validated thermodynamic cycle configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If YAML is malformed or its top level is not a mapping.
        pydantic.ValidationError: If values do not match the typed schema.
    """
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration in {config_path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"Physics configuration at {config_path} must contain a YAML mapping")
    return PhysicsConfig.model_validate(loaded)


def load_models_config(path: str | Path) -> ModelsConfig:
    """Load typed chunking, embedding, and vector-store settings from YAML.

    Args:
        path: YAML settings file.

    Returns:
        Validated model and vector store configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If YAML is malformed or its top level is not a mapping.
        pydantic.ValidationError: If values do not match the typed schema.
    """
    return ModelsConfig.model_validate(_load_yaml_mapping(path))


def load_knowledge_config(path: str | Path) -> KnowledgeConfig:
    """Load typed structured-constraint store settings from YAML.

    Args:
        path: YAML settings file.

    Returns:
        Validated knowledge store configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If YAML is malformed or its top level is not a mapping.
        pydantic.ValidationError: If values do not match the typed schema.
    """
    return KnowledgeConfig.model_validate(_load_yaml_mapping(path))


def load_agent_prompts_config(path: str | Path) -> AgentPromptsConfig:
    """Load typed design-agent prompt and retry settings from YAML.

    Args:
        path: YAML file containing agent prompt templates and bounded controls.

    Returns:
        Validated agent prompt configuration.
    """
    return AgentPromptsConfig.model_validate(_load_yaml_mapping(path))


def load_orchestration_config(path: str | Path) -> OrchestrationConfig:
    """Load graph iteration and convergence settings from YAML.

    Args:
        path: YAML settings file for the workflow state graph.

    Returns:
        Validated orchestration configuration.
    """
    return OrchestrationConfig.model_validate(_load_yaml_mapping(path))


def load_optimization_config(path: str | Path) -> OptimizationConfig:
    """Load validated Optuna study and objective settings from YAML.

    Args:
        path: YAML settings file for the scalar objective and study controls.

    Returns:
        Validated optimization configuration.
    """
    return OptimizationConfig.model_validate(_load_yaml_mapping(path))


def load_candidate_search_config(
    path: str | Path,
    source_ledger_path: str | Path = "data/curated/sources.md",
) -> CandidateSearchConfig:
    """Load sourced sampler bounds and ensure every provenance key exists in the ledger."""
    config = CandidateSearchConfig.model_validate(_load_yaml_mapping(path))
    _validate_source_ids(
        (
            source_id
            for field_name in CandidateSearchConfig.model_fields
            if field_name != "hot_section_materials"
            for source_id in getattr(config, field_name).source_ids
        ),
        source_ledger_path,
    )
    _validate_source_ids(
        (material.source_id for material in config.hot_section_materials),
        source_ledger_path,
    )
    return config


def load_engine_weight_lookup_config(
    path: str | Path,
    source_ledger_path: str | Path = "data/curated/sources.md",
) -> EngineWeightLookupConfig:
    """Load source-backed engine anchors and ensure provenance keys resolve."""
    config = EngineWeightLookupConfig.model_validate(_load_yaml_mapping(path))
    _validate_source_ids(
        (
            *[anchor.source_id for anchor in config.anchors],
            config.standard_gravity_source_id,
        ),
        source_ledger_path,
    )
    return config


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    """Load evaluation dataset, report, and tracing paths from YAML.

    Args:
        path: YAML evaluation configuration file.

    Returns:
        Validated evaluation configuration.
    """
    return EvaluationConfig.model_validate(_load_yaml_mapping(path))


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Read one YAML mapping and wrap syntax or shape errors consistently."""
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration in {config_path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration at {config_path} must contain a YAML mapping")
    return loaded


def _validate_source_ids(source_ids: Any, ledger_path: str | Path) -> None:
    """Require every configuration source ID to have a heading in the source ledger."""
    ledger = Path(ledger_path)
    text = ledger.read_text(encoding="utf-8")
    headings = {
        line[3:].strip()
        for line in text.splitlines()
        if line.startswith("## ")
    }
    missing = sorted(set(source_ids) - headings)
    if missing:
        raise ValueError(f"Unresolved source IDs in {ledger}: {', '.join(missing)}")


__all__ = [
    "AgentPromptsConfig",
    "AppConfig",
    "CandidateSearchConfig",
    "ChunkingConfig",
    "ComponentEfficiencies",
    "CritiqueAgentConfig",
    "DashboardConfig",
    "DesignAgentConfig",
    "EmbeddingConfig",
    "EngineWeightAnchor",
    "EngineWeightLookupConfig",
    "EvaluationConfig",
    "KnowledgeConfig",
    "LLMConfig",
    "LoggingConfig",
    "ModelsConfig",
    "NumericalConfig",
    "ObjectiveConfig",
    "OptimizationConfig",
    "OrchestrationConfig",
    "PhysicsConfig",
    "SourcedMaterialOption",
    "SourcedRange",
    "ValidationError",
    "VectorStoreConfig",
    "load_agent_prompts_config",
    "load_candidate_search_config",
    "load_config",
    "load_engine_weight_lookup_config",
    "load_evaluation_config",
    "load_knowledge_config",
    "load_models_config",
    "load_optimization_config",
    "load_orchestration_config",
    "load_physics_config",
]
