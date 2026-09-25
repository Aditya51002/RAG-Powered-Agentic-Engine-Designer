"""Load and validate application configuration from YAML and environment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class LoggingConfig(BaseModel):
    """Logging output settings."""

    model_config = ConfigDict(extra="forbid")

    level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    json_output: bool = Field(default=True, alias="json")


class AppConfig(BaseModel):
    """Validated settings shared across the application."""

    model_config = ConfigDict(extra="forbid")

    logging: LoggingConfig = Field(default_factory=LoggingConfig)


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


def _apply_environment_overrides(data: dict[str, Any], environ: dict[str, str]) -> None:
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


def load_config(path: str | Path, environ: dict[str, str] | None = None) -> AppConfig:
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


__all__ = [
    "AppConfig",
    "AgentPromptsConfig",
    "ChunkingConfig",
    "ComponentEfficiencies",
    "EmbeddingConfig",
    "DesignAgentConfig",
    "CritiqueAgentConfig",
    "KnowledgeConfig",
    "LoggingConfig",
    "ModelsConfig",
    "NumericalConfig",
    "ObjectiveConfig",
    "OptimizationConfig",
    "OrchestrationConfig",
    "PhysicsConfig",
    "VectorStoreConfig",
    "ValidationError",
    "load_config",
    "load_agent_prompts_config",
    "load_knowledge_config",
    "load_models_config",
    "load_orchestration_config",
    "load_optimization_config",
    "load_physics_config",
]
