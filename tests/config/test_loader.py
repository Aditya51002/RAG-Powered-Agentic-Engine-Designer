from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_phy.config import (
    load_agent_prompts_config,
    AppConfig,
    load_config,
    load_knowledge_config,
    load_models_config,
    load_orchestration_config,
    load_optimization_config,
    load_physics_config,
)


def test_load_agent_prompt_configuration() -> None:
    config = load_agent_prompts_config("config/agent_prompts.yaml")

    assert config.design_agent.max_attempts == 3
    assert config.design_agent.rejection_history_limit == 20
    assert "{schema}" in config.design_agent.prompt_template
    assert "{constraints}" in config.critique_agent.prompt_template


def test_load_orchestration_limits_from_yaml() -> None:
    config = load_orchestration_config("config/orchestration.yaml")

    assert config.max_iterations == 8
    assert config.max_invalid_revisions == 3
    assert config.convergence_tolerance == 0.001


def test_load_optimization_objective_and_study_config() -> None:
    config = load_optimization_config("config/optimization.yaml")

    assert config.trial_count == 12
    assert config.objective.thrust_to_weight_weight == 1.0
    assert config.objective.specific_fuel_consumption_reference_kg_per_n_s == 0.0000332


def test_load_config_reads_yaml_into_typed_settings(tmp_path: Path) -> None:
    config_file = tmp_path / "sample.yaml"
    config_file.write_text("logging:\n  level: DEBUG\n  json: false\n", encoding="utf-8")

    config = load_config(config_file, environ={})

    assert isinstance(config, AppConfig)
    assert config.logging.level == "DEBUG"
    assert config.logging.json_output is False


def test_environment_override_is_applied_by_loader(tmp_path: Path) -> None:
    config_file = tmp_path / "sample.yaml"
    config_file.write_text("logging:\n  level: INFO\n", encoding="utf-8")

    config = load_config(config_file, environ={"RAG_PHY_LOGGING__LEVEL": "WARNING"})

    assert config.logging.level == "WARNING"


def test_unknown_settings_fail_validation(tmp_path: Path) -> None:
    config_file = tmp_path / "sample.yaml"
    config_file.write_text("unknown: true\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(config_file, environ={})


def test_load_physics_config_validates_component_efficiencies() -> None:
    config = load_physics_config(Path(__file__).parents[2] / "config" / "physics_bounds.yaml")

    assert config.working_fluid == "Air"
    assert config.component_efficiencies.compressor == 0.87


def test_load_knowledge_configs_from_yaml() -> None:
    """Produce typed model and knowledge store settings from their YAML files."""
    project_root = Path(__file__).parents[2]
    models = load_models_config(project_root / "config" / "models.yaml")
    knowledge = load_knowledge_config(project_root / "config" / "knowledge.yaml")

    assert models.embedding.expected_dimension == 384
    assert models.embedding.max_sequence_tokens == 512
    assert models.chunking.overlap_characters < models.chunking.chunk_size_characters
    assert knowledge.material_constraints_csv_path.as_posix().endswith(
        "data/curated/material_constraints.csv"
    )
