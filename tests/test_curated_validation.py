from pathlib import Path

from scripts.evaluate_retrieval import calculate_source_metrics
from scripts.validate_curated_data import validate_curated_data
from scripts.validate_physics_bounds import main as validate_physics_bounds_main
from scripts.validate_physics_bounds import validate_physics_bounds
from scripts.validate_qa_dataset import validate_qa_dataset


def test_repository_curated_data_resolves_all_source_ids() -> None:
    root = Path(__file__).parents[1]

    assert validate_curated_data(
        root / "data/curated/material_constraints.csv",
        root / "data/curated/sources.md",
    ) == []


def test_validator_rejects_source_ids_missing_from_ledger(tmp_path: Path) -> None:
    csv_path = tmp_path / "material_constraints.csv"
    ledger_path = tmp_path / "sources.md"
    csv_path.write_text(
        "material_name,max_service_temperature_k,source_id\nTEST,300,missing-source\n",
        encoding="utf-8",
    )
    ledger_path.write_text("## known-source\nA documented source.\n", encoding="utf-8")

    assert any("missing-source" in error for error in validate_curated_data(csv_path, ledger_path))


def test_repository_physics_settings_have_current_evidence_dispositions() -> None:
    root = Path(__file__).parents[1]

    assert validate_physics_bounds(
        root / "config/physics_bounds.yaml",
        root / "KNOWN_ASSUMPTIONS.md",
    ) == []


def test_physics_bounds_cli_fails_on_missing_and_stale_register_rows(
    tmp_path: Path, capsys
) -> None:
    config_path = tmp_path / "physics.yaml"
    register_path = tmp_path / "KNOWN_ASSUMPTIONS.md"
    config_path.write_text("root:\n  pressure_pa: 100\n  iterations: 8\n", encoding="utf-8")
    register_path.write_text(
        "## `config/physics_bounds.yaml`\n\n"
        "| Setting | Config value | Evidence status | Evidence / limitation |\n"
        "| --- | --- | --- | --- |\n"
        "| `root.pressure_pa` | 90 | source-backed | Report. https://example.org/report |\n",
        encoding="utf-8",
    )

    exit_code = validate_physics_bounds_main(
        ["--config", str(config_path), "--register", str(register_path)]
    )
    errors = capsys.readouterr().err

    assert exit_code == 1
    assert "no KNOWN_ASSUMPTIONS.md entry" in errors
    assert "config has 100" in errors


def test_repository_qa_set_is_tied_to_real_corpus_pages() -> None:
    root = Path(__file__).parents[1]

    assert validate_qa_dataset(
        root / "data/evaluation/qa_set.json",
        root / "data/curated/corpus/manifest.json",
        root,
    ) == []


def test_retrieval_metrics_treat_unanswerable_cases_separately() -> None:
    metrics = calculate_source_metrics(
        [True, False],
        [{"source:a"}, {"source:spurious"}],
        [{"source:a", "source:extra"}, {"source:spurious"}],
    )

    assert metrics["retrieval_precision_macro_answerable"] == 0.5
    assert metrics["retrieval_recall_macro_answerable"] == 1.0
    assert metrics["unanswerable_retrieval_empty_rate"] == 0.0
