import subprocess
import sys
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
        "material_name,max_service_temperature_k,evidence_status,source_id\n"
        "TEST,300,screening_only,missing-source\n",
        encoding="utf-8",
    )
    ledger_path.write_text(
        "## known-source\n- Material evidence status: `screening_only`.\n",
        encoding="utf-8",
    )

    assert any(
        "missing-source" in error
        for error in validate_curated_data(csv_path, ledger_path, requirements_path=None)
    )


def test_validator_rejects_non_finite_material_temperature(tmp_path: Path) -> None:
    csv_path = tmp_path / "material_constraints.csv"
    ledger_path = tmp_path / "sources.md"
    csv_path.write_text(
        "material_name,max_service_temperature_k,evidence_status,source_id\n"
        "TEST,nan,screening_only,known-source\n",
        encoding="utf-8",
    )
    ledger_path.write_text(
        "## known-source\n"
        "- Material evidence status: `screening_only`.\n"
        "Reference: https://example.org/material.pdf\n",
        encoding="utf-8",
    )

    errors = validate_curated_data(csv_path, ledger_path, requirements_path=None)

    assert any("must be finite and positive" in error for error in errors)


def test_validator_rejects_rows_with_wrong_field_counts(tmp_path: Path) -> None:
    csv_path = tmp_path / "material_constraints.csv"
    ledger_path = tmp_path / "sources.md"
    ledger_path.write_text(
        "## known-source\n"
        "- Material evidence status: `screening_only`.\n"
        "Reference: https://example.invalid/material\n",
        encoding="utf-8",
    )

    for row, expected_error in (
        ("TEST,300,screening_only,known-source,unexpected", "unexpected fields"),
        ("TEST,300,screening_only", "missing one or more CSV fields"),
    ):
        csv_path.write_text(
            "material_name,max_service_temperature_k,evidence_status,source_id\n"
            f"{row}\n",
            encoding="utf-8",
        )

        errors = validate_curated_data(csv_path, ledger_path, requirements_path=None)

        assert any(expected_error in error for error in errors)


def test_curated_data_cli_exits_nonzero_for_invalid_row(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    csv_path = tmp_path / "material_constraints.csv"
    ledger_path = tmp_path / "sources.md"
    requirements_path = tmp_path / "requirements.json"
    csv_path.write_text(
        "material_name,max_service_temperature_k,evidence_status,source_id\n"
        "TEST,not-a-temperature,screening_only,TEST-SOURCE\n",
        encoding="utf-8",
    )
    ledger_path.write_text(
        "## TEST-SOURCE\n"
        "- Material evidence status: `screening_only`.\n"
        "Reference: https://example.invalid/test-source\n",
        encoding="utf-8",
    )
    requirements_path.write_text(
        '{"required_material_groups": [{"requirement": "test", '
        '"any_of_material_names": ["TEST"]}]}',
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/validate_curated_data.py"),
            "--csv",
            str(csv_path),
            "--ledger",
            str(ledger_path),
            "--requirements",
            str(requirements_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "max_service_temperature_k must be finite and positive" in completed.stderr


def test_validator_rejects_unclassified_material_evidence(tmp_path: Path) -> None:
    csv_path = tmp_path / "material_constraints.csv"
    ledger_path = tmp_path / "sources.md"
    csv_path.write_text(
        "material_name,max_service_temperature_k,evidence_status,source_id\n"
        "TEST,300,unspecified,known-source\n",
        encoding="utf-8",
    )
    ledger_path.write_text(
        "## known-source\n- Material evidence status: `screening_only`.\n",
        encoding="utf-8",
    )

    assert any(
        "evidence_status must be one of" in error
        for error in validate_curated_data(csv_path, ledger_path, requirements_path=None)
    )


def test_validator_rejects_ledger_heading_without_checkable_reference(tmp_path: Path) -> None:
    csv_path = tmp_path / "material_constraints.csv"
    ledger_path = tmp_path / "sources.md"
    csv_path.write_text(
        "material_name,max_service_temperature_k,evidence_status,source_id\n"
        "TEST,300,screening_only,known-source\n",
        encoding="utf-8",
    )
    ledger_path.write_text(
        "## known-source\n- Material evidence status: `screening_only`.\n"
        "A malformed citation URL: https://\n",
        encoding="utf-8",
    )

    errors = validate_curated_data(csv_path, ledger_path, requirements_path=None)

    assert any("no checkable HTTP(S) reference" in error for error in errors)


def test_validator_rejects_csv_evidence_status_not_recorded_for_source(tmp_path: Path) -> None:
    csv_path = tmp_path / "material_constraints.csv"
    ledger_path = tmp_path / "sources.md"
    csv_path.write_text(
        "material_name,max_service_temperature_k,evidence_status,source_id\n"
        "TEST,300,qualified_allowable,known-source\n",
        encoding="utf-8",
    )
    ledger_path.write_text(
        "## known-source\n"
        "- Material evidence status: `screening_only`.\n"
        "Reference: https://example.org/material.pdf\n",
        encoding="utf-8",
    )

    errors = validate_curated_data(csv_path, ledger_path, requirements_path=None)

    assert any("conflicts with source ledger status" in error for error in errors)


def test_validator_enforces_declarative_material_coverage(tmp_path: Path) -> None:
    csv_path = tmp_path / "material_constraints.csv"
    ledger_path = tmp_path / "sources.md"
    requirements_path = tmp_path / "requirements.json"
    csv_path.write_text(
        "material_name,max_service_temperature_k,evidence_status,source_id\n"
        "Only Material,300,screening_only,known-source\n",
        encoding="utf-8",
    )
    ledger_path.write_text("## known-source\nA documented source.\n", encoding="utf-8")
    requirements_path.write_text(
        '{"required_material_groups": [{"requirement": "CMC", '
        '"any_of_material_names": ["CMC A", "CMC B"]}]}',
        encoding="utf-8",
    )

    errors = validate_curated_data(csv_path, ledger_path, requirements_path)

    assert any("Required material coverage 'CMC' is missing" in error for error in errors)


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


def test_physics_source_backed_status_requires_real_url_or_doi_syntax(tmp_path: Path) -> None:
    config_path = tmp_path / "physics.yaml"
    register_path = tmp_path / "KNOWN_ASSUMPTIONS.md"
    config_path.write_text("pressure_pa: 100\n", encoding="utf-8")

    for citation_text in (
        "Claim contains this-ishttp-invalid",
        "Claim cites DOI: not-a-doi",
        "Claim includes an incomplete URL: https://",
    ):
        register_path.write_text(
            "## `config/physics_bounds.yaml`\n\n"
            "| Setting | Config value | Evidence status | Evidence / limitation |\n"
            "| --- | --- | --- | --- |\n"
            f"| `pressure_pa` | 100 | source-backed | {citation_text} |\n",
            encoding="utf-8",
        )

        errors = validate_physics_bounds(config_path, register_path)

        assert any("needs a citation URL or DOI" in error for error in errors)


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
