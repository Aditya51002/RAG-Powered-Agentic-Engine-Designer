from pathlib import Path

from scripts.validate_curated_data import validate_curated_data


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
