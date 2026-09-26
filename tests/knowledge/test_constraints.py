from pathlib import Path

import pytest

from rag_phy.knowledge import (
    ConstraintNotFoundError,
    MaterialConstraintStore,
)


def fixture_constraint_store() -> MaterialConstraintStore:
    """Load explicitly synthetic test records, not engineering constraint data."""
    csv_path = Path(__file__).parents[1] / "fixtures" / "material_constraints_synthetic.csv"
    return MaterialConstraintStore.from_csv(csv_path)


def test_curated_source_backed_records_load_from_project_data() -> None:
    csv_path = Path(__file__).parents[2] / "data" / "curated" / "material_constraints.csv"

    store = MaterialConstraintStore.from_csv(csv_path)

    assert store.get_material("Inconel 718 (bare)").max_service_temperature_k == pytest.approx(
        977.594
    )
    assert store.get_material("Inconel 718 (cooled)").max_service_temperature_k == pytest.approx(
        977.594
    )
    assert store.get_material("Ti-6Al-4V").source_id == "CARPENTER-TI6AL4V"
    assert store.get_material("NASA SiC/SiC System A CMC").source_id == (
        "NASA-TM-2006-20060054003"
    )


def test_exact_material_lookup_and_temperature_range_query() -> None:
    """Provide indexed case-insensitive exact lookup and sorted range results."""
    store = fixture_constraint_store()

    exact = store.get_material("test_alloy_alpha")
    high_temperature_materials = store.materials_rated_at_least(1000)

    assert exact.max_service_temperature_k == 1200
    assert exact.source_id == "synthetic-test-record-alpha"
    assert [record.material_name for record in high_temperature_materials] == [
        "TEST_ALLOY_ALPHA"
    ]


def test_missing_material_raises_explicit_error() -> None:
    """Never substitute a default or return None for a missing material record."""
    with pytest.raises(ConstraintNotFoundError, match="No curated constraint record"):
        fixture_constraint_store().get_material("MISSING_TEST_MATERIAL")


def test_material_record_without_source_is_rejected(tmp_path: Path) -> None:
    """Reject numeric constraints without a provenance identifier."""
    csv_path = tmp_path / "missing_source.csv"
    csv_path.write_text(
        "material_name,max_service_temperature_k,source_id\nTEST,1100,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid material constraint"):
        MaterialConstraintStore.from_csv(csv_path)


def test_empty_material_table_is_rejected(tmp_path: Path) -> None:
    """Do not treat an empty structured-data file as a populated constraint store."""
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text(
        "material_name,max_service_temperature_k,source_id\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contains no records"):
        MaterialConstraintStore.from_csv(csv_path)
