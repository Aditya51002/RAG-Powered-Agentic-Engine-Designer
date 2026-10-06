#!/usr/bin/env python3
"""Checks for the sourced, reproducible J85 comparison diagnostic."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.compare_reference_engine import compare_reference_engine

ROOT = Path(__file__).parents[2]


def test_reference_engine_comparison_reports_measured_model_discrepancy() -> None:
    report = compare_reference_engine()

    assert report["comparison_type"] == "diagnostic_only_not_model_validation"
    assert report["source_ids"] == [
        "NASA-CR-20170000884-J85-MODEL",
        "NIST-SP811-UNIT-CONVERSIONS",
    ]
    assert report["input_sha256"]["reference_case"]
    assert report["difference"]["thrust_percent"] > 0
    assert report["difference"]["specific_fuel_consumption_percent"] < 0
    assert "not independent validation" in report["interpretation"]


def test_reference_engine_comparison_rejects_unresolved_source_ids(tmp_path: Path) -> None:
    ledger = tmp_path / "sources.md"
    ledger.write_text("## NASA-CR-20170000884-J85-MODEL\n", encoding="utf-8")

    with pytest.raises(ValueError, match="NIST-SP811-UNIT-CONVERSIONS"):
        compare_reference_engine(ledger_path=ledger)
