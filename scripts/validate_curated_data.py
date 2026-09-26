#!/usr/bin/env python3
"""Validate provenance links in curated material-constraint data."""

from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/curated/material_constraints.csv"
LEDGER_PATH = ROOT / "data/curated/sources.md"
REQUIRED_COLUMNS = {"material_name", "max_service_temperature_k", "source_id"}
SOURCE_HEADING = re.compile(r"^## ([A-Za-z0-9][A-Za-z0-9._:-]*)\s*$", re.MULTILINE)


def validate_curated_data(csv_path: Path = CSV_PATH, ledger_path: Path = LEDGER_PATH) -> list[str]:
    """Return all schema, value, and unresolved-source errors without mutating files."""
    errors: list[str] = []
    try:
        ledger = ledger_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"Cannot read source ledger {ledger_path}: {exc}"]
    source_ids = set(SOURCE_HEADING.findall(ledger))
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
                missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or ()))
                return [f"CSV is missing required columns: {', '.join(missing)}"]
            rows = list(reader)
    except OSError as exc:
        return [f"Cannot read curated CSV {csv_path}: {exc}"]
    if not rows:
        errors.append("Curated material CSV contains no data rows")
    seen_materials: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        name = (row.get("material_name") or "").strip()
        source_id = (row.get("source_id") or "").strip()
        if not name:
            errors.append(f"Line {line_number}: material_name is empty")
        elif name.casefold() in seen_materials:
            errors.append(f"Line {line_number}: duplicate material_name {name!r}")
        else:
            seen_materials.add(name.casefold())
        try:
            temperature = float(row.get("max_service_temperature_k") or "")
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError
        except ValueError:
            errors.append(f"Line {line_number}: max_service_temperature_k must be finite and positive")
        if not source_id:
            errors.append(f"Line {line_number}: source_id is empty")
        elif source_id not in source_ids:
            errors.append(f"Line {line_number}: source_id {source_id!r} has no ledger entry")
    return errors


def main() -> int:
    errors = validate_curated_data()
    if errors:
        print("Curated data validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Curated material data and source ledger are structurally consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
