#!/usr/bin/env python3
"""Validate provenance links in curated material-constraint data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from rag_phy.provenance import has_checkable_http_reference

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/curated/material_constraints.csv"
LEDGER_PATH = ROOT / "data/curated/sources.md"
REQUIREMENTS_PATH = ROOT / "data/curated/material_requirements.json"
REQUIRED_COLUMNS = {
    "material_name",
    "max_service_temperature_k",
    "evidence_status",
    "source_id",
}
EVIDENCE_STATUSES = {"qualified_allowable", "screening_only"}
SOURCE_HEADING = re.compile(r"^## ([A-Za-z0-9][A-Za-z0-9._:-]*)\s*$", re.MULTILINE)
SOURCE_EVIDENCE_STATUS = re.compile(
    r"^- Material evidence status: `([^`]+)`\.?\s*$",
    re.MULTILINE,
)


def validate_curated_data(
    csv_path: Path = CSV_PATH,
    ledger_path: Path = LEDGER_PATH,
    requirements_path: Path | None = REQUIREMENTS_PATH,
) -> list[str]:
    """Return all schema, value, and unresolved-source errors without mutating files."""
    errors: list[str] = []
    try:
        ledger = ledger_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"Cannot read source ledger {ledger_path}: {exc}"]
    source_sections = _source_sections(ledger)
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
        if None in row:
            errors.append(f"Line {line_number}: row has unexpected fields beyond the CSV header")
        if any(value is None for key, value in row.items() if key is not None):
            errors.append(f"Line {line_number}: row is missing one or more CSV fields")
        name = (row.get("material_name") or "").strip()
        evidence_status = (row.get("evidence_status") or "").strip()
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
        elif source_id not in source_sections:
            errors.append(f"Line {line_number}: source_id {source_id!r} has no ledger entry")
        elif not has_checkable_http_reference(source_sections[source_id]):
            errors.append(
                f"Line {line_number}: source_id {source_id!r} has no checkable HTTP(S) "
                "reference in its ledger entry"
            )
        if evidence_status not in EVIDENCE_STATUSES:
            errors.append(
                f"Line {line_number}: evidence_status must be one of "
                f"{', '.join(sorted(EVIDENCE_STATUSES))}"
            )
        if source_id in source_sections:
            statuses = SOURCE_EVIDENCE_STATUS.findall(source_sections[source_id])
            if len(statuses) != 1:
                errors.append(
                    f"Line {line_number}: source_id {source_id!r} must declare exactly one "
                    "material evidence status in its ledger entry"
                )
            elif statuses[0] != evidence_status:
                errors.append(
                    f"Line {line_number}: evidence_status {evidence_status!r} conflicts with "
                    f"source ledger status {statuses[0]!r} for {source_id!r}"
                )
    if requirements_path is not None:
        errors.extend(_validate_material_coverage(seen_materials, requirements_path))
    return errors


def _source_sections(ledger: str) -> dict[str, str]:
    """Map each source heading to the text up to the following source heading."""
    matches = list(SOURCE_HEADING.finditer(ledger))
    return {
        match.group(1): ledger[match.end() : matches[index + 1].start() if index + 1 < len(matches) else None]
        for index, match in enumerate(matches)
    }


def _validate_material_coverage(material_names: set[str], requirements_path: Path) -> list[str]:
    """Validate material category coverage from a declarative requirements artifact."""
    try:
        requirements: Any = json.loads(requirements_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Cannot read material coverage requirements {requirements_path}: {exc}"]
    if not isinstance(requirements, dict):
        return ["Material coverage requirements must be a JSON object"]
    groups = requirements.get("required_material_groups")
    if not isinstance(groups, list) or not groups:
        return ["Material coverage requirements must define required_material_groups"]

    normalized_names = {" ".join(name.split()) for name in material_names}
    errors: list[str] = []
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            errors.append(f"Material requirement at index {index} must be an object")
            continue
        requirement = group.get("requirement")
        alternatives = group.get("any_of_material_names")
        if not isinstance(requirement, str) or not requirement.strip():
            errors.append(f"Material requirement at index {index} needs a non-empty name")
            continue
        if (
            not isinstance(alternatives, list)
            or not alternatives
            or any(not isinstance(name, str) or not name.strip() for name in alternatives)
        ):
            errors.append(f"Material requirement {requirement!r} needs non-empty name alternatives")
            continue
        normalized_alternatives = {" ".join(name.casefold().split()) for name in alternatives}
        if not normalized_names.intersection(normalized_alternatives):
            errors.append(
                f"Required material coverage {requirement!r} is missing; expected one of "
                f"{', '.join(alternatives)}"
            )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    parser.add_argument("--requirements", type=Path, default=REQUIREMENTS_PATH)
    args = parser.parse_args(argv)

    errors = validate_curated_data(args.csv, args.ledger, args.requirements)
    if errors:
        print("Curated data validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Curated material data, source ledger, and required material coverage are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
