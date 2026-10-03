#!/usr/bin/env python3
"""Ensure every configured physics setting has an explicit evidence disposition."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/physics_bounds.yaml"
REGISTER_PATH = ROOT / "KNOWN_ASSUMPTIONS.md"
SECTION_HEADING = "## `config/physics_bounds.yaml`"
ALLOWED_STATUSES = {
    "source-backed",
    "model-choice",
    "unverified-placeholder",
    "numerical-policy",
}


def _leaf_values(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    leaves: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, Mapping):
            leaves.update(_leaf_values(child, path))
        else:
            leaves[path] = child
    return leaves


def _assumption_rows(markdown: str) -> tuple[dict[str, tuple[str, str]], list[str]]:
    lines = markdown.splitlines()
    try:
        section_start = lines.index(SECTION_HEADING) + 1
    except ValueError:
        return {}, [f"Missing assumptions section {SECTION_HEADING}"]
    section_end = next(
        (index for index in range(section_start, len(lines)) if lines[index].startswith("## ")),
        len(lines),
    )
    rows: dict[str, tuple[str, str]] = {}
    errors: list[str] = []
    for line_number, line in enumerate(lines[section_start:section_end], start=section_start + 1):
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) != 4 or cells[0] == "Setting" or all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue
        if len(cells[0]) < 3 or cells[0][0] != "`" or cells[0][-1] != "`":
            errors.append(f"Line {line_number}: setting path must be enclosed in backticks")
            continue
        path = cells[0][1:-1]
        if path in rows:
            errors.append(f"Line {line_number}: duplicate setting path {path!r}")
            continue
        rows[path] = (cells[1], cells[2])
        if not cells[3]:
            errors.append(f"Line {line_number}: evidence/limitation is empty for {path!r}")
        if cells[2] not in ALLOWED_STATUSES:
            errors.append(f"Line {line_number}: unsupported evidence status {cells[2]!r}")
        elif cells[2] == "source-backed" and "http" not in cells[3].lower() and "doi:" not in cells[3].lower():
            errors.append(f"Line {line_number}: source-backed setting {path!r} needs a citation URL or DOI")
        elif cells[2] == "unverified-placeholder" and "pending review" not in cells[3].lower():
            errors.append(f"Line {line_number}: placeholder {path!r} must say it is pending review")
        elif cells[2] == "numerical-policy" and "no external source" not in cells[3].lower():
            errors.append(f"Line {line_number}: numerical policy {path!r} must state it has no external source")
    return rows, errors


def validate_physics_bounds(
    config_path: Path = CONFIG_PATH,
    register_path: Path = REGISTER_PATH,
) -> list[str]:
    """Return schema, value, evidence-status, and register-coverage errors."""
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return [f"Cannot read valid physics config {config_path}: {exc}"]
    if not isinstance(config, Mapping):
        return [f"Physics config {config_path} must contain a mapping"]
    try:
        register = register_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"Cannot read assumptions register {register_path}: {exc}"]

    configured = _leaf_values(config)
    rows, errors = _assumption_rows(register)
    for path in sorted(configured.keys() - rows.keys()):
        errors.append(f"Physics setting {path!r} has no KNOWN_ASSUMPTIONS.md entry")
    for path in sorted(rows.keys() - configured.keys()):
        errors.append(f"Assumptions entry {path!r} does not resolve to a physics config setting")
    for path in sorted(configured.keys() & rows.keys()):
        display_value, _ = rows[path]
        try:
            registered_value = yaml.safe_load(display_value)
        except yaml.YAMLError:
            registered_value = object()
        if registered_value != configured[path]:
            errors.append(
                f"Assumptions entry {path!r} has value {display_value!r}; "
                f"config has {configured[path]!r}"
            )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--register", type=Path, default=REGISTER_PATH)
    args = parser.parse_args(argv)

    errors = validate_physics_bounds(args.config, args.register)
    if errors:
        print("Physics-bound provenance validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Every configured physics setting has a current, classified assumptions entry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
