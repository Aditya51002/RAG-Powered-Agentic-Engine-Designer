#!/usr/bin/env python3
"""Compare the configured turbojet cycle model with a sourced reference point."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rag_phy.config import load_physics_config
from rag_phy.physics import CycleInput, simulate_cycle

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "data/curated/reference_engines/j85_takeoff.json"
DEFAULT_PHYSICS = ROOT / "config/physics_bounds.yaml"
DEFAULT_LEDGER = ROOT / "data/curated/sources.md"
SOURCE_HEADING = re.compile(r"^## ([A-Za-z0-9][A-Za-z0-9._:-]*)\s*$", re.MULTILINE)


class PublishedValues(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    altitude_ft: float
    ambient_temperature_rankine: float
    ambient_pressure_psia: float
    mach_number: float
    net_thrust_lbf: float = Field(gt=0)
    specific_fuel_consumption_lbm_per_hour_lbf: float = Field(gt=0)
    shaft_speed_rpm: float = Field(gt=0)
    air_mass_flow_lbm_per_second: float = Field(gt=0)
    compressor_pressure_ratio: float = Field(gt=1)
    turbine_inlet_temperature_rankine: float = Field(gt=0)


class ReferencePerformance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    thrust_n: float = Field(gt=0)
    specific_fuel_consumption_kg_per_n_s: float = Field(gt=0)


class ReferenceEngineCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: int = Field(ge=1)
    engine_name: str = Field(min_length=1)
    source_ids: tuple[str, ...] = Field(min_length=1)
    source_locator: str = Field(min_length=1)
    published_values: PublishedValues
    cycle_input_si: CycleInput
    reference_performance_si: ReferencePerformance
    conversion_note: str = Field(min_length=1)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def compare_reference_engine(
    reference_path: Path = DEFAULT_REFERENCE,
    physics_path: Path = DEFAULT_PHYSICS,
    ledger_path: Path = DEFAULT_LEDGER,
) -> dict[str, Any]:
    """Run one comparison only after every dataset source ID resolves in the ledger."""
    case = ReferenceEngineCase.model_validate_json(reference_path.read_text(encoding="utf-8"))
    source_headings = set(SOURCE_HEADING.findall(ledger_path.read_text(encoding="utf-8")))
    unresolved = sorted(set(case.source_ids) - source_headings)
    if unresolved:
        raise ValueError(f"Reference engine sources missing from ledger: {', '.join(unresolved)}")

    physics_config = load_physics_config(physics_path)
    predicted = simulate_cycle(case.cycle_input_si, physics_config)
    expected = case.reference_performance_si
    thrust_error = predicted.thrust_n - expected.thrust_n
    sfc_error = (
        predicted.specific_fuel_consumption_kg_per_n_s
        - expected.specific_fuel_consumption_kg_per_n_s
    )
    return {
        "schema_version": 1,
        "comparison_type": "diagnostic_only_not_model_validation",
        "engine_name": case.engine_name,
        "source_ids": list(case.source_ids),
        "source_locator": case.source_locator,
        "published_values": case.published_values.model_dump(mode="json"),
        "cycle_input_si": case.cycle_input_si.model_dump(mode="json"),
        "conversion_note": case.conversion_note,
        "reference_performance_si": expected.model_dump(mode="json"),
        "predicted_performance_si": {
            "thrust_n": predicted.thrust_n,
            "specific_fuel_consumption_kg_per_n_s": (
                predicted.specific_fuel_consumption_kg_per_n_s
            ),
        },
        "difference": {
            "thrust_n": thrust_error,
            "thrust_percent": 100 * thrust_error / expected.thrust_n,
            "specific_fuel_consumption_kg_per_n_s": sfc_error,
            "specific_fuel_consumption_percent": (
                100 * sfc_error / expected.specific_fuel_consumption_kg_per_n_s
            ),
        },
        "environment": {
            "rag_phy": _package_version("rag-phy"),
            "coolprop": _package_version("CoolProp"),
        },
        "input_sha256": {
            "reference_case": _sha256(reference_path),
            "physics_config": _sha256(physics_path),
            "source_ledger": _sha256(ledger_path),
        },
        "interpretation": (
            "The configured generic cycle model and the NASA J85 example use different, "
            "partly assumed component models. This point comparison is a discrepancy "
            "diagnostic, not independent validation, a calibrated J85 simulation, or an "
            "operating envelope."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--physics", type=Path, default=DEFAULT_PHYSICS)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/evaluation/j85_cycle_diagnostic.json"
    )
    args = parser.parse_args()
    try:
        report = compare_reference_engine(args.reference, args.physics, args.ledger)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as exc:
        print(f"Reference comparison failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output),
                "thrust_percent_difference": report["difference"]["thrust_percent"],
                "sfc_percent_difference": report["difference"][
                    "specific_fuel_consumption_percent"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
