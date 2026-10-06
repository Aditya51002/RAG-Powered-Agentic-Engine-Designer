#!/usr/bin/env python3
"""Run a sourced, provider-backed 50+ trial study and save an auditable artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from rag_phy.evaluation.metrics import (
    cumulative_validity_rate,
    save_validity_chart,
    validity_observations_from_study,
)
from rag_phy.optimization import OptimizationRun

ROOT = Path(__file__).resolve().parents[1]
MINIMUM_TRIAL_COUNT = 50


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def serialize_study(run: OptimizationRun, metadata: dict[str, Any]) -> dict[str, Any]:
    """Convert one completed optimization run into JSON-compatible evidence."""
    observations = validity_observations_from_study(run.study)
    validity_points = cumulative_validity_rate(observations)
    trials = [
        {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "params": trial.params,
            "user_attributes": trial.user_attrs,
        }
        for trial in run.study.trials
    ]
    pareto_frontier = [
        {
            "candidate": point.candidate.model_dump(mode="json"),
            "thrust_to_weight_ratio": point.thrust_to_weight_ratio,
            "specific_fuel_consumption_kg_per_n_s": (
                point.specific_fuel_consumption_kg_per_n_s
            ),
            "cited_sources": list(point.cited_sources),
        }
        for point in run.pareto_frontier
    ]
    best_valid = None
    if run.best_valid_candidate is not None:
        best_valid = {
            "candidate": run.best_valid_candidate.model_dump(mode="json"),
            "performance": asdict(run.best_valid_performance)
            if run.best_valid_performance is not None
            else None,
            "critique": run.best_valid_critique.model_dump(mode="json")
            if run.best_valid_critique is not None
            else None,
            "score": run.best_valid_score,
        }
    completed_count = len(observations)
    valid_count = validity_points[-1].valid_count if validity_points else 0
    return {
        "schema_version": 1,
        "metadata": metadata,
        "environment": {
            "python": sys.version.split()[0],
            "rag_phy": _package_version("rag-phy"),
            "optuna": _package_version("optuna"),
        },
        "summary": {
            "trial_count": len(trials),
            "completed_trial_count": completed_count,
            "valid_trial_count": valid_count,
            "validity_rate": valid_count / completed_count if completed_count else None,
            "best_objective_value": run.study.best_value,
            "best_valid_score": run.best_valid_score,
        },
        "trials": trials,
        "validity_rate": [asdict(point) for point in validity_points],
        "best_valid_result": best_valid,
        "pareto_frontier": pareto_frontier,
    }


def write_study_artifacts(
    run: OptimizationRun,
    metadata: dict[str, Any],
    run_directory: Path,
) -> dict[str, Any]:
    """Write the complete trial ledger, Pareto set, and validity chart."""
    artifact = serialize_study(run, metadata)
    _write_json(run_directory / "study.json", artifact)
    observations = validity_observations_from_study(run.study)
    save_validity_chart(
        cumulative_validity_rate(observations),
        run_directory / "validity_rate.png",
    )
    return artifact


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _new_run_directory(output_root: Path) -> tuple[str, Path]:
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex[:8]}"
    run_directory = output_root / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_id, run_directory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-goal", required=True)
    parser.add_argument("--trials", required=True, type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--candidate-config", type=Path, default=ROOT / "config/candidate_search.yaml"
    )
    args = parser.parse_args()
    if args.trials < MINIMUM_TRIAL_COUNT:
        parser.error(f"--trials must be at least {MINIMUM_TRIAL_COUNT}")
    if not args.design_goal.strip():
        parser.error("--design-goal must not be empty")

    args.candidate_config = args.candidate_config.resolve()
    os.chdir(ROOT)
    if not args.candidate_config.is_file():
        print(
            f"No sourced candidate ranges found at {args.candidate_config}; "
            "the study will not invent parameter bounds.",
            file=sys.stderr,
        )
        return 2

    if __package__:
        from scripts.validate_curated_data import validate_curated_data
        from scripts.validate_physics_bounds import validate_physics_bounds
    else:
        from validate_curated_data import validate_curated_data
        from validate_physics_bounds import validate_physics_bounds

    data_errors = [*validate_curated_data(), *validate_physics_bounds()]
    if data_errors:
        print("Cannot run a sourced optimizer study:", file=sys.stderr)
        for error in data_errors:
            print(f"- {error}", file=sys.stderr)
        return 2

    from rag_phy.api.composition import make_optimizer_factory

    source_files = {
        "candidate_search": args.candidate_config,
        "app": ROOT / "config/app.yaml",
        "physics_bounds": ROOT / "config/physics_bounds.yaml",
        "engine_weight": ROOT / "config/engine_weight.yaml",
        "optimization": ROOT / "config/optimization.yaml",
        "models": ROOT / "config/models.yaml",
        "agent_prompts": ROOT / "config/agent_prompts.yaml",
        "orchestration": ROOT / "config/orchestration.yaml",
        "material_constraints": ROOT / "data/curated/material_constraints.csv",
        "source_ledger": ROOT / "data/curated/sources.md",
        "corpus_manifest": ROOT / "data/curated/corpus/manifest.json",
    }
    run_directory: Path | None = None
    metadata: dict[str, Any] = {}
    try:
        optimizer_factory = make_optimizer_factory(
            candidate_search_config_path=args.candidate_config
        )
        output_root = (
            args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
        )
        output_root.mkdir(parents=True, exist_ok=True)
        run_id, run_directory = _new_run_directory(output_root)
        metadata = {
            "run_id": run_id,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "design_goal": args.design_goal.strip(),
            "requested_trials": args.trials,
            "input_sha256": {name: _sha256(path) for name, path in source_files.items()},
        }
        _write_json(run_directory / "status.json", {"status": "running", **metadata})
        optimizer = optimizer_factory(run_directory / "trace.jsonl")
        result = optimizer.run(
            args.design_goal.strip(),
            trace_id=run_id,
            trial_count=args.trials,
        )
        artifact = write_study_artifacts(result, metadata, run_directory)
        _write_json(
            run_directory / "status.json",
            {"status": "completed", "completed_at_utc": datetime.now(UTC).isoformat(), **metadata},
        )
        print(
            json.dumps(
                {"run_id": run_id, "run_directory": str(run_directory), **artifact["summary"]},
                indent=2,
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - persist failure state for any run-time error.
        if run_directory is not None:
            try:
                _write_json(
                    run_directory / "status.json",
                    {
                        "status": "failed",
                        "failed_at_utc": datetime.now(UTC).isoformat(),
                        "error_type": type(exc).__name__,
                        **metadata,
                    },
                )
            except OSError as status_error:
                print(f"Could not persist failed-run status: {status_error}", file=sys.stderr)
        print(f"Optimization study failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
