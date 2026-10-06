#!/usr/bin/env python3
"""Offline checks for auditable optimizer-study artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

optuna = pytest.importorskip("optuna")

from rag_phy.optimization.study import OptimizationRun
from scripts import run_optimizer_study


def _synthetic_run() -> OptimizationRun:
    study = optuna.create_study(direction="maximize")

    def objective(trial):
        trial.set_user_attr("valid", trial.number == 1)
        trial.set_user_attr("candidate", {"test_dimension": trial.number})
        return float(trial.number)

    study.optimize(objective, n_trials=2)
    return OptimizationRun(study=study, pareto_frontier=())


def test_study_artifacts_capture_trial_history_and_validity(tmp_path: Path) -> None:
    run = _synthetic_run()

    artifact = run_optimizer_study.write_study_artifacts(
        run,
        {"design_goal": "synthetic unit test"},
        tmp_path,
    )

    stored = json.loads((tmp_path / "study.json").read_text(encoding="utf-8"))
    assert stored == artifact
    assert artifact["summary"]["trial_count"] == 2
    assert artifact["summary"]["completed_trial_count"] == 2
    assert artifact["summary"]["valid_trial_count"] == 1
    assert artifact["summary"]["validity_rate"] == 0.5
    assert len(artifact["trials"]) == 2
    assert (tmp_path / "validity_rate.png").is_file()


def test_runner_refuses_to_invent_missing_candidate_ranges(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_config = tmp_path / "candidate_search.yaml"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_optimizer_study",
            "--design-goal",
            "synthetic test only",
            "--trials",
            "50",
            "--output-root",
            str(tmp_path / "runs"),
            "--candidate-config",
            str(missing_config),
        ],
    )

    assert run_optimizer_study.main() == 2
    assert "will not invent parameter bounds" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()
