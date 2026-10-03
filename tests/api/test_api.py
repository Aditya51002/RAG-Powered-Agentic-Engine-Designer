"""HTTP lifecycle tests with injected fakes and no model/provider dependencies."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from rag_phy.api.schemas import (
    CandidateResponse,
    DesignProgressResponse,
    DesignRunResultResponse,
    ParetoDesignResponse,
    PerformanceResponse,
)
from rag_phy.api.service import create_app


def _candidate() -> CandidateResponse:
    return CandidateResponse(
        ambient_temperature_k=290,
        ambient_pressure_pa=100000,
        flight_speed_m_per_s=0,
        air_mass_flow_kg_per_s=5,
        compressor_pressure_ratio=5,
        turbine_inlet_temperature_k=1000,
        hot_section_material_name="TEST-MATERIAL",
    )


class _Executor:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.seen: tuple[str, int] | None = None

    def run(self, design_goal, trial_count, on_progress, trace_path: Path):
        self.seen = (design_goal, trial_count)
        on_progress(
            DesignProgressResponse(
                iteration=1,
                total_iterations=trial_count,
                status="valid",
                best_candidate=_candidate(),
                best_score=1.5,
                best_valid=True,
            )
        )
        if self.fail:
            raise RuntimeError("internal exception must not reach client")
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            '{"trace_id":"trace-1","span_id":"span-1","parent_span_id":null,'
            '"node":"score","iteration":1,"message":"done","timestamp":"now"}\n',
            encoding="utf-8",
        )
        point = ParetoDesignResponse(
            candidate=_candidate(),
            thrust_to_weight_ratio=2,
            specific_fuel_consumption_kg_per_n_s=0.00003,
            cited_sources=("ledger:material",),
        )
        return DesignRunResultResponse(
            accepted_candidate=_candidate(),
            performance=PerformanceResponse(
                thrust_n=10000,
                specific_fuel_consumption_kg_per_n_s=0.00003,
                thermal_efficiency=0.3,
                fuel_air_ratio=0.02,
            ),
            pareto_frontier=(point,),
            cited_sources=("ledger:material",),
        )


def test_design_run_lifecycle_trace_search_and_readiness(tmp_path: Path) -> None:
    executor = _Executor()
    app = create_app(
        run_executor=executor,
        knowledge_search=lambda query: [
            SimpleNamespace(
                text=f"passage for {query}", source_refs=("source#page=1",), distance=0.1
            )
        ],
        readiness_check=lambda: True,
        trace_directory=tmp_path / "traces",
    )

    with TestClient(app) as client:
        submitted = client.post(
            "/design-runs", json={"design_goal": "minimize fuel", "trial_count": 3}
        )
        assert submitted.status_code == 202
        run_id = submitted.json()["run_id"]
        assert submitted.json()["status"] == "queued"

        run = client.get(f"/design-runs/{run_id}")
        assert run.status_code == 200
        assert run.json()["status"] == "completed"
        assert run.json()["progress"]["iteration"] == 1
        assert run.json()["result"]["accepted_candidate"]["hot_section_material_name"] == (
            "TEST-MATERIAL"
        )
        assert run.json()["result"]["cited_sources"] == ["ledger:material"]
        assert executor.seen == ("minimize fuel", 3)

        trace = client.get(f"/design-runs/{run_id}/trace")
        assert trace.status_code == 200
        assert trace.json()[0]["trace_id"] == "trace-1"

        search = client.get("/knowledge/search", params={"q": "CMC"})
        assert search.status_code == 200
        assert search.json()[0]["source_refs"] == ["source#page=1"]
        assert client.get("/health").json() == {"status": "ready"}


def test_structured_validation_not_found_and_not_configured_errors() -> None:
    with TestClient(create_app()) as client:
        invalid = client.post("/design-runs", json={"design_goal": "  ", "trial_count": 0})
        assert invalid.status_code == 422
        assert invalid.json()["error"]["code"] == "validation_error"
        missing = client.get("/design-runs/unknown")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "run_not_found"
        unconfigured = client.post(
            "/design-runs", json={"design_goal": "test", "trial_count": 1}
        )
        assert unconfigured.status_code == 503
        assert unconfigured.json()["error"]["code"] == "not_configured"
        unready = client.get("/health")
        assert unready.status_code == 503


def test_internal_worker_error_is_recorded_without_exposing_traceback(tmp_path: Path) -> None:
    app = create_app(
        run_executor=_Executor(fail=True),
        readiness_check=lambda: True,
        trace_directory=tmp_path / "traces",
    )
    with TestClient(app) as client:
        submitted = client.post(
            "/design-runs", json={"design_goal": "test", "trial_count": 1}
        )
        run = client.get(f"/design-runs/{submitted.json()['run_id']}")

    assert run.status_code == 200
    assert run.json()["status"] == "failed"
    assert "Traceback" not in run.text
    assert run.json()["error"] == "Design run failed; inspect server logs"
