"""Optimizer composition loads sourced ranges before initializing external adapters."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from rag_phy.api import OptimizationDesignRunExecutor
from rag_phy.api.composition import make_optimizer_factory
from rag_phy.api.service import create_app
from rag_phy.config import EmbeddingConfig, LLMConfig
from rag_phy.knowledge import RetrievalResult
from rag_phy.optimization import OptunaOptimizer
from rag_phy.orchestration import JsonlTraceSink, WorkflowTraceEvent


def _candidate_search_data() -> dict[str, object]:
    source_ids = ["TEST-BOUNDS"]
    fields = {
        "ambient_temperature_k": (280, 290, "K", "uniform"),
        "ambient_pressure_pa": (99000, 102000, "Pa", "uniform"),
        "flight_speed_m_per_s": (0, 100, "m/s", "uniform"),
        "air_mass_flow_kg_per_s": (2, 20, "kg/s", "log_uniform"),
        "compressor_pressure_ratio": (3, 15, "1", "log_uniform"),
        "turbine_inlet_temperature_k": (900, 1200, "K", "uniform"),
    }
    config: dict[str, object] = {
        name: {
            "minimum": minimum,
            "maximum": maximum,
            "unit": unit,
            "strategy": strategy,
            "source_ids": source_ids,
        }
        for name, (minimum, maximum, unit, strategy) in fields.items()
    }
    config["hot_section_materials"] = [
        {"name": "SC 180 single-crystal superalloy", "source_id": "TEST-BOUNDS"}
    ]
    return config


class _FakeLLM:
    def complete(self, prompt: str) -> str:
        return "offline explanation"


class _FakeEmbedder:
    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config

    def embed_query(self, text: str) -> list[float]:
        return [1.0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


class _FakeVectorStore:
    def __init__(self, results: Sequence[RetrievalResult] = ()) -> None:
        self.results = list(results)

    def upsert(self, chunks, embeddings) -> None:
        raise AssertionError("Composition test does not ingest documents")

    def search(self, query_embedding, limit: int) -> list[RetrievalResult]:
        return self.results[:limit]


class _FakeTraceSink:
    def emit(self, event: WorkflowTraceEvent) -> None:
        pass


def test_optimizer_factory_composes_from_sourced_ranges_and_injected_adapters(
    tmp_path: Path,
) -> None:
    search_path = tmp_path / "candidate_search.yaml"
    search_path.write_text(yaml.safe_dump(_candidate_search_data()), encoding="utf-8")
    source_ledger = tmp_path / "sources.md"
    project_ledger = Path("data/curated/sources.md").read_text(encoding="utf-8")
    source_ledger.write_text(f"{project_ledger}\n## TEST-BOUNDS\n", encoding="utf-8")
    calls: list[str] = []
    trace_paths: list[Path] = []

    def llm_factory(config: LLMConfig) -> _FakeLLM:
        calls.append("llm")
        return _FakeLLM()

    def embedder_factory(config: EmbeddingConfig) -> _FakeEmbedder:
        calls.append("embedder")
        return _FakeEmbedder(config)

    def vector_store_factory(path: Path, collection: str) -> _FakeVectorStore:
        calls.append("vector_store")
        return _FakeVectorStore()

    factory = make_optimizer_factory(
        candidate_search_config_path=search_path,
        source_ledger_path=source_ledger,
        llm_client_factory=llm_factory,
        embedder_factory=embedder_factory,
        vector_store_factory=vector_store_factory,
        trace_sink_factory=lambda path: trace_paths.append(path) or _FakeTraceSink(),
    )

    trace_path = tmp_path / "trace.jsonl"
    optimizer = factory(trace_path)

    assert isinstance(optimizer, OptunaOptimizer)
    assert calls == ["llm", "embedder", "vector_store"]
    assert trace_paths == [trace_path]


def test_missing_candidate_ranges_fail_before_external_adapter_initialization(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    factory = make_optimizer_factory(
        candidate_search_config_path=tmp_path / "missing_search.yaml",
        llm_client_factory=lambda config: calls.append("llm") or _FakeLLM(),
        embedder_factory=lambda config: calls.append("embedder") or _FakeEmbedder(config),
        vector_store_factory=lambda path, name: calls.append("vector_store")
        or _FakeVectorStore(),
    )

    with pytest.raises(FileNotFoundError, match="missing_search.yaml"):
        factory(tmp_path / "trace.jsonl")

    assert not calls


def test_real_optimizer_api_lifecycle_with_external_services_faked(tmp_path: Path) -> None:
    """Exercise submit-to-result with real optimization, physics, citations, and traces."""
    search_data = _candidate_search_data()
    search_data["ambient_temperature_k"] = {
        "minimum": 288.15,
        "maximum": 288.16,
        "unit": "K",
        "strategy": "uniform",
        "source_ids": ["TEST-BOUNDS"],
    }
    search_data["ambient_pressure_pa"] = {
        "minimum": 101325,
        "maximum": 101326,
        "unit": "Pa",
        "strategy": "uniform",
        "source_ids": ["TEST-BOUNDS"],
    }
    search_data["flight_speed_m_per_s"] = {
        "minimum": 0,
        "maximum": 0.01,
        "unit": "m/s",
        "strategy": "uniform",
        "source_ids": ["TEST-BOUNDS"],
    }
    search_data["air_mass_flow_kg_per_s"] = {
        "minimum": 21,
        "maximum": 22,
        "unit": "kg/s",
        "strategy": "uniform",
        "source_ids": ["TEST-BOUNDS"],
    }
    search_data["compressor_pressure_ratio"] = {
        "minimum": 5,
        "maximum": 5.01,
        "unit": "1",
        "strategy": "uniform",
        "source_ids": ["TEST-BOUNDS"],
    }
    search_data["turbine_inlet_temperature_k"] = {
        "minimum": 1100,
        "maximum": 1101,
        "unit": "K",
        "strategy": "uniform",
        "source_ids": ["TEST-BOUNDS"],
    }
    search_path = tmp_path / "candidate_search.yaml"
    search_path.write_text(yaml.safe_dump(search_data), encoding="utf-8")
    source_ledger = tmp_path / "sources.md"
    project_ledger = Path("data/curated/sources.md").read_text(encoding="utf-8")
    source_ledger.write_text(f"{project_ledger}\n## TEST-BOUNDS\n", encoding="utf-8")

    def make_store(path: Path, collection: str) -> _FakeVectorStore:
        return _FakeVectorStore(
            [
                RetrievalResult(
                    chunk_id="synthetic-chunk",
                    text="Synthetic retrieved passage for API integration coverage.",
                    distance=0.1,
                    source_refs=("synthetic:test-retrieval",),
                )
            ]
        )

    factory = make_optimizer_factory(
        candidate_search_config_path=search_path,
        source_ledger_path=source_ledger,
        llm_client_factory=lambda config: _FakeLLM(),
        embedder_factory=_FakeEmbedder,
        vector_store_factory=make_store,
        trace_sink_factory=JsonlTraceSink,
    )
    executor = OptimizationDesignRunExecutor(factory)
    app = create_app(
        run_executor=executor,
        readiness_check=lambda: True,
        trace_directory=tmp_path / "traces",
    )

    with TestClient(app) as client:
        submitted = client.post(
            "/design-runs",
            json={"design_goal": "synthetic API integration test", "trial_count": 2},
            headers={"x-request-id": "composition-integration"},
        )
        assert submitted.status_code == 202
        run_id = submitted.json()["run_id"]
        run = client.get(f"/design-runs/{run_id}")
        trace = client.get(f"/design-runs/{run_id}/trace")

    assert run.json()["status"] == "completed"
    assert run.json()["result"]["accepted_candidate"]["hot_section_material_name"] == (
        "SC 180 single-crystal superalloy"
    )
    assert run.json()["result"]["performance"]["thrust_n"] >= 12677
    assert "SEAH-SC180" in run.json()["result"]["cited_sources"]
    assert "synthetic:test-retrieval" in run.json()["result"]["cited_sources"]
    assert trace.status_code == 200
    assert trace.json()
    assert {event["trace_id"] for event in trace.json()} == {"composition-integration"}
