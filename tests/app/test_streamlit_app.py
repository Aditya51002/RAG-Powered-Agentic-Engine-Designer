"""Streamlit startup test for the unconfigured real-service state."""

from pathlib import Path

import pytest
import requests


def test_streamlit_entrypoint_renders_setup_status_without_backend() -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    entrypoint = Path(__file__).parents[2] / "src/rag_phy/app/streamlit_app.py"
    app = AppTest.from_file(str(entrypoint)).run()

    assert not app.exception
    assert [element.value for element in app.title] == ["RAG Phy"]
    assert any("No dashboard service is configured" in element.value for element in app.error)


def test_http_frontend_renders_submission_and_search_without_api(monkeypatch) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    def unavailable_api(self, url, *, timeout, params=None):
        del self, url, timeout, params
        raise requests.ConnectionError("test keeps API offline")

    monkeypatch.setattr(requests.sessions.Session, "get", unavailable_api)
    entrypoint = Path(__file__).parents[2] / "src/rag_phy/frontend/streamlit_app.py"
    app = AppTest.from_file(str(entrypoint)).run()

    assert not app.exception
    assert [element.value for element in app.title] == ["RAG Phy Design Runs"]
    assert any("Could not reach the RAG Phy API" in element.value for element in app.error)
    assert [element.label for element in app.text_area] == ["Design goal"]
    assert len(app.form) == 2


def test_http_frontend_submits_goal_and_trial_count_then_polls_run(monkeypatch) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    requests_made = []
    run_responses = [
        {
            "run_id": "run-test",
            "status": "queued",
            "progress": None,
            "error": None,
            "result": None,
        },
        {
            "run_id": "run-test",
            "status": "completed",
            "progress": None,
            "error": None,
            "result": {
                "accepted_candidate": {"compressor_pressure_ratio": 7.0},
                "performance": {"thrust_n": 12000.0},
                "pareto_frontier": [
                    {
                        "thrust_to_weight_ratio": 2.0,
                        "specific_fuel_consumption_kg_per_n_s": 0.00003,
                    }
                ],
                "cited_sources": ["NASA-report.pdf#page=12"],
            },
        },
    ]

    def fake_get(self, url, *, timeout, params=None):
        del self, timeout, params
        requests_made.append(("GET", url))
        if url.endswith("/health"):
            return Response({"status": "ready"})
        if url.endswith("/design-runs/run-test"):
            return Response(run_responses.pop(0))
        if url.endswith("/design-runs/run-test/trace"):
            return Response([])
        raise AssertionError(f"Unexpected API URL: {url}")

    def fake_post(self, url, *, json, timeout):
        del self, timeout
        requests_made.append(("POST", url, json))
        return Response({"run_id": "run-test", "status": "queued"})

    monkeypatch.setattr(requests.sessions.Session, "get", fake_get)
    monkeypatch.setattr(requests.sessions.Session, "post", fake_post)
    entrypoint = Path(__file__).parents[2] / "src/rag_phy/frontend/streamlit_app.py"
    app = AppTest.from_file(str(entrypoint)).run()
    app.text_area[0].set_value("reference engine study")
    app.number_input[0].set_value(3)
    app.button[0].click().run()

    assert not app.exception
    assert (
        "POST",
        "http://127.0.0.1:8000/design-runs",
        {"design_goal": "reference engine study", "trial_count": 3},
    ) in requests_made
    assert ("GET", "http://127.0.0.1:8000/design-runs/run-test") in requests_made
    assert any(element.value == "Run run-test" for element in app.subheader)
    assert any("Status: queued" in element.value for element in app.markdown)

    next(button for button in app.button if button.label == "Refresh run").click().run()

    assert not app.exception
    assert any("Status: completed" in element.value for element in app.markdown)
    assert any("NASA-report.pdf#page=12" in element.value for element in app.code)


def test_http_frontend_search_displays_passage_and_source(monkeypatch) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    requests_made = []

    def fake_get(self, url, *, timeout, params=None):
        del self, timeout
        requests_made.append((url, params))
        if url.endswith("/health"):
            return Response({"status": "ready"})
        if url.endswith("/knowledge/search"):
            return Response(
                [
                    {
                        "text": "NASA describes five SiC/SiC composite systems.",
                        "source_refs": ["NASA-TM-2004-213048.pdf#page=5"],
                        "distance": 0.2,
                    }
                ]
            )
        raise AssertionError(f"Unexpected API URL: {url}")

    monkeypatch.setattr(requests.sessions.Session, "get", fake_get)
    entrypoint = Path(__file__).parents[2] / "src/rag_phy/frontend/streamlit_app.py"
    app = AppTest.from_file(str(entrypoint)).run()
    next(element for element in app.text_input if element.label == "Query").set_value(
        "NASA composite systems"
    )
    app.button[1].click().run()

    assert not app.exception
    assert (
        "http://127.0.0.1:8000/knowledge/search",
        {"q": "NASA composite systems"},
    ) in requests_made
    assert any(
        "NASA describes five SiC/SiC composite systems." in element.value
        for element in app.markdown
    )
    assert any("NASA-TM-2004-213048.pdf#page=5" in element.value for element in app.code)


def test_http_frontend_renders_completed_run_pareto_and_citations(monkeypatch) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    run = {
        "run_id": "run-test",
        "status": "completed",
        "progress": None,
        "error": None,
        "result": {
            "accepted_candidate": {"compressor_pressure_ratio": 7.0},
            "performance": {"thrust_n": 12000.0},
            "pareto_frontier": [
                {
                    "thrust_to_weight_ratio": 2.0,
                    "specific_fuel_consumption_kg_per_n_s": 0.00003,
                }
            ],
            "cited_sources": ["NASA-report.pdf#page=12"],
        },
    }

    def fake_get(self, url, *, timeout, params=None):
        del self, timeout, params
        if url.endswith("/health"):
            return Response({"status": "ready"})
        if url.endswith("/design-runs/run-test"):
            return Response(run)
        if url.endswith("/design-runs/run-test/trace"):
            return Response([])
        raise AssertionError(f"Unexpected API URL: {url}")

    monkeypatch.setattr(requests.sessions.Session, "get", fake_get)
    entrypoint = Path(__file__).parents[2] / "src/rag_phy/frontend/streamlit_app.py"
    app = AppTest.from_file(str(entrypoint)).run()
    app.session_state["run_id"] = "run-test"
    app.run()

    assert not app.exception
    assert any("Pareto frontier" in element.value for element in app.markdown)
    assert any("NASA-report.pdf#page=12" in element.value for element in app.code)
    assert len(app.get("dataframe")) == 1
    assert len(app.get("vega_lite_chart")) == 1
