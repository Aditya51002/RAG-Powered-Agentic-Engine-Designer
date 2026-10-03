"""Frontend HTTP-client tests against a tiny fake session."""

from __future__ import annotations

import pytest
import requests

from rag_phy.frontend import APIClientError, RAGPhyAPIClient


class _Response:
    def __init__(self, payload, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error

    def raise_for_status(self) -> None:
        if self.error:
            self.error.response = self
            raise self.error

    def json(self):
        return self.payload


class _Session:
    def __init__(self, payloads) -> None:
        self.payloads = list(payloads)
        self.calls = []

    def get(self, url, *, timeout, params=None):
        self.calls.append(("GET", url, timeout, params))
        return self.payloads.pop(0)

    def post(self, url, *, json, timeout):
        self.calls.append(("POST", url, timeout, json))
        return self.payloads.pop(0)


def test_client_calls_only_public_http_endpoints() -> None:
    session = _Session(
        [
            _Response({"status": "ready"}),
            _Response({"run_id": "run-1", "status": "queued"}),
            _Response({"status": "running"}),
            _Response([]),
            _Response([]),
        ]
    )
    client = RAGPhyAPIClient("http://localhost:8000/", timeout_seconds=4, session=session)

    assert client.health() == {"status": "ready"}
    assert client.submit("goal", 4)["run_id"] == "run-1"
    assert client.get_run("run-1")["status"] == "running"
    assert client.get_trace("run-1") == []
    assert client.search("CMC") == []
    assert session.calls[1] == (
        "POST",
        "http://localhost:8000/design-runs",
        4,
        {"design_goal": "goal", "trial_count": 4},
    )
    assert session.calls[-1][3] == {"q": "CMC"}


def test_client_translates_structured_server_errors() -> None:
    error = requests.HTTPError("internal details")
    session = _Session(
        [_Response({"error": {"code": "not_configured", "message": "API not ready"}}, error)]
    )
    client = RAGPhyAPIClient("http://localhost:8000", timeout_seconds=1, session=session)

    with pytest.raises(APIClientError, match="API not ready"):
        client.health()


def test_client_rejects_invalid_configuration_and_shapes() -> None:
    with pytest.raises(ValueError, match="base URL"):
        RAGPhyAPIClient(" ", timeout_seconds=1)
    with pytest.raises(ValueError, match="positive"):
        RAGPhyAPIClient("http://localhost", timeout_seconds=0)

    client = RAGPhyAPIClient(
        "http://localhost",
        timeout_seconds=1,
        session=_Session([_Response({"not": "a list"})]),
    )
    with pytest.raises(APIClientError, match="invalid trace"):
        client.get_trace("run-1")
