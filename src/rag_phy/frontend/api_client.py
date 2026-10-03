"""Small HTTP client; this package depends only on the public API contract."""

from __future__ import annotations

from typing import Any, Protocol

import requests


class _Response(Protocol):
    def raise_for_status(self) -> None:
        """Raise for non-success HTTP status codes."""

    def json(self) -> Any:
        """Decode a JSON response body."""


class _Session(Protocol):
    def get(self, url: str, *, timeout: float, params: dict[str, str] | None = None) -> _Response:
        """Send a GET request."""

    def post(self, url: str, *, json: dict[str, Any], timeout: float) -> _Response:
        """Send a JSON POST request."""


class APIClientError(RuntimeError):
    """Safe error raised for API transport or structured error responses."""


class RAGPhyAPIClient:
    """Call public RAG Phy API endpoints without importing server-side Python modules."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float,
        session: _Session | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("API base URL must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("HTTP timeout must be positive")
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._session = session or requests.Session()

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def submit(self, design_goal: str, trial_count: int) -> dict[str, Any]:
        return self._post(
            "/design-runs",
            {"design_goal": design_goal, "trial_count": trial_count},
        )

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._get(f"/design-runs/{run_id}")

    def get_trace(self, run_id: str) -> list[dict[str, Any]]:
        result = self._get(f"/design-runs/{run_id}/trace")
        if not isinstance(result, list):
            raise APIClientError("API returned an invalid trace response")
        return result

    def search(self, query: str) -> list[dict[str, Any]]:
        result = self._get("/knowledge/search", params={"q": query})
        if not isinstance(result, list):
            raise APIClientError("API returned an invalid knowledge-search response")
        return result

    def _get(self, path: str, *, params: dict[str, str] | None = None) -> Any:
        return self._request(self._session.get, path, params=params)

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        return self._request(self._session.post, path, payload=payload)

    def _request(self, method, path: str, *, params=None, payload=None) -> Any:
        try:
            if payload is None:
                response = method(
                    f"{self._base_url}{path}",
                    params=params,
                    timeout=self._timeout_seconds,
                )
            else:
                response = method(
                    f"{self._base_url}{path}",
                    json=payload,
                    timeout=self._timeout_seconds,
                )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            message = "Could not reach the RAG Phy API"
            response = getattr(exc, "response", None)
            if response is not None:
                try:
                    error_message = response.json().get("error", {}).get("message")
                except (AttributeError, ValueError):
                    error_message = None
                if isinstance(error_message, str) and error_message:
                    message = error_message
            raise APIClientError(message) from exc


__all__ = ["APIClientError", "RAGPhyAPIClient"]
