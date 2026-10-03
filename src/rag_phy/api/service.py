"""Dependency-injected FastAPI routes for asynchronous design runs and retrieval."""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from fastapi import BackgroundTasks, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from rag_phy.api.schemas import (
    DesignProgressResponse,
    DesignRunRequest,
    DesignRunResponse,
    DesignRunResultResponse,
    DesignRunSubmissionResponse,
    HealthResponse,
    KnowledgeResultResponse,
    TraceEventResponse,
)
from rag_phy.api.store import InMemoryRunStore
from rag_phy.request_context import request_id_context

logger = logging.getLogger(__name__)


class DesignRunExecutor(Protocol):
    """Deployment adapter that maps internal optimizer objects to public schemas."""

    def run(
        self,
        design_goal: str,
        trial_count: int,
        on_progress: Callable[[DesignProgressResponse], None],
        trace_path: Path,
        request_id: str,
    ) -> DesignRunResultResponse:
        """Run the domain workflow and return explicitly mapped API response models."""


class KnowledgeSearchResult(Protocol):
    """Minimum passage surface mapped into the public knowledge response."""

    text: str
    source_refs: Sequence[str]
    distance: float


class APIConfigurationError(RuntimeError):
    """Raised when an endpoint's deployment-owned dependency is not configured."""


def create_app(
    *,
    run_executor: DesignRunExecutor | None = None,
    knowledge_search: Callable[[str], Sequence[KnowledgeSearchResult]] | None = None,
    readiness_check: Callable[[], bool] | None = None,
    trace_directory: str | Path = "reports/traces/design-runs",
    run_store: InMemoryRunStore | None = None,
) -> FastAPI:
    """Construct the API around injected runtime capabilities; no live dependency is implicit."""
    app = FastAPI(title="RAG Phy API", version="0.1.0")
    store = run_store or InMemoryRunStore()
    trace_root = Path(trace_directory)
    app.state.run_store = store

    @app.middleware("http")
    async def correlate_request(request: Request, call_next):
        supplied_id = request.headers.get("x-request-id", "").strip()
        valid_id = bool(re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", supplied_id))
        correlation_id = supplied_id if valid_id else uuid.uuid4().hex
        token = request_id_context.set(correlation_id)
        try:
            response = await call_next(request)
            response.headers["x-request-id"] = correlation_id
            return response
        finally:
            request_id_context.reset(token)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        del request, error
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "validation_error", "message": "Invalid request"}},
        )

    @app.exception_handler(Exception)
    async def internal_error_handler(request: Request, error: Exception) -> JSONResponse:
        logger.exception("Unhandled API request failure", extra={"path": request.url.path})
        del error
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": {"code": "internal_error", "message": "Internal server error"}},
        )

    def execute_run(run_id: str, payload: DesignRunRequest, request_id: str) -> None:
        assert run_executor is not None
        trace_path = trace_root / f"{run_id}.jsonl"
        store.mark_running(run_id)
        token = request_id_context.set(request_id)
        try:
            result = run_executor.run(
                payload.design_goal,
                payload.trial_count,
                lambda progress: store.update_progress(run_id, progress),
                trace_path,
                request_id,
            )
            store.complete(run_id, result)
        except Exception:
            logger.exception("Design run failed", extra={"run_id": run_id})
            store.fail(run_id)
        finally:
            request_id_context.reset(token)

    @app.post(
        "/design-runs",
        response_model=DesignRunSubmissionResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def submit_design_run(
        payload: DesignRunRequest, background_tasks: BackgroundTasks, request: Request
    ) -> DesignRunSubmissionResponse:
        if run_executor is None:
            raise APIConfigurationError("Design-run executor is not configured")
        run_id = str(uuid.uuid4())
        store.create(run_id)
        background_tasks.add_task(
            execute_run,
            run_id,
            payload,
            request_id_context.get() or request.headers.get("x-request-id", "") or run_id,
        )
        return DesignRunSubmissionResponse(run_id=run_id, status="queued")

    @app.get("/design-runs/{run_id}", response_model=DesignRunResponse)
    def get_design_run(run_id: str) -> DesignRunResponse:
        record = store.get(run_id)
        if record is None:
            return _error_response(404, "run_not_found", "Design run not found")
        return record

    @app.get("/design-runs/{run_id}/trace", response_model=list[TraceEventResponse])
    def get_design_run_trace(run_id: str) -> list[TraceEventResponse] | JSONResponse:
        if store.get(run_id) is None:
            return _error_response(404, "run_not_found", "Design run not found")
        trace_path = trace_root / f"{run_id}.jsonl"
        if not trace_path.exists():
            return []
        try:
            events = [
                TraceEventResponse.model_validate_json(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, ValueError):
            logger.exception("Could not read design-run trace", extra={"run_id": run_id})
            return _error_response(500, "trace_unavailable", "Trace is unavailable")
        return events

    @app.get("/knowledge/search", response_model=list[KnowledgeResultResponse])
    def search_knowledge(q: str) -> list[KnowledgeResultResponse]:
        if not q.strip():
            return _error_response(422, "invalid_query", "Query must not be empty")
        if knowledge_search is None:
            raise APIConfigurationError("Knowledge search is not configured")
        results = knowledge_search(q.strip())
        return [
            KnowledgeResultResponse(
                text=result.text,
                source_refs=tuple(result.source_refs),
                distance=result.distance,
            )
            for result in results
        ]

    @app.get("/health", response_model=HealthResponse)
    def health() -> JSONResponse | HealthResponse:
        try:
            ready = readiness_check is not None and readiness_check()
        except Exception:
            logger.exception("API readiness check failed")
            ready = False
        response = HealthResponse(status="ready" if ready else "not_ready")
        return response if ready else JSONResponse(status_code=503, content=response.model_dump())

    @app.exception_handler(APIConfigurationError)
    async def configuration_error_handler(
        request: Request, error: APIConfigurationError
    ) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": {"code": "not_configured", "message": str(error)}},
        )

    return app


def _error_response(code: int, error_code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content={"error": {"code": error_code, "message": message}},
    )


__all__ = ["APIConfigurationError", "DesignRunExecutor", "create_app"]
