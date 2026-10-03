"""Thread-safe in-memory design-run records for the API prototype."""

from __future__ import annotations

import threading
from datetime import UTC, datetime

from rag_phy.api.schemas import (
    DesignProgressResponse,
    DesignRunResponse,
    DesignRunResultResponse,
)


class InMemoryRunStore:
    """Store run state in process memory; records are lost when the API restarts."""

    def __init__(self) -> None:
        self._records: dict[str, DesignRunResponse] = {}
        self._lock = threading.RLock()

    def create(self, run_id: str) -> DesignRunResponse:
        now = datetime.now(UTC)
        record = DesignRunResponse(
            run_id=run_id,
            status="queued",
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._records[run_id] = record
        return record

    def get(self, run_id: str) -> DesignRunResponse | None:
        with self._lock:
            record = self._records.get(run_id)
            return record.model_copy(deep=True) if record is not None else None

    def mark_running(self, run_id: str) -> None:
        self._replace(run_id, status="running", error=None)

    def update_progress(self, run_id: str, progress: DesignProgressResponse) -> None:
        self._replace(run_id, status="running", progress=progress)

    def complete(self, run_id: str, result: DesignRunResultResponse) -> None:
        self._replace(run_id, status="completed", result=result, error=None)

    def fail(self, run_id: str) -> None:
        self._replace(run_id, status="failed", error="Design run failed; inspect server logs")

    def _replace(self, run_id: str, **updates: object) -> None:
        with self._lock:
            current = self._records.get(run_id)
            if current is None:
                raise KeyError(f"Unknown design run {run_id}")
            self._records[run_id] = current.model_copy(
                update={**updates, "updated_at": datetime.now(UTC)},
                deep=True,
            )


__all__ = ["InMemoryRunStore"]
