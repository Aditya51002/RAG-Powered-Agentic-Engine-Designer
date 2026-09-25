"""Correlated workflow span events and a dependency-free JSONL trace sink."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkflowTraceEvent:
    """One correlated workflow span; external backends can map this to their trace API."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    node: str
    iteration: int
    message: str
    timestamp: str


class WorkflowTraceSink(Protocol):
    """Persistence interface for workflow span events."""

    def emit(self, event: WorkflowTraceEvent) -> None:
        """Persist or export one span event."""


class JsonlTraceSink:
    """Thread-safe append-only local trace sink, usable without Langfuse credentials."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    def emit(self, event: WorkflowTraceEvent) -> None:
        """Append one JSON object per workflow transition."""
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(asdict(event), sort_keys=True) + "\n")
        except OSError:
            logger.exception("Failed to persist workflow trace", extra={"path": str(self._path)})
            raise


def new_trace_event(
    trace_id: str,
    parent_span_id: str | None,
    node: str,
    iteration: int,
    message: str,
) -> WorkflowTraceEvent:
    """Construct a timestamped event; kept separate for deterministic sink testing."""
    import uuid

    return WorkflowTraceEvent(
        trace_id=trace_id,
        span_id=uuid.uuid4().hex,
        parent_span_id=parent_span_id,
        node=node,
        iteration=iteration,
        message=message,
        timestamp=datetime.now(UTC).isoformat(),
    )


__all__ = ["JsonlTraceSink", "WorkflowTraceEvent", "WorkflowTraceSink"]
