"""Context-local request correlation ID shared by HTTP middleware and structured logs."""

from __future__ import annotations

from contextvars import ContextVar

request_id_context: ContextVar[str | None] = ContextVar("rag_phy_request_id", default=None)


__all__ = ["request_id_context"]
