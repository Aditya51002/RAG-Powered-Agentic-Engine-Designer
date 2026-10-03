import json
import logging

from rag_phy.logging import JsonFormatter
from rag_phy.request_context import request_id_context


def test_json_formatter_preserves_structured_context() -> None:
    """Include operational context fields in structured log output."""
    record = logging.LogRecord(
        name="rag_phy.ingestion",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Document ingestion completed",
        args=(),
        exc_info=None,
    )
    record.chunk_count = 7
    record.source = "fixture.pdf"

    formatted = json.loads(JsonFormatter().format(record))

    assert formatted["context"] == {"chunk_count": 7, "source": "fixture.pdf"}


def test_json_formatter_includes_context_local_request_id() -> None:
    record = logging.LogRecord(
        name="rag_phy.api",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Request accepted",
        args=(),
        exc_info=None,
    )
    token = request_id_context.set("request-123")
    try:
        formatted = json.loads(JsonFormatter().format(record))
    finally:
        request_id_context.reset(token)

    assert formatted["context"]["request_id"] == "request-123"
