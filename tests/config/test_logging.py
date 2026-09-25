import json
import logging

from rag_phy.logging import JsonFormatter


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
