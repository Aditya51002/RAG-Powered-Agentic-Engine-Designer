"""Shared structured logging setup."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from rag_phy.config import AppConfig


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize timestamp, severity, logger, message, and exception details."""
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        standard_fields = set(logging.makeLogRecord({}).__dict__)
        context = {
            key: value
            for key, value in record.__dict__.items()
            if key not in standard_fields and key not in {"message", "asctime"}
        }
        if context:
            payload["context"] = context
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True, default=str)


def configure_logging(config: AppConfig) -> None:
    """Configure the root logger from validated settings."""
    handler = logging.StreamHandler()
    if config.logging.json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(config.logging.level)


__all__ = ["JsonFormatter", "configure_logging"]
