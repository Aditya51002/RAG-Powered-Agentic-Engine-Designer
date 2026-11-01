"""Small pure helpers for checking citation references in provenance ledgers."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

_HTTP_URL = re.compile(r"https?://[^\s<>\"'`]+", re.IGNORECASE)


def has_checkable_http_reference(text: str) -> bool:
    """Return whether text contains an HTTP(S) URL with a valid authority and port."""
    for match in _HTTP_URL.finditer(text):
        trailing_text = text[match.end() :]
        if re.match(r"\s+[^\s/]+/", trailing_text):
            continue
        value = match.group().rstrip(".,;:")
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError:
            continue
        if (
            parsed.scheme.lower() in {"http", "https"}
            and parsed.hostname
            and not any(character.isspace() for character in parsed.netloc)
            and (port is None or 0 < port <= 65535)
        ):
            return True
    return False


__all__ = ["has_checkable_http_reference"]
