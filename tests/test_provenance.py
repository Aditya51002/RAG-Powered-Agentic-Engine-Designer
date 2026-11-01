"""Tests for checkable source references used by data/config validators."""

import pytest

from rag_phy.provenance import has_checkable_http_reference


@pytest.mark.parametrize(
    "text",
    [
        "Reference: https://ntrs.nasa.gov/report.pdf.",
        "[Report](http://example.org:8080/path)",
    ],
)
def test_accepts_http_references_with_host_and_valid_port(text: str) -> None:
    assert has_checkable_http_reference(text)


@pytest.mark.parametrize(
    "text",
    [
        "https://",
        "https://example.org:invalid/report",
        "https://example.org:70000/report",
        "https://example org/report",
        "This prose contains https without a URL.",
    ],
)
def test_rejects_malformed_or_missing_http_authority(text: str) -> None:
    assert not has_checkable_http_reference(text)
