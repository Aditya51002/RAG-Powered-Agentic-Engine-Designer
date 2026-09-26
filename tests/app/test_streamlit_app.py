"""Streamlit startup test for the unconfigured real-service state."""

from pathlib import Path

import pytest


def test_streamlit_entrypoint_renders_setup_status_without_backend() -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    entrypoint = Path(__file__).parents[2] / "src/rag_phy/app/streamlit_app.py"
    app = AppTest.from_file(str(entrypoint)).run()

    assert not app.exception
    assert [element.value for element in app.title] == ["RAG Phy"]
    assert any("No dashboard service is configured" in element.value for element in app.error)
