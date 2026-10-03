"""Uvicorn entry point; deployment code should call ``create_app`` with real adapters."""

from rag_phy.api.service import create_app

app = create_app()
