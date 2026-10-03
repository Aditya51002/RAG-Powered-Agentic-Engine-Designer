"""Uvicorn entry point; deployment code should call ``create_app`` with real adapters."""

from rag_phy.api.service import create_app
from rag_phy.api.readiness import make_readiness_check

app = create_app(readiness_check=make_readiness_check())
