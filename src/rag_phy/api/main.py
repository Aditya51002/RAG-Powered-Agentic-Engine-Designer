"""Uvicorn entry point; deployment code should call ``create_app`` with real adapters."""

from rag_phy.api.knowledge import ConfiguredKnowledgeSearch
from rag_phy.api.readiness import make_readiness_check
from rag_phy.api.service import create_app
from rag_phy.config import load_config
from rag_phy.logging import configure_logging

configure_logging(load_config("config/app.yaml"))

app = create_app(
    knowledge_search=ConfiguredKnowledgeSearch(),
    readiness_check=make_readiness_check(),
)
