"""HTTP API for injected RAG Phy design and retrieval services."""

from importlib import import_module

from rag_phy.api.service import APIConfigurationError, DesignRunExecutor, create_app
from rag_phy.api.store import InMemoryRunStore

_LAZY_EXPORTS = {
    "ConfiguredKnowledgeSearch": ("rag_phy.api.knowledge", "ConfiguredKnowledgeSearch"),
    "OptimizationDesignRunExecutor": (
        "rag_phy.api.runtime",
        "OptimizationDesignRunExecutor",
    ),
    "make_optimizer_factory": ("rag_phy.api.composition", "make_optimizer_factory"),
    "make_readiness_check": ("rag_phy.api.readiness", "make_readiness_check"),
}


def __getattr__(name: str):
    """Import optional runtime components only when a deployment requests them."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "APIConfigurationError",
    "ConfiguredKnowledgeSearch",
    "DesignRunExecutor",
    "InMemoryRunStore",
    "OptimizationDesignRunExecutor",
    "create_app",
    "make_optimizer_factory",
    "make_readiness_check",
]
