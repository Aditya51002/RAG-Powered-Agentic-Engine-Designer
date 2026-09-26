"""Resolve the deployment-provided optimization service from typed configuration."""

from __future__ import annotations

import importlib
import logging
from typing import Any

from rag_phy.config import AppConfig

logger = logging.getLogger(__name__)


class DashboardNotConfiguredError(RuntimeError):
    """Raised when no real optimization service factory has been configured."""


def load_dashboard_service(config: AppConfig) -> Any:
    """Import and invoke the configured ``module:function`` service factory."""
    factory_path = config.dashboard.service_factory
    if factory_path is None:
        raise DashboardNotConfiguredError(
            "No dashboard service is configured. Set dashboard.service_factory to a factory "
            "that assembles real LLM, retrieval, constraint, sampler, and weight adapters."
        )
    module_name, separator, function_name = factory_path.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("dashboard.service_factory must use the form 'module.path:function'")
    try:
        factory = getattr(importlib.import_module(module_name), function_name)
        service = factory()
    except Exception as exc:
        logger.exception("Dashboard service factory failed", extra={"factory": factory_path})
        raise RuntimeError(f"Could not initialize dashboard service {factory_path!r}") from exc
    if not callable(getattr(service, "run_optimization", None)):
        raise TypeError(
            f"Dashboard service factory {factory_path!r} returned an incompatible service"
        )
    return service


__all__ = ["DashboardNotConfiguredError", "load_dashboard_service"]
