"""Tests for deployment-owned dashboard service loading and setup errors."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from rag_phy.app.runtime import DashboardNotConfiguredError, load_dashboard_service
from rag_phy.config import AppConfig, DashboardConfig


def test_missing_dashboard_factory_fails_with_setup_guidance() -> None:
    with pytest.raises(DashboardNotConfiguredError, match="No dashboard service"):
        load_dashboard_service(AppConfig())


def test_dashboard_factory_is_loaded_from_configured_import_path(monkeypatch) -> None:
    module = ModuleType("synthetic_dashboard_backend")

    class Service:
        def run_optimization(self, design_goal, additional_constraints, on_progress):
            return None

    module.create_service = Service
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = AppConfig(dashboard=DashboardConfig(service_factory="synthetic_dashboard_backend:create_service"))

    assert isinstance(load_dashboard_service(config), Service)


def test_dashboard_factory_rejects_incompatible_service(monkeypatch) -> None:
    module = ModuleType("invalid_dashboard_backend")
    module.create_service = lambda: object()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = AppConfig(dashboard=DashboardConfig(service_factory="invalid_dashboard_backend:create_service"))

    with pytest.raises(TypeError, match="incompatible service"):
        load_dashboard_service(config)
