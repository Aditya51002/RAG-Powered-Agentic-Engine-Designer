"""Streamlit presentation layer and deployment service contract."""

from rag_phy.app.dashboard import (
    DashboardProgress,
    DashboardRunResult,
    DashboardService,
    render_dashboard,
)
from rag_phy.app.runtime import DashboardNotConfiguredError, load_dashboard_service
from rag_phy.app.service import OptimizationDashboardService

__all__ = [
    "DashboardNotConfiguredError",
    "DashboardProgress",
    "DashboardRunResult",
    "DashboardService",
    "OptimizationDashboardService",
    "load_dashboard_service",
    "render_dashboard",
]
