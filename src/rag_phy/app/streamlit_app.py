"""Streamlit entrypoint for ``streamlit run src/rag_phy/app/streamlit_app.py``."""

from __future__ import annotations

import logging

from rag_phy.app.dashboard import render_dashboard
from rag_phy.app.runtime import DashboardNotConfiguredError, load_dashboard_service
from rag_phy.config import load_config

logger = logging.getLogger(__name__)


def main() -> None:
    """Load app configuration and render the configured optimization dashboard."""
    import streamlit as st

    st.set_page_config(page_title="RAG Phy", layout="wide")
    config = load_config("config/app.yaml")
    try:
        service = load_dashboard_service(config)
    except DashboardNotConfiguredError as exc:
        st.title("RAG Phy")
        st.error(str(exc))
        st.caption(
            "Production inputs are not present in this workspace; no synthetic optimization "
            "or engineering verdict is substituted."
        )
        return
    except Exception as exc:
        logger.exception("Dashboard startup failed")
        st.title("RAG Phy")
        st.error(f"Dashboard startup failed: {exc}")
        return
    render_dashboard(service)


if __name__ == "__main__":
    main()
