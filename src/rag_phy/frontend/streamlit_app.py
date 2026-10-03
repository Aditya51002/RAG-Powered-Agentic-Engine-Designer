"""Streamlit browser UI that communicates exclusively with the HTTP API."""

from __future__ import annotations

import time

import streamlit as st

from rag_phy.frontend.api_client import APIClientError, RAGPhyAPIClient


def main() -> None:
    st.set_page_config(page_title="RAG Phy", layout="wide")
    st.title("RAG Phy Design Runs")

    api_url = st.sidebar.text_input("API URL", value="http://127.0.0.1:8000")
    timeout = st.sidebar.number_input(
        "Request timeout (seconds)", min_value=1, value=10, step=1
    )
    client = RAGPhyAPIClient(api_url, timeout_seconds=float(timeout))
    try:
        health = client.health()
        if health.get("status") != "ready":
            st.error("The API is not ready")
    except APIClientError as exc:
        st.error(str(exc))

    with st.form("design_run"):
        goal = st.text_area("Design goal")
        trials = st.number_input("Optimizer trials", min_value=1, value=50, step=1)
        submit = st.form_submit_button("Start run")
    if submit:
        try:
            submitted = client.submit(goal, int(trials))
            st.session_state["run_id"] = submitted["run_id"]
            st.rerun()
        except (APIClientError, KeyError) as exc:
            st.error(str(exc) if isinstance(exc, APIClientError) else "Invalid API response")

    run_id = st.session_state.get("run_id")
    if run_id:
        _render_run(client, run_id)
    _render_knowledge_search(client)


def _render_run(client: RAGPhyAPIClient, run_id: str) -> None:
    try:
        run = client.get_run(run_id)
    except APIClientError as exc:
        st.error(str(exc))
        return

    st.subheader(f"Run {run_id}")
    st.write(f"Status: {run['status']}")
    progress = run.get("progress")
    if progress:
        st.progress(min(progress["iteration"] / progress["total_iterations"], 1.0))
        st.caption(
            f"Trial {progress['iteration']} of {progress['total_iterations']} · "
            f"best score {progress['best_score']:.5g} · "
            f"{'valid' if progress['best_valid'] else 'invalid'}"
        )

    if run["status"] == "failed":
        st.error(run.get("error") or "Design run failed")
    result = run.get("result")
    if result:
        left, right = st.columns(2)
        with left:
            st.markdown("**Accepted candidate**")
            st.json(result.get("accepted_candidate"))
        with right:
            st.markdown("**Cycle performance**")
            st.json(result.get("performance"))

        points = result.get("pareto_frontier", [])
        if points:
            st.markdown("**Pareto frontier**")
            rows = [
                {
                    "thrust_to_weight_ratio": point["thrust_to_weight_ratio"],
                    "specific_fuel_consumption_kg_per_n_s": (
                        point["specific_fuel_consumption_kg_per_n_s"]
                    ),
                }
                for point in points
            ]
            st.dataframe(rows, use_container_width=True)
            st.scatter_chart(
                rows,
                x="specific_fuel_consumption_kg_per_n_s",
                y="thrust_to_weight_ratio",
            )
        with st.expander("Approval citations", expanded=True):
            for source in result.get("cited_sources", []):
                st.code(source)

    try:
        traces = client.get_trace(run_id)
        if traces:
            with st.expander("Workflow trace"):
                st.json(traces)
    except APIClientError as exc:
        st.warning(str(exc))

    if run["status"] in {"queued", "running"}:
        auto_refresh = st.checkbox("Auto-refresh", value=False, key=f"refresh-{run_id}")
        if auto_refresh:
            interval = st.selectbox("Refresh interval (seconds)", [2, 5, 10], index=1)
            time.sleep(interval)
            st.rerun()
        else:
            st.button("Refresh run", on_click=st.rerun)


def _render_knowledge_search(client: RAGPhyAPIClient) -> None:
    with st.expander("Knowledge search"):
        with st.form("knowledge_search"):
            query = st.text_input("Query")
            search = st.form_submit_button("Search")
        if search:
            try:
                for result in client.search(query):
                    st.write(result["text"])
                    st.caption(f"Distance: {result['distance']}")
                    for source in result["source_refs"]:
                        st.code(source)
            except (APIClientError, KeyError) as exc:
                st.error(str(exc) if isinstance(exc, APIClientError) else "Invalid API response")


if __name__ == "__main__":
    main()
