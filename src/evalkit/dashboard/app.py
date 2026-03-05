"""Streamlit dashboard for evalkit evaluation results.

Provides interactive views for:
- Evaluation result browsing and filtering
- Model version comparison
- Regression trend charts
- Per-criterion score breakdowns

Launch with: streamlit run src/evalkit/dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _check_streamlit() -> bool:
    """Check if streamlit is available."""
    try:
        import streamlit  # noqa: F401
        return True
    except ImportError:
        return False


def main() -> None:
    """Entry point for the evalkit dashboard."""
    if not _check_streamlit():
        print(
            "Streamlit is not installed. Install it with:\n"
            "  pip install evalkit[dashboard]",
            file=sys.stderr,
        )
        sys.exit(1)

    import streamlit as st
    _run_dashboard()


def _run_dashboard() -> None:
    """Main dashboard layout and logic."""
    import streamlit as st

    st.set_page_config(
        page_title="evalkit Dashboard",
        page_icon="<>",
        layout="wide",
    )

    st.title("evalkit -- LLM Evaluation Dashboard")
    st.markdown("---")

    # Sidebar: database connection
    st.sidebar.header("Configuration")
    db_path = st.sidebar.text_input(
        "DuckDB Database Path",
        value=":memory:",
        help="Path to the evalkit DuckDB database file, or :memory: for demo mode.",
    )

    if db_path == ":memory:":
        st.info(
            "Running in demo mode with an in-memory database. "
            "Connect to a database file to view real evaluation results."
        )
        _show_demo_dashboard(st)
        return

    db_file = Path(db_path)
    if not db_file.exists():
        st.error(f"Database file not found: {db_path}")
        return

    from evalkit.core.storage import DuckDBStorage

    try:
        storage = DuckDBStorage(db_path=db_path)
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        return

    _show_live_dashboard(st, storage)


def _show_demo_dashboard(st: object) -> None:
    """Display a demo dashboard with sample data."""
    import streamlit as _st
    st = _st

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Evaluations", "0", help="Connect a database to see real data")
    with col2:
        st.metric("Models Tracked", "0")
    with col3:
        st.metric("Regressions Detected", "0")

    st.markdown("## Getting Started")
    st.code(
        """
from evalkit.core.storage import DuckDBStorage
from evalkit.regression.tracker import RegressionTracker
from evalkit.core.models import EvalResult

# Create a persistent database
storage = DuckDBStorage(db_path="./evalkit.duckdb")
tracker = RegressionTracker(storage=storage)

# Record evaluation results
result = EvalResult(
    model_id="my-model",
    model_version="v1.0",
    input_text="What is Python?",
    output_text="Python is a programming language...",
    aggregate_score=4.2,
)
tracker.record(result)

# Then launch the dashboard pointing to your database:
# evalkit  (or: streamlit run src/evalkit/dashboard/app.py)
        """,
        language="python",
    )


def _show_live_dashboard(st_module: object, storage: object) -> None:
    """Display the live dashboard connected to a real database."""
    import streamlit as st
    from evalkit.core.storage import DuckDBStorage

    assert isinstance(storage, DuckDBStorage)

    # Overview metrics
    total = storage.count_results()
    results = storage.get_results(limit=100)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Evaluations", str(total))
    with col2:
        model_ids = {r["model_id"] for r in results}
        st.metric("Models Tracked", str(len(model_ids)))
    with col3:
        versions = {r["model_version"] for r in results}
        st.metric("Versions Recorded", str(len(versions)))

    st.markdown("---")

    # Results table
    st.subheader("Recent Evaluations")
    if results:
        display_data = [
            {
                "ID": r["id"][:8] + "...",
                "Model": r["model_id"],
                "Version": r["model_version"],
                "Score": r.get("aggregate_score", "N/A"),
                "Rubric": r.get("rubric_name", "default"),
                "Created": r.get("created_at", ""),
            }
            for r in results[:50]
        ]
        st.dataframe(display_data, use_container_width=True)
    else:
        st.info("No evaluation results found in the database.")

    # Model comparison
    st.markdown("---")
    st.subheader("Model Version Comparison")

    if results:
        model_ids_list = sorted({r["model_id"] for r in results})
        selected_model = st.selectbox("Select Model", model_ids_list)

        if selected_model:
            model_results = storage.get_results(model_id=selected_model)
            version_scores: dict[str, list[float]] = {}
            for r in model_results:
                v = r["model_version"]
                score = r.get("aggregate_score")
                if score is not None:
                    version_scores.setdefault(v, []).append(score)

            if version_scores:
                chart_data = {
                    v: sum(scores) / len(scores)
                    for v, scores in version_scores.items()
                }
                st.bar_chart(chart_data)
            else:
                st.info("No scored results for this model.")


if __name__ == "__main__":
    main()
