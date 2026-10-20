"""Streamlit monitoring dashboard.

Run: streamlit run src/dashboard/app.py          (reads data/demo/<scenario>/ written by the demo)
Env: DASH_DIR overrides the folder holding scenario sub-folders.
"""
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.api import storage
from src.monitoring import signals

ROOT = Path(os.environ.get("DASH_DIR", "data/demo"))


def _jsonl(path):
    if not Path(path).exists():
        return pd.DataFrame()
    rows = [json.loads(x) for x in open(path) if x.strip()]
    return pd.DataFrame(rows)


def production_version(folder):
    db = Path(folder) / "mlflow.db"
    if not db.exists():
        return None
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{db.as_posix()}")
        return int(mlflow.MlflowClient().get_model_version_by_alias("delay-predictor", "production").version)
    except Exception:
        return None


def load_state(folder):
    """Everything the dashboard shows, as plain data (kept separate from Streamlit for testing)."""
    folder = Path(folder)
    sig, hist = _jsonl(folder / "signals_log.jsonl"), _jsonl(folder / "retrain_history.jsonl")
    complaints = {}
    if (folder / "api.db").exists():
        conn = storage.connect(str(folder / "api.db"))
        res = signals.complaint_volume(conn, signals.load_thresholds()["complaints"], since=None)
        complaints = {k: v["n"] for k, v in res["trains"].items()}
    return {"model_version": production_version(folder), "signals": sig, "history": hist,
            "complaints": complaints}


def main():
    st.set_page_config(page_title="Railway delay model monitor", layout="wide")
    st.title("Delay-aware railway intelligence: model monitor")
    scenarios = sorted(p.name for p in ROOT.iterdir() if p.is_dir()) if ROOT.exists() else []
    if not scenarios:
        st.info(f"No demo data in {ROOT}. Run `python -m src.demo.run_demo` first.")
        return
    scenario = st.sidebar.selectbox("Scenario", scenarios)
    state = load_state(ROOT / scenario)
    sig, hist = state["signals"], state["history"]

    promoted = int(hist["promoted"].sum()) if not hist.empty else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Production model version", state["model_version"] or "n/a")
    c2.metric("Retrains", len(hist))
    c3.metric("Promotions", promoted)
    c4.metric("Latest drift share", f"{sig['drift_share'].iloc[-1]:.2f}" if not sig.empty else "n/a")

    if not sig.empty:
        sig = sig.set_index("ref_date")
        left, mid, right = st.columns(3)
        left.subheader("Worst route rolling MAE (min)")
        left.line_chart(sig["max_error_mae"])
        mid.subheader("Drift share (Evidently)")
        mid.line_chart(sig["drift_share"])
        right.subheader("Max complaints per route")
        right.line_chart(sig["max_complaints"])

    st.subheader("Complaints by route (all logged)")
    if state["complaints"]:
        st.bar_chart(pd.Series(state["complaints"], name="complaints"))
    st.subheader("Retraining history")
    if hist.empty:
        st.write("No retrains yet.")
    else:
        hist = hist.assign(reasons=hist["reasons"].map(", ".join))
        st.dataframe(hist.drop(columns=["run_id"]), use_container_width=True)


main()
