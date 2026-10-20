"""Statistical data-drift signal using Evidently (live feature window vs. training reference)."""
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

CATEGORICAL = {"region", "category", "station", "is_festival"}


def recent_features(conn, ref_date, window_days):
    """Feature rows logged by /predict for the window ending at ref_date (by prediction date)."""
    ref = date.fromisoformat(ref_date) if isinstance(ref_date, str) else ref_date
    lo = (ref - timedelta(days=window_days - 1)).isoformat()
    rows = conn.execute("SELECT features FROM predictions WHERE date BETWEEN ? AND ? AND features IS NOT NULL",
                        (lo, ref.isoformat())).fetchall()
    return pd.DataFrame([json.loads(r["features"]) for r in rows])


def detect_drift(reference, live, cfg, html_path=None):
    """Share of monitored columns Evidently flags as drifted; breach when share >= share_threshold."""
    cols = [c for c in cfg["columns"] if c in reference.columns and c in live.columns]
    if len(live) < cfg["min_rows"] or not cols:
        return {"breach": False, "share": 0.0, "drifted_columns": [], "n_rows": len(live),
                "note": "not enough live rows"}
    dd = DataDefinition(numerical_columns=[c for c in cols if c not in CATEGORICAL],
                        categorical_columns=[c for c in cols if c in CATEGORICAL])
    snap = Report([DataDriftPreset()]).run(Dataset.from_pandas(live[cols], data_definition=dd),
                                           Dataset.from_pandas(reference[cols], data_definition=dd))
    metrics = snap.dict()["metrics"]
    share = float(metrics[0]["value"]["share"])
    drifted = [m["config"]["column"] for m in metrics[1:]
               if m["config"].get("type", "").endswith("ValueDrift") and m["value"] > m["config"]["threshold"]]
    if html_path:
        Path(html_path).parent.mkdir(parents=True, exist_ok=True)
        snap.save_html(str(html_path))
    return {"breach": share >= cfg["share_threshold"], "share": round(share, 3),
            "drifted_columns": drifted, "n_rows": len(live)}

