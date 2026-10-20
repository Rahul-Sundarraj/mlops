"""Rolling prediction-error and complaint-volume signals (retrain triggers)."""
from datetime import date, timedelta

import yaml


def load_thresholds(path="configs/thresholds.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def _window(ref, days):
    ref = date.fromisoformat(ref) if isinstance(ref, str) else ref
    return (ref - timedelta(days=days - 1)).isoformat(), ref.isoformat()


def _ref_date(conn, table, ref):
    if ref:
        return ref
    r = conn.execute(f"SELECT MAX(date) d FROM {table}").fetchone()["d"]
    return r


def rolling_error(conn, cfg, ref=None):
    """Per-train MAE of predictions vs. GPS pings over a sliding window."""
    ref = _ref_date(conn, "pings", ref)
    if ref is None:
        return {"ref_date": None, "trains": {}, "breach": False}
    lo, hi = _window(ref, cfg["window_days"])
    rows = conn.execute(
        """SELECT p.train_no, COUNT(*) n, AVG(ABS(p.actual_delay - pr.predicted_delay)) mae
           FROM pings p JOIN predictions pr
             ON pr.id = (SELECT MAX(id) FROM predictions
                         WHERE train_no = p.train_no AND station = p.station AND date = p.date)
           WHERE p.date BETWEEN ? AND ? GROUP BY p.train_no""", (lo, hi)).fetchall()
    trains = {}
    for r in rows:
        breach = r["n"] >= cfg["min_samples"] and r["mae"] > cfg["mae_threshold"]
        trains[r["train_no"]] = {"n": r["n"], "mae": round(r["mae"], 2), "breach": breach}
    return {"ref_date": hi, "trains": trains, "breach": any(t["breach"] for t in trains.values())}


def complaint_volume(conn, cfg, ref=None):
    """Per-train complaint counts over a sliding window vs. route-specific thresholds."""
    ref = _ref_date(conn, "complaints", ref)
    if ref is None:
        return {"ref_date": None, "trains": {}, "breach": False}
    lo, hi = _window(ref, cfg["window_days"])
    rows = conn.execute("SELECT train_no, COUNT(*) n FROM complaints WHERE date BETWEEN ? AND ? "
                        "GROUP BY train_no", (lo, hi)).fetchall()
    trains = {}
    for r in rows:
        thr = cfg["per_train"].get(r["train_no"], cfg["default_threshold"])
        trains[r["train_no"]] = {"n": r["n"], "threshold": thr, "breach": r["n"] > thr}
    return {"ref_date": hi, "trains": trains, "breach": any(t["breach"] for t in trains.values())}
