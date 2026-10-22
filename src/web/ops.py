"""Read-only data for the /ops page (demo output folders). Standard library only."""
import json
from pathlib import Path

from src.monitoring import signals
from src.web import storage


def _jsonl(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def sparkline(values, w=220, h=46):
    """SVG polyline points for a series (None values skipped)."""
    pts = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(pts) < 2:
        return ""
    lo, hi = min(v for _, v in pts), max(v for _, v in pts)
    span = (hi - lo) or 1.0
    n = len(values) - 1 or 1
    return " ".join(f"{i / n * w:.1f},{h - 4 - (v - lo) / span * (h - 8):.1f}" for i, v in pts)


def load_state(folder):
    folder = Path(folder)
    sig, hist = _jsonl(folder / "signals_log.jsonl"), _jsonl(folder / "retrain_history.jsonl")
    version = None
    model_file = folder / "production.json"
    if model_file.exists():
        version = json.loads(model_file.read_text(encoding="utf-8")).get("version")
    complaints = {}
    if (folder / "api.db").exists():
        conn = storage.connect(str(folder / "api.db"))
        res = signals.complaint_volume(conn, signals.load_thresholds()["complaints"])
        complaints = dict(sorted(((k, v["n"]) for k, v in res["trains"].items()), key=lambda kv: -kv[1]))
        conn.close()
    return {
        "model_version": version, "history": hist, "signals": sig, "complaints": complaints,
        "promotions": sum(1 for h in hist if h.get("promoted")),
        "spark": {k: sparkline([s.get(k) for s in sig]) for k in ("max_error_mae", "drift_share", "max_complaints")},
    }
