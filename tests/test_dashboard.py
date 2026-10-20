import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.api import storage
from src.dashboard.app import load_state


def make_folder(tmp_path):
    (tmp_path / "s1").mkdir()
    d = tmp_path / "s1"
    (d / "signals_log.jsonl").write_text(json.dumps(
        {"ref_date": "2025-10-01", "drift_share": 0.4, "max_error_mae": 20.0, "max_complaints": 3,
         "triggers": ["drift"], "retrained": True, "promoted": True}) + "\n")
    (d / "retrain_history.jsonl").write_text(json.dumps(
        {"ref_date": "2025-10-01", "reasons": ["drift"], "candidate_model": "lightgbm", "candidate_version": 2,
         "candidate_holdout_mae": 9.0, "champion_holdout_mae": 9.5, "promoted": True, "run_id": "x"}) + "\n")
    conn = storage.connect(str(d / "api.db"))
    conn.execute("INSERT INTO complaints(train_no, date, text, created_at) VALUES (12675,'2025-10-01','x','t')")
    conn.commit()
    return tmp_path


def test_load_state(tmp_path):
    s = load_state(make_folder(tmp_path) / "s1")
    assert s["complaints"] == {12675: 1} and len(s["history"]) == 1 and s["model_version"] is None


def test_dashboard_renders(tmp_path, monkeypatch):
    monkeypatch.setenv("DASH_DIR", str(make_folder(tmp_path)))
    at = AppTest.from_file(str(Path(__file__).parents[1] / "src/dashboard/app.py"), default_timeout=60).run()
    assert not at.exception
    assert [m.label for m in at.metric] == ["Production model version", "Retrains", "Promotions", "Latest drift share"]
    assert at.metric[1].value == "1"
