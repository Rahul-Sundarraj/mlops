import json

import mlflow
import mlflow.sklearn
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from src.api import storage
from src.data.simulate import load_config, simulate_delays
from src.features.build import add_features, attach_hist, hist_avg_table
from src.models.train import fit_candidates
from src.monitoring.drift import detect_drift
from src.orchestrator import run as orch

TINY = {"xgboost": {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.2},
        "lightgbm": {"n_estimators": 20, "num_leaves": 7, "learning_rate": 0.2},
        "experiment": "t", "registered_model": "m"}
ORCH = {"cooldown_days": 3, "holdout_days": 7, "promote_tolerance_min": 0.0}
TH = {"drift": {"min_rows": 50, "share_threshold": 0.3, "columns": ["rain_mm", "is_festival", "region"]},
      "rolling_error": {"window_days": 7, "min_samples": 3, "mae_threshold": 15.0},
      "complaints": {"window_days": 7, "default_threshold": 2, "per_train": {}},
      "orchestrator": ORCH}


def test_promote_decision_and_cooldown():
    assert orch.promote_decision(8.0, 8.0, 0.0) and not orch.promote_decision(8.1, 8.0, 0.0)
    assert orch.promote_decision(8.1, 8.0, 0.2)
    hist = [{"ref_date": "2025-11-10"}]
    assert not orch.cooldown_ok("2025-11-12", 3, hist) and orch.cooldown_ok("2025-11-13", 3, hist)
    assert orch.cooldown_ok("2025-11-01", 3, [])


@pytest.fixture(scope="module")
def data():
    df = add_features(simulate_delays(load_config(), "2025-01-01", "2025-03-31", 5))
    train = df[df.date < "2025-03-01"]
    table = hist_avg_table(train)
    return attach_hist(train, table), attach_hist(df[df.date >= "2025-03-01"], table)


def test_drift_detected_on_shifted_inputs(data):
    train, _ = data
    cfg = TH["drift"]
    assert not detect_drift(train, train.sample(500, random_state=1), cfg)["breach"]
    shifted = train.sample(500, random_state=1).assign(rain_mm=lambda d: d.rain_mm + 40, is_festival=1)
    out = detect_drift(train, shifted, cfg)
    assert out["breach"] and "rain_mm" in out["drifted_columns"]


def test_retrain_cycle_registers_candidate_and_logs_history(tmp_path, data):
    train, new = data
    params = TINY | {"mlflow_uri": f"sqlite:///{tmp_path / 'm.db'}"}
    mlflow.set_tracking_uri(params["mlflow_uri"])
    mlflow.set_experiment("t")
    name, pipe, _ = fit_candidates(train, new, params)
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")
    mv = mlflow.register_model(info.model_uri, "m")
    MlflowClient().set_registered_model_alias("m", "production", mv.version)

    conn = storage.connect(str(tmp_path / "a.db"))
    for _ in range(3):  # complaint spike on one train
        conn.execute("INSERT INTO complaints(train_no, date, text, created_at) VALUES (12675,'2025-03-20','x','t')")
    conn.commit()
    hist_path = tmp_path / "h.jsonl"
    live = train.sample(10, random_state=0)  # too few rows: drift must stay quiet
    out = orch.run_cycle(conn, params, TH, train, train, live, new, "2025-03-20", hist_path)
    assert out["signals"]["triggers"] == ["complaints"] and out["retrained"]
    rec = out["record"]
    assert rec["reasons"] == ["complaints"] and rec["candidate_version"] == 2
    prod = MlflowClient().get_model_version_by_alias("m", "production").version
    assert str(prod) == ("2" if rec["promoted"] else "1")
    assert json.loads(hist_path.read_text().splitlines()[0])["ref_date"] == "2025-03-20"

    again = orch.run_cycle(conn, params, TH, train, train, live, new, "2025-03-21", hist_path)
    assert not again["retrained"] and again["skipped"] == "cooldown"
    quiet = orch.run_cycle(conn, params, TH, train, train, live, new, "2025-03-30", hist_path)
    assert quiet["skipped"] == "no trigger" or quiet["retrained"]  # complaints age out of the window
