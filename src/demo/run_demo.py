"""Replay a simulated incident through the API and watch the orchestrator react.

Scenarios (each runs in its own folder data/demo/<name>/ with its own MLflow registry, so the real one is untouched):
  surge       heavy-rain/festival style surge: delays double, rain inputs shift -> error + drift + complaints fire
  complaints  delays stay normal but one route gets a complaint flood -> only the complaint signal fires

Usage: python -m src.demo.run_demo --scenario surge   (needs `dvc repro` output in data/processed)
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from src.features.build import FEATURES, TARGET
from src.models.export import export_model, hist_from_csv
from src.models.train import fit_candidates
from src.monitoring import drift as drift_mod
from src.monitoring import signals
from src.orchestrator import run as orch
from src.web import storage
from src.web.app import create_app

DEMO_ROOT = Path("data/demo")
NORMAL_DAYS, EVENT_DAYS, SAMPLE_PER_DAY = 7, 14, 250


def bootstrap_champion(train, test, params, demo_dir):
    params = {**params, "mlflow_uri": f"sqlite:///{(demo_dir / 'mlflow.db').as_posix()}"}
    for m in ("xgboost", "lightgbm"):
        params[m] = {**params[m], "n_estimators": 150}
    mlflow.set_tracking_uri(params["mlflow_uri"])
    mlflow.set_experiment(params["experiment"])
    first_week = test[test.date <= sorted(test.date.unique())[NORMAL_DAYS - 1]]
    name, pipe, mae = fit_candidates(train, first_week, params)
    with mlflow.start_run(run_name=f"champion-{name}"):
        info = mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")
    mv = mlflow.register_model(info.model_uri, params["registered_model"])
    MlflowClient().set_registered_model_alias(params["registered_model"], "production", mv.version)
    export_model(pipe, hist_from_csv(), mv.version, demo_dir / "production.json")
    print(f"champion: {name} v{mv.version}, MAE on first week {mae:.2f}")
    return params


def day_frame(test, day, scenario, event):
    rows = test[test.date == day].copy()
    if event and scenario == "surge":
        rows["rain_mm"] += 35
        rows[TARGET] = rows[TARGET] * 2 + 8
    return rows


def run(scenario, seed=7):
    rng = np.random.default_rng(seed)
    demo_dir = DEMO_ROOT / scenario
    shutil.rmtree(demo_dir, ignore_errors=True)
    demo_dir.mkdir(parents=True)

    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    reference = train.sample(5000, random_state=0)
    params = bootstrap_champion(train, test, orch.load_params(), demo_dir)
    os.environ["MLFLOW_URI"] = params["mlflow_uri"]

    th = signals.load_thresholds()
    th["orchestrator"].update(holdout_days=3, cooldown_days=3)
    champ = mlflow.sklearn.load_model(f"models:/{params['registered_model']}@production")
    first_week = test[test.date <= sorted(test.date.unique())[NORMAL_DAYS - 1]]
    th["rolling_error"]["baseline"] = signals.per_train_mae(champ, first_week)
    client = create_app(str(demo_dir / "production.json"), str(demo_dir / "api.db"), th).test_client()
    conn = storage.connect(str(demo_dir / "api.db"))  # second connection for monitor reads

    days = sorted(test.date.unique())[:NORMAL_DAYS + EVENT_DAYS]
    new_rows, log = [], []
    history_path, log_path = demo_dir / "retrain_history.jsonl", demo_dir / "signals_log.jsonl"
    print(f"{'day':<11}{'event':<6}{'max MAE':>8}{'drift':>7}{'complaints':>11}  action")
    for i, day in enumerate(days):
        event = i >= NORMAL_DAYS
        rows = day_frame(test, day, scenario, event)
        sample = rows.sample(min(SAMPLE_PER_DAY, len(rows)), random_state=int(rng.integers(1e9)))
        for r in sample.itertuples():
            client.post("/predict", json={"train_no": int(r.train_no), "station": r.station, "date": day,
                                          "rain_mm": float(r.rain_mm)})
            client.post("/location-ping", json={"train_no": int(r.train_no), "station": r.station,
                                                "date": day, "actual_delay_min": float(getattr(r, TARGET))})
        final = rows[rows.groupby("train_no").stop_seq.transform("max") == rows.stop_seq]
        counts = rng.poisson(0.1 + 0.04 * np.clip(final[TARGET] - 30, 0, None))
        extra = {12675: 4} if (event and scenario == "complaints") else {}
        for tn, n in zip(final.train_no, counts):
            for _ in range(int(n) + extra.get(int(tn), 0)):
                client.post("/complain", json={"train_no": int(tn), "date": day, "text": "delayed"})
        new_rows.append(sample[FEATURES + [TARGET, "date"]])

        since = orch.last_retrain_date(orch.read_history(history_path))
        live = drift_mod.recent_features(conn, day, th["drift"]["window_days"], since)
        out = orch.run_cycle(conn, params, th, train, reference, live, pd.concat(new_rows), day,
                             history_path, log_path, demo_dir / "production.json")
        sig = out["signals"]
        action = out.get("skipped", "")
        if out["retrained"]:
            rec = out["record"]
            verdict = f"PROMOTED v{rec['candidate_version']}" if rec["promoted"] else "rejected"
            action = (f"RETRAIN [{','.join(rec['reasons'])}] cand {rec['candidate_holdout_mae']} vs "
                      f"champ {rec['champion_holdout_mae']} -> {verdict}")
            if rec["promoted"]:
                client.post("/reload-model")
                # the promoted model has seen the new regime: fold it into the drift reference
                reference = pd.concat([reference, pd.concat(new_rows)]).sample(5000, random_state=0)
        elif sig["triggers"]:
            action = f"trigger {sig['triggers']} but {action}"
        mae = max((t["mae"] for t in sig["rolling_error"]["trains"].values()), default=0)
        cmax = max((t["n"] for t in sig["complaints"]["trains"].values()), default=0)
        print(f"{day:<11}{'yes' if event else '':<6}{mae:>8.1f}{sig['drift']['share']:>7.2f}{cmax:>11}  {action}")
        log.append({"day": day, "triggers": sig["triggers"], "retrained": out["retrained"]})
    retrains = [e for e in log if e["retrained"]]
    summary = {"scenario": scenario, "event_start": days[NORMAL_DAYS], "retrains": len(retrains),
               "first_trigger": next((e["day"] for e in log if e["triggers"]), None)}
    (demo_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=["surge", "complaints", "all"], default="all")
    a = p.parse_args()
    for s in (["surge", "complaints"] if a.scenario == "all" else [a.scenario]):
        print(f"\n=== scenario: {s} ===")
        run(s)


if __name__ == "__main__":
    sys.exit(main())
