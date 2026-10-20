"""Retraining orchestrator: any of drift / rolling error / complaint spike triggers a retrain.

The candidate is registered as alias `staging`, compared with the current `production` model on a
holdout of the newest labelled days, and promoted only if it is not worse. Every decision is appended
to data/retrain_history.jsonl.
"""
import json
from datetime import date, timedelta
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error

from src.features.build import CAT, NUM, TARGET
from src.models.train import fit_candidates
from src.monitoring import drift, signals

HISTORY = Path("data/retrain_history.jsonl")


def read_history(path=HISTORY):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in open(path) if line.strip()]


def evaluate_signals(conn, thresholds, reference, live_features, ref_date=None):
    """Collect the three signals and the list of those that fired."""
    d = drift.detect_drift(reference, live_features, thresholds["drift"])
    e = signals.rolling_error(conn, thresholds["rolling_error"], ref_date)
    c = signals.complaint_volume(conn, thresholds["complaints"], ref_date)
    fired = [n for n, s in (("drift", d), ("rolling_error", e), ("complaints", c)) if s["breach"]]
    return {"drift": d, "rolling_error": e, "complaints": c, "triggers": fired}


def cooldown_ok(ref_date, cooldown_days, history):
    """True if no retrain happened within `cooldown_days` (logical data days) before ref_date."""
    if not history:
        return True
    last = date.fromisoformat(history[-1]["ref_date"])
    return (date.fromisoformat(ref_date) - last).days >= cooldown_days


def promote_decision(candidate_mae, champion_mae, tolerance):
    return candidate_mae <= champion_mae + tolerance


def retrain_and_promote(train_df, new_df, params, orch_cfg, reasons, ref_date, history_path=HISTORY):
    """Retrain on train_df + older new_df rows, compare with production on the newest holdout days."""
    cut = (date.fromisoformat(ref_date) - timedelta(days=orch_cfg["holdout_days"] - 1)).isoformat()
    holdout = new_df[new_df["date"] >= cut]
    fit_df = pd.concat([train_df, new_df[new_df["date"] < cut]], ignore_index=True)
    if holdout.empty:
        raise ValueError("no labelled holdout rows in the holdout window")

    mlflow.set_tracking_uri(params["mlflow_uri"])
    mlflow.set_experiment(params["experiment"])
    client, name = MlflowClient(), params["registered_model"]
    champ = mlflow.sklearn.load_model(f"models:/{name}@production")
    champ_mae = mean_absolute_error(holdout[TARGET], champ.predict(holdout[CAT + NUM]))

    with mlflow.start_run(run_name=f"retrain-{ref_date}") as run:
        best, pipe, cand_mae = fit_candidates(fit_df, holdout, params)
        mlflow.set_tags({"trigger": ",".join(reasons), "ref_date": ref_date, "candidate_model": best})
        mlflow.log_metrics({"holdout_mae_candidate": cand_mae, "holdout_mae_champion": champ_mae})
        mlflow.log_param("fit_rows", len(fit_df))
        info = mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")
    mv = mlflow.register_model(info.model_uri, name)
    client.set_registered_model_alias(name, "staging", mv.version)

    promoted = promote_decision(cand_mae, champ_mae, orch_cfg["promote_tolerance_min"])
    if promoted:
        client.set_registered_model_alias(name, "production", mv.version)
    client.set_model_version_tag(name, mv.version, "promoted", str(promoted).lower())

    record = {"ref_date": ref_date, "reasons": reasons, "candidate_model": best,
              "candidate_version": int(mv.version), "candidate_holdout_mae": round(cand_mae, 3),
              "champion_holdout_mae": round(champ_mae, 3), "promoted": promoted, "run_id": run.info.run_id}
    Path(history_path).parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def run_cycle(conn, params, thresholds, train_df, reference, live_features, new_df, ref_date,
              history_path=HISTORY):
    """One orchestrator tick: evaluate signals, respect cooldown, retrain if triggered."""
    sig = evaluate_signals(conn, thresholds, reference, live_features, ref_date)
    history = read_history(history_path)
    out = {"signals": sig, "retrained": False, "record": None}
    if not sig["triggers"]:
        out["skipped"] = "no trigger"
    elif not cooldown_ok(ref_date, thresholds["orchestrator"]["cooldown_days"], history):
        out["skipped"] = "cooldown"
    else:
        out["record"] = retrain_and_promote(train_df, new_df, params, thresholds["orchestrator"],
                                            sig["triggers"], ref_date, history_path)
        out["retrained"] = True
    return out


def load_params():
    return yaml.safe_load(open("params.yaml"))["train"]
