"""Train baseline + XGBoost + LightGBM, log to MLflow, register the best model.

Usage: python -m src.models.train
"""
import json
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.sklearn
import pandas as pd
import xgboost as xgb
import yaml
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.common import with_env_overrides
from src.features.build import CAT, FEATURES, NUM, TARGET


def make_pipeline(model):
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT)],
        remainder="passthrough")
    return Pipeline([("pre", pre), ("model", model)])


def fit_candidates(fit_df, eval_df, params):
    """Fit XGBoost and LightGBM on fit_df; return (name, pipeline, MAE) of the best on eval_df."""
    best = None
    for name, model in {
        "xgboost": xgb.XGBRegressor(random_state=42, objective="reg:absoluteerror", **params["xgboost"]),
        "lightgbm": lgb.LGBMRegressor(random_state=42, objective="mae", verbose=-1, **params["lightgbm"]),
    }.items():
        pipe = make_pipeline(model).fit(fit_df[CAT + NUM], fit_df[TARGET])
        mae = mean_absolute_error(eval_df[TARGET], pipe.predict(eval_df[CAT + NUM]))
        if best is None or mae < best[2]:
            best = (name, pipe, mae)
    return best


def main():
    p = with_env_overrides(yaml.safe_load(open("params.yaml"))["train"])
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    Xtr, ytr, Xte, yte = train[FEATURES], train[TARGET], test[FEATURES], test[TARGET]
    Xtr, Xte = Xtr[CAT + NUM], Xte[CAT + NUM]

    mlflow.set_tracking_uri(p["mlflow_uri"])
    mlflow.set_experiment(p["experiment"])
    results = {}

    # Baseline: historical (train, station) average delay.
    with mlflow.start_run(run_name="baseline-hist-avg"):
        mae = mean_absolute_error(yte, test["hist_avg_delay"])
        mlflow.log_param("model", "historical_average")
        mlflow.log_metric("test_mae", mae)
        results["baseline"] = {"mae": mae, "run_id": None}

    candidates = {
        "xgboost": xgb.XGBRegressor(random_state=42, objective="reg:absoluteerror", **p["xgboost"]),
        "lightgbm": lgb.LGBMRegressor(random_state=42, objective="mae", verbose=-1, **p["lightgbm"]),
    }
    for name, model in candidates.items():
        with mlflow.start_run(run_name=name) as run:
            pipe = make_pipeline(model).fit(Xtr, ytr)
            mae = mean_absolute_error(yte, pipe.predict(Xte))
            mlflow.log_param("model", name)
            mlflow.log_params({f"{name}.{k}": v for k, v in p[name].items()})
            mlflow.log_metric("test_mae", mae)
            mlflow.log_metric("train_mae", mean_absolute_error(ytr, pipe.predict(Xtr)))
            mlflow.log_param("features", ",".join(CAT + NUM))
            mlflow.sklearn.log_model(pipe, name="model", input_example=Xtr.head(3),
                                     serialization_format="cloudpickle")
            results[name] = {"mae": mae, "run_id": run.info.run_id, "pipe": pipe}

    best = min((k for k in results if k != "baseline"), key=lambda k: results[k]["mae"])
    mv = mlflow.register_model(f"runs:/{results[best]['run_id']}/model", p["registered_model"])
    MlflowClient().set_registered_model_alias(p["registered_model"], "production", mv.version)

    from src.models.export import export_model, hist_from_csv
    out = export_model(results[best]["pipe"], hist_from_csv(), mv.version, p["export_path"])
    print(f"exported {best} v{mv.version} -> {out}")

    summary = {"best_model": best, "registered_version": int(mv.version),
               **{f"{k}_mae": round(v["mae"], 3) for k, v in results.items()}}
    Path("metrics").mkdir(exist_ok=True)
    json.dump(summary, open("metrics/metrics.json", "w"), indent=2)
    print(summary)


if __name__ == "__main__":
    main()
