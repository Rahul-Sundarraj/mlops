"""Export a fitted (OrdinalEncoder + LightGBM/XGBoost) pipeline to one dependency-free JSON file.

The JSON also carries the (train, station) historical-average table, so the web app needs no CSVs.
Usage: python -m src.models.export            (exports the registered `production` model to models/production.json)
"""
import json
from pathlib import Path

import yaml

from src.common import with_env_overrides
from src.features.build import CAT, NUM


def _flatten_lightgbm(tree):
    nodes = []

    def walk(n):
        idx = len(nodes)
        nodes.append(None)
        if "leaf_value" in n:
            nodes[idx] = [round(n["leaf_value"], 7)]
            return idx
        assert n["decision_type"] == "<=", n["decision_type"]
        left, right = walk(n["left_child"]), walk(n["right_child"])
        nodes[idx] = [n["split_feature"], n["threshold"], left, right]
        return idx

    walk(tree["tree_structure"])
    return nodes


def _flatten_xgboost(tree):
    nodes = []

    def walk(n):
        idx = len(nodes)
        nodes.append(None)
        if "leaf" in n:
            nodes[idx] = [round(n["leaf"], 7)]
            return idx
        kids = {c["nodeid"]: c for c in n["children"]}
        left, right = walk(kids[n["yes"]]), walk(kids[n["no"]])
        nodes[idx] = [int(n["split"][1:]), n["split_condition"], left, right]
        return idx

    walk(tree)
    return nodes


def export_model(pipe, hist, version, path):
    """hist: {"train_no|station": mean_delay}. Returns the written Path."""
    pre, model = pipe.named_steps["pre"], pipe.named_steps["model"]
    cats = pre.named_transformers_["cat"].categories_
    spec = {
        "version": str(version), "cat_columns": CAT, "num_columns": NUM,
        "categories": {c: [v.item() if hasattr(v, "item") else v for v in arr] for c, arr in zip(CAT, cats)},
        "hist": hist, "hist_default": sum(hist.values()) / len(hist),
    }
    name = type(model).__name__
    if name == "LGBMRegressor":
        dump = model.booster_.dump_model()
        spec.update(kind="lightgbm", strict=False, base=0.0,
                    trees=[_flatten_lightgbm(t) for t in dump["tree_info"]])
    elif name == "XGBRegressor":
        booster = model.get_booster()
        raw = json.loads(booster.save_config())["learner"]["learner_model_param"]["base_score"]
        base = float(str(raw).strip("[]").split(",")[0])  # xgboost 3.x stores it as "[1.39E1]"
        spec.update(kind="xgboost", strict=True, base=base,
                    trees=[_flatten_xgboost(json.loads(t)) for t in booster.get_dump(dump_format="json")])
    else:
        raise ValueError(f"unsupported model type {name}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, separators=(",", ":")), encoding="utf-8")
    return path


def hist_from_csv(path="data/processed/hist_avg.csv"):
    import pandas as pd
    df = pd.read_csv(path)
    return {f"{int(r.train_no)}|{r.station}": round(float(r.hist_avg_delay), 4) for r in df.itertuples()}


def main():
    import mlflow
    import mlflow.sklearn
    p = with_env_overrides(yaml.safe_load(open("params.yaml"))["train"])
    mlflow.set_tracking_uri(p["mlflow_uri"])
    name = p["registered_model"]
    version = mlflow.MlflowClient().get_model_version_by_alias(name, "production").version
    pipe = mlflow.sklearn.load_model(f"models:/{name}@production")
    out = export_model(pipe, hist_from_csv(), version, p.get("export_path", "models/production.json"))
    print(f"exported production v{version} -> {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
