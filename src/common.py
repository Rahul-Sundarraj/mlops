"""Shared helpers."""
import os


def with_env_overrides(params):
    """Let MLFLOW_URI override params.yaml's mlflow_uri (used inside Docker / the demo)."""
    params = dict(params)
    params["mlflow_uri"] = os.environ.get("MLFLOW_URI", params["mlflow_uri"])
    return params
