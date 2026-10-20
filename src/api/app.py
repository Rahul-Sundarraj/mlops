"""FastAPI service: /predict, /location-ping, /complain plus monitor signals.

Run: uvicorn src.api.app:get_app --factory --port 8000  (needs a registered model; see src/models/train.py)
"""
import os
from datetime import date as Date, datetime, timezone

import mlflow
import mlflow.sklearn
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api import storage
from src.api.features import UnknownRoute, build_row, load_hist
from src.data.simulate import load_config
from src.monitoring import signals


class PredictIn(BaseModel):
    train_no: int
    station: str
    date: Date
    rain_mm: float | None = Field(default=None, ge=0)


class PingIn(BaseModel):
    train_no: int
    station: str
    date: Date
    actual_delay_min: float = Field(ge=0)


class ComplaintIn(BaseModel):
    train_no: int
    date: Date
    text: str = Field(min_length=1, max_length=1000)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_registered(params):
    mlflow.set_tracking_uri(params["mlflow_uri"])
    name = params["registered_model"]
    version = mlflow.MlflowClient().get_model_version_by_alias(name, "production").version
    return mlflow.sklearn.load_model(f"models:/{name}@production"), str(version)


def create_app(predictor=None, model_version="test", db_path=None, hist=None, thresholds=None):
    """`predictor` needs .predict(DataFrame); when None the production model is loaded lazily."""
    cfg = load_config()
    state = {"predictor": predictor, "version": model_version}
    conn = storage.connect(db_path or os.environ.get("RAILWAY_DB", "data/api.db"))
    hist = hist if hist is not None else load_hist()
    th = thresholds or signals.load_thresholds()
    app = FastAPI(title="Delay-Aware Railway Intelligence")

    def model():
        if state["predictor"] is None:
            params = yaml.safe_load(open("params.yaml"))["train"]
            state["predictor"], state["version"] = _load_registered(params)
        return state["predictor"]

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": state["predictor"] is not None,
                "model_version": state["version"]}

    @app.post("/predict")
    def predict(body: PredictIn):
        try:
            X = build_row(cfg, hist, body.train_no, body.station, body.date, body.rain_mm)
        except UnknownRoute as e:
            raise HTTPException(422, str(e))
        pred = max(0.0, float(model().predict(X)[0]))
        conn.execute("INSERT INTO predictions(train_no, station, date, predicted_delay, created_at) "
                     "VALUES (?,?,?,?,?)",
                     (body.train_no, body.station, body.date.isoformat(), pred, _now()))
        conn.commit()
        return {"train_no": body.train_no, "station": body.station, "date": body.date,
                "predicted_delay_min": round(pred, 1), "model_version": state["version"]}

    @app.post("/location-ping")
    def location_ping(body: PingIn):
        t = cfg["trains"].get(body.train_no)
        if t is None or body.station not in {c for c, _ in t["stops"]}:
            raise HTTPException(422, "unknown train or station for this train")
        d = body.date.isoformat()
        conn.execute("INSERT INTO pings(train_no, station, date, actual_delay, created_at) VALUES (?,?,?,?,?)",
                     (body.train_no, body.station, d, body.actual_delay_min, _now()))
        conn.commit()
        r = conn.execute("SELECT predicted_delay FROM predictions WHERE train_no=? AND station=? AND date=? "
                         "ORDER BY id DESC LIMIT 1", (body.train_no, body.station, d)).fetchone()
        err = None if r is None else round(abs(body.actual_delay_min - r["predicted_delay"]), 1)
        return {"stored": True, "abs_error_min": err}

    @app.post("/complain")
    def complain(body: ComplaintIn):
        if body.train_no not in cfg["trains"]:
            raise HTTPException(422, f"unknown train {body.train_no}")
        conn.execute("INSERT INTO complaints(train_no, date, text, created_at) VALUES (?,?,?,?)",
                     (body.train_no, body.date.isoformat(), body.text, _now()))
        conn.commit()
        return {"stored": True}

    @app.get("/monitor/error")
    def monitor_error(ref_date: str | None = None):
        return signals.rolling_error(conn, th["rolling_error"], ref_date)

    @app.get("/monitor/complaints")
    def monitor_complaints(ref_date: str | None = None):
        return signals.complaint_volume(conn, th["complaints"], ref_date)

    return app


def get_app():
    return create_app()
