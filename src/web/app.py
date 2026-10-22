"""Flask app: passenger page, JSON API (/predict, /location-ping, /complain) and monitor signals.

One small process serves everything, using only Flask + PyYAML + the standard library.
Run locally:  flask --app wsgi run --port 8000     Production: gunicorn wsgi:app
Env: MODEL_PATH (models/production.json), RAILWAY_DB (data/api.db), OPS_DIR (data/demo)
"""
import json
import os
import sqlite3
from datetime import date as Date
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from src.data.reference import load_config
from src.monitoring import signals
from src.web import storage
from src.web.features import UnknownRoute, build_row, scheduled_arrival
from src.web.logic import STATION_NAMES, classify, fmt_delay
from src.web.model import LiteModel

ROOT = Path(__file__).resolve().parents[2]


class BadRequest(ValueError):
    pass


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _int(body, key):
    try:
        return int(body[key])
    except (KeyError, TypeError, ValueError):
        raise BadRequest(f"'{key}' must be an integer") from None


def _day(body):
    try:
        return Date.fromisoformat(str(body["date"]))
    except (KeyError, ValueError):
        raise BadRequest("'date' must be YYYY-MM-DD") from None


def _num(body, key, required=True, lo=0.0, hi=1000.0):
    if body.get(key) is None:
        if required:
            raise BadRequest(f"'{key}' is required")
        return None
    try:
        v = float(body[key])
    except (TypeError, ValueError):
        raise BadRequest(f"'{key}' must be a number") from None
    if not lo <= v <= hi:
        raise BadRequest(f"'{key}' must be between {lo} and {hi}")
    return v


def create_app(model_path=None, db_path=None, thresholds=None, ops_dir=None):
    app = Flask(__name__)
    cfg = load_config()
    state = {"model": None, "path": model_path or os.environ.get("MODEL_PATH", ROOT / "models" / "production.json")}
    db = str(db_path or os.environ.get("RAILWAY_DB", ROOT / "data" / "api.db"))
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    storage.connect(db).close()  # create tables once
    th = thresholds or signals.load_thresholds(str(ROOT / "configs" / "thresholds.yaml"))
    ops_root = Path(ops_dir or os.environ.get("OPS_DIR", ROOT / "data" / "demo"))

    def model():
        if state["model"] is None:
            state["model"] = LiteModel.load(state["path"])
        return state["model"]

    def conn():
        c = sqlite3.connect(db, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def store_prediction(c, train_no, station, day, pred, row):
        c.execute("INSERT INTO predictions(train_no, station, date, predicted_delay, features, created_at) "
                  "VALUES (?,?,?,?,?,?)", (train_no, station, day.isoformat(), pred, json.dumps(row), _now()))

    def predict_one(c, train_no, station, day, rain):
        row = build_row(cfg, model(), train_no, station, day, rain)
        pred = max(0.0, model().predict(row))
        store_prediction(c, train_no, station, day, pred, row)
        return pred

    @app.errorhandler(BadRequest)
    def bad_request(e):
        return jsonify(error=str(e)), 422

    @app.errorhandler(UnknownRoute)
    def unknown_route(e):
        return jsonify(error=str(e)), 422

    @app.errorhandler(FileNotFoundError)
    def no_model(e):
        return jsonify(error="model file not found; run the training pipeline and export step"), 503

    def body():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise BadRequest("JSON object body required")
        return data

    # ---------- pages ----------
    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        ok = Path(state["path"]).exists()
        return jsonify(status="ok" if ok else "no-model", model_loaded=state["model"] is not None,
                       model_version=state["model"].version if state["model"] else None)

    @app.get("/api/trains")
    def trains():
        out = []
        for no, t in cfg["trains"].items():
            stops = [{"code": c, "name": STATION_NAMES.get(c, c), "km": km} for c, km in t["stops"]]
            out.append({"train_no": no, "name": t["name"], "stops": stops})
        return jsonify(out)

    # ---------- prediction ----------
    @app.post("/predict")
    def predict():
        b = body()
        train_no, station, day = _int(b, "train_no"), str(b.get("station", "")), _day(b)
        rain = _num(b, "rain_mm", required=False)
        with conn() as c:
            pred = predict_one(c, train_no, station, day, rain)
        return jsonify(train_no=train_no, station=station, date=day.isoformat(),
                       predicted_delay_min=round(pred, 1), model_version=model().version)

    @app.post("/predict-route")
    def predict_route():
        """Forecast for every stop of a train on a date (one call for the passenger page)."""
        b = body()
        train_no, day = _int(b, "train_no"), _day(b)
        rain = _num(b, "rain_mm", required=False)
        if train_no not in cfg["trains"]:
            raise UnknownRoute(f"unknown train {train_no}")
        stops = []
        with conn() as c:
            for i, (code, km) in enumerate(cfg["trains"][train_no]["stops"]):
                sched = scheduled_arrival(cfg, train_no, code, day)
                delay = 0.0 if i == 0 else predict_one(c, train_no, code, day, rain)
                label, colour = classify(delay)
                stops.append({"station": code, "name": STATION_NAMES.get(code, code), "distance_km": km,
                              "scheduled": sched.strftime("%H:%M"), "scheduled_date": sched.strftime("%d %b"),
                              "expected": (sched + timedelta(minutes=delay)).strftime("%H:%M"),
                              "delay_min": round(delay, 1), "delay_text": fmt_delay(delay),
                              "status": label, "colour": colour})
        return jsonify(train_no=train_no, date=day.isoformat(), model_version=model().version, stops=stops)

    @app.post("/reload-model")
    def reload_model():
        """Re-read the model file (call after a promotion rewrites it)."""
        state["model"] = None
        return jsonify(model_version=model().version)

    # ---------- feedback ----------
    @app.post("/location-ping")
    def location_ping():
        b = body()
        train_no, station, day = _int(b, "train_no"), str(b.get("station", "")), _day(b)
        actual = _num(b, "actual_delay_min", hi=1440.0)
        t = cfg["trains"].get(train_no)
        if t is None or station not in {c for c, _ in t["stops"]}:
            raise UnknownRoute("unknown train or station for this train")
        with conn() as c:
            c.execute("INSERT INTO pings(train_no, station, date, actual_delay, created_at) VALUES (?,?,?,?,?)",
                      (train_no, station, day.isoformat(), actual, _now()))
            r = c.execute("SELECT predicted_delay FROM predictions WHERE train_no=? AND station=? AND date=? "
                          "ORDER BY id DESC LIMIT 1", (train_no, station, day.isoformat())).fetchone()
        err = None if r is None else round(abs(actual - r["predicted_delay"]), 1)
        return jsonify(stored=True, abs_error_min=err)

    @app.post("/complain")
    def complain():
        b = body()
        train_no, day = _int(b, "train_no"), _day(b)
        text = str(b.get("text", "")).strip()
        if not text or len(text) > 1000:
            raise BadRequest("'text' must be 1-1000 characters")
        if train_no not in cfg["trains"]:
            raise UnknownRoute(f"unknown train {train_no}")
        with conn() as c:
            c.execute("INSERT INTO complaints(train_no, date, text, created_at) VALUES (?,?,?,?)",
                      (train_no, day.isoformat(), text, _now()))
        return jsonify(stored=True)

    # ---------- monitor signals ----------
    @app.get("/monitor/error")
    def monitor_error():
        with conn() as c:
            return jsonify(signals.rolling_error(c, th["rolling_error"], request.args.get("ref_date")))

    @app.get("/monitor/complaints")
    def monitor_complaints():
        with conn() as c:
            return jsonify(signals.complaint_volume(c, th["complaints"], request.args.get("ref_date")))

    # ---------- operator page ----------
    @app.get("/ops")
    def ops():
        from src.web import ops as ops_mod
        scenarios = sorted(p.name for p in ops_root.iterdir() if p.is_dir()) if ops_root.exists() else []
        chosen = request.args.get("scenario") or (scenarios[0] if scenarios else None)
        if chosen not in scenarios:
            chosen = scenarios[0] if scenarios else None
        data = ops_mod.load_state(ops_root / chosen) if chosen else None
        return render_template("ops.html", scenarios=scenarios, chosen=chosen, data=data)

    return app
