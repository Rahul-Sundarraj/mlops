import json

import pytest

from src.web.app import create_app
from src.web.logic import classify, fmt_delay
from src.web.model import LiteModel

TH = {"rolling_error": {"window_days": 7, "min_samples": 3, "ratio": 1.5, "mae_threshold": 15.0},
      "complaints": {"window_days": 7, "default_threshold": 2, "per_train": {12675: 1}}}


def tiny_model_spec(value=10.0):
    """A one-tree model that always predicts `value` (leaf only)."""
    return {"version": "t1", "kind": "lightgbm", "cat_columns": ["train_no", "station", "category", "region"],
            "num_columns": ["distance_km", "stop_seq", "month", "day_of_week", "is_festival", "rain_mm",
                            "sched_hour", "hist_avg_delay", "route_len_km"],
            "categories": {"train_no": [12675], "station": ["SA"], "category": ["superfast"], "region": ["tn_west"]},
            "hist": {"12675|SA": 20.0}, "hist_default": 20.0, "base": 0.0, "strict": False, "trees": [[[value]]]}


@pytest.fixture
def client(tmp_path):
    model = tmp_path / "m.json"
    model.write_text(json.dumps(tiny_model_spec()))
    return create_app(str(model), str(tmp_path / "t.db"), TH, tmp_path / "ops").test_client()


def post(c, path, **body):
    return c.post(path, json=body)


def predict(c, train=12675, station="SA", day="2025-11-10", **extra):
    return post(c, "/predict", train_no=train, station=station, date=day, **extra)


def ping(c, day, delay, station="SA", train=12675):
    return post(c, "/location-ping", train_no=train, station=station, date=day, actual_delay_min=delay)


def complain(c, train, day="2025-11-10"):
    return post(c, "/complain", train_no=train, date=day, text="late")


def test_pages_and_health(client):
    assert client.get("/health").get_json()["status"] == "ok"
    html = client.get("/").get_data(as_text=True)
    assert "Will my train be late?" in html and "/predict-route" in html
    assert len(client.get("/api/trains").get_json()) == 16
    assert client.get("/ops").status_code == 200


def test_predict_and_validation(client):
    r = predict(client)
    assert r.status_code == 200 and r.get_json()["predicted_delay_min"] == 10.0
    assert r.get_json()["model_version"] == "t1"
    assert predict(client, train=99999).status_code == 422
    assert predict(client, station="XXX").status_code == 422
    assert post(client, "/predict", train_no="abc", station="SA", date="2025-11-10").status_code == 422
    assert post(client, "/predict", train_no=12675, station="SA", date="not-a-date").status_code == 422
    assert client.post("/predict", data="x").status_code == 422
    assert predict(client, rain_mm=-5).status_code == 422


def test_predict_route_returns_every_stop(client):
    r = post(client, "/predict-route", train_no=20643, date="2025-11-10", rain_mm=40)
    d = r.get_json()
    assert r.status_code == 200 and [s["station"] for s in d["stops"]] == ["MAS", "SA", "ED", "TUP", "CBE"]
    assert d["stops"][0]["delay_min"] == 0 and all(s["delay_min"] == 10.0 for s in d["stops"][1:])
    assert d["stops"][1]["scheduled"] and d["stops"][-1]["status"] == "Slightly late"
    assert post(client, "/predict-route", train_no=1, date="2025-11-10").status_code == 422


def test_ping_returns_error_vs_prediction(client):
    predict(client)
    assert ping(client, "2025-11-10", 40).get_json() == {"stored": True, "abs_error_min": 30.0}
    assert ping(client, "2025-11-10", 5, station="ED").get_json()["abs_error_min"] is None
    assert ping(client, "2025-11-10", 5, station="XXX").status_code == 422
    assert ping(client, "2025-11-10", -3).status_code == 422


def test_rolling_error_breach_needs_min_samples(client):
    for i in range(3):
        day = f"2025-11-0{i + 1}"
        predict(client, day=day)
        ping(client, day, 50)
        if i == 1:
            assert client.get("/monitor/error").get_json()["breach"] is False  # n=2 < min_samples
    out = client.get("/monitor/error").get_json()
    assert out["trains"]["12675"] == {"n": 3, "mae": 40.0, "threshold": 15.0, "breach": True} and out["breach"]


def test_complaint_spike_uses_route_threshold(client):
    complain(client, 12675)
    assert client.get("/monitor/complaints").get_json()["breach"] is False
    complain(client, 12675)  # 2 > per-train threshold of 1
    complain(client, 12676)
    out = client.get("/monitor/complaints").get_json()
    assert out["trains"]["12675"]["breach"] and not out["trains"]["12676"]["breach"]
    assert post(client, "/complain", train_no=12675, date="2025-11-10", text="  ").status_code == 422


def test_missing_model_is_a_clean_503(tmp_path):
    c = create_app(str(tmp_path / "nope.json"), str(tmp_path / "t.db"), TH, tmp_path).test_client()
    assert c.get("/health").get_json()["status"] == "no-model"
    assert predict(c).status_code == 503


def test_helpers():
    assert classify(3)[0] == "On time" and classify(40) == ("Heavily delayed", "red")
    assert fmt_delay(28.4) == "28 min" and fmt_delay(75) == "1 h 15 min"


def test_litemodel_tree_semantics():
    spec = tiny_model_spec()
    spec["trees"] = [[[0, 5.0, 1, 2], [1.0], [2.0]]]  # first feature is the encoded train_no: 0 <= 5 -> left
    m = LiteModel(spec)
    row = {"train_no": 12675, "station": "SA", "category": "superfast", "region": "tn_west", "distance_km": 1,
           "stop_seq": 1, "month": 1, "day_of_week": 1, "is_festival": 0, "rain_mm": 0, "sched_hour": 1,
           "hist_avg_delay": 1, "route_len_km": 1}
    assert m.predict(row) == 1.0
    strict = {**spec, "strict": True, "trees": [[[0, 0.0, 1, 2], [1.0], [2.0]]]}
    assert LiteModel(strict).predict(row) == 2.0  # 0 < 0 is false


def test_since_excludes_data_before_last_retrain(tmp_path):
    from src.monitoring import signals
    from src.web import storage

    conn = storage.connect(str(tmp_path / "s.db"))
    for d in ("2025-11-09", "2025-11-10", "2025-11-11"):
        conn.execute("INSERT INTO complaints(train_no, date, text, created_at) VALUES (12676, ?, 'x', 't')", (d,))
    conn.commit()
    cfg = TH["complaints"]
    assert signals.complaint_volume(conn, cfg, "2025-11-11")["trains"][12676]["n"] == 3
    assert signals.complaint_volume(conn, cfg, "2025-11-11", since="2025-11-10")["trains"][12676]["n"] == 1


def test_old_data_falls_out_of_window(client):
    for _ in range(3):
        complain(client, 12676, day="2025-10-01")
    assert client.get("/monitor/complaints?ref_date=2025-10-01").get_json()["breach"]
    assert client.get("/monitor/complaints?ref_date=2025-11-01").get_json()["trains"] == {}
