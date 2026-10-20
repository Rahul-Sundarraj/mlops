import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app


class FakeModel:
    def __init__(self, value=10.0):
        self.value = value

    def predict(self, X):
        return [self.value] * len(X)


TH = {"rolling_error": {"window_days": 7, "min_samples": 3, "mae_threshold": 15.0},
      "complaints": {"window_days": 7, "default_threshold": 2, "per_train": {12675: 1}}}


@pytest.fixture
def client(tmp_path):
    app = create_app(FakeModel(), "t1", str(tmp_path / "t.db"), hist={(12675, "SA"): 20.0}, thresholds=TH)
    return TestClient(app)


def predict(c, train=12675, station="SA", day="2025-11-10"):
    return c.post("/predict", json={"train_no": train, "station": station, "date": day})


def ping(c, day, delay, station="SA", train=12675):
    return c.post("/location-ping", json={"train_no": train, "station": station, "date": day,
                                          "actual_delay_min": delay})


def complain(c, train, day="2025-11-10"):
    return c.post("/complain", json={"train_no": train, "date": day, "text": "late"})


def test_health_and_predict(client):
    assert client.get("/health").json()["status"] == "ok"
    r = predict(client)
    assert r.status_code == 200 and r.json()["predicted_delay_min"] == 10.0


def test_predict_rejects_unknown_route(client):
    assert predict(client, train=99999).status_code == 422
    assert predict(client, station="XXX").status_code == 422


def test_ping_returns_error_vs_prediction(client):
    predict(client)
    assert ping(client, "2025-11-10", 40).json() == {"stored": True, "abs_error_min": 30.0}
    assert ping(client, "2025-11-10", 5, station="ED").json()["abs_error_min"] is None
    assert ping(client, "2025-11-10", 5, station="XXX").status_code == 422


def test_rolling_error_breach_needs_min_samples(client):
    for i in range(3):
        day = f"2025-11-0{i + 1}"
        predict(client, day=day)
        ping(client, day, 50)
        if i == 1:
            assert client.get("/monitor/error").json()["breach"] is False  # n=2 < min_samples
    out = client.get("/monitor/error").json()
    assert out["trains"]["12675"] == {"n": 3, "mae": 40.0, "threshold": 15.0, "breach": True} and out["breach"]


def test_complaint_spike_uses_route_threshold(client):
    complain(client, 12675)
    assert client.get("/monitor/complaints").json()["breach"] is False
    complain(client, 12675)  # 2 > per-train threshold of 1
    complain(client, 12676)
    out = client.get("/monitor/complaints").json()
    assert out["trains"]["12675"]["breach"] and not out["trains"]["12676"]["breach"]


def test_old_data_falls_out_of_window(client):
    for _ in range(3):
        complain(client, 12676, day="2025-10-01")
    assert client.get("/monitor/complaints", params={"ref_date": "2025-10-01"}).json()["breach"]
    assert client.get("/monitor/complaints", params={"ref_date": "2025-11-01"}).json()["trains"] == {}


def test_since_excludes_data_before_last_retrain(tmp_path):
    from src.api import storage
    from src.monitoring import signals

    conn = storage.connect(str(tmp_path / "s.db"))
    for d in ("2025-11-09", "2025-11-10", "2025-11-11"):
        conn.execute("INSERT INTO complaints(train_no, date, text, created_at) VALUES (12676, ?, 'x', 't')", (d,))
    conn.commit()
    cfg = TH["complaints"]
    assert signals.complaint_volume(conn, cfg, "2025-11-11")["trains"][12676]["n"] == 3
    assert signals.complaint_volume(conn, cfg, "2025-11-11", since="2025-11-10")["trains"][12676]["n"] == 1
