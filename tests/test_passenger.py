from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.passenger.logic import classify, fmt_delay, station_label

APP = str(Path(__file__).parents[1] / "src/passenger/app.py")


def test_helpers():
    assert classify(3)[0] == "On time" and classify(40) == ("Heavily delayed", "red")
    assert fmt_delay(28.4) == "28 min" and fmt_delay(75) == "1 h 15 min"
    assert station_label("CBE") == "Coimbatore (CBE)"


def test_app_reports_api_down_gracefully(monkeypatch):
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")  # nothing listens here
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception and at.button[0].label == "Check delay"
    at.button[0].click().run()
    assert not at.exception and "Could not reach" in at.error[0].value
