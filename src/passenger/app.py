"""Passenger view: ask how late a train is likely to be, report a delay, or file a complaint.

Run: streamlit run src/passenger/app.py        (needs the API: uvicorn src.api.app:get_app --factory)
Env: API_URL (default http://localhost:8000)
"""
import os
from datetime import date, timedelta

import httpx
import pandas as pd
import streamlit as st

from src.api.features import scheduled_arrival
from src.data.simulate import load_config
from src.passenger.logic import WEATHER, classify, fmt_delay, station_label

API_URL = os.environ.get("API_URL", "http://localhost:8000")
CFG = load_config()


def api(method, path, **kw):
    r = httpx.request(method, f"{API_URL}{path}", timeout=30, **kw)
    r.raise_for_status()
    return r.json()


def predict(train_no, station, day, rain):
    body = {"train_no": train_no, "station": station, "date": day.isoformat()}
    if rain is not None:
        body["rain_mm"] = rain
    return api("POST", "/predict", json=body)["predicted_delay_min"]


def main():
    st.set_page_config(page_title="Will my train be late?", page_icon="🚆", layout="centered")
    st.title("🚆 Will my train be late?")
    st.caption("Expected delay for 16 Southern Railway trains, from a model that retrains itself as conditions change.")

    trains = {no: f"{no} · {t['name']}" for no, t in CFG["trains"].items()}
    c1, c2 = st.columns([3, 2])
    train_no = c1.selectbox("Train", list(trains), format_func=trains.get)
    day = c2.date_input("Travel date", value=date.today(), min_value=date.today() - timedelta(days=30))
    stops = [c for c, _ in CFG["trains"][train_no]["stops"]]
    c3, c4 = st.columns([3, 2])
    station = c3.selectbox("Your station", stops[1:] or stops, index=len(stops[1:]) - 1 if len(stops) > 1 else 0,
                           format_func=station_label)
    weather = c4.selectbox("Weather", list(WEATHER), help="Forecast rain along the route for that day")

    if st.button("Check delay", type="primary", width="stretch"):
        st.session_state["checked"] = True  # keeps results visible when the report/complaint buttons rerun the page
    if not st.session_state.get("checked"):
        st.info("Pick your train and station, then press **Check delay**.")
        return

    rain = WEATHER[weather]
    try:
        rows = []
        for code in stops:
            sched = scheduled_arrival(CFG, train_no, code, day)
            delay = 0.0 if code == stops[0] else predict(train_no, code, day, rain)
            rows.append({"Station": station_label(code), "Scheduled": sched.strftime("%H:%M (%d %b)"),
                         "Expected delay": fmt_delay(delay),
                         "Expected arrival": (sched + timedelta(minutes=delay)).strftime("%H:%M (%d %b)"),
                         "_code": code, "_delay": delay})
    except httpx.HTTPError as e:
        st.error(f"Could not reach the prediction service at {API_URL}. Is the API running?  ({e})")
        return

    mine = next(r for r in rows if r["_code"] == station)
    label, colour = classify(mine["_delay"])
    st.divider()
    st.markdown(f"### {station_label(station)}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Expected delay", fmt_delay(mine["_delay"]))
    m2.metric("Scheduled", mine["Scheduled"].split(" ")[0])
    m3.metric("Expected arrival", mine["Expected arrival"].split(" ")[0])
    st.markdown(f":{colour}[**{label}**]")

    st.subheader("Along the route")
    table = pd.DataFrame(rows)
    st.dataframe(table.drop(columns=["_code", "_delay"]).style.apply(
        lambda r: ["font-weight: bold" if r["Station"] == station_label(station) else "" for _ in r], axis=1),
        hide_index=True, width="stretch")
    st.line_chart(table.set_index("Station")["_delay"].rename("Expected delay (min)"))

    with st.expander("I'm on this train: report my current delay"):
        actual = st.number_input("Minutes late right now", min_value=0, max_value=600, value=0)
        if st.button("Send my delay report"):
            res = api("POST", "/location-ping", json={"train_no": train_no, "station": station,
                                                      "date": day.isoformat(), "actual_delay_min": float(actual)})
            err = res["abs_error_min"]
            st.success("Thanks, your report helps keep the forecast accurate."
                       + ("" if err is None else f" (The forecast was off by {err:.0f} min.)"))

    with st.expander("File a complaint about this train"):
        text = st.text_area("What went wrong?")
        if st.button("Submit complaint") and text.strip():
            api("POST", "/complain", json={"train_no": train_no, "date": day.isoformat(), "text": text.strip()})
            st.success("Complaint recorded.")


main()
