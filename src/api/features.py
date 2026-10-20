"""Build a single-row feature frame for /predict from the route config."""
from datetime import date as Date, datetime, timedelta

import pandas as pd

from src.data.simulate import RAIN_MEAN, festival_dates
from src.features.build import CAT, NUM

_FEST = festival_dates()


class UnknownRoute(ValueError):
    pass


def build_row(cfg, hist, train_no, station, day, rain_mm=None):
    t = cfg["trains"].get(train_no)
    if t is None:
        raise UnknownRoute(f"unknown train {train_no}")
    stops = {c: (i, km) for i, (c, km) in enumerate(t["stops"])}
    if station not in stops:
        raise UnknownRoute(f"train {train_no} does not stop at {station}")
    seq, km = stops[station]
    region = next(r for r, sts in cfg["regions"].items() if station in sts)
    hh, mm = map(int, t["dep"].split(":"))
    sched = datetime.combine(day, datetime.min.time()) + timedelta(hours=hh, minutes=mm + km / t["speed"] * 60)
    if rain_mm is None:
        rain_mm = RAIN_MEAN[region][day.month - 1]
    h = hist.get((train_no, station))
    if h is None:
        h = sum(hist.values()) / len(hist)
    row = {
        "distance_km": km, "stop_seq": seq, "month": day.month, "day_of_week": day.weekday(),
        "is_festival": int(day in _FEST), "rain_mm": rain_mm, "sched_hour": sched.hour,
        "hist_avg_delay": h, "route_len_km": t["stops"][-1][1],
        "train_no": train_no, "station": station, "category": t["category"], "region": region,
    }
    return pd.DataFrame([row])[CAT + NUM]


def load_hist(path="data/processed/hist_avg.csv"):
    df = pd.read_csv(path)
    return {(int(r.train_no), r.station): float(r.hist_avg_delay) for r in df.itertuples()}
