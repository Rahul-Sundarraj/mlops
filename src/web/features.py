"""Dependency-free feature building for one (train, station, date); mirrors src/features/build.py."""
from datetime import datetime, timedelta

from src.data.reference import RAIN_MEAN, festival_dates

_FEST = festival_dates()


class UnknownRoute(ValueError):
    pass


def scheduled_arrival(cfg, train_no, station, day):
    """Scheduled arrival datetime from origin departure + distance / average speed."""
    t = cfg["trains"][train_no]
    km = dict(t["stops"])[station]
    hh, mm = map(int, t["dep"].split(":"))
    return datetime.combine(day, datetime.min.time()) + timedelta(hours=hh, minutes=mm + km / t["speed"] * 60)


def region_of(cfg, station):
    return next(r for r, sts in cfg["regions"].items() if station in sts)


def build_row(cfg, model, train_no, station, day, rain_mm=None):
    """Feature dict (all categorical and numeric columns) for the model."""
    t = cfg["trains"].get(train_no)
    if t is None:
        raise UnknownRoute(f"unknown train {train_no}")
    stops = {c: (i, km) for i, (c, km) in enumerate(t["stops"])}
    if station not in stops:
        raise UnknownRoute(f"train {train_no} does not stop at {station}")
    seq, km = stops[station]
    region = region_of(cfg, station)
    sched = scheduled_arrival(cfg, train_no, station, day)
    if rain_mm is None:
        rain_mm = RAIN_MEAN[region][day.month - 1]
    return {
        "distance_km": km, "stop_seq": seq, "month": day.month, "day_of_week": day.weekday(),
        "is_festival": int(day in _FEST), "rain_mm": rain_mm, "sched_hour": sched.hour,
        "hist_avg_delay": model.hist_avg(train_no, station), "route_len_km": t["stops"][-1][1],
        "train_no": train_no, "station": station, "category": t["category"], "region": region,
    }
