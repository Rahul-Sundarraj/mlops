"""Simulate per-station train delays, weather proxy and grievances.

Usage:
    python -m src.data.simulate --start 2024-01-01 --end 2025-12-31 --seed 42
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.reference import RAIN_MEAN, festival_dates, load_config  # noqa: F401  (re-exported)


def train_stops(cfg, train):
    return train["stops"]


def simulate_delays(cfg, start, end, seed):
    rng = np.random.default_rng(seed)
    fest = festival_dates()
    regions = {st: r for r, sts in cfg["regions"].items() for st in sts}
    days = pd.date_range(start, end, freq="D")

    # Rain is a weather proxy shared by all trains on a given day and region.
    rain = {}
    for r, means in RAIN_MEAN.items():
        m = np.array([means[d.month - 1] for d in days])
        rain[r] = dict(zip(days, rng.gamma(0.5, m / 0.5 + 1e-9)))

    rows = []
    for no, t in cfg["trains"].items():
        stops = train_stops(cfg, t)
        n = len(stops) - 1
        cal = cfg["calibration"][t["category"]]
        rec = cal["recovery"] / max(n, 1) * 3  # per-segment recovery fraction
        rec = min(rec, 0.9)
        # per-segment mean shock so steady-state final delay ~ calibrated mean
        mu = 0.78 * cal["mean_final_delay"] * rec / (1 - (1 - rec) ** n)
        hh, mm = map(int, t["dep"].split(":"))
        for day in days:
            is_fest = day.date() in fest
            dow = day.dayofweek
            base_mult = (1.35 if is_fest else 1.0) * (1.1 if dow in (4, 6) else 1.0)
            dep = datetime.combine(day.date(), datetime.min.time()) + timedelta(hours=hh, minutes=mm)
            delay = max(0.0, rng.normal(2, 3))
            incident_seg = rng.integers(0, n) if rng.random() < 0.03 else -1
            rows.append(_row(no, t, stops[0], 0, dep, day, rain, regions, is_fest, delay))
            for i in range(1, n + 1):
                code, km = stops[i]
                reg = regions[code]
                r_mm = rain[reg][day]
                mult = base_mult * (1 + 0.025 * min(r_mm, 80))
                shock = rng.normal(mu * mult, cal["noise_sd"] / np.sqrt(n) * 1.5)
                if i - 1 == incident_seg:
                    shock += rng.uniform(30, 120)
                delay = max(0.0, delay * (1 - rec) + shock)
                sched = dep + timedelta(minutes=km / t["speed"] * 60)
                rows.append(_row(no, t, stops[i], i, sched, day, rain, regions, is_fest, delay))
    return pd.DataFrame(rows)


def _row(no, t, stop, seq, sched, day, rain, regions, is_fest, delay):
    code, km = stop
    return {
        "date": day.date().isoformat(), "train_no": no, "train_name": t["name"],
        "category": t["category"], "station": code, "region": regions[code],
        "stop_seq": seq, "distance_km": km, "scheduled_arrival": sched.isoformat(timespec="minutes"),
        "day_of_week": day.dayofweek, "month": day.month, "is_festival": int(is_fest),
        "rain_mm": round(float(rain[regions[code]][day]), 1), "delay_min": round(float(delay), 1),
    }


def simulate_grievances(delays, seed):
    """Grievance count per train-day: weak baseline + growth once final delay passes ~30 min."""
    rng = np.random.default_rng(seed + 1)
    final = delays.loc[delays.groupby(["date", "train_no"])["stop_seq"].idxmax(),
                       ["date", "train_no", "delay_min"]].rename(columns={"delay_min": "final_delay_min"})
    lam = 0.1 + 0.04 * np.clip(final["final_delay_min"] - 30, 0, None)
    final["grievance_count"] = rng.poisson(lam)
    return final.reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/simulated")
    a = p.parse_args()
    cfg = load_config()
    delays = simulate_delays(cfg, a.start, a.end, a.seed)
    griev = simulate_grievances(delays, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    delays.to_csv(out / "delays.csv", index=False)
    griev.to_csv(out / "grievances.csv", index=False)
    print(f"delays: {len(delays):,} rows, grievances: {len(griev):,} rows -> {out}")


if __name__ == "__main__":
    main()
