"""Dependency-light reference data (route config, festivals, rain climatology).

Shared by the simulator and the Flask web app, so it must only use the standard library and PyYAML.
"""
from datetime import date, timedelta
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "routes.yaml"

# Festival windows (approximate dates, inclusive) that raise passenger load / delays.
FESTIVALS = [
    ("2024-01-12", "2024-01-17"), ("2024-03-24", "2024-03-26"), ("2024-04-09", "2024-04-14"),
    ("2024-09-12", "2024-09-17"), ("2024-10-09", "2024-10-14"), ("2024-10-29", "2024-11-03"),
    ("2024-12-23", "2025-01-02"), ("2025-01-12", "2025-01-17"), ("2025-03-29", "2025-04-01"),
    ("2025-04-10", "2025-04-14"), ("2025-08-30", "2025-09-06"), ("2025-09-28", "2025-10-03"),
    ("2025-10-18", "2025-10-23"), ("2025-12-23", "2026-01-02"), ("2026-01-12", "2026-01-17"),
    # approximate 2026-27 windows added for live lookups (not used by the 2024-25 simulation)
    ("2026-11-06", "2026-11-10"), ("2026-12-23", "2027-01-02"), ("2027-01-12", "2027-01-17"),
]

# Mean daily rainfall (mm) by month for each region (0-indexed month -> Jan..Dec).
RAIN_MEAN = {
    "tn_north":        [1, 1, 1, 1, 2, 3, 4, 5, 5, 11, 14, 6],   # NE monsoon Oct-Dec
    "tn_west":         [0.5, 0.5, 1, 3, 3, 2, 2, 2, 3, 7, 6, 2],
    "kerala":          [0.5, 1, 2, 6, 9, 20, 18, 12, 8, 9, 5, 2],  # SW monsoon Jun-Aug
    "karnataka_coast": [0, 0, 0.5, 3, 8, 30, 35, 22, 8, 6, 2, 0.5],
}


def load_config(path=CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


def festival_dates():
    days = set()
    for s, e in FESTIVALS:
        d, end = date.fromisoformat(s), date.fromisoformat(e)
        while d <= end:
            days.add(d)
            d += timedelta(days=1)
    return days
