"""Build model features and a time-based train/test split.

Usage: python -m src.features.build
"""
import json
from pathlib import Path

import pandas as pd
import yaml

NUM = ["distance_km", "stop_seq", "month", "day_of_week", "is_festival", "rain_mm",
       "sched_hour", "hist_avg_delay", "route_len_km"]
CAT = ["train_no", "station", "category", "region"]
FEATURES = NUM + CAT
TARGET = "delay_min"


def add_features(df):
    df = df.copy()
    df["sched_hour"] = pd.to_datetime(df["scheduled_arrival"]).dt.hour
    df["route_len_km"] = df.groupby("train_no")["distance_km"].transform("max")
    return df


def hist_avg_table(train):
    """Mean delay per (train, station) from the training window only (no test leakage)."""
    g = train.groupby(["train_no", "station"])[TARGET].mean().rename("hist_avg_delay")
    return g.reset_index()


def attach_hist(df, table):
    out = df.merge(table, on=["train_no", "station"], how="left")
    out["hist_avg_delay"] = out["hist_avg_delay"].fillna(table["hist_avg_delay"].mean())
    return out


def main():
    p = yaml.safe_load(open("params.yaml"))
    df = add_features(pd.read_csv("data/simulated/delays.csv"))
    train = df[df["date"] < p["split"]["test_start"]]
    test = df[df["date"] >= p["split"]["test_start"]]
    table = hist_avg_table(train)
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    attach_hist(train, table).to_csv(out / "train.csv", index=False)
    attach_hist(test, table).to_csv(out / "test.csv", index=False)
    table.to_csv(out / "hist_avg.csv", index=False)
    print(json.dumps({"train_rows": len(train), "test_rows": len(test)}))


if __name__ == "__main__":
    main()
