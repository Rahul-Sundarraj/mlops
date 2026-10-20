import pandas as pd

from src.data.simulate import load_config, simulate_delays, simulate_grievances

TRAINS = {20644, 12680, 22638, 12676, 16160, 22640, 12672, 12674, 12675, 22637,
          20643, 12679, 12671, 12673, 16159, 12685}


def _run(seed=7):
    cfg = load_config()
    return cfg, simulate_delays(cfg, "2024-01-01", "2024-12-31", seed)


def test_all_requested_trains_present():
    cfg, df = _run()
    assert set(cfg["trains"]) == TRAINS
    assert set(df.train_no) == TRAINS


def test_reproducible_and_nonnegative():
    _, a = _run(1)
    _, b = _run(1)
    pd.testing.assert_frame_equal(a, b)
    assert (a.delay_min >= 0).all()


def test_final_delay_means_ordered_by_category():
    _, df = _run()
    final = df[df.groupby(["date", "train_no"]).stop_seq.transform("max") == df.stop_seq]
    m = final.groupby("category").delay_min.mean()
    assert m["vande_bharat"] < m["superfast"] < m["express"] + 15
    assert 2 < m["vande_bharat"] < 15


def test_monsoon_raises_delay():
    _, df = _run()
    kerala = df[df.region == "kerala"]
    jul = kerala[kerala.month == 7].delay_min.mean()
    feb = kerala[kerala.month == 2].delay_min.mean()
    assert jul > feb


def test_grievances_follow_delay():
    _, df = _run()
    g = simulate_grievances(df, 7)
    late = g[g.final_delay_min > 60].grievance_count.mean()
    ontime = g[g.final_delay_min < 15].grievance_count.mean()
    assert late > ontime
