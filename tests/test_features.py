
from src.data.simulate import load_config, simulate_delays
from src.features.build import TARGET, add_features, attach_hist, hist_avg_table


def test_hist_avg_uses_only_given_rows_and_fills_unseen():
    df = add_features(simulate_delays(load_config(), "2024-01-01", "2024-02-15", 3))
    tr, te = df[df.date < "2024-02-01"], df[df.date >= "2024-02-01"]
    table = hist_avg_table(tr)
    out = attach_hist(te, table)
    assert out["hist_avg_delay"].notna().all()
    assert len(out) == len(te)
    assert {"sched_hour", "route_len_km"} <= set(out.columns)
    assert out[TARGET].equals(te[TARGET].reset_index(drop=True))
