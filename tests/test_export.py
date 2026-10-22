"""The exported pure-Python model must reproduce the fitted sklearn pipeline."""
import numpy as np
import pytest

from src.data.simulate import load_config, simulate_delays
from src.features.build import CAT, NUM, TARGET, add_features, attach_hist, hist_avg_table
from src.models.export import export_model
from src.web.model import LiteModel

TINY = {"xgboost": {"n_estimators": 30, "max_depth": 4, "learning_rate": 0.2},
        "lightgbm": {"n_estimators": 30, "num_leaves": 15, "learning_rate": 0.2}}


@pytest.fixture(scope="module")
def fitted():
    df = add_features(simulate_delays(load_config(), "2025-01-01", "2025-02-28", 3))
    table = hist_avg_table(df)
    df = attach_hist(df, table)
    return df, {f"{r.train_no}|{r.station}": float(r.hist_avg_delay) for r in table.itertuples()}


@pytest.mark.parametrize("name", ["lightgbm", "xgboost"])
def test_export_matches_pipeline(tmp_path, fitted, name):
    df, hist = fitted
    import lightgbm as lgb
    import xgboost as xgb

    import src.models.train as train
    model = (lgb.LGBMRegressor(random_state=0, objective="mae", verbose=-1, **TINY["lightgbm"]) if name == "lightgbm"
             else xgb.XGBRegressor(random_state=0, objective="reg:absoluteerror", **TINY["xgboost"]))
    pipe = train.make_pipeline(model).fit(df[CAT + NUM], df[TARGET])
    path = export_model(pipe, hist, "7", tmp_path / "m.json")
    lite = LiteModel.load(path)
    sample = df.sample(300, random_state=1)
    ref = pipe.predict(sample[CAT + NUM])
    mine = np.array([lite.predict(r) for r in sample[CAT + NUM].to_dict("records")])
    assert lite.kind == name and lite.version == "7"
    assert np.abs(ref - mine).max() < 1e-3
