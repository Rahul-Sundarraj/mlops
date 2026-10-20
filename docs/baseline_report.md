# Baseline evaluation (Phase 2)

Reproduce: `dvc repro` (simulate -> features -> train). Runs are logged in MLflow (`sqlite:///mlflow.db`, experiment `railway-delay`); view with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

Split: time-based. Train = 2024-01-01..2025-09-30 (129,078 rows); test = 2025-10-01..2025-12-31 (18,584 rows; NE monsoon + festivals).

| Model | Test MAE (min) |
|---|---|
| Baseline: historical (train, station) average | 8.813 |
| XGBoost (MAE objective) | 8.677 |
| LightGBM (MAE objective) | **8.652** (registered as `delay-predictor` v1, alias `production`) |

Reading: both gradient-boosting models beat the baseline, but only by ~1.8%. That is expected on simulated data: delays are generated from the
same few drivers (rain, festival, weekday, route) plus large random noise, so the noise floor dominates and the historical average is already close to it.
The model's value should show once drift is injected (Phase 4), when the baseline's static averages go stale. Real data would likely widen the gap.

Leakage controls: `hist_avg_delay` is computed from training rows only. Features exclude any delay at earlier stops of the same run.
