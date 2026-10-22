# Final report: Delay-Aware Railway Intelligence

## 1. What was built
An end-to-end, locally runnable MLOps pipeline for 16 South Indian trains (see `configs/routes.yaml`):
simulated delay data -> DVC pipeline -> MLflow tracking/registry -> a single Flask app (passenger page, JSON API with GPS-ping and complaint feedback, monitor page) serving a model exported to plain JSON ->
rolling-error, complaint and Evidently drift signals -> OR-trigger retraining with gated registry promotion -> operator monitor page,
packaged with Docker Compose and GitHub Actions. The serving app is deliberately light (Flask + PyYAML only) to fit Azure's free tier. Phase-by-phase commits are in the git log.

## 2. Data and its limits
No public per-station historical delay archive exists, so delays are simulated (`src/data/simulate.py`): delays accumulate along a route with partial
recovery, scaled by rain (region/month based), festival windows, Fri/Sun load and rare incidents. Final-stop mean delay is calibrated to roughly
8 / 25 / 32 minutes for Vande Bharat / superfast / express trains; this calibration is **not yet tied to published Rajya Sabha figures** and should be before
any claim about realism. Route stops, distances and timings were taken from third-party timetable sites (confirmtkt, railmitra, ixigo), not NTES; long trains list major stops only.
Because the simulator only uses a few drivers plus noise, results below show the mechanics of the pipeline, not real-world accuracy.

## 3. Model results (time-based split, test = Oct-Dec 2025)
| Model | Test MAE (min) |
|---|---|
| Historical (train, station) average | 8.813 |
| XGBoost | 8.677 |
| LightGBM (registered v1) | 8.652 |

The gain over the baseline is small (~1.8%) because noise dominates the simulated data.

## 4. Retraining demo (`python -m src.demo.run_demo --scenario all`)
Each scenario replays 7 normal days then 14 incident days (250 predictions/day plus pings and complaints) through the API, running the orchestrator daily.

**Surge** (delays about double, rain input +35 mm from day 8): no trigger during the 7 normal days; rolling error fired on the first incident day (2025-10-08) and
the candidate was promoted (holdout MAE 13.68 vs 13.75). Complaints triggered further retrains on 10-11 and 10-14 (promoted: 22.74 vs 24.54, then 17.12 vs 18.45),
then 10-18 (error) and 10-21 (complaints). 5 retrains, 5 promotions.

**Complaints only** (normal delays, +4 complaints/day on train 12675): the complaint signal fired on 2025-10-10 (day 3 of the flood). Retrains at 10-10 and 10-13 were promoted
by small margins (9.07 -> 8.99, 7.98 -> 7.91); at 10-16 and 10-19 the candidate was no better and was rejected by the promotion rule. 4 retrains, 2 promotions.

### Honest caveats about the demo
- **The drift signal did not fire in these final demo runs.** After every retrain the monitors only count data newer than the retrain, and the drift check needs at least
  1,500 live rows (about six days of demo traffic, because rain is a daily value shared by all trains). Retrains came more often than that, so drift stayed starved.
  Drift itself was verified separately: on the Oct 2025 data it stays quiet on a normal week, flags a +35 mm rain shift, and flags the Diwali window (`drift` tests in `tests/test_orchestrator.py`).
  The proposal's festival-triggered retrain therefore is shown by the unit test and calibration check, not by the live replay.
- Thresholds (`configs/thresholds.yaml`) were tuned by hand on this simulated data after earlier runs raised false alarms in the normal week (per-train relative error baseline,
  same-season drift reference, larger minimum sample sizes, a minimum promotion gain). They are not validated on real data.
- Under a sustained surge the system keeps retraining every few days because the surge never goes away; some candidates beat production by very little.
- The hist-average feature is frozen at the original training window.
- Docker files and CI were written but the image build and the GitHub Actions run were not executed here (Docker engine was not running; nothing pushed).

## 5. Reproducing
See `README.md`.
