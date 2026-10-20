# Delay-Aware Railway Intelligence — Phase-by-Phase Plan

Based on `MLOPS_Project_Proposal.docx` (5-week timeline) and `Mlops_writeup.docx`.
Rule: **every phase ends with one or more local git commits** (no pushing). Commit messages use Conventional Commits (`feat:`, `chore:`, `docs:`, `ci:`, `test:`).

## Target repo layout

```
.
├── data/                  # DVC-tracked (raw/, simulated/, processed/)
├── src/
│   ├── data/              # metadata loader, delay + GPS simulator
│   ├── features/          # feature engineering
│   ├── models/            # baseline, xgboost/lightgbm training, evaluation
│   ├── api/               # FastAPI app (/predict, /location-ping, /complain)
│   ├── monitoring/        # rolling error, complaint aggregator, drift (Evidently)
│   ├── orchestrator/      # 3-signal retrain trigger + registry promotion
│   └── dashboard/         # Streamlit app
├── tests/
├── configs/               # thresholds, simulator params (YAML)
├── docker/ + docker-compose.yml
├── .github/workflows/ci.yml
├── dvc.yaml, params.yaml
└── docs/                  # report, methodology, demo script
```

---

## Phase 0 — Project bootstrap (before Week 1)
- Init git, `.gitignore`, `README.md`, `pyproject.toml`/`requirements.txt`, folder skeleton, `dvc init`, pre-commit (ruff).
- **Commits:**
  1. `chore: initialize repo, gitignore, and project skeleton`
  2. `chore: add dependencies, DVC init, and lint config`
- **Done when:** `pip install -r requirements.txt` works; `ruff check` passes.

## Phase 1 — Data foundation (Week 1)
- Pull station / route / zone metadata from `datameet/railways` (CC0); optional small live-API pull to align train numbers.
- Design the **delay simulator**: per-train/per-station delays calibrated to published aggregate stats (Rajya Sabha / Ministry of Railways), with configurable effects for fog (Dec–Jan, North), monsoon, festivals, day-of-week.
- GPS-ping simulator (actual delay at timestamps) and grievance generator (volume correlated with delay spikes, with noise).
- Document the data limitations (required by proposal §4).
- **Commits:**
  1. `feat(data): load station and route metadata from datameet`
  2. `feat(data): delay simulator with seasonal/festival effects`
  3. `feat(data): simulated GPS ping and grievance generators`
  4. `docs: data strategy and calibration notes`
- **Done when:** one command generates a reproducible dataset from a seed; unit tests for the simulator pass.

## Phase 2 — Modeling & experiment tracking (Week 2)
- Put datasets under DVC (`dvc.yaml` stages: ingest → simulate → features → train).
- Feature engineering: route, distance, zone, month/season, weather proxy, day-of-week, festival flag, historical route average delay.
- Baseline (historical route average) vs. XGBoost and LightGBM; metric = MAE (minutes).
- Log params/metrics/artifacts to MLflow; register best model.
- **Commits:**
  1. `feat(features): feature engineering pipeline`
  2. `feat(models): baseline and gradient-boosting training with MLflow`
  3. `build: DVC pipeline stages and params.yaml`
  4. `docs: baseline evaluation report`
- **Done when:** `dvc repro` runs end-to-end; ≥2 models + baseline compared in MLflow; GBM beats baseline MAE.

## Phase 3 — Serving & feedback loops (Week 3)
- FastAPI: `/predict`, `/location-ping`, `/complain`, `/health`; loads the Production model from the MLflow registry.
- Persist pings, predictions, and complaints (SQLite or Parquet).
- Rolling-error monitor (predicted vs. actual from pings, per-route sliding window).
- Complaint aggregator (per-route rolling volume vs. route-specific threshold).
- **Commits:**
  1. `feat(api): FastAPI app with /predict and /health`
  2. `feat(api): /location-ping and /complain endpoints with storage`
  3. `feat(monitoring): rolling prediction-error monitor`
  4. `feat(monitoring): complaint aggregator with route thresholds`
  5. `test: API and monitor tests`
- **Done when:** replaying simulated pings/complaints against the API produces correct rolling-error and complaint-spike signals.

## Phase 4 — Ops automation (Week 4)
- Evidently AI drift monitor: live feature window vs. training reference; drift flag + HTML report.
- Retraining orchestrator: **OR** of drift / rolling error / complaint spike (thresholds in `configs/`), cooldown to avoid retrain loops.
- Retrain → evaluate → promote in MLflow registry (Staging → Production) only if candidate MAE ≤ current; log trigger reason and history.
- **Commits:**
  1. `feat(monitoring): Evidently drift detection`
  2. `feat(orchestrator): three-signal retrain trigger with cooldown`
  3. `feat(orchestrator): conditional registry promotion and retrain history`
  4. `test: trigger and promotion tests`
- **Done when:** injecting each of the three signals separately triggers a retrain and (when better) a promotion.

## Phase 5 — Packaging, CI, dashboard, demo (Week 5)
- Dockerfiles + `docker-compose.yml` (API, MLflow server, monitoring/orchestrator job, dashboard).
- GitHub Actions: ruff + pytest + Docker build on push (workflow file committed; not pushed until you decide).
- Streamlit dashboard: current model version, drift report, complaints by route, retraining history, rolling MAE.
- Demo scenarios: fog-season surge, festival surge, complaint spike → visible retrain + promotion. Scripted via `make demo`.
- Final report (methodology, simulated-data justification, limitations, results).
- **Commits:**
  1. `build: Dockerfiles and docker-compose stack`
  2. `ci: GitHub Actions for lint, tests, and image build`
  3. `feat(dashboard): Streamlit monitoring dashboard`
  4. `feat: demo scenarios and make demo target`
  5. `docs: final report and README usage`
  6. `chore: tag v1.0`
- **Done when:** `docker compose up` + demo script shows a triggered retrain end-to-end on a clean machine.

---

## Risks & mitigations
| Risk | Mitigation |
|---|---|
| No public historical delay data | Calibrated simulation, stated openly in report |
| datameet data is ~10 yrs old | Use only stable attributes; optional live-API refresh |
| Retrain loops / flapping | Cooldown + promote-only-if-better |
| Windows + Docker friction | Keep all components runnable locally without Docker too |

## Open decisions (defaults assumed unless you say otherwise)
- LightGBM vs XGBoost as primary → train both, keep the better.
- Storage for pings/complaints → SQLite.
- Scope of routes → ~20–50 popular routes to keep data volume manageable.
