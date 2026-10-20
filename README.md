# Delay-Aware Railway Intelligence

An MLOps pipeline that predicts train delays for 16 South Indian trains (Chennai - Coimbatore / Mettupalayam / Mangaluru / Alappuzha / Egmore corridors)
and retrains itself when drift, rolling prediction error or a complaint spike is detected. See `PLAN.md` for the phase plan and `docs/` for details.

| Stage | Tool | Where |
|---|---|---|
| Data (simulated, calibrated) | pandas / numpy | `src/data/simulate.py`, `configs/routes.yaml` |
| Versioning + pipeline | DVC | `dvc.yaml`, `params.yaml` |
| Tracking + registry | MLflow | `src/models/train.py` |
| Serving + feedback | FastAPI | `src/api/app.py` (`/predict`, `/location-ping`, `/complain`) |
| Monitoring | Evidently + rolling signals | `src/monitoring/` |
| Retraining | orchestrator | `src/orchestrator/run.py`, thresholds in `configs/thresholds.yaml` |
| Dashboard | Streamlit | `src/dashboard/app.py` |
| Packaging / CI | Docker Compose, GitHub Actions | `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml` |

## Quick start (local)

```bash
pip install -r requirements.txt
dvc repro                                  # simulate -> features -> train (logs to MLflow, registers the model)
uvicorn src.api.app:get_app --factory      # API on :8000
python -m src.demo.run_demo --scenario all # replay incidents; writes data/demo/<scenario>/
streamlit run src/dashboard/app.py         # dashboard on :8501
mlflow ui --backend-store-uri sqlite:///mlflow.db
pytest -q
```

## Docker

```bash
docker compose up --build                          # pipeline, then api :8000 and mlflow :5000
docker compose --profile demo up demo dashboard    # incident replay + dashboard :8501
```

## Demo scenarios

`python -m src.demo.run_demo` replays 7 normal days then 14 incident days through the API (predict, GPS ping, complaints)
and runs the orchestrator daily. Each scenario gets its own MLflow registry under `data/demo/<scenario>/`.

- `surge`: delays roughly double and rain inputs shift (festival/monsoon-style surge). Rolling error, drift and complaints fire.
- `complaints`: delays stay normal but one route gets a complaint flood. Only the complaint signal fires; retrains are gated by the promotion rule.

Retraining runs on any signal, then the candidate is promoted only if it beats production on a holdout of the newest labelled days by at least `min_gain_min`.

## Caveats

Delay data is simulated (no public historical per-station delay archive exists). Route stops and distances come from third-party timetable sites, not NTES.
See `docs/data_simulation.md` and `docs/final_report.md`.
