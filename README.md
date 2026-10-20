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
| Passenger app (end-user view) | Streamlit | `src/passenger/app.py` |
| Operator dashboard (model health) | Streamlit | `src/dashboard/app.py` |
| Packaging / CI | Docker Compose, GitHub Actions | `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml` |

## Quick start (local)

```bash
pip install -r requirements.txt
dvc repro                                  # simulate -> features -> train (logs to MLflow, registers the model)
uvicorn src.api.app:get_app --factory --port 8000          # prediction API (start this first)
streamlit run src/passenger/app.py --server.port 8502      # passenger app: http://localhost:8502
python -m src.demo.run_demo --scenario all                 # replay incidents; writes data/demo/<scenario>/
streamlit run src/dashboard/app.py --server.port 8501      # operator dashboard: http://localhost:8501
mlflow ui --backend-store-uri sqlite:///mlflow.db
pytest -q
```

## Passenger app (the end-user demo)

`src/passenger/app.py` is what a rider sees: pick a train, date, station and weather, press **Check delay**, and get the expected delay,
expected arrival and a per-stop forecast. Riders can also report their current delay (feeds the rolling-error monitor) and file a complaint
(feeds the complaint monitor). It talks to the API over HTTP, so **the API must be running**; set `API_URL` if it is not on `http://localhost:8000`.
The first lookup loads the model and takes a few seconds. `.streamlit/config.toml` hides Streamlit's Deploy button and developer menu.

Showcase flow: start the API and the passenger app, look up a train, switch Weather to "Heavy rain" to see the forecast change,
submit a delay report and a complaint. Then open the operator dashboard to show monitoring and the retraining history from the demo replay.

## Deploying

The passenger app is only a front end. Deploy the API (with a trained model in the MLflow registry) somewhere reachable, then run the passenger app with
`API_URL` pointing at it. The Docker stack below does both. Deploying the Streamlit page alone will show "Could not reach the prediction service".

## Docker

```bash
docker compose up --build                          # pipeline, then api :8000, passenger app :8502, mlflow :5000
docker compose --profile demo up demo dashboard    # incident replay + dashboard :8501
```

## Demo scenarios

`python -m src.demo.run_demo` replays 7 normal days then 14 incident days through the API (predict, GPS ping, complaints)
and runs the orchestrator daily. Each scenario gets its own MLflow registry under `data/demo/<scenario>/`.

- `surge`: delays roughly double and rain inputs shift (festival/monsoon-style surge). Rolling error fires first, then complaints (drift is covered by tests; it was starved of data in the replay, see `docs/final_report.md`).
- `complaints`: delays stay normal but one route gets a complaint flood. The complaint signal drives the retrains, and the promotion rule rejects candidates that are not clearly better.

Retraining runs on any signal, then the candidate is promoted only if it beats production on a holdout of the newest labelled days by at least `min_gain_min`.

## Caveats

Docker files and the CI workflow were written but not run end to end on the author's machine. Delay data is simulated (no public historical per-station delay archive exists). Route stops and distances come from third-party timetable sites, not NTES.
See `docs/data_simulation.md` and `docs/final_report.md`.
