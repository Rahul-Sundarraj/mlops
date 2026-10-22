# Delay-Aware Railway Intelligence

An MLOps pipeline that predicts train delays for 16 South Indian trains (Chennai - Coimbatore / Mettupalayam / Mangaluru / Alappuzha / Egmore corridors)
and retrains itself when drift, rolling prediction error or a complaint spike is detected.

The deployed piece is **one small Flask app** that serves the passenger page, the JSON API and an operator monitor page.
It needs only Flask, gunicorn and PyYAML (about 20 MB installed): the trained model is exported to a plain JSON file (`models/production.json`, ~0.5 MB)
and evaluated in pure Python, so no numpy, pandas, LightGBM or MLflow is shipped.

| Stage | Tool | Where |
|---|---|---|
| Data (simulated, calibrated) | pandas / numpy | `src/data/simulate.py`, `configs/routes.yaml` |
| Versioning + pipeline | DVC | `dvc.yaml`, `params.yaml` |
| Tracking + registry | MLflow | `src/models/train.py` |
| Model export (for serving) | JSON, pure Python | `src/models/export.py`, `src/web/model.py` |
| Web app: passenger page + API + monitor | Flask | `src/web/`, `wsgi.py` |
| Monitoring | Evidently + rolling signals | `src/monitoring/` |
| Retraining | orchestrator | `src/orchestrator/run.py`, thresholds in `configs/thresholds.yaml` |
| Packaging / CI | Docker Compose, GitHub Actions | `Dockerfile`, `Dockerfile.pipeline`, `docker-compose.yml`, `.github/workflows/ci.yml` |

## Run the web app (what you demo)

```bash
pip install -r requirements.txt           # just Flask, gunicorn, PyYAML
flask --app wsgi run --port 8000          # development server
# production: gunicorn --bind 0.0.0.0:8000 --workers 1 --threads 4 wsgi:app
```

Then open:

- `http://localhost:8000/` is the **passenger page**. Pick a train, date, station and weather, press **Check delay**, and see the expected delay, expected arrival
  and a per-stop forecast. Riders can report their current delay (feeds the rolling-error monitor) and file a complaint (feeds the complaint monitor).
- `http://localhost:8000/ops` is the **operator monitor**: production model version, retrain history, signal trends and complaints for each demo scenario
  (needs the demo output from `python -m src.demo.run_demo`).

The repo ships a trained `models/production.json`, so the app works straight after cloning. API summary:

| Endpoint | Purpose |
|---|---|
| `GET /health` | status and model version |
| `GET /api/trains` | trains and their stops |
| `POST /predict` | `{train_no, station, date, rain_mm?}` -> predicted delay |
| `POST /predict-route` | `{train_no, date, rain_mm?}` -> forecast for every stop |
| `POST /location-ping` | `{train_no, station, date, actual_delay_min}` -> stored, returns forecast error |
| `POST /complain` | `{train_no, date, text}` |
| `GET /monitor/error`, `GET /monitor/complaints` | retrain signals |
| `POST /reload-model` | re-read the model file after a promotion |

Environment variables: `MODEL_PATH` (default `models/production.json`), `RAILWAY_DB` (SQLite file, default `data/api.db`), `OPS_DIR` (default `data/demo`).

## Train, track and retrain (full MLOps stack)

```bash
pip install -r requirements-ml.txt
dvc repro                                  # simulate -> features -> train; logs to MLflow, registers the model, exports models/production.json
python -m src.demo.run_demo --scenario all # replay incidents through the app; writes data/demo/<scenario>/
mlflow ui --backend-store-uri sqlite:///mlflow.db
pytest -q
```

After `dvc repro`, commit the refreshed `models/production.json` if you want to deploy the new model.

## Deploy to Azure App Service (free F1 tier)

One app now replaces the old separate backend and frontend, so it uses a single slot of the free quota. Steps (Linux, Python 3.12):

```bash
az webapp up --name <app-name> --resource-group <rg> --runtime "PYTHON:3.12" --sku F1
az webapp config set --name <app-name> --resource-group <rg> \
  --startup-file "gunicorn --bind=0.0.0.0 --timeout 120 --workers 1 --threads 4 wsgi:app"
az webapp config appsettings set --name <app-name> --resource-group <rg> \
  --settings RAILWAY_DB=/home/api.db SCM_DO_BUILD_DURING_DEPLOYMENT=true
```

Notes:
- Deploy from the repo root so `models/production.json`, `configs/` and `wsgi.py` are included. Heavy folders (`mlruns/`, `mlflow.db`, `data/demo/`) are git-ignored.
- `RAILWAY_DB=/home/api.db` keeps pings and complaints on Azure's persistent `/home` volume. The free tier has small CPU and disk quotas, which is why the deployed app avoids the ML libraries.
- `/ops` will show "No demo data" on Azure unless you also upload a `data/demo/<scenario>/` folder; it is meant for the local demo.
- These Azure commands are untested here (no Azure access); adjust names and flags to your subscription.

## Docker

```bash
docker compose up --build                       # trains + exports the model, then web on :8000
docker compose --profile demo up demo           # incident replay; then open http://localhost:8000/ops
```

`Dockerfile` builds only the light web image; `Dockerfile.pipeline` builds the full ML image used for training, MLflow and the demo.

## Demo scenarios

`python -m src.demo.run_demo` replays 7 normal days then 14 incident days through the app (predict, GPS ping, complaints)
and runs the orchestrator daily. Each scenario gets its own MLflow registry and exported model under `data/demo/<scenario>/`.

- `surge`: delays roughly double and rain inputs shift (festival/monsoon-style surge). Rolling error fires first, then complaints (drift is covered by tests; it was starved of data in the replay, see `docs/final_report.md`).
- `complaints`: delays stay normal but one route gets a complaint flood. The complaint signal drives the retrains, and the promotion rule rejects candidates that are not clearly better.

Retraining runs on any signal; the candidate is promoted only if it beats production on a holdout of the newest labelled days by at least `min_gain_min`,
and on promotion the exported JSON model is rewritten.

## Caveats

Delay data is simulated (no public historical per-station delay archive exists). Route stops and distances come from third-party timetable sites, not NTES.
The Docker files, CI workflow and Azure steps were written but not run end to end on the author's machine.
See `docs/data_simulation.md` and `docs/final_report.md`.
