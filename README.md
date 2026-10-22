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

The app is a single Flask process with a 22 MB dependency set, so it fits the F1 plan as **one** web app. It replaces the separate backend and frontend you
deployed before, so use the one app for both (see "Free-tier quota" below if you hit the quota error).

### Before you start
- An Azure subscription and the [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) (`az login` works).
- The repo checked out with `models/production.json` committed (it is, at the time of writing). If you retrained, run `dvc repro` and commit the new file first.
- Pick a globally unique app name; it becomes `https://<app-name>.azurewebsites.net`.

### Step by step (Azure CLI)
Run from the repo root. Commands are single-line so they work in Bash, PowerShell and cmd; replace the `<...>` values.

```bash
# 1. sign in and create a resource group + a free Linux plan
az login
az group create --name rg-railway --location centralindia
az appservice plan create --name plan-railway --resource-group rg-railway --sku F1 --is-linux

# 2. create the web app (Python 3.12 runtime)
az webapp create --name <app-name> --resource-group rg-railway --plan plan-railway --runtime "PYTHON:3.12"

# 3. settings: build dependencies on deploy, keep the SQLite file on the persistent /home volume
az webapp config appsettings set --name <app-name> --resource-group rg-railway --settings SCM_DO_BUILD_DURING_DEPLOYMENT=true RAILWAY_DB=/home/api.db

# 4. tell Azure how to start the app
az webapp config set --name <app-name> --resource-group rg-railway --startup-file "gunicorn --bind=0.0.0.0 --timeout 120 --workers 1 --threads 4 wsgi:app"

# 5. package the committed files only (no data/, mlruns/ or local databases) and deploy
git archive --format=zip -o deploy.zip HEAD
az webapp deploy --name <app-name> --resource-group rg-railway --src-path deploy.zip --type zip
```

Then open `https://<app-name>.azurewebsites.net/` (passenger page) and `https://<app-name>.azurewebsites.net/health`.
A healthy response looks like `{"status": "ok", ...}`. The first request after a deploy or idle period can take 20-60 seconds on F1 while the app starts.

To redeploy after changes: commit, then repeat step 5 (`git archive` only packs what is committed).

### Deploying through the Azure portal instead
Create a Web App (Publish: Code, Runtime stack: Python 3.12, OS: Linux, Pricing plan: Free F1). Under **Settings > Environment variables** add `SCM_DO_BUILD_DURING_DEPLOYMENT=true`
and `RAILWAY_DB=/home/api.db`. Under **Settings > Configuration > Stack settings** set the startup command to
`gunicorn --bind=0.0.0.0 --timeout 120 --workers 1 --threads 4 wsgi:app`. Then deploy with **Deployment Center** (GitHub) or by running step 5 above.

### Free-tier quota ("quota exceeded")
- The F1 plan gives a small daily CPU allowance (about 60 CPU-minutes per day) that is **shared by every app on the plan**, plus 1 GB of RAM and disk. A separate frontend app consumes that same allowance.
- Run only this one app on the plan: delete the old backend/frontend apps (`az webapp delete --name <old-app> --resource-group <rg>`) or deploy into the existing backend app and change its startup command (step 4).
- Keep `--workers 1`. Do not enable Always On (not available on F1); the app sleeps when idle and wakes on the next request.
- When the allowance is used up, Azure stops the app until the quota resets, so avoid load tests against the free plan.

### Troubleshooting
| Symptom | Fix |
|---|---|
| "Application Error" / site does not start | `az webapp log tail --name <app-name> --resource-group rg-railway`; check the startup command (step 4) is exactly as shown |
| `/health` shows `"status": "no-model"` or `/predict` returns 503 | `models/production.json` was not in the deployed zip; commit it and redeploy |
| Pings or complaints vanish after a restart | `RAILWAY_DB` is not set to `/home/api.db` (step 3) |
| Deploy is very large or slow | you zipped the working folder; use `git archive` as in step 5 |
| `/ops` says "No demo data" | expected on Azure; it reads `data/demo/` from the local demo run |

These steps follow Azure's documented CLI workflow but have not been run end to end on a real subscription by the author, so adjust names, region and flags to your account.

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
