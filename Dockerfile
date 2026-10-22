<<<<<<< HEAD
# Lightweight web image: Flask app + the exported model file only (no ML libraries).
=======

>>>>>>> 08e789a48bfcdecfdfbc027f7857197f9332682e
FROM python:3.12-slim

WORKDIR /app
<<<<<<< HEAD
=======

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

>>>>>>> 08e789a48bfcdecfdfbc027f7857197f9332682e
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
<<<<<<< HEAD
ENV PYTHONUNBUFFERED=1 RAILWAY_DB=/app/data/api.db
EXPOSE 8000
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8000} --workers 1 --threads 4 --timeout 120 wsgi:app"]
=======

ENV PYTHONUNBUFFERED=1 \
    MLFLOW_URI=sqlite:////app/db/mlflow.db \
    RAILWAY_DB=/app/data/api.db

# Generate data, build features, and register the trained model
RUN mkdir -p /app/db /app/data /app/mlruns \
    && python -m src.data.simulate \
    && python -m src.features.build \
    && test -f /app/data/processed/hist_avg.csv \
    && ls -l /app/data/processed/hist_avg.csv \
    && python -m src.models.train

EXPOSE 8000

CMD ["uvicorn", "src.api.app:get_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
>>>>>>> 08e789a48bfcdecfdfbc027f7857197f9332682e
