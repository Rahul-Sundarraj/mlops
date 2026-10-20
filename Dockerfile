FROM python:3.12-slim
WORKDIR /app
# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1 MLFLOW_URI=sqlite:////app/db/mlflow.db RAILWAY_DB=/app/data/api.db
CMD ["uvicorn", "src.api.app:get_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
