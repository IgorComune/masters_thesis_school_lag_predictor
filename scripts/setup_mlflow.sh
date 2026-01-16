#!/bin/bash

echo "🔧 Iniciando MLflow..."
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000 &

echo "🚀 MLflow UI disponível em http://localhost:5000"
