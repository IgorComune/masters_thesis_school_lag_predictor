# src/api/main.py
from fastapi import FastAPI
import joblib
from pathlib import Path

# Routers
from src.api.routers import inference
from src.api.routers import health
from src.api.routers import metrics as metrics_router

# -----------------------------
# 🔥 App FastAPI
# -----------------------------
app = FastAPI(title="XGBoost Inference API")

# -----------------------------
# 🔥 Carrega modelo uma vez
# -----------------------------
MODEL_PATH = Path("src/models/model.pkl")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}. Treine e salve um modelo primeiro.")

model = joblib.load(MODEL_PATH)

# Injeta o modelo no router de inference
# (o router vai usar isso no predict)
inference.router.model = model

# -----------------------------
# 🔥 Registra routers
# -----------------------------
app.include_router(inference.router, prefix="/inference")
app.include_router(health.router, prefix="/health")
app.include_router(metrics_router.router, prefix="/metrics")
