from fastapi import FastAPI
import joblib

from api.routers import inference

app = FastAPI(title="XGBoost Inference API")

# 🔥 Carrega modelo uma vez só
model = joblib.load("src/models/xgboost_20260212_221848.pkl")

# injeta modelo no router
inference.router.model = model

# registra router
app.include_router(inference.router)
