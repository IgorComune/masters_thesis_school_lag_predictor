from fastapi import FastAPI
import joblib

from src.api.routers import inference

app = FastAPI(title="XGBoost Inference API")

# 🔥 Carrega modelo uma vez só
model = joblib.load("src/models/model.pkl")

# injeta modelo no router
inference.router.model = model

# registra router
app.include_router(inference.router)
