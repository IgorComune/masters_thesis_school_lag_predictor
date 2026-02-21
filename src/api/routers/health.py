from fastapi import APIRouter
from src.monitoring.model_health import health_report, sanitize_for_json

router = APIRouter()

@router.get("/model_health")
def get_model_health():
    report = health_report()
    return sanitize_for_json(report)