# src/api/core/config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = PROJECT_ROOT / "monitoring" / "inference_logs.jsonl"

API_HOST = "0.0.0.0"
API_PORT = 8000

MODEL_VERSION = "v1"  # atualize conforme sua versão ou carregue do modelo
