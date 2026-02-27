# src/monitoring/logger.py
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from . import metrics
from ..api.core.config import LOG_PATH, MODEL_VERSION

# LOG_PATH comes from config; ensure parent exists
LOG_PATH = Path(LOG_PATH)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

def log_inference(input_data: dict, prediction: float, latency: float, true_label: Optional[int] = None):
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "inputs": input_data,
        "prediction": float(prediction),
        "latency": float(latency),
        "true_label": int(true_label) if true_label is not None else None,
    }

    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

    # Emit metrics
    metrics.observe_inference(MODEL_VERSION, latency, error=False)
