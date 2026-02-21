# src/monitoring/prediction_drift.py
import pandas as pd
from . import metrics
from pathlib import Path
import json

LOG_PATH = Path("src/monitoring/inference_logs.jsonl")

def load_predictions():
    records = []
    with open(LOG_PATH, "r") as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)

def update_prediction_metrics():
    df = load_predictions()
    if df.empty:
        return
    # exemplo: proporção de predictions > 0.5
    ratio = (df["prediction"] > 0.5).mean()
    # expor como gauge (re-uso DRIFT_KS por falta de outro gauge — adicione um novo gauge se quiser)
    metrics.DRIFT_KS.labels(feature="prediction_above_0_5_ratio").set(float(ratio))
