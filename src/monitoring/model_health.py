import pandas as pd
import json
import math
from pathlib import Path
import math


LOG_PATH = Path("src/monitoring/inference_logs.jsonl")

def health_report():
    records = []

    if LOG_PATH.exists():
        with open(LOG_PATH, "r") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    df = pd.DataFrame(records)

    total_requests = len(df)
    
    # substitui NaN e inf por 0 ou None
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].apply(lambda x: None if pd.isna(x) or math.isinf(x) else x)
    
    avg_latency = df["latency"].mean() if total_requests > 0 else 0.0
    max_latency = df["latency"].max() if total_requests > 0 else 0.0

    if math.isnan(avg_latency) or math.isinf(avg_latency):
        avg_latency = 0.0
    if math.isnan(max_latency) or math.isinf(max_latency):
        max_latency = 0.0

    last_5 = []
    if total_requests > 0:
        last_5 = df.tail(5).to_dict(orient="records")

    return {
        "total_requests": total_requests,
        "average_latency": avg_latency,
        "max_latency": max_latency,
        "last_5_requests": last_5
    }


def sanitize_for_json(obj):
    """Recursively replace NaN/inf in dict/list with None"""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj
