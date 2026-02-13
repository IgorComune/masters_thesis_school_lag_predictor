import json
from datetime import datetime
from pathlib import Path

LOG_PATH = Path("monitoring/inference_logs.jsonl")

def log_inference(input_data: dict, prediction: float):
    record = {
        "timestamp": datetime.now().isoformat(),
        "inputs": input_data,
        "prediction": prediction,
    }

    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
