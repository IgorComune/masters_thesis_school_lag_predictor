import pandas as pd
import json
from pathlib import Path

LOG_PATH = Path("monitoring/inference_logs.jsonl")

def load_predictions():
    records = []

    with open(LOG_PATH, "r") as f:
        for line in f:
            records.append(json.loads(line))

    df = pd.DataFrame(records)
    return df

def prediction_distribution():
    df = load_predictions()

    print("\nPrediction statistics:")
    print(df["prediction"].describe())

    print("\nClass 1 rate:")
    print((df["prediction"] > 0.5).mean())
