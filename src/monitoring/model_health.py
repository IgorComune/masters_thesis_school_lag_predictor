import pandas as pd
import json

LOG_PATH = "src/monitoring/inference_logs.jsonl"

def health_report():
    records = []

    with open(LOG_PATH, "r") as f:
        for line in f:
            records.append(json.loads(line))

    df = pd.DataFrame(records)

    print("Total requests:", len(df))
    print("Average latency:", df["latency"].mean())
    print("Max latency:", df["latency"].max())
