import pandas as pd
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

LOG_PATH = "src/monitoring/inference_logs.jsonl"

def evaluate_performance():
    records = []

    with open(LOG_PATH, "r") as f:
        for line in f:
            records.append(json.loads(line))

    df = pd.DataFrame(records)

    df = df.dropna(subset=["true_label"])

    if len(df) == 0:
        print("No labeled data yet.")
        return

    y_true = df["true_label"]
    y_pred = (df["prediction"] > 0.5).astype(int)
    y_prob = df["prediction"]

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("ROC AUC:", roc_auc_score(y_true, y_prob))
