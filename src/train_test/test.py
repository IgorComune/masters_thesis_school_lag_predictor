"""
Test Pipeline for XGBoost Model

Features:
- Loads trained model from ./models
- Loads dataset
- Evaluates model
- Saves test metrics to ./models
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ==========================================================
# CONFIG
# ==========================================================

FEATURE_COLUMNS = [
    "ipv", "ips", "iaa", "ieg", "no_av", "ida", "media"
]

TARGET_COLUMN = "defasagem"


# ==========================================================
# LOADERS
# ==========================================================

def load_model(models_dir: Path):
    model_files = sorted(models_dir.glob("xgboost_*.pkl"))

    if not model_files:
        raise FileNotFoundError("No trained model found in ./models")

    latest_model = model_files[-1]
    logger.info(f"Loading model: {latest_model}")

    model = joblib.load(latest_model)
    return model, latest_model.stem.split("_")[-1]


def load_data(data_path: Path):
    if not data_path.exists():
        raise FileNotFoundError("Dataset not found")

    df = pd.read_csv(data_path)

    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df


# ==========================================================
# EVALUATION
# ==========================================================

def evaluate(model, X, y):

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    metrics = {
        "recall": recall_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba),
        "pr_auc": average_precision_score(y, y_proba),
    }

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics.update({
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    })

    return metrics


# ==========================================================
# SAVE
# ==========================================================

def save_metrics(models_dir: Path, metrics: dict, timestamp: str):

    output_path = models_dir / f"test_metrics_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Test metrics saved at: {output_path}")


# ==========================================================
# MAIN
# ==========================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/students_feature_engineering.csv"),
    )
    args = parser.parse_args()

    models_dir = Path("src/models")

    model, timestamp = load_model(models_dir)

    df = load_data(args.data_path)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    logger.info("Running evaluation...")
    metrics = evaluate(model, X, y)

    save_metrics(models_dir, metrics, timestamp)

    logger.info("Testing completed successfully.")


if __name__ == "__main__":
    main()
