import json
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.train_test.test import (
    load_model,
    load_data,
    evaluate,
    save_metrics,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


# ==========================================================
# LOAD MODEL
# ==========================================================

def test_load_model_success(tmp_path):
    models_dir = tmp_path
    model_path = models_dir / "model.pkl"
    model_path.write_bytes(b"fake")  # arquivo fake

    with patch("your_module.joblib.load") as mock_load:
        mock_load.return_value = "mock_model"
        model, timestamp = load_model(models_dir)

    assert model == "mock_model"
    assert isinstance(timestamp, str)


def test_load_model_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path)


# ==========================================================
# LOAD DATA
# ==========================================================

def test_load_data_success(tmp_path):
    file = tmp_path / "data.csv"

    df = pd.DataFrame({
        "ipv": [1, 2],
        "ips": [1, 2],
        "iaa": [1, 2],
        "ieg": [1, 2],
        "no_av": [1, 2],
        "ida": [1, 2],
        "media": [1, 2],
        "defasagem": [0, 1]
    })

    df.to_csv(file, index=False)

    result = load_data(file)
    assert not result.empty


def test_load_data_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "does_not_exist.csv")


def test_load_data_missing_column(tmp_path):
    file = tmp_path / "data.csv"

    df = pd.DataFrame({
        "ipv": [1],
        "defasagem": [0]
    })

    df.to_csv(file, index=False)

    with pytest.raises(ValueError):
        load_data(file)


# ==========================================================
# EVALUATE
# ==========================================================

def test_evaluate():
    mock_model = MagicMock()
    mock_model.predict.return_value = [0, 1]
    mock_model.predict_proba.return_value = [[0.8, 0.2], [0.1, 0.9]]

    X = pd.DataFrame({col: [1, 2] for col in FEATURE_COLUMNS})
    y = pd.Series([0, 1])

    metrics = evaluate(mock_model, X, y)

    assert "recall" in metrics
    assert metrics["true_positives"] == 1
    assert metrics["false_negatives"] == 0


# ==========================================================
# SAVE METRICS
# ==========================================================

def test_save_metrics(tmp_path):
    metrics = {"recall": 0.9}
    timestamp = "123456"

    save_metrics(tmp_path, metrics, timestamp)

    output_file = tmp_path / f"test_metrics_{timestamp}.json"
    assert output_file.exists()

    with open(output_file) as f:
        data = json.load(f)

    assert data["recall"] == 0.9
