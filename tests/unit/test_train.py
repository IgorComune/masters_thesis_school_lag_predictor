import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import src.train_test.train as train


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

    config = train.Config(data_path=file)
    result = train.load_data(config)

    assert not result.empty


def test_load_data_missing_column(tmp_path):
    file = tmp_path / "data.csv"

    df = pd.DataFrame({
        "ipv": [1],
        "defasagem": [0]
    })
    df.to_csv(file, index=False)

    config = train.Config(data_path=file)

    with pytest.raises(ValueError):
        train.load_data(config)


# ==========================================================
# EVALUATE (corrigido)
# ==========================================================

def test_evaluate():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0, 1])
    mock_model.predict_proba.return_value = np.array([
        [0.8, 0.2],
        [0.1, 0.9]
    ])

    X = pd.DataFrame({"a": [1, 2]})
    y = pd.Series([0, 1])

    metrics = train.evaluate(mock_model, X, y)

    assert "recall" in metrics
    assert metrics["true_positives"] == 1


# ==========================================================
# SAVE ARTIFACTS (corrigido)
# ==========================================================

@patch("src.train_test.train.joblib.dump")
def test_save_artifacts(mock_dump, tmp_path):
    config = train.Config(data_path=tmp_path / "dummy.csv")
    config.models_dir = tmp_path

    dummy_model = object()

    train.save_artifacts(config, dummy_model, {"a": 1}, {"recall": 0.9})

    mock_dump.assert_called_once()

    assert (tmp_path / "params.json").exists()
    assert (tmp_path / "metrics.json").exists()


# ==========================================================
# OPTIMIZE (corrigido)
# ==========================================================

@patch("src.train_test.train.optuna.create_study")
def test_optimize_mocked(mock_create_study):

    mock_study = MagicMock()
    mock_study.best_params = {"n_estimators": 100}
    mock_create_study.return_value = mock_study

    config = train.Config(data_path=Path("dummy.csv"), n_trials=1)

    X = pd.DataFrame({
        "ipv": [1,2,3,4],
        "ips": [1,2,3,4],
        "iaa": [1,2,3,4],
        "ieg": [1,2,3,4],
        "no_av": [1,2,3,4],
        "ida": [1,2,3,4],
        "media": [1,2,3,4],
    })
    y = pd.Series([0,1,0,1])

    params = train.optimize(config, X, y)

    assert "n_estimators" in params
    assert "scale_pos_weight" in params
