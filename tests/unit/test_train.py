import json
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# ⚠️ TROQUE "your_module" pelo nome real do seu arquivo
from src.train_test import (
    Config,
    load_data,
    split_data,
    save_train_stats,
    evaluate,
    save_artifacts,
    optimize
)


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

    config = Config(data_path=file)
    result = load_data(config)

    assert not result.empty


def test_load_data_missing_column(tmp_path):
    file = tmp_path / "data.csv"

    df = pd.DataFrame({
        "ipv": [1],
        "defasagem": [0]
    })
    df.to_csv(file, index=False)

    config = Config(data_path=file)

    with pytest.raises(ValueError):
        load_data(config)


# ==========================================================
# SPLIT DATA
# ==========================================================

def test_split_data_shapes(tmp_path):
    df = pd.DataFrame({
        "ipv": [1]*40,
        "ips": [1]*40,
        "iaa": [1]*40,
        "ieg": [1]*40,
        "no_av": [1]*40,
        "ida": [1]*40,
        "media": [1]*40,
        "defasagem": [0, 1]*20
    })

    config = Config(data_path=tmp_path / "dummy.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(config, df)

    assert len(X_train) > 0
    assert len(X_test) > 0


# ==========================================================
# SAVE TRAIN STATS
# ==========================================================

def test_save_train_stats(tmp_path):
    config = Config(data_path=tmp_path / "dummy.csv")
    config.models_dir = tmp_path

    df = pd.DataFrame({
        "ipv": [1, 2, 3],
        "ips": [4, 5, 6],
    })

    path = save_train_stats(config, df)

    assert path.exists()

    with open(path) as f:
        data = json.load(f)

    assert "ipv" in data
    assert "mean" in data["ipv"]


# ==========================================================
# EVALUATE
# ==========================================================

def test_evaluate():
    mock_model = MagicMock()
    mock_model.predict.return_value = [0, 1]
    mock_model.predict_proba.return_value = [[0.8, 0.2], [0.1, 0.9]]

    X = pd.DataFrame({"a": [1, 2]})
    y = pd.Series([0, 1])

    metrics = evaluate(mock_model, X, y)

    assert "recall" in metrics
    assert metrics["true_positives"] == 1


# ==========================================================
# SAVE ARTIFACTS
# ==========================================================

def test_save_artifacts(tmp_path):
    config = Config(data_path=tmp_path / "dummy.csv")
    config.models_dir = tmp_path

    mock_model = MagicMock()

    save_artifacts(config, mock_model, {"a": 1}, {"recall": 0.9})

    assert (tmp_path / "model.pkl").exists()
    assert (tmp_path / "params.json").exists()
    assert (tmp_path / "metrics.json").exists()


# ==========================================================
# OPTIMIZE (mockado)
# ==========================================================

@patch("your_module.optuna.create_study")
def test_optimize_mocked(mock_create_study):
    mock_study = MagicMock()
    mock_study.best_params = {"n_estimators": 100}
    mock_create_study.return_value = mock_study

    config = Config(data_path=Path("dummy.csv"), n_trials=1)

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

    params = optimize(config, X, y)

    assert "n_estimators" in params
    assert "scale_pos_weight" in params
