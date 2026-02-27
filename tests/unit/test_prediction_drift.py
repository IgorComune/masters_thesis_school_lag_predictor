import json
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

import src.monitoring.prediction_drift as module  # ajuste se necessário


# ==========================================================
# helper
# ==========================================================

def write_jsonl(path: Path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ==========================================================
# load_predictions
# ==========================================================

def test_load_predictions(tmp_path):
    fake_path = tmp_path / "logs.jsonl"

    records = [
        {"prediction": 0.9},
        {"prediction": 0.1},
    ]

    write_jsonl(fake_path, records)

    with patch.object(module, "LOG_PATH", fake_path):
        df = module.load_predictions()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "prediction" in df.columns


# ==========================================================
# update_prediction_metrics — DF vazio
# ==========================================================

@patch.object(module.metrics, "DRIFT_KS")
def test_update_prediction_metrics_empty_df(mock_gauge):
    with patch.object(module, "load_predictions", return_value=pd.DataFrame()):
        module.update_prediction_metrics()

    mock_gauge.labels.assert_not_called()


# ==========================================================
# update_prediction_metrics — cálculo correto
# ==========================================================

@patch.object(module.metrics, "DRIFT_KS")
def test_update_prediction_metrics_ratio(mock_gauge):
    df = pd.DataFrame({
        "prediction": [0.9, 0.8, 0.2, 0.1]  # 2 acima de 0.5 → ratio = 0.5
    })

    mock_label = MagicMock()
    mock_gauge.labels.return_value = mock_label

    with patch.object(module, "load_predictions", return_value=df):
        module.update_prediction_metrics()

    mock_gauge.labels.assert_called_once_with(
        feature="prediction_above_0_5_ratio"
    )
    mock_label.set.assert_called_once_with(0.5)
