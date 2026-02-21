import json
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.monitoring.drift_detector import detect_and_set_gauges


# ==========================================================
# KS NORMAL FLOW
# ==========================================================

@patch("src.monitoring.drift_detector.metrics.set_drift_for_feature")
@patch("src.monitoring.drift_detector.ks_2samp")
def test_detect_and_set_gauges_ks(mock_ks, mock_set_gauge, tmp_path):

    # mock KS retornando estatística conhecida
    mock_ks.return_value = (0.42, 0.01)

    training_stats = {
        "feature1": {
            "mean": 10,
            "sample": [1, 2, 3, 4]
        }
    }

    json_path = tmp_path / "train_stats.json"
    with open(json_path, "w") as f:
        json.dump(training_stats, f)

    prod_df = pd.DataFrame({
        "feature1": [5, 6, 7]
    })

    detect_and_set_gauges(json_path, prod_df)

    mock_ks.assert_called_once()
    mock_set_gauge.assert_called_once_with("feature1", 0.42)


# ==========================================================
# FALLBACK (quando sample não existe)
# ==========================================================

@patch("src.monitoring.drift_detector.metrics.set_drift_for_feature")
@patch("src.monitoring.drift_detector.ks_2samp")
def test_detect_and_set_gauges_fallback(mock_ks, mock_set_gauge, tmp_path):

    # força erro no KS
    mock_ks.side_effect = Exception("KS failed")

    training_stats = {
        "feature1": {
            "mean": 10
            # sem sample
        }
    }

    json_path = tmp_path / "train_stats.json"
    with open(json_path, "w") as f:
        json.dump(training_stats, f)

    prod_df = pd.DataFrame({
        "feature1": [15, 15, 15]
    })

    detect_and_set_gauges(json_path, prod_df)

    # fallback = diferença absoluta das médias
    expected_diff = abs(10 - 15)

    mock_set_gauge.assert_called_once_with("feature1", expected_diff)
