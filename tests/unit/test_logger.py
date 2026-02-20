import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

# ajuste se o caminho for diferente
from src.monitoring.logger import log_inference


# ==========================================================
# TEST LOG SUCCESS
# ==========================================================

@patch("src.monitoring.logger.metrics.observe_inference")
@patch("src.monitoring.logger.open", new_callable=mock_open)
@patch("src.monitoring.logger.LOG_PATH", Path("test_log.json"))
@patch("src.monitoring.logger.MODEL_VERSION", "v1")
def test_log_inference_success(mock_model_version,
                               mock_log_path,
                               mock_file,
                               mock_metrics):

    input_data = {"a": 1}
    prediction = 0.87
    latency = 0.12

    log_inference(input_data, prediction, latency, true_label=1)

    # Verifica se escreveu no arquivo
    mock_file.assert_called_once_with(Path("test_log.json"), "a")

    written_content = mock_file().write.call_args[0][0]
    record = json.loads(written_content.strip())

    assert record["inputs"] == input_data
    assert record["prediction"] == float(prediction)
    assert record["latency"] == float(latency)
    assert record["true_label"] == 1
    assert "timestamp" in record

    # Verifica métrica
    mock_metrics.assert_called_once_with("v1", latency, error=False)


# ==========================================================
# TEST WITHOUT TRUE LABEL
# ==========================================================

@patch("src.monitoring.logger.metrics.observe_inference")
@patch("src.monitoring.logger.open", new_callable=mock_open)
@patch("src.monitoring.logger.LOG_PATH", Path("test_log.json"))
@patch("src.monitoring.logger.MODEL_VERSION", "v1")
def test_log_inference_without_label(mock_model_version,
                                     mock_log_path,
                                     mock_file,
                                     mock_metrics):

    log_inference({"x": 10}, 0.5, 0.2)

    written_content = mock_file().write.call_args[0][0]
    record = json.loads(written_content.strip())

    assert record["true_label"] is None
