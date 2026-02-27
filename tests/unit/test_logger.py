import json
from pathlib import Path
from unittest.mock import patch, mock_open

import src.monitoring.logger as logger_module


# ==========================================================
# TEST LOG SUCCESS
# ==========================================================

@patch("src.monitoring.logger.metrics.observe_inference")
@patch("builtins.open", new_callable=mock_open)
def test_log_inference_success(mock_file, mock_metrics):

    test_path = Path("test_log.json")

    with patch.object(logger_module, "LOG_PATH", test_path), \
         patch.object(logger_module, "MODEL_VERSION", "v1"):

        input_data = {"a": 1}
        prediction = 0.87
        latency = 0.12

        logger_module.log_inference(input_data, prediction, latency, true_label=1)

    mock_file.assert_called_once_with(test_path, "a")

    written_content = mock_file().write.call_args[0][0]
    record = json.loads(written_content.strip())

    assert record["inputs"] == input_data
    assert record["prediction"] == float(prediction)
    assert record["latency"] == float(latency)
    assert record["true_label"] == 1
    assert "timestamp" in record

    mock_metrics.assert_called_once_with("v1", latency, error=False)


# ==========================================================
# TEST WITHOUT TRUE LABEL
# ==========================================================

@patch("src.monitoring.logger.metrics.observe_inference")
@patch("builtins.open", new_callable=mock_open)
def test_log_inference_without_label(mock_file, mock_metrics):

    test_path = Path("test_log.json")

    with patch.object(logger_module, "LOG_PATH", test_path), \
         patch.object(logger_module, "MODEL_VERSION", "v1"):

        logger_module.log_inference({"x": 10}, 0.5, 0.2)

    written_content = mock_file().write.call_args[0][0]
    record = json.loads(written_content.strip())

    assert record["true_label"] is None
