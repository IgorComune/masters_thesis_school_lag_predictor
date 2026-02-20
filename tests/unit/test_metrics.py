from unittest.mock import MagicMock, patch

# ajuste se necessário
import src.monitoring.metrics as metrics


# ==========================================================
# observe_request
# ==========================================================

@patch.object(metrics, "REQUESTS_TOTAL")
@patch.object(metrics, "REQUEST_DURATION_SECONDS")
def test_observe_request(mock_duration, mock_requests):

    mock_counter = MagicMock()
    mock_hist = MagicMock()

    mock_requests.labels.return_value = mock_counter
    mock_duration.labels.return_value = mock_hist

    metrics.observe_request("GET", "/predict", "200", 0.35)

    mock_requests.labels.assert_called_once_with(
        method="GET", route="/predict", status="200"
    )
    mock_counter.inc.assert_called_once()

    mock_duration.labels.assert_called_once_with(route="/predict")
    mock_hist.observe.assert_called_once_with(0.35)


# ==========================================================
# observe_inference
# ==========================================================

@patch.object(metrics, "INFERENCE_REQUESTS_TOTAL")
@patch.object(metrics, "INFERENCE_LATENCY_SECONDS")
@patch.object(metrics, "INFERENCE_ERRORS_TOTAL")
def test_observe_inference_no_error(mock_errors, mock_latency, mock_requests):

    mock_req = MagicMock()
    mock_lat = MagicMock()

    mock_requests.labels.return_value = mock_req
    mock_latency.labels.return_value = mock_lat

    metrics.observe_inference("v1", 0.2, error=False)

    mock_requests.labels.assert_called_once_with(model_version="v1")
    mock_req.inc.assert_called_once()

    mock_latency.labels.assert_called_once_with(model_version="v1")
    mock_lat.observe.assert_called_once_with(0.2)

    mock_errors.labels.assert_not_called()


@patch.object(metrics, "INFERENCE_REQUESTS_TOTAL")
@patch.object(metrics, "INFERENCE_LATENCY_SECONDS")
@patch.object(metrics, "INFERENCE_ERRORS_TOTAL")
def test_observe_inference_with_error(mock_errors, mock_latency, mock_requests):

    mock_req = MagicMock()
    mock_lat = MagicMock()
    mock_err = MagicMock()

    mock_requests.labels.return_value = mock_req
    mock_latency.labels.return_value = mock_lat
    mock_errors.labels.return_value = mock_err

    metrics.observe_inference("v1", 0.5, error=True)

    mock_err.inc.assert_called_once()


# ==========================================================
# set_model_version
# ==========================================================

@patch.object(metrics, "MODEL_VERSION_INFO")
def test_set_model_version(mock_gauge):

    mock_label = MagicMock()
    mock_gauge.labels.return_value = mock_label

    metrics.set_model_version("v2")

    mock_gauge.labels.assert_called_once_with(model_version="v2")
    mock_label.set.assert_called_once_with(1)


# ==========================================================
# set_drift_for_feature
# ==========================================================

@patch.object(metrics, "DRIFT_KS")
def test_set_drift_for_feature(mock_gauge):

    mock_label = MagicMock()
    mock_gauge.labels.return_value = mock_label

    metrics.set_drift_for_feature("ipv", 0.73)

    mock_gauge.labels.assert_called_once_with(feature="ipv")
    mock_label.set.assert_called_once_with(0.73)
