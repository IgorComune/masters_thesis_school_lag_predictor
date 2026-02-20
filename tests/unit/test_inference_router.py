from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pytest

import src.api.routers.inference as inference_module


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def create_app():
    app = FastAPI()
    app.include_router(inference_module.router, prefix="/inference")
    return app


VALID_PAYLOAD = {
    "ipv": 1,
    "ips": 2,
    "iaa": 3,
    "ieg": 4,
    "no_av": 5,
    "ida": 6,
}


# ---------------------------------------------------------
# SUCCESS - predict_proba
# ---------------------------------------------------------

@patch("src.api.routers.inference.load_model")
@patch("src.api.routers.inference.metrics.observe_request")
@patch("src.api.routers.inference.metrics.observe_inference")
@patch("src.api.routers.inference.log_inference")
def test_predict_success_with_predict_proba(
    mock_log,
    mock_observe_inference,
    mock_observe_request,
    mock_load_model,
):
    fake_model = MagicMock()
    fake_model.predict_proba.return_value = [[0.2, 0.8]]
    mock_load_model.return_value = fake_model

    app = create_app()
    client = TestClient(app)

    response = client.post("/inference/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()

    assert body["probability_class_0"] == 0.2
    assert body["probability_class_1"] == 0.8

    mock_log.assert_called_once()
    mock_observe_inference.assert_called()
    mock_observe_request.assert_called()


# ---------------------------------------------------------
# SUCCESS - fallback predict()
# ---------------------------------------------------------

@patch("src.api.routers.inference.load_model")
def test_predict_fallback_predict(mock_load_model):
    fake_model = MagicMock()
    fake_model.predict_proba.side_effect = Exception("no proba")
    fake_model.predict.return_value = [1]
    mock_load_model.return_value = fake_model

    app = create_app()
    client = TestClient(app)

    response = client.post("/inference/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()

    assert body["probability_class_1"] == 1.0
    assert body["probability_class_0"] == 0.0


# ---------------------------------------------------------
# ERROR - missing features
# ---------------------------------------------------------

@patch("src.api.routers.inference.load_model")
def test_predict_missing_feature(mock_load_model):
    mock_load_model.return_value = MagicMock()

    app = create_app()
    client = TestClient(app)

    bad_payload = {"ipv": 1}  # faltando tudo

    response = client.post("/inference/predict", json=bad_payload)

    assert response.status_code == 400
    assert "Missing required features" in response.text


# ---------------------------------------------------------
# ERROR - model not found
# ---------------------------------------------------------

@patch("src.api.routers.inference.load_model")
def test_predict_model_load_error(mock_load_model):
    mock_load_model.side_effect = FileNotFoundError("no model")

    app = create_app()
    client = TestClient(app)

    response = client.post("/inference/predict", json=VALID_PAYLOAD)

    assert response.status_code == 500
