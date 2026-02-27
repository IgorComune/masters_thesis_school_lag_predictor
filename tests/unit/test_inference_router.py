from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pytest

import src.api.routers.inference as inference_module


# ---------------------------------------------------------
# Helper: criar app com router
# ---------------------------------------------------------
def create_app():
    app = FastAPI()
    app.include_router(inference_module.router, prefix="/inference")
    return app


# ---------------------------------------------------------
# Payload válido
# ---------------------------------------------------------
VALID_PAYLOAD = {
    "ipv": 1,
    "ips": 2,
    "iaa": 3,
    "ieg": 4,
    "no_av": 5,
    "ida": 6,
    "media": 3.5,
}


# =========================================================
# TESTES - SUCESSO
# =========================================================

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


# =========================================================
# TESTES - ERRO
# =========================================================

@patch("src.api.routers.inference.load_model")
def test_predict_missing_feature(mock_load_model):
    """Testa payload com features faltando. FastAPI/Pydantic retorna 422."""
    mock_load_model.return_value = MagicMock()
    app = create_app()
    client = TestClient(app)

    bad_payload = {"ipv": 1}  # faltando campos obrigatórios

    response = client.post("/inference/predict", json=bad_payload)
    assert response.status_code == 422

    # agora inspecionamos o JSON detail para cada campo faltante
    detail = response.json().get("detail", [])
    missing_fields = [d["loc"][-1] for d in detail]
    for field in ["ips", "iaa", "ieg", "no_av", "ida"]:
        assert field in missing_fields


@patch("src.api.routers.inference.load_model")
def test_predict_model_load_error(mock_load_model):
    """Simula erro ao carregar o modelo (.pkl não existe)."""
    mock_load_model.side_effect = FileNotFoundError("no model")

    app = create_app()
    client = TestClient(app)

    response = client.post("/inference/predict", json=VALID_PAYLOAD)
    assert response.status_code == 500
    assert "Unexpected error" in response.text or "Model prediction error" in response.text
