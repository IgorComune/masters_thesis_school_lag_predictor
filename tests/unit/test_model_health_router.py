from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch

import src.api.routers.health as module  # <-- ajuste se necessário


def create_test_app():
    app = FastAPI()
    app.include_router(module.router)
    return app


@patch("src.api.routers.health.sanitize_for_json")
@patch("src.api.routers.health.health_report")
def test_get_model_health(mock_health_report, mock_sanitize):
    mock_health_report.return_value = {"raw": "data"}
    mock_sanitize.return_value = {"clean": "data"}

    app = create_test_app()
    client = TestClient(app)

    response = client.get("/model_health")

    assert response.status_code == 200
    assert response.json() == {"clean": "data"}

    mock_health_report.assert_called_once()
    mock_sanitize.assert_called_once_with({"raw": "data"})
