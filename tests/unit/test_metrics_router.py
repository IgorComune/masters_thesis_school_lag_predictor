from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch

import src.api.routers.metrics as metrics_module


def create_app():
    app = FastAPI()
    app.include_router(metrics_module.router)
    return app


@patch("src.api.routers.metrics.generate_latest")
def test_metrics_endpoint(mock_generate_latest):
    fake_metrics_output = b"fake_prometheus_metrics 1\n"
    mock_generate_latest.return_value = fake_metrics_output

    app = create_app()
    client = TestClient(app)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.content == fake_metrics_output
    assert response.headers["content-type"] == metrics_module.CONTENT_TYPE_LATEST

    mock_generate_latest.assert_called_once()
