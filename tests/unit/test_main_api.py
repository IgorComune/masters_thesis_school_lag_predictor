import sys
import importlib
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


MODULE_PATH = "src.api.main"


def reload_module():
    if MODULE_PATH in sys.modules:
        del sys.modules[MODULE_PATH]
    return importlib.import_module(MODULE_PATH)


# ---------------------------------------------------------------------
# Caso feliz: modelo existe
# ---------------------------------------------------------------------

@patch("src.api.main.joblib.load")
@patch("src.api.main.Path.exists")
def test_app_initialization_success(mock_exists, mock_load):
    mock_exists.return_value = True

    fake_model = MagicMock()
    mock_load.return_value = fake_model

    module = reload_module()

    # app criado
    assert module.app.title == "XGBoost Inference API"

    # modelo carregado
    mock_load.assert_called_once()

    # modelo injetado no router
    assert module.inference.router.model == fake_model

    # routers registrados
    routes = [route.path for route in module.app.routes]

    assert any("/inference" in r for r in routes)
    assert any("/health" in r for r in routes)
    assert any("/metrics" in r for r in routes)


# ---------------------------------------------------------------------
# Caso erro: modelo não existe
# ---------------------------------------------------------------------

@patch("src.api.main.Path.exists")
def test_model_not_found_raises(mock_exists):
    mock_exists.return_value = False

    if MODULE_PATH in sys.modules:
        del sys.modules[MODULE_PATH]

    with pytest.raises(FileNotFoundError):
        importlib.import_module(MODULE_PATH)
