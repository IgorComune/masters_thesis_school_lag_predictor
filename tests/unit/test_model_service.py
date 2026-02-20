from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.your_module import ModelService  # <-- ajuste aqui


# ---------------------------------------------------------------------
# Inicialização
# ---------------------------------------------------------------------

def test_init_sets_path():
    service = ModelService("model.pkl")

    assert isinstance(service.model_path, Path)
    assert service.model is None


# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------

@patch("src.your_module.joblib.load")
def test_load_model(mock_load):
    fake_model = MagicMock()
    mock_load.return_value = fake_model

    service = ModelService("model.pkl")
    service.load()

    mock_load.assert_called_once_with(service.model_path)
    assert service.model == fake_model


# ---------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------

def test_predict_calls_model():
    fake_model = MagicMock()
    fake_model.predict.return_value = [1, 0]

    service = ModelService("model.pkl")
    service.model = fake_model

    result = service.predict([[1], [2]])

    fake_model.predict.assert_called_once()
    assert result == [1, 0]


# ---------------------------------------------------------------------
# Predict Proba (exists)
# ---------------------------------------------------------------------

def test_predict_proba_when_available():
    fake_model = MagicMock()
    fake_model.predict_proba.return_value = [[0.1, 0.9]]

    service = ModelService("model.pkl")
    service.model = fake_model

    result = service.predict_proba([[1]])

    fake_model.predict_proba.assert_called_once()
    assert result == [[0.1, 0.9]]


# ---------------------------------------------------------------------
# Predict Proba (not available)
# ---------------------------------------------------------------------

def test_predict_proba_when_not_available():
    class ModelWithoutProba:
        def predict(self, X):
            return [1]

    service = ModelService("model.pkl")
    service.model = ModelWithoutProba()

    result = service.predict_proba([[1]])

    assert result is None
