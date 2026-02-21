import pytest
from pydantic import ValidationError

from src.api.schemas.inference import InferenceRequest


# ---------------------------------------------------------
# SUCCESS - criação válida
# ---------------------------------------------------------

def test_inference_request_valid():
    payload = {
        "ipv": 1,
        "ips": 2,
        "iaa": 3,
        "ieg": 4,
        "no_av": 5,
        "ida": 6,
    }

    obj = InferenceRequest(**payload)

    assert obj.ipv == 1.0
    assert obj.ips == 2.0
    assert obj.iaa == 3.0
    assert obj.ieg == 4.0
    assert obj.no_av == 5.0
    assert obj.ida == 6.0


# ---------------------------------------------------------
# SUCCESS - coerção de string para float
# ---------------------------------------------------------

def test_inference_request_type_coercion():
    payload = {
        "ipv": "1.5",
        "ips": "2.5",
        "iaa": "3.5",
        "ieg": "4.5",
        "no_av": "5.5",
        "ida": "6.5",
    }

    obj = InferenceRequest(**payload)

    assert isinstance(obj.ipv, float)
    assert obj.ipv == 1.5


# ---------------------------------------------------------
# ERROR - campo ausente
# ---------------------------------------------------------

def test_inference_request_missing_field():
    payload = {
        "ipv": 1,
        "ips": 2,
        "iaa": 3,
        "ieg": 4,
        "no_av": 5,
        # "ida" faltando
    }

    with pytest.raises(ValidationError):
        InferenceRequest(**payload)


# ---------------------------------------------------------
# ERROR - tipo inválido
# ---------------------------------------------------------

def test_inference_request_invalid_type():
    payload = {
        "ipv": "invalid",
        "ips": 2,
        "iaa": 3,
        "ieg": 4,
        "no_av": 5,
        "ida": 6,
    }

    with pytest.raises(ValidationError):
        InferenceRequest(**payload)
