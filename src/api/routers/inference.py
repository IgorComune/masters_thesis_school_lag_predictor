from fastapi import APIRouter
from api.schemas.inference import InferenceRequest
import pandas as pd

router = APIRouter()

MODEL_COLUMNS = ['ipv','ips','iaa','ieg','nº_av','ida','media']

@router.post("/predict")
def predict(request: InferenceRequest):

    df = pd.DataFrame([request.model_dump()])  # se estiver usando Pydantic v2

    base_cols = ['ipv','ips','iaa','ieg','nº_av','ida']
    df["media"] = df[base_cols].mean(axis=1)

    X = df[MODEL_COLUMNS]

    # 🔥 Probabilidades
    proba = router.model.predict_proba(X)[0]

    # assumindo classificação binária
    prob_class_1 = float(proba[1])
    prob_class_0 = float(proba[0])

    return {
        "probability_class_0": prob_class_0,
        "probability_class_1": prob_class_1
    }
