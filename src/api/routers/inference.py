from fastapi import APIRouter
from api.schemas.inference import InferenceRequest
import pandas as pd
import time

# monitoring modules
from monitoring.logger import log_inference
from monitoring.prediction_drift import prediction_distribution
from monitoring.performance_tracker import evaluate_performance
from monitoring.model_health import health_report
from monitoring.drift_detector import detect_drift

router = APIRouter()

MODEL_COLUMNS = ['ipv','ips','iaa','ieg','no_av','ida','media']


@router.post("/predict")
def predict(request: InferenceRequest):

    # -----------------------------
    # 1️⃣ Converte request em DataFrame
    # -----------------------------
    df = pd.DataFrame([request.model_dump()])

    base_cols = ['ipv','ips','iaa','ieg','no_av','ida']
    df["media"] = df[base_cols].mean(axis=1)

    X = df[MODEL_COLUMNS]

    # -----------------------------
    # 2️⃣ Predição + latência
    # -----------------------------
    start_time = time.time()
    proba = router.model.predict_proba(X)[0]
    latency = time.time() - start_time

    prob_class_1 = float(proba[1])
    prob_class_0 = float(proba[0])

    # -----------------------------
    # 3️⃣ Log da inferência
    # -----------------------------
    log_inference(
        input_data=request.model_dump(),
        prediction=prob_class_1,
        latency=latency,
        true_label=None
    )

    # -----------------------------
    # 4️⃣ Monitoramento em tempo real (forçado)
    # -----------------------------

    try:
        print("\n🔎 Prediction Drift")
        prediction_distribution()

        print("\n📊 Performance Drift")
        evaluate_performance()

        print("\n⚙️ Model Health")
        health_report()

        print("\n📈 Data Drift")
        detect_drift(
            training_stats_path="monitoring/training_stats.json",
            production_data_df=df
        )

    except Exception as e:
        print("Monitoring error:", e)

    # -----------------------------
    # 5️⃣ Resposta da API
    # -----------------------------
    return {
        "probability_class_0": prob_class_0,
        "probability_class_1": prob_class_1,
        "latency": latency
    }
