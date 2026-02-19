from fastapi import APIRouter
from src.api.schemas.inference import InferenceRequest
import pandas as pd
import time

# monitoring modules
from src.monitoring.logger import log_inference
from src.monitoring.prediction_drift import load_predictions, update_prediction_metrics
from src.monitoring.performance_tracker import evaluate_performance
from src.monitoring.model_health import health_report
from src.monitoring.drift_detector import detect_and_set_gauges
from src.monitoring import metrics

router = APIRouter()

MODEL_COLUMNS = ['ipv','ips','iaa','ieg','no_av','ida','media']

@router.post("/predict")
def predict(request: InferenceRequest):
    start_request = time.time()

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
    start_inference = time.time()
    proba = router.model.predict_proba(X)[0]  # mantém sua lógica
    inference_latency = time.time() - start_inference

    prob_class_1 = float(proba[1])
    prob_class_0 = float(proba[0])

    # -----------------------------
    # 3️⃣ Log da inferência
    # -----------------------------
    log_inference(
        input_data=request.model_dump(),
        prediction=prob_class_1,
        latency=inference_latency,
        true_label=None
    )

    # -----------------------------
    # 4️⃣ Emissão de métricas Prometheus
    # -----------------------------
    try:
        # Incrementa métricas custom de inferência
        metrics.observe_inference(
            model_version="v1",  # ou MODEL_VERSION do config
            latency=inference_latency,
            error=False
        )

        # Atualiza drift (exemplo rápido, seu drift_detector já calcula)
        detect_drift(
            training_stats_path="monitoring/training_stats.json",
            production_data_df=df
        )
        prediction_distribution()  # opcional, atualizar gauges aqui
        evaluate_performance()
        health_report()

    except Exception as e:
        # marca erro nas métricas
        metrics.observe_inference(model_version="v1", latency=inference_latency, error=True)
        print("Monitoring error:", e)

    # -----------------------------
    # 5️⃣ Observa métricas HTTP via instrumentator
    # -----------------------------
    request_latency = time.time() - start_request
    metrics.REQUEST_DURATION_SECONDS.labels(route="/predict").observe(request_latency)

    # -----------------------------
    # 6️⃣ Resposta da API
    # -----------------------------
    return {
        "probability_class_0": prob_class_0,
        "probability_class_1": prob_class_1,
        "latency": inference_latency
    }
