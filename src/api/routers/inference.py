# # src/api/routers/inference.py
# from fastapi import APIRouter, HTTPException
# from src.api.schemas.inference import InferenceRequest
# import pandas as pd
# import time
# from pathlib import Path
# import joblib
# import glob
# import json
# import warnings

# # monitoring modules
# from src.monitoring.logger import log_inference
# from src.monitoring.prediction_drift import update_prediction_metrics
# from src.monitoring.performance_tracker import evaluate_performance
# from src.monitoring.model_health import health_report
# from src.monitoring.drift_detector import detect_and_set_gauges
# from src.monitoring import metrics

# # optional config (if you have it); fallback handled below
# try:
#     from src.api.core.config import MODEL_VERSION as CONFIG_MODEL_VERSION
# except Exception:
#     CONFIG_MODEL_VERSION = "v1"

# router = APIRouter()

# MODEL_COLUMNS = ['ipv', 'ips', 'iaa', 'ieg', 'no_av', 'ida', 'media']

# # module-level cached model and version
# _MODEL = None
# _MODEL_VERSION = None


# def find_latest_model(models_dir: str = "src/models") -> Path | None:
#     p = Path(models_dir)
#     candidates = sorted(p.glob("*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)
#     return candidates[0] if candidates else None


# def load_model() -> object:
#     global _MODEL, _MODEL_VERSION
#     if _MODEL is not None:
#         return _MODEL

#     model_path = find_latest_model()
#     if not model_path:
#         raise FileNotFoundError("No model .pkl found in src/models. Train and save a model first.")

#     _MODEL = joblib.load(model_path)
#     # derive version from filename (e.g. xgboost_20260101_120000.pkl)
#     _MODEL_VERSION = model_path.stem
#     return _MODEL


# def get_model_version() -> str:
#     # prefer config value if set; otherwise use derived model filename stem
#     global _MODEL_VERSION
#     if _MODEL_VERSION:
#         return _MODEL_VERSION
#     model_path = find_latest_model()
#     if model_path:
#         return model_path.stem
#     return CONFIG_MODEL_VERSION or "v1"


# def find_latest_train_stats(models_dir: str = "src/models") -> Path | None:
#     p = Path(models_dir)
#     candidates = sorted(p.glob("train_stats_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
#     return candidates[0] if candidates else None


# @router.post("/predict")
# def predict(request: InferenceRequest):
#     start_request = time.time()

#     # -----------------------------
#     # 0️⃣ Ensure model is loaded
#     # -----------------------------
#     try:
#         model = load_model()
#     except Exception as e:
#         # if model missing, fail fast with helpful message
#         raise HTTPException(status_code=500, detail=f"Model load error: {e}")

#     model_version = get_model_version()

#     # -----------------------------
#     # 1️⃣ Build DataFrame from request
#     # -----------------------------
#     # support Pydantic v1 (dict()) and v2 (model_dump())
#     try:
#         payload = request.model_dump()
#     except Exception:
#         payload = dict(request)

#     df = pd.DataFrame([payload])
#     base_cols = ['ipv', 'ips', 'iaa', 'ieg', 'no_av', 'ida']
#     # ensure base cols exist, else raise
#     missing_cols = set(base_cols) - set(df.columns)
#     if missing_cols:
#         raise HTTPException(status_code=400, detail=f"Missing required features: {missing_cols}")

#     df["media"] = df[base_cols].mean(axis=1)
#     X = df[MODEL_COLUMNS]

#     # -----------------------------
#     # 2️⃣ Prediction + inference latency
#     # -----------------------------
#     start_inference = time.time()
#     try:
#         # if model implements predict_proba with array-like input
#         proba = model.predict_proba(X)[0]
#     except Exception as e:
#         # try a fallback for models that return single value or different API
#         try:
#             pred = model.predict(X)[0]
#             proba = [1.0 - float(pred), float(pred)]
#         except Exception as e2:
#             raise HTTPException(status_code=500, detail=f"Model prediction error: {e} / {e2}")
#     inference_latency = time.time() - start_inference

#     prob_class_1 = float(proba[1])
#     prob_class_0 = float(proba[0])

#     # -----------------------------
#     # 3️⃣ Log the inference (JSONL)
#     # -----------------------------
#     try:
#         log_inference(
#             input_data=payload,
#             prediction=prob_class_1,
#             latency=inference_latency,
#             true_label=None
#         )
#     except Exception as e:
#         # logging should not break the response; warn
#         warnings.warn(f"Failed to write inference log: {e}")

#     # -----------------------------
#     # 4️⃣ Update monitoring metrics (non-blocking)
#     # -----------------------------
#     try:
#         metrics.observe_inference(model_version=model_version, latency=inference_latency, error=False)

#         # update prediction metrics (reads logs)
#         try:
#             update_prediction_metrics()
#         except Exception as _:
#             warnings.warn("update_prediction_metrics failed")

#         # update performance & health reports (they print / read logs)
#         try:
#             evaluate_performance()
#         except Exception:
#             # it's normal if there are no labeled records yet
#             pass

#         try:
#             health_report()
#         except Exception:
#             pass

#         # Data drift: find latest train stats and call detector with a production df
#         try:
#             stats_path = find_latest_train_stats()
#             if stats_path:
#                 # production_data_df expected to have features as columns; we already have df
#                 # pass the training stats path and the production df (single row OK)
#                 detect_and_set_gauges(str(stats_path), df[MODEL_COLUMNS])
#             else:
#                 warnings.warn("No train_stats_*.json found in src/models — skipping data drift detection")
#         except Exception:
#             warnings.warn("detect_and_set_gauges failed")
#     except Exception as e:
#         # any metric emission failure shouldn't break endpoint; increment error metric
#         try:
#             metrics.observe_inference(model_version=model_version, latency=inference_latency, error=True)
#         except Exception:
#             pass
#         warnings.warn(f"Monitoring emission failed: {e}")

#     # -----------------------------
#     # 5️⃣ Observe HTTP request duration metric
#     # -----------------------------
#     request_latency = time.time() - start_request
#     try:
#         metrics.REQUEST_DURATION_SECONDS.labels(route="/predict").observe(request_latency)
#     except Exception:
#         # don't break on metrics errors
#         pass

#     # -----------------------------
#     # 6️⃣ Return response
#     # -----------------------------
#     return {
#         "probability_class_0": prob_class_0,
#         "probability_class_1": prob_class_1,
#         "latency": inference_latency,
#         "model_version": model_version
#     }

# src/api/routers/inference.py
from fastapi import APIRouter, HTTPException
from src.api.schemas.inference import InferenceRequest
import pandas as pd
import time
from pathlib import Path
import joblib
import warnings

# monitoring modules
from src.monitoring.logger import log_inference
from src.monitoring.prediction_drift import update_prediction_metrics
from src.monitoring.performance_tracker import evaluate_performance
from src.monitoring.model_health import health_report
from src.monitoring.drift_detector import detect_and_set_gauges
from src.monitoring import metrics

# optional config
try:
    from src.api.core.config import MODEL_VERSION as CONFIG_MODEL_VERSION
except Exception:
    CONFIG_MODEL_VERSION = "v1"

router = APIRouter()
MODEL_COLUMNS = ['ipv', 'ips', 'iaa', 'ieg', 'no_av', 'ida', 'media']

# cached model
_MODEL = None
_MODEL_VERSION = None


def find_latest_model(models_dir: str = "src/models") -> Path | None:
    p = Path(models_dir)
    candidates = sorted(p.glob("*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_model() -> object:
    global _MODEL, _MODEL_VERSION
    if _MODEL is not None:
        return _MODEL

    model_path = find_latest_model()
    if not model_path:
        raise FileNotFoundError("No model .pkl found in src/models.")

    _MODEL = joblib.load(model_path)
    _MODEL_VERSION = model_path.stem
    # Set model version in Prometheus
    metrics.set_model_version(_MODEL_VERSION)
    return _MODEL


def get_model_version() -> str:
    global _MODEL_VERSION
    if _MODEL_VERSION:
        return _MODEL_VERSION
    model_path = find_latest_model()
    if model_path:
        _MODEL_VERSION = model_path.stem
        metrics.set_model_version(_MODEL_VERSION)
        return _MODEL_VERSION
    return CONFIG_MODEL_VERSION or "v1"


def find_latest_train_stats(models_dir: str = "src/models") -> Path | None:
    p = Path(models_dir)
    candidates = sorted(p.glob("train_stats_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


@router.post("/predict")
def predict(request: InferenceRequest):
    start_request = time.time()
    model_version = get_model_version()
    http_status = "200"

    try:
        # -----------------------------
        # Load model
        # -----------------------------
        model = load_model()

        # -----------------------------
        # Build DataFrame
        # -----------------------------
        try:
            payload = request.model_dump()
        except Exception:
            payload = dict(request)

        df = pd.DataFrame([payload])
        base_cols = ['ipv', 'ips', 'iaa', 'ieg', 'no_av', 'ida']
        missing_cols = set(base_cols) - set(df.columns)
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing required features: {missing_cols}")

        df["media"] = df[base_cols].mean(axis=1)
        X = df[MODEL_COLUMNS]

        # -----------------------------
        # Inference
        # -----------------------------
        start_inference = time.time()
        try:
            proba = model.predict_proba(X)[0]
        except Exception:
            try:
                pred = model.predict(X)[0]
                proba = [1.0 - float(pred), float(pred)]
            except Exception as e2:
                raise HTTPException(status_code=500, detail=f"Model prediction error: {e2}")
        inference_latency = time.time() - start_inference
        prob_class_0 = float(proba[0])
        prob_class_1 = float(proba[1])

        # -----------------------------
        # Log inference
        # -----------------------------
        try:
            log_inference(input_data=payload, prediction=prob_class_1, latency=inference_latency, true_label=None)
        except Exception as e:
            warnings.warn(f"Failed to log inference: {e}")

        # -----------------------------
        # Update Prometheus metrics
        # -----------------------------
        try:
            metrics.observe_inference(model_version=model_version, latency=inference_latency, error=False)
        except Exception as e:
            warnings.warn(f"Prometheus observe_inference failed: {e}")

        # -----------------------------
        # Background / optional: drift & performance
        # -----------------------------
        try:
            # Only update if train stats exist
            stats_path = find_latest_train_stats()
            if stats_path:
                detect_and_set_gauges(str(stats_path), df[MODEL_COLUMNS])
        except Exception:
            warnings.warn("detect_and_set_gauges failed")

    except HTTPException as e:
        http_status = str(e.status_code)
        metrics.observe_inference(model_version=model_version, latency=0.0, error=True)
        raise
    except Exception:
        http_status = "500"
        metrics.observe_inference(model_version=model_version, latency=0.0, error=True)
        raise HTTPException(status_code=500, detail="Unexpected error during prediction")
    finally:
        # -----------------------------
        # Observe HTTP request
        # -----------------------------
        request_latency = time.time() - start_request
        try:
            metrics.observe_request(method="POST", route="/predict", status=http_status, duration=request_latency)
        except Exception:
            warnings.warn("Prometheus observe_request failed")

    # -----------------------------
    # Return response
    # -----------------------------
    return {
        "probability_class_0": prob_class_0,
        "probability_class_1": prob_class_1,
        "latency": inference_latency,
        "model_version": model_version
    }
