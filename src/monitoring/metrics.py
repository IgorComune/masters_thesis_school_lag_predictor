# src/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Requests & inference
REQUESTS_TOTAL = Counter(
    "app_requests_total", "Total HTTP requests", ["method", "route", "status"]
)

INFERENCE_REQUESTS_TOTAL = Counter(
    "inference_requests_total", "Total model inference requests", ["model_version"]
)

INFERENCE_ERRORS_TOTAL = Counter(
    "inference_errors_total", "Total inference errors", ["model_version"]
)

# Latency histograms (buckets default; ajuste se quiser)
REQUEST_DURATION_SECONDS = Histogram(
    "app_request_duration_seconds", "Request duration in seconds", ["route"]
)

INFERENCE_LATENCY_SECONDS = Histogram(
    "inference_latency_seconds", "Inference latency seconds", ["model_version"]
)

# Model info
MODEL_VERSION_INFO = Gauge(
    "model_version_info", "Info about current model version", ["model_version"]
)

# Drift gauges (example generic metric; seja prudente com cardinalidade)
# Use labels apenas para features que fizer sentido (ou agrupe features).
DRIFT_KS = Gauge("feature_drift_ks", "KS statistic per feature", ["feature"])

# Helper functions
def observe_request(method: str, route: str, status: str, duration: float):
    REQUESTS_TOTAL.labels(method=method, route=route, status=status).inc()
    REQUEST_DURATION_SECONDS.labels(route=route).observe(duration)

def observe_inference(model_version: str, latency: float, error: bool = False):
    INFERENCE_REQUESTS_TOTAL.labels(model_version=model_version).inc()
    INFERENCE_LATENCY_SECONDS.labels(model_version=model_version).observe(latency)
    if error:
        INFERENCE_ERRORS_TOTAL.labels(model_version=model_version).inc()

def set_model_version(version: str):
    # zera as antigas? aqui apenas seta um gauge com label version=1 -> value 1
    MODEL_VERSION_INFO.labels(model_version=version).set(1)

def set_drift_for_feature(feature: str, ks_value: float):
    DRIFT_KS.labels(feature=feature).set(float(ks_value))
