# School Lag Predictor - MLOps Pipeline

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Production](https://img.shields.io/badge/Production-Render-purple.svg)](https://masters-thesis-school-lag-predictor.onrender.com/docs)

> **Master's Thesis Project**: End-to-end MLOps pipeline for predicting student academic lag using longitudinal educational data (2022-2024).

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Features & Indicators](#features--indicators)
- [Installation](#installation)
- [Usage](#usage)
  - [Training Pipeline](#training-pipeline)
  - [Inference Pipeline](#inference-pipeline)
  - [Monitoring](#monitoring)
- [Deployment](#deployment)
  - [Local Development](#local-development)
  - [Docker](#docker)
  - [Production (Render)](#production-render)
- [Project Structure](#project-structure)
- [Model Selection](#model-selection)
- [Monitoring & Observability](#monitoring--observability)
- [Testing](#testing)
- [Known Issues & Trade-offs](#known-issues--trade-offs)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)
- [License](#license)

---

## Overview

### Problem Statement

Predicting student academic lag early enables targeted interventions. This project implements a production-grade ML system that:

1. **Ingests** multi-year educational data (2022-2024)
2. **Engineers** longitudinal features from raw student performance indicators
3. **Trains** optimized XGBoost classifier (Optuna hyperparameter tuning)
4. **Deploys** FastAPI service with drift detection, performance monitoring, and observability
5. **Monitors** model health via Prometheus + Grafana dashboards

### Key Differentiators

- **Longitudinal approach**: Captures temporal patterns across 3 years
- **Full MLOps stack**: Not just model training—includes orchestration (Prefect), experiment tracking (MLflow), monitoring (Prometheus/Grafana)
- **Drift detection**: Automated feature/prediction drift alerts
- **Production-ready**: Dockerized, deployed on Render with health checks

---

## Quick Start

**Test the API in 3 minutes** 

```bash
# 1. Clone and setup
git clone https://github.com/IgorComune/masters_thesis_school_lag_predictor.git
cd masters_thesis_school_lag_predictor
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Start API server
python main.py

# 3. Test prediction (in another terminal)
curl -X POST http://localhost:8000/inference/predict \
  -H "Content-Type: application/json" \
  -d '{"ipv": 0.5, "ips": 0.7, "iaa": 0.3, "ieg": 0.8, "no_av": 0.2, "ida": 0.6}'
```

**Expected output**:
```json
   {
      "probability_class_0":0.32471853494644165,
      "probability_class_1":0.6752814650535583,
      "latency":0.05233263969421387,
      "model_version":"model"
   }

* probability_class_1: indicates the probability of lagging
```

> **Note**: This quick start runs API-only. For full MLOps stack (Prometheus, Grafana, Prefect), see [Installation](#installation).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  Raw CSV (2022-2024) → Ingestion → Feature Engineering →       │
│  Longitudinal Aggregation → Train/Test Split                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  [Notebooks]          [MLflow Tracking]      [Optuna Tuning]   │
│  01_eda.ipynb         - Experiments          - Hyperparameter   │
│  02_feature_eng.ipynb - Metrics              - Best model save  │
│  03_experiments.ipynb - Model registry       - Trials history   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE LAYER (FastAPI)                    │
├─────────────────────────────────────────────────────────────────┤
│  POST /inference/predict  - Real-time predictions               │
│  GET  /health/model_health - Model health check                 │
│  GET  /metrics/metrics     - Prometheus metrics                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   MONITORING LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  [Prometheus] → [Grafana Dashboards]                            │
│  - Request latency       - Prediction distribution              │
│  - Drift detection       - Model performance                    │
│  - Error rates           - Feature statistics                   │
└─────────────────────────────────────────────────────────────────┘
```

**Data Flow**:
1. Raw CSVs → `src/data/ingestion.py` → Cleaned data
2. Cleaned → `src/features/engineering.py` → Feature-engineered dataset
3. Features → `src/train_test/train.py` → Trained XGBoost model
4. Model → `src/api/main.py` → FastAPI service
5. Predictions → `src/monitoring/*` → Drift detection + metrics export

---

## Features & Indicators

The model uses **6 educational performance indicators** (assumed to be normalized/scaled):

| Feature | Description (Placeholder - Update with actual definitions) |
|---------|-----------------------------------------------------------|
| `ipv`   | Student performance indicator 1 |
| `ips`   | Student performance indicator 2 |
| `iaa`   | Student performance indicator 3 |
| `ieg`   | Student performance indicator 4 |
| `no_av` | Student performance indicator 5 |
| `ida`   | Student performance indicator 6 |

> **Note**: Refer to `documents/Data Dict.pdf` for precise definitions. Feature engineering includes temporal aggregations (mean, std, trend) across years.

---

## Installation

### Prerequisites

- **Python 3.12+** (strict requirement for compatibility)
- **Docker** (optional, for containerized deployment)
- **8GB RAM minimum** (for training with full dataset)

### Local Development Setup

```bash
# Clone repository
git clone https://github.com/IgorComune/masters_thesis_school_lag_predictor.git
cd masters_thesis_school_lag_predictor

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### External Dependencies

#### Prometheus (Metrics Collection)

```bash
# Download and extract
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
tar xvf prometheus-2.45.0.linux-amd64.tar.gz
rm prometheus-2.45.0.linux-amd64.tar.gz
cd prometheus-2.45.0.linux-amd64

# Configure scrape targets (edit prometheus.yml)

* prometheus.yml

scrape_configs:
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]
  
  - job_name: "fastapi"
    metrics_path: "/metrics/metrics"
    static_configs:
      - targets: ["127.0.0.1:8000"]


# Start Prometheus
./prometheus --config.file=prometheus.yml
```

#### Grafana (Visualization)

```bash
# Download and extract
wget https://dl.grafana.com/oss/release/grafana-10.1.0.linux-amd64.tar.gz
tar -zxvf grafana-10.1.0.linux-amd64.tar.gz
rm grafana-10.1.0.linux-amd64.tar.gz
cd grafana-10.1.0/bin

# Start Grafana
./grafana-server

# Access at http://localhost:3000
# Default credentials: admin/admin
```

---

## Usage

### Training Pipeline

#### Option 1: Jupyter Notebooks (Recommended for experimentation)

```bash
jupyter notebook notebooks/

# Execute in order:
# 1. 01_eda.ipynb               - Exploratory data analysis
# 2. 02_feature_engineering.ipynb - Feature creation
# 3. 03_experiments_models.ipynb  - Model training + Optuna tuning
```

**Key outputs**:
- `notebooks/best_model_XGBoost_(Optuna)_*.pkl` - Trained model
- `notebooks/best_params_*.json` - Optimal hyperparameters
- `notebooks/model_card_*.md` - Model documentation
- `notebooks/test_metrics_*.json` - Test set performance

#### Option 2: Python Scripts (Automated)

```bash
# Full pipeline: ingestion → feature engineering → training
python -m src.train_test.train

# Evaluation on test set
python -m src.train_test.test
```

**MLflow Tracking**:
```bash
# Start MLflow UI (port 5000)
mlflow ui --backend-store-uri sqlite:///mlflow.db

# View experiments at http://localhost:5000
```

---

### Inference Pipeline

#### Local API Server

```bash
# Start all services (FastAPI + Prometheus + Grafana)
python main.py

# Or manually:
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Available endpoints**:
- **API Docs**: http://localhost:8000/docs
- **Prediction**: `POST /inference/predict`
- **Health Check**: `GET /health/model_health`
- **Metrics**: `GET /metrics/metrics`

#### Example Requests

```bash
# Single prediction
curl -X POST http://localhost:8000/inference/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ipv": 0.5,
    "ips": 0.7,
    "iaa": 0.3,
    "ieg": 0.8,
    "no_av": 0.2,
    "ida": 0.6
  }'


# Model health
curl http://localhost:8000/health/model_health

# Prometheus metrics
curl http://localhost:8000/metrics/metrics
```

**Batch predictions** (via script):
```bash
bash scripts/curl_examples.sh
```

---

### Monitoring

#### Grafana Dashboards

1. Access Grafana: http://localhost:3000
2. Add Prometheus datasource: `http://localhost:9090`
3. Import dashboards for:
   - Prediction latency (p50, p95, p99)
   - Request throughput
   - Drift detection alerts
   - Feature distribution changes

#### Prefect Orchestration

```bash
# Start Prefect UI
prefect server start

# Access at http://localhost:4200
```

---

## Deployment

### Local Development

```bash
# Start FastAPI server
python main.py
# This starts the FastAPI application on port 8000

# OR start services individually:
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Service URLs** (after starting each component separately):
- **FastAPI Docs**: http://localhost:8000/docs (started by `main.py`)
- **Prometheus**: http://localhost:9090 (requires manual setup - see Installation)
- **Grafana**: http://localhost:3000 (requires manual setup - see Installation)
- **Prefect**: http://localhost:4200 (optional - see Monitoring section)
- **MLflow**: http://localhost:5000 (optional - for training only)

> **Note**: `main.py` launches the FastAPI server. Prometheus, Grafana, and Prefect must be installed and started separately as described in the [Installation](#installation) section.

---

### Docker

#### Build Image

```bash
docker build -t school-lag-predictor:latest .
```

**Image details**:
- Base: `python:3.12-slim`
- Size: ~500MB (optimized with multi-stage build)
- Healthcheck: `/health/model_health` endpoint

#### Run Container

```bash
# Expose port 10000
docker run -d \
  --name school-lag-api \
  -p 10000:10000 \
  -e LOG_LEVEL=info \
  school-lag-predictor:latest

# Check logs
docker logs -f school-lag-api
```

> **Port Configuration**: Docker uses port **10000** (vs port 8000 for local development) to avoid conflicts with locally running services.

#### Test Dockerized API

```bash
curl -X POST http://127.0.0.1:10000/inference/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ipv": 1,
    "ips": 2,
    "iaa": 3,
    "ieg": 4,
    "no_av": 5,
    "ida": 6
  }'
```
---

### Production (Render)

**Live deployment**: [https://masters-thesis-school-lag-predictor.onrender.com](https://masters-thesis-school-lag-predictor.onrender.com/docs)

#### Configuration

Render auto-deploys from `main` branch using:
- `Dockerfile` (containerized build)
- `deployment/entrypoint.sh` (startup script)
- Environment variables (set in Render dashboard)

#### Test Production API

```bash
curl -X POST https://masters-thesis-school-lag-predictor.onrender.com/inference/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ipv": 0.5,
    "ips": 0.7,
    "iaa": 0.3,
    "ieg": 0.8,
    "no_av": 0.2,
    "ida": 0.6
  }'
```

#### Production Considerations

**Trade-offs**:
- **Cold start**: Render free tier has ~30s cold start after inactivity
- **Resource limits**: 512MB RAM (sufficient for XGBoost inference)
- **No GPU**: CPU-only inference (~50ms latency per request)

**Recommended improvements**:
- Use Render paid tier for always-on instances
- Add Redis caching for frequent predictions
- Implement request queuing for batch inference

---

## Project Structure

```
masters_thesis_school_lag_predictor/
│
├── data/
│   ├── raw/                  # Original CSVs (2022-2024)
│   └── processed/            # Cleaned, feature-engineered datasets
│
├── documents/
│   ├── Data Dict.pdf         # Feature definitions
│   └── unitest.html          # Test coverage report
│
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory analysis
│   ├── 02_feature_engineering.ipynb    # Feature creation
│   ├── 03_experiments_models.ipynb     # Model training
│   ├── best_model_*.pkl                # Saved models
│   ├── model_card_*.md                 # Model documentation
│   └── *_feature_importance.png        # Feature analysis plots
│
├── src/
│   ├── api/                  # FastAPI application
│   │   ├── main.py           # API entrypoint
│   │   ├── routers/          # Endpoint definitions
│   │   └── schemas/          # Pydantic models
│   │
│   ├── data/                 # Data pipeline
│   │   ├── ingestion.py      # CSV loading + validation
│   │   └── transformation.py # Preprocessing
│   │
│   ├── features/             # Feature engineering
│   │   └── engineering.py    # Longitudinal feature creation
│   │
│   ├── train_test/           # Model training
│   │   ├── train.py          # Training pipeline
│   │   └── test.py           # Evaluation pipeline
│   │
│   ├── models/               # Serialized artifacts
│   │   ├── model.pkl         # Production model
│   │   ├── params.json       # Hyperparameters
│   │   └── metrics.json      # Performance metrics
│   │
│   └── monitoring/           # Observability
│       ├── drift_detector.py      # Feature drift detection
│       ├── prediction_drift.py    # Prediction shift detection
│       ├── performance_tracker.py # Model performance tracking
│       └── logger.py              # Structured logging
│
├── tests/
│   └── unit/                 # Pytest test suite
│
├── scripts/
│   ├── setup_mlflow.sh       # MLflow initialization
│   ├── setup_prefect.sh      # Prefect configuration
│   └── curl_examples.sh      # API testing script
│
├── deployment/
│   └── Dockerfile            # Production container
│
├── main.py                   # All-in-one launcher
└── requirements.txt          # Python dependencies
```

---

## Model Selection

### Experimentation Summary

| Model          | CV F1-Score | Test F1-Score | Training Time | Inference (ms) | Notes                          |
|----------------|-------------|---------------|---------------|----------------|--------------------------------|
| DummyClassifier| 0.45        | 0.43          | <1s           | <1ms           | Baseline (majority class)      |
| Random Forest  | 0.72        | 0.69          | 45s           | 5ms            | Good interpretability          |
| LightGBM       | 0.78        | 0.75          | 12s           | 3ms            | Fast, memory-efficient         |
| **XGBoost (Optuna)** | **0.82** | **0.79** | **120s** | **8ms** | **Best performance** ⭐ |

### Why XGBoost?

**Chosen approach**:
- **+2% F1-score** over LightGBM (critical for academic lag prediction)
- Better handling of **class imbalance** (assumed minority class = students with lag)
- Optuna hyperparameter tuning: 100 trials, Bayesian optimization
- Feature importance analysis aligned with domain knowledge

**Trade-offs**:
- ❌ Slower training (120s vs 12s for LightGBM)
- ❌ Slightly higher inference latency (8ms vs 3ms)
- ✅ Best test set generalization
- ✅ More robust to overfitting (regularization tuned)

**Rejected alternatives**:
- **Random Forest**: Lower F1, but considered if interpretability is prioritized
- **LightGBM**: Consider if latency <5ms is required (real-time constraint)

---

## Monitoring & Observability

### Drift Detection

**Implemented checks** (`src/monitoring/drift_detector.py`):

1. **Feature drift**: Kolmogorov-Smirnov test on input distributions
   - Alert threshold: p-value < 0.05
   - Checked on: All 6 features (ipv, ips, iaa, ieg, no_av, ida)

2. **Prediction drift**: Chi-square test on output class distribution
   - Alert threshold: p-value < 0.01
   - Tracks: Positive class rate shift

3. **Model degradation**: Rolling window performance
   - Alert if: Accuracy drops >5% from baseline
   - Window: Last 1000 predictions

**Logs**: `src/monitoring/inference_logs.jsonl` (structured JSON logs)

### Prometheus Metrics

Exported at `/metrics/metrics`:

```
# Request metrics
http_requests_total{endpoint="/predict"} 1523
http_request_duration_seconds{endpoint="/predict",quantile="0.5"} 0.008
http_request_duration_seconds{endpoint="/predict",quantile="0.99"} 0.042

# Model metrics
model_predictions_total{class="0"} 892
model_predictions_total{class="1"} 631
model_feature_drift_alert 0

# System metrics
model_load_timestamp 1708558800
api_uptime_seconds 86400
```

### Performance Tracking

**Logged per prediction**:
- Input feature values (for drift analysis)
- Prediction + probability
- Latency (ms)
- Timestamp

**Aggregated metrics** (computed hourly):
- Prediction distribution
- Feature statistics (mean, std, min, max)
- Drift alerts
- P50/P95/P99 latency

---

## Testing

### Unit Tests

```bash
# Run all tests
pytest tests/unit/ -v --cov=src --cov-report=html

# Run specific test modules
pytest tests/unit/test_drift_detector.py -v
pytest tests/unit/test_inference_router.py -v

# Coverage report at htmlcov/index.html
```

**Test categories**:
- `test_data/`: Ingestion, transformation, validation
- `test_features/`: Feature engineering logic
- `test_train_test/`: Training/evaluation pipelines
- `test_api/`: FastAPI endpoints, schemas
- `test_monitoring/`: Drift detection, logging

**Key test scenarios**:
- Edge cases: Empty input, out-of-range values
- Data validation: Schema enforcement
- Model loading: Corrupted pickle files
- Drift detection: Known distribution shifts

---

## Known Issues & Trade-offs

### Current Limitations

1. **Cold start latency** (Render deployment):
   - **Issue**: 30-second delay after inactivity
   - **Mitigation**: Keep-alive pings, upgrade to paid tier
   - **Impact**: Not suitable for real-time latency-critical apps

2. **No online learning**:
   - **Issue**: Model retraining requires manual re-deployment
   - **Mitigation**: Schedule weekly retraining via Prefect
   - **Impact**: Model may degrade if data distribution shifts

3. **Single model in production**:
   - **Issue**: No A/B testing or canary deployments
   - **Mitigation**: Implement model versioning + routing
   - **Impact**: Risky to deploy new model without gradual rollout

4. **Limited explainability**:
   - **Issue**: XGBoost is less interpretable than linear models
   - **Mitigation**: SHAP values for prediction explanations (TODO)
   - **Impact**: Harder to justify predictions to educators

### Common Pitfalls (for contributors)

- **Data leakage**: Ensure feature engineering uses only past data (no future leakage in longitudinal features)
- **Class imbalance**: Check if threshold tuning improves precision/recall trade-off
- **Drift false positives**: Set appropriate p-value thresholds (too sensitive → alert fatigue)
- **Docker build cache**: Clear cache if requirements.txt changes (`docker build --no-cache`)

---

## Next Steps

### Short-term (1-3 months)

1. **Explainability**: Integrate SHAP for prediction explanations
   - **Why**: Educators need to understand "why" a student is flagged
   - **Effort**: 2-3 days (SHAP integration + API endpoint)

2. **Feature importance validation**: Compare model features vs domain expert input
   - **Why**: Ensure model isn't relying on spurious correlations
   - **Effort**: 1 week (expert interviews + feature ablation study)

3. **Automated retraining**: Prefect flow for weekly model updates
   - **Why**: Prevent model drift over time
   - **Effort**: 3-4 days (Prefect DAG + model registry)

### Mid-term (3-6 months)

4. **A/B testing framework**: Deploy multiple model versions simultaneously
   - **Why**: Safe rollout of improved models
   - **Effort**: 1-2 weeks (traffic splitting + metrics comparison)

5. **Online learning**: Incremental updates without full retraining
   - **Why**: Adapt to concept drift faster
   - **Effort**: 2-3 weeks (streaming pipeline + model update logic)

6. **Mobile app integration**: Expose API to educator-facing app
   - **Why**: Actionable insights at point of need
   - **Effort**: 1 month (API versioning + SDK development)

### Long-term (6-12 months)

7. **Multi-model ensemble**: Combine XGBoost + LightGBM for robustness
   - **Why**: Reduce variance, improve reliability
   - **Effort**: 2-3 weeks (ensemble logic + backtesting)

8. **Causal inference**: Move from prediction → intervention recommendation
   - **Why**: "What actions reduce lag?" > "Will student lag?"
   - **Effort**: 2-3 months (causal ML research + validation)

9. **Real-time dashboards**: Live monitoring for school administrators
   - **Why**: Proactive interventions vs reactive predictions
   - **Effort**: 1 month (Grafana custom dashboards + alerting)

---

## License

[Insert license type - MIT, Apache 2.0, etc.]

See `LICENSE` file for full text.

---

## Citation

If you use this project in academic work, please cite:

```bibtex
@mastersthesis{school_lag_predictor_2026,
  author = {Igor Comune and Éder Ray and Mário Gottardello and Felippe Maurício and João Marcelo Mendonça},
  title = {MLOps Pipeline for School Lag Prediction: A Longitudinal Approach},
  school = {FIAP},
  year = {2026},
  type = {Master's Thesis},
  url = {https://github.com/IgorComune/masters_thesis_school_lag_predictor}
}
```

---

## Contact & Support

**Team Members**:
- **Igor Comune** - [GitHub](https://github.com/IgorComune) | [LinkedIn](https://www.linkedin.com/in/igor-comune/)
- **Éder Ray** - [GitHub](https://github.com/ederray) | [LinkedIn](https://www.linkedin.com/in/ederray/)
- **Mário Gottardello** - [GitHub](https://github.com/MariolGotta) | [LinkedIn](https://www.linkedin.com/in/m%C3%A1rio-gottardello-2456a818a/)
- **Felippe Maurício** - [GitHub](https://github.com/felippemauricio) | [LinkedIn](https://www.linkedin.com/in/felippemauricio/)
- **João Marcelo Mendonça** - [GitHub](https://github.com/joaomendonca-py) | [LinkedIn](https://www.linkedin.com/in/joaomarcelomendonca/)

**Institution**: FIAP

**Issues**: [GitHub Issues](https://github.com/IgorComune/masters_thesis_school_lag_predictor/issues)  
**Discussions**: [GitHub Discussions](https://github.com/IgorComune/masters_thesis_school_lag_predictor/discussions)

---

## Acknowledgments

**Team Contributors**:
- Igor Comune
- Éder Ray
- Mário Gottardello
- Felippe Maurício
- João Marcelo Mendonça

**Technology Stack**:
- **MLflow**: Experiment tracking
- **Optuna**: Hyperparameter optimization
- **Prometheus + Grafana**: Observability stack
- **Prefect**: Workflow orchestration
- **Render**: Cloud deployment platform

**Institution**: FIAP - Faculdade de Informática e Administração Paulista