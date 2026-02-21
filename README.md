# Master's Thesis - MLOps - School Lag Predictor

## Installation for Local Deployment and/or Development

## Python 3.12
`python3 -m venv .venv`
`source .venv/bin/activate`
`pip install -r requirements.txt`


## Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
tar xvf prometheus-2.45.0.linux-amd64.tar.gz
rm -f prometheus-2.45.0.linux-amd64.tar.gz

### Edit .yml
#### scrape configs './prometheus --config.file=prometheus.yml'

´´´
scrape_configs:
  # Prometheus se monitorando
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]

  # sua FastAPI
  - job_name: "fastapi"
    metrics_path: "/metrics/metrics" 
    static_configs:
      - targets: ["127.0.0.1:8000"]
´´´



## grafana
wget https://dl.grafana.com/oss/release/grafana-10.1.0.linux-amd64.tar.gz
tar -zxvf grafana-10.1.0.linux-amd64.tar.gz
rm -f grafana-10.1.0.linux-amd64.tar.gz


user: admin
psw: admin

# For local deployment
## Main flow
Execute ./main.py #it will lauch the API, Prometheus and Grafana
Execute scripts/curl_examples.sh to send data to the prediction endpoint and get the output on the terminal

Grafana UI disponível em: http://localhost:3000/ # usado para dashboards
Prometheus UI disponível em http://localhost:9090 # usado para verificar o ETL
Prefect UI disponível em http://localhost:4200 # usado para verificar o ETL
MLflow UI disponível em http://localhost:5000 # usado para o treino/teste na pasta "notebooks", não é usado no script principal.


Deploy local docker

docker build -t xgboost-api .
docker run -p 10000:10000 xgboost-api

## Test docker deploy
curl -X POST http://127.0.0.1:10000/inference/predict \
    -H "Content-Type: application/json" \
    -d '{"ipv": 1, "ips": 2, "iaa": 3, "ieg": 4, "no_av": 5, "ida": 6}'


# Cloud (Render)