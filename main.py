import subprocess
import signal
import sys
import time
import socket
import requests

from src.data.ingestion import run_ingestion
from src.data.transformation import build_longitudinal_dataset
from src.train_test import train, test

processes = []


# ==============================
# UTILITÁRIOS
# ==============================

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def wait_for_http(url: str, timeout: int = 60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=5)
            if 200 <= r.status_code < 300:
                return
        except Exception:
            pass
        time.sleep(2)

    raise RuntimeError(f"Serviço {url} não respondeu no tempo esperado.")


def start_process(cmd):
    p = subprocess.Popen(cmd)
    processes.append(p)
    return p


def shutdown(signum=None, frame=None):
    print("\nEncerrando todos os serviços...")
    for p in processes:
        if p.poll() is None:
            p.terminate()

    for p in processes:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()

    sys.exit(0)


# ==============================
# INFRA
# ==============================

def start_mlflow():
    if is_port_in_use(5000):
        print("MLflow já está rodando na porta 5000.")
        return

    print("🔧 Iniciando MLflow...")
    start_process([
        "mlflow", "server",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--default-artifact-root", "./mlruns",
        "--host", "0.0.0.0",
        "--port", "5000"
    ])

    wait_for_http("http://localhost:5000")
    print("🚀 MLflow disponível em http://localhost:5000")


def start_prefect():
    if is_port_in_use(4200):
        print("Prefect já está rodando na porta 4200.")
        return

    print("🔧 Iniciando Prefect...")
    start_process([
        "prefect", "server", "start",
        "--host", "0.0.0.0",
        "--port", "4200"
    ])

    wait_for_http("http://localhost:4200")
    print("🚀 Prefect disponível em http://localhost:4200")


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("Subindo infraestrutura...")
    start_mlflow()
    start_prefect()
    print("Infraestrutura pronta.\n")

    try:
        print("Downloading data")
        run_ingestion()
        print("[OK] Data ingestion completed")

        print("[PIPELINE] Building longitudinal dataset...")
        build_longitudinal_dataset()
        print("[OK] Longitudinal dataset built successfully")

        print("\n--------------------")
        print("Iniciando treinamento")
        print("--------------------")
        train.main()

        print("\n--------------------")
        print("Iniciando teste")
        print("--------------------")
        test.main()

    except Exception as e:
        print("Erro na pipeline:", e)

    print("\nSubindo API, Prometheus e Grafana...")

    start_process([
        "uvicorn",
        "src.api.main:app",
        "--reload",
        "--port", "8000"
    ])

    start_process([
        "prometheus-2.45.0.linux-amd64/prometheus",
        "--config.file=prometheus-2.45.0.linux-amd64/prometheus.yml"
    ])

    start_process([
        "grafana-10.1.0/bin/grafana-server"
    ])

    print("Serviços ativos.")
    signal.pause()