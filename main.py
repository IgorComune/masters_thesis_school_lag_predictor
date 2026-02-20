import uvicorn
import subprocess
import signal
import sys

from src.data.ingestion import run_ingestion
from src.data.transformation import build_longitudinal_dataset
from src.train_test import train, test
from scripts.start_services import ServiceLauncher


processes = []

def start_process(cmd):
    p = subprocess.Popen(cmd)
    processes.append(p)

def shutdown():
    print("\nEncerrando todos os serviços...")
    for p in processes:
        p.terminate()
    for p in processes:
        p.wait()
    sys.exit(0)


if __name__ == "__main__":
        # ==============================
    # SUBINDO SERVIÇOS PRIMEIRO
    # ==============================
    print('Iniciando Prefect e MLFlow')
    service_launcher = ServiceLauncher()
    service_launcher.start_all()
    print("Infrastructure ready.")

    try:
        print("Downloading data")
        run_ingestion()
        print("[OK] Data ingestion completed")

        print("[PIPELINE] Building longitudinal dataset...")
        build_longitudinal_dataset()
        print("[OK] Longitudinal dataset built successfully")

    except Exception as e:
        print("Erro no download:", e)

    print('--------------------')
    print('Iniciando treinamento')
    print('--------------------')
    train.main()

    print('Treinamento concluído')
    print('--------------------')

    print('Iniciando teste')
    print('--------------------')
    test.main()

    print('--------------------')
    print('Teste concluído')
    print('--------------------')

    # ==============================
    # INICIANDO SERVIÇOS
    # ==============================

    signal.signal(signal.SIGINT, lambda s, f: shutdown())
    signal.signal(signal.SIGTERM, lambda s, f: shutdown())

    print("Subindo API, Prometheus e Grafana...")

    start_process([
        "uvicorn",
        "src.api.main:app",
        "--reload",
        "--port", "8000"
    ])

    start_process([
        "prometheus-2.45.0.linux-amd64/prometheus",
        "--config.file=prometheus.yml"
    ])

    start_process([
        "grafana-10.1.0/bin/grafana-server"
    ])

    print("Serviços ativos.")
    signal.pause()
