from scripts.start_services import ServiceLauncher
from src.data.ingestion import run_ingestion
from src.data.transformation import build_longitudinal_dataset


if __name__ == "__main__":
    print("[BOOT] Starting services...")

    launcher = ServiceLauncher()
    launcher.start_all()

    print("[OK] MLflow and Prefect are up and responding")

    print("[PIPELINE] Starting data ingestion...")
    run_ingestion()
    print("[OK] Data ingestion completed")

    print("[PIPELINE] Building longitudinal dataset...")
    build_longitudinal_dataset()
    print("[OK] Longitudinal dataset built successfully")

    print("[SYSTEM] All services running. Container is healthy.")
    launcher.keep_alive()
