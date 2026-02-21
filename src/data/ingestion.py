"""
Data ingestion flow.

Downloads public Google Sheets tabs (by GID) as CSV files,
stores them in data/raw, and generates metadata files
containing schema and shape information.
"""

from pathlib import Path
from typing import Dict

import json
import requests
import pandas as pd
from prefect import flow, task, get_run_logger


SPREADSHEET_ID = "1td91KoeSgXrUrCVOUkLmONG9Go3LVcXpcNEw_XrL2R0"
BASE_EXPORT_URL = (
    "https://docs.google.com/spreadsheets/d/"
    f"{SPREADSHEET_ID}/export"
)

RAW_DATA_DIR = Path("data/raw")
METADATA_DIR = Path("documents/data")

SHEETS: Dict[int, int] = {
    2022: 90992733,
    2023: 555005642,
    2024: 215885893,
}


def build_export_url(gid: int) -> str:
    """
    Build the Google Sheets CSV export URL for a given GID.
    """
    return f"{BASE_EXPORT_URL}?format=csv&gid={gid}"


@task(
    name="Download Google Sheets tab",
    retries=3,
    retry_delay_seconds=5,
)
def download_sheet(year: int, gid: int) -> Path:
    """
    Download a single Google Sheets tab and save it as a CSV file.

    Returns:
        Path: Path to the downloaded CSV file.
    """
    logger = get_run_logger()

    url = build_export_url(gid)
    output_path = RAW_DATA_DIR / f"{year}.csv"

    logger.info("Downloading data for year %s", year)

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    output_path.write_bytes(response.content)

    logger.info("Saved raw file to %s", output_path.resolve())

    return output_path


@task(name="Generate dataset metadata")
def generate_metadata(csv_path: Path) -> None:
    """
    Generate metadata (shape and schema) for a CSV dataset.
    """
    logger = get_run_logger()

    df = pd.read_csv(csv_path)

    metadata = {
        "file_name": csv_path.name,
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "columns": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },
    }

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = METADATA_DIR / f"{csv_path.stem}_metadata.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Metadata saved to %s", output_path.resolve())


@flow(name="School Lag Data Ingestion")
def run_ingestion() -> None:
    """
    Orchestrate data ingestion and metadata generation.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for year, gid in SHEETS.items():
        csv_path = download_sheet(year=year, gid=gid)
        generate_metadata(csv_path)


# if __name__ == "__main__":
#     run_ingestion()
