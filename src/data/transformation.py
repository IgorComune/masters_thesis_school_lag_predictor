"""
Build a longitudinal dataset by stacking yearly CSV files,
preventing leakage, and generating dataset metadata.

Author: You (now actually acting like a data engineer)
"""

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from prefect import flow, task


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")
DOCUMENTS_DATA_PATH = Path("documents/data")

OUTPUT_CSV = PROCESSED_DATA_PATH / "students_longitudinal.csv"
OUTPUT_METADATA = DOCUMENTS_DATA_PATH / "students_longitudinal.json"

YEAR_FILES = {
    2022: RAW_DATA_PATH / "2022.csv",
    2023: RAW_DATA_PATH / "2023.csv",
    2024: RAW_DATA_PATH / "2024.csv",
}

NUMERIC_COLUMNS = ["idade", "defasagem", "fase"]

LEAKY_COLUMNS = [
    "atingiu_pv",
    "indicado",
    "ponto_de_virada",
    "rec_av1",
    "rec_av2",
    "rec_av3",
    "avaliador1",
    "avaliador2",
    "avaliador3",
]


# ---------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------

@task
def load_csv(path: Path, year: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ano"] = year
    return df


@task
def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("/", "_")
        .str.replace("__+", "_", regex=True)
    )
    return df


@task
def fix_defasagem_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    2022 uses 'defas', other years use 'defasagem'.
    Standardize to 'defasagem'.
    """
    df = df.copy()

    if "defas" in df.columns and "defasagem" not in df.columns:
        df = df.rename(columns={"defas": "defasagem"})

    return df


@task
def collapse_year_specific_columns(
    df: pd.DataFrame,
    base_name: str,
) -> pd.DataFrame:
    df = df.copy()
    matching_cols = [c for c in df.columns if c.startswith(base_name)]

    if not matching_cols:
        return df

    df[base_name] = df[matching_cols].bfill(axis=1).iloc[:, 0]
    return df.drop(columns=matching_cols)


@task
def cast_numeric_columns(
    df: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:
    df = df.copy()

    for col in columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@task
def remove_leaky_columns(
    df: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:
    df = df.copy()
    to_drop = [c for c in columns if c in df.columns]
    return df.drop(columns=to_drop)


@task
def validate_primary_key(df: pd.DataFrame, key: str) -> None:
    if key not in df.columns:
        raise ValueError(f"Primary key '{key}' not found.")


@task
def align_schema(dfs: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
    common_columns = set.intersection(
        *[set(df.columns) for df in dfs.values()]
    )
    common_columns = sorted(common_columns)

    return {
        year: df[common_columns].copy()
        for year, df in dfs.items()
    }


@task
def stack_dataframes(dfs: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(dfs.values(), ignore_index=True)


@task
def export_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


@task
def generate_metadata(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_types": df.dtypes.astype(str).to_dict(),
        "missing_values_ratio": (
            df.isna().mean().round(4).to_dict()
        ),
    }


@task
def export_metadata(metadata: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------

@flow(name="build_longitudinal_student_dataset")
def build_longitudinal_dataset() -> None:
    dataframes: Dict[int, pd.DataFrame] = {}

    for year, path in YEAR_FILES.items():
        df = load_csv(path, year)
        df = normalize_column_names(df)

        df = fix_defasagem_column(df)

        df = collapse_year_specific_columns(df, "inde")
        df = collapse_year_specific_columns(df, "pedra")

        df = cast_numeric_columns(df, NUMERIC_COLUMNS)
        df = remove_leaky_columns(df, LEAKY_COLUMNS)

        validate_primary_key(df, key="ra")

        dataframes[year] = df

    aligned_dfs = align_schema(dataframes)
    final_df = stack_dataframes(aligned_dfs)

    export_csv(final_df, OUTPUT_CSV)

    metadata = generate_metadata(final_df)
    export_metadata(metadata, OUTPUT_METADATA)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    build_longitudinal_dataset()
