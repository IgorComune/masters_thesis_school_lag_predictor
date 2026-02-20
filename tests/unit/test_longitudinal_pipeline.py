import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

import src.your_module as module  # <-- ajuste aqui


# ---------------------------------------------------------------------
# Tasks isoladas
# ---------------------------------------------------------------------

def test_normalize_column_names():
    df = pd.DataFrame(columns=[" Nome ", "Idade/Aluno", "DEFAS  "])
    result = module.normalize_column_names.fn(df)

    assert "nome" in result.columns
    assert "idade_aluno" in result.columns
    assert "defas" in result.columns


def test_fix_defasagem_column():
    df = pd.DataFrame({"defas": [1, 2]})
    result = module.fix_defasagem_column.fn(df)

    assert "defasagem" in result.columns
    assert "defas" not in result.columns


def test_collapse_year_specific_columns():
    df = pd.DataFrame({
        "inde_2022": [None, 5],
        "inde_2023": [3, None],
        "ra": [1, 2],
    })

    result = module.collapse_year_specific_columns.fn(df, "inde")

    assert "inde" in result.columns
    assert "inde_2022" not in result.columns
    assert result["inde"].tolist() == [3, 5]


def test_cast_numeric_columns():
    df = pd.DataFrame({
        "idade": ["10", "11"],
        "fase": ["1,5", "2,5"],
    })

    result = module.cast_numeric_columns.fn(df, ["idade", "fase"])

    assert result["idade"].dtype != object
    assert result["fase"].iloc[0] == 1.5


def test_remove_leaky_columns():
    df = pd.DataFrame({
        "ra": [1],
        "atingiu_pv": [1],
        "avaliador1": [2],
    })

    result = module.remove_leaky_columns.fn(df, module.LEAKY_COLUMNS)

    assert "atingiu_pv" not in result.columns
    assert "avaliador1" not in result.columns
    assert "ra" in result.columns


def test_validate_primary_key_raises():
    df = pd.DataFrame({"x": [1]})

    with pytest.raises(ValueError):
        module.validate_primary_key.fn(df, key="ra")


def test_align_schema():
    df1 = pd.DataFrame({"ra": [1], "a": [1]})
    df2 = pd.DataFrame({"ra": [2], "b": [2]})

    aligned = module.align_schema.fn({2022: df1, 2023: df2})

    for df in aligned.values():
        assert list(df.columns) == ["ra"]


def test_stack_dataframes():
    df1 = pd.DataFrame({"ra": [1]})
    df2 = pd.DataFrame({"ra": [2]})

    result = module.stack_dataframes.fn({2022: df1, 2023: df2})

    assert len(result) == 2


def test_generate_metadata():
    df = pd.DataFrame({
        "a": [1, None],
        "b": [2, 3],
    })

    metadata = module.generate_metadata.fn(df)

    assert metadata["rows"] == 2
    assert metadata["columns"] == 2
    assert "a" in metadata["column_types"]
    assert "a" in metadata["missing_values_ratio"]


# ---------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------

@patch("src.your_module.export_metadata")
@patch("src.your_module.generate_metadata")
@patch("src.your_module.export_csv")
@patch("src.your_module.stack_dataframes")
@patch("src.your_module.align_schema")
@patch("src.your_module.validate_primary_key")
@patch("src.your_module.remove_leaky_columns")
@patch("src.your_module.cast_numeric_columns")
@patch("src.your_module.collapse_year_specific_columns")
@patch("src.your_module.fix_defasagem_column")
@patch("src.your_module.normalize_column_names")
@patch("src.your_module.load_csv")
def test_build_longitudinal_dataset(
    mock_load,
    mock_norm,
    mock_fix,
    mock_collapse,
    mock_cast,
    mock_remove,
    mock_validate,
    mock_align,
    mock_stack,
    mock_export_csv,
    mock_generate_meta,
    mock_export_meta,
):
    fake_df = pd.DataFrame({"ra": [1]})

    mock_load.return_value = fake_df
    mock_norm.return_value = fake_df
    mock_fix.return_value = fake_df
    mock_collapse.return_value = fake_df
    mock_cast.return_value = fake_df
    mock_remove.return_value = fake_df
    mock_align.return_value = {2022: fake_df}
    mock_stack.return_value = fake_df
    mock_generate_meta.return_value = {"rows": 1}

    module.build_longitudinal_dataset()

    assert mock_load.call_count == len(module.YEAR_FILES)
    assert mock_export_csv.called
    assert mock_export_meta.called
