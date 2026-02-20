from unittest.mock import patch
import pandas as pd
import pytest

from src.data.transformation import build_longitudinal_dataset, YEAR_FILES

# mocks de todas as tarefas que fazem I/O ou dependem de serviços externos
@patch("src.data.transformation.load_csv")
@patch("src.data.transformation.normalize_column_names")
@patch("src.data.transformation.fix_defasagem_column")
@patch("src.data.transformation.collapse_year_specific_columns")
@patch("src.data.transformation.cast_numeric_columns")
@patch("src.data.transformation.remove_leaky_columns")
@patch("src.data.transformation.validate_primary_key")
@patch("src.data.transformation.align_schema")
@patch("src.data.transformation.stack_dataframes")
@patch("src.data.transformation.export_csv")
@patch("src.data.transformation.generate_metadata")
@patch("src.data.transformation.export_metadata")
def test_build_longitudinal_dataset_safe(
    mock_export_meta,
    mock_generate_meta,
    mock_export_csv,
    mock_stack,
    mock_align,
    mock_validate,
    mock_remove,
    mock_cast,
    mock_collapse,
    mock_fix,
    mock_norm,
    mock_load,
):
    # dataframe fake para todas as tarefas
    fake_df = pd.DataFrame({"ra": [1]})
    
    mock_load.return_value = fake_df
    mock_norm.return_value = fake_df
    mock_fix.return_value = fake_df
    mock_collapse.return_value = fake_df
    mock_cast.return_value = fake_df
    mock_remove.return_value = fake_df
    mock_align.return_value = {year: fake_df for year in YEAR_FILES}
    mock_stack.return_value = fake_df
    mock_generate_meta.return_value = {"rows": 1, "columns": 1}

    # chama o fluxo - nenhum I/O real será executado
    build_longitudinal_dataset()

    # checagens básicas
    assert mock_load.call_count == len(YEAR_FILES)
    assert mock_export_csv.called
    assert mock_export_meta.called
