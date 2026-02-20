import pandas as pd
import pytest

from src.your_module import LineAverager  # ajuste o import


# ==========================================================
# caso básico
# ==========================================================

def test_apply_basic_average():
    df = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4]
    })

    averager = LineAverager()
    result = averager.apply(df)

    assert "media" in result.columns
    assert result["media"].tolist() == [2.0, 3.0]


# ==========================================================
# múltiplas colunas
# ==========================================================

def test_apply_multiple_columns():
    df = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4],
        "c": [5, 6]
    })

    averager = LineAverager()
    result = averager.apply(df)

    assert result["media"].tolist() == [3.0, 4.0]


# ==========================================================
# target_col customizado
# ==========================================================

def test_apply_custom_target_column():
    df = pd.DataFrame({
        "x": [10, 20],
        "y": [30, 40]
    })

    averager = LineAverager(target_col="avg")
    result = averager.apply(df)

    assert "avg" in result.columns
    assert result["avg"].tolist() == [20.0, 30.0]


# ==========================================================
# comportamento com NaN (pandas ignora por padrão)
# ==========================================================

def test_apply_with_nan():
    df = pd.DataFrame({
        "a": [1, None],
        "b": [3, 4]
    })

    averager = LineAverager()
    result = averager.apply(df)

    # linha 1: (1+3)/2 = 2
    # linha 2: pandas ignora None → média = 4
    assert result["media"].tolist() == [2.0, 4.0]


# ==========================================================
# dataframe vazio
# ==========================================================

def test_apply_empty_dataframe():
    df = pd.DataFrame()

    averager = LineAverager()
    result = averager.apply(df)

    assert "media" in result.columns
    assert result["media"].empty
