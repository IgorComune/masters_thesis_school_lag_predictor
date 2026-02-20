import json
import pytest
from pathlib import Path
from unittest.mock import patch

import src.monitoring.your_module_name as module  # ajuste aqui


# ==========================================================
# helper
# ==========================================================

def write_jsonl(path: Path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ==========================================================
# caso: sem dados rotulados
# ==========================================================

def test_evaluate_performance_no_labeled_data(tmp_path, capsys):
    fake_path = tmp_path / "logs.jsonl"

    records = [
        {"prediction": 0.8, "true_label": None},
        {"prediction": 0.3},  # sem true_label
    ]

    write_jsonl(fake_path, records)

    with patch.object(module, "LOG_PATH", str(fake_path)):
        module.evaluate_performance()

    captured = capsys.readouterr()
    assert "No labeled data yet." in captured.out


# ==========================================================
# caso: métricas corretas
# ==========================================================

def test_evaluate_performance_with_data(tmp_path, capsys):
    fake_path = tmp_path / "logs.jsonl"

    records = [
        {"prediction": 0.9, "true_label": 1},
        {"prediction": 0.8, "true_label": 1},
        {"prediction": 0.2, "true_label": 0},
        {"prediction": 0.1, "true_label": 0},
    ]

    write_jsonl(fake_path, records)

    with patch.object(module, "LOG_PATH", str(fake_path)):
        module.evaluate_performance()

    captured = capsys.readouterr().out

    assert "Accuracy:" in captured
    assert "Precision:" in captured
    assert "Recall:" in captured
    assert "ROC AUC:" in captured


# ==========================================================
# caso: threshold funcionando (> 0.5)
# ==========================================================

def test_evaluate_performance_threshold_logic(tmp_path, capsys):
    fake_path = tmp_path / "logs.jsonl"

    records = [
        {"prediction": 0.6, "true_label": 1},
        {"prediction": 0.4, "true_label": 0},
    ]

    write_jsonl(fake_path, records)

    with patch.object(module, "LOG_PATH", str(fake_path)):
        module.evaluate_performance()

    captured = capsys.readouterr().out

    # ambos devem ser classificados corretamente
    assert "Accuracy: 1.0" in captured
