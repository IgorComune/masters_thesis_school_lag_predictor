import json
import pytest
from pathlib import Path
from unittest.mock import patch

import src.monitoring.your_module_name as module  # ajuste aqui


# ==========================================================
# helper para criar arquivo temporário JSONL
# ==========================================================

def write_jsonl(path: Path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ==========================================================
# caso: arquivo não existe
# ==========================================================

def test_health_report_no_file(tmp_path):
    fake_path = tmp_path / "missing.jsonl"

    with patch.object(module, "LOG_PATH", fake_path):
        result = module.health_report()

    assert result["total_requests"] == 0
    assert result["average_latency"] == 0.0
    assert result["max_latency"] == 0.0
    assert result["last_5_requests"] == []


# ==========================================================
# caso: arquivo com registros válidos
# ==========================================================

def test_health_report_with_data(tmp_path):
    fake_path = tmp_path / "logs.jsonl"

    records = [
        {"latency": 0.1},
        {"latency": 0.3},
        {"latency": 0.2},
    ]

    write_jsonl(fake_path, records)

    with patch.object(module, "LOG_PATH", fake_path):
        result = module.health_report()

    assert result["total_requests"] == 3
    assert result["average_latency"] == pytest.approx(0.2)
    assert result["max_latency"] == 0.3
    assert len(result["last_5_requests"]) == 3


# ==========================================================
# caso: latência com NaN e inf
# ==========================================================

def test_health_report_with_invalid_latency(tmp_path):
    fake_path = tmp_path / "logs.jsonl"

    records = [
        {"latency": float("nan")},
        {"latency": float("inf")},
        {"latency": 0.5},
    ]

    write_jsonl(fake_path, records)

    with patch.object(module, "LOG_PATH", fake_path):
        result = module.health_report()

    assert result["total_requests"] == 3
    assert result["max_latency"] == 0.5
    assert result["average_latency"] == pytest.approx(0.5)


# ==========================================================
# caso: mais de 5 registros → last_5 limitado
# ==========================================================

def test_health_report_last_5(tmp_path):
    fake_path = tmp_path / "logs.jsonl"

    records = [{"latency": i} for i in range(10)]

    write_jsonl(fake_path, records)

    with patch.object(module, "LOG_PATH", fake_path):
        result = module.health_report()

    assert result["total_requests"] == 10
    assert len(result["last_5_requests"]) == 5
    assert result["last_5_requests"][0]["latency"] == 5
    assert result["last_5_requests"][-1]["latency"] == 9


# ==========================================================
# sanitize_for_json
# ==========================================================

def test_sanitize_for_json():
    data = {
        "a": float("nan"),
        "b": [1, float("inf"), {"c": float("nan")}],
    }

    result = module.sanitize_for_json(data)

    assert result["a"] is None
    assert result["b"][1] is None
    assert result["b"][2]["c"] is None
