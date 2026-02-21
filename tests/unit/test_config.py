from pathlib import Path

import src.api.core.config as config


def test_project_root_is_path():
    assert isinstance(config.PROJECT_ROOT, Path)
    assert config.PROJECT_ROOT.exists()


def test_log_path_construction():
    expected = config.PROJECT_ROOT / "monitoring" / "inference_logs.jsonl"
    assert config.LOG_PATH == expected


def test_api_host_and_port():
    assert config.API_HOST == "0.0.0.0"
    assert config.API_PORT == 8000


def test_model_version():
    assert isinstance(config.MODEL_VERSION, str)
    assert config.MODEL_VERSION == "v1"
