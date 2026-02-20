import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd

import src.data.ingestion as run_ingestion 


def test_build_export_url():
    gid = 123
    url = module.build_export_url(gid)

    assert "format=csv" in url
    assert f"gid={gid}" in url
    assert module.SPREADSHEET_ID in url


@patch("src.your_module.requests.get")
@patch("src.your_module.get_run_logger")
def test_download_sheet(mock_logger, mock_get, tmp_path):
    fake_content = b"col1,col2\n1,2\n"

    mock_response = MagicMock()
    mock_response.content = fake_content
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    with patch.object(module, "RAW_DATA_DIR", tmp_path):
        output_path = module.download_sheet.fn(year=2022, gid=999)

    assert output_path.exists()
    assert output_path.read_bytes() == fake_content


@patch("src.your_module.get_run_logger")
def test_generate_metadata(mock_logger, tmp_path):
    csv_path = tmp_path / "2022.csv"

    df = pd.DataFrame({
        "a": [1, 2],
        "b": ["x", "y"]
    })
    df.to_csv(csv_path, index=False)

    with patch.object(module, "METADATA_DIR", tmp_path):
        module.generate_metadata.fn(csv_path)

    metadata_file = tmp_path / "2022_metadata.json"
    assert metadata_file.exists()

    metadata = json.loads(metadata_file.read_text())

    assert metadata["file_name"] == "2022.csv"
    assert metadata["num_rows"] == 2
    assert metadata["num_columns"] == 2
    assert "a" in metadata["columns"]
    assert "b" in metadata["columns"]


@patch("src.your_module.generate_metadata")
@patch("src.your_module.download_sheet")
def test_run_ingestion(mock_download, mock_metadata, tmp_path):
    fake_csv = tmp_path / "file.csv"
    fake_csv.write_text("a,b\n1,2")

    mock_download.return_value = fake_csv

    with patch.object(module, "RAW_DATA_DIR", tmp_path):
        module.run_ingestion()

    assert mock_download.call_count == len(module.SHEETS)
    assert mock_metadata.call_count == len(module.SHEETS)
