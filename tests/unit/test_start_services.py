import pytest
from unittest.mock import patch, MagicMock
from requests.exceptions import RequestException

from scripts.start_services import ServiceLauncher 


# ----------------------------
# _wait_for_http_service
# ----------------------------

@patch("your_module.requests.get")
@patch("your_module.time.sleep", return_value=None)
def test_wait_for_http_service_success(mock_sleep, mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    ServiceLauncher._wait_for_http_service(
        url="http://fake-service",
        timeout=5,
        interval=1,
    )


@patch("your_module.requests.get")
@patch("your_module.time.sleep", return_value=None)
def test_wait_for_http_service_expected_status(mock_sleep, mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_get.return_value = mock_response

    ServiceLauncher._wait_for_http_service(
        url="http://fake-service",
        timeout=5,
        interval=1,
        expected_status=201,
    )


@patch("your_module.requests.get", side_effect=RequestException)
@patch("your_module.time.sleep", return_value=None)
def test_wait_for_http_service_timeout(mock_sleep, mock_get):
    with pytest.raises(RuntimeError):
        ServiceLauncher._wait_for_http_service(
            url="http://fake-service",
            timeout=1,
            interval=0,
        )


# ----------------------------
# start_mlflow / start_prefect
# ----------------------------

@patch("your_module.subprocess.Popen")
@patch.object(ServiceLauncher, "_wait_for_http_service")
def test_start_mlflow(mock_wait, mock_popen):
    mock_process = MagicMock()
    mock_popen.return_value = mock_process

    launcher = ServiceLauncher()
    launcher.start_mlflow()

    mock_popen.assert_called_once()
    mock_wait.assert_called_once()
    assert launcher.processes[0] == mock_process


@patch("your_module.subprocess.Popen")
@patch.object(ServiceLauncher, "_wait_for_http_service")
def test_start_prefect(mock_wait, mock_popen):
    mock_process = MagicMock()
    mock_popen.return_value = mock_process

    launcher = ServiceLauncher()
    launcher.start_prefect()

    mock_popen.assert_called_once()
    mock_wait.assert_called_once()
    assert launcher.processes[0] == mock_process


# ----------------------------
# _shutdown
# ----------------------------

@patch("your_module.sys.exit")
def test_shutdown_terminates_processes(mock_exit):
    launcher = ServiceLauncher()

    mock_process = MagicMock()
    mock_process.poll.return_value = None
    launcher.processes.append(mock_process)

    launcher._shutdown(signum=15, frame=None)

    mock_process.terminate.assert_called_once()
    mock_process.wait.assert_called()
    mock_exit.assert_called_once_with(0)
