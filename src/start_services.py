import signal
import subprocess
import sys
import time
from typing import List, Optional

import requests
from requests.exceptions import RequestException


class ServiceLauncher:
    def __init__(self) -> None:
        self.processes: List[subprocess.Popen] = []

    @staticmethod
    def _wait_for_http_service(
        url: str,
        timeout: int = 60,
        interval: int = 2,
        expected_status: Optional[int] = None,
    ) -> None:
        """
        Wait until an HTTP service becomes available.

        Args:
            url: Endpoint to check.
            timeout: Max seconds to wait.
            interval: Seconds between retries.
            expected_status: If provided, require this exact status code.

        Raises:
            RuntimeError: If the service does not become available in time.
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)

                if expected_status is not None:
                    if response.status_code == expected_status:
                        return
                else:
                    if response.status_code < 500:
                        return

            except RequestException:
                pass

            time.sleep(interval)

        raise RuntimeError(
            f"Service at {url} did not become available within {timeout} seconds"
        )

    def start_mlflow(self) -> None:
        process = subprocess.Popen(
            ["bash", "scripts/setup_mlflow.sh"],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        self.processes.append(process)

        # MLflow UI health check
        self._wait_for_http_service(
            url="http://localhost:5000",
            timeout=60,
            interval=2,
        )

    def start_prefect(self) -> None:
        process = subprocess.Popen(
            ["bash", "scripts/setup_prefect.sh"],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        self.processes.append(process)

        # Prefect UI health check
        self._wait_for_http_service(
            url="http://localhost:4200",
            timeout=90,
            interval=3,
        )

    def start_all(self) -> None:
        self.start_mlflow()
        self.start_prefect()

        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT, self._shutdown)

    def _shutdown(self, signum, frame) -> None:
        print(f"Shutting down services (signal={signum})")

        for process in self.processes:
            if process.poll() is None:
                process.terminate()

        for process in self.processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

        sys.exit(0)

    def keep_alive(self) -> None:
        while True:
            time.sleep(1)
