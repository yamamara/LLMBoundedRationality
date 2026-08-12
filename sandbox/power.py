from __future__ import annotations

import platform
import subprocess
import threading
from contextlib import contextmanager
from typing import Iterator


class SystemSleepInhibitor:
    """Keeps macOS awake while one or more simulation jobs are active."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._holders = 0
        self._process: subprocess.Popen | None = None

    @contextmanager
    def hold(self) -> Iterator[None]:
        self._acquire()
        try:
            yield
        finally:
            self._release()

    def _acquire(self) -> None:
        with self._lock:
            self._holders += 1
            if self._holders == 1 and platform.system() == "Darwin":
                try:
                    self._process = subprocess.Popen(
                        ["/usr/bin/caffeinate", "-dimsu"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except OSError:
                    self._process = None

    def _release(self) -> None:
        with self._lock:
            self._holders = max(0, self._holders - 1)
            if self._holders or self._process is None:
                return
            process, self._process = self._process, None
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)


SYSTEM_SLEEP_INHIBITOR = SystemSleepInhibitor()


@contextmanager
def prevent_system_sleep() -> Iterator[None]:
    with SYSTEM_SLEEP_INHIBITOR.hold():
        yield
