from __future__ import annotations

import platform
import subprocess
import threading
from contextlib import contextmanager
from typing import Iterator


ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _set_windows_execution_state(flags: int) -> bool:
    try:
        import ctypes

        return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))
    except (AttributeError, OSError):
        return False


class SystemSleepInhibitor:
    """Keeps macOS or Windows awake while simulation jobs are active."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._holders = 0
        self._process: subprocess.Popen | None = None
        self._windows_inhibited = False

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
            if self._holders != 1:
                return
            system = platform.system()
            if system == "Darwin":
                try:
                    self._process = subprocess.Popen(
                        ["/usr/bin/caffeinate", "-dimsu"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except OSError:
                    self._process = None
            elif system == "Windows":
                self._windows_inhibited = _set_windows_execution_state(
                    ES_CONTINUOUS | ES_SYSTEM_REQUIRED
                )

    def _release(self) -> None:
        with self._lock:
            self._holders = max(0, self._holders - 1)
            if self._holders:
                return
            if self._process is not None:
                process, self._process = self._process, None
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            if self._windows_inhibited:
                _set_windows_execution_state(ES_CONTINUOUS)
                self._windows_inhibited = False


SYSTEM_SLEEP_INHIBITOR = SystemSleepInhibitor()


@contextmanager
def prevent_system_sleep() -> Iterator[None]:
    with SYSTEM_SLEEP_INHIBITOR.hold():
        yield
