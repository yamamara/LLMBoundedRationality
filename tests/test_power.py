from __future__ import annotations

import subprocess
import unittest
from unittest.mock import Mock, patch

from sandbox.power import SystemSleepInhibitor


class SystemSleepInhibitorTests(unittest.TestCase):
    @patch("sandbox.power.platform.system", return_value="Darwin")
    @patch("sandbox.power.subprocess.Popen")
    def test_nested_jobs_share_one_caffeinate_process(self, popen, _system):
        process = Mock()
        popen.return_value = process
        inhibitor = SystemSleepInhibitor()

        with inhibitor.hold():
            with inhibitor.hold():
                popen.assert_called_once_with(
                    ["/usr/bin/caffeinate", "-dimsu"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            process.terminate.assert_not_called()

        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=2)

    @patch("sandbox.power.platform.system", return_value="Linux")
    @patch("sandbox.power.subprocess.Popen")
    def test_non_macos_jobs_do_not_start_caffeinate(self, popen, _system):
        with SystemSleepInhibitor().hold():
            pass
        popen.assert_not_called()

    @patch("sandbox.power._set_windows_execution_state", return_value=True)
    @patch("sandbox.power.platform.system", return_value="Windows")
    def test_windows_jobs_set_and_restore_execution_state(self, _system, set_state):
        inhibitor = SystemSleepInhibitor()

        with inhibitor.hold():
            with inhibitor.hold():
                pass

        self.assertEqual(set_state.call_count, 2)
        self.assertEqual(set_state.call_args_list[0].args[0], 0x80000001)
        self.assertEqual(set_state.call_args_list[1].args[0], 0x80000000)


if __name__ == "__main__":
    unittest.main()
