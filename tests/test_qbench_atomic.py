"""Deterministic retry tests, usable from staging or the merged repository."""
from __future__ import annotations

import errno
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock


source = Path(__file__).resolve().parents[1] / "src/ai4ct/qbench/atomic.py"
spec = importlib.util.spec_from_file_location("staged_qbench_atomic", source)
atomic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(atomic)


def windows_error(code):
    error = PermissionError(errno.EACCES, "simulated transient Windows lock")
    error.winerror = code
    return error


class FakeClock:
    def __init__(self):
        self.elapsed = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.elapsed

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.elapsed += seconds


class AtomicReplaceTests(unittest.TestCase):
    def test_two_transient_errors_then_success(self):
        clock = FakeClock()
        replace = Mock(side_effect=[windows_error(5), windows_error(32), None])
        atomic.replace_with_retry("from.tmp", "to.json", replace=replace,
                                  sleep=clock.sleep, monotonic=clock.monotonic)
        self.assertEqual(replace.call_count, 3)
        self.assertEqual(clock.sleeps, [.01, .02])
        self.assertTrue(all(call.args == ("from.tmp", "to.json") for call in replace.call_args_list))

    def test_permanent_lock_raises_original_exception_after_bounded_delay(self):
        clock = FakeClock()
        error = windows_error(5)
        replace = Mock(side_effect=error)
        with self.assertRaises(PermissionError) as raised:
            atomic.replace_with_retry("from.tmp", "to.pt", timeout_seconds=.25,
                                      initial_delay_seconds=.1, maximum_delay_seconds=.2,
                                      replace=replace, sleep=clock.sleep, monotonic=clock.monotonic)
        self.assertIs(raised.exception, error)
        self.assertAlmostEqual(sum(clock.sleeps), .25)
        self.assertEqual(replace.call_count, 3)

    def test_other_errors_are_never_retried(self):
        errors = [OSError(errno.ENOSPC, "disk full"), FileNotFoundError(errno.ENOENT, "missing source"),
                  PermissionError(errno.EACCES, "non-Windows permission"), windows_error(19)]
        for error in errors:
            with self.subTest(error=repr(error)):
                clock = FakeClock()
                replace = Mock(side_effect=error)
                with self.assertRaises(type(error)) as raised:
                    atomic.replace_with_retry("from.tmp", "to.json", replace=replace,
                                              sleep=clock.sleep, monotonic=clock.monotonic)
                self.assertIs(raised.exception, error)
                replace.assert_called_once()
                self.assertEqual(clock.sleeps, [])

    def test_zero_timeout_attempts_once(self):
        clock = FakeClock()
        replace = Mock(side_effect=windows_error(32))
        with self.assertRaises(PermissionError):
            atomic.replace_with_retry("from.tmp", "to.json", timeout_seconds=0,
                                      replace=replace, sleep=clock.sleep, monotonic=clock.monotonic)
        replace.assert_called_once()
        self.assertEqual(clock.sleeps, [])

    def test_success_does_not_sleep(self):
        clock = FakeClock()
        replace = Mock(return_value=None)
        atomic.replace_with_retry("from.tmp", "to.json", replace=replace,
                                  sleep=clock.sleep, monotonic=clock.monotonic)
        replace.assert_called_once()
        self.assertEqual(clock.sleeps, [])

    def test_invalid_budget_rejected_before_replacing(self):
        for value in (-1, float("inf"), float("nan")):
            replace = Mock()
            with self.assertRaises(ValueError):
                atomic.replace_with_retry("from.tmp", "to.json", timeout_seconds=value, replace=replace)
            replace.assert_not_called()


if __name__ == "__main__":
    unittest.main()
