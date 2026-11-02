"""Bounded retries for transient Windows atomic-replace sharing failures.

Only a replacement is retried: bytes already serialized to the temporary file
are left unchanged. This helper never deletes the destination as a fallback.
"""
from __future__ import annotations

import math
import os
import time


def replace_with_retry(source, destination, *, timeout_seconds=2.0,
                       initial_delay_seconds=0.01, maximum_delay_seconds=0.2,
                       replace=None, sleep=None, monotonic=None):
    """Atomically replace, retrying only PermissionError with winerror 5 or 32.

    The timeout bounds retries, not the operating system call itself. On expiry
    the last original exception is raised with its traceback; source and target
    are not removed. Injectable callables support deterministic tests without a
    global monkeypatch or actual sleeping.
    """
    values = (timeout_seconds, initial_delay_seconds, maximum_delay_seconds)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Retry timing must be finite")
    if timeout_seconds < 0 or initial_delay_seconds <= 0 or maximum_delay_seconds < initial_delay_seconds:
        raise ValueError("Invalid retry timing")
    replace = os.replace if replace is None else replace
    sleep = time.sleep if sleep is None else sleep
    monotonic = time.monotonic if monotonic is None else monotonic
    deadline = monotonic() + timeout_seconds
    delay = initial_delay_seconds
    while True:
        try:
            return replace(source, destination)
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32):
                raise
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise
            sleep(min(delay, remaining))
            delay = min(maximum_delay_seconds, delay * 2)
