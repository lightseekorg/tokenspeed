# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""CPU-only coverage of log I/O, independent of model/runtime dependencies."""

import contextlib
import io
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tokenspeed._logging import (
    AsyncStreamHandler,
    configure_root_logger,
    stop_async_logging,
)


class BlockingStream(io.StringIO):
    def __init__(self, *, stall_in: str, stalled_calls: set[int]):
        super().__init__()
        self.stall_in = stall_in
        self.stalled_calls = stalled_calls
        self.calls = 0
        self.entered = queue.Queue()
        self.release = threading.Semaphore(0)

    def _stall(self, operation):
        if operation != self.stall_in:
            return
        self.calls += 1
        if self.calls in self.stalled_calls:
            self.entered.put(self.calls)
            if not self.release.acquire(timeout=5):
                raise TimeoutError("test did not release the blocked stream")

    def write(self, text):
        self._stall("write")
        return super().write(text)

    def flush(self):
        self._stall("flush")
        super().flush()


@pytest.fixture
def root_logger():
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    # basicConfig(force=True) must not close pytest's own capture handlers.
    root.handlers.clear()
    try:
        yield root
    finally:
        stop_async_logging()
        for handler in root.handlers:
            handler.close()
        root.handlers[:] = handlers
        root.setLevel(level)


def configure(stream, enabled):
    configure_root_logger(
        level=logging.INFO,
        log_format="%(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=stream,
        enable_async_logging=enabled,
    )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("stall_in", ["write", "flush"])
def test_intermittent_stalls_only_block_synchronous_producers(
    root_logger, enabled, stall_in
):
    stream = BlockingStream(stall_in=stall_in, stalled_calls={1, 7, 14})
    configure(stream, enabled)
    done = threading.Event()

    def produce():
        for i in range(20):
            root_logger.info(f"record {i}")
        done.set()

    producer = threading.Thread(target=produce)
    producer.start()
    try:
        for expected in (1, 7, 14):
            assert stream.entered.get(timeout=5) == expected
            if enabled:
                assert done.wait(1), "async producer waited on the output stream"
            else:
                assert not done.is_set(), "synchronous output returned before its write"
            stream.release.release()
        assert done.wait(5)
    finally:
        stream.release.release(3)
        producer.join(timeout=5)
        stop_async_logging()
    assert stream.getvalue().splitlines() == [f"INFO record {i}" for i in range(20)]


def test_full_queue_drops_new_records_and_can_be_closed_while_full():
    stream = BlockingStream(stall_in="write", stalled_calls={1})
    handler = AsyncStreamHandler(stream, capacity=2, shutdown_timeout=2)
    logger = logging.Logger("bounded", logging.INFO)
    logger.addHandler(handler)
    errors = []
    closed = threading.Event()

    class NoFallback(io.StringIO):
        def write(self, text):
            raise AssertionError(f"overflow wrote synchronously to stderr: {text}")

    def close():
        try:
            handler.close()
        except Exception as error:
            errors.append(error)
        finally:
            closed.set()

    closer = threading.Thread(target=close)
    try:
        logger.info("active")
        stream.entered.get(timeout=5)
        with contextlib.redirect_stderr(NoFallback()):
            for i in range(100):
                logger.info(f"queued {i}")
        assert handler.queue.qsize() == 2
        assert handler.dropped_records == 98
        closer.start()
        assert not closed.wait(0.05), "close returned before queued records drained"
        stream.release.release()
        assert closed.wait(3)
        assert not errors
        assert stream.getvalue().splitlines() == ["active", "queued 0", "queued 1"]
    finally:
        stream.release.release()
        handler.close()
        if closer.ident is not None:
            closer.join(timeout=3)


def test_close_has_a_deadline_and_is_idempotent():
    stream = BlockingStream(stall_in="write", stalled_calls={1})
    handler = AsyncStreamHandler(stream, capacity=2, shutdown_timeout=0.05)
    logger = logging.Logger("shutdown", logging.INFO)
    logger.addHandler(handler)
    try:
        logger.info("active")
        stream.entered.get(timeout=5)
        started = time.monotonic()
        handler.close()
        handler.close()
        assert time.monotonic() - started < 1
        logger.info("after close")
        assert handler.dropped_records == 1
    finally:
        stream.release.release()
        handler.queue.join()
        handler.close()
    assert stream.getvalue().splitlines() == ["active"]


def test_reconfiguration_drains_old_records_and_preserves_exception_text(root_logger):
    before, after = io.StringIO(), io.StringIO()
    configure(before, True)
    for i in range(50):
        root_logger.info(f"before {i}")
    try:
        raise ValueError("example failure")
    except ValueError:
        root_logger.exception("failed")
    configure(after, False)
    root_logger.info("after")
    assert before.getvalue().splitlines()[:50] == [
        f"INFO before {i}" for i in range(50)
    ]
    assert "ERROR failed\nTraceback" in before.getvalue()
    assert "ValueError: example failure" in before.getvalue()
    assert after.getvalue() == "INFO after\n"


def test_normal_python_exit_drains_pending_logs():
    code = """
import logging, sys, time
from tokenspeed._logging import configure_root_logger
class SlowStream:
    def write(self, text):
        time.sleep(0.005)
        return sys.stdout.write(text)
    def flush(self):
        sys.stdout.flush()
configure_root_logger(level=logging.INFO, log_format='%(message)s', datefmt='',
                      stream=SlowStream(), enable_async_logging=True)
for i in range(20):
    logging.info(f'record {i}')
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "python")
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert result.stdout.splitlines() == [f"record {i}" for i in range(20)]


@pytest.mark.parametrize("enabled", [False, True])
def test_periodic_slow_write_measurement(root_logger, enabled):
    class PeriodicStream(io.StringIO):
        def __init__(self):
            super().__init__()
            self.writes = 0

        def write(self, text):
            self.writes += 1
            if self.writes % 10 == 0:
                time.sleep(0.05)
            return super().write(text)

    stream = PeriodicStream()
    configure(stream, enabled)
    latencies = []
    started = time.monotonic()
    for i in range(40):
        before = time.monotonic()
        root_logger.info(f"record {i}")
        latencies.append(time.monotonic() - before)
        time.sleep(0.001)
    producer_time = time.monotonic() - started
    stop_async_logging()
    assert stream.getvalue().splitlines() == [f"INFO record {i}" for i in range(40)]
    print(
        f"Python async={enabled}: producer={producer_time * 1000:.2f}ms, "
        f"max_log_call={max(latencies) * 1000:.2f}ms, records=40/40; "
        "50ms write stall every 10 records"
    )
