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

import multiprocessing as mp
import threading
import time

import pytest

from tokenspeed.runtime.process_startup import wait_for_process_startup


def _wait_without_reporting(writer, stop) -> None:
    stop.wait()
    writer.close()


def _report_error(writer, error: str) -> None:
    writer.send({"status": "error", "error": error})
    writer.close()
    raise SystemExit(3)


def _report_ready(writer, rank: int, stop) -> None:
    writer.send({"status": "ready", "rank": rank})
    writer.close()
    stop.wait()


def _exit_without_reporting() -> None:
    raise SystemExit(4)


def _stop_processes(processes, stop=None) -> None:
    if stop is not None:
        stop.set()
    for process in processes:
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)


def test_later_rank_failure_is_not_hidden_by_silent_first_rank():
    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    silent_reader, silent_writer = ctx.Pipe(duplex=False)
    failed_reader, failed_writer = ctx.Pipe(duplex=False)
    processes = [
        ctx.Process(target=_wait_without_reporting, args=(silent_writer, stop)),
        ctx.Process(
            target=_report_error,
            args=(failed_writer, "DistNetworkError: EADDRINUSE"),
        ),
    ]
    for process in processes:
        process.start()
    silent_writer.close()
    failed_writer.close()

    # Bound the regression case: a rank-ordered recv would only unblock when
    # the silent first rank exits and closes its writer five seconds later.
    fallback_stop = threading.Timer(5, stop.set)
    fallback_stop.start()
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="EADDRINUSE"):
            wait_for_process_startup(
                [silent_reader, failed_reader],
                processes,
                description="scheduler rank",
            )
        assert time.monotonic() - started < 5
    finally:
        fallback_stop.cancel()
        silent_reader.close()
        failed_reader.close()
        _stop_processes(processes, stop)


def test_process_exit_is_detected_even_if_parent_writer_stays_open():
    ctx = mp.get_context("spawn")
    reader, writer = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_exit_without_reporting)
    process.start()

    try:
        with pytest.raises(RuntimeError, match="exited before reporting readiness"):
            wait_for_process_startup([reader], [process], description="scheduler rank")
    finally:
        # Deliberately keep this open until after supervision returns: the
        # process sentinel, rather than pipe EOF, must surface the failure.
        writer.close()
        reader.close()
        _stop_processes([process])


def test_readiness_results_preserve_process_order():
    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    pairs = [ctx.Pipe(duplex=False) for _ in range(2)]
    processes = [
        ctx.Process(target=_report_ready, args=(writer, rank, stop))
        for rank, (_, writer) in enumerate(pairs)
    ]
    for process in reversed(processes):
        process.start()
    for _, writer in pairs:
        writer.close()

    try:
        results = wait_for_process_startup(
            [reader for reader, _ in pairs],
            processes,
            description="scheduler rank",
        )
        assert [result["rank"] for result in results] == [0, 1]
    finally:
        for reader, _ in pairs:
            reader.close()
        _stop_processes(processes, stop)
