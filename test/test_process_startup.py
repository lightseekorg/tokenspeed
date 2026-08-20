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

import pytest

from tokenspeed.runtime.process_startup import wait_for_process_startup


def _wait_without_reporting(writer, stop) -> None:
    stop.wait()
    writer.close()


def _exit_without_reporting() -> None:
    return None


def _stop_processes(processes, stop=None) -> None:
    if stop is not None:
        stop.set()
    for process in processes:
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)


def test_later_rank_exit_is_not_hidden_by_silent_first_rank():
    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    silent_reader, silent_writer = ctx.Pipe(duplex=False)
    exited_reader, exited_writer = ctx.Pipe(duplex=False)
    processes = [
        ctx.Process(target=_wait_without_reporting, args=(silent_writer, stop)),
        ctx.Process(target=_exit_without_reporting),
    ]
    for process in processes:
        process.start()
    silent_writer.close()

    try:
        with pytest.raises(RuntimeError, match=r"scheduler rank 1 exited.*code 0"):
            wait_for_process_startup(
                [silent_reader, exited_reader],
                processes,
                description="scheduler rank",
                timeout_seconds=5,
            )
    finally:
        exited_writer.close()
        silent_reader.close()
        exited_reader.close()
        _stop_processes(processes, stop)


def test_startup_wait_has_a_finite_deadline():
    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    reader, writer = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_wait_without_reporting, args=(writer, stop))
    process.start()
    writer.close()

    try:
        with pytest.raises(TimeoutError, match="Timed out after 0.1s"):
            wait_for_process_startup(
                [reader],
                [process],
                description="scheduler rank",
                timeout_seconds=0.1,
            )
    finally:
        reader.close()
        _stop_processes([process], stop)
