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

"""Tests for the scheduler ready pipe when a child dies before it reports."""

import contextlib
import multiprocessing as mp
import os
import signal
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.utils.process import (  # noqa: E402
    receive_ready_messages,
    start_process_with_ready_pipe,
)

# Long enough for a slow spawn; a blocked recv() would wait forever instead.
_TIMEOUT_S = 60


@pytest.fixture(scope="module", autouse=True)
def _spawn_start_method():
    """Run this module's children on production's spawn start method.

    Several children import the engine stack inside the child; a forked child
    cannot re-initialize a parent's CUDA state on that import path, which is
    why production sets spawn in ``_set_envs_and_config``. These tests bypass
    that entry point, so set the same method here: every process, Event and
    Barrier created in this module then shares one context. Restore the
    previous method so other modules are unaffected.
    """
    previous = mp.get_start_method(allow_none=True)
    mp.set_start_method("spawn", force=True)
    try:
        yield
    finally:
        mp.set_start_method(previous, force=True)


def _report_ready(value, writer):
    writer.send({"status": "ready", "value": value})
    # A scheduler keeps running after it reports.
    time.sleep(_TIMEOUT_S * 10)


def _die_before_ready(writer):
    os.kill(os.getpid(), signal.SIGKILL)


def _report_then_die(writer):
    writer.send({"status": "ready"})
    os.kill(os.getpid(), signal.SIGKILL)


def _never_ready(writer):
    # Like a rank blocked in a collective on a peer that died.
    time.sleep(_TIMEOUT_S * 10)


def _report_close_and_wait(finish_cleanup, writer):
    writer.send({"status": "ready"})
    writer.close()
    finish_cleanup.wait(_TIMEOUT_S * 10)


def _fail_with_lingering_cleanup(finish_cleanup, writer):
    # A rank whose startup failed: it closes its ready channel first, so the
    # launcher can stop the other ranks while its own cleanup (here a
    # transport thread a partial constructor started) still runs.
    threading.Thread(target=finish_cleanup.wait, daemon=False).start()
    writer.close()


def _engine_owns_failed_startup_cleanup(writer):
    from tokenspeed.runtime.entrypoints.engine import Engine

    finish_cleanup = mp.Event()
    started = [
        start_process_with_ready_pipe(_never_ready, (), contextlib.nullcontext()),
        start_process_with_ready_pipe(
            _fail_with_lingering_cleanup,
            (finish_cleanup,),
            contextlib.nullcontext(),
        ),
    ]
    # The constructor's existing failure handler calls shutdown before its
    # frontend has been installed. Bypass only model/network construction.
    engine = Engine.__new__(Engine)
    engine.llm = None
    engine.tokenizer_manager = None
    try:
        try:
            receive_ready_messages(
                [reader for _, reader in started], [proc for proc, _ in started]
            )
        except RuntimeError as exc:
            engine.shutdown()
            for proc, _ in started:
                proc.join(_TIMEOUT_S)
            writer.send(
                {
                    "error": str(exc),
                    "alive": [proc.is_alive() for proc, _ in started],
                    "exitcodes": [proc.exitcode for proc, _ in started],
                }
            )
    finally:
        for proc, reader in started:
            if proc.is_alive():
                proc.kill()
            proc.join(_TIMEOUT_S)
            reader.close()
        writer.close()


class TestReadyPipe:
    def test_killed_child_raises_eof_instead_of_blocking(self) -> None:
        proc, reader = start_process_with_ready_pipe(
            _die_before_ready, (), contextlib.nullcontext()
        )
        proc.join(_TIMEOUT_S)
        assert proc.exitcode == -signal.SIGKILL
        message = "recv() would block: the parent still holds the write end"
        assert reader.poll(_TIMEOUT_S), message
        with pytest.raises(EOFError):
            reader.recv()

    def _receive_or_fail(self, started):
        """receive_ready_messages, failing the test instead of hanging."""
        result = {}

        def run():
            try:
                result["messages"] = receive_ready_messages(
                    [reader for _, reader in started], [proc for proc, _ in started]
                )
            except RuntimeError as exc:
                result["error"] = exc

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(_TIMEOUT_S)
        assert not thread.is_alive(), "receive_ready_messages hung"
        return result

    @pytest.fixture
    def start_process(self, request):
        def start(target, args=()):
            proc, reader = start_process_with_ready_pipe(
                target, args, contextlib.nullcontext()
            )

            def cleanup():
                if proc.is_alive():
                    proc.kill()
                proc.join(_TIMEOUT_S)
                reader.close()

            request.addfinalizer(cleanup)
            return proc, reader

        return start

    def test_receive_returns_messages_in_reader_order(self, start_process) -> None:
        started = [start_process(_report_ready, (value,)) for value in (1, 2)]
        messages = self._receive_or_fail(started)["messages"]
        assert [m["value"] for m in messages] == [1, 2]

    def test_receive_fails_when_another_rank_dies(self, start_process) -> None:
        # Rank 0 stays alive without reporting; rank 1 is killed. Waiting on
        # rank 0's pipe alone would hang.
        started = [start_process(_never_ready), start_process(_die_before_ready)]
        assert "Scheduler process 1" in str(self._receive_or_fail(started)["error"])

    def test_receive_fails_when_a_reported_rank_dies(self, start_process) -> None:
        # Rank 1 reports and then dies while rank 0 is still loading: only
        # its process exit, not its pipe, shows it.
        started = [start_process(_never_ready), start_process(_report_then_die)]
        assert "Scheduler process 1" in str(self._receive_or_fail(started)["error"])

    def test_receive_fails_when_reported_rank_closes_pipe_while_alive(
        self, start_process
    ) -> None:
        finish_cleanup = mp.Event()
        started = [
            start_process(_never_ready),
            start_process(_report_close_and_wait, (finish_cleanup,)),
        ]
        failed, _ = started[1]
        assert "Scheduler process 1" in str(self._receive_or_fail(started)["error"])
        # Detecting EOF must neither wait for nor terminate the worker's cleanup.
        assert failed.is_alive()
        assert failed.exitcode is None
        finish_cleanup.set()
        failed.join(_TIMEOUT_S)
        assert failed.exitcode == 0

    def test_failed_startup_reports_without_killing_worker_cleanup(
        self, start_process
    ) -> None:
        finish_cleanup = mp.Event()
        started = [
            start_process(_never_ready),
            start_process(_fail_with_lingering_cleanup, (finish_cleanup,)),
        ]
        failed, _ = started[1]
        assert "Scheduler process 1" in str(self._receive_or_fail(started)["error"])
        assert failed.is_alive()
        finish_cleanup.set()
        failed.join(_TIMEOUT_S)
        assert not failed.is_alive()
        assert failed.exitcode == 0

    def test_engine_failure_cleanup_reaps_workers(self, start_process) -> None:
        supervisor, reader = start_process(_engine_owns_failed_startup_cleanup)
        assert reader.poll(_TIMEOUT_S), "launcher failed to unwind startup"
        result = reader.recv()
        assert "Scheduler process 1" in result["error"]
        assert result["alive"] == [False, False]
        assert result["exitcodes"] == [-signal.SIGKILL, -signal.SIGKILL]
        supervisor.join(_TIMEOUT_S)
        assert supervisor.exitcode == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
