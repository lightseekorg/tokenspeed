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

"""Launcher cleanup after real worker failures and termination signals."""

import contextlib
import multiprocessing as mp
import os
import signal
import sys
import threading
import time

import psutil
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-1gpu")

from tokenspeed.runtime.utils.process import (  # noqa: E402
    register_child_failure_signals,
    start_process_with_ready_pipe,
)

_TIMEOUT_S = 60


@pytest.fixture(scope="module", autouse=True)
def _spawn_start_method():
    """Run this module's children on production's spawn start method.

    Several children import the engine stack inside the child; a forked child
    cannot re-initialize a parent's CUDA state on that import path, which is
    why production sets spawn in ``_set_envs_and_config``. These tests bypass
    that entry point, so set the same method here: every process, Event,
    Barrier and nested worker created in this module then shares one context.
    Restore the previous method so other modules are unaffected.
    """
    previous = mp.get_start_method(allow_none=True)
    mp.set_start_method("spawn", force=True)
    try:
        yield
    finally:
        mp.set_start_method(previous, force=True)


def _wait_for_termination(writer):
    writer.send({"status": "ready"})
    threading.Event().wait(_TIMEOUT_S * 10)


def _engine_startup_failure(signum, writer):
    from tokenspeed.runtime.entrypoints import engine as engine_module
    from tokenspeed.runtime.utils.server_args import ServerArgs

    started = []

    def launch_without_model(server_args, port_args):
        # Replace GPU/model loading only; Engine's constructor, failure handler,
        # signal callbacks and process-tree shutdown are the real implementation.
        register_child_failure_signals()
        for _ in range(2):
            proc, reader = start_process_with_ready_pipe(
                _wait_for_termination, (), contextlib.nullcontext()
            )
            started.append((proc, reader))
            assert reader.poll(_TIMEOUT_S)
            assert reader.recv() == {"status": "ready"}
        os.kill(os.getpid(), signum)
        raise AssertionError("child-failure signal did not unwind startup")

    args = ServerArgs(model="unused", device="cpu", world_size=1, nprocs_per_node=1)
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(engine_module, "_launch_subprocesses", launch_without_model)
            try:
                engine_module.Engine(server_args=args)
            except SystemExit as exc:
                for proc, _ in started:
                    proc.join(_TIMEOUT_S)
                writer.send(
                    {
                        "status": exc.code,
                        "exitcodes": [proc.exitcode for proc, _ in started],
                        "alive": [proc.is_alive() for proc, _ in started],
                    }
                )
    finally:
        for proc, reader in started:
            if proc.is_alive():
                proc.kill()
            proc.join(_TIMEOUT_S)
            reader.close()
        writer.close()


def _notify_failure(start, notified, signum, writer):
    parent_pid = os.getppid()
    writer.send({"status": "ready"})
    assert start.wait(_TIMEOUT_S)
    for sig in (signum, signal.SIGUSR1, signal.SIGQUIT):
        os.kill(parent_pid, sig)
    notified.set()
    threading.Event().wait(_TIMEOUT_S * 10)


@contextlib.contextmanager
def _failure_signals_on_thread(trigger, pending_failure):
    """Route worker signals away from the blocked main thread."""
    failures = {signal.SIGUSR1, signal.SIGQUIT}
    ready = threading.Event()
    stop = threading.Event()

    def receive_signals():
        signal.pthread_sigmask(signal.SIG_UNBLOCK, failures)
        ready.set()
        # The owner holds this condition until wait() releases it. Its native
        # wait must therefore start before the worker can send the first signal.
        with pending_failure:
            trigger.set()
        stop.wait(_TIMEOUT_S * 10)

    receiver = threading.Thread(target=receive_signals)
    receiver.start()
    try:
        assert ready.wait(_TIMEOUT_S), "signal receiver did not start"
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, failures)
        try:
            yield
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    finally:
        stop.set()
        receiver.join(_TIMEOUT_S)
        assert not receiver.is_alive(), "signal receiver did not stop"


def _repeated_failure_cleanup(signal_on_thread, writer):
    from tokenspeed.runtime.entrypoints.engine import Engine

    # A launcher installed after a failed instance must receive its own first
    # failure, even after an earlier launcher latched the shutdown state.
    lifecycle_exitcodes = []
    for _ in range(2):
        register_child_failure_signals()
        try:
            os.kill(os.getpid(), signal.SIGUSR1)
        except SystemExit as exc:
            lifecycle_exitcodes.append(exc.code)

    register_child_failure_signals()
    triggers = [mp.Event(), mp.Event()]
    notified = [mp.Event(), mp.Event()]
    started = [
        start_process_with_ready_pipe(
            _notify_failure,
            (trigger, sent, sig),
            contextlib.nullcontext(),
        )
        for trigger, sent, sig in zip(
            triggers, notified, (signal.SIGUSR1, signal.SIGQUIT)
        )
    ]
    engine = Engine.__new__(Engine)
    engine.llm = None
    engine.tokenizer_manager = None
    finished_cleanup = False
    outcome = "failure signal was not delivered"
    try:
        for _, reader in started:
            assert reader.poll(_TIMEOUT_S)
            assert reader.recv() == {"status": "ready"}
        try:
            pending_failure = threading.Condition()
            with pending_failure:
                signal_context = (
                    _failure_signals_on_thread(triggers[0], pending_failure)
                    if signal_on_thread
                    else contextlib.nullcontext()
                )
                with signal_context:
                    if not signal_on_thread:
                        triggers[0].set()
                    # A native library thread may receive the OS signal while
                    # Python's handler stays pending until the main thread runs
                    # again. Return to Python before the second worker times out.
                    deadline = time.monotonic() + _TIMEOUT_S
                    while time.monotonic() < deadline:
                        pending_failure.wait(0.1)
        except SystemExit as exc:
            outcome = exc.code
        finally:
            # The second real worker reports only after the owner is already
            # unwinding. Its notifications must not bypass actual Engine cleanup.
            triggers[1].set()
            assert all(event.wait(_TIMEOUT_S) for event in notified)
            engine.shutdown()
            finished_cleanup = True
    except SystemExit:
        # Record the failed behavior before fallback cleanup can hide survivors.
        outcome = "cleanup interrupted"
    finally:
        if finished_cleanup:
            for proc, _ in started:
                proc.join(_TIMEOUT_S)
        writer.send(
            {
                "lifecycle_exitcodes": lifecycle_exitcodes,
                "outcome": outcome,
                "finished_cleanup": finished_cleanup,
                "alive": [proc.is_alive() for proc, _ in started],
                "exitcodes": [proc.exitcode for proc, _ in started],
            }
        )
        for proc, reader in started:
            if proc.is_alive():
                proc.kill()
            proc.join(_TIMEOUT_S)
            reader.close()
        writer.close()


# A metric reporter plugin whose sink records that it was closed.
_RELAY_PROBE = """
import pathlib

from tokenspeed.runtime.metrics.registry import register_metric_reporter


class _Sink:
    def __init__(self, path):
        self._path = pathlib.Path(path)

    def emit(self, event):
        pass

    def close(self, timeout):
        self._path.write_text("closed")


def register():
    register_metric_reporter(
        "relay-probe",
        validate=lambda config: config,
        create_sink=lambda config: _Sink(config["path"]),
        start_node=None,
        override=False,
    )
"""


def _install_relay_probe(directory):
    """Install the probe plugin as a distribution on ``directory``."""
    (directory / "relay_probe.py").write_text(_RELAY_PROBE)
    dist = directory / "relay_probe-0.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: relay-probe\nVersion: 0.0\n"
    )
    (dist / "entry_points.txt").write_text(
        "[tokenspeed.plugins]\nrelay-probe = relay_probe:register\n"
    )


def _engine_shutdown_with_relay(directory, marker, writer):
    # The relay's spawned process inherits this path, so it loads the probe.
    sys.path.insert(0, directory)
    from tokenspeed.runtime.engine.async_llm import AsyncLLM
    from tokenspeed.runtime.entrypoints.engine import Engine
    from tokenspeed.runtime.metrics.relay import MetricsRelay

    relay = MetricsRelay({"relay-probe": {"path": marker}})
    # Bypass model and network construction only: the relay and the shutdown
    # are the real implementation.
    tokenizer_manager = AsyncLLM.__new__(AsyncLLM)
    tokenizer_manager.bootstrap_server = None
    tokenizer_manager.metrics_relay = relay
    engine = Engine.__new__(Engine)
    engine.llm = None
    engine.tokenizer_manager = tokenizer_manager
    engine.shutdown()
    writer.send(open(marker).read() if os.path.exists(marker) else None)


def _stop_supervisor(proc):
    if proc.is_alive():
        # Test-only fallback must stop descendants if the regression hangs.
        try:
            for child in psutil.Process(proc.pid).children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
        except psutil.NoSuchProcess:
            pass
        proc.kill()
    proc.join(_TIMEOUT_S)


def test_engine_startup_signal_owns_worker_cleanup():
    # SIGUSR1 and SIGQUIT share one handler; the repeated-signal test below
    # delivers both.
    proc, reader = start_process_with_ready_pipe(
        _engine_startup_failure, (signal.SIGUSR1,), contextlib.nullcontext()
    )
    try:
        assert reader.poll(_TIMEOUT_S), "Engine did not finish startup cleanup"
        result = reader.recv()
        assert result["status"] == 1
        assert result["alive"] == [False, False]
        assert result["exitcodes"] == [-signal.SIGKILL, -signal.SIGKILL]
        proc.join(_TIMEOUT_S)
        assert proc.exitcode == 0
    finally:
        _stop_supervisor(proc)
        reader.close()


@pytest.mark.parametrize("signal_on_thread", [False, True], ids=["main", "thread"])
def test_repeated_failure_signals_cannot_interrupt_owner_cleanup(signal_on_thread):
    proc, reader = start_process_with_ready_pipe(
        _repeated_failure_cleanup, (signal_on_thread,), contextlib.nullcontext()
    )
    try:
        # The child imports the engine stack and waits on two spawned workers
        # with their own _TIMEOUT_S budgets, so its worst case exceeds a
        # single _TIMEOUT_S on a busy runner.
        assert reader.poll(
            _TIMEOUT_S * 3
        ), "owner did not finish repeated-signal cleanup"
        result = reader.recv()
        assert result["lifecycle_exitcodes"] == [1, 1]
        assert result["outcome"] == 1
        assert result["finished_cleanup"]
        assert result["alive"] == [False, False]
        assert result["exitcodes"] == [-signal.SIGKILL, -signal.SIGKILL]
        proc.join(_TIMEOUT_S)
        assert proc.exitcode == 0
    finally:
        _stop_supervisor(proc)
        reader.close()


def test_engine_shutdown_lets_the_metrics_relay_flush(tmp_path):
    # The relay closes its sinks on a stop; killed with the process tree, it
    # would drop what they still buffer.
    _install_relay_probe(tmp_path)
    marker = tmp_path / "closed"
    proc, reader = start_process_with_ready_pipe(
        _engine_shutdown_with_relay,
        (str(tmp_path), str(marker)),
        contextlib.nullcontext(),
    )
    try:
        assert reader.poll(_TIMEOUT_S), "Engine did not finish its shutdown"
        assert reader.recv() == "closed"
        proc.join(_TIMEOUT_S)
        assert proc.exitcode == 0
    finally:
        _stop_supervisor(proc)
        reader.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
