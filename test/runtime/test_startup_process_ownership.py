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
import functools
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


class _CleanupReadyWriter:
    """A real ready connection with a gate for controlled peer notifications."""

    def __init__(self, connection, cleanup_started, notifications_sent):
        self.connection = connection
        self.cleanup_started = cleanup_started
        self.notifications_sent = notifications_sent

    @property
    def closed(self):
        return self.connection.closed

    def send(self, message):
        self.connection.send(message)

    def close(self):
        self.connection.close()
        self.cleanup_started.set()
        assert self.notifications_sent.wait(_TIMEOUT_S)


def _dp_model_start(
    mode,
    startup_barrier,
    status_writer,
    cleanup_started,
    notifications_sent,
    server_args,
    port_args,
    writer,
):
    rank = server_args.mapping.attn.dp_rank
    startup_barrier.wait(_TIMEOUT_S)
    status_writer.send((rank, os.getpid()))
    startup_barrier.wait(_TIMEOUT_S)
    if mode in ("failed_group", "eof_then_failures") and rank == 0:
        # Failure while the other group is still in startup. A lingering thread
        # can hold the process alive even after its ready channel closes.
        writer.close()
    if mode in ("terminate_then_failures", "eof_then_failures"):
        assert cleanup_started.wait(_TIMEOUT_S)
        for sig in (signal.SIGUSR1, signal.SIGQUIT):
            os.kill(os.getppid(), sig)
        startup_barrier.wait(_TIMEOUT_S)
        if rank == 0:
            notifications_sent.set()
    threading.Event().wait(_TIMEOUT_S * 10)


def _dp_controller(mode, result_writer, writer):
    from tokenspeed.runtime.engine import data_parallel_controller as controller_module
    from tokenspeed.runtime.engine import event_loop as event_loop_module
    from tokenspeed.runtime.utils.server_args import PortArgs, ServerArgs

    status_reader, status_writer = mp.Pipe(duplex=False)
    startup_barrier = mp.Barrier(2)
    cleanup_started = mp.Event()
    notifications_sent = mp.Event()
    if mode in ("terminate_then_failures", "eof_then_failures"):
        writer = _CleanupReadyWriter(writer, cleanup_started, notifications_sent)
    worker_pids = []

    def observe_workers():
        for _ in range(2):
            if not status_reader.poll(_TIMEOUT_S):
                return
            worker_pids.append(status_reader.recv()[1])
        if mode in ("terminate", "terminate_then_failures"):
            os.kill(os.getpid(), signal.SIGTERM)

    args = ServerArgs(
        model="unused",
        device="cpu",
        world_size=2,
        nprocs_per_node=2,
        attn_tp_size=1,
        data_parallel_size=2,
        dist_init_addr="127.0.0.1:28000",
    )
    ports = PortArgs.init_new(args)
    observer = threading.Thread(target=observe_workers, daemon=True)
    observer.start()
    # Use real DP launch, readiness supervision and wrapper teardown; replace
    # only GPU/model loading with two real processes in a startup collective.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            event_loop_module,
            "run_event_loop",
            functools.partial(
                _dp_model_start,
                mode,
                startup_barrier,
                status_writer,
                cleanup_started,
                notifications_sent,
            ),
        )
        try:
            controller_module.run_data_parallel_controller_process(args, ports, writer)
        except (RuntimeError, SystemExit) as exc:
            outcome = str(exc) if isinstance(exc, RuntimeError) else exc.code
        else:
            outcome = "returned without failure"
    observer.join(_TIMEOUT_S)
    assert not observer.is_alive(), "worker observer did not finish"
    assert len(worker_pids) == 2, "both DP groups must start before the failure"
    # The wrapper owns termination although its constructor never returned.
    # Joining here reaps children; it must not be needed to stop them.
    for child in mp.active_children():
        child.join(_TIMEOUT_S)
    result_writer.send(
        {
            "outcome": outcome,
            "alive": [psutil.pid_exists(pid) for pid in worker_pids],
            "ready_closed": writer.closed,
        }
    )
    status_reader.close()
    status_writer.close()
    result_writer.close()


def _dp_supervisor(mode, result_writer, writer):
    # Consume the controller's real child-failure notification in an isolated
    # parent, rather than delivering it to pytest itself.
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    proc = mp.Process(target=_dp_controller, args=(mode, result_writer, writer))
    proc.start()
    writer.close()
    result_writer.close()
    proc.join(_TIMEOUT_S)
    assert not proc.is_alive(), "DP controller did not finish cleanup"
    assert proc.exitcode == 0


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


@pytest.mark.parametrize("signum", [signal.SIGUSR1, signal.SIGQUIT])
def test_engine_startup_signal_owns_worker_cleanup(signum):
    proc, reader = start_process_with_ready_pipe(
        _engine_startup_failure, (signum,), contextlib.nullcontext()
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


@pytest.mark.parametrize(
    "mode",
    ["failed_group", "terminate", "eof_then_failures", "terminate_then_failures"],
)
def test_dp_partial_startup_owns_worker_cleanup(mode):
    result_reader, result_writer = mp.Pipe(duplex=False)
    proc, ready_reader = start_process_with_ready_pipe(
        _dp_supervisor, (mode, result_writer), contextlib.nullcontext()
    )
    result_writer.close()
    try:
        assert result_reader.poll(_TIMEOUT_S), "DP controller did not unwind startup"
        result = result_reader.recv()
        if mode in ("failed_group", "eof_then_failures"):
            assert "Scheduler process 0" in result["outcome"]
        else:
            assert result["outcome"] == 0
        assert result["alive"] == [False, False]
        assert result["ready_closed"]
        assert ready_reader.poll(_TIMEOUT_S)
        with pytest.raises(EOFError):
            ready_reader.recv()
        proc.join(_TIMEOUT_S)
        assert proc.exitcode == 0
    finally:
        _stop_supervisor(proc)
        ready_reader.close()
        result_reader.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
