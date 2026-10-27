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

"""Process and signal helpers."""

from __future__ import annotations

import contextlib
import ctypes
import logging
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from multiprocessing import connection
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any

import psutil

logger = logging.getLogger(__name__)

_process_shutdown_started = False

_PR_SET_PDEATHSIG = 1

if sys.platform.startswith("linux"):
    # Bound once so a post-fork child (a ``preexec_fn``) only calls prctl.
    _libc = ctypes.CDLL(None, use_errno=True)


def begin_process_shutdown() -> bool:
    """Latch owner cleanup, returning whether this is the first shutdown request.

    Failure and termination callbacks share this latch so a second worker's
    notification cannot interrupt cleanup already unwinding the process.
    """
    global _process_shutdown_started
    if _process_shutdown_started:
        return False
    _process_shutdown_started = True
    return True


def register_child_failure_signals():
    """Unwind once to the launcher that owns cleanup after a child failure."""
    global _process_shutdown_started
    _process_shutdown_started = False

    def signal_handler(sig, frame):
        if begin_process_shutdown():
            raise SystemExit(1)

    signal.signal(signal.SIGUSR1, signal_handler)
    signal.signal(signal.SIGQUIT, signal_handler)


def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
    """Kill the target process and all of its child processes."""
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    children = itself.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    if include_parent:
        try:
            if parent_pid == os.getpid():
                itself.kill()
                sys.exit(0)

            itself.kill()
            itself.send_signal(signal.SIGQUIT)
        except psutil.NoSuchProcess:
            pass


def set_parent_death_signal(parent_pid: int) -> None:
    """Ask the kernel to SIGKILL this process when ``parent_pid`` dies.

    The kernel sends SIGKILL even when the parent itself was SIGKILLed.
    Linux only; elsewhere this does nothing. The signal follows the parent
    *thread* this process was started from, so start the process from a
    thread that lives as long as the parent needs it.
    """
    if not sys.platform.startswith("linux"):
        return
    if _libc.prctl(_PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    # The parent may have died before the signal was set up.
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)


def run_with_parent_death_signal(parent_pid: int, target: Callable, *args):
    """Run ``target(*args)`` in a process that is killed when its parent dies.

    Meant as a ``multiprocessing.Process`` target, with ``parent_pid`` the
    parent's ``os.getpid()``; see ``set_parent_death_signal``. Elsewhere
    than Linux ``target`` just runs.
    """
    set_parent_death_signal(parent_pid)
    return target(*args)


def stop_processes(processes: Sequence[BaseProcess], timeout: float) -> None:
    """Stop child ``processes`` and everything they started.

    Each live one gets SIGTERM, and together they get ``timeout`` seconds to
    exit. Then the ones still running, and every process that was below any
    of them when this was called, get SIGKILL, and the killed ones get
    ``timeout`` seconds more to be gone. Listing descendants first matters:
    one whose parent exits in the meantime is no longer found below it.
    """
    processes = [process for process in processes if process.is_alive()]
    descendants = []
    for process in processes:
        with contextlib.suppress(psutil.NoSuchProcess):
            descendants += psutil.Process(process.pid).children(recursive=True)
        process.terminate()

    def join(waiting):
        deadline = time.monotonic() + timeout
        for process in waiting:
            process.join(max(0.0, deadline - time.monotonic()))

    join(processes)
    stuck = [process for process in processes if process.exitcode is None]
    for process in stuck:
        logger.warning(f"Process {process.pid!s} still running after {timeout!s} s")
        process.kill()
    for descendant in descendants:
        with contextlib.suppress(psutil.NoSuchProcess):
            descendant.kill()
    join(stuck)


def start_process_with_ready_pipe(
    target: Callable[..., None],
    args: tuple,
    start_context: AbstractContextManager,
) -> tuple[mp.Process, Connection]:
    """Start ``target(*args, writer)`` and return it with the pipe's read end.

    The child reports readiness by sending on ``writer``. The parent closes
    its own copy of ``writer`` once the child has started, so ``recv()`` on
    the returned reader raises ``EOFError`` if the child dies first (e.g. it
    is OOM-killed) instead of blocking forever.

    Args:
        target: The child process entry point.
        args: Positional arguments passed before ``writer``.
        start_context: Entered around ``Process.start()``.

    Returns:
        The started process and the read end of its pipe.
    """
    reader, writer = mp.Pipe(duplex=False)
    proc = mp.Process(target=target, args=(*args, writer))
    with start_context:
        proc.start()
    writer.close()
    return proc, reader


def receive_ready_messages(
    readers: Sequence[Connection], procs: Sequence[mp.Process]
) -> list[Any]:
    """Receive one message per reader, failing when any child exits or closes its pipe.

    ``readers[i]`` is the ready pipe of ``procs[i]``. Waiting on every pipe
    and every process at once matters with several ranks: when one rank is
    killed, the others can block in a collective and never report, so
    waiting on their pipes alone would hang.

    Detection never waits for cleanup or terminates workers. The launcher
    owns shutdown of the process tree.

    Args:
        readers: The read ends from ``start_process_with_ready_pipe``.
        procs: The processes, in the same order.

    Returns:
        The messages, in the order of ``readers``.

    Raises:
        RuntimeError: A process exited before every reader got its message,
            including one that had already reported, or closed its pipe
            while cleanup still runs: its peers may be waiting on it.
    """
    messages: list[Any] = [None] * len(readers)
    reader_indices = {reader: index for index, reader in enumerate(readers)}
    pending = set(readers)
    sentinels = {proc.sentinel: index for index, proc in enumerate(procs)}
    while pending:
        ready = connection.wait([*reader_indices, *sentinels])
        for reader in [obj for obj in ready if obj in reader_indices]:
            index = reader_indices[reader]
            try:
                message = reader.recv()
            except EOFError:
                _raise_process_died(procs, index)
            if reader not in pending:
                raise RuntimeError(
                    f"Scheduler process {index} reported readiness twice"
                )
            messages[index] = message
            pending.remove(reader)
        for index in [sentinels[obj] for obj in ready if obj in sentinels]:
            _raise_process_died(procs, index)
    return messages


def _raise_process_died(procs: Sequence[mp.Process], index: int) -> None:
    proc = procs[index]
    raise RuntimeError(
        f"Scheduler process {index} (pid {proc.pid}) exited or closed its "
        f"ready pipe before startup finished (exit code {proc.exitcode}). "
        "Please check if there are relevant logs."
    )
