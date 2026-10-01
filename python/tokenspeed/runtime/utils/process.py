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
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Sequence
from multiprocessing.process import BaseProcess

import psutil

logger = logging.getLogger(__name__)

_PR_SET_PDEATHSIG = 1


def register_usr_signal():
    parent_process = psutil.Process().parent()

    def signal_handler(sig, frame):
        logger.error("recv usr signal, kill usr signal to parent")
        parent_process.send_signal(signal.SIGUSR1)

    signal.signal(signal.SIGUSR1, signal_handler)


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


def run_with_parent_death_signal(parent_pid: int, target: Callable, *args):
    """Run ``target(*args)`` in a process that is killed when its parent dies.

    Meant as a ``multiprocessing.Process`` target, with ``parent_pid`` the
    parent's ``os.getpid()``. The kernel sends SIGKILL even when the parent
    itself was SIGKILLed. Linux only; elsewhere ``target`` just runs. The
    signal follows the parent *thread* that started this process, so start
    it from a thread that lives as long as the parent needs it.
    """
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(_PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0) != 0:
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno))
        # The parent may have died before the signal was set up.
        if os.getppid() != parent_pid:
            os.kill(os.getpid(), signal.SIGKILL)
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
