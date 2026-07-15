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

import logging
import os
import signal
import sys
import time
from ctypes import CDLL, get_errno

import psutil

logger = logging.getLogger(__name__)

_PR_SET_CHILD_SUBREAPER = 36


def _enable_child_subreaper() -> bool:
    """Ask Linux to adopt orphaned descendants so this process can reap them."""

    if sys.platform != "linux":
        return False
    libc = CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) == 0:
        return True
    logger.warning("Could not enable child subreaper: errno=%s", get_errno())
    return False


def _descendant_depths(
    descendants: list[psutil.Process], root_pid: int
) -> dict[int, int]:
    parent_by_pid = {}
    for process in descendants:
        try:
            parent_by_pid[process.pid] = process.ppid()
        except psutil.NoSuchProcess:
            continue

    depths = {}
    for pid in parent_by_pid:
        depth = 0
        current = pid
        seen = set()
        while current != root_pid and current in parent_by_pid and current not in seen:
            seen.add(current)
            current = parent_by_pid[current]
            depth += 1
        depths[pid] = depth
    return depths


def _wait_and_reap_owned(processes: list[psutil.Process], timeout: float) -> list[int]:
    """Wait for captured PIDs and reap each one if it becomes our child."""

    pending = {process.pid for process in processes}
    deadline = time.monotonic() + timeout
    while pending and time.monotonic() < deadline:
        for pid in tuple(pending):
            try:
                waited_pid, _ = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                if not psutil.pid_exists(pid):
                    pending.discard(pid)
            else:
                if waited_pid == pid:
                    pending.discard(pid)
            if pid in pending and not psutil.pid_exists(pid):
                pending.discard(pid)
        if pending:
            time.sleep(0.01)
    return sorted(pending)


def register_usr_signal():
    parent_process = psutil.Process().parent()

    def signal_handler(sig, frame):
        logger.error("recv usr signal, kill usr signal to parent")
        parent_process.send_signal(signal.SIGUSR1)

    signal.signal(signal.SIGUSR1, signal_handler)


def kill_process_tree(
    parent_pid: int | None,
    include_parent: bool = True,
    skip_pid: int | None = None,
    wait_timeout: float = 5.0,
) -> None:
    """Kill a process tree and reap children before returning.

    Descendants are killed deepest-first. On Linux, the caller becomes a child
    subreaper before termination, so orphaned scheduler grandchildren are
    adopted and can be waited by exact PID instead of escaping to container
    PID 1. TokenSpeed owns these scheduler and ``resource_tracker`` processes;
    leaving them to a minimal non-reaping PID 1 is unsafe.

    Args:
        parent_pid: Root of the tree. ``None`` means the current process while
            implicitly excluding that process itself.
        include_parent: Whether to kill the root after its descendants.
        skip_pid: Optional descendant PID to leave untouched.
        wait_timeout: Total number of seconds to wait for killed descendants.
    """
    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    _enable_child_subreaper()
    children = [
        child for child in itself.children(recursive=True) if child.pid != skip_pid
    ]
    depths = _descendant_depths(children, parent_pid)
    children.sort(key=lambda child: depths.get(child.pid, 0), reverse=True)
    for child in children:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    if children:
        pending = _wait_and_reap_owned(children, wait_timeout)
        if pending:
            logger.warning(
                "Timed out waiting for descendant PIDs to exit: %s",
                pending,
            )

    if include_parent:
        try:
            if parent_pid == os.getpid():
                itself.kill()
                sys.exit(0)

            itself.kill()
            pending = _wait_and_reap_owned([itself], wait_timeout)
            if pending:
                logger.warning("Timed out waiting for root PID to exit: %s", pending)
        except psutil.NoSuchProcess:
            pass
