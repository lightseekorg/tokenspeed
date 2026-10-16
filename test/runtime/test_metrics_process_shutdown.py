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

"""Graceful metric cleanup and bounded teardown of real subprocess trees."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from contextlib import contextmanager

import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

_PROCESS_SCRIPT = r"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

directory = Path(sys.argv[1])
name = sys.argv[2]
behavior = sys.argv[3]
script = sys.argv[4]
children = []


def cleanup(signum, frame):
    (directory / f"cleanup-{name}").write_text("flushed")
    raise SystemExit(0)


signal.signal(signal.SIGTERM, cleanup if behavior == "graceful" or name == "parent" else signal.SIG_IGN)
if name == "parent":
    names = ["child0", "child1"]
elif name == "child0":
    names = ["grandchild"]
else:
    names = []
for child_name in names:
    children.append(subprocess.Popen([sys.executable, "-c", script, str(directory), child_name, behavior, script]))
for child_name in names:
    while not (directory / f"ready-{child_name}").exists():
        time.sleep(0.005)
(directory / f"ready-{name}").write_text(str(os.getpid()))
while True:
    for child in children:
        child.poll()
    time.sleep(0.005)
"""


def _wait_until(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "subprocess did not reach expected state"
        time.sleep(0.01)


@contextmanager
def _process_tree(directory, behavior):
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _PROCESS_SCRIPT,
            str(directory),
            "parent",
            behavior,
            _PROCESS_SCRIPT,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_until(lambda: (directory / "ready-parent").exists())
        descendants = [
            psutil.Process(int((directory / f"ready-{name}").read_text()))
            for name in ("grandchild", "child0", "child1")
        ]
        yield parent, descendants
    finally:
        for marker in directory.glob("ready-*"):
            try:
                psutil.Process(int(marker.read_text())).kill()
            except psutil.NoSuchProcess:
                pass
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=5)


def _has_exited(process):
    try:
        return not process.is_running() or process.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def test_recursive_children_flush_without_signalling_parent(tmp_path):
    from tokenspeed.runtime.utils.process import terminate_child_processes

    with _process_tree(tmp_path, "graceful") as (parent, descendants):
        terminate_child_processes(parent.pid, timeout=2.0)

        for name in ("child0", "child1", "grandchild"):
            assert (tmp_path / f"cleanup-{name}").read_text() == "flushed"
        _wait_until(lambda: all(_has_exited(child) for child in descendants))
        assert parent.poll() is None
        assert not (tmp_path / "cleanup-parent").exists()


def test_term_ignoring_children_are_killed_after_one_shared_timeout(tmp_path):
    from tokenspeed.runtime.utils.process import terminate_child_processes

    with _process_tree(tmp_path, "ignore") as (parent, descendants):
        started = time.monotonic()
        terminate_child_processes(parent.pid, timeout=0.2)
        elapsed = time.monotonic() - started

        assert elapsed < 0.5
        _wait_until(lambda: all(_has_exited(child) for child in descendants))
        assert parent.poll() is None
        assert not (tmp_path / "cleanup-parent").exists()


def test_signal_permission_error_does_not_skip_other_children(
    tmp_path, monkeypatch, caplog
):
    from tokenspeed.runtime.utils.process import terminate_child_processes

    with _process_tree(tmp_path, "graceful") as (parent, descendants):
        denied_pid = descendants[-1].pid
        send_signal = psutil.Process.send_signal

        def deny_one_signal(process, signum):
            if process.pid == denied_pid:
                raise psutil.AccessDenied(pid=denied_pid)
            return send_signal(process, signum)

        with monkeypatch.context() as patch:
            patch.setattr(psutil.Process, "send_signal", deny_one_signal)
            terminate_child_processes(parent.pid, timeout=0.2)

        assert (tmp_path / "cleanup-child0").read_text() == "flushed"
        assert (tmp_path / "cleanup-grandchild").read_text() == "flushed"
        assert "denied" in caplog.text.lower()
        assert parent.poll() is None
        assert not (tmp_path / "cleanup-parent").exists()


def test_enumeration_permission_error_is_logged(monkeypatch, caplog):
    from tokenspeed.runtime.utils.process import terminate_child_processes

    def deny_enumeration(process, recursive):
        raise PermissionError("process enumeration not permitted")

    monkeypatch.setattr(psutil.Process, "children", deny_enumeration)

    terminate_child_processes(os.getpid(), timeout=0.2)

    assert "process enumeration not permitted" in caplog.text
