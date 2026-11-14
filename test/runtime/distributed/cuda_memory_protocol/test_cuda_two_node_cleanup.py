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

"""CPU regressions for deadline-safe cleanup of the CUDA test launchers."""

import os
import select
import signal
import subprocess
import sys
import threading
import time
from test.ci_system.ci_register import register_cuda_ci

import pytest
from test_cuda_two_node import stop_processes

# CPU regressions use the runtime job's complete dependency environment.
register_cuda_ci(est_time=5, suite="runtime-1gpu")

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX launcher cleanup")

CHILD = """
import signal
import time

def terminate(signum, frame):
    time.sleep(0.2)
    raise SystemExit(0)

signal.signal(signal.SIGTERM, terminate)
print("ready", flush=True)
while True:
    signal.pause()
"""


@pytest.fixture
def process_sessions():
    children = []
    background_done = threading.Event()
    background_ready = threading.Event()

    def background():
        # A process-directed deadline can reach another thread even when the
        # main thread masks it. Exercise the same condition as native IO threads.
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGALRM, signal.SIGTERM})
        background_ready.set()
        background_done.wait()

    thread = threading.Thread(target=background)
    thread.start()

    def start():
        session = []
        for _ in range(2):
            process = subprocess.Popen(
                [sys.executable, "-c", CHILD],
                stdout=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            children.append(process)
            session.append(process)
            assert select.select([process.stdout], [], [], 5)[0], "child not ready"
            assert process.stdout.readline().strip() == "ready"
        return session

    try:
        assert background_ready.wait(5)
        yield start
    finally:
        try:
            # These test children have no descendants. Recover their exact
            # PIDs independently if the cleanup regression fails.
            for process in children:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
                process.stdout.close()
        finally:
            background_done.set()
            thread.join(timeout=5)


@pytest.fixture
def processes(process_sessions):
    return process_sessions()


def assert_reaped(processes):
    assert [process.returncode for process in processes] == [0, 0]
    for process in processes:
        with pytest.raises(ChildProcessError):
            os.waitpid(process.pid, os.WNOHANG)


@pytest.mark.parametrize(
    ("signum", "error"),
    [(signal.SIGALRM, TimeoutError), (signal.SIGTERM, KeyboardInterrupt)],
)
def test_deadline_waits_until_both_children_are_reaped(processes, signum, error):
    observed = []

    def expired(_signum, _frame):
        observed.append([process.returncode for process in processes])
        raise error("test deadline")

    previous = signal.signal(signum, expired)
    timer = None
    try:
        if signum == signal.SIGALRM:
            signal.setitimer(signal.ITIMER_REAL, 0.05)
        else:
            timer = threading.Timer(0.05, os.kill, (os.getpid(), signum))
            timer.start()
        with pytest.raises(error, match="test deadline"):
            stop_processes(processes)
        assert observed == [[0, 0]]
        assert_reaped(processes)
        # A later session still uses the original deadline handler.
        with pytest.raises(error, match="test deadline"):
            signal.raise_signal(signum)
        assert observed == [[0, 0], [0, 0]]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if timer is not None:
            timer.cancel()
            timer.join(timeout=5)
        signal.signal(signum, previous)


def test_cleanup_keeps_a_later_watchdog_armed_across_sessions(process_sessions):
    def expired(_signum, _frame):
        raise TimeoutError("next session deadline")

    previous = signal.signal(signal.SIGALRM, expired)
    try:
        for _ in range(2):
            processes = process_sessions()
            signal.setitimer(signal.ITIMER_REAL, 0.7)
            assert stop_processes(processes) == [0, 0]
            assert_reaped(processes)
            with pytest.raises(TimeoutError, match="next session deadline"):
                time.sleep(2)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def test_ignored_alarm_does_not_hide_pending_termination(processes):
    def expired(_signum, _frame):
        raise TimeoutError("termination during cleanup")

    previous_alarm = signal.signal(signal.SIGALRM, signal.SIG_IGN)
    previous_term = signal.signal(signal.SIGTERM, expired)
    timer = threading.Timer(0.06, os.kill, (os.getpid(), signal.SIGTERM))
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.025)
        timer.start()
        with pytest.raises(TimeoutError, match="termination during cleanup"):
            stop_processes(processes)
        assert_reaped(processes)
        signal.raise_signal(signal.SIGALRM)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        timer.cancel()
        timer.join(timeout=5)
        try:
            signal.signal(signal.SIGALRM, previous_alarm)
        finally:
            signal.signal(signal.SIGTERM, previous_term)
