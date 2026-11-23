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

"""Real subprocess stalls produce persistent, attributed stacks without GPUs."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


def start_worker(tmp_path, mode):
    script = tmp_path / "worker.py"
    script.write_text(
        "import sys,time\n"
        "from pathlib import Path\n"
        "from ci_system.worker_diagnostics import worker_diagnostics\n"
        "def wait_for_release():\n"
        "    sys.stdin.readline()\n"
        "with worker_diagnostics(Path(sys.argv[1]), 1, dump_after_seconds=1) as phase:\n"
        "    phase('cpu_fixture')\n"
        "    if sys.argv[2] == 'stall': wait_for_release()\n"
        "    phase('complete')\n"
        # Reopening reuses the descriptor faulthandler captured. Without
        # cancellation, the timer appends here instead of hitting a closed fd.
        "if sys.argv[2] == 'success':\n"
        "    with (Path(sys.argv[1]) / 'rank-1.log').open('a'):\n"
        "        time.sleep(1.2)\n"
    )
    return subprocess.Popen(
        [sys.executable, str(script), str(tmp_path), mode],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
    )


def test_completed_worker_cancels_the_diagnostic_timer(tmp_path):
    process = start_worker(tmp_path, "success")
    stdout, stderr = process.communicate(timeout=10)
    assert process.returncode == 0, stderr
    assert not stdout and not stderr
    assert (tmp_path / "rank-1.log").read_text() == (
        f"rank=1 pid={process.pid}\nphase=cpu_fixture\nphase=complete\n"
    )


@pytest.mark.parametrize("terminate", [False, True])
def test_periodic_worker_dumps_are_attributed_nonfatal_and_survive_kill(
    tmp_path, terminate
):
    process = start_worker(tmp_path, "stall")
    path = tmp_path / "rank-1.log"
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            output = path.read_text() if path.exists() else ""
            if (
                output.count("Timeout (") >= 2
                and output.count("in wait_for_release") >= 2
            ):
                break
            time.sleep(0.02)
        else:
            pytest.fail("The stalled CPU worker did not write repeated stacks")
        assert f"rank=1 pid={process.pid}\nphase=cpu_fixture\n" in output
        assert process.poll() is None  # The diagnostic did not end the worker.
        if terminate:
            process.kill()
            process.communicate(timeout=10)
            assert process.returncode != 0
        else:
            _, stderr = process.communicate(input="continue\n", timeout=10)
            assert process.returncode == 0, stderr
            assert "phase=complete" in path.read_text()
        assert "in wait_for_release" in path.read_text()
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
