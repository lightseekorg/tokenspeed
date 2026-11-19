import sys
import time

import psutil
import pytest
from native_timeout import run


@pytest.mark.parametrize("code", [0, 7])
def test_command_exit_status_is_preserved(code):
    assert (
        run(
            [sys.executable, "-c", f"raise SystemExit({code})"],
            dump_after=2,
            timeout=3,
            debugger=["false"],
        )
        == code
    )


def test_timeout_kills_workers_even_when_debugger_fails(tmp_path, capfd):
    child_pid_path = tmp_path / "child-pid"
    child = "import time; time.sleep(30)"
    parent = (
        "import pathlib, subprocess, sys, time; "
        f"child = subprocess.Popen([sys.executable, '-c', {child!r}]); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid)); "
        "time.sleep(30)"
    )
    assert (
        run(
            [sys.executable, "-c", parent],
            dump_after=1,
            timeout=2,
            debugger=["false"],
        )
        == 124
    )
    child_pid = int(child_pid_path.read_text())
    # An orphan can remain a zombie briefly until init reaps it; it must no
    # longer run or hold GPU resources when the wrapper returns.
    deadline = time.monotonic() + 5
    while True:
        try:
            if psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        assert time.monotonic() < deadline
        time.sleep(0.01)
    output = capfd.readouterr().out
    assert "Native stack dump failed" in output
    assert "Command timed out" in output
