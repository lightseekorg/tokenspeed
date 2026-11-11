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

"""The MI450 simulator job's report digest: when a run without test results
passes."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

DIGEST = Path(__file__).resolve().parent / "mi450_sim_report_digest.py"


def _digest(tmp_path: Path, events: list[dict], pytest_status: int) -> int:
    log = tmp_path / "report.jsonl"
    log.write_text("".join(json.dumps(event) + "\n" for event in events))
    run = subprocess.run(
        [sys.executable, str(DIGEST), "--pytest-status", str(pytest_status), str(log)],
        capture_output=True,
        text=True,
        check=False,
    )
    return run.returncode


START = {"$report_type": "SessionStart", "pytest_version": "9.2.0"}
FINISH = {"$report_type": "SessionFinish", "exitstatus": 5}


@pytest.mark.parametrize(
    "events, pytest_status, expected",
    [
        # The changed-paths selection emptied the run: pytest passed.
        ([START, FINISH], 0, 0),
        # Nothing collected by itself: pytest exits 5.
        ([START, FINISH], 5, 1),
        # Killed before the session finished.
        ([START], 0, 1),
    ],
    ids=["selection-emptied", "nothing-collected", "killed"],
)
def test_a_run_without_results_passes_only_when_pytest_passed(
    tmp_path, events, pytest_status, expected
):
    assert _digest(tmp_path, events, pytest_status) == expected
