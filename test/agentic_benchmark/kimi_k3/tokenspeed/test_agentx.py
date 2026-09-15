# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import copy
import importlib.util
import json
import os
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location(
    "agentx_result", ROOT / "agentx_result.py"
)
RESULT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RESULT)


class ManifestTest(unittest.TestCase):
    def test_server_parameters_are_recorded_without_mandating_a_launcher(self):
        environment = dict(
            MODEL_DIR="/models/target",
            DRAFT_DIR="/models/draft",
            MODEL_REVISION="target-revision",
            DRAFT_REVISION="draft-revision",
            SERVER_VENV="/opt/server-venv",
            GPU_MEMORY_UTILIZATION="0.9",
            SERVER_SEED="7",
        )
        baseline = RESULT.server_configuration(environment)
        self.assertEqual(baseline, environment)
        for field, changed in (
            ("MODEL_DIR", "/models/other-target"),
            ("DRAFT_DIR", "/models/other-draft"),
            ("MODEL_REVISION", "other-target-revision"),
            ("DRAFT_REVISION", "other-draft-revision"),
            ("SERVER_SEED", "8"),
            ("GPU_MEMORY_UTILIZATION", "0.8"),
            ("SERVER_VENV", ""),
        ):
            with self.subTest(field=field):
                updated = RESULT.server_configuration(
                    dict(environment, **{field: changed})
                )
                self.assertNotEqual(baseline, updated)
                self.assertEqual(updated[field], changed)
        for field in environment:
            with self.subTest(missing=field):
                incomplete = dict(environment)
                del incomplete[field]
                self.assertIsNone(RESULT.server_configuration(incomplete)[field])


class ResultTest(unittest.TestCase):
    def test_phase_and_request_evidence(self):
        summary = {
            "status": "completed",
            "error_summary": [],
            "scenario": {"mode": "smoke"},
            "submission_valid": False,
            "metrics": {"request_count": {"avg": 1}},
        }
        records = [
            {
                "metadata": {
                    "benchmark_phase": "profiling",
                    "was_cancelled": False,
                    "context_overflow_skip": False,
                },
                "metrics": {"output_sequence_length": {"value": 7}},
            }
        ]
        for case in (
            "success",
            "hidden_cancel_smoke",
            "hidden_cancel_benchmark",
            "error",
            "overflow",
            "missing_phase",
            "grace_timeout",
            "count_mismatch",
        ):
            with self.subTest(case=case):
                s, rows = copy.deepcopy(summary), copy.deepcopy(records)
                cancelled = int(case.startswith("hidden_cancel"))
                log = f"Phase profiling (profiling) complete | completed=1, cancelled={cancelled}, errors=0"
                if case == "hidden_cancel_benchmark":
                    s["scenario"]["mode"] = "benchmark"
                elif case == "error":
                    rows[0]["error"] = {"message": "failed"}
                elif case == "overflow":
                    rows[0]["metadata"]["context_overflow_skip"] = True
                elif case == "grace_timeout":
                    s["scenario"]["mode"] = "benchmark"
                    log += " | elapsed=2400.00s | grace_period_timeout=True"
                elif case == "missing_phase":
                    log = ""
                elif case == "count_mismatch":
                    s["metrics"]["request_count"]["avg"] = 2
                if case in ("success", "hidden_cancel_benchmark"):
                    self.assertEqual(RESULT.audit(s, rows, log)["cancelled"], cancelled)
                else:
                    with self.assertRaises(ValueError):
                        RESULT.audit(s, rows, log)


class LifecycleTest(unittest.TestCase):
    def test_cleanup_and_hold(self):
        for case in (
            "success",
            "client_failure",
            "hold",
            "hold_marker",
            "hold_sigterm",
            "hold_sigint",
            "occupied",
            "startup_failure",
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                binaries = root / "bin"
                binaries.mkdir()
                scripts = {
                    "squeue": """case "$*" in
*"%u"*) id -un ;;
*"%T"*) echo RUNNING ;;
*"%N"*) echo testnode ;;
*"%i %j"*) if [ -f "$TEST_ROOT/step" ]; then cat "$TEST_ROOT/step"; fi; echo "123.99 unrelated_server" ;;
esac""",
                    "scontrol": """if [ "$1" = show ] && [ "$2" = hostnames ]; then echo testnode; else echo allocation; fi""",
                    "srun": """for arg in "$@"; do
case "$arg" in --job-name=*) echo "123.1 ${arg#--job-name=}" > "$TEST_ROOT/step"; echo $$ > "$TEST_ROOT/server-pid"; exec /usr/bin/sleep 30 ;; esac
done
exit 0""",
                    "scancel": '''echo "$*" >> "$TEST_ROOT/cancelled"
[ "$1" = 123.1 ] && kill "$(cat "$TEST_ROOT/server-pid")"''',
                    "curl": """if [ "$CASE" = occupied ]; then exit 0; fi
if [ ! -f "$TEST_ROOT/step" ] || [ "$CASE" = startup_failure ]; then exit 7; fi
echo "{}"
""",
                    "sleep": "exec /usr/bin/sleep 0.05",
                    "python": """if [ "$1" = -c ]; then
 touch "$TEST_ROOT/client-called"
 if [ "$CASE" = hold_marker ]; then touch "$RUN_ROOT/hold-after-client"; fi
 [ "$CASE" != client_failure ]
else exit 0
fi""",
                }
                for name, body in scripts.items():
                    path = binaries / name
                    path.write_text("#!/bin/bash\n" + body + "\n")
                    path.chmod(0o755)
                server = root / "server.sh"
                server.touch()
                scenario = root / "scenario.json"
                scenario.write_text("{}")
                run = root / "output"
                env = dict(
                    os.environ,
                    PATH=str(binaries) + os.pathsep + os.environ["PATH"],
                    TEST_ROOT=str(root),
                    CASE=case,
                    SLURM_JOB_ID="123",
                    SOURCE_ROOT=str(ROOT.parents[4]),
                    SERVER_SCRIPT=str(server),
                    CONTAINER_IMAGE="test-image",
                    CONTAINER_MOUNTS="/tmp:/tmp",
                    CLIENT_PYTHON=str(binaries / "python"),
                    SCENARIO_FILE=str(scenario),
                    MODEL_NAME="model",
                    TOKENIZER_PATH="tokenizer",
                    DATASET_PATH=str(root),
                    CONCURRENCY="1",
                    DURATION="60",
                    SEED="1",
                    API_PORT="8000",
                    READINESS_PATH="/readiness",
                    READINESS_TIMEOUT="1",
                    CLIENT_TIMEOUT="90",
                    HOLD_AFTER_RUN=(
                        "1" if case in ("hold", "hold_sigterm", "hold_sigint") else "0"
                    ),
                    RUN_ROOT=str(run),
                )
                process = subprocess.Popen(
                    ["bash", str(ROOT / "agentx.slurm")],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                try:
                    if case.startswith("hold"):
                        deadline = time.monotonic() + 5
                        while (
                            not (run / "allocation-held.txt").exists()
                            and time.monotonic() < deadline
                        ):
                            time.sleep(0.05)
                        self.assertTrue((run / "allocation-held.txt").exists())
                        self.assertIsNone(process.poll())
                        if case in ("hold_sigterm", "hold_sigint"):
                            process.send_signal(
                                signal.SIGTERM
                                if case == "hold_sigterm"
                                else signal.SIGINT
                            )
                        else:
                            (run / "release-requested").touch()
                    stdout, stderr = process.communicate(timeout=8)
                    self.assertEqual(
                        process.returncode == 0,
                        case in ("success", "hold", "hold_marker"),
                        stderr,
                    )
                    if case in ("hold_sigterm", "hold_sigint"):
                        expected = 143 if case == "hold_sigterm" else 130
                        self.assertEqual(process.returncode, expected)
                        self.assertEqual(
                            int((run / "client-exit-code.txt").read_text()), expected
                        )
                    if case == "occupied":
                        self.assertFalse((root / "step").exists())
                    else:
                        self.assertEqual(
                            (root / "cancelled").read_text().splitlines(), ["123.1"]
                        )
                    if case in ("occupied", "startup_failure"):
                        self.assertFalse((root / "client-called").exists())
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.communicate(timeout=5)
                    # Also reap the fake server if the behavior under test leaked it.
                    if (root / "server-pid").exists():
                        try:
                            os.kill(
                                int((root / "server-pid").read_text()), signal.SIGTERM
                            )
                        except ProcessLookupError:
                            pass


if __name__ == "__main__":
    unittest.main()
