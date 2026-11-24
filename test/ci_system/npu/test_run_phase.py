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

"""CPU-only tests for the NPU phase runner using real subprocesses and fixtures."""

import argparse
import contextlib
import hashlib
import io
import json
import os
import runpy
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from run_phase import (
    PhaseInterrupted,
    capture_test,
    execute_phase,
    handle_interrupt,
    load_inputs,
    main,
    parse_args,
    positive_seconds,
    report_error,
    run_operator,
    save_failure_summary,
    save_text,
    stop_process_group,
)

ROOT = Path(__file__).resolve().parents[3]
CHANGES = ROOT / "test" / "ci_system" / "npu" / "changes.py"
REPORTING = ROOT / "test" / "ci_system" / "npu" / "reporting.py"


def reporting_namespace():
    """Load the real reporting module exactly as load_inputs does."""
    return runpy.run_path(str(REPORTING))


def operator(name, accuracy_cases, performance_configs):
    return {
        "name": name,
        "accuracy_cases": accuracy_cases,
        "performance_configs": performance_configs,
        "script": f"test/ci/npu/tests/{name}-test.py",
    }


def accuracy_plan(*names):
    return [
        {"case": {"name": name, "width": 64}, "metadata": {"dtype": "float32"}}
        for name in names
    ]


def performance_plan(*names):
    return [
        {"config": {"name": name, "width": 64}, "metadata": {"dtype": "float32"}}
        for name in names
    ]


def write_manifest(root, operators):
    manifest = {
        "schema_version": 3,
        "watch_paths": ["test/ci/npu/**", "test/ci_system/npu/**"],
        "expected_plans": "test/ci/npu/expected-plans.json",
        "execution": {"phases": ["accuracy", "performance"], "fail_fast": True},
        "performance_policy": {"mode": "record_only"},
        "operators": operators,
    }
    path = root / "test" / "ci" / "npu" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def write_expected_plans(root, operators, plans):
    manifest_path = root / "test" / "ci" / "npu" / "manifest.json"
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    sources = {}
    for item in operators:
        script = root / item["script"]
        sources[item["script"]] = hashlib.sha256(script.read_bytes()).hexdigest()
    plan = {
        "schema_version": 2,
        "manifest_sha256": manifest_sha256,
        "source_sha256": sources,
        "operators": {item["name"]: plans[item["name"]] for item in operators},
    }
    path = root / "test" / "ci" / "npu" / "expected-plans.json"
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan


def write_operator_script(root, name, content):
    path = root / "test" / "ci" / "npu" / "tests" / f"{name}-test.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def pass_accuracy_script(operator):
    return """import json, sys

def emit(event, **fields):
    value = {"schema_version": 1, "operator": "%s", "phase": "accuracy", "event": event}
    value.update(fields)
    print("NPU_RESULT_JSON=" + json.dumps(value, allow_nan=False), flush=True)

mode = sys.argv[sys.argv.index("--mode") + 1]
if mode != "accuracy":
    sys.exit(2)
emit("started", expected_cases=1, case_names=["width64"])
case = {"name": "width64", "width": 64}
emit("case_started", case=case, dtype="float32")
emit("case_completed", case=case, dtype="float32", status="passed")
emit("completed", expected_cases=1, completed_cases=1, passed_cases=1, failed_cases=0)
sys.exit(0)
""" % operator


def fail_accuracy_script(operator):
    return """import json, sys

def emit(event, **fields):
    value = {"schema_version": 1, "operator": "%s", "phase": "accuracy", "event": event}
    value.update(fields)
    print("NPU_RESULT_JSON=" + json.dumps(value, allow_nan=False), flush=True)

mode = sys.argv[sys.argv.index("--mode") + 1]
if mode != "accuracy":
    sys.exit(2)
emit("started", expected_cases=1, case_names=["width64"])
case = {"name": "width64", "width": 64}
emit("case_started", case=case, dtype="float32")
emit("case_completed", case=case, dtype="float32", status="failed")
emit(
    "failed",
    expected_cases=1,
    completed_cases=1,
    passed_cases=0,
    failed_cases=1,
    current_case="width64",
    error_type="RuntimeError",
    error="boom",
)
sys.exit(1)
""" % operator


def pass_performance_script(operator):
    return """import json, sys

def emit(event, **fields):
    value = {"schema_version": 1, "operator": "%s", "phase": "performance", "event": event}
    value.update(fields)
    print("NPU_RESULT_JSON=" + json.dumps(value, allow_nan=False), flush=True)

mode = sys.argv[sys.argv.index("--mode") + 1]
if mode != "performance":
    sys.exit(2)
emit("started", expected_configs=1, config_names=["cfg0"])
config = {"name": "cfg0", "width": 64}
emit(
    "config_started",
    config=config,
    dtype="float32",
    warmup=0,
    iterations=1,
    expected_samples=2,
)
emit("sample", config="cfg0", sample_index=0, latency_ms=1.0)
emit("sample", config="cfg0", sample_index=1, latency_ms=2.0)
emit(
    "config_completed",
    config=config,
    dtype="float32",
    warmup=0,
    iterations=1,
    sample_count=2,
    samples_ms=[1.0, 2.0],
    median_latency_ms=1.5,
    threshold_ms=None,
    performance_verdict="not_evaluated",
)
emit("completed", expected_configs=1, completed_configs=1)
sys.exit(0)
""" % operator


def sleeping_script():
    return "import time\ntime.sleep(30)\n"


def build_repo(root, operators, scripts, kernel_dirs=True):
    """Create a self-contained checkout using the real scanner and reporting code."""
    npu_dir = root / "test" / "ci_system" / "npu"
    npu_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(CHANGES, npu_dir / "changes.py")
    shutil.copy(REPORTING, npu_dir / "reporting.py")
    plans = {}
    for item in operators:
        name = item["name"]
        write_operator_script(root, name, scripts[name](name))
        plans[name] = {
            # The fixture operator scripts emit one case/config with these names.
            "accuracy": accuracy_plan("width64"),
            "performance": performance_plan("cfg0"),
            "timing": {"warmup": 0, "iterations": 1, "sample_count": 2},
        }
    manifest = write_manifest(root, operators)
    write_expected_plans(root, operators, plans)
    if kernel_dirs:
        (root / "tokenspeed-kernel" / "python").mkdir(parents=True, exist_ok=True)
        (root / "tokenspeed-kernel-npu" / "python").mkdir(parents=True, exist_ok=True)
    return manifest, plans


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


class PositiveSecondsTest(unittest.TestCase):
    def test_accepts_positive_integer(self):
        self.assertEqual(positive_seconds("7"), 7)

    def test_rejects_zero(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            positive_seconds("0")

    def test_rejects_negative(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            positive_seconds("-3")

    def test_rejects_non_integer(self):
        with self.assertRaises(ValueError):
            positive_seconds("soon")


class ParseArgsTest(unittest.TestCase):
    def test_full_valid_arguments(self):
        args = parse_args(
            [
                "--repo-root",
                "/tmp/repo",
                "--phase",
                "accuracy",
                "--output-dir",
                "/tmp/out",
                "--timeout-seconds",
                "30",
            ]
        )
        self.assertEqual(args.repo_root, Path("/tmp/repo"))
        self.assertEqual(args.phase, "accuracy")
        self.assertEqual(args.output_dir, Path("/tmp/out"))
        self.assertEqual(args.timeout_seconds, 30)
        self.assertIsNone(args.github_summary)

    def test_missing_required_argument_exits(self):
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--repo-root",
                    "/tmp/repo",
                    "--output-dir",
                    "/tmp/out",
                    "--timeout-seconds",
                    "30",
                ]
            )

    def test_rejects_unknown_phase(self):
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--repo-root",
                    "/tmp/repo",
                    "--phase",
                    "smoke",
                    "--output-dir",
                    "/tmp/out",
                    "--timeout-seconds",
                    "30",
                ]
            )

    def test_rejects_non_positive_timeout(self):
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--repo-root",
                    "/tmp/repo",
                    "--phase",
                    "accuracy",
                    "--output-dir",
                    "/tmp/out",
                    "--timeout-seconds",
                    "0",
                ]
            )

    def test_github_summary_defaults_to_environment(self):
        previous = os.environ.get("GITHUB_STEP_SUMMARY")
        os.environ["GITHUB_STEP_SUMMARY"] = "/tmp/step-summary.md"
        try:
            args = parse_args(
                [
                    "--repo-root",
                    "/tmp/repo",
                    "--phase",
                    "performance",
                    "--output-dir",
                    "/tmp/out",
                    "--timeout-seconds",
                    "30",
                ]
            )
            self.assertEqual(args.github_summary, Path("/tmp/step-summary.md"))
        finally:
            if previous is None:
                os.environ.pop("GITHUB_STEP_SUMMARY", None)
            else:
                os.environ["GITHUB_STEP_SUMMARY"] = previous

    def test_github_summary_flag_overrides_environment(self):
        args = parse_args(
            [
                "--repo-root",
                "/tmp/repo",
                "--phase",
                "accuracy",
                "--output-dir",
                "/tmp/out",
                "--timeout-seconds",
                "30",
                "--github-summary",
                "/tmp/explicit.md",
            ]
        )
        self.assertEqual(args.github_summary, Path("/tmp/explicit.md"))


class LoadInputsTest(unittest.TestCase):
    def test_loads_real_repo_consistently(self):
        root, manifest, plan, reporting = load_inputs(ROOT)
        self.assertEqual(root, ROOT.resolve())
        self.assertEqual(
            {item["name"] for item in manifest["operators"]}, set(plan["operators"])
        )
        manifest_path = ROOT / "test" / "ci" / "npu" / "manifest.json"
        self.assertEqual(
            plan["manifest_sha256"],
            hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        )
        for key in (
            "require",
            "summarize_phase",
            "phase_succeeded",
            "make_report",
            "render_summary",
            "save_report",
        ):
            self.assertIn(key, reporting)

    def test_loads_mini_repo_consistently(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "repo"
            build_repo(root, [operator("scatter", 1, 1)], {"scatter": pass_accuracy_script})
            _, manifest, plan, reporting = load_inputs(root)
            self.assertEqual(manifest["operators"][0]["name"], "scatter")
            case = plan["operators"]["scatter"]["accuracy"][0]["case"]
            self.assertEqual(case["name"], "width64")
            self.assertIn("summarize_phase", reporting)


class CaptureTestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def capture(self, command, name="test.log", timeout=5):
        log = self.root / name
        with quiet():
            return capture_test(command, self.root, log, timeout)

    def test_success_round_trips_output(self):
        result = self.capture([sys.executable, "-c", "print('hello')"])
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(result["log_saved"])
        self.assertIsNone(result["capture_error"])
        self.assertIn("hello", Path(result["log_path"]).read_text(encoding="utf-8"))

    def test_merges_stderr_into_log(self):
        code = "import sys; sys.stderr.write('warning'); print('out')"
        result = self.capture([sys.executable, "-c", code])
        text = Path(result["log_path"]).read_text(encoding="utf-8")
        self.assertIn("warning", text)
        self.assertIn("out", text)

    def test_preserves_nonzero_exit_code(self):
        result = self.capture([sys.executable, "-c", "import sys; sys.exit(7)"])
        self.assertEqual(result["exit_code"], 7)
        self.assertTrue(result["log_saved"])
        self.assertIsNone(result["capture_error"])

    def test_timeout_records_error_and_stops_process(self):
        result = self.capture(
            [sys.executable, "-c", "import time; time.sleep(30)"], timeout=1
        )
        self.assertIsNotNone(result["capture_error"])
        self.assertEqual(result["capture_error"]["type"], "TimeoutError")
        self.assertFalse(result["log_saved"])
        self.assertIsNotNone(result["exit_code"])
        self.assertLess(result["exit_code"], 0)

    def test_existing_log_path_is_reported_as_error(self):
        log = self.root / "taken.log"
        log.write_text("occupied", encoding="utf-8")
        with quiet():
            result = capture_test(
                [sys.executable, "-c", "print('hello')"], self.root, log, 5
            )
        self.assertIsNotNone(result["capture_error"])
        self.assertEqual(result["capture_error"]["type"], "FileExistsError")
        self.assertFalse(result["log_saved"])
        self.assertIsNone(result["exit_code"])


class StopProcessGroupTest(unittest.TestCase):
    def test_stops_live_process(self):
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            start_new_session=True,
        )
        stop_process_group(process)
        self.assertIsNotNone(process.returncode)
        self.assertNotEqual(process.returncode, 0)

    def test_stops_already_exited_process_without_error(self):
        process = subprocess.Popen([sys.executable, "-c", "pass"])
        process.wait()
        stop_process_group(process)  # must not raise


class SaveTextTest(unittest.TestCase):
    def test_writes_and_verifies_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "evidence.json"
            save_text(path, '{"ok": true}\n')
            self.assertEqual(path.read_text(encoding="utf-8"), '{"ok": true}\n')

    def test_rejects_existing_path(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "evidence.json"
            path.write_text("x", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                save_text(path, "y")


class ReportErrorTest(unittest.TestCase):
    def test_prints_to_stderr(self):
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            report_error("boom")
        self.assertIn("boom", stream.getvalue())

    def test_actions_mode_escapes_error_message(self):
        out, err = io.StringIO(), io.StringIO()
        previous = os.environ.get("GITHUB_ACTIONS")
        os.environ["GITHUB_ACTIONS"] = "true"
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                report_error("100% done\nline2\r\n")
        finally:
            if previous is None:
                os.environ.pop("GITHUB_ACTIONS", None)
            else:
                os.environ["GITHUB_ACTIONS"] = previous
        self.assertIn("100% done\nline2\r\n", err.getvalue())
        self.assertIn("::error::100%25 done%0Aline2%0D%0A", out.getvalue())


class SaveFailureSummaryTest(unittest.TestCase):
    def test_appends_html_escaped_message(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "summary.md"
            save_failure_summary(path, "accuracy", "bad <x> & boom")
            text = path.read_text(encoding="utf-8")
            self.assertIn("### ARM NPU phase execution failed", text)
            self.assertIn("bad &lt;x&gt; &amp; boom", text)
            self.assertIn("`accuracy`", text)

    def test_none_path_is_noop(self):
        save_failure_summary(None, "accuracy", "boom")


class InterruptTest(unittest.TestCase):
    def test_phase_interrupted_carries_signal_number(self):
        exc = PhaseInterrupted(15)
        self.assertEqual(exc.signum, 15)
        self.assertIn("15", str(exc))

    def test_handle_interrupt_ignores_signals_and_raises(self):
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            with self.assertRaises(PhaseInterrupted) as raised:
                handle_interrupt(signal.SIGINT, None)
            self.assertEqual(raised.exception.signum, signal.SIGINT)
            self.assertEqual(signal.getsignal(signal.SIGINT), signal.SIG_IGN)
            self.assertEqual(signal.getsignal(signal.SIGTERM), signal.SIG_IGN)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


class RunOperatorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "repo"
        (self.root / "test" / "ci" / "npu" / "tests").mkdir(parents=True)
        self.output = Path(self.tmp.name) / "out"
        self.output.mkdir()
        self.reporting = reporting_namespace()
        self.entry = operator("scatter", 1, 1)
        self.plan = {
            "accuracy": accuracy_plan("width64"),
            "performance": performance_plan("cfg0"),
            "timing": {"warmup": 0, "iterations": 1, "sample_count": 2},
        }

    def run_op(self, script, phase="accuracy", timeout=10):
        write_operator_script(self.root, "scatter", script)
        with quiet():
            return run_operator(
                self.root,
                self.entry,
                phase,
                self.plan,
                self.reporting,
                self.output,
                timeout,
            )

    def test_accuracy_pass(self):
        result = self.run_op(pass_accuracy_script("scatter"))
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(result["structured_log_valid"])
        self.assertTrue(result["log_saved"])
        self.assertIsNone(result.get("evidence_error"))
        directory = self.output / "scatter"
        self.assertTrue((directory / "process.json").is_file())
        self.assertTrue((directory / "result.json").is_file())
        self.assertEqual((directory / "accuracy.rc").read_text(), "0\n")
        process = json.loads((directory / "process.json").read_text(encoding="utf-8"))
        self.assertEqual(process["exit_code"], 0)
        self.assertTrue(process["log_saved"])

    def test_accuracy_failure(self):
        result = self.run_op(fail_accuracy_script("scatter"))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertEqual(result["error"], "boom")
        self.assertEqual(result["exit_code"], 1)
        self.assertTrue(result["structured_log_valid"])
        directory = self.output / "scatter"
        self.assertEqual((directory / "accuracy.rc").read_text(), "1\n")

    def test_performance_completed(self):
        result = self.run_op(
            pass_performance_script("scatter"), phase="performance"
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["performance_verdict"], "not_evaluated")
        self.assertEqual(result["completed_configs"], 1)
        directory = self.output / "scatter"
        self.assertEqual((directory / "performance.rc").read_text(), "0\n")

    def test_empty_log_without_structured_records_fails(self):
        # An operator that prints nothing still runs; the empty log then fails
        # validation, so the run is reported as failed rather than crashing.
        result = self.run_op("import sys\nsys.exit(0)\n")
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["structured_log_valid"])
        self.assertEqual(result["error_type"], "ReportError")
        self.assertIsNone(result.get("evidence_error"))

    def test_evidence_write_conflict_recorded(self):
        # A pre-existing operator directory makes run_operator's own evidence
        # step fail; the failure is captured as evidence_error, not a crash.
        (self.output / "scatter").mkdir()
        result = self.run_op(pass_accuracy_script("scatter"))
        self.assertEqual(result["status"], "failed")
        self.assertIsNotNone(result.get("evidence_error"))
        self.assertEqual(result["evidence_error"]["type"], "FileExistsError")
        self.assertIsNone(result["exit_code"])

    def test_capture_error_becomes_failure_with_validation_error(self):
        result = self.run_op(sleeping_script(), timeout=1)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_type"], "TimeoutError")
        self.assertIsNotNone(result.get("validation_error"))
        self.assertLess(result["exit_code"], 0)


class ExecutePhaseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "repo"
        self.reporting = reporting_namespace()

    def phase(self, operators, scripts, phase="accuracy", github_summary=None, timeout=10):
        build_repo(self.root, operators, scripts)
        _, manifest, plan, _ = load_inputs(self.root)
        output = Path(self.tmp.name) / "phase-out"
        with quiet():
            code = execute_phase(
                self.root,
                manifest,
                plan,
                self.reporting,
                phase,
                output,
                timeout,
                github_summary,
            )
        return code, output

    def test_phase_success(self):
        code, output = self.phase(
            [operator("scatter", 1, 1)], {"scatter": pass_accuracy_script}
        )
        self.assertEqual(code, 0)
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["exit_code"], 0)
        self.assertTrue((output / "summary.md").is_file())
        context = json.loads((output / "context.json").read_text(encoding="utf-8"))
        self.assertTrue(context["manifest_sha256"])

    def test_phase_fails_fast_and_marks_later_operator_not_run(self):
        code, output = self.phase(
            [
                operator("scatter", 1, 1),
                operator("ghost", 1, 1),
            ],
            {"scatter": fail_accuracy_script, "ghost": pass_accuracy_script},
        )
        self.assertEqual(code, 1)
        self.assertFalse((output / "ghost").exists())
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        by_name = {item["operator"]: item for item in report["operators"]}
        self.assertEqual(by_name["scatter"]["status"], "failed")
        self.assertEqual(by_name["ghost"]["status"], "not_run")

    def test_unknown_phase_rejected(self):
        operators = [operator("scatter", 1, 1)]
        build_repo(self.root, operators, {"scatter": pass_accuracy_script})
        manifest = load_inputs(self.root)[1]
        plan = json.loads(
            (self.root / "test" / "ci" / "npu" / "expected-plans.json").read_text(
                encoding="utf-8"
            )
        )
        output = Path(self.tmp.name) / "phase-out"
        with self.assertRaisesRegex(ValueError, "Unknown phase"):
            execute_phase(
                self.root, manifest, plan, self.reporting, "smoke", output, 10
            )

    def test_actions_mode_emits_group_markers(self):
        stream = io.StringIO()
        code = None
        previous = os.environ.get("GITHUB_ACTIONS")
        os.environ["GITHUB_ACTIONS"] = "true"
        try:
            build_repo(
                self.root,
                [operator("scatter", 1, 1)],
                {"scatter": pass_accuracy_script},
            )
            manifest = load_inputs(self.root)[1]
            plan = json.loads(
                (self.root / "test" / "ci" / "npu" / "expected-plans.json").read_text(
                    encoding="utf-8"
                )
            )
            output = Path(self.tmp.name) / "phase-out"
            with contextlib.redirect_stdout(stream):
                code = execute_phase(
                    self.root, manifest, plan, self.reporting, "accuracy", output, 10
                )
        finally:
            if previous is None:
                os.environ.pop("GITHUB_ACTIONS", None)
            else:
                os.environ["GITHUB_ACTIONS"] = previous
        self.assertEqual(code, 0)
        self.assertIn("::group::scatter/accuracy", stream.getvalue())
        self.assertIn("::endgroup::", stream.getvalue())

    def test_github_summary_appended(self):
        with tempfile.TemporaryDirectory() as td:
            summary = Path(td) / "step-summary.md"
            code, output = self.phase(
                [operator("scatter", 1, 1)],
                {"scatter": pass_accuracy_script},
                github_summary=summary,
            )
            self.assertEqual(code, 0)
            text = summary.read_text(encoding="utf-8")
            self.assertIn("## Ascend NPU operator accuracy tests", text)


class MainTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def base_args(self, root, output, phase="accuracy", **extra):
        args = [
            "--repo-root",
            str(root),
            "--phase",
            phase,
            "--output-dir",
            str(output),
            "--timeout-seconds",
            "30",
        ]
        for key, value in extra.items():
            args.extend([f"--{key.replace('_', '-')}", str(value)])
        return args

    def test_main_success(self):
        root = Path(self.tmp.name) / "repo"
        output = Path(self.tmp.name) / "out"
        build_repo(root, [operator("scatter", 1, 1)], {"scatter": pass_accuracy_script})
        with quiet():
            code = main(self.base_args(root, output))
        self.assertEqual(code, 0)
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "passed")

    def test_main_returns_one_on_operator_failure(self):
        root = Path(self.tmp.name) / "repo"
        output = Path(self.tmp.name) / "out"
        build_repo(root, [operator("scatter", 1, 1)], {"scatter": fail_accuracy_script})
        stream = io.StringIO()
        with quiet(), contextlib.redirect_stderr(stream):
            code = main(self.base_args(root, output))
        self.assertEqual(code, 1)
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")

    def test_main_missing_source_directory_returns_one(self):
        root = Path(self.tmp.name) / "repo"
        output = Path(self.tmp.name) / "out"
        summary = Path(self.tmp.name) / "step-summary.md"
        build_repo(
            root,
            [operator("scatter", 1, 1)],
            {"scatter": pass_accuracy_script},
            kernel_dirs=False,
        )
        stream = io.StringIO()
        with quiet(), contextlib.redirect_stderr(stream):
            code = main(
                self.base_args(
                    root,
                    output,
                    **{"github-summary": summary},
                )
            )
        self.assertEqual(code, 1)
        text = summary.read_text(encoding="utf-8")
        self.assertIn("Phase execution failed", text)

    def test_main_restores_pythonpath_and_handlers(self):
        root = Path(self.tmp.name) / "repo"
        output = Path(self.tmp.name) / "out"
        build_repo(root, [operator("scatter", 1, 1)], {"scatter": pass_accuracy_script})
        previous_pp = os.environ.get("PYTHONPATH")
        previous_int = signal.getsignal(signal.SIGINT)
        previous_term = signal.getsignal(signal.SIGTERM)
        os.environ["PYTHONPATH"] = "sentinel"
        try:
            with quiet():
                code = main(self.base_args(root, output))
        finally:
            restored_pp = os.environ.get("PYTHONPATH")
            if previous_pp is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = previous_pp
        self.assertEqual(code, 0)
        self.assertEqual(restored_pp, "sentinel")
        self.assertEqual(signal.getsignal(signal.SIGINT), previous_int)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous_term)

    def test_main_actions_requires_github_summary(self):
        root = Path(self.tmp.name) / "repo"
        output = Path(self.tmp.name) / "out"
        build_repo(root, [operator("scatter", 1, 1)], {"scatter": pass_accuracy_script})
        previous = os.environ.get("GITHUB_ACTIONS")
        os.environ["GITHUB_ACTIONS"] = "true"
        stream = io.StringIO()
        try:
            with quiet(), contextlib.redirect_stderr(stream):
                code = main(self.base_args(root, output))
        finally:
            if previous is None:
                os.environ.pop("GITHUB_ACTIONS", None)
            else:
                os.environ["GITHUB_ACTIONS"] = previous
        self.assertEqual(code, 1)
        self.assertIn("GITHUB_STEP_SUMMARY is required inside Actions", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
