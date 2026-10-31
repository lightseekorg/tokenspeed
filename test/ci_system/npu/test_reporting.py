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

"""CPU-only tests for NPU report validation using real file and log fixtures."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from reporting import (
    RecordCursor,
    ReportError,
    check_count,
    decode_record,
    latency_value,
    make_report,
    read_expected_plan,
    read_records,
    same_json,
    save_report,
    summarize_phase,
    validate_accuracy,
    validate_performance,
)

OPERATOR = "scatter"


def make_operator(name, accuracy_cases, performance_configs):
    return {
        "name": name,
        "accuracy_cases": accuracy_cases,
        "performance_configs": performance_configs,
        "script": f"test/ci/npu/tests/{name}-test.py",
    }


def make_manifest(operators):
    return {
        "schema_version": 3,
        "watch_paths": ["tokenspeed-kernel-npu/**"],
        "expected_plans": "test/ci/npu/expected-plans.json",
        "execution": {"phases": ["accuracy", "performance"], "fail_fast": True},
        "performance_policy": {"mode": "record_only"},
        "operators": operators,
    }


def write_json(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def materialize_plan(root, manifest):
    """Write manifest and operator scripts; return (plan dict, manifest sha256)."""
    manifest_path = root / "test/ci/npu/manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    sources = {}
    for item in manifest["operators"]:
        script = root / item["script"]
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(f"fixture for {item['name']}\n", encoding="utf-8")
        sources[item["script"]] = hashlib.sha256(script.read_bytes()).hexdigest()
    plan = {
        "schema_version": 2,
        "manifest_sha256": manifest_sha256,
        "operators": {
            item["name"]: {
                "accuracy": [
                    {"case": {"name": "case0"}, "metadata": {"dtype": "float32"}}
                ]
                * item["accuracy_cases"],
                "performance": [
                    {"config": {"name": "cfg0"}, "metadata": {"dtype": "float32"}}
                ]
                * item["performance_configs"],
            }
            for item in manifest["operators"]
        },
        "source_sha256": sources,
    }
    return plan, manifest_sha256


def record(operator, phase, event, **fields):
    value = {
        "schema_version": 1,
        "operator": operator,
        "phase": phase,
        "event": event,
    }
    value.update(fields)
    return value


def log_text(records, padding_lines=()):
    lines = list(padding_lines)
    for item in records:
        lines.append("NPU_RESULT_JSON=" + json.dumps(item))
    return "\n".join(lines) + "\n"


def accuracy_plan(*names):
    return [
        {"case": {"name": name, "width": 64}, "metadata": {"dtype": "float32"}}
        for name in names
    ]


def passed_accuracy_records(operator, planned):
    names = [item["case"]["name"] for item in planned]
    records = [
        record(
            operator,
            "accuracy",
            "started",
            expected_cases=len(planned),
            case_names=names,
        )
    ]
    for item in planned:
        records.append(
            record(
                operator,
                "accuracy",
                "case_started",
                case=item["case"],
                dtype="float32",
            )
        )
        records.append(
            record(
                operator,
                "accuracy",
                "case_completed",
                case=item["case"],
                dtype="float32",
                status="passed",
            )
        )
    records.append(
        record(
            operator,
            "accuracy",
            "completed",
            expected_cases=len(planned),
            completed_cases=len(planned),
            passed_cases=len(planned),
            failed_cases=0,
        )
    )
    return records


def failed_accuracy_records(operator, planned):
    names = [item["case"]["name"] for item in planned]
    records = [
        record(
            operator,
            "accuracy",
            "started",
            expected_cases=len(planned),
            case_names=names,
        )
    ]
    records.append(
        record(
            operator,
            "accuracy",
            "case_started",
            case=planned[0]["case"],
            dtype="float32",
        )
    )
    records.append(
        record(
            operator,
            "accuracy",
            "case_completed",
            case=planned[0]["case"],
            dtype="float32",
            status="failed",
        )
    )
    records.append(
        record(
            operator,
            "accuracy",
            "failed",
            expected_cases=len(planned),
            completed_cases=1,
            passed_cases=0,
            failed_cases=1,
            current_case=planned[0]["case"]["name"],
            error_type="RuntimeError",
            error="boom",
        )
    )
    return records


def passed_accuracy_result(operator):
    return {
        "operator": operator,
        "phase": "accuracy",
        "status": "passed",
        "exit_code": 0,
        "structured_log_valid": True,
        "log_saved": True,
        "log_path": "/tmp/fixture.log",
    }


def failed_accuracy_result(operator):
    return {
        "operator": operator,
        "phase": "accuracy",
        "status": "failed",
        "exit_code": 1,
        "structured_log_valid": True,
        "log_saved": True,
        "log_path": "/tmp/fixture.log",
        "error_type": "RuntimeError",
        "error": "boom",
    }


class ExpectedPlanTest(unittest.TestCase):
    """read_expected_plan gates the whole operator suite against manifest."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = make_manifest(
            [
                make_operator("scatter", 2, 3),
                make_operator("pick", 1, 1),
            ]
        )
        plan, self.manifest_sha256 = materialize_plan(self.root, self.manifest)
        self.plan_path = write_json(
            self.root, "test/ci/npu/expected-plans.json", plan
        )

    def read_plan(self):
        return read_expected_plan(
            self.plan_path, self.manifest, self.manifest_sha256, self.root
        )

    def test_valid_plan_passes(self):
        plan = self.read_plan()
        self.assertEqual(plan["schema_version"], 2)
        self.assertEqual(plan["manifest_sha256"], self.manifest_sha256)

    def test_rejects_wrong_schema_version(self):
        plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
        plan["schema_version"] = 1
        write_json(self.root, "test/ci/npu/expected-plans.json", plan)
        with self.assertRaisesRegex(ReportError, "schema_version=2"):
            self.read_plan()

    def test_rejects_manifest_sha_mismatch(self):
        plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
        plan["manifest_sha256"] = "0" * 64
        write_json(self.root, "test/ci/npu/expected-plans.json", plan)
        with self.assertRaisesRegex(ReportError, "different manifest"):
            self.read_plan()

    def test_rejects_operator_set_mismatch(self):
        plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
        del plan["operators"]["pick"]
        write_json(self.root, "test/ci/npu/expected-plans.json", plan)
        with self.assertRaisesRegex(ReportError, "operators differ from manifest"):
            self.read_plan()

    def test_rejects_source_set_mismatch(self):
        plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
        plan["source_sha256"]["test/ci/npu/tests/ghost-test.py"] = "0" * 64
        write_json(self.root, "test/ci/npu/expected-plans.json", plan)
        with self.assertRaisesRegex(ReportError, "source list differs from manifest"):
            self.read_plan()

    def test_rejects_invalid_source_hash_format(self):
        plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
        plan["source_sha256"]["test/ci/npu/tests/pick-test.py"] = "xyz"
        write_json(self.root, "test/ci/npu/expected-plans.json", plan)
        with self.assertRaisesRegex(ReportError, "Invalid source hash"):
            self.read_plan()

    def test_rejects_script_content_mismatch(self):
        script = self.root / "test/ci/npu/tests/pick-test.py"
        script.write_text("changed after plan generation\n", encoding="utf-8")
        with self.assertRaisesRegex(ReportError, "Script differs from expected plan"):
            self.read_plan()

    def test_rejects_accuracy_count_mismatch(self):
        plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
        plan["operators"]["scatter"]["accuracy"].pop()
        write_json(self.root, "test/ci/npu/expected-plans.json", plan)
        with self.assertRaisesRegex(
            ReportError, "scatter/accuracy: plan count differs from manifest"
        ):
            self.read_plan()

    def test_rejects_performance_count_mismatch(self):
        plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
        plan["operators"]["scatter"]["performance"].pop()
        write_json(self.root, "test/ci/npu/expected-plans.json", plan)
        with self.assertRaisesRegex(
            ReportError, "scatter/performance: plan count differs from manifest"
        ):
            self.read_plan()


class StructuredRecordsTest(unittest.TestCase):
    """decode_record, read_records and the small validation helpers."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write_log(self, records, padding_lines=()):
        path = self.root / "operator.log"
        path.write_text(log_text(records, padding_lines), encoding="utf-8")
        return path

    def test_decode_rejects_duplicate_key(self):
        with self.assertRaisesRegex(ReportError, "Duplicate JSON key: a"):
            decode_record('{"a": 1, "a": 2}')

    def test_decode_rejects_nan_constant(self):
        with self.assertRaisesRegex(ReportError, "Invalid JSON number: NaN"):
            decode_record('{"latency_ms": NaN}')

    def test_decode_rejects_non_finite_float(self):
        with self.assertRaisesRegex(ReportError, "Non-finite JSON number"):
            decode_record('{"latency_ms": 1e999}')

    def test_decode_rejects_invalid_json(self):
        with self.assertRaisesRegex(ReportError, "Invalid structured JSON"):
            decode_record('{"latency_ms":')

    def test_decode_rejects_non_object(self):
        with self.assertRaisesRegex(ReportError, "Record must be a JSON object"):
            decode_record("[1, 2]")

    def test_read_records_skips_non_prefixed_lines(self):
        path = self.write_log(
            [
                record(OPERATOR, "accuracy", "started", expected_cases=0),
                record(OPERATOR, "accuracy", "completed"),
            ],
            padding_lines=("plain log line", "another plain line"),
        )
        parsed = read_records(path, OPERATOR, "accuracy")
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["event"], "started")

    def test_read_records_reports_empty_log(self):
        path = self.write_log([])
        with self.assertRaisesRegex(ReportError, "No structured records"):
            read_records(path, OPERATOR, "accuracy")

    def test_read_records_reports_unreadable_log(self):
        path = self.root / "missing.log"
        with self.assertRaisesRegex(ReportError, "Cannot read log"):
            read_records(path, OPERATOR, "accuracy")

    def test_read_records_reports_invalid_json_with_line(self):
        path = self.write_log(
            [record(OPERATOR, "accuracy", "started", expected_cases=0)]
        )
        with path.open("a", encoding="utf-8") as handle:
            handle.write("NPU_RESULT_JSON={broken\n")
        with self.assertRaises(ReportError) as raised:
            read_records(path, OPERATOR, "accuracy")
        self.assertIn(f"{path}:2", str(raised.exception))

    def test_read_records_rejects_misplaced_marker(self):
        path = self.root / "operator.log"
        path.write_text(
            "noise NPU_RESULT_JSON={}\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ReportError, "Misplaced JSON marker"):
            read_records(path, OPERATOR, "accuracy")

    def test_read_records_rejects_wrong_schema_version(self):
        value = record(OPERATOR, "accuracy", "started")
        value["schema_version"] = 2
        path = self.write_log([value])
        with self.assertRaisesRegex(ReportError, "Unsupported schema_version"):
            read_records(path, OPERATOR, "accuracy")

    def test_read_records_rejects_wrong_operator(self):
        path = self.write_log([record("other", "accuracy", "started")])
        with self.assertRaisesRegex(ReportError, "Operator mismatch"):
            read_records(path, OPERATOR, "accuracy")

    def test_read_records_rejects_wrong_phase(self):
        path = self.write_log([record(OPERATOR, "performance", "started")])
        with self.assertRaisesRegex(ReportError, "Phase mismatch"):
            read_records(path, OPERATOR, "accuracy")

    def test_read_records_rejects_unknown_event(self):
        path = self.write_log([record(OPERATOR, "accuracy", "exploded")])
        with self.assertRaisesRegex(ReportError, "Unknown event"):
            read_records(path, OPERATOR, "accuracy")

    def test_read_records_rejects_bad_operator_name_and_phase(self):
        with self.assertRaisesRegex(ReportError, "Invalid operator name"):
            read_records(Path("x.log"), "BadName", "accuracy")
        with self.assertRaisesRegex(ReportError, "Unknown phase"):
            read_records(Path("x.log"), OPERATOR, "smoke")

    def test_cursor_takes_in_order_and_finishes(self):
        records = [{"event": "started"}, {"event": "completed"}]
        cursor = RecordCursor(records)
        self.assertEqual(cursor.take("started"), records[0])
        self.assertEqual(cursor.take("completed"), records[1])
        cursor.finish()

    def test_cursor_missing_event(self):
        cursor = RecordCursor([{"event": "started"}])
        cursor.take("started")
        with self.assertRaisesRegex(ReportError, "Record 2: missing completed"):
            cursor.take("completed")

    def test_cursor_unexpected_event(self):
        cursor = RecordCursor([{"event": "completed"}])
        with self.assertRaisesRegex(ReportError, "expected started, got"):
            cursor.take("started")

    def test_cursor_extra_records_at_finish(self):
        cursor = RecordCursor([{"event": "started"}, {"event": "completed"}])
        cursor.take("started")
        with self.assertRaisesRegex(ReportError, "Unexpected records after record 1"):
            cursor.finish()

    def test_check_count(self):
        check_count({"n": 1}, "n", 1)
        with self.assertRaisesRegex(ReportError, "Invalid count: n"):
            check_count({}, "n", 1)
        with self.assertRaisesRegex(ReportError, "n: expected 2, got 1"):
            check_count({"n": 1}, "n", 2)

    def test_latency_value(self):
        self.assertEqual(latency_value(1.5), 1.5)
        self.assertEqual(latency_value(0), 0.0)
        for bad in (-1, -0.5, float("inf"), float("nan"), "5", None):
            with self.subTest(value=bad):
                with self.assertRaises(ReportError):
                    latency_value(bad)

    def test_same_json_ignores_key_order(self):
        self.assertTrue(same_json({"a": 1, "b": 2}, {"b": 2, "a": 1}))
        self.assertFalse(same_json({"a": 1}, {"a": 2}))


class EvidenceValidationTest(unittest.TestCase):
    """validate_accuracy and validate_performance on real record streams."""

    def test_accuracy_all_cases_pass(self):
        planned = accuracy_plan("case0", "case1")
        records = passed_accuracy_records(OPERATOR, planned)
        result = validate_accuracy(records, planned, 0)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["expected_cases"], 2)
        self.assertEqual(result["completed_cases"], 2)
        self.assertEqual(result["passed_cases"], 2)
        self.assertEqual(result["failed_cases"], 0)

    def test_accuracy_failed_case_reports_error(self):
        planned = accuracy_plan("case0")
        records = failed_accuracy_records(OPERATOR, planned)
        result = validate_accuracy(records, planned, 1)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["current_case"], "case0")
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertEqual(result["error"], "boom")

    def test_accuracy_completed_log_with_failed_case_rejected(self):
        planned = accuracy_plan("case0")
        records = failed_accuracy_records(OPERATOR, planned)
        records[-1] = record(
            OPERATOR,
            "accuracy",
            "completed",
            expected_cases=1,
            completed_cases=1,
            passed_cases=0,
            failed_cases=1,
        )
        with self.assertRaisesRegex(
            ReportError, "Accuracy completed without all cases passing"
        ):
            validate_accuracy(records, planned, 0)

    def test_accuracy_duplicate_planned_names(self):
        case = {"name": "case0", "width": 64}
        planned = [
            {"case": case, "metadata": {"dtype": "float32"}},
            {"case": case, "metadata": {"dtype": "float32"}},
        ]
        with self.assertRaisesRegex(ReportError, "Duplicate planned case names"):
            validate_accuracy([], planned, 0)

    def test_accuracy_case_parameters_mismatch(self):
        planned = accuracy_plan("case0")
        records = [
            record(
                OPERATOR,
                "accuracy",
                "started",
                expected_cases=1,
                case_names=["case0"],
            ),
            record(
                OPERATOR,
                "accuracy",
                "case_started",
                case={"name": "case0", "width": 32},
                dtype="float32",
            ),
        ]
        with self.assertRaisesRegex(ReportError, "Case parameters mismatch"):
            validate_accuracy(records, planned, 0)

    def test_accuracy_missing_started_event(self):
        planned = accuracy_plan("case0")
        with self.assertRaisesRegex(ReportError, "missing started"):
            validate_accuracy([], planned, 0)

    def test_performance_all_configs_complete(self):
        config = {"name": "cfg0", "width": 64}
        planned = [{"config": config, "metadata": {"dtype": "float32"}}]
        records = [
            record(
                OPERATOR,
                "performance",
                "started",
                expected_configs=1,
                config_names=["cfg0"],
            ),
            record(
                OPERATOR,
                "performance",
                "config_started",
                config=config,
                dtype="float32",
                warmup=0,
                iterations=1,
                expected_samples=2,
            ),
            record(
                OPERATOR,
                "performance",
                "sample",
                config="cfg0",
                sample_index=0,
                latency_ms=1.0,
            ),
            record(
                OPERATOR,
                "performance",
                "sample",
                config="cfg0",
                sample_index=1,
                latency_ms=2.0,
            ),
            record(
                OPERATOR,
                "performance",
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
            ),
            record(
                OPERATOR,
                "performance",
                "completed",
                expected_configs=1,
                completed_configs=1,
            ),
        ]
        result = validate_performance(records, planned, 0, OPERATOR, 0, 1, 2)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["completed_configs"], 1)
        self.assertEqual(result["configs"][0]["median_latency_ms"], 1.5)

    def test_performance_sample_list_mismatch(self):
        config = {"name": "cfg0", "width": 64}
        planned = [{"config": config, "metadata": {"dtype": "float32"}}]
        records = [
            record(
                OPERATOR,
                "performance",
                "started",
                expected_configs=1,
                config_names=["cfg0"],
            ),
            record(
                OPERATOR,
                "performance",
                "config_started",
                config=config,
                dtype="float32",
                warmup=0,
                iterations=1,
                expected_samples=2,
            ),
            record(
                OPERATOR,
                "performance",
                "sample",
                config="cfg0",
                sample_index=0,
                latency_ms=1.0,
            ),
            record(
                OPERATOR,
                "performance",
                "sample",
                config="cfg0",
                sample_index=1,
                latency_ms=2.0,
            ),
            record(
                OPERATOR,
                "performance",
                "config_completed",
                config=config,
                dtype="float32",
                warmup=0,
                iterations=1,
                sample_count=2,
                samples_ms=[1.0, 3.0],
                median_latency_ms=2.0,
                threshold_ms=None,
                performance_verdict="not_evaluated",
            ),
        ]
        with self.assertRaisesRegex(
            ReportError, "Sample list differs from individual records"
        ):
            validate_performance(records, planned, 0, OPERATOR, 0, 1, 2)

    def test_performance_failure_stops_later_configs(self):
        first = {"name": "cfg0", "width": 64}
        second = {"name": "cfg1", "width": 128}
        planned = [
            {"config": first, "metadata": {"dtype": "float32"}},
            {"config": second, "metadata": {"dtype": "float32"}},
        ]
        records = [
            record(
                OPERATOR,
                "performance",
                "started",
                expected_configs=2,
                config_names=["cfg0", "cfg1"],
            ),
            record(
                OPERATOR,
                "performance",
                "config_started",
                config=first,
                dtype="float32",
                warmup=0,
                iterations=1,
                expected_samples=1,
            ),
            record(
                OPERATOR,
                "performance",
                "sample",
                config="cfg0",
                sample_index=0,
                latency_ms=1.0,
            ),
            record(
                OPERATOR,
                "performance",
                "config_completed",
                config=first,
                dtype="float32",
                warmup=0,
                iterations=1,
                sample_count=1,
                samples_ms=[1.0],
                median_latency_ms=1.0,
                threshold_ms=None,
                performance_verdict="not_evaluated",
            ),
            record(
                OPERATOR,
                "performance",
                "config_started",
                config=second,
                dtype="float32",
                warmup=0,
                iterations=1,
                expected_samples=1,
            ),
            record(
                OPERATOR,
                "performance",
                "failed",
                expected_configs=2,
                completed_configs=1,
                current_config="cfg1",
                error_type="RuntimeError",
                error="boom",
            ),
        ]
        result = validate_performance(records, planned, 1, OPERATOR, 0, 1, 1)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["attempted_configs"], 2)
        self.assertEqual(result["not_run_configs"], 0)
        self.assertEqual(result["current_config"], "cfg1")


class ReportGenerationTest(unittest.TestCase):
    """make_report ordering and fail-fast behavior, summarize and save."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = {"operators": [{"name": "scatter"}, {"name": "pick"}]}
        self.metadata = {"commit": "abc"}

    def test_make_report_passes_all_results(self):
        report = make_report(
            [passed_accuracy_result("scatter"), passed_accuracy_result("pick")],
            self.metadata,
            self.manifest,
            "accuracy",
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["exit_code"], 0)
        self.assertIsNone(report["first_failure"])
        self.assertEqual(report["errors"], [])

    def test_make_report_fail_fast_violation(self):
        report = make_report(
            [failed_accuracy_result("scatter"), passed_accuracy_result("pick")],
            self.metadata,
            self.manifest,
            "accuracy",
        )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["first_failure"], "scatter")
        blocked = report["operators"][1]
        self.assertEqual(blocked["status"], "failed")
        self.assertEqual(blocked["error_type"], "FailFastViolation")
        self.assertTrue(
            any("result received after failure of scatter" in e for e in report["errors"])
        )

    def test_make_report_not_run_after_failure(self):
        report = make_report(
            [failed_accuracy_result("scatter")],
            self.metadata,
            self.manifest,
            "accuracy",
        )
        not_run = report["operators"][1]
        self.assertEqual(not_run["status"], "not_run")
        self.assertEqual(not_run["blocked_by"], "scatter")

    def test_make_report_unexpected_operator(self):
        report = make_report(
            [passed_accuracy_result("ghost")],
            self.metadata,
            self.manifest,
            "accuracy",
        )
        self.assertEqual(len(report["unexpected_results"]), 1)
        self.assertTrue(
            any("Unexpected operator or phase results" in e for e in report["errors"])
        )

    def test_make_report_duplicate_operator_result(self):
        report = make_report(
            [passed_accuracy_result("scatter"), passed_accuracy_result("scatter")],
            self.metadata,
            self.manifest,
            "accuracy",
        )
        self.assertTrue(
            any("expected one result, got 2" in e for e in report["errors"])
        )

    def test_summarize_accuracy_log(self):
        planned = accuracy_plan("case0")
        path = self.root / "operator.log"
        path.write_text(
            log_text(passed_accuracy_records(OPERATOR, planned)), encoding="utf-8"
        )
        result = summarize_phase(path, OPERATOR, "accuracy", planned, 0)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["structured_log_valid"])
        self.assertEqual(result["parsed_records"], 4)

    def test_summarize_invalid_log(self):
        path = self.root / "operator.log"
        path.write_text("no structured data here\n", encoding="utf-8")
        result = summarize_phase(path, OPERATOR, "accuracy", accuracy_plan("case0"), 1)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["structured_log_valid"])
        self.assertEqual(result["error_type"], "ReportError")

    def test_save_report_writes_files(self):
        report = make_report(
            [passed_accuracy_result("scatter")],
            self.metadata,
            {"operators": [{"name": "scatter"}]},
            "accuracy",
        )
        output = self.root / "out"
        saved = save_report(report, output)
        for key in ("json", "markdown"):
            path = Path(saved[key]["path"])
            self.assertTrue(path.exists())
            self.assertEqual(saved[key]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(json.loads(Path(saved["json"]["path"]).read_text())["status"], "passed")


if __name__ == "__main__":
    unittest.main()
