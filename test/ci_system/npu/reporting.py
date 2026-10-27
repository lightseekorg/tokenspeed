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

"""Validate structured operator records before producing CI reports."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from pathlib import Path

PREFIX = "NPU_RESULT_JSON="


class ReportError(ValueError):
    """Evidence is malformed, incomplete, or inconsistent."""


def require(condition, message):
    if not condition:
        raise ReportError(message)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ReportError(f"Invalid JSON number: {value}")


def _finite_float(value):
    number = float(value)
    require(math.isfinite(number), f"Non-finite JSON number: {value}")
    return number


def decode_record(text):
    try:
        record = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except ValueError as exc:
        raise ReportError(f"Invalid structured JSON: {exc}") from exc
    require(type(record) is dict, "Record must be a JSON object")
    return record


def read_expected_plan(path, manifest, manifest_sha256, repo_root):
    """Check plan identity, registered scripts and expected item counts."""
    plan = decode_record(Path(path).read_text(encoding="utf-8"))
    require(
        type(plan.get("schema_version")) is int and plan["schema_version"] == 2,
        "Expected plan schema_version=2",
    )
    require(
        plan.get("manifest_sha256") == manifest_sha256,
        "Plan was generated for a different manifest",
    )
    operators = plan.get("operators")
    registered = manifest["operators"]
    require(type(operators) is dict, "Missing plan operators")
    require(
        set(operators) == {item["name"] for item in registered},
        "Plan operators differ from manifest",
    )
    sources = plan.get("source_sha256")
    scripts = {item["script"] for item in registered}
    require(
        type(sources) is dict and set(sources) == scripts,
        "Plan source list differs from manifest",
    )
    for relative, expected in sources.items():
        require(
            type(expected) is str and re.fullmatch(r"[0-9a-f]{64}", expected),
            f"Invalid source hash: {relative}",
        )
        actual = hashlib.sha256((Path(repo_root) / relative).read_bytes()).hexdigest()
        require(actual == expected, f"Script differs from expected plan: {relative}")
    for item in registered:
        name = item["name"]
        entry = operators[name]
        require(type(entry) is dict, f"Invalid operator plan: {name}")
        for phase, count_key in (
            ("accuracy", "accuracy_cases"),
            ("performance", "performance_configs"),
        ):
            values = entry.get(phase)
            require(
                type(values) is list and len(values) == item[count_key],
                f"{name}/{phase}: plan count differs from manifest",
            )
    return plan


def read_records(log_path, operator, phase):
    require(
        type(operator) is str and re.fullmatch(r"[a-z][a-z0-9_]*", operator),
        "Invalid operator name",
    )
    events = {
        "accuracy": (
            "started",
            "case_started",
            "case_completed",
            "completed",
            "failed",
        ),
        "performance": (
            "started",
            "config_started",
            "sample",
            "config_completed",
            "completed",
            "failed",
        ),
    }
    require(phase in ("accuracy", "performance"), "Unknown phase")
    path = Path(log_path)
    records = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if PREFIX not in line:
                    continue
                try:
                    require(line.startswith(PREFIX), "Misplaced JSON marker")
                    record = decode_record(line[len(PREFIX) :])
                    version = record.get("schema_version")
                    require(
                        type(version) is int and version == 1,
                        "Unsupported schema_version",
                    )
                    require(record.get("operator") == operator, "Operator mismatch")
                    require(record.get("phase") == phase, "Phase mismatch")
                    require(record.get("event") in events[phase], "Unknown event")
                except ReportError as exc:
                    raise ReportError(f"{path}:{line_number}: {exc}") from exc
                records.append(record)
    except (OSError, UnicodeError) as exc:
        raise ReportError(f"Cannot read log {path}: {exc}") from exc
    require(records, f"No structured records in {path}")
    return records


def check_count(record, key, expected):
    value = record.get(key)
    require(type(value) is int and value >= 0, f"Invalid count: {key}")
    require(value == expected, f"{key}: expected {expected}, got {value}")


def latency_value(value):
    require(type(value) in (int, float), "Latency must be a number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ReportError("Invalid latency value") from exc
    require(math.isfinite(number) and number >= 0, "Invalid latency value")
    return number


class RecordCursor:
    """Consume records in order and reject missing or extra events."""

    def __init__(self, records):
        self.records = records
        self.index = 0

    def take(self, event):
        position = self.index + 1
        require(
            self.index < len(self.records),
            f"Record {position}: missing {event}",
        )
        record = self.records[self.index]
        actual = record.get("event")
        require(
            actual == event,
            f"Record {position}: expected {event}, got {actual!r}",
        )
        self.index += 1
        return record

    def finish(self):
        require(
            self.index == len(self.records),
            f"Unexpected records after record {self.index}",
        )


def same_json(actual, expected):
    options = {"sort_keys": True, "allow_nan": False}
    return json.dumps(actual, **options) == json.dumps(expected, **options)


def validate_accuracy(records, planned, exit_code):
    require(type(exit_code) is int, "Missing or invalid process exit code")
    require(type(planned) is list and planned, "Empty accuracy plan")
    names = []
    for item in planned:
        require(
            type(item) is dict and set(item) == {"case", "metadata"},
            "Invalid accuracy plan item",
        )
        case, metadata = item["case"], item["metadata"]
        require(
            type(case) is dict and type(case.get("name")) is str and case["name"],
            "Invalid planned case name",
        )
        require(
            type(metadata) is dict
            and type(metadata.get("dtype")) is str
            and metadata["dtype"],
            "Missing planned accuracy dtype",
        )
        names.append(case["name"])
    require(len(set(names)) == len(names), "Duplicate planned case names")
    require(
        type(records) is list and all(type(r) is dict for r in records),
        "Invalid accuracy records",
    )
    cursor = RecordCursor(records)

    def peek():
        return (
            records[cursor.index].get("event") if cursor.index < len(records) else None
        )

    def match(record, item):
        require(same_json(record.get("case"), item["case"]), "Case parameters mismatch")
        for key, expected in item["metadata"].items():
            require(
                key in record and same_json(record[key], expected),
                f"{item['case']['name']}: {key} mismatch",
            )

    start = cursor.take("started")
    check_count(start, "expected_cases", len(planned))
    require(start.get("case_names") == names, "Accuracy case names mismatch")
    results, current, started = [], None, 0
    for item in planned:
        if peek() == "failed":
            break
        match(cursor.take("case_started"), item)
        current = item["case"]["name"]
        started += 1
        if peek() == "failed":
            break
        record = cursor.take("case_completed")
        match(record, item)
        status = record.get("status")
        require(status in ("passed", "failed"), "Invalid accuracy case status")
        results.append(
            {
                "case": item["case"],
                "metadata": item["metadata"],
                "dtype": item["metadata"]["dtype"],
                "status": status,
            }
        )
        if status == "failed":
            break
    terminal = "failed" if peek() == "failed" else "completed"
    end = cursor.take(terminal)
    cursor.finish()
    passed = sum(item["status"] == "passed" for item in results)
    failed = len(results) - passed
    for key, expected in (
        ("expected_cases", len(planned)),
        ("completed_cases", len(results)),
        ("passed_cases", passed),
        ("failed_cases", failed),
    ):
        check_count(end, key, expected)
    result = {
        "status": "failed",
        "exit_code": exit_code,
        "expected_cases": len(planned),
        "started_cases": started,
        "completed_cases": len(results),
        "passed_cases": passed,
        "failed_cases": failed,
        "incomplete_cases": started - len(results),
        "not_run_cases": len(planned) - started,
        "cases": results,
    }
    if terminal == "failed":
        require(exit_code != 0, "Failed accuracy log has zero exit code")
        require(
            "current_case" in end and end["current_case"] == current,
            "Failed accuracy current_case mismatch",
        )
        require(
            type(end.get("error_type")) is str and end["error_type"],
            "Missing accuracy error type",
        )
        require(type(end.get("error")) is str, "Missing accuracy error message")
        result.update(
            current_case=current, error_type=end["error_type"], error=end["error"]
        )
    else:
        require(
            len(results) == len(planned) and failed == 0,
            "Accuracy completed without all cases passing",
        )
        result["status"] = "passed" if exit_code == 0 else "failed"
    return result


def validate_config(cursor, config, metadata, warmup, iterations, sample_count):
    require(
        type(config) is dict and type(config.get("name")) is str and config["name"],
        "Invalid planned config",
    )
    require(
        type(metadata) is dict
        and type(metadata.get("dtype")) is str
        and metadata["dtype"],
        "Missing planned performance dtype",
    )
    reserved = {
        "config",
        "metadata",
        "status",
        "warmup",
        "iterations",
        "expected_samples",
        "sample_count",
        "samples_ms",
        "median_latency_ms",
        "threshold_ms",
        "performance_verdict",
        "schema_version",
        "operator",
        "phase",
        "event",
    }
    require(not reserved.intersection(metadata), "Reserved key in config metadata")
    for key, value, minimum in (
        ("warmup", warmup, 0),
        ("iterations", iterations, 1),
        ("sample_count", sample_count, 1),
    ):
        require(type(value) is int and value >= minimum, f"Invalid plan: {key}")

    def pending_failure():
        return (
            cursor.index < len(cursor.records)
            and cursor.records[cursor.index].get("event") == "failed"
        )

    def match(record):
        require(same_json(record.get("config"), config), "Config parameters mismatch")
        check_count(record, "warmup", warmup)
        check_count(record, "iterations", iterations)
        for key, expected in metadata.items():
            require(
                key in record and same_json(record[key], expected),
                f"{config['name']}: {key} mismatch",
            )

    start = cursor.take("config_started")
    match(start)
    check_count(start, "expected_samples", sample_count)
    samples = []
    for index in range(sample_count):
        if pending_failure():
            break
        record = cursor.take("sample")
        require(record.get("config") == config["name"], "Sample config mismatch")
        check_count(record, "sample_index", index)
        samples.append(latency_value(record.get("latency_ms")))
    result = {
        "config": config,
        "metadata": metadata,
        **metadata,
        "status": "incomplete",
        "warmup": warmup,
        "iterations": iterations,
        "expected_samples": sample_count,
        "sample_count": len(samples),
        "samples_ms": samples,
        "median_latency_ms": None,
        "threshold_ms": None,
        "performance_verdict": "not_evaluated",
    }
    if pending_failure():
        return result
    end = cursor.take("config_completed")
    match(end)
    check_count(end, "sample_count", sample_count)
    reported = end.get("samples_ms")
    require(
        type(reported) is list and len(reported) == sample_count,
        "Missing or incomplete sample list",
    )
    require(
        [latency_value(value) for value in reported] == samples,
        "Sample list differs from individual records",
    )
    median = statistics.median(samples)
    require(
        latency_value(end.get("median_latency_ms")) == median,
        "Median differs from recorded samples",
    )
    require(
        "threshold_ms" in end and end["threshold_ms"] is None,
        "Unexpected or missing performance threshold",
    )
    require(
        end.get("performance_verdict") == "not_evaluated",
        "Unexpected performance verdict",
    )
    result.update(status="completed", median_latency_ms=median)
    return result


def validate_performance(
    records, planned, exit_code, operator, warmup, iterations, sample_count
):
    require(type(exit_code) is int, "Missing or invalid process exit code")
    require(
        type(operator) is str and re.fullmatch(r"[a-z][a-z0-9_]*", operator),
        "Invalid operator name",
    )
    require(type(planned) is list and planned, "Empty performance plan")
    for key, value, minimum in (
        ("warmup", warmup, 0),
        ("iterations", iterations, 1),
        ("sample_count", sample_count, 1),
    ):
        require(type(value) is int and value >= minimum, f"Invalid plan: {key}")
    names = []
    for item in planned:
        require(
            type(item) is dict and set(item) == {"config", "metadata"},
            "Invalid performance plan item",
        )
        config, metadata = item["config"], item["metadata"]
        require(
            type(config) is dict and type(config.get("name")) is str and config["name"],
            "Invalid planned config name",
        )
        require(
            type(metadata) is dict
            and type(metadata.get("dtype")) is str
            and metadata["dtype"],
            "Missing planned performance dtype",
        )
        names.append(config["name"])
    require(len(set(names)) == len(names), "Duplicate planned config names")
    require(
        type(records) is list and all(type(r) is dict for r in records),
        "Invalid performance records",
    )
    cursor = RecordCursor(records)

    def peek():
        return (
            records[cursor.index].get("event") if cursor.index < len(records) else None
        )

    start = cursor.take("started")
    check_count(start, "expected_configs", len(planned))
    require(start.get("config_names") == names, "Performance config names mismatch")
    results, current = [], None
    for item in planned:
        if peek() == "failed":
            break
        current = item["config"]["name"]
        result = validate_config(
            cursor, item["config"], item["metadata"], warmup, iterations, sample_count
        )
        results.append(result)
        if result["status"] != "completed":
            break
    terminal = "failed" if peek() == "failed" else "completed"
    end = cursor.take(terminal)
    cursor.finish()
    completed = sum(item["status"] == "completed" for item in results)
    check_count(end, "expected_configs", len(planned))
    check_count(end, "completed_configs", completed)
    attempted = len(results)
    error = {}
    if terminal == "failed":
        require(exit_code != 0, "Failed performance log has zero exit code")
        require("current_config" in end, "Missing failed current_config")
        observed = end["current_config"]
        if observed != current:
            require(
                completed == len(results)
                and completed < len(planned)
                and observed == names[completed],
                "Failed current_config mismatch",
            )
            attempted += 1
        require(
            type(end.get("error_type")) is str and end["error_type"],
            "Missing performance error type",
        )
        require(type(end.get("error")) is str, "Missing performance error message")
        error = dict(
            current_config=observed, error_type=end["error_type"], error=end["error"]
        )
    else:
        require(
            completed == len(planned), "Performance completed with unfinished configs"
        )
    return {
        "status": (
            "completed" if terminal == "completed" and exit_code == 0 else "failed"
        ),
        "exit_code": exit_code,
        "expected_configs": len(planned),
        "attempted_configs": attempted,
        "started_configs": len(results),
        "completed_configs": completed,
        "incomplete_configs": attempted - completed,
        "not_run_configs": len(planned) - attempted,
        "expected_samples": len(planned) * sample_count,
        "completed_samples": sum(item["sample_count"] for item in results),
        "performance_verdict": "not_evaluated",
        "configs": results,
        **error,
    }


def summarize_phase(
    log_path,
    operator,
    phase,
    planned,
    exit_code,
    *,
    warmup=None,
    iterations=None,
    sample_count=None,
):
    result = {
        "operator": operator,
        "phase": phase,
        "status": "failed",
        "log_path": str(log_path),
        "exit_code": exit_code if type(exit_code) is int else None,
        "structured_log_valid": False,
        "parsed_records": None,
    }
    if phase == "performance":
        result["performance_verdict"] = "not_evaluated"
    try:
        require(type(exit_code) is int, "Missing or invalid process exit code")
        records = read_records(log_path, operator, phase)
        result["parsed_records"] = len(records)
        if phase == "accuracy":
            details = validate_accuracy(records, planned, exit_code)
        else:
            details = validate_performance(
                records,
                planned,
                exit_code,
                operator,
                warmup,
                iterations,
                sample_count,
            )
        result.update(details)
        result["structured_log_valid"] = True
        if exit_code != 0 and "error" not in result:
            result["error_type"] = "ProcessExitError"
            result["error"] = f"Process exited with code {exit_code}"
        elif result["status"] == "failed" and "error" not in result:
            result["error_type"] = "AccuracyFailure"
            result["error"] = "One or more accuracy cases failed"
    except Exception as exc:
        # Keep reporting possible even when the log or validation fails.
        result["status"] = "failed"
        result["structured_log_valid"] = False
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    return result


def phase_succeeded(result, phase):
    require(phase in ("accuracy", "performance"), "Unknown report phase")
    wanted = "passed" if phase == "accuracy" else "completed"
    return (
        type(result) is dict
        and result.get("status") == wanted
        and type(result.get("exit_code")) is int
        and result["exit_code"] == 0
        and result.get("structured_log_valid") is True
        and result.get("log_saved") is True
    )


def make_report(results, metadata, manifest, phase):
    require(phase in ("accuracy", "performance"), "Unknown report phase")
    require(
        type(results) is list and all(type(r) is dict for r in results),
        "Operator results must be a list of objects",
    )
    require(type(metadata) is dict, "Report metadata must be an object")
    require(
        type(manifest) is dict
        and type(manifest.get("operators")) is list
        and manifest["operators"],
        "Invalid report manifest",
    )
    names = []
    for entry in manifest["operators"]:
        require(
            type(entry) is dict
            and type(entry.get("name")) is str
            and re.fullmatch(r"[a-z][a-z0-9_]*", entry["name"]),
            "Invalid registered operator name",
        )
        names.append(entry["name"])
    require(len(set(names)) == len(names), "Duplicate registered operators")
    expected = [(name, phase) for name in names]
    actual = [(item.get("operator"), item.get("phase")) for item in results]
    errors = []
    if actual != expected[: len(actual)]:
        errors.append("Results must follow registry order in the selected phase")
    unexpected = [
        item
        for item in results
        if item.get("operator") not in names or item.get("phase") != phase
    ]
    if unexpected:
        errors.append("Unexpected operator or phase results")
    ordered, first_failure = [], None
    for name in names:
        matches = [
            item
            for item in results
            if item.get("operator") == name and item.get("phase") == phase
        ]
        if first_failure is not None:
            if matches:
                message = (
                    f"{name}/{phase}: result received after failure of {first_failure}"
                )
                errors.append(message)
                ordered.append(
                    {
                        "operator": name,
                        "phase": phase,
                        "status": "failed",
                        "error_type": "FailFastViolation",
                        "error": message,
                        "reported_results": matches,
                    }
                )
            else:
                ordered.append(
                    {
                        "operator": name,
                        "phase": phase,
                        "status": "not_run",
                        "exit_code": None,
                        "structured_log_valid": False,
                        "log_saved": False,
                        "reason": "not_run_after_failure",
                        "blocked_by": first_failure,
                    }
                )
            continue
        if len(matches) != 1:
            message = f"{name}/{phase}: expected one result, got {len(matches)}"
            errors.append(message)
            ordered.append(
                {
                    "operator": name,
                    "phase": phase,
                    "status": "failed",
                    "exit_code": None,
                    "structured_log_valid": False,
                    "log_saved": False,
                    "error_type": "EvidenceError",
                    "error": message,
                    "conflicting_results": matches,
                }
            )
            first_failure = name
            continue
        item = dict(matches[0])
        if not phase_succeeded(item, phase):
            item["test_status"] = item.get("status")
            item["status"] = "failed"
            item.setdefault("error_type", "PhaseOrEvidenceFailure")
            item.setdefault("error", "Test or required evidence collection failed")
            errors.append(f"{name}/{phase}: {item['error']}")
            first_failure = name
        ordered.append(item)
    report = {
        "schema_version": 2,
        "phase": phase,
        "status": "failed" if errors else "passed",
        "exit_code": 1 if errors else 0,
        "expected_operators": names,
        "first_failure": first_failure,
        "metadata": metadata,
        "operators": ordered,
        "errors": errors,
        "unexpected_results": unexpected,
    }
    if phase == "performance":
        report["performance_verdict"] = "not_evaluated"
    return report


def _md_cell(value):
    import html

    if value is None:
        return "-"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, allow_nan=False)
    text = html.escape(str(value), quote=False)
    for character in "\\`*_[]|":
        text = text.replace(character, "\\" + character)
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")


def _md_row(values):
    return "| " + " | ".join(_md_cell(value) for value in values) + " |"


def render_summary(report):
    phase = report["phase"]
    require(phase in ("accuracy", "performance"), "Unknown summary phase")
    phase_name = "accuracy tests" if phase == "accuracy" else "performance tests"
    lines = [
        f"## Ascend NPU operator {phase_name}",
        "",
        "Phase status: **" + _md_cell(report["status"]) + "**",
        "",
        "Phase exit code: " + _md_cell(report["exit_code"]),
        "",
    ]
    if phase == "performance":
        lines += [
            "Performance verdict: **not_evaluated** (no latency thresholds are configured).",
            "",
        ]
    lines += [
        "| Operator | Result | Progress | Exit code | Structured log valid | Log saved |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in report["operators"]:
        progress = "No data available"
        if item["status"] == "not_run":
            progress = "Not run after failure of " + str(
                item.get("blocked_by", "a previous test")
            )
        elif phase == "accuracy" and "completed_cases" in item:
            progress = (
                f"Completed {item['completed_cases']}/{item['expected_cases']}; "
                f"passed {item['passed_cases']}; failed {item['failed_cases']}; "
                f"incomplete {item['incomplete_cases']}; not run {item['not_run_cases']}"
            )
        elif phase == "performance" and "completed_configs" in item:
            progress = (
                f"Completed configs {item['completed_configs']}/{item['expected_configs']}; "
                f"Collected samples {item['completed_samples']}/{item['expected_samples']}; "
                f"incomplete {item['incomplete_configs']}; not run {item['not_run_configs']}"
            )
        lines.append(
            _md_row(
                [
                    item["operator"],
                    item["status"],
                    progress,
                    item.get("exit_code"),
                    item.get("structured_log_valid"),
                    item.get("log_saved"),
                ]
            )
        )
    if phase == "performance":
        lines += [
            "",
            "### Configuration sampling results",
            "",
            "| Operator | Configuration | Result | Samples | Median latency (ms/execution) |",
            "| --- | --- | --- | --- | --- |",
        ]
        for item in report["operators"]:
            if not item.get("structured_log_valid"):
                continue
            for config in item.get("configs", []):
                median = config.get("median_latency_ms")
                lines.append(
                    _md_row(
                        [
                            item["operator"],
                            config["config"]["name"],
                            config["status"],
                            f"{config['sample_count']}/{config['expected_samples']}",
                            format(median, ".9g") if median is not None else None,
                        ]
                    )
                )
    details = list(report["errors"])
    for item in report["operators"]:
        if item.get("error") is not None:
            details.append(
                f"{item['operator']}: {item.get('error_type', 'Error')}: {item['error']}"
            )
        elif item["status"] == "not_run":
            details.append(
                f"{item['operator']}: not run after failure of {item['blocked_by']}"
            )
    if details:
        lines += ["", "### Failure and execution details", ""]
        lines += ["- " + _md_cell(reason) for reason in dict.fromkeys(details)]
    lines += ["", "### Evidence", "", "| Operator | Log path |", "| --- | --- |"]
    for item in report["operators"]:
        lines.append(_md_row([item["operator"], item.get("log_path")]))
    lines += ["", "### Run information", "", "| Field | Value |", "| --- | --- |"]
    for key, value in report["metadata"].items():
        lines.append(_md_row([key, value]))
    return "\n".join(lines) + "\n"


def save_report(report, output_dir, github_summary=None):
    """Save both report formats; propagate any write failure to the caller."""
    require(
        type(report) is dict and report.get("status") in ("passed", "failed"),
        "Invalid report status",
    )
    require(
        type(report.get("exit_code")) is int
        and report["exit_code"] == (0 if report["status"] == "passed" else 1),
        "Report status and exit code differ",
    )
    json_text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    markdown = render_summary(report)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    saved = {}
    for key, filename, text in (
        ("json", "report.json", json_text),
        ("markdown", "summary.md", markdown),
    ):
        target = directory / filename
        expected = text.encode("utf-8")
        with target.open("xb") as handle:
            require(
                handle.write(expected) == len(expected), f"Incomplete write: {target}"
            )
        require(target.read_bytes() == expected, f"Saved file differs: {target}")
        saved[key] = {
            "path": str(target),
            "sha256": hashlib.sha256(expected).hexdigest(),
        }
    saved["github_summary"] = None
    if github_summary is not None:
        target = Path(github_summary)
        content = "\n\n" + markdown
        with target.open("a", encoding="utf-8") as handle:
            require(
                handle.write(content) == len(content),
                "Incomplete Actions summary write",
            )
        saved["github_summary"] = {"path": str(target), "appended": True}
    return saved
