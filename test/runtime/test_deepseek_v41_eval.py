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

"""CPU-only checks of the installed EvalScope 1.11.1 report contract."""

import json
import signal
import subprocess
import sys
from contextlib import nullcontext
from io import StringIO
from test.runtime import run_deepseek_v41_eval as harness
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def report():
    # Serialized by EvalScope 1.11.1 ReportGenerator + ExecutionSummary, with
    # synthetic scores (1200/1319), not a claimed model evaluation result.
    return {
        "schema_version": 2,
        "name": "DeepSeek-V4.1-Flash@gsm8k",
        "dataset_name": "gsm8k",
        "dataset_pretty_name": "GSM8K",
        "dataset_description": "",
        "model_name": "DeepSeek-V4.1-Flash",
        "metrics": [
            {
                "identity": {
                    "name": "accuracy",
                    "aggregation": "mean",
                    "dimensions": {},
                },
                "legacy_name": None,
                "num": 1319,
                "score": 0.9098,
                "macro_score": 0.9098,
                "categories": [
                    {
                        "name": ["default"],
                        "num": 1319,
                        "score": 0.9098,
                        "macro_score": 0.9098,
                        "subsets": [
                            {
                                "name": "main",
                                "score": 0.9098,
                                "num": 1319,
                                "is_aggregate": False,
                            }
                        ],
                    }
                ],
                "semantics": {
                    "semantic_id": "quality.accuracy.ratio",
                    "metric_name": "Accuracy",
                    "display_name": None,
                    "kind": "quality",
                    "direction": "higher_is_better",
                    "raw_unit": None,
                    "value_range": {"min": 0.0, "max": 1.0},
                    "display_kind": "percent",
                    "display_multiplier": 100.0,
                    "display_unit": "%",
                    "display_precision": 1,
                },
            }
        ],
        "analysis": "N/A",
        "perf_metrics": None,
        "primary_metric_identity": {
            "name": "accuracy",
            "aggregation": "mean",
            "dimensions": {},
        },
        "primary_metric_unavailable_reason": None,
        "judge_summary": None,
        "execution_summary": {
            "requested": 1319,
            "succeeded": 1319,
            "errored": 0,
            "incomplete": False,
            "subsets": {"main": {"requested": 1319, "succeeded": 1319, "errored": 0}},
        },
        "num": 1319,
    }


def write_report(output, report, timestamp):
    path = (
        output
        / "evalscope"
        / timestamp
        / "reports"
        / "DeepSeek-V4.1-Flash"
        / "gsm8k.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def set_accuracy(report, accuracy):
    metric = report["metrics"][0]
    category = metric["categories"][0]
    for node in (metric, category, category["subsets"][0]):
        node["score"] = accuracy
    metric["macro_score"] = category["macro_score"] = accuracy


def test_public_result_uses_raw_accuracy(tmp_path, report):
    report["analysis"] = "Do not publish this report text"
    report["perf_metrics"] = {"local_path": str(tmp_path)}
    write_report(tmp_path, report, "20000101_000000")
    assert harness.read_gsm8k_result(tmp_path / "evalscope", "1.11.1") == {
        "model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "dataset": "openai/gsm8k",
        "subset": "main",
        "split": "test",
        "evalscope_version": "1.11.1",
        "samples": 1319,
        "accuracy": 0.9098,
        "threshold": 0.90,
        "passed": True,
    }


@pytest.mark.parametrize(
    "accuracy,passed", [(0.0, False), (0.90, False), (0.9001, True), (1.0, True)]
)
def test_strict_threshold(tmp_path, report, accuracy, passed):
    set_accuracy(report, accuracy)
    write_report(tmp_path, report, "20000101_000000")
    assert (
        harness.read_gsm8k_result(tmp_path / "evalscope", "1.11.1")["passed"] is passed
    )


@pytest.mark.parametrize(
    "path,value",
    [
        (("schema_version",), 1),
        (("schema_version",), 2.0),
        (("dataset_name",), "other"),
        (("model_name",), "other"),
        (("primary_metric_identity",), None),
        (("primary_metric_unavailable_reason",), "Missing accuracy"),
        (("metrics",), []),
        (("metrics", 0, "identity", "name"), "acc"),
        (("metrics", 0, "identity", "aggregation"), "max"),
        (("metrics", 0, "identity", "dimensions"), {"k": 2}),
        (("metrics", 0, "categories"), []),
        (("metrics", 0, "categories", 0, "name"), ["other"]),
        (("metrics", 0, "categories", 0, "subsets", 0, "name"), "train"),
        (("metrics", 0, "categories", 0, "subsets", 0, "is_aggregate"), True),
        (("execution_summary",), None),
        (("execution_summary",), {"incomplete": False}),
        (("execution_summary", "incomplete"), True),
        (("execution_summary", "incomplete"), 0),
        (("execution_summary", "requested"), 1318),
        (("execution_summary", "requested"), 1320),
        (("execution_summary", "requested"), "1319"),
        (("execution_summary", "succeeded"), 1318),
        (("execution_summary", "succeeded"), 1319.0),
        (("execution_summary", "errored"), 1),
        (("execution_summary", "errored"), False),
        (("execution_summary", "subsets"), {}),
        (("execution_summary", "subsets", "main", "requested"), 1318),
        (("execution_summary", "subsets", "main", "succeeded"), 1318),
        (("execution_summary", "subsets", "main", "errored"), 1),
        (("num",), 1318),
        (("num",), "1319"),
        (("metrics", 0, "num"), 1318),
        (("metrics", 0, "categories", 0, "num"), 1318),
        (("metrics", 0, "categories", 0, "subsets", 0, "num"), 1318),
        (("metrics", 0, "macro_score"), 0.99),
        (("metrics", 0, "categories", 0, "score"), 0.99),
        (("metrics", 0, "categories", 0, "subsets", 0, "score"), 0.99),
    ],
)
def test_rejects_malformed_or_incomplete_report(tmp_path, report, path, value):
    node = report
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    write_report(tmp_path, report, "20000101_000000")
    with pytest.raises(RuntimeError, match="Invalid or incomplete GSM8K report"):
        harness.read_gsm8k_result(tmp_path / "evalscope", "1.11.1")


@pytest.mark.parametrize(
    "accuracy", [float("nan"), float("inf"), -0.01, 90.98, True, "0.99", None]
)
def test_rejects_invalid_accuracy(tmp_path, report, accuracy):
    set_accuracy(report, accuracy)
    write_report(tmp_path, report, "20000101_000000")
    with pytest.raises(
        RuntimeError, match="accuracy must be a consistent finite number"
    ):
        harness.read_gsm8k_result(tmp_path / "evalscope", "1.11.1")


@pytest.mark.parametrize(
    "content", ["{", "null", "[]", "{}", '{"score": 0.99, "num": 1319}']
)
def test_rejects_broken_or_legacy_json(tmp_path, report, content):
    path = write_report(tmp_path, report, "20000101_000000")
    path.write_text(content, encoding="utf-8")
    with pytest.raises(RuntimeError, match="Invalid or incomplete GSM8K report"):
        harness.read_gsm8k_result(tmp_path / "evalscope", "1.11.1")


@pytest.mark.parametrize("timestamps", [[], ["20000101_000000", "20000101_000001"]])
def test_requires_one_report(tmp_path, report, timestamps):
    for timestamp in timestamps:
        write_report(tmp_path, report, timestamp)
    with pytest.raises(RuntimeError, match="Expected exactly one GSM8K report"):
        harness.read_gsm8k_result(tmp_path / "evalscope", "1.11.1")


@pytest.fixture
def main_env(tmp_path, report, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_deepseek_v41_eval.py",
            "--model",
            str(tmp_path / "checkpoint"),
            "--output-dir",
            str(tmp_path),
            "--port",
            "12345",
            "--server-timeout",
            "1",
            "--eval-timeout",
            "7200",
            "--run-eval",
        ],
    )
    monkeypatch.setattr(harness.socket, "socket", MagicMock())
    monkeypatch.setattr(
        harness,
        "urlopen",
        MagicMock(
            side_effect=[
                nullcontext(SimpleNamespace(status=200)),
                StringIO(json.dumps({"choices": [{"message": {"content": "4"}}]})),
            ]
        ),
    )
    launch = MagicMock()
    launch.return_value.poll.return_value = None
    monkeypatch.setattr(harness.subprocess, "Popen", launch)
    monkeypatch.setattr(harness.os, "killpg", MagicMock())
    process = MagicMock()
    process.return_value.children.return_value = []
    monkeypatch.setattr(harness.psutil, "Process", process)
    monkeypatch.setattr(harness.psutil, "wait_procs", MagicMock(return_value=([], [])))
    monkeypatch.setattr(harness, "version", MagicMock(return_value="1.11.1"))

    def complete(*args, **kwargs):
        assert not (tmp_path / "result.json").exists()
        write_report(tmp_path, report, "20000101_000000")

    evaluate = MagicMock(side_effect=complete)
    monkeypatch.setattr(harness.subprocess, "run", evaluate)
    return evaluate, launch


def test_full_eval_writes_result_only_after_completion(tmp_path, main_env):
    evaluate, launch = main_env
    harness.main()
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["passed"] is True
    assert result["samples"] == 1319
    assert str(tmp_path) not in json.dumps(result)
    evaluate.assert_called_once()
    command = evaluate.call_args.args[0]
    assert command[:2] == ["evalscope", "eval"]
    assert not {"--limit", "--ignore-errors", "--use-cache"}.intersection(command)
    assert evaluate.call_args.kwargs["check"] is True
    assert evaluate.call_args.kwargs["timeout"] == 7200
    launch.return_value.wait.assert_called_once_with(timeout=45)


@pytest.mark.parametrize(
    "error",
    [
        subprocess.CalledProcessError(1, "evalscope"),
        subprocess.CalledProcessError(-signal.SIGTERM, "evalscope"),
        subprocess.TimeoutExpired("evalscope", 7200),
    ],
)
def test_failed_process_never_accepts_even_a_complete_report(
    tmp_path, report, main_env, error
):
    evaluate, launch = main_env

    def fail(*args, **kwargs):
        write_report(tmp_path, report, "20000101_000000")
        raise error

    evaluate.side_effect = fail
    with pytest.raises(type(error)):
        harness.main()
    assert not (tmp_path / "result.json").exists()
    launch.return_value.wait.assert_called_once_with(timeout=45)


def test_complete_below_threshold_records_failure(tmp_path, report, main_env):
    set_accuracy(report, 0.90)
    with pytest.raises(RuntimeError, match="must be > 0.90"):
        harness.main()
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["accuracy"] == 0.90
    assert result["passed"] is False


@pytest.mark.parametrize(
    "requested,succeeded,errored", [(1319, 1318, 1), (1319, 1318, 0), (20, 20, 0)]
)
def test_partial_eval_has_no_result(
    tmp_path, report, main_env, requested, succeeded, errored
):
    # Real incomplete/filtered and --limit report shapes, even with high accuracy.
    set_accuracy(report, 0.99)
    execution = report["execution_summary"]
    execution["incomplete"] = succeeded < requested
    for counts in (execution, execution["subsets"]["main"]):
        counts.update(requested=requested, succeeded=succeeded, errored=errored)
    metric = report["metrics"][0]
    category = metric["categories"][0]
    for node in (report, metric, category, category["subsets"][0]):
        node["num"] = succeeded
    with pytest.raises(RuntimeError, match="Invalid or incomplete GSM8K report"):
        harness.main()
    assert not (tmp_path / "result.json").exists()


def test_server_exit_has_no_result(tmp_path, main_env):
    evaluate, launch = main_env
    launch.return_value.poll.side_effect = [None, -signal.SIGTERM, -signal.SIGTERM]
    launch.return_value.returncode = -signal.SIGTERM
    with pytest.raises(RuntimeError, match="Server exited .* during evaluation"):
        harness.main()
    evaluate.assert_called_once()
    assert not (tmp_path / "result.json").exists()


def test_smoke_only_has_no_result(tmp_path, main_env, monkeypatch):
    evaluate, _ = main_env
    monkeypatch.setattr(sys, "argv", sys.argv[:-1])
    harness.main()
    evaluate.assert_not_called()
    assert not (tmp_path / "result.json").exists()


def test_nonempty_output_preserves_logs_without_launching(tmp_path, main_env):
    evaluate, launch = main_env
    previous = tmp_path / "server.log"
    previous.write_text("previous run\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        harness.main()
    assert exc.value.code == 2
    assert previous.read_text(encoding="utf-8") == "previous run\n"
    assert list(tmp_path.iterdir()) == [previous]
    launch.assert_not_called()
    evaluate.assert_not_called()
