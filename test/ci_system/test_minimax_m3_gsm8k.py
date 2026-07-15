import json
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest
from minimax_m3_gsm8k import (
    MemoryMonitor,
    parse_args,
    parse_gpu_memory,
    probe_health,
    validate_gsm8k_outputs,
)


def _write_fixture(
    root: Path, *, count: int = 1319, score: float = 1.0
) -> tuple[Path, Path]:
    predictions = root / "predictions.jsonl"
    reviews = root / "reviews.jsonl"
    prediction_rows = []
    review_rows = []
    for index in reversed(range(count)):
        prediction_rows.append(
            {
                "index": index,
                "model_output": {
                    "choices": [{"message": {"content": f"answer {index}"}}],
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                    "error": None,
                },
            }
        )
        review_rows.append(
            {
                "index": index,
                "sample_score": {"score": {"value": {"acc": score}}},
            }
        )
    predictions.write_text("".join(json.dumps(row) + "\n" for row in prediction_rows))
    reviews.write_text("".join(json.dumps(row) + "\n" for row in review_rows))
    return predictions, reviews


def test_exact_1319_outputs_pass_in_any_order(tmp_path: Path) -> None:
    predictions, reviews = _write_fixture(tmp_path)

    result = validate_gsm8k_outputs(
        predictions,
        reviews,
        expected_samples=1319,
        minimum_score=0.99,
    )

    assert result["failures"] == []
    assert result["counts"] == {
        "predictions": 1319,
        "reviews": 1319,
        "nonempty_outputs": 1319,
        "positive_usage": 1319,
        "model_errors": 0,
    }
    assert result["quality"] == {
        "correct": 1319,
        "score": 1.0,
        "finite_binary_scores": 1319,
    }


def test_missing_duplicate_empty_usage_and_nonfinite_score_fail(tmp_path: Path) -> None:
    predictions, reviews = _write_fixture(tmp_path)
    prediction_rows = [
        json.loads(line) for line in predictions.read_text().splitlines()
    ]
    review_rows = [json.loads(line) for line in reviews.read_text().splitlines()]
    prediction_rows[-1]["index"] = prediction_rows[-2]["index"]
    prediction_rows[0]["model_output"]["choices"][0]["message"]["content"] = ""
    prediction_rows[0]["model_output"]["usage"]["output_tokens"] = 0
    prediction_rows[0]["model_output"]["error"] = "transport error"
    review_rows[0]["sample_score"]["score"]["value"]["acc"] = float("nan")
    predictions.write_text("".join(json.dumps(row) + "\n" for row in prediction_rows))
    reviews.write_text("".join(json.dumps(row) + "\n" for row in review_rows))

    result = validate_gsm8k_outputs(
        predictions,
        reviews,
        expected_samples=1319,
        minimum_score=0.99,
    )

    assert any("duplicate" in failure for failure in result["failures"])
    assert any("missing indices" in failure for failure in result["failures"])
    assert any("empty content" in failure for failure in result["failures"])
    assert any("invalid usage" in failure for failure in result["failures"])
    assert any("model error" in failure for failure in result["failures"])
    assert result["counts"]["model_errors"] == 1
    assert any("invalid score" in failure for failure in result["failures"])


def test_score_threshold_is_enforced(tmp_path: Path) -> None:
    predictions, reviews = _write_fixture(tmp_path)
    rows = [json.loads(line) for line in reviews.read_text().splitlines()]
    for row in rows[:20]:
        row["sample_score"]["score"]["value"]["acc"] = 0.0
    reviews.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = validate_gsm8k_outputs(
        predictions,
        reviews,
        expected_samples=1319,
        minimum_score=0.99,
    )

    assert result["quality"]["score"] < 0.99
    assert any("below minimum" in failure for failure in result["failures"])


def test_memory_parser_requires_every_declared_gpu(tmp_path: Path) -> None:
    memory = tmp_path / "memory.csv"
    memory.write_text("0, 100\n1, 200\n2, 300\n")

    result = parse_gpu_memory(memory, (0, 1, 2, 3))

    assert result["peak_mib"]["2"] == 300
    assert any("GPU 3" in failure for failure in result["failures"])


class _HealthHandler(BaseHTTPRequestHandler):
    statuses = {"/readiness": 200, "/health": 200, "/health_check": 200}

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(self.statuses.get(self.path, 404))
        self.end_headers()

    def log_message(self, _format: str, *_args) -> None:
        pass


def test_health_probe_checks_gateway_control_and_engine() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _HealthHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        import requests

        with requests.Session() as session:
            result = probe_health(url, url, session=session)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result["failures"] == []
    assert all(
        result[name]["status_code"] == 200
        for name in ("gateway_readiness", "control_health", "engine_health")
    )


def test_minimum_score_cli_is_required(tmp_path: Path) -> None:
    base = [
        "--evalscope",
        "/tmp/evalscope",
        "--model",
        "minimax-m3",
        "--base-url",
        "http://127.0.0.1:8000",
        "--control-url",
        "http://127.0.0.1:8001",
        "--output-dir",
        str(tmp_path),
        "--gpu-ids",
        "0,1,2,3",
    ]
    with pytest.raises(SystemExit):
        parse_args(base)


class _FakeProcess:
    def __init__(self, *, timeout_once: bool = False) -> None:
        self.returncode = None
        self.terminated = False
        self.killed = False
        self.timeout_once = timeout_once

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        if self.timeout_once:
            self.timeout_once = False
            raise subprocess.TimeoutExpired("nvidia-smi", timeout)
        self.returncode = -15 if not self.killed else -9
        return self.returncode


def test_memory_monitor_escalates_only_its_exact_process() -> None:
    process = _FakeProcess(timeout_once=True)
    monitor = MemoryMonitor(process)  # type: ignore[arg-type]

    monitor.stop()

    assert process.terminated is True
    assert process.killed is True
