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

"""Record AgentX inputs and audit exports together with phase-level counts."""

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def server_configuration(environment):
    # Record choices without imposing a particular custom launcher's environment.
    # Revisions are caller-provided identifiers, not verified weight hashes.
    return {
        name: environment.get(name)
        for name in (
            "MODEL_DIR",
            "DRAFT_DIR",
            "SERVER_VENV",
            "GPU_MEMORY_UTILIZATION",
            "SERVER_SEED",
            "MODEL_REVISION",
            "DRAFT_REVISION",
        )
    }


def prepare(run_root, server_arguments):
    from evalscope.perf.scenarios.agentx import AgentXScenario

    scenario = AgentXScenario.model_validate_json(
        (run_root / "scenario.json").read_text()
    )
    server = server_configuration(os.environ)
    duration = int(os.environ["DURATION"])
    if scenario.mode == "benchmark" and duration < 900:
        raise ValueError("Benchmark duration must be >=900 seconds")
    if int(os.environ["CLIENT_TIMEOUT"]) <= duration:
        raise ValueError("CLIENT_TIMEOUT must include initialization and drain time")
    trace = run_root / "dataset" / "traces.jsonl"
    environment = {
        key: os.environ[key]
        for key in (
            "SOURCE_ROOT",
            "SERVER_SCRIPT",
            "CONTAINER_IMAGE",
            "CONTAINER_MOUNTS",
            "CLIENT_PYTHON",
            "MODEL_NAME",
            "TOKENIZER_PATH",
            "DATASET_PATH",
            "CONCURRENCY",
            "DURATION",
            "SEED",
            "API_PORT",
            "READINESS_PATH",
            "READINESS_TIMEOUT",
            "CLIENT_TIMEOUT",
            "HOLD_AFTER_RUN",
            "SLURM_JOB_ID",
        )
    }
    commit = subprocess.check_output(
        ["git", "-C", environment["SOURCE_ROOT"], "rev-parse", "HEAD"], text=True
    ).strip()
    import aiperf
    import evalscope.perf.scenarios.agentx as adapter

    manifest = {
        "environment": environment,
        "scenario": scenario.model_dump(),
        "server_configuration": server,
        "server_arguments": server_arguments,
        "harness_commit": commit,
        "harness_sha256": file_sha256(run_root / "harness.slurm"),
        "auditor_sha256": file_sha256(Path(__file__)),
        "server_script_path": str(run_root / "server.sh"),
        "server_script_sha256": file_sha256(run_root / "server.sh"),
        "dataset_path": str(trace.parent),
        "dataset_sha256": file_sha256(trace),
        "client_arch": platform.machine(),
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name) for name in ("evalscope", "aiperf")
        },
        "aiperf_source_root": str(Path(aiperf.__file__).parent),
        "aiperf_source_sha256": {
            name: file_sha256(Path(aiperf.__file__).parent / name)
            for name in ("timing/phase/runner.py", "credit/callback_handler.py")
        },
        "adapter_file": adapter.__file__,
        "adapter_sha256": file_sha256(Path(adapter.__file__)),
    }
    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def profiling_counts(log_text):
    return [
        dict(zip(("completed", "cancelled", "errors"), map(int, counts)))
        for counts in re.findall(
            r"Phase profiling \(profiling\) complete \| completed=(\d+), cancelled=(\d+), errors=(\d+)",
            log_text,
        )
    ]


def audit(summary, records, log_text):
    if summary["status"] != "completed" or summary["error_summary"]:
        raise ValueError("Benchmark did not complete without reported errors")
    for row in records:
        if row.get("error") or row.get("errors"):
            raise ValueError("Request error in export")
        metadata = row["metadata"]
        if metadata["was_cancelled"] or metadata["context_overflow_skip"]:
            raise ValueError("Cancelled or overflow-skipped exported request")
        if row["metrics"]["output_sequence_length"]["value"] <= 0:
            raise ValueError("Missing output token evidence")
    measured = [
        row for row in records if row["metadata"]["benchmark_phase"] == "profiling"
    ]
    count = summary["metrics"]["request_count"]["avg"]
    if not measured or len(measured) != count:
        raise ValueError("Summary and exported request counts disagree")
    if (
        "error_request_count" in summary["metrics"]
        and summary["metrics"]["error_request_count"]["avg"] != 0
    ):
        raise ValueError("Request errors in aggregate metric")
    phases = profiling_counts(log_text)
    if len(phases) != 1:
        raise ValueError("Missing or ambiguous final profiling phase counts")
    if re.search(
        r"Phase profiling \(profiling\) complete[^\n]*grace_period_timeout=True",
        log_text,
    ):
        raise ValueError(
            "Profiling exhausted the grace period; throughput includes an incomplete drain"
        )
    completed, cancelled, errors = (
        phases[0][key] for key in ("completed", "cancelled", "errors")
    )
    if completed != count or errors:
        raise ValueError("Phase totals disagree or include errors")
    if summary["scenario"]["mode"] == "smoke" and (
        cancelled or summary["submission_valid"]
    ):
        raise ValueError(
            "Smoke must have zero cancellations and must not be submission-valid"
        )
    return {
        "status": "completed_with_cancellation" if cancelled else "completed",
        "completed": completed,
        "cancelled": cancelled,
        "errors": errors,
        "warmup_records": sum(
            row["metadata"]["benchmark_phase"] == "warmup" for row in records
        ),
        "submission_valid": summary["submission_valid"],
    }


def main(operation, run_root, server_arguments):
    if operation == "prepare":
        prepare(run_root, server_arguments)
        return
    if operation != "audit":
        raise ValueError("Expected prepare or audit")
    if server_arguments:
        raise ValueError("Server arguments are only accepted by prepare")
    log_text = ""
    try:
        paths = list((run_root / "client").rglob("agentx_summary.json"))
        if len(paths) != 1:
            raise ValueError("Expected exactly one AgentX summary")
        artifact = paths[0].parent / "aiperf"
        log_text = (artifact / "logs/aiperf.log").read_text()
        summary = json.loads(paths[0].read_text())
        records = [
            json.loads(line)
            for line in (artifact / "profile_export.jsonl").read_text().splitlines()
            if line.strip()
        ]
        result = audit(summary, records, log_text)
    except Exception as error:
        phases = profiling_counts(log_text)
        result = {
            "status": "rejected",
            "reason": str(error),
            "error_type": type(error).__name__,
            "completed": None,
            "cancelled": None,
            "errors": None,
            "profiling_phase_counts": phases,
        }
        if len(phases) == 1:
            result.update(phases[0])
        (run_root / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
        raise
    (run_root / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]), sys.argv[3:])
