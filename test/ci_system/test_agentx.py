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

import argparse
import io
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest
import run_agentx
from pipeline import normalize_task
from slurm_submit import parse_args, select_tasks

MODEL = "nvidia/Qwen3.8-Flash-Next-NVFP4"
REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def agentx_result():
    return {
        "status": "completed",
        "submission_valid": True,
        "concurrency": 4,
        "duration": run_agentx.AGENTX_DURATION_SECONDS,
        "scenario": {"mode": "benchmark"},
        "dataset": {"verified": True},
        "metrics": {"output_token_throughput": {"avg": 1000, "unit": "tokens/sec"}},
    }


def test_models_must_advertise_requested_id():
    run_agentx.validate_models({"data": [{"id": MODEL}]}, MODEL)
    with pytest.raises(ValueError, match="advertised exactly once"):
        run_agentx.validate_models({"data": [{"id": "another-model"}]}, MODEL)


def test_control_model_info_must_report_matching_256k_context():
    run_agentx.validate_model_info(
        {"served_model_name": MODEL, "max_context_length": 262144}, MODEL, 262144
    )
    for max_context_length in (32768, None, "262144"):
        with pytest.raises(ValueError, match="262144"):
            run_agentx.validate_model_info(
                {"served_model_name": MODEL, "max_context_length": max_context_length},
                MODEL,
                262144,
            )
    with pytest.raises(ValueError, match="served_model_name"):
        run_agentx.validate_model_info(
            {"served_model_name": "another-model", "max_context_length": 262144},
            MODEL,
            262144,
        )


@pytest.mark.parametrize(
    "override",
    [
        {"status": "failed"},
        {"submission_valid": False},
        {"concurrency": 8},
        {"duration": 60},
        {"scenario": {"mode": "smoke"}},
        {"dataset": {"verified": False}},
        {"metrics": {}},
    ],
)
def test_invalid_result_fails_and_keeps_diagnostics(tmp_path, agentx_result, override):
    agentx_result.update(override)
    summary = tmp_path / "agentx_summary.json"
    summary.write_text(json.dumps(agentx_result))
    with pytest.raises(ValueError):
        run_agentx.validate_result(tmp_path, 4)
    assert json.loads(summary.read_text()) == agentx_result


@pytest.mark.parametrize("count", [0, 2])
def test_missing_or_ambiguous_summary_fails(tmp_path, agentx_result, count):
    for index in range(count):
        directory = tmp_path / str(index)
        directory.mkdir()
        (directory / "agentx_summary.json").write_text(json.dumps(agentx_result))
    with pytest.raises(ValueError, match="expected one AgentX summary"):
        run_agentx.validate_result(tmp_path, 4)


@pytest.mark.parametrize("returncode", [0, 1])
def test_run_records_tokenspeed_scenario_and_propagates_failure(
    monkeypatch, tmp_path, agentx_result, returncode
):
    args = argparse.Namespace(
        outputs_dir=tmp_path / "benchmark",
        base_url="http://127.0.0.1:8000/",
        control_url="http://127.0.0.1:8001/",
        model=MODEL,
        revision="pinned-model-revision",
        engine_version="pinned-engine-version",
        num_gpus=4,
        context_length=262144,
        evalscope="/tmp/client/bin/evalscope",
        parallel=4,
    )

    def server_info(url, timeout):
        assert timeout == 30
        if url == "http://127.0.0.1:8000/v1/models":
            return io.BytesIO(json.dumps({"data": [{"id": MODEL}]}).encode())
        assert url == "http://127.0.0.1:8001/get_model_info"
        return io.BytesIO(
            json.dumps(
                {"served_model_name": MODEL, "max_context_length": 262144}
            ).encode()
        )

    def execute(command, check):
        assert check is False
        assert command[:2] == [args.evalscope, "perf"]
        assert command[command.index("--parallel") + 1] == "4"
        assert command[command.index("--duration") + 1] == "1800"
        assert command[command.index("--data-source") + 1] == "huggingface"
        scenario = json.loads(command[command.index("--scenario") + 1])
        assert scenario["name"] == "agentx"
        assert scenario["mode"] == "benchmark"
        assert scenario["variant"] == "256k"
        assert scenario["num_gpus"] == 4
        assert scenario["engine"] == "tokenspeed"
        assert scenario["engine_version"] == args.engine_version
        assert scenario["tokenizer_revision"] == args.revision
        assert scenario["tokenizer_trust_remote_code"] is True
        assert not {"--number", "--max-tokens", "--multi-turn", "--warmup-num"} & set(
            command
        )
        directory = args.outputs_dir / "model" / "agentx_256k" / "parallel_4"
        directory.mkdir(parents=True)
        (directory / "agentx_summary.json").write_text(json.dumps(agentx_result))
        return subprocess.CompletedProcess(command, returncode)

    monkeypatch.setattr(run_agentx, "urlopen", server_info)
    monkeypatch.setattr(run_agentx.subprocess, "run", execute)
    if returncode:
        with pytest.raises(subprocess.CalledProcessError):
            run_agentx.run(args)
    else:
        run_agentx.run(args)
    assert (args.outputs_dir / "server-models.json").exists()
    assert (args.outputs_dir / "server-model-info.json").exists()
    assert (args.outputs_dir / "command.json").exists()
    with pytest.raises(ValueError, match="must be empty"):
        run_agentx.run(args)


@pytest.mark.parametrize(
    ("context_length", "num_gpus", "message"),
    [(32768, 4, "262144"), (262144, 0, "num_gpus must be positive")],
)
def test_invalid_scenario_settings_fail_before_network(
    tmp_path, context_length, num_gpus, message
):
    args = argparse.Namespace(
        context_length=context_length,
        num_gpus=num_gpus,
        outputs_dir=tmp_path / "benchmark",
    )
    with pytest.raises(ValueError, match=message):
        run_agentx.run(args)
    assert not args.outputs_dir.exists()


def test_slurm_selects_six_agentx_points_with_one_server_configuration(tmp_path):
    args = parse_args(
        [
            "--all",
            "--runner",
            "slurm-gb200-4gpu",
            "--type",
            "perf",
            "--trigger",
            "slurm",
            "--match",
            "tokenspeed-qwen3.8-flash-next-nvfp4-agentx",
            "--artifact-root",
            "/tmp/artifacts",
            "--cache-dir",
            "/tmp/cache",
            "--container-image",
            "example/runner:fixed",
        ]
    )
    tasks = select_tasks(args, REPO_ROOT)
    assert len(tasks) == 6
    assert len({task.name for task in tasks}) == 6
    assert {(task.runner, task.nodes, task.gpus) for task in tasks} == {
        ("slurm-gb200-4gpu", 1, 4)
    }
    configs = [normalize_task(REPO_ROOT / task.config, REPO_ROOT) for task in tasks]
    assert len({config["server"]["command"] for config in configs}) == 1
    shell_python = tmp_path / "python3"
    shell_python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    shell_python.chmod(0o755)
    shell_env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "MODEL_REVISION": "pinned-model-revision",
        "TOKENSPEED_CI_COMMIT": TEST_COMMIT,
    }

    parallels = set()
    for config in configs:
        assert config["triggers"] == ["slurm"]
        assert config["env"]["TOKENSPEED_QWEN4_EXP_QSA_MAX_LOGITS_MB"] == "2048"
        server_args = shlex.split(config["server"]["command"])
        perf_args = shlex.split(config["perf"]["command"])

        def option_value(tokens, option):
            assert tokens.count(option) == 1
            return tokens[tokens.index(option) + 1]

        assert server_args[:2] == ["ts", "serve"]
        assert option_value(server_args, "--model") == MODEL
        assert option_value(server_args, "--tensor-parallel-size") == "4"
        assert option_value(server_args, "--dense-tp-size") == "4"
        assert option_value(server_args, "--expert-parallel-size") == "4"
        assert option_value(server_args, "--moe-tp-size") == "1"
        assert option_value(server_args, "--speculative-algorithm") == "MTP"
        assert option_value(server_args, "--speculative-num-steps") == "3"
        assert json.loads(option_value(server_args, "--hf-overrides")) == {
            "index_share_for_mtp_iteration": True,
            "max_position_embeddings": 262156,
        }
        assert option_value(server_args, "--max-model-len") == "262144"
        assert option_value(perf_args, "--num-gpus") == "4"
        assert option_value(perf_args, "--context-length") == "262144"
        assert option_value(perf_args, "--control-url") == "http://127.0.0.1:8001"
        assert option_value(perf_args, "--engine-version") == "$TOKENSPEED_CI_COMMIT"
        assert any(
            '"$TOKENSPEED_CI_COMMIT"' in command
            for command in config["perf"]["install"]
        )
        parallels.add(int(option_value(perf_args, "--parallel")))
        expanded = subprocess.run(
            ["sh", "-c", config["perf"]["command"]],
            env=shell_env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        assert expanded[expanded.index("--engine-version") + 1] == TEST_COMMIT

        for command in [
            *config["install"],
            config["server"]["command"],
            *config["perf"]["install"],
            config["perf"]["command"],
        ]:
            assert "git rev-parse" not in command
            subprocess.run(["sh", "-n", "-c", command], check=True)

    assert parallels == {1, 2, 4, 8, 16, 32}
