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

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).with_name("serve_deepseek_v41_flash_pd_1p1d.sh")
CONFIG = (
    SCRIPT.parents[1]
    / "ci/eval/deepseek-v4.1-flash-pd-1p1d-dspark-evalscope-gsm8k-gb300-slurm.yaml"
)


@pytest.fixture
def launcher(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls_dir = tmp_path / "calls"
    calls_dir.mkdir()
    python = bin_dir / "python3"
    python.write_text(f"#!{sys.executable}\n" + r"""
import json, os, signal, sys, time
from pathlib import Path
args = sys.argv[1:]
if args[0] == '-':
    code = sys.stdin.read()
    if 'get_local_ip_by_remote' in code:
        print(os.environ['MOCK_IP'])
        sys.exit(0)
    if 'wait_grpc_serving' not in code:
        raise AssertionError('unexpected inline Python script')
    kind = 'health'
else:
    kind = ('router' if args[1] == 'smg' else
            args[args.index('--disaggregation-mode') + 1])
record = {'kind': kind, 'args': args, 'env': dict(os.environ), 'pid': os.getpid()}
target = Path(os.environ['CALLS_DIR'], str(os.getpid()) + '.json')
pending = target.with_suffix('.tmp')
pending.write_text(json.dumps(record))
pending.replace(target)
if kind == 'health':
    sys.exit(int(os.environ.get('HEALTH_EXIT', '0')))
print(f'{kind}: health status -> SERVING', flush=True)
if kind == os.environ.get('EXIT_ROLE'):
    time.sleep(0.1)
    sys.exit(0)
while True:
    signal.pause()
""")
    python.chmod(0o755)
    for name, body in (("curl", "exit 0"), ("sleep", "/bin/sleep 0.05")):
        path = bin_dir / name
        path.write_text("#!/bin/bash\n" + body + "\n")
        path.chmod(0o755)
    processes = []

    def start(overrides):
        env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "MODEL_PATH": str(tmp_path),
            "PD_CI_LOG_DIR": str(tmp_path / "logs"),
            "CALLS_DIR": str(calls_dir),
            "STARTUP_TIMEOUT": "3",
            "WORKER_SHUTDOWN_TIMEOUT": "1",
            **overrides,
        }
        log = (tmp_path / f"launcher-{len(processes)}.log").open("w")
        process = subprocess.Popen(
            ["bash", str(SCRIPT)], env=env, stdout=log, stderr=log
        )
        log.close()
        processes.append(process)
        return process

    def calls():
        return [json.loads(p.read_text()) for p in calls_dir.glob("*.json")]

    yield start, calls, tmp_path
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        process.wait(timeout=5)
    for call in calls():
        with pytest.raises(ProcessLookupError):
            os.kill(call["pid"], 0)


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("launcher did not reach expected state")


def slurm_env(node):
    return {
        **yaml.safe_load(CONFIG.read_text())["env"],
        "SLURM_JOB_ID": "123",
        "SLURM_STEP_ID": "1",
        "SLURM_STEP_NUM_NODES": "2",
        "SLURM_NODEID": str(node),
        "SLURM_STEP_NODELIST": "worker[0-1]",
        "MOCK_IP": f"192.0.2.{node + 1}",
        "STARTUP_TIMEOUT": "3",
    }


def value(args, flag):
    return args[args.index(flag) + 1]


def test_two_node_roles_keep_engines_independent(launcher):
    start, calls, tmp_path = launcher
    decode = start(slurm_env(1))
    prefill = start(slurm_env(0))
    wait_for(lambda: any(c["kind"] == "router" for c in calls()))
    assert decode.poll() is None and prefill.poll() is None
    records = {c["kind"]: c for c in calls()}
    assert records.keys() == {"prefill", "decode", "router", "health"}
    for role, host in (("prefill", "192.0.2.1"), ("decode", "192.0.2.2")):
        call = records[role]
        args, env = call["args"], call["env"]
        assert (
            not {"SLURM_NODEID", "SLURM_STEP_NODELIST", "SLURM_STEP_NUM_NODES"}
            & env.keys()
        )
        assert env["SLURM_JOB_ID"] == "123"
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
        assert value(args, "--host") == host
        assert value(args, "--dist-init-addr").startswith(host + ":")
        assert (
            value(args, "--world-size") == value(args, "--tensor-parallel-size") == "4"
        )
        assert value(args, "--speculative-algorithm") == "DSPARK"
        assert value(args, "--max-cudagraph-capture-size") == "8"
        assert value(args, "--disaggregation-layerwise-interval") == "0"
        assert (
            value(args, "--disaggregation-ib-device") == "mlx5_0,mlx5_1,mlx5_2,mlx5_3"
        )
        assert "--enable-expert-parallel" in args
        assert ("--disable-prefix-caching" in args) == (role == "decode")
        assert (tmp_path / "logs/123-1" / f"{role}.log").is_file()
    assert records["health"]["args"] == ["-", "192.0.2.1:18346", "192.0.2.2:18347"]
    router = records["router"]
    assert router["env"]["SLURM_STEP_NUM_NODES"] == "2"
    assert value(router["args"], "--prefill") == "grpc://192.0.2.1:18346"
    assert value(router["args"], "--decode") == "grpc://192.0.2.2:18347"
    assert value(router["args"], "--reasoning-parser") == "deepseek_v31"
    decode.terminate()
    assert decode.wait(timeout=5) == 143
    assert not (tmp_path / "logs/123-1/decode.ready").exists()
    prefill.terminate()
    assert prefill.wait(timeout=5) == 143
    decode_output = (tmp_path / "launcher-0.log").read_text()
    prefill_output = (tmp_path / "launcher-1.log").read_text()
    assert "decode: health status" in decode_output
    assert "prefill: health status" not in decode_output
    assert "prefill: health status" in prefill_output
    assert "decode: health status" not in prefill_output


def test_single_node_defaults_still_launch_both_roles(launcher):
    start, calls, _ = launcher
    start({"PD_SLURM": "0"})
    wait_for(lambda: any(c["kind"] == "router" for c in calls()))
    records = {c["kind"]: c for c in calls()}
    assert records.keys() == {"prefill", "decode", "router"}
    for role, gpus in (("prefill", "0,1"), ("decode", "2,3")):
        assert records[role]["env"]["CUDA_VISIBLE_DEVICES"] == gpus
        assert value(records[role]["args"], "--host") == "127.0.0.1"
        assert value(records[role]["args"], "--world-size") == "2"
        assert value(records[role]["args"], "--max-cudagraph-capture-size") == "16"
    assert value(records["router"]["args"], "--reasoning-parser") == "passthrough"


@pytest.mark.parametrize("node", [0, 1])
def test_successful_worker_exit_fails_serving_job(launcher, node):
    start, _, tmp_path = launcher
    role = "prefill" if node == 0 else "decode"
    process = start({**slurm_env(node), "EXIT_ROLE": role})
    assert process.wait(timeout=5) != 0
    assert not (tmp_path / "logs/123-1" / f"{role}.ready").exists()


def test_unreachable_decode_never_starts_router(launcher):
    start, calls, _ = launcher
    start(slurm_env(1))
    process = start({**slurm_env(0), "HEALTH_EXIT": "1"})
    assert process.wait(timeout=5) == 1
    assert not any(c["kind"] == "router" for c in calls())


def test_missing_decode_readiness_times_out(launcher):
    start, calls, _ = launcher
    process = start({**slurm_env(0), "STARTUP_TIMEOUT": "1"})
    assert process.wait(timeout=5) == 1
    assert not any(c["kind"] == "router" for c in calls())


@pytest.mark.parametrize("nodes,rank", [("1", "0"), ("3", "0"), ("2", "2"), ("2", "")])
def test_invalid_slurm_topology_fails_before_workers_start(launcher, nodes, rank):
    start, calls, _ = launcher
    process = start(
        {**slurm_env(0), "SLURM_STEP_NUM_NODES": nodes, "SLURM_NODEID": rank}
    )
    assert process.wait(timeout=5) == 2
    assert calls() == []
