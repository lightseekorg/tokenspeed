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
# What the stand-in python3 answers for the gateway's handshake-port derivation.
DERIVED_HANDSHAKE_PORT = "25152"


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
    raise AssertionError('unexpected inline Python script')
if args[0] == '-c':
    if 'zmq_handshake_port' not in args[1]:
        raise AssertionError('unexpected inline Python script')
    print(os.environ['DERIVED_HANDSHAKE_PORT'])
    sys.exit(0)
kind = ('router' if args[1] == 'smg' else
        args[args.index('--disaggregation-mode') + 1])
record = {'kind': kind, 'args': args, 'env': dict(os.environ), 'pid': os.getpid()}
target = Path(os.environ['CALLS_DIR'], str(os.getpid()) + '.json')
pending = target.with_suffix('.tmp')
pending.write_text(json.dumps(record))
pending.replace(target)
print(f'{kind}: started', flush=True)
if kind == os.environ.get('EXIT_ROLE'):
    time.sleep(0.1)
    sys.exit(0)
while True:
    signal.pause()
""")
    python.chmod(0o755)
    curl = bin_dir / "curl"
    # The gateway answers at once, its readiness unless READINESS_FAILS.
    curl.write_text(
        "#!/bin/bash\n"
        'if [[ -n "${READINESS_FAILS:-}" && "$*" == *readiness* ]]; then exit 22; fi\n'
        "exit 0\n"
    )
    curl.chmod(0o755)
    sleep = bin_dir / "sleep"
    sleep.write_text("#!/bin/bash\n/bin/sleep 0.05\n")
    sleep.chmod(0o755)
    processes = []

    def start(overrides):
        env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "MODEL_PATH": str(tmp_path),
            "PD_CI_LOG_DIR": str(tmp_path / "logs"),
            "TMPDIR": str(tmp_path),
            "CALLS_DIR": str(calls_dir),
            "DERIVED_HANDSHAKE_PORT": DERIVED_HANDSHAKE_PORT,
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


def test_two_node_roles_dial_the_gateway_on_node_0(launcher):
    start, calls, tmp_path = launcher
    decode = start(slurm_env(1))
    prefill = start(slurm_env(0))
    wait_for(lambda: {c["kind"] for c in calls()} == {"prefill", "decode", "router"})
    assert decode.poll() is None and prefill.poll() is None
    records = {c["kind"]: c for c in calls()}
    # Node 1's decode engine dials node 0's gateway over TCP; node 0's prefill
    # engine dials the handshake the gateway derives from its ipc:// worker.
    handshakes = {
        "prefill": ("127.0.0.1", DERIVED_HANDSHAKE_PORT),
        "decode": ("192.0.2.1", "18348"),
    }
    for role, host in (("prefill", "192.0.2.1"), ("decode", "192.0.2.2")):
        call = records[role]
        args, env = call["args"], call["env"]
        assert args[:3] == ["-m", "tokenspeed.runtime.entrypoints.engine", "--model"]
        assert (
            not {"SLURM_NODEID", "SLURM_STEP_NODELIST", "SLURM_STEP_NUM_NODES"}
            & env.keys()
        )
        assert env["SLURM_JOB_ID"] == "123"
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
        assert value(args, "--host") == host
        assert value(args, "--dist-init-addr").startswith(host + ":")
        assert (
            value(args, "--data-parallel-address"),
            value(args, "--data-parallel-rpc-port"),
        ) == handshakes[role]
        assert value(args, "--zmq-engine-index") == "0"
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
        # Only the prefill role captures prefill graphs; decode never extends.
        assert ("--disable-prefill-graph" in args) == (role == "decode")
        assert "--enforce-eager" not in args
        assert (tmp_path / "logs/123-1" / f"{role}.log").is_file()
    router = records["router"]
    assert router["env"]["SLURM_STEP_NUM_NODES"] == "2"
    assert value(router["args"], "--backend") == "tokenspeed"
    assert value(router["args"], "--prefill").startswith("ipc://")
    assert value(router["args"], "--decode") == "tcp://192.0.2.1:18348"
    decode.terminate()
    assert decode.wait(timeout=5) == 143
    prefill.terminate()
    assert prefill.wait(timeout=5) == 143
    assert not (tmp_path / "logs/123-1/gateway.ready").exists()
    decode_output = (tmp_path / "launcher-0.log").read_text()
    prefill_output = (tmp_path / "launcher-1.log").read_text()
    assert "decode: started" in decode_output
    assert "prefill: started" not in decode_output
    assert "prefill: started" in prefill_output
    assert "decode: started" not in prefill_output


def test_single_node_defaults_still_launch_both_roles(launcher):
    start, calls, _ = launcher
    start({"PD_SLURM": "0"})
    wait_for(lambda: {c["kind"] for c in calls()} == {"prefill", "decode", "router"})
    records = {c["kind"]: c for c in calls()}
    for role, gpus in (("prefill", "0,1"), ("decode", "2,3")):
        args = records[role]["args"]
        assert records[role]["env"]["CUDA_VISIBLE_DEVICES"] == gpus
        assert value(args, "--host") == "127.0.0.1"
        assert value(args, "--world-size") == "2"
        assert value(args, "--max-cudagraph-capture-size") == "16"
        assert value(args, "--data-parallel-address") == "127.0.0.1"
        assert value(args, "--data-parallel-rpc-port") == DERIVED_HANDSHAKE_PORT
    router = records["router"]["args"]
    for role in ("prefill", "decode"):
        assert value(router, f"--{role}").startswith("ipc://")
        assert value(router, f"--{role}").endswith(f"/{role}")


@pytest.mark.parametrize("node", [0, 1])
def test_successful_worker_exit_fails_serving_job(launcher, node):
    start, _, tmp_path = launcher
    role = "prefill" if node == 0 else "decode"
    if node == 1:
        ready = tmp_path / "logs/123-1/gateway.ready"
        ready.parent.mkdir(parents=True)
        ready.write_text("192.0.2.1\n")
    process = start({**slurm_env(node), "EXIT_ROLE": role})
    assert process.wait(timeout=5) != 0
    if node == 0:
        assert not (tmp_path / "logs/123-1/gateway.ready").exists()


def test_missing_gateway_node_times_out(launcher):
    start, calls, _ = launcher
    process = start({**slurm_env(1), "STARTUP_TIMEOUT": "1"})
    assert process.wait(timeout=5) == 1
    assert calls() == []


def test_unregistered_engines_time_out(launcher):
    start, calls, _ = launcher
    process = start({**slurm_env(0), "STARTUP_TIMEOUT": "1", "READINESS_FAILS": "1"})
    assert process.wait(timeout=10) == 1
    assert {c["kind"] for c in calls()} == {"router", "prefill"}


@pytest.mark.parametrize("nodes,rank", [("1", "0"), ("3", "0"), ("2", "2"), ("2", "")])
def test_invalid_slurm_topology_fails_before_workers_start(launcher, nodes, rank):
    start, calls, _ = launcher
    process = start(
        {**slurm_env(0), "SLURM_STEP_NUM_NODES": nodes, "SLURM_NODEID": rank}
    )
    assert process.wait(timeout=5) == 2
    assert calls() == []
