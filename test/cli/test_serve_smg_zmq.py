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

"""CPU tests for the direct TokenSpeed serving lifecycle."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tokenspeed.cli import _argsplit, _proc, serve_smg


@pytest.fixture
def engine_flags(monkeypatch):
    monkeypatch.setattr(
        _argsplit,
        "_engine_recognized_flags",
        lambda: {"--tokenizer", "--revision", "--trust-remote-code"},
    )


def test_connection_mode_and_tokenizer_fanout(engine_flags):
    result = _argsplit.split_argv(
        ["model", "--tokenizer", "custom", "--connection-mode", "zmq"]
    )
    assert result.opts.connection_mode == "zmq"
    assert result.engine == ["--model", "model", "--tokenizer", "custom"]
    assert result.gateway == ["--model", "model", "--tokenizer-path", "custom"]
    assert _argsplit.split_argv([]).opts.connection_mode == "grpc"
    assert (
        _argsplit.split_argv(["--connection-mode=grpc"]).opts.connection_mode == "grpc"
    )
    with pytest.raises(ValueError, match="zmq or grpc"):
        _argsplit.split_argv(["--connection-mode", "http"])
    with pytest.raises(ValueError, match="--headless"):
        _argsplit.split_headless_argv(
            ["model", "--headless", "--connection-mode", "zmq"]
        )


@pytest.mark.asyncio
async def test_gateway_zmq_worker_spec(monkeypatch):
    spawn = AsyncMock()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    spec = {
        "url": "ipc:///tmp/ts-test/engine",
        "connection_mode": "zmq",
        "runtime_type": "tokenspeed",
        "labels": {"zmq_control_url": "grpc://127.0.0.1:9000"},
    }
    await _proc.spawn_gateway(
        [], engine_host="127.0.0.1", engine_port=9000, worker_spec=spec
    )
    argv = spawn.await_args.args
    assert "--worker-urls" not in argv
    assert json.loads(argv[argv.index("--worker-specs") + 1]) == [spec]


class Process:
    def __init__(self):
        self.returncode = None
        self.pid = 0
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.stdout.feed_eof()
        self.stderr.feed_eof()
        self.exited = asyncio.Event()

    async def wait(self):
        await self.exited.wait()
        return self.returncode


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "engine", "gateway"])
async def test_zmq_launch_handshake_and_cleanup(monkeypatch, failure):
    engine, gateway = Process(), Process()
    stop = asyncio.Event()
    events, config = [], {}

    async def spawn_engine(args, **kwargs):
        events.append("engine")
        config["engine"] = args
        return engine

    async def spawn_gateway(args, **kwargs):
        events.append("gateway")
        config["gateway"] = args
        config["spec"] = kwargs["worker_spec"]
        config["directory"] = Path(config["spec"]["url"].removeprefix("ipc://")).parent
        assert config["directory"].is_dir()
        return gateway

    async def health(*args, **kwargs):
        assert events == ["engine", "gateway"]  # Both peers must exist to handshake.
        if failure is not None:
            proc = engine if failure == "engine" else gateway
            proc.returncode = 2
            proc.exited.set()
            await asyncio.Event().wait()

    async def ready(*args, **kwargs):
        stop.set()

    terminated = []

    async def terminate(proc, **kwargs):
        terminated.append(proc)
        proc.returncode = 0
        proc.exited.set()

    monkeypatch.setattr(serve_smg, "spawn_engine", spawn_engine)
    monkeypatch.setattr(serve_smg, "spawn_gateway", spawn_gateway)
    monkeypatch.setattr(serve_smg, "wait_grpc_serving", health)
    monkeypatch.setattr(serve_smg, "wait_http_ready", ready)
    monkeypatch.setattr(serve_smg, "terminate_then_kill", terminate)
    monkeypatch.setattr(
        serve_smg, "_start_control_server", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(serve_smg, "kill_process_tree", lambda *args, **kwargs: None)
    ports = iter([19001, 19002, 19003, 19004])
    monkeypatch.setattr(
        serve_smg, "_free_port_avoiding_ephemeral_range", lambda: next(ports)
    )
    code = await asyncio.wait_for(
        serve_smg.run_smg(
            engine_args=["--model", "test/model", "--data-parallel-size", "2"],
            gateway_args=[],
            opts=_argsplit.OrchestratorOpts(connection_mode="zmq"),
            user_host="127.0.0.1",
            user_port=19000,
            _stop_event=stop,
        ),
        timeout=2,
    )
    assert code == (0 if failure is None else 1)
    assert "--zmq-msgpack" in config["engine"]
    assert (
        "--skip-tokenizer-init" not in config["engine"]
    )  # Control health + RL retain tokenizer.
    gateway_args = config["gateway"]
    assert (
        gateway_args[gateway_args.index("--worker-startup-timeout-secs") + 1] == "3600"
    )
    spec = config["spec"]
    assert spec["runtime_type"] == "tokenspeed"
    assert spec["connection_mode"] == "zmq"
    assert spec["dp_size"] == 2
    assert spec["labels"]["zmq_control_url"] == "grpc://127.0.0.1:19001"
    assert spec["zmq_handshake_address"].startswith("tcp://127.0.0.1:")
    assert terminated == [gateway, engine]
    assert not config["directory"].exists()


@pytest.mark.asyncio
async def test_gateway_merges_peer_specs(monkeypatch):
    spawn = AsyncMock()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    local = {"url": "ipc:///tmp/local", "worker_type": "decode"}
    peer = {"url": "ipc:///tmp/prefill", "worker_type": "prefill"}
    await _proc.spawn_gateway(
        ["--worker-specs", json.dumps([peer])],
        engine_host="127.0.0.1",
        engine_port=9000,
        worker_spec=local,
    )
    argv = spawn.await_args.args
    assert argv.count("--worker-specs") == 1
    assert json.loads(argv[argv.index("--worker-specs") + 1]) == [local, peer]


def test_encode_role_has_control_service_without_zmq_handshake():
    args, spec = serve_smg._zmq_launch_args(
        ["--disaggregation-mode", "encode", "--disaggregation-bootstrap-port", "19008"],
        engine_port=19001,
        ipc_directory="/tmp/unused",
        rl_control_url="http://127.0.0.1:19002",
    )
    assert "--zmq-msgpack" not in args
    assert spec["connection_mode"] == "grpc"
    assert spec["worker_type"] == "encode"
    assert spec["bootstrap_port"] == 19008


@pytest.mark.parametrize("mode", ["grpc", "zmq"])
def test_pd_epd_smoke_transport_configuration(mode):
    import os
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / "ci_system" / "smg_transport.sh"
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
set -eu
source "$1"
trap cleanup_smg_transport EXIT
configure_smg_worker encode encode 19100 19200
[[ "${SMG_ENGINE_TRANSPORT_ARGS[*]-}" == "" ]]
configure_smg_worker prefill prefill 19101 19201
configure_smg_worker decode decode 19102 ""
configure_smg_gateway
python3 - "${SMG_GATEWAY_WORKER_ARGS[@]}" <<'PRINT'
import json, sys
print(json.dumps(sys.argv[1:]))
PRINT
""",
            "bash",
            str(script),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "SMG_CONNECTION_MODE": mode,
            "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}",
        },
    )
    argv = json.loads(result.stdout)
    if mode == "grpc":
        assert argv == [
            "--encode",
            "grpc://127.0.0.1:19100",
            "19200",
            "--prefill",
            "grpc://127.0.0.1:19101",
            "19201",
            "--decode",
            "grpc://127.0.0.1:19102",
        ]
    else:
        assert argv[0] == "--worker-specs"
        encode, prefill, decode = json.loads(argv[1])
        assert encode["connection_mode"] == "grpc"
        assert prefill["bootstrap_port"] == 19201
        assert prefill["connection_mode"] == decode["connection_mode"] == "zmq"
        assert prefill["zmq_handshake_address"] != decode["zmq_handshake_address"]
        assert decode["labels"]["zmq_control_url"] == "grpc://127.0.0.1:19102"
        assert not Path(decode["url"].removeprefix("ipc://")).parent.exists()


def test_zmq_rejects_old_gateway_before_spawning(monkeypatch):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(
        sys.modules,
        "smg.router_args",
        SimpleNamespace(RouterArgs=type("OldRouterArgs", (), {})),
    )
    with pytest.raises(SystemExit, match="matching SMG build"):
        serve_smg._check_zmq_gateway_supported()
    monkeypatch.setitem(
        sys.modules,
        "smg.router_args",
        SimpleNamespace(RouterArgs=type("RouterArgs", (), {"worker_specs": None})),
    )
    serve_smg._check_zmq_gateway_supported()


def test_zmq_rejects_remote_dp_owners():
    with pytest.raises(ValueError, match="host-local IPC"):
        serve_smg._zmq_launch_args(
            ["--nnodes", "2", "--data-parallel-size", "2"],
            engine_port=19001,
            ipc_directory="/tmp/unused",
            rl_control_url="http://127.0.0.1:19002",
        )


@pytest.mark.parametrize(
    "args,expected",
    [
        (["--world-size", "4", "--attn-tp-size", "1"], 4),
        (["--tensor-parallel-size", "2"], 1),
        (["--world-size", "4"], 1),
        (["--world-size", "4", "--attn-tp-size", "2"], 2),
    ],
)
def test_zmq_worker_count_follows_engine_parallelism(monkeypatch, args, expected):
    monkeypatch.setattr(serve_smg, "_free_port_avoiding_ephemeral_range", lambda: 19005)
    _, spec = serve_smg._zmq_launch_args(
        args,
        engine_port=19001,
        ipc_directory="/tmp/unused",
        rl_control_url="http://127.0.0.1:19002",
    )
    assert spec["dp_size"] == expected


def test_zmq_pd_spec_advertises_wire_capability(monkeypatch):
    monkeypatch.setattr(serve_smg, "_free_port_avoiding_ephemeral_range", lambda: 19005)
    _, spec = serve_smg._zmq_launch_args(
        ["--disaggregation-mode", "prefill"],
        engine_port=19001,
        ipc_directory="/tmp/unused",
        rl_control_url="http://127.0.0.1:19002",
    )
    assert spec["labels"]["zmq_pd"] == "true"
    assert spec["bootstrap_port"] == 8998
