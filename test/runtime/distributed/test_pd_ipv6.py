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

"""The PD control plane on IPv6 hosts: endpoint formatting, when a ZMQ socket
enables IPv6, the control socket's port, the Mooncake engine's MC_USE_IPV6 and
session id, and one request through real Prefill and Decode executors over IPv4
and IPv6 loopback."""

from __future__ import annotations

import functools
import os
import socket
import sys
import threading
import time
import types

import pytest
import zmq

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from test.runtime import cache_pd_test_utils as cache_pd  # noqa: E402

from tokenspeed.runtime.pd.base.bootstrap import BootstrapInfo  # noqa: E402
from tokenspeed.runtime.pd.base.manager import (  # noqa: E402
    DisaggManagerBase,
    enable_ipv6_for_endpoint,
)
from tokenspeed.runtime.pd.base.mooncake_engine import (  # noqa: E402
    MooncakeTransferEngine,
    enable_mooncake_ipv6_for_host,
)
from tokenspeed.runtime.pd.base.status import TransferPoll  # noqa: E402
from tokenspeed.runtime.pd.decode_executor import DisaggDecodeExecutor  # noqa: E402
from tokenspeed.runtime.pd.mooncake.conn import MooncakeKVBootstrapServer  # noqa: E402
from tokenspeed.runtime.pd.mooncake.entities import KVArgs, KVManagerArgs  # noqa: E402
from tokenspeed.runtime.pd.prefill_executor import DisaggPrefillExecutor  # noqa: E402
from tokenspeed.runtime.pd.topology import PDParallelTopology  # noqa: E402
from tokenspeed.runtime.utils.network import get_free_port, join_host_port  # noqa: E402


def _ipv6_loopback_available() -> bool:
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock:
            sock.bind(("::1", 0))
    except OSError:
        return False
    return True


requires_ipv6_loopback = pytest.mark.skipif(
    not _ipv6_loopback_available(), reason="IPv6 loopback is unavailable"
)


def _free_port_for_both_families() -> int:
    """A port free on IPv4 and IPv6 alike, for a server that binds both, as
    the bootstrap server does; get_free_port() only probes IPv4."""
    if not _ipv6_loopback_available():
        return get_free_port()
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        sock.bind(("::", 0))
        return sock.getsockname()[1]


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("10.0.0.1", "10.0.0.1:8998"),
        ("fd00::1", "[fd00::1]:8998"),
        ("[fd00::1]", "[fd00::1]:8998"),
        ("fe80::1%eth0", "[fe80::1%eth0]:8998"),
        ("prefill-0.prefill", "prefill-0.prefill:8998"),
    ],
)
def test_join_host_port_brackets_only_ipv6_literals(host, expected):
    assert join_host_port(host, 8998) == expected


@pytest.mark.parametrize(
    ("host", "ipv6"),
    [
        ("[fd00::1]", 1),
        ("10.0.0.1", 0),
        ("v6-only.example", 1),
        # With IPv6 on, libzmq resolves a name to IPv6 first and would miss an
        # IPv4-only listener, so a dual-stack name keeps the IPv4 default.
        ("dual-stack.example", 0),
        ("unresolvable.example", 0),
    ],
)
def test_zmq_ipv6_follows_endpoint_host_address_family(monkeypatch, host, ipv6):
    v6 = (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("fd00::1", 0, 0, 0))
    v4 = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 0))
    addresses = {"v6-only.example": [v6], "dual-stack.example": [v6, v4]}

    def getaddrinfo(name, *args, **kwargs):
        if name not in addresses:
            raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
        return addresses[name]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    with zmq.Context() as context, context.socket(zmq.PUSH) as sock:
        enable_ipv6_for_endpoint(sock, f"tcp://{host}:8998")
        assert sock.getsockopt(zmq.IPV6) == ipv6


def test_zmq_ipv6_stays_off_for_malformed_name():
    # socket.getaddrinfo raises UnicodeError, not OSError, for "a..b".
    with zmq.Context() as context, context.socket(zmq.PUSH) as sock:
        enable_ipv6_for_endpoint(sock, "tcp://a..b:8998")
        assert sock.getsockopt(zmq.IPV6) == 0


class _ControlSocketManager(DisaggManagerBase):
    """A manager without transfer buffers to register."""

    def register_buffer_to_engine(self):
        pass


@pytest.mark.parametrize(
    ("host", "ipv6"),
    [("127.0.0.1", 0), pytest.param("::1", 1, marks=requires_ipv6_loopback)],
)
def test_control_socket_is_reachable_on_the_port_it_reports(host, ipv6):
    """bind_server_socket stores the port its own bind took in rank_port, and
    a peer reaches the control socket there."""
    manager = _ControlSocketManager(engine=None)
    try:
        manager.bind_server_socket(host)
        assert isinstance(manager.rank_port, int)
        # IPv4 keeps libzmq's default socket options.
        assert manager.server_socket.getsockopt(zmq.IPV6) == ipv6
        endpoint = f"tcp://{join_host_port(host, manager.rank_port)}"
        with zmq.Context() as context, context.socket(zmq.PUSH) as peer:
            peer.setsockopt(zmq.LINGER, 0)
            enable_ipv6_for_endpoint(peer, endpoint)
            peer.connect(endpoint)
            peer.send(b"ping")
            assert manager.server_socket.poll(5000) == zmq.POLLIN
            assert manager.server_socket.recv() == b"ping"
    finally:
        # Garbage-collecting the context of a socket left open would block.
        manager.server_socket.close()


def test_mooncake_ipv6_keeps_explicit_setting(monkeypatch):
    monkeypatch.setenv("MC_USE_IPV6", "on")
    enable_mooncake_ipv6_for_host("fd00::1")
    assert os.environ["MC_USE_IPV6"] == "on"


@pytest.fixture
def fake_mooncake(monkeypatch):
    """Fake Mooncake's native engine module, which needs RDMA, and start with
    MC_USE_IPV6 unset. Returns the (hostname, MC_USE_IPV6) each engine starts
    with and the session ids that transfers write to."""
    initialized, written_sessions = [], []
    rpc_ports = iter((15123, 15432))

    class TransferEngine:
        def __init__(self) -> None:
            self.rpc_port: int | None = None

        def initialize(self, hostname, metadata_server, protocol, device_name):
            initialized.append((hostname, os.environ.get("MC_USE_IPV6")))
            self.rpc_port = next(rpc_ports)
            return 0

        def get_rpc_port(self):
            return self.rpc_port

        def register_memory(self, ptr, length):
            return 0

        def batch_transfer_sync_write_pages(self, session_id, *pages):
            written_sessions.append(session_id)
            return 0

    engine_module = types.ModuleType("mooncake.engine")
    engine_module.TransferEngine = TransferEngine
    monkeypatch.setitem(sys.modules, "mooncake", types.ModuleType("mooncake"))
    monkeypatch.setitem(sys.modules, "mooncake.engine", engine_module)
    # setenv first, so that monkeypatch also removes the value an engine sets.
    monkeypatch.setenv("MC_USE_IPV6", "")
    monkeypatch.delenv("MC_USE_IPV6")
    return initialized, written_sessions


@pytest.mark.parametrize(
    ("hostname", "mc_use_ipv6", "session_id"),
    [
        ("10.0.0.1", None, "10.0.0.1:15123"),
        ("fd00::1", "1", "[fd00::1]:15123"),
    ],
)
def test_mooncake_engine_follows_local_address_family(
    fake_mooncake, hostname, mc_use_ipv6, session_id
):
    initialized, _ = fake_mooncake
    engine = MooncakeTransferEngine(hostname, gpu_id=0, ib_device=None)
    assert initialized == [(hostname, mc_use_ipv6)]
    assert engine.session_id == session_id


def _wait_for(condition) -> None:
    deadline = time.monotonic() + 10
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting for the PD peer"
        time.sleep(0.01)


@pytest.mark.parametrize(
    ("host", "mc_use_ipv6", "decode_session"),
    [
        ("127.0.0.1", None, "127.0.0.1:15432"),
        pytest.param("::1", "1", "[::1]:15432", marks=requires_ipv6_loopback),
    ],
)
def test_request_crosses_pd_control_plane_over_loopback(
    monkeypatch, fake_mooncake, host, mc_use_ipv6, decode_session
):
    """Bootstrap, Decode registration, the request manifest, the transfer and
    Prefill's final status between real executors on ``host``, which both roles
    take as their local address."""
    initialized, written_sessions = fake_mooncake
    # Both roles take this host's address from get_local_ip_by_remote(), and
    # Prefill its bootstrap server's from HOST_IP.
    for module in ("conn", "decode", "prefill", "receiver"):
        monkeypatch.setattr(
            f"tokenspeed.runtime.pd.mooncake.{module}.get_local_ip_by_remote",
            lambda: host,
        )
    monkeypatch.setenv("HOST_IP", host)
    monkeypatch.delenv("TOKENSPEED_HOST_IP", raising=False)
    # The managers start endless worker threads and have no shutdown; as
    # daemons they cannot keep the test process alive.
    monkeypatch.setattr(
        threading, "Thread", functools.partial(threading.Thread, daemon=True)
    )

    cache_layout = cache_pd.layout(
        cache_pd.group("history", cache_pd.segment("layer.0.kv", shape=(8,)))
    )
    bootstrap_port = _free_port_for_both_families()
    bootstrap = MooncakeKVBootstrapServer(bootstrap_port)

    def start(executor_cls, kv_data_ptr):
        args = KVManagerArgs(
            bootstrap_port=bootstrap_port,
            dist_init_addr=None,
            topology=PDParallelTopology(
                tp_size=1, tp_rank=0, dp_size=1, dp_rank=0, world_size=1, global_rank=0
            ),
            enable_metrics=False,
            enable_dp_attention=False,
            served_model_name="model",
            app_key=None,
            # What resolve_metrics_options() leaves for a server started without
            # --metrics-reporters.
            metrics_reporters=["prometheus"],
        )
        kv_args = KVArgs(
            engine_rank=0,
            kv_data_ptr=kv_data_ptr,
            ib_device=None,
            gpu_id=0,
            cache_layout=cache_layout,
            cache_fields_by_stage=(("layer.0.kv",),),
        )
        return executor_cls(args, kv_args, gloo_group=None)

    try:
        prefill = start(DisaggPrefillExecutor, 0x10000)
        decode = start(DisaggDecodeExecutor, 0x20000)
        # Mooncake reads MC_USE_IPV6 when its first engine starts.
        assert initialized == [(host, mc_use_ipv6), (host, mc_use_ipv6)]
        assert decode.kv_manager.get_session_id() == decode_session

        info = BootstrapInfo(
            bootstrap_host=host, bootstrap_port=bootstrap_port, bootstrap_room=7
        )
        prefill.register("req", info)
        # Decode finds Prefill through the bootstrap server and registers.
        decode.register("req", info)
        status = decode.kv_manager.check_status(7)
        assert status == TransferPoll.Bootstrapped, decode.kv_manager.failure_records
        decode.receivers["req"].prefill(
            block_manifest=cache_pd.block_manifest(("history", (1,)))
        )
        _wait_for(
            lambda: prefill.kv_manager.check_status(7) == TransferPoll.Bootstrapped
        )

        prefill.senders["req"].send(
            is_last=True,
            bootstrap_token=42,
            block_manifest=cache_pd.block_manifest(("history", (2,))),
        )
        _wait_for(
            lambda: decode.kv_manager.check_status(7)
            in (TransferPoll.Success, TransferPoll.Failed)
        )
        assert decode.kv_manager.check_status(7) == TransferPoll.Success
        assert written_sessions == [decode_session]
    finally:
        bootstrap.close()
