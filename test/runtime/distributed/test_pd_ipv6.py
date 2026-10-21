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

"""PD and EPD transfer setup on IPv6 hosts: endpoint formatting, when a ZMQ
socket enables IPv6, the control socket's port, and the Mooncake engine's
MC_USE_IPV6 and session id."""

from __future__ import annotations

import os
import socket
import sys
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


from tokenspeed.runtime.pd.base.manager import (  # noqa: E402
    enable_ipv6_for_endpoint,
)
from tokenspeed.runtime.pd.base.mooncake_engine import (  # noqa: E402
    MooncakeTransferEngine,
    enable_mooncake_ipv6_for_host,
)
from tokenspeed.runtime.utils.network import join_host_port  # noqa: E402


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
