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

"""PD, EPD and multi-node setup on IPv6 hosts: ``host:port`` brackets only
IPv6 literals and parses them back, cross-node listeners bind the host
address's family, the head node's interface is found over IPv6, and the
host's Mooncake engines start with ``MC_USE_IPV6`` on an IPv6 address. The
transfer peers' sockets are ``ts-pd``'s."""

from __future__ import annotations

import os
import socket
import sys
from types import SimpleNamespace

import pytest

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")


from tokenspeed.runtime.pd.host_peer import (  # noqa: E402
    enable_mooncake_ipv6_for_host,
)
from tokenspeed.runtime.utils import launcher  # noqa: E402
from tokenspeed.runtime.utils.network import (  # noqa: E402
    join_host_port,
    split_host_port,
    wildcard_host_for,
)


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
    ("address", "expected"),
    [
        ("10.0.0.1:4000", ("10.0.0.1", 4000)),
        ("prefill-0.prefill:4000", ("prefill-0.prefill", 4000)),
        ("[fd00::1]:4000", ("fd00::1", 4000)),
        ("[fe80::1%eth0]:4000", ("fe80::1%eth0", 4000)),
        # Bare, the last colon separates the port.
        ("fd00::1:4000", ("fd00::1", 4000)),
    ],
)
def test_split_host_port_inverts_join_host_port(address, expected):
    assert split_host_port(address) == expected
    assert split_host_port(join_host_port(*expected)) == expected


@pytest.mark.parametrize(
    "address",
    ["10.0.0.1", "[fd00::1]", "[fd00::1]4000", "fd00::1", "10.0.0.1:0", ":4000"],
)
def test_split_host_port_rejects_a_missing_port(address):
    with pytest.raises(ValueError):
        split_host_port(address)


@pytest.mark.parametrize(
    ("ip", "expected"), [("fd00::1", "::"), ("10.0.0.1", "0.0.0.0")]
)
def test_cross_node_listeners_bind_the_host_ip_family(ip, expected):
    assert wildcard_host_for(ip) == expected


def test_the_interface_reaching_an_ipv6_head_is_found(monkeypatch):
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM) as sock:
            sock.connect(("::1", 9))
    except OSError:
        pytest.skip("no IPv6 loopback")
    monkeypatch.setattr(
        launcher.psutil,
        "net_if_addrs",
        lambda: {
            "lo4": [SimpleNamespace(family=socket.AF_INET, address="127.0.0.1")],
            "lo6": [SimpleNamespace(family=socket.AF_INET6, address="::1")],
        },
    )
    assert launcher.interface_for_host("::1") == "lo6"
    assert launcher.interface_for_host("[::1]") == "lo6"


@pytest.mark.parametrize(
    ("hostname", "expected"), [("fd00::1", "1"), ("10.0.0.1", None)]
)
def test_mooncake_ipv6_follows_the_local_address_family(
    monkeypatch, hostname, expected
):
    # setenv first, so that monkeypatch also removes the value the call sets.
    monkeypatch.setenv("MC_USE_IPV6", "")
    monkeypatch.delenv("MC_USE_IPV6")
    enable_mooncake_ipv6_for_host(hostname)
    assert os.environ.get("MC_USE_IPV6") == expected


def test_mooncake_ipv6_keeps_explicit_setting(monkeypatch):
    monkeypatch.setenv("MC_USE_IPV6", "on")
    enable_mooncake_ipv6_for_host("fd00::1")
    assert os.environ["MC_USE_IPV6"] == "on"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
