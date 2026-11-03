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

"""PD and EPD transfer setup on IPv6 hosts: ``host:port`` brackets only IPv6
literals, and the host's Mooncake engines start with ``MC_USE_IPV6`` on an
IPv6 address. The transfer peers' sockets are ``ts-pd``'s."""

from __future__ import annotations

import os
import sys

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
