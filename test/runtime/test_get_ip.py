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


"""The address a host advertises to its peers is never a loopback address."""

from __future__ import annotations

import ipaddress
import pathlib
import socket
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

import pytest  # noqa: E402

from tokenspeed.runtime.utils.network import (  # noqa: E402
    get_ip,
    get_local_ip_by_remote,
)


class _NoRoute(socket.socket):
    def connect(self, address):
        raise OSError("Network is unreachable")


@pytest.fixture
def hostname_on_127_0_1_1(monkeypatch):
    # Debian-style /etc/hosts.
    monkeypatch.delenv("TOKENSPEED_HOST_IP", raising=False)
    monkeypatch.delenv("HOST_IP", raising=False)
    monkeypatch.setattr(socket, "gethostbyname", lambda _hostname: "127.0.1.1")


def test_get_ip_skips_a_hostname_on_127_0_1_1(hostname_on_127_0_1_1) -> None:
    assert not ipaddress.ip_address(get_ip()).is_loopback


def test_get_local_ip_by_remote_never_falls_back_to_127_0_1_1(
    hostname_on_127_0_1_1, monkeypatch
) -> None:
    monkeypatch.setattr(socket, "socket", _NoRoute)
    with pytest.raises(ValueError, match="Can not get local ip"):
        get_local_ip_by_remote()
