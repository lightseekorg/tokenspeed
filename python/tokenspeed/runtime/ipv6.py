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

"""Lightweight IPv6 address and transport bootstrap helpers.

Kept free of ``torch`` and device-environment imports so standalone host
services (such as shared L2 providers) can import it without a GPU runtime.
"""

from __future__ import annotations

import ipaddress
import os


def is_valid_ipv6_address(address: str) -> bool:
    """Return True when ``address`` parses as an IPv6 literal."""
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def enable_mooncake_ipv6_for_host(hostname: str) -> None:
    """Make Mooncake's handshake use IPv6 when ``hostname``, the address the
    host's transfer engine starts on, is an IPv6 literal.

    Mooncake opens its P2P handshake sockets as IPv4 unless ``MC_USE_IPV6``
    is set, and reads it when its first engine starts, so callers set it
    before starting an engine; an explicit setting is kept.
    """
    if is_valid_ipv6_address(hostname):
        os.environ.setdefault("MC_USE_IPV6", "1")
