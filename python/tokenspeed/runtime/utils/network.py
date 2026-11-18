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

"""Port, socket, and host networking helpers."""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
import warnings

logger = logging.getLogger(__name__)


def is_port_available(port):
    """Return whether a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))
            s.listen(1)
            return True
        except OSError:
            return False
        except OverflowError:
            return False


def get_free_port():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def join_host_port(host: str, port: int | str) -> str:
    """Join ``host`` and ``port`` for a URL or a ZMQ TCP endpoint.

    An IPv6 literal is bracketed (``[::1]:80``) so its colons do not read as
    the port separator. IPv4 addresses, hostnames and already-bracketed hosts
    are kept as given. A link-local literal keeps its zone ID inside the
    brackets (``[fe80::1%eth0]:80``), the form libzmq accepts. ``port`` may
    also be ZMQ's ``*`` wildcard.
    """
    if is_valid_ipv6_address(host):
        return f"[{host}]:{port}"
    return f"{host}:{port}"


def split_host_port(address: str) -> tuple[str, int]:
    """Split ``host:port`` into the host (brackets removed) and the port.

    Accepts an IPv4 address, a hostname, a bracketed IPv6 literal
    (``[2001:db8::1]:4000``) or a bare IPv6 literal, where the last colon
    separates the port (``2001:db8::1:4000``).
    """
    address = address.strip()
    if address.startswith("["):
        host, sep, port = address[1:].partition("]:")
        if not sep:
            raise ValueError(f"{address!r}: expected [ipv6]:port")
    else:
        host, sep, port = address.rpartition(":")
        if not sep or (":" in host and not is_valid_ipv6_address(host)):
            raise ValueError(f"{address!r}: expected host:port")
    if not host or not port.isdigit() or not 0 < int(port) < 65536:
        raise ValueError(f"{address!r}: expected host:port")
    return host, int(port)


def wildcard_host_for(ip: str) -> str:
    """The wildcard bind address of ``ip``'s family: ``"::"`` for an IPv6
    address, else ``"0.0.0.0"``. A listener that peers dial at ``ip`` binds
    this, as an IPv4 wildcard cannot accept on an IPv6-only host."""
    return "::" if is_valid_ipv6_address(ip) else "0.0.0.0"


def resolve_host(host: str) -> str:
    """The numeric address of ``host`` (an IPv4/IPv6 literal, bracketed or
    not, or a hostname), IPv4 or IPv6, in getaddrinfo's preference order;
    unlike ``socket.gethostbyname`` this resolves IPv6-only names."""
    host = host.strip("[]")
    try:
        ipaddress.ip_address(host.split("%", 1)[0])
        return host
    except ValueError:
        pass
    infos = socket.getaddrinfo(host, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    return infos[0][4][0]


def get_local_ip_by_remote() -> str | None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        pass

    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if ip and ip != "0.0.0.0" and not ipaddress.ip_address(ip).is_loopback:
            return ip
    except Exception:
        pass

    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        s.connect(("2001:4860:4860::8888", 80))
        return s.getsockname()[0]
    except Exception:
        raise ValueError("Can not get local ip")


def get_ip() -> str:
    from tokenspeed.runtime.utils.env import envs

    host_ip = envs.TOKENSPEED_HOST_IP.get() or os.getenv("HOST_IP", "")
    if host_ip:
        return host_ip

    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        # Debian-style /etc/hosts maps the hostname to 127.0.1.1, which no peer can reach.
        if ip and ip != "0.0.0.0" and not ipaddress.ip_address(ip).is_loopback:
            return ip
    except Exception:
        pass

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        pass

    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        s.connect(("2001:4860:4860::8888", 80))
        return s.getsockname()[0]
    except Exception:
        pass

    warnings.warn(
        "Failed to get the IP address, using 0.0.0.0 by default."
        "The value can be set by the environment variable"
        " TOKENSPEED_HOST_IP or HOST_IP.",
        stacklevel=2,
    )
    return "0.0.0.0"
