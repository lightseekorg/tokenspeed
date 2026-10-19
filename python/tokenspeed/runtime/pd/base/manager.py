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

from __future__ import annotations

import ipaddress
import socket
import threading
from functools import cache

import zmq

from tokenspeed.runtime.pd.base.status import TransferPoll
from tokenspeed.runtime.utils.network import join_host_port


def enable_ipv6_for_endpoint(sock: zmq.Socket, endpoint: str) -> None:
    """Enable IPv6 on ``sock`` when the host of the ``tcp://`` ``endpoint`` is
    an IPv6 literal (``join_host_port`` brackets it) or a name without an IPv4
    address.

    libzmq resolves only IPv4 addresses unless IPv6 is enabled: binding an
    IPv6 address fails with ENODEV and a connect never delivers. With IPv6
    enabled it resolves a name to its IPv6 address first and would miss an
    IPv4-only listener, so only a name without an IPv4 address enables IPv6.
    A dual-stack name keeps the IPv4 behaviour, even on an IPv6-only host, as
    the lookup does not use ``AI_ADDRCONFIG``.
    """
    host = endpoint.split("://", 1)[-1].rsplit(":", 1)[0]
    if host.startswith("["):
        sock.setsockopt(zmq.IPV6, 1)
        return
    try:
        ipaddress.IPv4Address(host)
        return
    except ValueError:
        pass
    try:
        addresses = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError):
        # A name that does not resolve, or a malformed one such as "a..b"
        # (UnicodeError), keeps the default options and is left to libzmq.
        return
    if all(address[0] != socket.AF_INET for address in addresses):
        sock.setsockopt(zmq.IPV6, 1)


class DisaggManagerBase:
    """Transfer-engine handle + ZMQ control socket + room-keyed status FSM shared
    by the KV (prefill->decode) and embedding (encode->prefill) managers.

    Transport-neutral: the data-plane ``engine`` is constructed by the subclass
    and injected here, so this base carries no vendor (Mooncake) binding -- only
    the control-plane ZMQ socket and the status FSM. It carries no
    KV/MLA/embedding-specific state either; subclasses add their role args/buffers
    and implement :meth:`register_buffer_to_engine`. The subclass MUST set the
    attributes that :meth:`register_buffer_to_engine` reads (e.g. ``kv_args`` /
    ``embedding_args``) BEFORE calling ``super().__init__`` here, because this
    constructor registers the buffers as its last engine step.

    ``request_status`` is the room-keyed FSM the senders/receivers poll;
    :meth:`update_status` advances it monotonically and makes ``Failed`` sticky
    so a late success cannot resurrect a broken transfer.
    """

    def __init__(self, *, engine):
        self.engine = engine
        self.server_socket = zmq.Context().socket(zmq.PULL)
        self.register_buffer_to_engine()

        self.rank_port = None
        self.request_status: dict[int, int] = {}
        self.failure_records: dict[int, str] = {}
        self.failure_lock = threading.Lock()

    def register_buffer_to_engine(self):
        """Register this role's RDMA buffers with the engine. Abstract."""
        raise NotImplementedError

    def get_session_id(self) -> str:
        return self.engine.get_session_id()

    def bind_server_socket(self, host: str) -> None:
        """Bind the control socket to ``host`` on a port the OS picks in the
        same call, and store that port in ``rank_port``.

        A port probed before the bind could be taken by then, and an IPv4 probe
        does not see a port held on IPv6.
        """
        endpoint = f"tcp://{join_host_port(host, '*')}"
        enable_ipv6_for_endpoint(self.server_socket, endpoint)
        self.server_socket.bind(endpoint)
        # The bound endpoint, such as "tcp://[::1]:40123".
        bound = self.server_socket.getsockopt_string(zmq.LAST_ENDPOINT)
        self.rank_port = int(bound.rsplit(":", 1)[1])

    @cache
    def _connect(self, endpoint: str):
        socket = zmq.Context().socket(zmq.PUSH)
        enable_ipv6_for_endpoint(socket, endpoint)
        socket.connect(endpoint)
        # PyZMQ sockets are not thread-safe. Managers run multiple transfer
        # workers, so serialize sends that reuse this cached endpoint socket.
        return socket, threading.Lock()

    def check_status(self, bootstrap_room: int):
        """Status of ``bootstrap_room``; raises ``KeyError`` if never seen."""
        return self.request_status[bootstrap_room]

    def room_status(self, bootstrap_room: int):
        """Status of ``bootstrap_room``, or ``None`` if unknown -- the non-raising
        read for lease/reap probes (an unknown or already-reaped room is a normal
        answer rather than a ``KeyError``)."""
        return self.request_status.get(bootstrap_room)

    def update_status(self, bootstrap_room: int, status: int):
        if bootstrap_room not in self.request_status:
            self.request_status[bootstrap_room] = status
        else:
            # Status is only allowed to be incremented unless either side has
            # observed a failure. Failed is sticky so a late success cannot
            # resurrect a broken transfer.
            if (
                self.request_status[bootstrap_room] == TransferPoll.Failed
                or status == TransferPoll.Failed
            ):
                self.request_status[bootstrap_room] = TransferPoll.Failed
            else:
                self.request_status[bootstrap_room] = max(
                    self.request_status[bootstrap_room], status
                )

    def record_failure(self, bootstrap_room: int, failure_reason: str):
        with self.failure_lock:
            self.failure_records[bootstrap_room] = failure_reason
