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

import threading
from functools import cache
from typing import Dict

import zmq

from tokenspeed.runtime.disaggregation.base.poll import TransferPoll


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
        self.request_status: Dict[int, int] = {}
        self.failure_records: Dict[int, str] = {}
        self.failure_lock = threading.Lock()

    def register_buffer_to_engine(self):
        """Register this role's RDMA buffers with the engine. Abstract."""
        raise NotImplementedError

    def get_session_id(self) -> str:
        return self.engine.get_session_id()

    @cache
    def _connect(self, endpoint: str):
        socket = zmq.Context().socket(zmq.PUSH)
        socket.connect(endpoint)
        return socket

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
