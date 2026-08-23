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

"""Frontend-side scheduler IPC client for ``AsyncLLM``.

``EngineCoreClient`` owns the ZMQ context and the three sockets that
``AsyncLLM`` uses to talk to the scheduler subprocess:

* ``send_to_scheduler`` — ``PUSH`` socket on
  ``PortArgs.scheduler_input_ipc_name``; carries tokenized requests,
  weight-sync / session / memory-occupation control messages, and the
  load-update watcher.
* ``recv_from_detokenizer`` — ``PULL`` socket on
  ``PortArgs.tokenizer_ipc_name``; receives ``BatchStrOut`` /
  ``BatchTokenIDOut`` / ``BatchEmbeddingOut`` and control-plane replies from
  the scheduler.
* ``recv_load_snapshot`` — ``PULL`` socket on
  ``PortArgs.metrics_ipc_name``; receives the newest immutable load snapshot
  published independently by each scheduler rank.

Concrete (not ABC): tokenspeed has a single transport (ZMQ in-proc
over ``PortArgs``-provided names) and a single caller (``AsyncLLM``),
so the client stays a plain class. If a second transport ever lands,
promoting this to ``EngineCoreClient(ABC)`` is a purely additive
change.
"""

import zmq
import zmq.asyncio

from tokenspeed.runtime.engine.io_struct import AsyncIpcReceiver, IpcSender
from tokenspeed.runtime.utils import get_zmq_socket
from tokenspeed.runtime.utils.server_args import PortArgs


class EngineCoreClient:
    """Owns the scheduler-facing ZMQ sockets for ``AsyncLLM``.

    Instantiated once per ``AsyncLLM`` in the front-end process. The sockets
    are wrapped in the msgpack IPC adapters from ``io_struct``, which keep
    the existing ``send_pyobj`` / ``recv_pyobj`` call-site ergonomics while
    switching the serialization off pickle.
    """

    def __init__(self, port_args: PortArgs):
        self.context = zmq.asyncio.Context(2)
        self.recv_from_detokenizer = AsyncIpcReceiver(
            get_zmq_socket(self.context, zmq.PULL, port_args.tokenizer_ipc_name, True)
        )
        self.send_to_scheduler = IpcSender(
            get_zmq_socket(
                self.context, zmq.PUSH, port_args.scheduler_input_ipc_name, True
            )
        )
        # Metrics are published by multiple scheduler ranks. Keep only one
        # outstanding snapshot per frontend receiver without conflating the
        # independent publisher streams.
        recv_load_snapshot = self.context.socket(zmq.PULL)
        recv_load_snapshot.setsockopt(zmq.RCVHWM, 1)
        recv_load_snapshot.bind(port_args.metrics_ipc_name)
        self.recv_load_snapshot = AsyncIpcReceiver(recv_load_snapshot)
