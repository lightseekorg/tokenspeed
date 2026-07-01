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

"""Embedding-role transport substrate for EPD (encode -> prefill).

The mechanical plumbing (the Mooncake engine + ZMQ control socket + room-keyed
status FSM, and the bootstrap rendezvous server) is the role-neutral shared base
in :mod:`...disaggregation.base`; this module just layers the embedding role's
buffer args onto it. The embedding-specific *semantics* -- the wire frames and
the senders/receivers -- live in :mod:`embedding_transfer`; nothing here carries
any KV/MLA/metrics state.
"""

from __future__ import annotations

import dataclasses

from tokenspeed.runtime.disaggregation.base.bootstrap import DisaggBootstrapServer
from tokenspeed.runtime.disaggregation.base.manager import DisaggManagerBase
from tokenspeed.runtime.disaggregation.mooncake_transfer_engine import (
    MooncakeTransferEngine,
)
from tokenspeed.runtime.disaggregation.utils import DisaggregationMode
from tokenspeed.runtime.utils.network import get_local_ip_by_remote


class EmbeddingTransferError(Exception):
    """Raised on an embedding (encode->prefill) transfer failure."""


@dataclasses.dataclass
class EmbeddingManagerArgs:
    """Connection config for the embedding Mooncake endpoint.

    Embedding transfer has a single DP group today. ``tp_size`` is the number
    of encode/prefill endpoint ranks that participate in that group; the shared
    bootstrap layer still receives it as ``world_size`` with ``dp_size=1``.
    """

    bootstrap_port: int
    tp_size: int
    bootstrap_host: str | None = None


class EmbeddingManagerBase(DisaggManagerBase):
    """Embedding (encode->prefill) manager: the shared engine/socket/status FSM
    (:class:`...base.manager.DisaggManagerBase`) plus the embedding buffer args.
    Carries none of the KV manager's KV/MLA/metrics fields.
    """

    def __init__(
        self,
        args: EmbeddingManagerArgs,
        embedding_args,  # EmbeddingArgs: embedding buffer ptrs + gpu/ib device
        disaggregation_mode: DisaggregationMode,
    ):
        self.args = args
        self.embedding_args = embedding_args
        self.disaggregation_mode = disaggregation_mode
        self.bootstrap_port = args.bootstrap_port
        self.world_size = args.tp_size
        self.dp_size = 1
        # Vendor binding stays out of the neutral base. embedding_args must be
        # set above before super().__init__, which calls register_buffer_to_engine.
        engine = MooncakeTransferEngine(
            hostname=get_local_ip_by_remote(),
            gpu_id=embedding_args.gpu_id,
            ib_device=embedding_args.ib_device,
        )
        super().__init__(engine=engine)

    def register_buffer_to_engine(self):
        ea = self.embedding_args
        if ea.embedding_data_ptr:
            self.engine.register(ea.embedding_data_ptr, ea.embedding_data_len)
        if ea.deepstack_data_ptr:
            self.engine.register(ea.deepstack_data_ptr, ea.deepstack_data_len)


class EmbeddingBootstrapServer(DisaggBootstrapServer):
    """Embedding bootstrap rendezvous: the shared server with no extra fields --
    the encode->prefill handshake needs only the base ip/port + parallel-size
    sync, so there is nothing to layer onto :class:`DisaggBootstrapServer`.
    """
