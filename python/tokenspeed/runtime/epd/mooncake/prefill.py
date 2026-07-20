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

"""Prefill-side (data sink) manager for the Mooncake embedding transfer.

Split out of :mod:`tokenspeed.runtime.epd.mooncake.embedding_transfer`; the
per-request receiver it backs lives in
:mod:`tokenspeed.runtime.epd.mooncake.receiver`.
"""

from __future__ import annotations

import threading

from tokenspeed.runtime.epd.entities import (
    EmbeddingArgs,
    EmbeddingManagerArgs,
)
from tokenspeed.runtime.epd.mooncake.conn import (
    MooncakeEmbeddingManagerBase,
)
from tokenspeed.runtime.pd.base.status import TransferPoll
from tokenspeed.runtime.pd.utils import DisaggregationMode
from tokenspeed.runtime.utils.network import get_free_port, get_local_ip_by_remote


class MooncakeEmbeddingManagerPrefill(MooncakeEmbeddingManagerBase):
    """Prefill-side (data sink) manager: holds the discovery caches and a thread
    that consumes the encode side's per-request completion-status frames, marking
    the room Success once all expected responses arrive.
    """

    def __init__(self, args: EmbeddingManagerArgs, embedding_args: EmbeddingArgs):
        super().__init__(args, embedding_args, DisaggregationMode.PREFILL)
        self.required_response_num: dict[int, int] = {}
        self.response_tracker: dict[int, set] = {}
        self.connection_pool: dict[str, list] = {}
        self.prefill_parallel_info: dict[str, dict] = {}
        self._start_status_thread()

    def _start_status_thread(self):
        self.rank_port = get_free_port()
        self.server_socket.bind(f"tcp://{get_local_ip_by_remote()}:{self.rank_port}")

        def loop():
            while True:
                parts = self.server_socket.recv_multipart()
                room = int(parts[0].decode("ascii"))
                status = int(parts[1].decode("ascii"))
                rank = int(parts[2].decode("ascii"))
                if status == TransferPoll.Success and room in self.request_status:
                    self.response_tracker.setdefault(room, set()).add(rank)
                    if len(
                        self.response_tracker[room]
                    ) >= self.required_response_num.get(room, 1):
                        self.update_status(room, TransferPoll.Success)
                elif status == TransferPoll.Failed:
                    self.record_failure(room, "encode failed to send embedding")
                    self.update_status(room, TransferPoll.Failed)

        threading.Thread(target=loop, daemon=True).start()
