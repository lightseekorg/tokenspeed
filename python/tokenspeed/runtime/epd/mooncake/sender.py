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

"""Encode-side per-request sender for the Mooncake embedding transfer.

Split out of :mod:`tokenspeed.runtime.epd.mooncake.embedding_transfer`; the
manager it drives lives in :mod:`tokenspeed.runtime.epd.mooncake.encode`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tokenspeed.runtime.epd.entities import (
    EmbeddingChunk,
    EmbeddingTransferError,
)
from tokenspeed.runtime.pd.base.status import TransferPoll

if TYPE_CHECKING:
    import torch


class MooncakeEmbeddingSender:
    """Encode-side per-request sender for one bootstrap room.

    ``poll`` / ``clear`` / ``failure_exception`` are status-only (they touch
    only the manager's status maps). ``send`` queues one contiguous embedding
    tensor described by scalar fields, keeping this class free of any torch
    dependency.
    """

    def __init__(self, mgr, bootstrap_addr: str, bootstrap_room: int):
        self.mgr = mgr  # MooncakeEmbeddingManagerEncode
        self.bootstrap_server_url = bootstrap_addr
        self.bootstrap_room = bootstrap_room
        self.mgr.update_status(bootstrap_room, TransferPoll.Bootstrapping)
        self.conclude_state = None

    def send(
        self,
        *,
        src_embedding_ptr: int,
        n_tokens: int,
        hidden: int,
        dtype: str,
        nbytes: int,
        src_deepstack_ptr: int = 0,
        deepstack_width: int = 0,
        deepstack_nbytes: int = 0,
        copy_event: torch.cuda.Event | None = None,
    ) -> None:
        chunk = EmbeddingChunk(
            room=self.bootstrap_room,
            src_embedding_ptr=src_embedding_ptr,
            n_tokens=n_tokens,
            hidden=hidden,
            dtype=dtype,
            nbytes=nbytes,
            src_deepstack_ptr=src_deepstack_ptr,
            deepstack_width=deepstack_width,
            deepstack_nbytes=deepstack_nbytes,
            copy_event=copy_event,
        )
        self.mgr.add_transfer_request(self.bootstrap_room, chunk)

    def poll(self) -> int:
        # No Bootstrapping timeout here: the never-registered-receiver case is
        # handled by the manager's _park_reaper (it fails rooms whose parked chunk
        # outlives bootstrap_time_out), so this only reports terminal status.
        if self.conclude_state is None:
            status = self.mgr.check_status(self.bootstrap_room)
            if status in (TransferPoll.Success, TransferPoll.Failed):
                self.conclude_state = status
            return status
        return self.conclude_state

    def clear(self) -> None:
        if self.bootstrap_room in self.mgr.request_status:
            self.mgr.request_status.pop(self.bootstrap_room)

    def failure_exception(self):
        if self.conclude_state is None:
            self.conclude_state = TransferPoll.Failed
        self.clear()
        with self.mgr.failure_lock:
            failure_reason = self.mgr.failure_records.pop(
                self.bootstrap_room, "Failed due to an unknown reason from another rank"
            )
        raise EmbeddingTransferError(
            self.bootstrap_room, failure_reason, self.bootstrap_server_url
        )
