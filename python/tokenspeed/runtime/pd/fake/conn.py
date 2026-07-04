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

import logging

import numpy as np
import numpy.typing as npt

from tokenspeed.runtime.pd.base.conn import (
    BaseKVManager,
    BaseKVReceiver,
    BaseKVSender,
    KVArgs,
    KVPoll,
)
from tokenspeed.runtime.pd.utils import DisaggregationMode
from tokenspeed.runtime.utils.server_args import ServerArgs

logger = logging.getLogger(__name__)


# For warmup reqs, we don't kv transfer, we use the fake sender and receiver
class FakeKVSender(BaseKVSender):
    def __init__(self, mgr: BaseKVManager, bootstrap_addr: str, bootstrap_room: int):
        self.has_sent = False

    def poll(self) -> KVPoll:
        if self.has_sent is False:
            # Assume handshake completed instantly
            return KVPoll.WaitingForInput
        else:
            # Assume transfer completed instantly
            logger.info("FakeKVSender poll success")
            return KVPoll.Success

    def init(
        self,
        kv_indices: list[int],
        aux_index: int | None = None,
        decode_prefix_len: int | None = 0,
    ):
        self.decode_prefix_len = decode_prefix_len
        logger.info(
            "FakeKVSender init with kv_indices: %s, aux_index: %s, decode_prefix_len: %s",
            kv_indices,
            aux_index,
            decode_prefix_len,
        )

    def send(self, kv_indices: npt.NDArray[np.int64], start_idx: int | None = 0):
        self.has_sent = True
        logger.info("FakeKVSender send with kv_indices: %s", kv_indices)

    def failure_exception(self):
        raise RuntimeError("Fake KVSender Exception")


class FakeKVReceiver(BaseKVReceiver):
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: int | None = None,
    ):
        self.has_init = False
        self.decode_prefix_len = 0

    def poll(self) -> KVPoll:
        if self.has_init is False:
            # Assume handshake completed instantly
            return KVPoll.WaitingForInput
        else:
            # Assume transfer completed instantly
            logger.info("FakeKVReceiver poll success")
            return KVPoll.Success

    def init(
        self,
        kv_indices: list[int],
        aux_index: int | None = None,
        decode_prefix_len: int | None = 0,
    ):
        self.has_init = True
        self.decode_prefix_len = decode_prefix_len
        logger.info(
            "FakeKVReceiver init with kv_indices: %s, aux_index: %s, decode_prefix_len: %s",
            kv_indices,
            aux_index,
            decode_prefix_len,
        )

    def failure_exception(self):
        raise RuntimeError("Fake KVReceiver Exception")


class FakeKVManager(BaseKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: bool | None = False,
        draft_is_mla_backend: bool | None = False,
    ):
        self.kv_args = args
        self.is_mla_backend = is_mla_backend
        self.draft_is_mla_backend = draft_is_mla_backend
        self.disaggregation_mode = disaggregation_mode
        self.decode_prefix_lengths = {}

    def receive_decode_prefix_info(self, bootstrap_room: int) -> int:
        """Receive decode prefix info from decode side"""
        return self.decode_prefix_lengths.get(bootstrap_room, 0)

    def store_decode_prefix_info(self, bootstrap_room: int, decode_prefix_len: int):
        """Store decode prefix info from decode side"""
        self.decode_prefix_lengths[bootstrap_room] = decode_prefix_len

    def send_decode_prefix_info(self, bootstrap_room: int, decode_prefix_len: int):
        """Fake backend keeps the real interface but uses direct method calls."""
        # In fake implementation, decode prefix info is handled via direct method calls
        pass
