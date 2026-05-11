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

from typing import Dict

import numpy as np

from tokenspeed.runtime.pd.base import BootstrapInfo, KVPoll
from tokenspeed.runtime.pd.mooncake.prefill import (
    MooncakeKVManagerPrefill,
    MooncakeKVSender,
)
from tokenspeed.runtime.pd.utils import (
    TransferBackend,
    poll_and_all_reduce,
)
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.dispatch import TypeBasedDispatcher

logger = get_colorful_logger(__name__)

from tokenspeed_scheduler import PD, Forward


class DisaggPrefillExecutor:
    def __init__(
        self, backend: TransferBackend, args, kv_args, gloo_group, page_size: int
    ):
        self.transfer_backend = backend
        self.bootstrap_port = args.bootstrap_port
        self.page_size = page_size
        self._dispatcher = TypeBasedDispatcher(
            [
                (Forward.FlatForwardOp, self._decode),
            ]
        )
        self.senders: Dict[int, MooncakeKVSender] = {}
        self.kv_manager = MooncakeKVManagerPrefill(args, kv_args)
        self.gloo_group = gloo_group
        self._local_states = {}
        self._layerwise_enabled = False
        self._layerwise_interval = 1
        # aux_index -> bootstrap_token, populated by store_prefill_token() after
        # the prefill forward pass, consumed by _decode() when sending KV.
        self._aux_token: Dict[int, int] = {}
        self._layerwise_token_published = set()

    def store_prefill_token(self, aux_index: int, token: int) -> None:
        """Called by event_loop after prefill forward to record the first output token."""
        self._aux_token[aux_index] = token
        if self._layerwise_enabled:
            self.kv_manager.set_bootstrap_token(aux_index, token)
            self._layerwise_token_published.add(aux_index)

    def register_layerwise_step_counter(self, step_counter, interval: int) -> None:
        self._layerwise_enabled = True
        self._layerwise_interval = max(int(interval), 1)
        self.kv_manager.register_layerwise_step_counter(
            step_counter, self._layerwise_interval
        )

    def _bootstrap(self, request_id, info):
        self.senders[request_id] = MooncakeKVSender(
            mgr=self.kv_manager,
            bootstrap_addr=f"{info.bootstrap_host}:{info.bootstrap_port}",
            bootstrap_room=info.bootstrap_room,
        )

    @staticmethod
    def _mamba_indices(op, index: int):
        indices = getattr(op, "mamba_pool_indices", None)
        if indices is None or index >= len(indices):
            return None
        slot = int(indices[index])
        if slot < 0:
            return None
        return np.array([slot], dtype=np.int64)

    def _decode_prefix_len(self, bootstrap_room: int) -> int:
        transfer_info = next(
            t
            for t in self.kv_manager.transfer_infos[bootstrap_room].values()
            if not t.is_dummy
        )
        return transfer_info.decode_prefix_len

    def _prefill_page_window(self, op, index: int, sender):
        decode_prefix_len = self._decode_prefix_len(sender.bootstrap_room)
        assert (
            decode_prefix_len % self.page_size == 0
        ), f"decode_prefix_len % page_size != 0 ! {decode_prefix_len=} {self.page_size=}"

        chunk_begin = op.extend_prefix_lens[index]
        chunk_end = chunk_begin + op.input_lengths[index]
        is_last = chunk_end >= op.prefill_lengths[index]

        decode_prefix_pages = decode_prefix_len // self.page_size
        start_page = max(
            decode_prefix_pages + sender.curr_idx,
            chunk_begin // self.page_size,
        )
        if is_last:
            end_page = (chunk_end + self.page_size - 1) // self.page_size
        else:
            end_page = chunk_end // self.page_size
        end_page = min(end_page, len(op.occupied_pages[index]))
        start_page = min(start_page, end_page)

        index_slice = slice(
            start_page - decode_prefix_pages,
            end_page - decode_prefix_pages,
        )
        return (
            np.array(op.occupied_pages[index][start_page:end_page], dtype=np.int64),
            index_slice,
            is_last,
        )

    def prepare_prefill(self, op) -> None:
        if not self._layerwise_enabled or op.num_extends() == 0:
            return
        begin_cache_step = self.kv_manager.reserve_layerwise_cache_steps()
        for i, request_id in enumerate(op.request_ids[: op.num_extends()]):
            sender = self.senders[request_id]
            kv_indices, index_slice, is_last = self._prefill_page_window(op, i, sender)
            if len(kv_indices) == 0 and not is_last:
                continue
            sender.send_layerwise(
                kv_indices,
                index_slice,
                op.request_pool_indices[i],
                is_last,
                begin_cache_step=begin_cache_step,
                layerwise_interval=self._layerwise_interval,
                wait_for_bootstrap_token=is_last,
                mamba_indices=self._mamba_indices(op, i) if is_last else None,
            )

    def _decode(self, op):
        is_last = True

        for i, request_id in enumerate(op.request_ids):
            aux_index = op.request_pool_indices[i]
            bootstrap_token = self._aux_token.pop(aux_index, -1)
            if self.senders[request_id].has_layerwise_transfer():
                if aux_index not in self._layerwise_token_published:
                    self.kv_manager.set_bootstrap_token(aux_index, bootstrap_token)
                self._layerwise_token_published.discard(aux_index)
                continue

            bootstrap_room = self.senders[request_id].bootstrap_room
            decode_prefix_len = self._decode_prefix_len(bootstrap_room)
            assert (
                decode_prefix_len % self.page_size == 0
            ), f"decode_prefix_len % page_size != 0 ! {decode_prefix_len=} {self.page_size=}"
            kv_indices = np.array(
                op.occupied_pages[i][decode_prefix_len // self.page_size :],
                dtype=np.int64,
            )

            logger.debug(
                "[prefill][_decode] rid=%s aux_index=%d kv_indices(len=%d)=%s bootstrap_token=%d",
                request_id,
                aux_index,
                len(kv_indices),
                kv_indices,
                bootstrap_token,
            )
            mamba_indices = self._mamba_indices(op, i)
            self.senders[request_id].send(
                kv_indices,
                aux_index,
                is_last,
                bootstrap_token=bootstrap_token,
                mamba_indices=mamba_indices,
            )

    def register(self, request_id: str, bootstrap_info: BootstrapInfo):
        self._local_states[request_id] = KVPoll.Bootstrapping
        self._bootstrap(request_id, bootstrap_info)

    def execute(self, op):
        self._dispatcher(op)

    def generate_events(self):
        if not self.senders:
            return []
        polls = poll_and_all_reduce(self.senders.values(), self.gloo_group)

        events = []
        to_remove = []
        for req_id, poll in zip(list(self.senders.keys()), polls):
            if (
                self._local_states[req_id] == KVPoll.Bootstrapping
                and poll == KVPoll.Bootstrapped
            ):
                logger.debug(
                    "[prefill][generate_events] rid=%s -> BootstrappedEvent", req_id
                )
                events.append(PD.BootstrappedEvent(req_id))
                self._local_states[req_id] = KVPoll.Bootstrapped
            elif poll == KVPoll.Failed:
                logger.warning(
                    "[prefill][generate_events] rid=%s -> FailedEvent", req_id
                )
                events.append(PD.FailedEvent(req_id))
            elif (
                self._local_states[req_id] == KVPoll.Bootstrapped
                and poll == KVPoll.Success
            ):
                self._local_states[req_id] = KVPoll.Success
                logger.debug(
                    "[prefill][generate_events] rid=%s -> SucceededEvent", req_id
                )
                events.append(PD.SucceededEvent(req_id))
                to_remove.append(req_id)
            else:
                pass
        for req_id in to_remove:
            del self.senders[req_id]

        return events
