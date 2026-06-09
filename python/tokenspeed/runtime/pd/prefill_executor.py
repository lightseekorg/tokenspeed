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
        # request_id -> bootstrap metadata, populated after the prefill forward pass.
        # Request ids and bootstrap rooms are stable across request-pool slot reuse.
        self._request_token: Dict[str, int] = {}
        self._request_spec_candidate_ids: Dict[str, list[int]] = {}
        self._layerwise_token_published = set()

    def store_prefill_token(
        self,
        request_id: str,
        aux_index: int,
        token: int,
        spec_candidate_ids: list[int] | None = None,
    ) -> None:
        """Called by event_loop after prefill forward to record the first output token."""
        self._request_token[request_id] = token
        if spec_candidate_ids is not None:
            self._request_spec_candidate_ids[request_id] = spec_candidate_ids
        if self._layerwise_enabled:
            sender = self.senders.get(request_id)
            if sender is None:
                logger.warning(
                    "Prefill token arrived before sender registration for request_id=%s",
                    request_id,
                )
                return
            self.kv_manager.set_prefill_metadata(
                sender.bootstrap_room, token, spec_candidate_ids
            )
            self._layerwise_token_published.add(request_id)

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

    def _drop_request_state(self, req_id: str) -> None:
        # Best-effort cleanup of all per-request state so failed/aborted
        # requests do not leak into the bookkeeping dicts. request_id is
        # stable (not a reusable slot index), so without explicit pop here
        # these entries would live until the engine restarts.
        sender = self.senders.pop(req_id, None)
        if sender is not None:
            self.kv_manager.discard_expired_metadata_room(sender.bootstrap_room)
        self._local_states.pop(req_id, None)
        self._request_token.pop(req_id, None)
        self._request_spec_candidate_ids.pop(req_id, None)
        self._layerwise_token_published.discard(req_id)

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
            sender = self.senders.get(request_id)
            if sender is None:
                logger.debug(
                    "[prefill][prepare_prefill] skipping request_id=%s without sender",
                    request_id,
                )
                self._drop_request_state(request_id)
                continue
            kv_indices, index_slice, is_last = self._prefill_page_window(op, i, sender)
            if len(kv_indices) == 0 and not is_last:
                continue
            mamba_indices = self._mamba_indices(op, i) if is_last else None
            sender.send_layerwise(
                kv_indices,
                index_slice,
                op.request_pool_indices[i],
                is_last,
                begin_cache_step=begin_cache_step,
                layerwise_interval=self._layerwise_interval,
                wait_for_bootstrap_token=is_last,
                mamba_indices=mamba_indices,
            )

    def _decode(self, op):
        is_last = True

        for i, request_id in enumerate(op.request_ids):
            sender = self.senders.get(request_id)
            if sender is None:
                logger.debug(
                    "[prefill][_decode] skipping request_id=%s without sender",
                    request_id,
                )
                self._drop_request_state(request_id)
                continue
            aux_index = op.request_pool_indices[i]
            bootstrap_token = self._request_token.pop(request_id, -1)
            spec_candidate_ids = self._request_spec_candidate_ids.pop(request_id, None)
            if sender.has_layerwise_transfer():
                if request_id not in self._layerwise_token_published:
                    self.kv_manager.set_prefill_metadata(
                        sender.bootstrap_room, bootstrap_token, spec_candidate_ids
                    )
                self._layerwise_token_published.discard(request_id)
                continue

            bootstrap_room = sender.bootstrap_room
            decode_prefix_len = self._decode_prefix_len(bootstrap_room)
            assert (
                decode_prefix_len % self.page_size == 0
            ), f"decode_prefix_len % page_size != 0 ! {decode_prefix_len=} {self.page_size=}"
            kv_indices = np.array(
                op.occupied_pages[i][decode_prefix_len // self.page_size :],
                dtype=np.int64,
            )
            mamba_indices = self._mamba_indices(op, i)
            logger.debug(
                "[prefill][_decode] rid=%s aux_index=%d kv_indices(len=%d)=%s bootstrap_token=%d",
                request_id,
                aux_index,
                len(kv_indices),
                kv_indices,
                bootstrap_token,
            )
            sender.send(
                kv_indices,
                aux_index,
                is_last,
                bootstrap_token=bootstrap_token,
                spec_candidate_ids=spec_candidate_ids,
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
                to_remove.append(req_id)
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
            self._drop_request_state(req_id)

        return events
