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

import numpy as np

from tokenspeed.runtime.execution.block_table import (
    select_block_table,
    unpadded_block_table_row,
)
from tokenspeed.runtime.pd.base.bootstrap import BootstrapInfo
from tokenspeed.runtime.pd.base.status import TransferPoll
from tokenspeed.runtime.pd.cache_protocol import (
    build_cache_layerwise_page_selection,
    build_cache_page_manifest,
    validate_cache_manifest_pair,
)
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
                (Forward.Batch, self._decode),
            ]
        )
        self.uses_cache_contract = kv_args.cache_layout is not None
        self.cache_layout = kv_args.cache_layout
        self.senders: dict[str, MooncakeKVSender] = {}
        self.kv_manager = MooncakeKVManagerPrefill(args, kv_args)
        self.gloo_group = gloo_group
        self._local_states = {}
        self._layerwise_enabled = False
        self._layerwise_interval = 1
        # request_id -> bootstrap metadata, populated after the prefill forward pass.
        # Request ids and bootstrap rooms are stable across request-pool slot reuse.
        self._request_token: dict[str, int] = {}
        self._request_spec_candidate_ids: dict[str, list[int]] = {}
        self._layerwise_token_published = set()

    def store_prefill_token(
        self,
        request_id: str,
        aux_index: int,
        token: int,
        spec_candidate_ids: list[int] | None = None,
    ) -> None:
        """Called by event_loop after prefill forward to record the first output token."""
        if self.uses_cache_contract and spec_candidate_ids is not None:
            raise NotImplementedError(
                "Paged cache PD does not support speculative/MTP bootstrap candidates"
            )
        if self.uses_cache_contract and (
            isinstance(token, bool) or not isinstance(token, int) or token < 0
        ):
            raise ValueError("Paged cache PD requires a non-negative bootstrap token")
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
                sender.bootstrap_room,
                token,
                spec_candidate_ids,
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

    def _decode_prefix_len(self, bootstrap_room: int) -> int:
        transfer_info = next(
            t
            for t in self.kv_manager.transfer_infos[bootstrap_room].values()
            if not t.is_dummy
        )
        return transfer_info.decode_prefix_len

    def _prefill_page_window(self, op, index: int, sender, table=None):
        if table is None:
            table = select_block_table(op)
        pages = unpadded_block_table_row(table, index)
        decode_prefix_len = self._decode_prefix_len(sender.bootstrap_room)
        if decode_prefix_len % self.page_size != 0:
            raise ValueError(
                "decode_prefix_len must be divisible by page_size: "
                f"{decode_prefix_len=} {self.page_size=}"
            )

        chunk_begin = op.extend_prefix_lens[index]
        chunk_end = chunk_begin + op.input_lengths[index]
        is_last = chunk_end >= op.prefill_lengths[index]

        decode_prefix_pages = decode_prefix_len // self.page_size
        # Transfer progress is relative to D's missing-page window. P's local
        # prefix may be deeper and must not hide pages that D still needs.
        start_page = decode_prefix_pages + sender.curr_idx
        if is_last:
            end_page = (chunk_end + self.page_size - 1) // self.page_size
        else:
            end_page = chunk_end // self.page_size
        end_page = min(end_page, len(pages))
        start_page = min(start_page, end_page)

        index_slice = slice(
            start_page - decode_prefix_pages,
            end_page - decode_prefix_pages,
        )
        return (
            np.array(pages[start_page:end_page], dtype=np.int64),
            index_slice,
            is_last,
        )

    def prepare_prefill(self, op) -> None:
        if not self._layerwise_enabled or op.num_extends() == 0:
            return
        if self.uses_cache_contract:
            self._prepare_cache_prefill(op)
            return
        table = select_block_table(op)
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
            kv_indices, index_slice, is_last = self._prefill_page_window(
                op, i, sender, table
            )
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
                mamba_indices=None,
            )

    def _prepare_cache_prefill(self, op) -> None:
        """Preflight and enqueue one group-aware CachePD chunk per request.

        Decode's manifest is immutable for the request. Each Prefill chunk
        selects newly ready source pages and positions inside that manifest;
        the producer step range is reserved once for the whole forward batch.
        """
        pending = []
        for index, request_id in enumerate(op.request_ids[: op.num_extends()]):
            sender = self.senders.get(request_id)
            if sender is None:
                logger.debug(
                    "[prefill][prepare_prefill] skipping request_id=%s without sender",
                    request_id,
                )
                self._drop_request_state(request_id)
                continue
            transfer_infos = self.kv_manager.transfer_infos.get(
                sender.bootstrap_room, {}
            )
            if not transfer_infos:
                raise RuntimeError("Paged cache destination metadata is unavailable")
            destinations = tuple(
                info for info in transfer_infos.values() if not info.is_dummy
            )
            # Idle representative ranks use the normal post-forward dummy
            # completion path; they have no cache fields to overlap.
            if not destinations:
                continue
            first = destinations[0]
            if first.page_manifest is None or first.peer_cache_layout is None:
                raise RuntimeError("Paged cache destination metadata is unavailable")
            prefix_len = first.page_manifest.prefix_len
            prompt_len = first.page_manifest.prompt_len
            chunk_begin = int(op.extend_prefix_lens[index])
            chunk_end = chunk_begin + int(op.input_lengths[index])
            if chunk_end > prompt_len:
                raise ValueError(
                    "Paged cache Prefill chunk extends past Decode's prompt manifest"
                )
            is_last = chunk_end == prompt_len
            selection = build_cache_layerwise_page_selection(
                op,
                layout=self.cache_layout,
                request_row=index,
                prefix_len=prefix_len,
                prompt_len=prompt_len,
                chunk_start=chunk_begin,
                chunk_end=chunk_end,
            )
            source_pages = tuple(
                page_id
                for group in selection.groups
                for page_id in group.source_page_ids
            )
            if not source_pages and not is_last:
                continue
            pending.append(
                (
                    sender,
                    np.asarray(source_pages, dtype=np.int64),
                    slice(0, 0),
                    op.request_pool_indices[index],
                    is_last,
                    selection,
                )
            )

        # The attention backend records one producer range for every forward,
        # including batches whose current chunk completes no history page.
        begin_cache_step = self.kv_manager.reserve_layerwise_cache_steps()
        for sender, page_ids, index_slice, aux_index, is_last, selection in pending:
            sender.send_layerwise(
                page_ids,
                index_slice,
                aux_index,
                is_last,
                begin_cache_step=begin_cache_step,
                layerwise_interval=self._layerwise_interval,
                wait_for_bootstrap_token=is_last,
                cache_page_selection=selection,
            )

    def _decode(self, op):
        if self.uses_cache_contract:
            self._cache_decode(op)
            return
        is_last = True
        table = select_block_table(op)

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
                        sender.bootstrap_room,
                        bootstrap_token,
                        spec_candidate_ids,
                    )
                self._layerwise_token_published.discard(request_id)
                continue

            bootstrap_room = sender.bootstrap_room
            decode_prefix_len = self._decode_prefix_len(bootstrap_room)
            if decode_prefix_len % self.page_size != 0:
                raise ValueError(
                    "decode_prefix_len must be divisible by page_size: "
                    f"{decode_prefix_len=} {self.page_size=}"
                )
            kv_indices = np.array(
                unpadded_block_table_row(table, i)[
                    decode_prefix_len // self.page_size :
                ],
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
            sender.send(
                kv_indices,
                aux_index,
                is_last,
                bootstrap_token=bootstrap_token,
                spec_candidate_ids=spec_candidate_ids,
                mamba_indices=None,
            )

    def _cache_decode(self, op) -> None:
        if self._layerwise_enabled:
            # Layerwise already streamed the KV during prepare_prefill; only the
            # bootstrap token still needs publishing on the last chunk.
            for request_id in op.request_ids:
                sender = self.senders.get(request_id)
                if sender is None or not sender.has_layerwise_final():
                    continue
                token = self._request_token.get(request_id)
                if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                    raise RuntimeError("Paged cache bootstrap token is unavailable")
                self.kv_manager.set_prefill_metadata(sender.bootstrap_room, token, None)
                self._request_token.pop(request_id, None)
            return
        pending = []
        for index, request_id in enumerate(op.request_ids):
            sender = self.senders.get(request_id)
            if sender is None:
                continue
            if sender.has_layerwise_final():
                token = self._request_token.pop(request_id, None)
                if request_id not in self._layerwise_token_published:
                    if (
                        isinstance(token, bool)
                        or not isinstance(token, int)
                        or token < 0
                    ):
                        raise RuntimeError("Paged cache bootstrap token is unavailable")
                    self.kv_manager.set_prefill_metadata(
                        sender.bootstrap_room, token, None
                    )
                self._layerwise_token_published.discard(request_id)
                continue
            transfer_infos = self.kv_manager.transfer_infos.get(
                sender.bootstrap_room, {}
            )
            if not transfer_infos:
                raise RuntimeError("Paged cache destination metadata is unavailable")
            destinations = [
                info for info in transfer_infos.values() if not info.is_dummy
            ]
            if not destinations:
                # A rank with no data edge still participates in the Prefill
                # TP status collective. Decode TP rank zero sends it one dummy
                # rendezvous; enqueueing a final no-op lets the existing
                # transfer worker mark this rank successful without DMA.
                token = self._request_token.get(request_id)
                if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                    raise RuntimeError("Paged cache bootstrap token is unavailable")
                pending.append(
                    (
                        request_id,
                        sender,
                        op.request_pool_indices[index],
                        token,
                        None,
                        np.empty(0, dtype=np.int64),
                    )
                )
                continue
            destination = destinations[0]
            if (
                destination.page_manifest is None
                or destination.peer_cache_layout is None
            ):
                raise RuntimeError("Paged cache destination metadata is unavailable")
            manifest = build_cache_page_manifest(
                op,
                layout=self.cache_layout,
                request_row=index,
                prefix_len=destination.page_manifest.prefix_len,
                prompt_len=destination.page_manifest.prompt_len,
            )
            for destination in destinations:
                if (
                    destination.page_manifest is None
                    or destination.peer_cache_layout is None
                ):
                    raise RuntimeError(
                        "Paged cache destination metadata is unavailable"
                    )
                validate_cache_manifest_pair(
                    manifest,
                    destination.page_manifest,
                    self.cache_layout,
                    destination.peer_cache_layout,
                )
            token = self._request_token.get(request_id)
            if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                raise RuntimeError("Paged cache bootstrap token is unavailable")
            page_ids = np.asarray(
                tuple(
                    page_id for group in manifest.groups for page_id in group.page_ids
                ),
                dtype=np.int64,
            )
            pending.append(
                (
                    request_id,
                    sender,
                    op.request_pool_indices[index],
                    token,
                    manifest,
                    page_ids,
                )
            )

        for request_id, sender, aux_index, token, manifest, page_ids in pending:
            # If an earlier non-final layerwise task was submitted but the
            # final task was not, fall back to one complete typed transfer.
            # The full-manifest ABI requires its flat page slice to start at 0.
            if sender.has_layerwise_transfer() and not sender.has_layerwise_final():
                sender.curr_idx = 0
            sender.send(
                page_ids,
                aux_index,
                True,
                bootstrap_token=token,
                page_manifest=manifest,
            )
            self._request_token.pop(request_id, None)

    def register(
        self,
        request_id: str,
        bootstrap_info: BootstrapInfo,
    ):
        self._local_states[request_id] = TransferPoll.Bootstrapping
        self._bootstrap(request_id, bootstrap_info)

    def abort(self, request_id: str, bootstrap_info: BootstrapInfo) -> None:
        """EPD: the prefill aborted this request before registering a KV sender
        (embedding receive timed out). Signal the dual-dispatched decode so its KV
        receiver fails instead of waiting forever. No sender was registered, so
        there is nothing to tear down on this side."""
        self.kv_manager.abort_room(
            bootstrap_info.bootstrap_room,
            f"EPD: prefill aborted request {request_id} (embedding receive timed out)",
        )

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
                self._local_states[req_id] == TransferPoll.Bootstrapping
                and poll == TransferPoll.Bootstrapped
            ):
                logger.debug(
                    "[prefill][generate_events] rid=%s -> BootstrappedEvent", req_id
                )
                events.append(PD.BootstrappedEvent(req_id))
                self._local_states[req_id] = TransferPoll.Bootstrapped
            elif poll == TransferPoll.Failed:
                logger.warning(
                    "[prefill][generate_events] rid=%s -> FailedEvent", req_id
                )
                events.append(PD.FailedEvent(req_id))
                to_remove.append(req_id)
            elif (
                self._local_states[req_id] == TransferPoll.Bootstrapped
                and poll == TransferPoll.Success
            ):
                self._local_states[req_id] = TransferPoll.Success
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
