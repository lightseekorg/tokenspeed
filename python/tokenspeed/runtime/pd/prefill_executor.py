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

from tokenspeed.runtime.pd.base.bootstrap import BootstrapInfo
from tokenspeed.runtime.pd.base.status import TransferPoll
from tokenspeed.runtime.pd.cache_protocol import (
    build_cache_block_manifest,
    build_cache_layerwise_block_selection,
)
from tokenspeed.runtime.pd.mooncake.prefill import (
    MooncakeKVManagerPrefill,
    MooncakeKVSender,
)
from tokenspeed.runtime.pd.utils import poll_and_all_reduce
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

from tokenspeed_scheduler import PD, Forward


class DisaggPrefillExecutor:
    def __init__(self, args, kv_args, gloo_group):
        self.cache_layout = kv_args.cache_layout
        self.senders: dict[str, MooncakeKVSender] = {}
        self.kv_manager = MooncakeKVManagerPrefill(args, kv_args)
        self.gloo_group = gloo_group
        self._local_states = {}
        self._layerwise_enabled = False
        self._layerwise_interval = 1

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
            sender.clear()
            self.kv_manager.discard_room(sender.bootstrap_room)
        self._local_states.pop(req_id, None)

    def prepare_prefill(self, op) -> None:
        if not self._layerwise_enabled or op.num_extends() == 0:
            return
        self._prepare_cache_prefill(op)

    def _prepare_cache_prefill(self, op) -> None:
        """Preflight and enqueue one group-aware CachePD chunk per request.

        Decode's manifest is immutable for the request. Each Prefill chunk
        selects newly ready source blocks and positions inside that manifest;
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
                raise RuntimeError("Cache-transfer destination metadata is unavailable")
            destinations = tuple(
                info for info in transfer_infos.values() if not info.is_dummy
            )
            # Idle representative ranks use the normal post-forward dummy
            # completion path; they have no cache fields to overlap.
            if not destinations:
                continue
            first = destinations[0]
            if first.block_manifest is None:
                raise RuntimeError("Cache-transfer destination metadata is unavailable")
            prefix_len = first.block_manifest.prefix_len
            prompt_len = first.block_manifest.prompt_len
            chunk_begin = int(op.extend_prefix_lens[index])
            chunk_end = chunk_begin + int(op.input_lengths[index])
            if chunk_end > prompt_len:
                raise ValueError(
                    "Cache-transfer Prefill chunk extends past Decode's prompt manifest"
                )
            is_last = chunk_end == prompt_len
            # On the first submitted chunk, include blocks that Prefill found in
            # its local cache but Decode still needs. Once a chunk has been
            # submitted, each later selection starts at its own chunk boundary
            # so already transferred blocks are not sent again.
            selection_start = chunk_begin
            if not sender.layerwise_chunk_submitted():
                selection_start = min(selection_start, prefix_len)
            block_selection = build_cache_layerwise_block_selection(
                op,
                layout=self.cache_layout,
                request_row=index,
                prefix_len=prefix_len,
                prompt_len=prompt_len,
                chunk_start=selection_start,
                chunk_end=chunk_end,
            )
            for destination in destinations:
                if destination.block_manifest is None:
                    raise RuntimeError(
                        "Cache-transfer destination metadata is unavailable"
                    )
                if (
                    destination.block_manifest.prefix_len != prefix_len
                    or destination.block_manifest.prompt_len != prompt_len
                ):
                    raise ValueError(
                        "Cache-transfer destinations disagree on the prompt window"
                    )
            if (
                not any(group.source_block_ids for group in block_selection.groups)
                and not is_last
            ):
                continue
            pending.append(
                (
                    sender,
                    is_last,
                    block_selection,
                )
            )

        # The attention backend records one producer range for every forward,
        # including batches whose current chunk completes no history block.
        begin_cache_step = self.kv_manager.reserve_layerwise_cache_steps()
        for sender, is_last, block_selection in pending:
            sender.send_layerwise(
                is_last,
                begin_cache_step=begin_cache_step,
                layerwise_interval=self._layerwise_interval,
                wait_for_bootstrap_token=is_last,
                cache_block_selection=block_selection,
            )

    def _cache_decode(self, op) -> None:
        """Publish each request's bootstrap payload and finish its transfer.

        ``op`` is the plan's remote-decode batch, and it is self-contained:
        the scheduler emits a row only once the final chunk's result has
        landed, so ``decode_input_ids[i]`` is the bootstrap token (the
        sampled first decode token) and ``spec_candidate_ids[i]`` the
        drafter's candidate rows (empty without speculation).
        """
        pending = []
        for index, request_id in enumerate(op.request_ids):
            sender = self.senders.get(request_id)
            if sender is None:
                continue
            token = int(op.decode_input_ids[index])
            if token < 0:
                raise RuntimeError("Cache-transfer bootstrap token is unavailable")
            spec_candidate_ids = list(op.spec_candidate_ids[index]) or None

            if self._layerwise_enabled:
                if not sender.layerwise_final_chunk_submitted():
                    transfer_infos = self.kv_manager.transfer_infos.get(
                        sender.bootstrap_room, {}
                    )
                    if transfer_infos and all(
                        info.is_dummy for info in transfer_infos.values()
                    ):
                        # Idle representative ranks have no cache fields to
                        # stream, but still participate in the Prefill TP
                        # status collective. Submit their final no-op only
                        # after the Prefill forward has completed.
                        sender.send(
                            True,
                            bootstrap_token=token,
                            spec_candidate_ids=spec_candidate_ids,
                            block_manifest=None,
                        )
                    continue
                # Layerwise already streamed the KV during prepare_prefill;
                # only the bootstrap payload still needs publishing.
                self.kv_manager.set_prefill_metadata(
                    sender.bootstrap_room, token, spec_candidate_ids
                )
                continue

            if sender.layerwise_final_chunk_submitted():
                self.kv_manager.set_prefill_metadata(
                    sender.bootstrap_room, token, spec_candidate_ids
                )
                continue
            transfer_infos = self.kv_manager.transfer_infos.get(
                sender.bootstrap_room, {}
            )
            if not transfer_infos:
                raise RuntimeError("Cache-transfer destination metadata is unavailable")
            destinations = [
                info for info in transfer_infos.values() if not info.is_dummy
            ]
            if not destinations:
                # A rank with no data edge still participates in the Prefill
                # TP status collective. Decode TP rank zero sends it one dummy
                # rendezvous; enqueueing a final no-op lets the existing
                # transfer worker mark this rank successful without DMA.
                pending.append((sender, token, spec_candidate_ids, None))
                continue
            destination = destinations[0]
            if destination.block_manifest is None:
                raise RuntimeError("Cache-transfer destination metadata is unavailable")
            block_manifest = build_cache_block_manifest(
                op,
                layout=self.cache_layout,
                request_row=index,
                prefix_len=destination.block_manifest.prefix_len,
                prompt_len=destination.block_manifest.prompt_len,
            )
            for destination in destinations:
                if destination.block_manifest is None:
                    raise RuntimeError(
                        "Cache-transfer destination metadata is unavailable"
                    )
                if (
                    destination.block_manifest.prefix_len != block_manifest.prefix_len
                    or destination.block_manifest.prompt_len
                    != block_manifest.prompt_len
                ):
                    raise ValueError(
                        "Cache-transfer destinations disagree on the prompt window"
                    )
            pending.append((sender, token, spec_candidate_ids, block_manifest))

        for sender, token, spec_candidate_ids, block_manifest in pending:
            sender.send(
                True,
                bootstrap_token=token,
                spec_candidate_ids=spec_candidate_ids,
                block_manifest=block_manifest,
            )

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
        """Send this completed prompt's KV to the node that will decode it.

        The P-role half of the plan's remote streams, submitted on the
        forward thread like any forward; completion arrives as a transfer
        event, not from here.
        """
        if not isinstance(op, Forward.Batch):
            raise TypeError(f"Expected Batch, got {type(op).__name__}.")
        self._cache_decode(op)

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
