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
from tokenspeed.runtime.utils.dispatch import TypeBasedDispatcher

logger = get_colorful_logger(__name__)

from tokenspeed_scheduler import PD, Forward


class DisaggPrefillExecutor:
    def __init__(self, args, kv_args, gloo_group):
        self._dispatcher = TypeBasedDispatcher(
            [
                (Forward.Batch, self._cache_decode),
            ]
        )
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
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
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

    def setup_layerwise_transfer(self, wiring, gpu_id: int, interval: int) -> None:
        """Stream KV out layer-by-layer during the prefill forward.

        The shared step counter is ticked by the attention backends as each
        layer's KV lands and read by this executor's sender. Installing it is
        backend surgery, so the device wiring does that and hands back the
        counter — this side never touches a backend.

        Args:
            wiring: The engine's startup ``DeviceWiring``.
            gpu_id: Device index the counter is created against.
            interval: Layer interval between sends; ``<= 0`` disables
                layerwise transfer (this call becomes a no-op).
        """
        if interval <= 0:
            return
        self.register_layerwise_step_counter(
            wiring.install_pd_step_counter(gpu_id), interval
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
        self._request_token.pop(req_id, None)
        self._request_spec_candidate_ids.pop(req_id, None)
        self._layerwise_token_published.discard(req_id)

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
                raise RuntimeError("Paged cache destination metadata is unavailable")
            destinations = tuple(
                info for info in transfer_infos.values() if not info.is_dummy
            )
            # Idle representative ranks use the normal post-forward dummy
            # completion path; they have no cache fields to overlap.
            if not destinations:
                continue
            first = destinations[0]
            if first.block_manifest is None:
                raise RuntimeError("Paged cache destination metadata is unavailable")
            prefix_len = first.block_manifest.prefix_len
            prompt_len = first.block_manifest.prompt_len
            chunk_begin = int(op.extend_prefix_lens[index])
            chunk_end = chunk_begin + int(op.input_lengths[index])
            if chunk_end > prompt_len:
                raise ValueError(
                    "Paged cache Prefill chunk extends past Decode's prompt manifest"
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
                        "Paged cache destination metadata is unavailable"
                    )
                if (
                    destination.block_manifest.prefix_len != prefix_len
                    or destination.block_manifest.prompt_len != prompt_len
                ):
                    raise ValueError(
                        "Paged cache destinations disagree on the prompt window"
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
        if self._layerwise_enabled:
            # Layerwise already streamed the KV during prepare_prefill; only the
            # bootstrap token still needs publishing on the last chunk.
            for request_id in op.request_ids:
                sender = self.senders.get(request_id)
                if sender is None:
                    continue
                if not sender.layerwise_final_chunk_submitted():
                    transfer_infos = self.kv_manager.transfer_infos.get(
                        sender.bootstrap_room, {}
                    )
                    if transfer_infos and all(
                        info.is_dummy for info in transfer_infos.values()
                    ):
                        # Idle representative ranks have no cache fields to
                        # stream, but still participate in the Prefill TP status
                        # collective. Submit their final no-op only after the
                        # Prefill forward has completed.
                        token = self._request_token.get(request_id)
                        spec_candidate_ids = self._request_spec_candidate_ids.get(
                            request_id
                        )
                        if (
                            isinstance(token, bool)
                            or not isinstance(token, int)
                            or token < 0
                        ):
                            raise RuntimeError(
                                "Paged cache bootstrap token is unavailable"
                            )
                        sender.send(
                            True,
                            bootstrap_token=token,
                            spec_candidate_ids=spec_candidate_ids,
                            block_manifest=None,
                        )
                        self._request_token.pop(request_id, None)
                        self._request_spec_candidate_ids.pop(request_id, None)
                        self._layerwise_token_published.discard(request_id)
                    continue
                token = self._request_token.get(request_id)
                spec_candidate_ids = self._request_spec_candidate_ids.get(request_id)
                if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                    raise RuntimeError("Paged cache bootstrap token is unavailable")
                if request_id not in self._layerwise_token_published:
                    self.kv_manager.set_prefill_metadata(
                        sender.bootstrap_room,
                        token,
                        spec_candidate_ids,
                    )
                self._request_token.pop(request_id, None)
                self._request_spec_candidate_ids.pop(request_id, None)
                self._layerwise_token_published.discard(request_id)
            return
        pending = []
        for index, request_id in enumerate(op.request_ids):
            sender = self.senders.get(request_id)
            if sender is None:
                continue
            if sender.layerwise_final_chunk_submitted():
                token = self._request_token.pop(request_id, None)
                spec_candidate_ids = self._request_spec_candidate_ids.pop(
                    request_id, None
                )
                if request_id not in self._layerwise_token_published:
                    if (
                        isinstance(token, bool)
                        or not isinstance(token, int)
                        or token < 0
                    ):
                        raise RuntimeError("Paged cache bootstrap token is unavailable")
                    self.kv_manager.set_prefill_metadata(
                        sender.bootstrap_room, token, spec_candidate_ids
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
                spec_candidate_ids = self._request_spec_candidate_ids.get(request_id)
                if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                    raise RuntimeError("Paged cache bootstrap token is unavailable")
                pending.append(
                    (
                        request_id,
                        sender,
                        token,
                        spec_candidate_ids,
                        None,
                    )
                )
                continue
            destination = destinations[0]
            if destination.block_manifest is None:
                raise RuntimeError("Paged cache destination metadata is unavailable")
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
                        "Paged cache destination metadata is unavailable"
                    )
                if (
                    destination.block_manifest.prefix_len != block_manifest.prefix_len
                    or destination.block_manifest.prompt_len
                    != block_manifest.prompt_len
                ):
                    raise ValueError(
                        "Paged cache destinations disagree on the prompt window"
                    )
            token = self._request_token.get(request_id)
            spec_candidate_ids = self._request_spec_candidate_ids.get(request_id)
            if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                raise RuntimeError("Paged cache bootstrap token is unavailable")
            pending.append(
                (
                    request_id,
                    sender,
                    token,
                    spec_candidate_ids,
                    block_manifest,
                )
            )

        for (
            request_id,
            sender,
            token,
            spec_candidate_ids,
            block_manifest,
        ) in pending:
            sender.send(
                True,
                bootstrap_token=token,
                spec_candidate_ids=spec_candidate_ids,
                block_manifest=block_manifest,
            )
            self._request_token.pop(request_id, None)
            self._request_spec_candidate_ids.pop(request_id, None)

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
