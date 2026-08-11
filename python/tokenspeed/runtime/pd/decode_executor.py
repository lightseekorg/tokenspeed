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


import numpy as np
import torch
from tokenspeed_scheduler import PD, Forward

from tokenspeed.runtime.execution.block_table import (
    select_block_table,
    unpadded_block_table_row,
)
from tokenspeed.runtime.pd.base.bootstrap import BootstrapInfo
from tokenspeed.runtime.pd.base.status import TransferPoll
from tokenspeed.runtime.pd.cache_protocol import (
    build_cache_page_manifest,
    cache_manifest_page_ids,
)
from tokenspeed.runtime.pd.mooncake.decode import MooncakeKVManagerDecode
from tokenspeed.runtime.pd.mooncake.receiver import MooncakeKVReceiver
from tokenspeed.runtime.pd.utils import (
    TransferBackend,
    poll_and_all_reduce,
)
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.dispatch import TypeBasedDispatcher

logger = get_colorful_logger(__name__)


class DisaggDecodeExecutor:
    def __init__(
        self, backend: TransferBackend, args, kv_args, gloo_group, page_size: int
    ):
        self.transfer_backend = backend
        self.bootstrap_port = args.bootstrap_port
        self.page_size = page_size
        self._dispatcher = TypeBasedDispatcher(
            [
                (Forward.Batch, self._prefill),
            ]
        )
        self.uses_cache_contract = kv_args.cache_layout is not None
        self.cache_layout = kv_args.cache_layout
        self.receivers: dict[str, MooncakeKVReceiver] = {}
        self.kv_manager = MooncakeKVManagerDecode(args, kv_args)
        self.gloo_group = gloo_group
        self._local_states = {}
        self._request_pool_indices: dict[str, int] = {}
        self._remote_spec_candidate_ids: dict[str, tuple[int, list[int]]] = {}

    def _bootstrap(self, request_id, info):
        self.receivers[request_id] = MooncakeKVReceiver(
            mgr=self.kv_manager,
            bootstrap_addr=f"{info.bootstrap_host}:{info.bootstrap_port}",
            bootstrap_room=info.bootstrap_room,
        )

    def _cache_prefill(self, op) -> None:
        pending = []
        num_extends = op.num_extends()
        if num_extends != len(op.request_ids):
            raise RuntimeError(
                "Paged cache decode destination admission does not support mixed batches"
            )
        for index, request_id in enumerate(op.request_ids):
            receiver = self.receivers.get(request_id)
            if receiver is None:
                continue
            prefix_len = int(op.extend_prefix_lens[index])
            manifest = build_cache_page_manifest(
                op,
                layout=self.cache_layout,
                request_row=index,
                prefix_len=prefix_len,
                prompt_len=int(op.prefill_lengths[index]),
            )
            page_ids = np.asarray(
                cache_manifest_page_ids(
                    manifest,
                    layout=self.cache_layout,
                    peer="destination",
                ),
                dtype=np.int64,
            )
            pending.append(
                (
                    request_id,
                    receiver,
                    op.request_pool_indices[index],
                    prefix_len,
                    manifest,
                    page_ids,
                )
            )

        # Validate every row before publishing any destination vector. A later
        # invalid row must not leave an earlier Prefill sender waiting forever.
        for request_id, receiver, aux_index, prefix_len, manifest, page_ids in pending:
            self._request_pool_indices[request_id] = aux_index
            receiver.prefill(
                page_ids,
                aux_index,
                prefix_len,
                None,
                None,
                page_manifest=manifest,
            )

    def _prefill(self, op):
        if self.uses_cache_contract:
            self._cache_prefill(op)
            return
        table = select_block_table(op)
        page_rows = [
            unpadded_block_table_row(table, row_index)
            for row_index in range(len(table))
        ]
        logger.debug(
            "[decode][_prefill] op: request_ids=%s page_rows=%s "
            "request_pool_indices=%s extend_prefix_lens=%s",
            list(op.request_ids),
            page_rows,
            list(op.request_pool_indices),
            list(op.extend_prefix_lens),
        )

        for i, request_id in enumerate(op.request_ids):
            if request_id not in self.receivers:
                # Request failed and its receiver was cleaned up in generate_events;
                # the scheduler may still dispatch its forward op one last time.
                continue
            extend_prefix_len = op.extend_prefix_lens[i]
            # Exclude pages held only for reserved decode input token(s); P has
            # source KV only through the logical end of the prompt.
            prompt_end_page = (
                op.prefill_lengths[i] + self.page_size - 1
            ) // self.page_size
            kv_indices = np.array(
                page_rows[i][extend_prefix_len // self.page_size : prompt_end_page],
                dtype=np.int64,
            )
            aux_index = op.request_pool_indices[i]
            self._request_pool_indices[request_id] = aux_index
            self.receivers[request_id].prefill(
                kv_indices,
                aux_index,
                extend_prefix_len,
                None,
            )

    def register(
        self,
        request_id: str,
        bootstrap_info: BootstrapInfo,
    ):
        self._local_states[request_id] = TransferPoll.Bootstrapping
        self._bootstrap(request_id, bootstrap_info)

    def execute(self, op):
        if not isinstance(op, Forward.Batch):
            raise TypeError(f"Expected Batch, got {type(op).__name__}.")
        self._dispatcher(op)

    def generate_events(self):
        if not self.receivers:
            return []
        polls = poll_and_all_reduce(self.receivers.values(), self.gloo_group)

        events = []
        to_remove = []
        for req_id, poll in zip(list(self.receivers.keys()), polls):
            if (
                self._local_states[req_id] == TransferPoll.Bootstrapping
                and poll == TransferPoll.Bootstrapped
            ):
                logger.debug(
                    "[decode][generate_events] rid=%s -> BootstrappedEvent", req_id
                )
                events.append(PD.BootstrappedEvent(req_id))
                self._local_states[req_id] = TransferPoll.Bootstrapped
            elif poll == TransferPoll.Failed:
                logger.warning(
                    "[decode][generate_events] rid=%s -> FailedEvent", req_id
                )
                events.append(PD.FailedEvent(req_id))
                # Drop the failed receiver so it is not polled again. Without this
                # a single failed request keeps re-emitting FailedEvent every loop
                # (poll stays Failed), wedging the whole conn-1 scheduler.
                to_remove.append(req_id)
            elif (
                self._local_states[req_id] == TransferPoll.Bootstrapped
                and poll == TransferPoll.Success
            ):
                # Read bootstrap_token from the ZMQ-delivered table in kv_manager.
                # The decode_thread stored it there when it received the Success status
                # message from the prefill side.  bootstrap_room == bootstrap_info.bootstrap_room,
                # which is the key used in MooncakeKVReceiver.
                self._local_states[req_id] = TransferPoll.Success
                bootstrap_room = self.receivers[req_id].bootstrap_room
                bootstrap_token, spec_candidate_ids = (
                    self.kv_manager.pop_prefill_metadata(bootstrap_room)
                )
                receiver = self.receivers[req_id]
                if (
                    spec_candidate_ids is not None
                    and req_id in self._request_pool_indices
                    and getattr(
                        receiver,
                        "supports_remote_spec_candidates",
                        True,
                    )
                ):
                    self._remote_spec_candidate_ids[req_id] = (
                        self._request_pool_indices[req_id],
                        spec_candidate_ids,
                    )
                logger.debug(
                    "[decode][generate_events] rid=%s -> RemotePrefillDoneEvent bootstrap_token=%s",
                    req_id,
                    bootstrap_token,
                )
                # Use RemotePrefillDoneEvent to carry the bootstrap_token to event_loop;
                # the C++ FSM will extend it into the TokenContainer via
                # fsm::RemotePrefillDoneEvent::operator()(Prefilling&&).
                event = PD.RemotePrefillDoneEvent(
                    req_id, bootstrap_token if bootstrap_token != -1 else -1
                )
                events.append(event)
                to_remove.append(req_id)
            else:
                pass
        for req_id in to_remove:
            # Best-effort cleanup mirroring prefill side; request_id is stable
            # so without explicit pop these dicts would grow unbounded across
            # failed requests. NOTE: _remote_spec_candidate_ids must NOT be
            # popped here — its consumer pop_remote_spec_candidate_ids runs
            # later inside event_loop._process_kv_transfer_events, after we return.
            # That dict is small (one tuple per Success request, between
            # generate_events emitting RemotePrefillDoneEvent and event_loop
            # consuming it) and is naturally drained by the pop path; an
            # eager pop here drops the spec candidates on the floor and the
            # next decode forward reads uninitialized future_input_map tail,
            # causing CUDA illegal memory access on embedding lookup.
            self.receivers.pop(req_id, None)
            self._request_pool_indices.pop(req_id, None)
            self._local_states.pop(req_id, None)

        return events

    def pop_remote_spec_candidate_ids(self, request_id: str):
        return self._remote_spec_candidate_ids.pop(request_id, None)

    def reset_valid_cache_length(
        self, forward_op, runtime_states, execution_stream, device
    ):
        num_extends = forward_op.num_extends()
        if num_extends <= 0:
            return

        # A decode destination never executes the prompt locally, so the
        # model executor cannot infer this state from a forward pass.  Seed
        # the runtime row with the complete remotely-computed prompt length
        # before the first local decode.  This is required for both cache
        # layouts: Paged cache additionally uses the resulting sequence length to
        # select the transferred recurrent-state snapshot page.
        extend_request_pool_indices = torch.tensor(
            forward_op.request_pool_indices[:num_extends],
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        ).to(device, non_blocking=True)
        extend_prefix_lens = torch.tensor(
            forward_op.prefill_lengths[:num_extends],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        ).to(device, non_blocking=True)
        # HostTodevice segment ends

        execution_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(execution_stream):
            if num_extends > 0:
                runtime_states.reset_states(
                    extend_request_pool_indices, extend_prefix_lens
                )
