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


from tokenspeed_scheduler import PD, Forward

from tokenspeed.runtime.pd.base.bootstrap import BootstrapInfo
from tokenspeed.runtime.pd.base.status import TransferPoll
from tokenspeed.runtime.pd.cache_protocol import build_cache_block_manifest
from tokenspeed.runtime.pd.mooncake.decode import MooncakeKVManagerDecode
from tokenspeed.runtime.pd.mooncake.receiver import MooncakeKVReceiver
from tokenspeed.runtime.pd.utils import poll_and_all_reduce
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


class DisaggDecodeExecutor:
    def __init__(self, args, kv_args, gloo_group):
        self.cache_layout = kv_args.cache_layout
        self.receivers: dict[str, MooncakeKVReceiver] = {}
        self.kv_manager = MooncakeKVManagerDecode(args, kv_args)
        self.gloo_group = gloo_group
        self._local_states = {}
        self._request_pool_indices: dict[str, int] = {}
        self._remote_cache_slots: dict[str, int] = {}
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
                "CachePD Decode admission does not support mixed batches"
            )
        for index, request_id in enumerate(op.request_ids[:num_extends]):
            receiver = self.receivers.get(request_id)
            if receiver is None:
                continue
            prefix_len = int(op.extend_prefix_lens[index])
            block_manifest = build_cache_block_manifest(
                op,
                layout=self.cache_layout,
                request_row=index,
                prefix_len=prefix_len,
                prompt_len=int(op.prefill_lengths[index]),
            )
            pending.append(
                (
                    request_id,
                    receiver,
                    op.request_pool_indices[index],
                    block_manifest,
                )
            )

        # Validate every row before publishing any destination manifest. A later
        # invalid row must not leave an earlier Prefill sender waiting forever.
        for request_id, receiver, request_pool_index, block_manifest in pending:
            self._request_pool_indices[request_id] = request_pool_index
            receiver.prefill(block_manifest=block_manifest)

    def register(
        self,
        request_id: str,
        bootstrap_info: BootstrapInfo,
    ):
        self._local_states[request_id] = TransferPoll.Bootstrapping
        self._bootstrap(request_id, bootstrap_info)

    def execute(self, op):
        """Pull this admitted prompt's KV from the prefill node.

        The D-role half of the plan's remote streams, submitted on the
        forward thread like any forward; completion arrives as a transfer
        event, not from here.
        """
        if not isinstance(op, Forward.Batch):
            raise TypeError(f"Expected Batch, got {type(op).__name__}.")
        self._cache_prefill(op)

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
                request_pool_index = self._request_pool_indices[req_id]
                self._remote_cache_slots[req_id] = request_pool_index
                if spec_candidate_ids is not None:
                    self._remote_spec_candidate_ids[req_id] = (
                        request_pool_index,
                        spec_candidate_ids,
                    )
                logger.debug(
                    "[decode][generate_events] rid=%s -> RemotePrefillDoneEvent bootstrap_token=%s",
                    req_id,
                    bootstrap_token,
                )
                # The C++ FSM extends the token into the TokenContainer as it
                # applies this event (RemotePrefilling -> PrefillDone).
                events.append(PD.RemotePrefillDoneEvent(req_id, bootstrap_token))
                to_remove.append(req_id)
            else:
                pass
        for req_id in to_remove:
            # Best-effort cleanup mirroring prefill side; request_id is stable
            # so without explicit pop these dicts would grow unbounded across
            # failed requests. The remote-cache/spec handoff dictionaries must
            # stay alive until event_loop consumes the event after this returns.
            receiver = self.receivers.pop(req_id, None)
            if receiver is not None:
                receiver.clear()
            self._request_pool_indices.pop(req_id, None)
            self._local_states.pop(req_id, None)

        return events

    def pop_remote_spec_candidate_ids(self, request_id: str):
        return self._remote_spec_candidate_ids.pop(request_id, None)

    def pop_remote_cache_slot(self, request_id: str) -> int | None:
        return self._remote_cache_slots.pop(request_id, None)
