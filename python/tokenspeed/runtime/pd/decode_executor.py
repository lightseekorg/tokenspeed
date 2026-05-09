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

from typing import Dict

import numpy as np
import torch
from tokenspeed_scheduler import PD, Forward

from tokenspeed.runtime.pd.base import BootstrapInfo, KVPoll
from tokenspeed.runtime.pd.mooncake import (
    MooncakeKVManagerDecode,
    MooncakeKVReceiver,
)
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
                (Forward.FlatForwardOp, self._prefill),
            ]
        )
        self.receivers: Dict[int, MooncakeKVReceiver] = {}
        self.kv_manager = MooncakeKVManagerDecode(args, kv_args)
        self.gloo_group = gloo_group
        self._local_states = {}

    def _bootstrap(self, request_id, info):
        self.receivers[request_id] = MooncakeKVReceiver(
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

    def _prefill(self, op):
        logger.debug(
            "[decode][_prefill] op: request_ids=%s occupied_pages=%s "
            "begins=%s sizes=%s request_pool_indices=%s extend_prefix_lens=%s",
            list(op.request_ids),
            [list(p) for p in op.occupied_pages],
            list(op.begins),
            list(op.sizes),
            list(op.request_pool_indices),
            list(op.extend_prefix_lens),
        )

        for i, request_id in enumerate(op.request_ids):
            extend_prefix_len = op.extend_prefix_lens[i]
            kv_indices = np.array(
                op.occupied_pages[i][extend_prefix_len // self.page_size :],
                dtype=np.int64,
            )
            aux_index = op.request_pool_indices[i]
            mamba_indices = self._mamba_indices(op, i)
            self.receivers[request_id].prefill(
                kv_indices,
                aux_index,
                extend_prefix_len,
                None,  # mla_l1_5_args
                mamba_indices,
            )

    def register(self, request_id: str, bootstrap_info: BootstrapInfo):
        self._local_states[request_id] = KVPoll.Bootstrapping
        self._bootstrap(request_id, bootstrap_info)

    def execute(self, op):
        assert isinstance(op, Forward.FlatForwardOp)
        self._dispatcher(op)

    def generate_events(self):
        if not self.receivers:
            return []
        polls = poll_and_all_reduce(self.receivers.values(), self.gloo_group)

        events = []
        to_remove = []
        for req_id, poll in zip(list(self.receivers.keys()), polls):
            if (
                self._local_states[req_id] == KVPoll.Bootstrapping
                and poll == KVPoll.Bootstrapped
            ):
                logger.debug(
                    "[decode][generate_events] rid=%s -> BootstrappedEvent", req_id
                )
                events.append(PD.BootstrappedEvent(req_id))
                self._local_states[req_id] = KVPoll.Bootstrapped
            elif poll == KVPoll.Failed:
                logger.warning(
                    "[decode][generate_events] rid=%s -> FailedEvent", req_id
                )
                events.append(PD.FailedEvent(req_id))
            elif (
                self._local_states[req_id] == KVPoll.Bootstrapped
                and poll == KVPoll.Success
            ):
                self._local_states[req_id] = KVPoll.Success
                # Read bootstrap_token from the ZMQ-delivered table in kv_manager.
                # The decode_thread stored it there when it received the Success status
                # message from the prefill side.  bootstrap_room == bootstrap_info.bootstrap_room,
                # which is the key used in MooncakeKVReceiver.
                bootstrap_room = self.receivers[req_id].bootstrap_room
                bootstrap_token = self.kv_manager.pop_bootstrap_token(bootstrap_room)
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
            del self.receivers[req_id]

        return events

    def reset_valid_cache_length(
        self, forward_op, runtime_states, execution_stream, device
    ):
        num_extends = forward_op.num_extends()
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
