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

import time
from typing import Optional

import numpy as np
import numpy.typing as npt

from tokenspeed.runtime.pd.base.conn import (
    KVPoll,
)
from tokenspeed.runtime.pd.mooncake.entities import KVTransferError
from tokenspeed.runtime.pd.utils import PageTransferMetadata
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


class MooncakeKVSender:
    def __init__(
        self,
        mgr,  # MooncakeKVManagerPrefill
        bootstrap_addr: str,
        bootstrap_room: int,
    ):
        self.kv_mgr = mgr
        self.bootstrap_server_url = bootstrap_addr
        self.bootstrap_room = bootstrap_room
        self.kv_mgr.update_status(bootstrap_room, KVPoll.Bootstrapping)
        logger.info(
            "[MooncakeKVSender.__init__] bootstrap_room=%s bootstrap_addr=%s status=Bootstrapping",
            bootstrap_room,
            bootstrap_addr,
        )

        # inner state
        self.init_time = None
        self.conclude_state = None
        self.curr_idx = 0
        self._layerwise_transfer_started = False

    def has_layerwise_transfer(self) -> bool:
        return self._layerwise_transfer_started

    def send(
        self,
        kv_indices: npt.NDArray[np.int64],
        aux_index,
        is_last,
        mla_l1_5_args: Optional[PageTransferMetadata] = None,
        bootstrap_token: int = -1,
        mamba_indices: Optional[npt.NDArray[np.int64]] = None,
    ):
        """
        Send the kv cache at the given kv indices to the decoder server
        mla_l1_5_args: optional (page_transfer_mask, page_local_indices)
            page_transfer_mask: boolean mask to select decode pages that will receive data from this prefill rank
            page_local_indices: remapped local page indices that this prefill rank will send
        bootstrap_token: first output token produced by prefill (shipped via ZMQ status msg).
        """
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)

        logger.info(
            "[MooncakeKVSender.send] bootstrap_room=%s kv_indices_len=%d is_last=%s curr_idx=%d bootstrap_token=%s",
            self.bootstrap_room,
            len(kv_indices),
            is_last,
            self.curr_idx,
            bootstrap_token,
        )

        if not is_last:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                False,
                mla_l1_5_args=mla_l1_5_args,
                mamba_indices=mamba_indices,
            )
        else:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                True,
                aux_index=aux_index,
                mla_l1_5_args=mla_l1_5_args,
                bootstrap_token=bootstrap_token,
                mamba_indices=mamba_indices,
            )

    def send_layerwise(
        self,
        kv_indices: npt.NDArray[np.int64],
        index_slice: slice,
        aux_index,
        is_last,
        begin_cache_step: int,
        layerwise_interval: int,
        mla_l1_5_args: Optional[PageTransferMetadata] = None,
        bootstrap_token: int = -1,
        wait_for_bootstrap_token: bool = False,
        mamba_indices: Optional[npt.NDArray[np.int64]] = None,
    ):
        self._layerwise_transfer_started = True
        self.curr_idx = max(self.curr_idx, index_slice.stop)

        if len(kv_indices) == 0 and not is_last:
            return

        logger.info(
            "[MooncakeKVSender.send_layerwise] bootstrap_room=%s kv_indices_len=%d "
            "slice=(%s,%s) is_last=%s begin_cache_step=%s interval=%s",
            self.bootstrap_room,
            len(kv_indices),
            index_slice.start,
            index_slice.stop,
            is_last,
            begin_cache_step,
            layerwise_interval,
        )
        self.kv_mgr.add_transfer_request(
            self.bootstrap_room,
            kv_indices,
            index_slice,
            is_last,
            aux_index=aux_index if is_last else None,
            mla_l1_5_args=mla_l1_5_args,
            bootstrap_token=bootstrap_token,
            begin_cache_step=begin_cache_step,
            layerwise_interval=layerwise_interval,
            wait_for_bootstrap_token=wait_for_bootstrap_token,
            mamba_indices=mamba_indices,
        )

    def poll(self) -> KVPoll:
        if self.conclude_state is None:
            status = self.kv_mgr.check_status(self.bootstrap_room)
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status
            elif status == KVPoll.Bootstrapping:
                if self.init_time is not None:
                    now = time.time()
                    elapsed = now - self.init_time
                    if elapsed >= self.kv_mgr.bootstrap_time_out:
                        logger.warning_once(
                            "Some requests timed out when bootstrapping, "
                            "which means prefill instances fail to receive the KV indices from the decode instance of this request. "
                            "If a greater mean TTFT is acceptable, you can 'export TOKENSPEED_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600' (10 minutes) to relax the timeout condition. "
                        )
                        self.kv_mgr.record_failure(
                            self.bootstrap_room,
                            f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.Bootstrapping",
                        )
                        self.conclude_state = KVPoll.Failed
                        return KVPoll.Failed

            return status
        else:
            return self.conclude_state

    def clear(self) -> None:
        if self.bootstrap_room in self.kv_mgr.request_status:
            self.kv_mgr.request_status.pop(self.bootstrap_room)

    def failure_exception(self):
        # Explicitly set the status to failure since this request has failed in another rank
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "Failed due to an unknown reason from another rank"
            )
        raise KVTransferError(
            self.bootstrap_room, failure_reason, self.bootstrap_server_url
        )
