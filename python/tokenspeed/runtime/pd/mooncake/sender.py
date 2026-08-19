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

from tokenspeed.runtime.pd.base.status import TransferPoll
from tokenspeed.runtime.pd.cache_protocol import (
    CachePDBlockManifest,
    CachePDLayerwiseBlockSelection,
)
from tokenspeed.runtime.pd.mooncake.entities import KVTransferError
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
        self.kv_mgr.begin_room(bootstrap_room)
        logger.info(
            "[MooncakeKVSender.__init__] bootstrap_room=%s bootstrap_addr=%s status=Bootstrapping",
            bootstrap_room,
            bootstrap_addr,
        )

        # inner state
        self.init_time = time.time()
        self.conclude_state = None
        self._layerwise_chunk_submitted = False
        self._layerwise_final_chunk_submitted = False

    def layerwise_chunk_submitted(self) -> bool:
        return self._layerwise_chunk_submitted

    def layerwise_final_chunk_submitted(self) -> bool:
        return self._layerwise_final_chunk_submitted

    def send(
        self,
        is_last,
        bootstrap_token: int = -1,
        spec_candidate_ids: list[int] | None = None,
        block_manifest: CachePDBlockManifest | None = None,
    ):
        """Submit one final, manifest-backed CachePD transfer."""
        if not is_last:
            raise ValueError("CachePD full-manifest transfer must be final")

        logger.info(
            "[MooncakeKVSender.send] bootstrap_room=%s is_last=%s bootstrap_token=%s",
            self.bootstrap_room,
            is_last,
            bootstrap_token,
        )

        self.kv_mgr.add_transfer_request(
            self.bootstrap_room,
            True,
            bootstrap_token=bootstrap_token,
            spec_candidate_ids=spec_candidate_ids,
            block_manifest=block_manifest,
        )

    def send_layerwise(
        self,
        is_last,
        begin_cache_step: int,
        layerwise_interval: int,
        bootstrap_token: int = -1,
        wait_for_bootstrap_token: bool = False,
        spec_candidate_ids: list[int] | None = None,
        cache_block_selection: CachePDLayerwiseBlockSelection | None = None,
    ):
        self._layerwise_chunk_submitted = True
        self._layerwise_final_chunk_submitted = (
            self._layerwise_final_chunk_submitted or is_last
        )

        logger.info(
            "[MooncakeKVSender.send_layerwise] bootstrap_room=%s "
            "is_last=%s begin_cache_step=%s interval=%s",
            self.bootstrap_room,
            is_last,
            begin_cache_step,
            layerwise_interval,
        )
        self.kv_mgr.add_transfer_request(
            self.bootstrap_room,
            is_last,
            bootstrap_token=bootstrap_token,
            begin_cache_step=begin_cache_step,
            layerwise_interval=layerwise_interval,
            wait_for_bootstrap_token=wait_for_bootstrap_token,
            spec_candidate_ids=spec_candidate_ids,
            cache_block_selection=cache_block_selection,
        )

    def poll(self) -> TransferPoll:
        if self.conclude_state is None:
            status = self.kv_mgr.check_status(self.bootstrap_room)
            if status in (TransferPoll.Success, TransferPoll.Failed):
                self.conclude_state = status
            elif status == TransferPoll.Bootstrapping:
                if self.init_time is not None:
                    now = time.time()
                    elapsed = now - self.init_time
                    if elapsed >= self.kv_mgr.bootstrap_time_out:
                        logger.warning_once(
                            "Some requests timed out when bootstrapping, "
                            "which means prefill instances fail to receive the cache manifest from the decode instance of this request. "
                            "If a greater mean TTFT is acceptable, you can 'export TOKENSPEED_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600' (10 minutes) to relax the timeout condition. "
                        )
                        self.kv_mgr.record_failure(
                            self.bootstrap_room,
                            f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in TransferPoll.Bootstrapping",
                        )
                        self.conclude_state = TransferPoll.Failed
                        return TransferPoll.Failed

            return status
        else:
            return self.conclude_state

    def clear(self) -> None:
        self.kv_mgr.request_status.pop(self.bootstrap_room, None)
        with self.kv_mgr.failure_lock:
            self.kv_mgr.failure_records.pop(self.bootstrap_room, None)

    def failure_exception(self):
        # Explicitly set the status to failure since this request has failed in another rank
        if self.conclude_state is None:
            self.conclude_state = TransferPoll.Failed

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.get(
                self.bootstrap_room, "Failed due to an unknown reason from another rank"
            )
        self.clear()
        raise KVTransferError(
            self.bootstrap_room, failure_reason, self.bootstrap_server_url
        )
