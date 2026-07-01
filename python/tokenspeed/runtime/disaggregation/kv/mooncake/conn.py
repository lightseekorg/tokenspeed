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

from tokenspeed.runtime.disaggregation.base.bootstrap import DisaggBootstrapServer
from tokenspeed.runtime.disaggregation.base.manager import DisaggManagerBase
from tokenspeed.runtime.disaggregation.kv.mooncake.entities import ManagerArgs
from tokenspeed.runtime.disaggregation.kv.types import KVArgs
from tokenspeed.runtime.disaggregation.mooncake_transfer_engine import (
    MooncakeTransferEngine,
)
from tokenspeed.runtime.disaggregation.utils import DisaggregationMode
from tokenspeed.runtime.metrics.collector import KVTransferMetrics
from tokenspeed.runtime.utils.network import get_local_ip_by_remote


class MooncakeKVManagerBase(DisaggManagerBase):
    """KV (prefill->decode) manager: the shared engine/socket/status FSM plus the
    KV-specific args, MLA flags, dp-attention wiring, and metrics."""

    def __init__(
        self,
        args: ManagerArgs,
        kv_args: KVArgs,
        disaggregation_mode: DisaggregationMode,
    ):
        self.args = args
        self.kv_args = kv_args
        self.attn_tp_rank = args.attn_tp_rank
        self.src_mode = "ON" if bool(args.enable_mla_l1_5_cache) else "OFF"
        self.is_mla_backend = args.is_mla_backend
        self.draft_is_mla_backend = args.draft_is_mla_backend
        self.disaggregation_mode = disaggregation_mode
        self.bootstrap_port = args.bootstrap_port
        self.dist_init_addr = args.dist_init_addr
        self.world_size = args.world_size
        self.dp_size = args.dp_size
        self.attn_dp_rank = args.attn_dp_rank
        self.enable_dp_attention = args.enable_dp_attention
        if not args.enable_dp_attention and args.dp_size != 1:
            raise ValueError(
                "If dp_attention is not enabled, dp size must be 1 in disaggregation mode."
            )

        if hasattr(args, "enable_metrics") and args.enable_metrics:
            labels = {
                "model_name": args.served_model_name,
                "app_key": args.app_key,
            }
            self.kv_transfer_metrics = KVTransferMetrics(labels, args.metrics_reporters)
        else:
            self.kv_transfer_metrics = None

        # Build the Mooncake data-plane engine here (the vendor binding stays in
        # the mooncake package, not in the neutral base) and inject it. self.kv_args
        # is set above so register_buffer_to_engine (called by the base) sees the
        # KV buffers.
        engine = MooncakeTransferEngine(
            hostname=get_local_ip_by_remote(),
            gpu_id=kv_args.gpu_id,
            ib_device=kv_args.ib_device,
        )
        super().__init__(engine=engine)

    def register_buffer_to_engine(self):
        for kv_data_ptr, kv_data_len in zip(
            self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
        ):
            self.engine.register(kv_data_ptr, kv_data_len)
        for state_data_ptr, state_data_len in zip(
            self.kv_args.state_data_ptrs, self.kv_args.state_data_lens
        ):
            self.engine.register(state_data_ptr, state_data_len)


class MooncakeKVBootstrapServer(DisaggBootstrapServer):
    """KV bootstrap rendezvous: the shared server plus the prefill-side MLA /
    kv-page length fields the decode side needs in the parallel-info sync."""

    def __init__(self, port: int):
        # Set before super() -- super() starts the server thread, after which a
        # register PUT can call _ingest_put_extra and read these.
        self.enable_mla_l1_5_cache = False
        self.prefill_kv_item_lens = []
        self.prefill_kv_unit_lens = []
        self.prefill_state_item_lens = []
        self.prefill_state_unit_lens = []
        super().__init__(port)

    def _ingest_put_extra(self, data: dict) -> None:
        self.enable_mla_l1_5_cache = bool(data["enable_mla_l1_5_cache"])
        self.prefill_kv_item_lens = data.get("kv_item_lens", self.prefill_kv_item_lens)
        self.prefill_kv_unit_lens = data.get("kv_unit_lens", self.prefill_kv_unit_lens)
        self.prefill_state_item_lens = data.get(
            "state_item_lens", self.prefill_state_item_lens
        )
        self.prefill_state_unit_lens = data.get(
            "state_unit_lens", self.prefill_state_unit_lens
        )

    def _extra_parallel_info(self) -> dict:
        return {
            "enable_mla_l1_5_cache": self.enable_mla_l1_5_cache,
            "kv_item_lens": self.prefill_kv_item_lens,
            "kv_unit_lens": self.prefill_kv_unit_lens,
            "state_item_lens": self.prefill_state_item_lens,
            "state_unit_lens": self.prefill_state_unit_lens,
        }
