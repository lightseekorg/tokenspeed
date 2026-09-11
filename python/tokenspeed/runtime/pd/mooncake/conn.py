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

from tokenspeed.runtime.metrics.collector import KVTransferMetrics
from tokenspeed.runtime.pd.base.bootstrap import DisaggBootstrapServerBase
from tokenspeed.runtime.pd.base.manager import DisaggManagerBase
from tokenspeed.runtime.pd.base.mooncake_engine import (
    MooncakeTransferEngine,
)
from tokenspeed.runtime.pd.cache_protocol import (
    CacheTransferContract,
)
from tokenspeed.runtime.pd.mooncake.entities import KVArgs, KVManagerArgs
from tokenspeed.runtime.pd.utils import DisaggregationMode
from tokenspeed.runtime.utils.network import get_local_ip_by_remote


class MooncakeKVManagerBase(DisaggManagerBase):
    """CachePD manager: shared engine/socket/status state and typed slabs."""

    def __init__(
        self,
        args: KVManagerArgs,
        kv_args: KVArgs,
        disaggregation_mode: DisaggregationMode,
    ):
        self.kv_args = kv_args
        self.topology = args.topology
        self.topology.require_cache_pd_supported()
        self.disaggregation_mode = disaggregation_mode
        self.bootstrap_port = args.bootstrap_port
        self.dist_init_addr = args.dist_init_addr
        if not args.enable_dp_attention and self.topology.dp_size != 1:
            raise ValueError(
                "If dp_attention is not enabled, dp size must be 1 in disaggregation mode."
            )

        if args.enable_metrics:
            labels = {
                "model_name": args.served_model_name,
                "app_key": args.app_key,
            }
            self.kv_transfer_metrics = KVTransferMetrics(labels, args.metrics_reporters)
        else:
            self.kv_transfer_metrics = None

        # Build the Mooncake data-plane engine here and inject it into the
        # transfer manager. self.kv_args is set above so register_buffer_to_engine
        # (called by the base) sees the KV buffers.
        engine = MooncakeTransferEngine(
            hostname=get_local_ip_by_remote(),
            gpu_id=kv_args.gpu_id,
            ib_device=kv_args.ib_device,
        )
        super().__init__(engine=engine)

    def register_buffer_to_engine(self):
        layout = self.kv_args.cache_layout
        self.engine.register(
            self.kv_args.kv_data_ptr,
            layout.plan.arena_bytes,
        )


class MooncakeKVBootstrapServer(DisaggBootstrapServerBase):
    """CachePD bootstrap rendezvous for one typed Prefill layout."""

    def __init__(self, port: int):
        # Set before super() -- super() starts the server thread, after which a
        # register PUT can call _ingest_put_extra and read these.
        self.prefill_cache_layout_wire: str | None = None
        self.prefill_num_target_cache_layers: int | None = None
        super().__init__(port)

    def _ingest_put_extra(self, data: dict) -> None:
        cache_layout_wire = data.get("cache_layout")
        if not isinstance(cache_layout_wire, str):
            raise ValueError("CachePD bootstrap layout must be a string")
        try:
            cache_layout = CacheTransferContract.from_wire_bytes(
                cache_layout_wire.encode("ascii")
            )
        except (UnicodeEncodeError, ValueError) as exc:
            raise ValueError("CachePD bootstrap layout is invalid") from exc
        # Keep the existing wire key; its value counts target cache layers.
        num_target_cache_layers = data["num_target_layers"]
        from tokenspeed.runtime.distributed.partition import (
            target_execution_stage_windows,
        )
        from tokenspeed.runtime.layers.attention.kv_cache.recipes.ownership import (
            pipeline_cache_ownership,
        )
        from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
            cache_field_layer_id,
        )

        cache_layer_extent = (
            max(
                cache_field_layer_id(field.field_id)
                for field in cache_layout.plan.fields
            )
            + 1
        )
        # Supported PP targets use the identity execution-to-cache mapping.
        target_cache_windows = target_execution_stage_windows(
            num_target_cache_layers,
            int(data.get("pp_size", 1)),
            (
                tuple(data["pp_layer_partition"])
                if data.get("pp_layer_partition")
                else None
            ),
        )
        pipeline_cache_ownership(
            num_target_cache_layers,
            cache_layer_extent - num_target_cache_layers,
            target_cache_windows,
        )
        if self.prefill_num_target_cache_layers not in (None, num_target_cache_layers):
            raise ValueError(
                "CachePD prefill ranks registered incompatible target layers"
            )
        canonical_wire = cache_layout.to_wire_bytes().decode("ascii")
        if self.prefill_cache_layout_wire not in (None, canonical_wire):
            raise ValueError("CachePD prefill ranks registered incompatible layouts")
        self.prefill_cache_layout_wire = canonical_wire
        self.prefill_num_target_cache_layers = num_target_cache_layers

    def _extra_parallel_info(self) -> dict:
        return {
            "cache_layout": self.prefill_cache_layout_wire,
            "num_target_layers": self.prefill_num_target_cache_layers,
        }
