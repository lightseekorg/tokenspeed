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

"""The PD transfer description of a rank's KV arena."""

import dataclasses

from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import CacheMemoryPlan
from tokenspeed.runtime.layers.attention.kv_cache.recipes.transfer import (
    build_cache_transfer_schema,
)
from tokenspeed.runtime.pd.cache_protocol import (
    CacheProducerSchedule,
    CacheTransferContract,
    build_arena_cache_transfer_contract,
    build_cache_fields_by_producer_step,
)


@dataclasses.dataclass
class KVArgs:
    """A rank's KV arena as PD transfers it: the arena, its cache transfer
    contract and the cache's placement over pipeline stages and producer
    steps."""

    engine_rank: int
    kv_data_ptr: int
    ib_device: str
    gpu_id: int
    cache_layout: CacheTransferContract
    cache_fields_by_stage: tuple[tuple[str, ...], ...]
    registration_regions: tuple[tuple[int, int], ...] = ()
    cache_producer_schedule: CacheProducerSchedule | None = None
    # Full-model logical contract for the PD wire when the local arena/plan
    # is narrowed to a stage window; None means cache_layout is already it.
    wire_cache_layout: CacheTransferContract | None = None

    @property
    def wire_layout(self) -> CacheTransferContract:
        layout = (
            self.wire_cache_layout
            if self.wire_cache_layout is not None
            else self.cache_layout
        )
        # Bootstrap compares the logical layout across ranks. Physical bindings
        # are rank-local and belong only to the transfer endpoint contract.
        return dataclasses.replace(layout, field_addresses=None)


def get_kv_args(
    engine_rank: int,
    gpu_id,
    ib_device,
    token_to_kv_pool,
    *,
    model_config,
    draft_model_config,
    cache_fields_by_stage: tuple[tuple[str, ...], ...],
    producer_fields_by_step: tuple[tuple[str, ...], ...],
    logical_plan: CacheMemoryPlan | None,
):
    """Build PD arguments from the arena and explicit cache-placement metadata.

    ``logical_plan`` is the complete layout before PP physical narrowing, or
    None when the arena already holds the complete plan.
    """
    # One big model, one arena: a draft's continuation-layer planes live in
    # the target pool's merged plan, so exactly one typed slab registration is
    # published for both target and draft caches.
    transfer_schema = build_cache_transfer_schema(
        token_to_kv_pool.arena.plan,
        model_config=model_config,
        draft_model_config=draft_model_config,
    )
    producer_schedule = build_cache_fields_by_producer_step(
        token_to_kv_pool.arena.plan,
        producer_fields_by_step=producer_fields_by_step,
    )
    layout, base_addr = build_arena_cache_transfer_contract(
        token_to_kv_pool.arena,
        transfer_schema=transfer_schema,
    )
    # Chunk-pipeline: the arena holds only this stage's layers (narrowed
    # plan), which is what local source addressing must use. The WIRE
    # contract, however, must be the full logical plan: every stage
    # registers the same layout, and Decode validates + plans stage windows
    # against the complete field set.
    wire_layout = None
    if logical_plan is not None:
        wire_layout = CacheTransferContract(
            plan=logical_plan,
            group_specs=token_to_kv_pool.arena.cache_group_specs,
            transfer_schema=build_cache_transfer_schema(
                logical_plan,
                model_config=model_config,
                draft_model_config=draft_model_config,
            ),
        )
    return KVArgs(
        engine_rank=engine_rank,
        kv_data_ptr=base_addr,
        registration_regions=token_to_kv_pool.arena.registration_regions(),
        ib_device=ib_device,
        gpu_id=gpu_id,
        cache_layout=layout,
        cache_producer_schedule=producer_schedule,
        cache_fields_by_stage=cache_fields_by_stage,
        wire_cache_layout=wire_layout,
    )
