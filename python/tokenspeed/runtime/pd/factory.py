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

"""Factories for PD KV transfer helpers."""

from tokenspeed.runtime.pd.cache_protocol import (
    build_cache_fields_by_producer_step,
    build_cache_transfer_schema,
    build_pool_cache_transfer_contract,
)
from tokenspeed.runtime.pd.decode_executor import DisaggDecodeExecutor
from tokenspeed.runtime.pd.mooncake.entities import KVArgs, KVManagerArgs
from tokenspeed.runtime.pd.prefill_executor import DisaggPrefillExecutor
from tokenspeed.runtime.pd.utils import TransferBackend


def _get_contiguous_buf_unit_lens(pool, item_lens):
    getter = getattr(pool, "get_contiguous_buf_unit_lens", None)
    if getter is None:
        return [1] * len(item_lens)
    unit_lens = list(getter())
    if len(unit_lens) != len(item_lens):
        raise ValueError(
            f"contiguous buffer unit count mismatch: units={len(unit_lens)}, items={len(item_lens)}"
        )
    return unit_lens


def _get_cache_contract(pool, *, model_config, draft_model_config):
    if getattr(pool, "supports_disaggregation", False) is not True:
        return None
    transfer_schema = build_cache_transfer_schema(
        pool.plan,
        model_config=model_config,
        draft_model_config=draft_model_config,
    )
    producer_schedule = build_cache_fields_by_producer_step(
        pool.plan,
        num_target_layers=model_config.num_attention_layers,
    )
    layout, base_addr = build_pool_cache_transfer_contract(
        pool,
        transfer_schema=transfer_schema,
    )
    return layout, base_addr, producer_schedule


def get_kv_args(
    engine_rank: int,
    gpu_id,
    ib_device,
    token_to_kv_pool,
    draft_token_to_kv_pool,
    *,
    model_config,
    draft_model_config=None,
):
    cache_contract = _get_cache_contract(
        token_to_kv_pool,
        model_config=model_config,
        draft_model_config=draft_model_config,
    )
    if cache_contract is not None:
        # One big model, one arena: the draft's continuation-layer planes
        # live inside the same merged plan the contract describes, so slab
        # pages carry the draft KV with no extra registration
        # (draft_token_to_kv_pool is a layer-mapped view of the same pool).
        layout, base_addr, producer_schedule = cache_contract
        item_len = layout.plan.lcm_block_bytes
        return KVArgs(
            engine_rank=engine_rank,
            kv_data_ptrs=[base_addr],
            kv_data_lens=[layout.plan.arena_bytes],
            kv_item_lens=[item_len],
            target_layer_num=1,
            draft_layer_num=0,
            kv_layer_ids=[0],
            # One logical Mooncake unit is one complete raw-arena parent page.
            kv_unit_lens=[item_len],
            state_data_ptrs=[],
            state_data_lens=[],
            state_item_lens=[],
            state_unit_lens=[],
            state_type="none",
            state_layer_ids=[],
            mamba_offsets=[],
            offsets=[(0,)],
            aux_data_ptrs=[],
            aux_data_lens=[],
            aux_item_lens=[],
            ib_device=ib_device,
            gpu_id=gpu_id,
            cache_layout=layout,
            cache_producer_schedule=producer_schedule,
        )

    # One big model, one pool: the pool's buffers already cover the draft's
    # continuation layers (the pool binds every planned layer), so a single
    # enumeration registers everything. The draft pool is a layer-mapped
    # view of the same pool; only the layer partition is derived from it.
    kv_data_ptrs, kv_data_lens, kv_item_lens = (
        token_to_kv_pool.get_contiguous_buf_infos()
    )
    kv_unit_lens = _get_contiguous_buf_unit_lens(token_to_kv_pool, kv_item_lens)
    # [[layer0buf0, layer0buf1...], [layer1buf0, layer1buf1...], ...]
    offsets = token_to_kv_pool.get_layerwise_buf_info_offsets()
    total_layer_num = token_to_kv_pool.layer_num
    kv_layer_ids = list(getattr(token_to_kv_pool, "layer_ids", range(total_layer_num)))
    draft_layer_num = (
        len(draft_token_to_kv_pool.layer_ids)
        if draft_token_to_kv_pool is not None
        else 0
    )
    target_layer_num = total_layer_num - draft_layer_num

    state_data_ptrs = []
    state_data_lens = []
    state_item_lens = []
    state_unit_lens = []
    state_type = "none"
    state_layer_ids = []
    kv_args = KVArgs(
        engine_rank=engine_rank,
        kv_data_ptrs=kv_data_ptrs,
        kv_data_lens=kv_data_lens,
        kv_item_lens=kv_item_lens,
        target_layer_num=target_layer_num,
        draft_layer_num=draft_layer_num,
        kv_layer_ids=kv_layer_ids,
        kv_unit_lens=kv_unit_lens,
        state_data_ptrs=state_data_ptrs,
        state_data_lens=state_data_lens,
        state_item_lens=state_item_lens,
        state_unit_lens=state_unit_lens,
        state_type=state_type,
        state_layer_ids=state_layer_ids,
        mamba_offsets=[],
        offsets=offsets,
        aux_data_ptrs=[],
        aux_data_lens=[],
        aux_item_lens=[],
        ib_device=ib_device,
        gpu_id=gpu_id,
    )

    return kv_args


def create_kv_transfer(
    mode: str,
    backend: TransferBackend,
    args: KVManagerArgs,
    kv_args: KVArgs,
    gloo_group,
    page_size,
):
    if kv_args.cache_layout is not None:
        if backend not in (TransferBackend.MOONCAKE, TransferBackend.MOONCAKE.value):
            raise NotImplementedError(
                "Paged-cache PD currently supports only the Mooncake backend"
            )
    if mode == "prefill":
        return DisaggPrefillExecutor(backend, args, kv_args, gloo_group, page_size)
    elif mode == "decode":
        return DisaggDecodeExecutor(backend, args, kv_args, gloo_group, page_size)
    else:
        raise NotImplementedError(f"Unsupported disaggregation mode: {mode}")
