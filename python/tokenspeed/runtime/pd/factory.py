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

from tokenspeed.runtime.layers.attention.kv_cache.recipes.transfer import (
    build_cache_transfer_schema,
)
from tokenspeed.runtime.pd.cache_protocol import (
    build_cache_fields_by_producer_step,
    build_pool_cache_transfer_contract,
)
from tokenspeed.runtime.pd.decode_executor import DisaggDecodeExecutor
from tokenspeed.runtime.pd.mooncake.entities import KVArgs, KVManagerArgs
from tokenspeed.runtime.pd.prefill_executor import DisaggPrefillExecutor
from tokenspeed.runtime.pd.utils import TransferBackend


def get_kv_args(
    engine_rank: int,
    gpu_id,
    ib_device,
    token_to_kv_pool,
    *,
    model_config,
    draft_model_config=None,
):
    # One big model, one arena: a draft's continuation-layer planes live in
    # the target pool's merged plan, so exactly one typed slab registration is
    # published for both target and draft caches.
    transfer_schema = build_cache_transfer_schema(
        token_to_kv_pool.plan,
        model_config=model_config,
        draft_model_config=draft_model_config,
    )
    producer_schedule = build_cache_fields_by_producer_step(
        token_to_kv_pool.plan,
        num_target_layers=model_config.num_attention_layers,
    )
    layout, base_addr = build_pool_cache_transfer_contract(
        token_to_kv_pool,
        transfer_schema=transfer_schema,
    )
    return KVArgs(
        engine_rank=engine_rank,
        kv_data_ptr=base_addr,
        ib_device=ib_device,
        gpu_id=gpu_id,
        cache_layout=layout,
        cache_producer_schedule=producer_schedule,
    )


def create_kv_transfer(
    mode: str,
    backend: TransferBackend,
    args: KVManagerArgs,
    kv_args: KVArgs,
    gloo_group,
):
    if backend not in (TransferBackend.MOONCAKE, TransferBackend.MOONCAKE.value):
        raise NotImplementedError("CachePD supports only the Mooncake backend")
    if mode == "prefill":
        return DisaggPrefillExecutor(args, kv_args, gloo_group)
    elif mode == "decode":
        return DisaggDecodeExecutor(args, kv_args, gloo_group)
    else:
        raise NotImplementedError(f"Unsupported disaggregation mode: {mode}")
