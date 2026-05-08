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

"""Factories for disaggregation KV transfer helpers."""

from tokenspeed.runtime.pd.base import KVArgs
from tokenspeed.runtime.pd.decode_executor import DisaggDecodeExecutor
from tokenspeed.runtime.pd.mooncake.entities import ManagerArgs
from tokenspeed.runtime.pd.prefill_executor import DisaggPrefillExecutor
from tokenspeed.runtime.pd.utils import (
    DisaggregationMode,
    TransferBackend,
)


def get_kv_args(
    engine_rank: int,
    gpu_id,
    ib_device,
    token_to_kv_pool,
    draft_token_to_kv_pool,
    mamba_pool=None,
):
    kv_data_ptrs, kv_data_lens, kv_item_lens = (
        token_to_kv_pool.get_contiguous_buf_infos()
    )
    # [[layer0buf0, layer0buf1...], [layer1buf0, layer1buf1...], ...]
    offsets = token_to_kv_pool.get_layerwise_buf_info_offsets()
    target_layer_num = token_to_kv_pool.layer_num
    kv_layer_ids = list(getattr(token_to_kv_pool, "layer_ids", range(target_layer_num)))

    if draft_token_to_kv_pool is not None:
        draft_layer_num = draft_token_to_kv_pool.layer_num
        # We should also transfer draft model kv cache. The indices are
        # always shared with a target model.
        draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = (
            draft_token_to_kv_pool.get_contiguous_buf_infos()
        )
        draft_offsets = draft_token_to_kv_pool.get_layerwise_buf_info_offsets(
            len(kv_data_ptrs)
        )

        kv_data_ptrs += draft_kv_data_ptrs
        kv_data_lens += draft_kv_data_lens
        kv_item_lens += draft_kv_item_lens
        offsets += draft_offsets
        draft_base_layer_id = max(kv_layer_ids) + 1 if kv_layer_ids else target_layer_num
        kv_layer_ids += list(
            range(draft_base_layer_id, draft_base_layer_id + draft_layer_num)
        )
    else:
        draft_layer_num = 0

    state_data_ptrs = []
    state_data_lens = []
    state_item_lens = []
    state_type = "none"
    state_layer_ids = []
    if mamba_pool is not None:
        state_data_ptrs, state_data_lens, state_item_lens = (
            mamba_pool.get_contiguous_buf_infos()
        )
        state_layer_ids = mamba_pool.get_contiguous_buf_layer_ids()
        state_type = "mamba"

    kv_args = KVArgs(
        engine_rank=engine_rank,
        kv_data_ptrs=kv_data_ptrs,
        kv_data_lens=kv_data_lens,
        kv_item_lens=kv_item_lens,
        target_layer_num=target_layer_num,
        draft_layer_num=draft_layer_num,
        kv_layer_ids=kv_layer_ids,
        state_data_ptrs=state_data_ptrs,
        state_data_lens=state_data_lens,
        state_item_lens=state_item_lens,
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


def create_pd_kv_transfer(
    mode: DisaggregationMode,
    backend: TransferBackend,
    args: ManagerArgs,
    kv_args: KVArgs,
    gloo_group,
    page_size,
):
    if mode == "prefill":
        return DisaggPrefillExecutor(backend, args, kv_args, gloo_group, page_size)
    elif mode == "decode":
        return DisaggDecodeExecutor(backend, args, kv_args, gloo_group, page_size)
    else:
        raise NotImplementedError(f"Unsupported disaggregation mode: {mode}")
