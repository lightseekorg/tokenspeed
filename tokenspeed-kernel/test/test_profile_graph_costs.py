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

from dataclasses import replace
from math import prod

import pytest
from tokenspeed_kernel.op_profile import OpRecord, TensorDescription
from tokenspeed_kernel.profile_attention_moe import grouped_expert_cost, mla_decode_cost


def desc(shape, size=2, dtype="torch.bfloat16"):
    return TensorDescription(
        tuple(shape),
        dtype,
        tuple(prod(shape[i + 1 :]) for i in range(len(shape))),
        "cuda:0",
        prod(shape) * size,
    )


def record(inputs, output, metadata, values):
    return OpRecord(
        0,
        0,
        0,
        0,
        "test.op",
        "test",
        None,
        inputs,
        {"return": output},
        0,
        1,
        None,
        "test",
        scalar_args=metadata,
        tensor_values=values,
    )


@pytest.mark.parametrize("layout,width", [("query_axis", 1), ("flattened", 8)])
def test_native_mla_adapter_uses_actual_lengths_and_matches_known_cost(layout, width):
    rows, axis = (1, width) if layout == "query_axis" else (width, 1)
    lengths = (136,) if layout == "query_axis" else tuple(range(137 - width, 137))
    op = record(
        dict(
            q=desc((rows, axis, 8, 576)),
            kv_cache=desc((32, 64, 1, 576)),
            page_table=desc((rows, 16), 4, "torch.int32"),
            cache_seqlens=desc((rows,), 4, "torch.int32"),
        ),
        desc((rows, axis, 8, 512)),
        dict(
            logical_batch=1,
            query_width=width,
            attention_tp_size=8,
            dcp_size=1,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            window_left=-1,
            noncausal_block_size=1,
            return_lse=False,
        ),
        {"cache_seqlens": lengths},
    )
    cost = mla_decode_cost(op, hbm_policy="logical_io_once")
    # Width 1 attends 136 KV rows; width 8 attends 129+...+136.
    # KV bytes are the 136-row request union, with metadata per actual row.
    expected = {
        "query_axis": (2367488, 165904, 8192),
        "flattened": (18452480, 230528, 65536),
    }[layout]
    assert (
        cost.useful_flops,
        cost.logical_read_bytes,
        cost.logical_write_bytes,
    ) == expected
    assert cost.estimated_hbm_bytes == expected[1] + expected[2]
    assert cost.compute_type is None
    with pytest.raises(ValueError, match="Actual per-call"):
        mla_decode_cost(replace(op, tensor_values={}), hbm_policy="logical_io_once")
    with pytest.raises(ValueError, match="geometry"):
        mla_decode_cost(
            replace(op, scalar_args={**op.scalar_args, "logical_batch": None}),
            hbm_policy="logical_io_once",
        )


def test_native_expert_adapter_keeps_route_instances_unique_weights_and_prepadding_size():
    op = record(
        {
            "hidden_states": desc((2, 4)),
            "topk_output.topk_ids": desc((2, 2), 4, "torch.int32"),
            "topk_output.topk_weights": desc((2, 2), 4, "torch.float32"),
            "self.w13_weight": desc((2, 16, 4), 1, "torch.float8_e4m3fn"),
            "self.w2_weight": desc((2, 4, 8), 1, "torch.float8_e4m3fn"),
            "self.w13_weight_scale_inv": desc((2, 4, 1), 4, "torch.float32"),
            "self.w2_weight_scale_inv": desc((2, 1, 2), 4, "torch.float32"),
        },
        desc((2, 4)),
        {
            "self.hidden_size": 4,
            "self.profile_useful_intermediate_size": 6,
            "self.num_experts": 8,
            "self.num_local_experts": 2,
            "self.ep_rank": 1,
            "self.ep_size": 4,
            "self.tp_size": 1,
            "self.zero_expert_num": 0,
            "self.ep_num_redundant_experts": 0,
            "self._quant_kind": "fp8",
            "self.profile_scale_block_shape": (4, 4),
            "self.plan.supports_all_to_all_ep": False,
            "moe_group_count": 1,
            "moe_route_policy": "global_expert_id",
            "moe_expert_placement": "trivial",
            "moe_enable_eplb": False,
            "logical_batch": 1,
            "query_width": 2,
            "dcp_size": 1,
            "do_finalize": True,
        },
        {"topk_output.topk_ids": (2, 7, 2, 3)},
    )
    cost = grouped_expert_cost(op, hbm_policy="logical_io_once")
    # Three local route instances, two unique local weights/scales: F=432;
    # R=16 hidden +16 IDs +16 route weights +192 expert weights +48 scales.
    assert (cost.useful_flops, cost.logical_read_bytes, cost.logical_write_bytes) == (
        432,
        288,
        16,
    )
    assert cost.compute_type is None and cost.estimated_hbm_bytes is not None
    packed = {**op.scalar_args, "logical_batch": 2, "logical_tokens": 2}
    del packed["query_width"]
    assert (
        grouped_expert_cost(
            replace(op, scalar_args=packed), hbm_policy="logical_io_once"
        )
        == cost
    )
    with pytest.raises(ValueError, match="differs from logical batch/query width"):
        grouped_expert_cost(
            replace(op, scalar_args={**op.scalar_args, "logical_tokens": 3}),
            hbm_policy="logical_io_once",
        )
    with pytest.raises(ValueError, match="Actual per-call"):
        grouped_expert_cost(replace(op, tensor_values={}), hbm_policy="logical_io_once")
    with pytest.raises(ValueError, match="pre-padding"):
        grouped_expert_cost(
            replace(
                op,
                scalar_args={
                    **op.scalar_args,
                    "self.profile_useful_intermediate_size": None,
                },
            ),
            hbm_policy="logical_io_once",
        )
