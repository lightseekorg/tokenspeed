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

"""CPU layout regression; replace only GPU experts, routing and collectives.

NVFP4 arithmetic is covered by the Blackwell MegaMoE kernel test. Here the
real Qwen block, router projections, shared MLP and CommManager exercise
scattered layouts, uneven token shards and empty collective owners.
"""

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.tuning import autotune, is_autotuning
from torch import nn

from tokenspeed.runtime.configs.qwen3_5_text_base_config import Qwen3_5BaseTextConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.moe import topk as topk_module
from tokenspeed.runtime.layers.moe import utils as moe_utils
from tokenspeed.runtime.layers.moe.topk import TopKOutputFormat
from tokenspeed.runtime.layers.moe.utils import All2AllBackend, MoeBackend
from tokenspeed.runtime.models import qwen3_5_moe
from tokenspeed.runtime.models.qwen3_5_moe import Qwen3_5MoeSparseMoeBlock


class CPUExperts(nn.Module):
    """Stand-in for GPU experts: expert e computes (e+1)*x."""

    def __init__(self, *, num_experts, hidden_size, **kwargs):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.topk_output_format = TopKOutputFormat.STANDARD
        self.w13_input_scale_quant = torch.ones(1)
        self.rows = None

    def forward(self, *, hidden_states, topk_output, **kwargs):
        x = hidden_states
        self.rows = x.shape[0]
        if not self.rows:
            return torch.empty((0, self.hidden_size))
        factor = (topk_output.topk_weights * (topk_output.topk_ids + 1)).sum(-1)
        return x * factor[:, None]


def _cpu_topk(logits, k, *, renormalize, **kwargs):
    weights, ids = logits.float().softmax(-1).topk(k, dim=-1)
    if renormalize:
        weights = weights / weights.sum(-1, keepdim=True)
    return weights, ids


def _cpu_shared_add(x, gate, shared, output):
    output.add_(torch.sigmoid(x @ gate)[:, None] * shared)


@pytest.mark.parametrize("attn_tp", [1, 2, 4])
@pytest.mark.parametrize("tokens", [0, 1, 3, 8, 11])
@pytest.mark.parametrize("tune_mode", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_mega_moe_restores_attention_rows(attn_tp, tokens, shared, tune_mode):
    hidden = 32
    config = Qwen3_5BaseTextConfig(
        hidden_size=hidden,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32 if shared else 0,
    )
    for rank in range(8):
        torch.manual_seed(42)
        mapping = Mapping(
            rank=rank,
            world_size=8,
            attn_tp_size=attn_tp,
            dense_tp_size=attn_tp,
            moe_tp_size=1,
            moe_ep_size=8,
        )
        with (
            patch.object(moe_utils, "MOE_BACKEND", MoeBackend.MEGA_MOE),
            patch.object(moe_utils, "ALL2ALL_BACKEND", All2AllBackend.NONE),
            patch.object(qwen3_5_moe, "MoELayer", CPUExperts),
        ):
            block = Qwen3_5MoeSparseMoeBlock(config, mapping)
        with torch.no_grad():
            for p in block.parameters():
                p.uniform_(-0.1, 0.1)
        x = torch.randn(tokens, hidden)
        weights, ids = _cpu_topk(F.linear(x, block.gate.weight), 2, renormalize=True)
        expected = x * (weights * (ids + 1)).sum(-1, keepdim=True)
        if shared:
            gate, up = F.linear(x, block.shared_expert.gate_up_proj.weight).chunk(2, -1)
            shared_out = F.linear(
                F.silu(gate) * up, block.shared_expert.down_proj.weight
            )
            expected += (
                F.linear(x, block.shared_expert_gate.weight).sigmoid() * shared_out
            )
        ctx = ForwardContext(
            attn_backend=None,
            token_to_kv_pool=None,
            bs=1,
            num_extends=1,
            input_num_tokens=tokens,
            forward_mode=None,
            output_layout=None,
            global_num_tokens=[tokens] * 8,
        )
        parts = torch.tensor_split(x, attn_tp)
        expected_parts = torch.tensor_split(expected, attn_tp)
        local_rank = mapping.attn.tp_rank
        inputs = parts[local_rank]
        wanted = expected_parts[local_rank]

        with (
            autotune(tune_mode=tune_mode, tuning_buckets=None, round_up=None),
            patch.object(topk_module, "moe_topk", _cpu_topk),
            patch.object(qwen3_5_moe, "fused_gate_sigmoid_mul_add", _cpu_shared_add),
        ):
            total, capacity = block.comm_manager.get_num_tokens(ctx)
            actual = block(inputs, total, capacity, ctx)
            assert is_autotuning() == tune_mode
        torch.testing.assert_close(actual, wanted)
        assert block.experts.rows == len(parts[local_rank])


@pytest.mark.parametrize(
    "moe_tp,moe_ep,all2all,options,message",
    [
        (8, 1, All2AllBackend.NONE, {}, "EP>1"),
        (1, 8, All2AllBackend.DEEPEP, {}, "all2all-backend none"),
        (1, 8, All2AllBackend.NONE, {"enable_eplb": True}, "trivial expert placement"),
        (
            1,
            8,
            All2AllBackend.NONE,
            {"ep_num_redundant_experts": 1},
            "trivial expert placement",
        ),
        (
            1,
            8,
            All2AllBackend.NONE,
            {"init_expert_location": "random"},
            "trivial expert placement",
        ),
    ],
)
def test_mega_moe_rejects_incompatible_options(
    moe_tp, moe_ep, all2all, options, message
):
    config = Qwen3_5BaseTextConfig(hidden_size=32, num_experts=8)
    mapping = Mapping(
        rank=0,
        world_size=8,
        attn_tp_size=2,
        dense_tp_size=2,
        moe_tp_size=moe_tp,
        moe_ep_size=moe_ep,
    )
    with (
        patch.object(moe_utils, "MOE_BACKEND", MoeBackend.MEGA_MOE),
        patch.object(moe_utils, "ALL2ALL_BACKEND", all2all),
        patch.dict(qwen3_5_moe.global_server_args_dict, options, clear=True),
        pytest.raises(ValueError, match=message),
    ):
        Qwen3_5MoeSparseMoeBlock(config, mapping)


@pytest.mark.parametrize("attn_tp,ep", [(8, 8), (4, 4), (2, 2)])
def test_mega_moe_rejects_replicated_rows(attn_tp, ep):
    config = Qwen3_5BaseTextConfig(hidden_size=32, num_experts=8)
    mapping = Mapping(
        rank=0,
        world_size=8,
        attn_tp_size=attn_tp,
        dense_tp_size=attn_tp,
        moe_tp_size=1,
        moe_ep_size=ep,
    )
    with (
        patch.object(moe_utils, "MOE_BACKEND", MoeBackend.MEGA_MOE),
        patch.object(moe_utils, "ALL2ALL_BACKEND", All2AllBackend.NONE),
        patch.dict(qwen3_5_moe.global_server_args_dict, {}, clear=True),
        pytest.raises(ValueError, match="requires scattered MoE rows"),
    ):
        Qwen3_5MoeSparseMoeBlock(config, mapping)
