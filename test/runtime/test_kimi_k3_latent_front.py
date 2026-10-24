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

"""Front fusion through the attention-DP MoE block with unloaded experts."""

import math
import os
import sys

import pytest
import torch
from tokenspeed_kernel.ops.moe import try_latent_moe_input_projections
from tokenspeed_kernel.platform import ArchVersion, current_platform
from tokenspeed_kernel.thirdparty.cuda.flashinfer_kimi_k3 import latent_front_available
from tokenspeed_kernel.thirdparty.cute_dsl.mega_moe import runner
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.layers.moe.topk import TopKOutputFormat
from tokenspeed.runtime.layers.moe.utils import All2AllBackend, MoeBackend
from tokenspeed.runtime.models import kimi_k3


class UnloadedExperts(nn.Module):
    """Replace missing expert weights/EP ranks while exercising real dense layers."""

    def __init__(self, *, top_k, num_experts, hidden_size, **kwargs):
        super().__init__()
        self.plan = {"weight_dtype": "bf16", "a2a_backend": "none"}
        self.supports_precomputed_topk = True
        self.num_experts = num_experts
        self.activation = torch.empty(8192, hidden_size, dtype=torch.bfloat16)
        self.scales = torch.empty(8192, hidden_size // 16, dtype=torch.uint8)
        self.ids = torch.empty(8192, top_k, dtype=torch.int64)
        self.weights = torch.empty(8192, top_k, dtype=torch.float32)

    def input_buffers(self, tokens, capacity):
        return (
            self.activation[:tokens],
            self.scales[:tokens],
            self.ids[:tokens],
            self.weights[:tokens],
        )

    def forward(
        self,
        *,
        hidden_states,
        topk_output,
        num_global_tokens,
        max_num_tokens_per_gpu,
        do_finalize
    ):
        factor = (
            topk_output.topk_weights.float()
            * (1.0 + topk_output.topk_ids.float() / self.num_experts)
        ).sum(-1, keepdim=True)
        return (hidden_states.float() * factor).to(hidden_states.dtype)


@pytest.mark.parametrize("rows", [64, 2048, 8192])
def test_attention_dp_front_preserves_routing_and_output(monkeypatch, rows):
    platform = current_platform()
    if not (
        platform.is_nvidia
        and platform.arch_version == ArchVersion(10, 3)
        and platform.sm_count == 152
        and latent_front_available()
    ):
        pytest.skip("requires the GB300 Kimi-K3 source bundle")
    monkeypatch.setattr(kimi_k3, "MoELayer", UnloadedExperts)
    monkeypatch.setattr(runner, "get_runner", lambda plan, experts: experts)
    monkeypatch.setattr(kimi_k3, "get_moe_backend", lambda: MoeBackend.MEGA_MOE)
    monkeypatch.setattr(kimi_k3, "get_all2all_backend", lambda: All2AllBackend.NONE)
    monkeypatch.setenv("TOKENSPEED_KIMI_K3_SHARED_EXPERT_TP_SIZE", "1")
    mapping = Mapping(rank=0, world_size=2, attn_dp_size=2, moe_ep_size=2)
    config = KimiLinearConfig(
        hidden_size=7168,
        routed_expert_hidden_size=3584,
        moe_intermediate_size=6144,
        num_shared_experts=1,
        num_experts=896,
        num_experts_per_token=16,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            layer = kimi_k3.KimiLinearMoE(
                config=config,
                mapping=mapping,
                layer_index=1,
                model_scope="front_test",
                moe_block_count=1,
                quant_config=None,
                prefix="moe",
                alt_stream=None,
            )
    finally:
        torch.set_default_dtype(dtype)
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=1,
        num_extends=1,
        input_num_tokens=rows,
        forward_mode=ForwardMode.EXTEND,
        output_layout=ForwardOutputLayout(1, 1, 0, 1),
        global_num_tokens=[rows, rows],
    )
    torch.manual_seed(271 + rows)
    with torch.no_grad():
        for parameter in layer.parameters():
            if parameter.ndim == 1:
                parameter.fill_(1.0)
            else:
                parameter.normal_(std=1.0 / math.sqrt(parameter.shape[-1]))
        layer.gate.e_score_correction_bias.zero_()
        hidden = torch.randn(rows, 7168, device="cuda", dtype=torch.bfloat16) * 0.25
        prefix = torch.randn_like(hidden)

        def reference():
            logits = layer.gate(hidden)
            routing = layer.topk(
                hidden,
                logits,
                output_format=TopKOutputFormat.STANDARD,
                num_token_non_padded=None,
                expert_location_dispatch_info=None,
                out=(layer.experts.weights[:rows], layer.experts.ids[:rows]),
            )
            latent, _ = layer.routed_expert_down_proj(hidden)
            routed = layer.experts(
                hidden_states=latent,
                topk_output=routing,
                num_global_tokens=2 * rows,
                max_num_tokens_per_gpu=rows,
                do_finalize=True,
            )
            if layer.routed_expert_norm is not None:
                routed = layer.routed_expert_norm(routed)
            shared = layer.shared_experts(hidden, down_out=None)
            return layer.routed_expert_up_proj.forward_add3(
                routed,
                prefix,
                shared,
                norm_weight=None,
                eps=None,
            )

        fused = try_latent_moe_input_projections(
            hidden,
            layer.gate.weight,
            layer.routed_expert_down_proj.weight,
            layer.shared_experts.gate_up_proj.weight,
            gate_clamp=4.0,
            up_clamp=25.0,
        )
        assert (fused is not None) == (rows >= 2048)
        expected = reference()
        actual = layer(hidden, prefix, 2 * rows, rows, ctx, prefix_is_sharded=False)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = layer(
                hidden, prefix, 2 * rows, rows, ctx, prefix_is_sharded=False
            )
        hidden.normal_(std=0.125)
        graph.replay()
        torch.testing.assert_close(captured, reference(), atol=1e-2, rtol=1e-2)
