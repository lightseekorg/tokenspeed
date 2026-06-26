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

import pytest
import tokenspeed_kernel
import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.gemm.fp8_utils import per_token_group_quant_fp8
from tokenspeed_kernel.platform import current_platform


def test_triton_fp8_moe_matches_dynamic_fp8_reference(device: str) -> None:
    if not current_platform().is_cdna4:
        pytest.skip("triton FP8 MoE bring-up path is currently targeted at AMD CDNA4")

    torch.manual_seed(0)
    num_tokens = 4
    hidden_size = 128
    intermediate_size = 128
    num_experts = 2
    top_k = 2
    fp8_dtype = current_platform().fp8e4m3fn.dtype

    x = (
        torch.randn((num_tokens, hidden_size), device=device, dtype=torch.bfloat16)
        * 0.1
    )
    w13_float = (
        torch.randn(
            (num_experts, 2 * intermediate_size, hidden_size),
            device=device,
            dtype=torch.float32,
        )
        * 0.05
    )
    w2_float = (
        torch.randn(
            (num_experts, hidden_size, intermediate_size),
            device=device,
            dtype=torch.float32,
        )
        * 0.05
    )

    w = torch.nn.Module()
    w.num_experts = num_experts
    w.num_local_experts = num_experts
    w.top_k = top_k
    w.w13_weight = w13_float.to(fp8_dtype).contiguous()
    w.w2_weight = w2_float.to(fp8_dtype).contiguous()
    w.w13_weight_scale_inv = torch.ones(
        (num_experts, 2, 1),
        device=device,
        dtype=torch.float32,
    )
    w.w2_weight_scale_inv = torch.ones(
        (num_experts, 1, 1),
        device=device,
        dtype=torch.float32,
    )

    topk_ids = torch.tensor(
        [[0, 1], [1, 0], [0, 1], [1, 0]], device=device, dtype=torch.int32
    )
    topk_weights = torch.tensor(
        [[0.7, 0.3], [0.6, 0.4], [1.0, 0.0], [0.2, 0.8]],
        device=device,
        dtype=torch.float32,
    )
    router_logits = torch.empty(
        (num_tokens, num_experts), device=device, dtype=torch.float32
    )
    plan = tokenspeed_kernel.moe_plan(
        "fp8",
        input_dtype=torch.bfloat16,
        activation="silu",
        ep_size=1,
        fp8_scale_block_shape=(128, 128),
        solution="triton",
    )

    actual = tokenspeed_kernel.moe_apply(
        plan,
        x,
        w,
        router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    torch.cuda.synchronize()

    expected = torch.zeros_like(actual)
    for token in range(num_tokens):
        for route in range(top_k):
            expert = int(topk_ids[token, route].item())
            route_weight = topk_weights[token, route]
            x_q, x_s = per_token_group_quant_fp8(
                x[token : token + 1].contiguous(),
                128,
                column_major_scales=False,
            )
            x_dequant = x_q.float() * x_s
            gate_up = x_dequant @ w.w13_weight[expert].float().T
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = (F.silu(gate) * up).to(torch.bfloat16)
            hidden_q, hidden_s = per_token_group_quant_fp8(
                hidden.contiguous(),
                128,
                column_major_scales=False,
            )
            hidden_dequant = hidden_q.float() * hidden_s
            out = hidden_dequant @ w.w2_weight[expert].float().T
            expected[token] += (out.squeeze(0) * route_weight).to(torch.bfloat16)

    torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.003)


def test_triton_fp8_moe_cuda_graph_capture(device: str) -> None:
    if not current_platform().is_cdna4:
        pytest.skip("triton FP8 MoE bring-up path is currently targeted at AMD CDNA4")

    torch.manual_seed(1)
    num_tokens = 4
    hidden_size = 128
    intermediate_size = 128
    num_experts = 2
    top_k = 2
    fp8_dtype = current_platform().fp8e4m3fn.dtype

    x = (
        torch.randn((num_tokens, hidden_size), device=device, dtype=torch.bfloat16)
        * 0.1
    )
    w13_float = (
        torch.randn(
            (num_experts, 2 * intermediate_size, hidden_size),
            device=device,
            dtype=torch.float32,
        )
        * 0.05
    )
    w2_float = (
        torch.randn(
            (num_experts, hidden_size, intermediate_size),
            device=device,
            dtype=torch.float32,
        )
        * 0.05
    )

    w = torch.nn.Module()
    w.num_experts = num_experts
    w.num_local_experts = num_experts
    w.top_k = top_k
    w.w13_weight = w13_float.to(fp8_dtype).contiguous()
    w.w2_weight = w2_float.to(fp8_dtype).contiguous()
    w.w13_weight_scale_inv = torch.ones(
        (num_experts, 2, 1),
        device=device,
        dtype=torch.float32,
    )
    w.w2_weight_scale_inv = torch.ones(
        (num_experts, 1, 1),
        device=device,
        dtype=torch.float32,
    )

    topk_ids = torch.tensor(
        [[0, 1], [1, 0], [0, 1], [1, 0]], device=device, dtype=torch.int32
    )
    topk_weights = torch.tensor(
        [[0.7, 0.3], [0.6, 0.4], [1.0, 0.0], [0.2, 0.8]],
        device=device,
        dtype=torch.float32,
    )
    router_logits = torch.empty(
        (num_tokens, num_experts), device=device, dtype=torch.float32
    )
    plan = tokenspeed_kernel.moe_plan(
        "fp8",
        input_dtype=torch.bfloat16,
        activation="silu",
        ep_size=1,
        fp8_scale_block_shape=(128, 128),
        solution="triton",
    )

    expected = tokenspeed_kernel.moe_apply(
        plan,
        x,
        w,
        router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = tokenspeed_kernel.moe_apply(
            plan,
            x,
            w,
            router_logits,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.003)
