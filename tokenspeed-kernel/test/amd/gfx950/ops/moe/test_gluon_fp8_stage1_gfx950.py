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
import torch


def _requires_gfx950() -> None:
    if torch.version.hip is None or torch.cuda.get_device_capability() != (9, 5):
        pytest.skip("requires gfx950")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_compact_fp8_stage1_graph_replay_matches_materialized_bf16() -> None:
    _requires_gfx950()
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8 import (
        gluon_fp8_block_dequantize,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.moe_align_device import (
        moe_align_block_size_device,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.stage1_kernel import invoke_stage1

    generator = torch.Generator(device="cuda").manual_seed(29)
    tokens, hidden, intermediate = 64, 4096, 512
    experts, topk, block_m = 16, 8, 16
    activations = torch.randn(
        tokens,
        hidden,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    weight = torch.randn(
        experts,
        2 * intermediate,
        hidden,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).to(torch.float8_e4m3fn)
    scale = (
        torch.rand(
            experts,
            2 * intermediate // 128,
            hidden // 128,
            device="cuda",
            generator=generator,
        )
        * 0.002
    )
    topk_ids = torch.randint(
        0,
        experts,
        (tokens, topk),
        device="cuda",
        dtype=torch.int32,
        generator=generator,
    )
    topk_weights = torch.softmax(
        torch.randn(tokens, topk, device="cuda", generator=generator), dim=-1
    )
    sorted_ids, sorted_experts, _, num_valid = moe_align_block_size_device(
        topk_ids, topk_weights, experts, block_m
    )
    actual = torch.empty(
        tokens * topk,
        intermediate,
        device="cuda",
        dtype=torch.bfloat16,
    )
    expected = torch.empty_like(actual)

    invoke_stage1(
        activations,
        gluon_fp8_block_dequantize(weight, scale),
        sorted_ids,
        sorted_experts,
        num_valid,
        expected,
        topk,
        BLOCK_M=block_m,
        BLOCK_N=64,
        BLOCK_K=64,
        split_k=1,
    )
    invoke_stage1(
        activations,
        weight,
        sorted_ids,
        sorted_experts,
        num_valid,
        actual,
        topk,
        BLOCK_M=block_m,
        BLOCK_N=64,
        split_k=1,
        w1_scale=scale,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke_stage1(
            activations,
            weight,
            sorted_ids,
            sorted_experts,
            num_valid,
            actual,
            topk,
            BLOCK_M=block_m,
            BLOCK_N=64,
            split_k=1,
            w1_scale=scale,
        )
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
