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

"""Preserve the pre-refactor LongCat torch router and test batch/graph ties."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="the batch-invariant torch router is registered for NVIDIA only",
)

from tokenspeed_kernel.ops.moe import plan_softmax_bias_topk, softmax_bias_topk
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.selection import NoKernelFoundError

from tokenspeed.runtime.layers.moe.topk import torch_router_topk

# torch_softmax_bias_topk declares batch invariance only on NVIDIA.
pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia,
    reason="the batch-invariant torch router is registered for NVIDIA only",
)


def original_router(logits, bias, top_k, num_real, scale, indices_dtype):
    # Frozen arithmetic from the old runtime torch_router_topk. Do not call the
    # extracted implementation here: this is the compatibility reference.
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    _, ids = torch.topk(probs + bias, k=top_k, dim=-1, sorted=True)
    weights = probs.gather(1, ids) * scale
    ids = ids.masked_fill(ids >= num_real, -1).to(indices_dtype)
    return weights, ids


def bits(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def rows(experts, dtype):
    g = torch.Generator(device="cuda").manual_seed(791)
    x = torch.randn(5, experts, generator=g, device="cuda", dtype=torch.float32).to(
        dtype
    )
    x[0].zero_()  # all candidates tie, including the top-k cutoff
    x[1] = torch.arange(experts, device="cuda").remainder(3).to(dtype)  # partial ties
    x[2].fill_(1)
    x[2, ::2] = torch.nextafter(x[2, ::2], torch.full_like(x[2, ::2], 2))
    x[3] = (x[3] * 2).round() / 2
    x[4].fill_(-1000)
    x[4, :4] = 1000  # saturated probabilities and zero-probability ties
    return x


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("experts,top_k", [(18, 3), (416, 12), (512, 12), (2048, 12)])
def test_original_longcat_router_and_planned_router_are_bitwise(dtype, experts, top_k):
    assert torch.cuda.is_available(), "real GPU routing acceptance requires CUDA"
    x = rows(experts, dtype)
    plan = plan_softmax_bias_topk(dtype=dtype, experts=experts, top_k=top_k)
    noise = torch.randn(1024, experts, device="cuda", dtype=dtype)
    for biased in (False, True):
        bias = torch.zeros(experts, device="cuda", dtype=torch.float32)
        if biased:
            bias[-2:] = 2  # identity candidates win; weights must stay unbiased
        for num_real in (experts, experts - 2):
            for index_dtype in (torch.int32, torch.int64):
                expected = original_router(x, bias, top_k, num_real, 2.5, index_dtype)
                for offset in (0, 1, 31, 32, 33, 129, 1024):
                    batch = torch.cat((noise[:offset], x))
                    legacy = torch_router_topk(
                        batch, bias, top_k, num_real, 2.5, index_dtype
                    )
                    planned = softmax_bias_topk(
                        batch,
                        bias,
                        top_k=top_k,
                        num_real_experts=num_real,
                        routed_scaling_factor=2.5,
                        indices_dtype=index_dtype,
                        plan=plan,
                    )
                    frozen = original_router(
                        batch, bias, top_k, num_real, 2.5, index_dtype
                    )
                    for actual, old, reference in zip(legacy, frozen, planned):
                        bits(actual, old)  # LongCat behavior, same input batch
                        bits(reference, old)
                    for actual, baseline in zip(planned, expected):
                        bits(actual[offset:], baseline)
                for row in range(len(x)):
                    single = softmax_bias_topk(
                        x[row : row + 1],
                        bias,
                        top_k=top_k,
                        num_real_experts=num_real,
                        routed_scaling_factor=2.5,
                        indices_dtype=index_dtype,
                        plan=plan,
                    )
                    for actual, baseline in zip(single, expected):
                        bits(actual, baseline[row : row + 1])
                order = torch.arange(len(x) - 1, -1, -1, device="cuda")
                reordered = softmax_bias_topk(
                    x[order],
                    bias,
                    top_k=top_k,
                    num_real_experts=num_real,
                    routed_scaling_factor=2.5,
                    indices_dtype=index_dtype,
                    plan=plan,
                )
                for actual, baseline in zip(reordered, expected):
                    bits(actual, baseline[order])


def test_tied_router_cuda_graph_updates_logits_and_bias():
    x = rows(416, torch.float32)
    bias = torch.zeros(416, device="cuda")
    plan = plan_softmax_bias_topk(dtype=x.dtype, experts=416, top_k=12)

    def run():
        return softmax_bias_topk(
            x,
            bias,
            top_k=12,
            num_real_experts=384,
            routed_scaling_factor=6,
            indices_dtype=torch.int32,
            plan=plan,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    for shift in (0, 1, 17):
        x.copy_(x.roll(shift, 1))
        bias.zero_()
        bias[shift : shift + 2] = 2
        graph.replay()
        torch.cuda.synchronize()
        expected = original_router(x, bias, 12, 384, 6, torch.int32)
        for a, b in zip(output, expected):
            bits(a, b)


def test_router_plan_refuses_unvalidated_expert_width():
    with pytest.raises(NoKernelFoundError):
        plan_softmax_bias_topk(dtype=torch.float32, experts=4096, top_k=12)
