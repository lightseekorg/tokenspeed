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

"""Correctness coverage for fused ordinary softmax top-k routing."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.moe.softmax_topk import moe_softmax_topk
from tokenspeed_kernel.ops.moe.triton.softmax_topk import triton_softmax_topk
from tokenspeed_kernel.platform import Platform


def test_torch_reference_runs_on_cpu():
    # FP64 is intentionally outside the fused signature and exercises the
    # portable fallback while preserving the previous runtime behavior.
    logits = torch.tensor([[0.0, 2.0, 1.0, -1.0]], dtype=torch.float64)
    weights, ids = moe_softmax_topk(
        logits,
        2,
        renormalize=True,
        routed_scaling_factor=2.0,
    )
    assert torch.equal(ids, torch.tensor([[1, 2]], dtype=torch.int64))
    torch.testing.assert_close(
        weights,
        torch.softmax(torch.tensor([[2.0, 1.0]]), dim=-1) * 2.0,
    )


needs_nvidia = pytest.mark.skipif(
    not torch.cuda.is_available() or not Platform.get().is_nvidia,
    reason="fused softmax-topk needs NVIDIA CUDA",
)


def _assert_valid_topk(
    logits: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    topk: int,
    renormalize: bool,
    scale: float,
) -> None:
    assert weights.dtype == torch.float32
    assert ids.dtype == torch.int64
    assert weights.shape == ids.shape == (logits.shape[0], topk)
    assert torch.all((ids >= 0) & (ids < logits.shape[1]))
    assert torch.all(torch.sort(ids, dim=-1).values.diff(dim=-1) != 0)

    logits_fp32 = logits.float()
    selected_logits = logits_fp32.gather(1, ids)
    reference_values = torch.topk(logits_fp32, topk, dim=-1).values
    torch.testing.assert_close(
        selected_logits.sort(dim=-1, descending=True).values,
        reference_values,
        atol=0.0,
        rtol=0.0,
    )
    if renormalize:
        expected_weights = torch.softmax(selected_logits, dim=-1)
    else:
        expected_weights = torch.softmax(logits_fp32, dim=-1).gather(1, ids)
    torch.testing.assert_close(
        weights,
        expected_weights * scale,
        atol=2e-6,
        rtol=2e-5,
    )


@needs_nvidia
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize(
    "tokens,experts,topk,dtype",
    [
        (1, 256, 8, torch.bfloat16),  # Qwen3.5 decode
        (8, 256, 8, torch.bfloat16),
        (64, 256, 8, torch.float16),
        (7, 16, 3, torch.float32),  # non-power-of-two top-k
        (2, 896, 16, torch.float32),
    ],
)
def test_triton_matches_softmax_topk(
    tokens: int,
    experts: int,
    topk: int,
    dtype: torch.dtype,
    renormalize: bool,
):
    torch.manual_seed(tokens * 1000 + experts)
    logits = torch.randn(tokens, experts, device="cuda", dtype=dtype)
    scale = 2.5
    weights, ids = triton_softmax_topk(
        router_logits=logits,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=scale,
    )
    _assert_valid_topk(logits, weights, ids, topk, renormalize, scale)


@needs_nvidia
def test_lowest_id_tie_break_and_strided_rows():
    storage = torch.zeros((6, 272), dtype=torch.bfloat16, device="cuda")
    logits = storage[::2, :256]
    assert not logits.is_contiguous() and logits.stride(1) == 1
    weights, ids = moe_softmax_topk(logits, 8, renormalize=True)
    expected_ids = torch.arange(8, dtype=torch.int64, device="cuda")
    assert torch.equal(ids, expected_ids.expand(3, 8))
    torch.testing.assert_close(weights, torch.full_like(weights, 1.0 / 8.0))


@needs_nvidia
def test_public_entry_point_is_cuda_graph_capturable():
    logits = torch.randn(1, 256, dtype=torch.bfloat16, device="cuda")
    # Compile before capture; production warms the kernels before capturing a
    # decode graph as well.
    moe_softmax_topk(logits, 8, renormalize=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        weights, ids = moe_softmax_topk(logits, 8, renormalize=True)
    graph.replay()
    torch.cuda.synchronize()
    _assert_valid_topk(logits, weights, ids, 8, True, 1.0)


@needs_nvidia
def test_empty_batch():
    logits = torch.empty((0, 256), dtype=torch.bfloat16, device="cuda")
    weights, ids = moe_softmax_topk(logits, 8)
    assert weights.shape == ids.shape == (0, 8)
    assert weights.dtype == torch.float32 and ids.dtype == torch.int64
