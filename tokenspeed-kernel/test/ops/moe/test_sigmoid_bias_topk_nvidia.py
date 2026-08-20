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

"""NVIDIA sigmoid_bias_topk: the fused minimax adapter must match the torch
reference (same expert set, same weights) and win selection on NVIDIA."""

import pytest
import torch
from tokenspeed_kernel.ops.moe.sigmoid_topk import (
    moe_sigmoid_bias_topk,
    torch_sigmoid_bias_topk,
    triton_minimax_sigmoid_bias_topk,
)
from tokenspeed_kernel.platform import Platform

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if not Platform.get().is_nvidia:
    pytest.skip("NVIDIA-only registration under test", allow_module_level=True)


@pytest.mark.parametrize("tokens", [1, 3, 8, 64])
@pytest.mark.parametrize("experts,topk", [(896, 16), (256, 8)])
@pytest.mark.parametrize("normalize", [True, False])
def test_minimax_adapter_matches_torch(tokens, experts, topk, normalize):
    torch.manual_seed(tokens * experts)
    logits = torch.randn(tokens, experts, dtype=torch.float32, device="cuda")
    bias = torch.randn(experts, dtype=torch.float32, device="cuda")
    kwargs = dict(
        router_logits=logits,
        correction_bias=bias,
        topk=topk,
        routed_scaling_factor=2.5,
        normalize_topk_weights=normalize,
    )
    ref_w, ref_i = torch_sigmoid_bias_topk(**kwargs)
    got_w, got_i = triton_minimax_sigmoid_bias_topk(**kwargs)
    assert got_w.dtype == torch.float32 and got_i.dtype == torch.int32
    for row in range(tokens):
        ref_map = dict(zip(ref_i[row].tolist(), ref_w[row].tolist(), strict=True))
        got_map = dict(zip(got_i[row].tolist(), got_w[row].tolist(), strict=True))
        assert set(ref_map) == set(got_map)
        for expert_id, ref_weight in ref_map.items():
            assert got_map[expert_id] == pytest.approx(ref_weight, abs=1e-4)


def test_entry_point_selects_fused_kernel():
    """The public entry point must not fall back to the multi-launch torch
    reference on NVIDIA."""
    logits = torch.randn(2, 896, dtype=torch.float32, device="cuda")
    bias = torch.randn(896, dtype=torch.float32, device="cuda")
    ref_w, ref_i = triton_minimax_sigmoid_bias_topk(
        router_logits=logits,
        correction_bias=bias,
        topk=16,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
    )
    got_w, got_i = moe_sigmoid_bias_topk(
        logits, bias, 16, routed_scaling_factor=1.0, normalize_topk_weights=True
    )
    torch.testing.assert_close(got_w, ref_w)
    assert torch.equal(got_i, ref_i)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("scale", [1.0, 2.5])
@pytest.mark.parametrize("tokens", [1, 2, 16, 128])
def test_kimi3_shape_uses_packed_key_kernel(tokens, normalize, scale):
    """The packed-key row kernel is exact and CUDA-graph capturable."""
    torch.manual_seed(7)
    logits = (torch.randn(tokens, 896, device="cuda") * 0.2).float()
    bias = (torch.randn(896, device="cuda") * 0.01).float()
    scores = logits.sigmoid()
    expected_ids = torch.topk(
        scores + bias.unsqueeze(0), 16, dim=-1, sorted=False
    ).indices
    expected_weights = scores.gather(1, expected_ids)
    if normalize:
        expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
    expected_weights *= scale

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        weights, ids = moe_sigmoid_bias_topk(
            logits,
            bias,
            16,
            routed_scaling_factor=scale,
            normalize_topk_weights=normalize,
        )
    graph.replay()
    torch.cuda.synchronize()

    assert weights.dtype == torch.float32 and ids.dtype == torch.int32
    for row in range(tokens):
        assert set(ids[row].tolist()) == set(expected_ids[row].tolist())
        expected_by_id = dict(
            zip(
                expected_ids[row].tolist(),
                expected_weights[row].tolist(),
                strict=True,
            )
        )
        for expert_id, weight in zip(
            ids[row].tolist(), weights[row].tolist(), strict=True
        ):
            assert weight == pytest.approx(expected_by_id[expert_id], abs=1e-5)


@pytest.mark.parametrize("tokens", [1, 2])
@pytest.mark.parametrize("map_dtype", [torch.int32, torch.int64])
def test_static_dispatch_map_and_weights_dtype(tokens, map_dtype):
    """The optional logical->physical map must translate the selected ids on
    the lean decode kernel, and bf16 weight output must match the fp32 path."""
    torch.manual_seed(11)
    logits = torch.randn(tokens, 896, dtype=torch.float32, device="cuda")
    bias = torch.randn(896, dtype=torch.float32, device="cuda")
    dispatch = torch.randperm(896, dtype=map_dtype, device="cuda")

    ref_w, ref_i = moe_sigmoid_bias_topk(
        logits, bias, 16, routed_scaling_factor=1.0, normalize_topk_weights=True
    )
    got_w, got_i = moe_sigmoid_bias_topk(
        logits,
        bias,
        16,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
        logical_to_physical_map=dispatch,
        weights_dtype=torch.bfloat16,
    )
    assert got_w.dtype == torch.bfloat16 and got_i.dtype == torch.int32
    assert torch.equal(dispatch[ref_i.long()].to(torch.int32), got_i)
    torch.testing.assert_close(got_w.float(), ref_w, atol=8e-3, rtol=8e-3)
