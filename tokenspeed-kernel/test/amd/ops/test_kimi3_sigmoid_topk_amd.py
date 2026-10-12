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

"""Sigmoid-bias top-k routing on AMD: Kimi K3 decode and prefill shapes."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.moe import moe_topk as kernel_moe_topk
from tokenspeed_kernel.ops.moe.sigmoid_topk import (
    _gluon_eligible,
)
from tokenspeed_kernel.ops.moe.sigmoid_topk import (
    _moe_sigmoid_bias_topk as moe_sigmoid_bias_topk,
)
from utils import is_cdna4, is_cdna5


def _sigmoid_topk(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    routed_scaling_factor: float = 1.0,
    normalize_topk_weights: bool = True,
    logical_to_physical_map: torch.Tensor | None = None,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return kernel_moe_topk(
        router_logits,
        topk,
        score_function="sigmoid",
        selection_method="topk",
        renormalize=normalize_topk_weights,
        routed_scaling_factor=routed_scaling_factor,
        correction_bias=correction_bias,
        logical_to_physical_map=logical_to_physical_map,
        solution=solution,
    )


if not (is_cdna4() or is_cdna5()):
    pytest.skip(
        "AMD CDNA4/CDNA5 is required for Kimi K3 sigmoid-bias top-k tests",
        allow_module_level=True,
    )


@pytest.mark.skipif(not is_cdna4(), reason="gfx950 decode routing is CDNA4")
@pytest.mark.parametrize("rows", [1, 8, 16, 32, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_kimi3_sigmoid_bias_topk_matches_torch_and_captures(
    rows: int,
    normalize: bool,
    scale: float,
) -> None:
    torch.manual_seed(7)
    logits = (torch.randn(rows, 896, device="cuda") * 0.2).float()
    bias = (torch.randn(896, device="cuda") * 0.01).float()
    scores = logits.sigmoid()
    expected_ids = torch.topk(
        scores + bias.unsqueeze(0),
        16,
        dim=-1,
        sorted=False,
    ).indices
    expected_weights = scores.gather(1, expected_ids)
    if normalize:
        expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
    expected_weights *= scale

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        weights, ids = _sigmoid_topk(
            logits,
            bias,
            16,
            routed_scaling_factor=scale,
            normalize_topk_weights=normalize,
        )
    graph.replay()
    torch.cuda.synchronize()

    expected_sorted, expected_order = expected_ids.sort(dim=1)
    actual_sorted, actual_order = ids.sort(dim=1)
    torch.testing.assert_close(actual_sorted.long(), expected_sorted, rtol=0, atol=0)
    torch.testing.assert_close(
        weights.gather(1, actual_order),
        expected_weights.gather(1, expected_order),
        rtol=0 if rows == 1 else 2e-7,
        atol=0 if rows == 1 else 2e-7,
    )


@pytest.mark.skipif(not is_cdna5(), reason="gfx1250 prefill routing is CDNA5")
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_prefill_sigmoid_bias_topk_matches_torch_and_captures(
    normalize: bool,
    scale: float,
) -> None:
    torch.manual_seed(16)
    logits = (torch.randn(16, 896, device="cuda") * 0.2).float()
    bias = (torch.randn(896, device="cuda") * 0.01).float()
    expected_weights, expected_ids = _sigmoid_topk(
        logits,
        bias,
        16,
        routed_scaling_factor=scale,
        normalize_topk_weights=normalize,
        solution="torch",
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        weights, ids = _sigmoid_topk(
            logits,
            bias,
            16,
            routed_scaling_factor=scale,
            normalize_topk_weights=normalize,
        )
    graph.replay()
    torch.cuda.synchronize()

    expected_sorted, expected_order = expected_ids.sort(dim=1)
    actual_sorted, actual_order = ids.sort(dim=1)
    torch.testing.assert_close(actual_sorted, expected_sorted, rtol=0, atol=0)
    torch.testing.assert_close(
        weights.gather(1, actual_order),
        expected_weights.gather(1, expected_order),
        rtol=2e-7,
        atol=2e-7,
    )

    logits.copy_(torch.randn_like(logits) * 0.3)
    bias.copy_(torch.randn_like(bias) * 0.02)
    expected_weights, expected_ids = _sigmoid_topk(
        logits,
        bias,
        16,
        routed_scaling_factor=scale,
        normalize_topk_weights=normalize,
        solution="torch",
    )
    graph.replay()
    torch.cuda.synchronize()
    expected_sorted, expected_order = expected_ids.sort(dim=1)
    actual_sorted, actual_order = ids.sort(dim=1)
    torch.testing.assert_close(actual_sorted, expected_sorted, rtol=0, atol=0)
    torch.testing.assert_close(
        weights.gather(1, actual_order),
        expected_weights.gather(1, expected_order),
        rtol=2e-7,
        atol=2e-7,
    )


@pytest.mark.skipif(not is_cdna4(), reason="gfx950 decode routing is CDNA4")
@pytest.mark.parametrize("experts,topk", [(256, 8), (1024, 16)])
def test_decode_sigmoid_bias_topk_generalizes_expert_geometry(
    experts: int,
    topk: int,
) -> None:
    normalize = True
    scale = 1.0
    torch.manual_seed(7)
    logits = (torch.randn(1, experts, device="cuda") * 0.2).float()
    bias = (torch.randn(experts, device="cuda") * 0.01).float()
    scores = logits.sigmoid()
    expected_ids = torch.topk(
        scores + bias.unsqueeze(0),
        topk,
        dim=-1,
        sorted=False,
    ).indices
    expected_weights = scores.gather(1, expected_ids)
    if normalize:
        expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
    expected_weights *= scale

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        weights, ids = _sigmoid_topk(
            logits,
            bias,
            topk,
            routed_scaling_factor=scale,
            normalize_topk_weights=normalize,
        )
    graph.replay()
    torch.cuda.synchronize()

    assert set(ids[0].tolist()) == set(expected_ids[0].tolist())
    expected_by_id = {
        expert_id: weight
        for expert_id, weight in zip(
            expected_ids[0].tolist(),
            expected_weights[0].tolist(),
        )
    }
    actual_by_id = {
        expert_id: weight
        for expert_id, weight in zip(ids[0].tolist(), weights[0].tolist())
    }
    assert actual_by_id.keys() == expected_by_id.keys()
    selected_ids = sorted(actual_by_id)
    torch.testing.assert_close(
        torch.tensor([actual_by_id[expert_id] for expert_id in selected_ids]),
        torch.tensor([expected_by_id[expert_id] for expert_id in selected_ids]),
        rtol=2e-7,
        atol=2e-7,
    )


@pytest.mark.skipif(not is_cdna4(), reason="gfx950 decode routing is CDNA4")
def test_decode_sigmoid_bias_topk_fuses_logical_to_physical_map() -> None:
    torch.manual_seed(11)
    logits = (torch.randn(1, 896, device="cuda") * 0.2).float()
    bias = (torch.randn(896, device="cuda") * 0.01).float()
    logical_to_physical = torch.arange(
        895,
        -1,
        -1,
        device="cuda",
        dtype=torch.int32,
    )
    expected_weights, logical_ids = _sigmoid_topk(
        logits,
        bias,
        16,
    )
    expected_ids = logical_to_physical[logical_ids]

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        weights, ids = _sigmoid_topk(
            logits,
            bias,
            16,
            logical_to_physical_map=logical_to_physical,
        )
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)


@pytest.mark.skipif(not is_cdna4(), reason="gfx950 decode routing is CDNA4")
@pytest.mark.parametrize("tokens", [1, 2])
def test_decode_sigmoid_bias_topk_accepts_int64_map(tokens: int) -> None:
    torch.manual_seed(13)
    logits = torch.randn(tokens, 896, device="cuda", dtype=torch.float32)
    bias = torch.randn(896, device="cuda", dtype=torch.float32)
    logical_to_physical = torch.randperm(896, device="cuda", dtype=torch.int64)

    expected_weights, logical_ids = _sigmoid_topk(
        logits,
        bias,
        16,
    )
    weights, ids = _sigmoid_topk(
        logits,
        bias,
        16,
        logical_to_physical_map=logical_to_physical,
    )

    torch.testing.assert_close(ids, logical_to_physical[logical_ids.long()].int())
    torch.testing.assert_close(weights, expected_weights)


@pytest.mark.skipif(not is_cdna4(), reason="Gluon sigmoid top-k is gfx950-only")
@pytest.mark.parametrize("tokens", [1, 17, 8192])
def test_kimi_topk_prefill_matches_reference(tokens: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(41 + tokens)
    logits = torch.randn(tokens, 896, device="cuda", generator=generator)
    bias = torch.randn(896, device="cuda", generator=generator) * 0.1
    scores = logits.sigmoid()
    _, expected_ids = torch.topk(scores + bias, 16, dim=-1, sorted=True)
    expected_weights = scores.gather(1, expected_ids)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)

    actual_weights, actual_ids = moe_sigmoid_bias_topk(
        logits,
        bias,
        16,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
    )
    torch.testing.assert_close(actual_ids, expected_ids.to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=2e-6, atol=2e-7)


@pytest.mark.skipif(not is_cdna4(), reason="Gluon sigmoid top-k is gfx950-only")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_kimi_topk_prefill_scales_beyond_8k(dtype: torch.dtype) -> None:
    tokens = 16384
    generator = torch.Generator(device="cuda").manual_seed(73)
    logits = torch.randn(tokens, 896, device="cuda", dtype=dtype, generator=generator)
    bias = torch.randn(896, device="cuda", generator=generator) * 0.1
    scores = logits.float().sigmoid().to(dtype)
    _, expected_ids = torch.topk(scores.float() + bias, 16, dim=-1, sorted=True)
    expected_weights = scores.gather(1, expected_ids)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)

    assert _gluon_eligible(logits, bias, 16)
    actual_weights, actual_ids = moe_sigmoid_bias_topk(
        logits,
        bias,
        16,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
    )

    torch.testing.assert_close(actual_ids, expected_ids.to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(
        actual_weights,
        expected_weights.float(),
        rtol=5e-3,
        atol=5e-4,
    )


@pytest.mark.skipif(not is_cdna4(), reason="Gluon sigmoid top-k is gfx950-only")
def test_kimi_topk_prefill_ties_choose_smaller_expert_id() -> None:
    logits = torch.zeros(3, 896, device="cuda")
    bias = torch.zeros(896, device="cuda")
    weights, ids = moe_sigmoid_bias_topk(logits, bias, 16)
    expected_ids = torch.arange(16, device="cuda", dtype=torch.int32).expand(3, -1)
    assert torch.equal(ids, expected_ids)
    torch.testing.assert_close(weights, torch.full_like(weights, 1 / 16))
