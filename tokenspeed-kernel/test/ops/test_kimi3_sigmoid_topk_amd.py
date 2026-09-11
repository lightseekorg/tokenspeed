from __future__ import annotations

import pytest
import tokenspeed_kernel
import torch
from utils import is_cdna4, is_cdna5

if not (is_cdna4() or is_cdna5()):
    pytest.skip(
        "AMD CDNA4/CDNA5 is required for Kimi K3 sigmoid-bias top-k tests",
        allow_module_level=True,
    )


@pytest.mark.skipif(not is_cdna4(), reason="gfx950 decode routing is CDNA4")
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_kimi3_sigmoid_bias_topk_is_exact_and_captures(
    normalize: bool,
    scale: float,
) -> None:
    torch.manual_seed(7)
    logits = (torch.randn(1, 896, device="cuda") * 0.2).float()
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
        weights, ids = tokenspeed_kernel.moe_sigmoid_bias_topk(
            logits,
            bias,
            16,
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
    assert actual_by_id == expected_by_id


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
    expected_weights, expected_ids = tokenspeed_kernel.moe_sigmoid_bias_topk(
        logits,
        bias,
        16,
        routed_scaling_factor=scale,
        normalize_topk_weights=normalize,
        solution="torch",
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        weights, ids = tokenspeed_kernel.moe_sigmoid_bias_topk(
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
    expected_weights, expected_ids = tokenspeed_kernel.moe_sigmoid_bias_topk(
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
