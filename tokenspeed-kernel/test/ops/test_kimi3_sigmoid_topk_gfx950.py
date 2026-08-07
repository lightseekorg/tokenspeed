from __future__ import annotations

import pytest
import tokenspeed_kernel
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("requires AMD CDNA4", allow_module_level=True)


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
