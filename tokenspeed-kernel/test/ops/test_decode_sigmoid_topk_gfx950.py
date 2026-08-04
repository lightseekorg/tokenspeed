from __future__ import annotations

import pytest
import tokenspeed_kernel
import torch

if not torch.cuda.is_available():
    pytest.skip("requires a GPU", allow_module_level=True)

if torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0] != "gfx950":
    pytest.skip("requires gfx950", allow_module_level=True)


@pytest.mark.parametrize("experts,topk", [(256, 8), (896, 16), (1024, 16)])
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
        weights, ids = tokenspeed_kernel.moe_sigmoid_bias_topk(
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


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_decode_sigmoid_bias_topk_k3_numerics(
    normalize: bool,
    scale: float,
) -> None:
    torch.manual_seed(7)
    logits = (torch.randn(1, 896, device="cuda") * 0.2).float()
    bias = (torch.randn(896, device="cuda") * 0.01).float()
    expected_weights, expected_ids = tokenspeed_kernel.moe_sigmoid_bias_topk(
        logits,
        bias,
        16,
        routed_scaling_factor=scale,
        normalize_topk_weights=normalize,
        solution="torch",
    )
    weights, ids = tokenspeed_kernel.moe_sigmoid_bias_topk(
        logits,
        bias,
        16,
        routed_scaling_factor=scale,
        normalize_topk_weights=normalize,
    )
    assert set(ids[0].tolist()) == set(expected_ids[0].tolist())
    expected_by_id = dict(
        zip(expected_ids[0].tolist(), expected_weights[0].tolist(), strict=True)
    )
    actual_by_id = dict(zip(ids[0].tolist(), weights[0].tolist(), strict=True))
    assert actual_by_id == expected_by_id


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
    expected_weights, logical_ids = tokenspeed_kernel.moe_sigmoid_bias_topk(
        logits,
        bias,
        16,
    )
    expected_ids = logical_to_physical[logical_ids]

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        weights, ids = tokenspeed_kernel.moe_sigmoid_bias_topk(
            logits,
            bias,
            16,
            logical_to_physical_map=logical_to_physical,
        )
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [1, 2])
def test_decode_sigmoid_bias_topk_accepts_int64_map(tokens: int) -> None:
    torch.manual_seed(13)
    logits = torch.randn(tokens, 896, device="cuda", dtype=torch.float32)
    bias = torch.randn(896, device="cuda", dtype=torch.float32)
    logical_to_physical = torch.randperm(896, device="cuda", dtype=torch.int64)

    expected_weights, logical_ids = tokenspeed_kernel.moe_sigmoid_bias_topk(
        logits,
        bias,
        16,
    )
    weights, ids = tokenspeed_kernel.moe_sigmoid_bias_topk(
        logits,
        bias,
        16,
        logical_to_physical_map=logical_to_physical,
    )

    torch.testing.assert_close(ids, logical_to_physical[logical_ids.long()].int())
    torch.testing.assert_close(weights, expected_weights)
