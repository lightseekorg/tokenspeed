from __future__ import annotations

import pytest
import tokenspeed_kernel
import torch
from tokenspeed_kernel.ops.moe.triton import kimi3_sigmoid_topk
from utils import assert_no_triton_compile, is_cdna4, is_cdna5


def _sigmoid_topk(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    routed_scaling_factor: float = 1.0,
    normalize_topk_weights: bool = True,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return tokenspeed_kernel.moe_topk(
        router_logits,
        topk,
        score_function="sigmoid",
        selection_method="topk",
        renormalize=normalize_topk_weights,
        routed_scaling_factor=routed_scaling_factor,
        correction_bias=correction_bias,
        solution=solution,
    )


if not (is_cdna4() or is_cdna5()):
    pytest.skip(
        "AMD CDNA4/CDNA5 is required for Kimi K3 sigmoid-bias top-k tests",
        allow_module_level=True,
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


@pytest.mark.skipif(not is_cdna5(), reason="gfx1250 packed routing is CDNA5")
@pytest.mark.parametrize("mapped", [False, True], ids=["unmapped", "mapped"])
def test_packed_sigmoid_bias_topk_batch_size_does_not_recompile(
    mapped: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public 2..512-row packed route reuses each warmed specialization."""
    torch.manual_seed(23)
    logits = torch.randn(512, 896, device="cuda", dtype=torch.float32) * 0.2
    bias = torch.randn(896, device="cuda", dtype=torch.float32) * 0.01
    dispatch = torch.randperm(896, device="cuda", dtype=torch.int32) if mapped else None
    scores = logits.sigmoid()
    expected_ids = (scores + bias).topk(16, dim=-1).indices
    expected_weights = scores.gather(1, expected_ids)
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
    expected_weights *= 2.5
    if dispatch is not None:
        expected_ids = dispatch[expected_ids]
    expected_ids = expected_ids.to(torch.int32)

    calls = []
    packed = kimi3_sigmoid_topk.kimi3_sigmoid_bias_topk

    def spy(*args, **kwargs):
        # A mapped call must not fall back to unmapped routing plus a gather.
        assert kwargs["logical_to_physical_map"] is dispatch
        calls.append(args[0].shape[0])
        return packed(*args, **kwargs)

    monkeypatch.setattr(kimi3_sigmoid_topk, "kimi3_sigmoid_bias_topk", spy)

    def run(tokens: int):
        return tokenspeed_kernel.moe_topk(
            logits[:tokens],
            16,
            score_function="sigmoid",
            selection_method="topk",
            renormalize=True,
            routed_scaling_factor=2.5,
            correction_bias=bias,
            logical_to_physical_map=dispatch,
            topk_weights_dtype=torch.float32,
        )

    # Row count changes only the grid, not a kernel argument. All feature
    # choices and dtypes stay fixed, so each mapped/unmapped case warms once.
    run(16)
    torch.cuda.synchronize()
    token_counts = (2, 3, 15, 17, 255, 257, 511, 512)
    with assert_no_triton_compile(kimi3_sigmoid_topk._kimi3_sigmoid_bias_topk_kernel):
        results = [run(tokens) for tokens in token_counts]
        torch.cuda.synchronize()
    assert calls == [16, *token_counts]

    for tokens, (weights, ids) in zip(token_counts, results):
        expected_sorted, expected_order = expected_ids[:tokens].sort(dim=1)
        actual_sorted, actual_order = ids.sort(dim=1)
        torch.testing.assert_close(actual_sorted, expected_sorted, rtol=0, atol=0)
        torch.testing.assert_close(
            weights.gather(1, actual_order),
            expected_weights[:tokens].gather(1, expected_order),
            rtol=2e-7,
            atol=2e-7,
        )
