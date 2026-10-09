# Copyright (c) 2026 LightSeek Foundation

"""Correctness coverage for gfx1250 biased-sigmoid top-k routing."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.moe import moe_topk
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from utils import assert_no_triton_compile, is_cdna5

if not is_cdna5():
    pytest.skip(
        "AMD CDNA5 is required for gfx1250 sigmoid-bias top-k tests",
        allow_module_level=True,
    )

from tokenspeed_kernel_amd.ops.gfx1250.moe.sigmoid_bias_topk import (  # noqa: E402
    gluon_sigmoid_bias_topk_route_gfx1250,
    launch_gluon_sigmoid_bias_topk_route_gfx1250,
)


def _route(
    logits: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    *,
    normalize: bool = True,
    scale: float = 1.0,
    weights_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    return launch_gluon_sigmoid_bias_topk_route_gfx1250(
        router_logits=logits,
        correction_bias=bias,
        topk=topk,
        routed_scaling_factor=scale,
        normalize_topk_weights=normalize,
        weights_dtype=weights_dtype,
    )


def _assert_matches_fp32_reference(
    logits: torch.Tensor,
    bias: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    topk: int,
    normalize: bool,
    scale: float,
) -> None:
    assert ids.dtype == torch.int32
    assert weights.shape == ids.shape == (logits.shape[0], topk)
    selected = ids.long()
    assert torch.all((selected >= 0) & (selected < logits.shape[1]))
    assert torch.all(selected.sort(dim=-1).values.diff(dim=-1) != 0)
    scores = logits.float().sigmoid()
    choice = scores + bias.float()
    # Compare selected scores, not ids: near-ties may order differently.
    torch.testing.assert_close(
        choice.gather(1, selected).sort(dim=-1, descending=True).values,
        torch.topk(choice, topk, dim=-1).values,
        rtol=0,
        atol=1e-6,
    )
    expected_weights = scores.gather(1, selected)
    if normalize:
        expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(
        weights.float(),
        expected_weights * scale,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "tokens,experts,topk,dtype,normalize,scale",
    [
        (1, 256, 8, torch.bfloat16, True, 2.5),  # GLM-5.3 draft decode
        (4, 256, 8, torch.bfloat16, True, 2.5),  # GLM-5.3 verify decode
        (8192, 256, 8, torch.bfloat16, True, 2.5),  # GLM-5.3 prefill chunk
        (4, 256, 8, torch.bfloat16, False, 1.0),
        (7, 160, 6, torch.bfloat16, True, 2.5),  # non-power-of-two experts and top-k
    ],
)
def test_matches_fp32_reference(
    tokens: int,
    experts: int,
    topk: int,
    dtype: torch.dtype,
    normalize: bool,
    scale: float,
) -> None:
    gen = torch.Generator(device="cuda").manual_seed(tokens * 1000 + experts)
    logits = torch.randn(tokens, experts, device="cuda", generator=gen).to(dtype)
    bias = torch.randn(experts, device="cuda", generator=gen) * 0.05
    weights, ids = _route(logits, bias, topk, normalize=normalize, scale=scale)
    _assert_matches_fp32_reference(logits, bias, weights, ids, topk, normalize, scale)


def test_reduced_precision_weights_and_strided_rows() -> None:
    gen = torch.Generator(device="cuda").manual_seed(5)
    storage = torch.randn(10, 272, device="cuda", generator=gen).bfloat16()
    logits = storage[::2, :256]
    assert not logits.is_contiguous() and logits.stride(1) == 1
    bias = torch.randn(256, device="cuda", generator=gen) * 0.05
    weights, ids = _route(logits, bias, 8, scale=2.5, weights_dtype=torch.bfloat16)
    assert weights.dtype == torch.bfloat16
    reference_weights, reference_ids = _route(logits.contiguous(), bias, 8, scale=2.5)
    torch.testing.assert_close(ids, reference_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights.float(), reference_weights, rtol=2**-7, atol=0)


def test_lowest_id_tie_break_and_nan_scores() -> None:
    logits = torch.zeros(2, 256, device="cuda", dtype=torch.bfloat16)
    logits[1, :4] = float("nan")
    bias = torch.zeros(256, device="cuda")
    weights, ids = _route(logits, bias, 8)
    expected = torch.arange(8, device="cuda", dtype=torch.int32)
    torch.testing.assert_close(ids[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(ids[1], expected + 4, rtol=0, atol=0)
    torch.testing.assert_close(weights, torch.full_like(weights, 1.0 / 8.0))


def test_public_entry_point_selects_and_captures() -> None:
    selected = select_kernel(
        "moe",
        "sigmoid_bias_topk",
        format_signature(router_logits=dense_tensor_format(torch.bfloat16)),
        traits={"tokens": 4, "experts": 256, "topk": 8},
    )
    assert selected.name == "gluon_sigmoid_bias_topk_route_gfx1250"

    gen = torch.Generator(device="cuda").manual_seed(17)
    logits = torch.randn(4, 256, device="cuda", generator=gen).bfloat16()
    bias = torch.randn(256, device="cuda", generator=gen) * 0.05

    def route() -> tuple[torch.Tensor, torch.Tensor]:
        return moe_topk(
            logits,
            8,
            score_function="sigmoid",
            selection_method="topk",
            renormalize=True,
            routed_scaling_factor=2.5,
            correction_bias=bias,
        )

    route()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        weights, ids = route()
    logits.copy_(torch.randn(4, 256, device="cuda", generator=gen).bfloat16())
    graph.replay()
    torch.cuda.synchronize()
    _assert_matches_fp32_reference(logits, bias, weights, ids, 8, True, 2.5)


def test_token_count_does_not_recompile() -> None:
    bias = torch.randn(256, device="cuda") * 0.05
    logits = torch.randn(1024, 256, device="cuda").bfloat16()
    _route(logits[:1], bias, 8, scale=2.5)
    with assert_no_triton_compile(gluon_sigmoid_bias_topk_route_gfx1250):
        for tokens in (2, 4, 16, 17, 48, 64, 130, 255, 512, 1000):
            _route(logits[:tokens], bias, 8, scale=2.5)
