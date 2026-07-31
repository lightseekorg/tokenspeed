# Copyright (c) 2026 LightSeek Foundation

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.gemm import moe_input_projections
from tokenspeed_kernel.ops.gemm.moe_input_projections import fused_weight_view

if not torch.cuda.is_available():
    pytest.skip("requires a GPU", allow_module_level=True)

HIDDEN = 1024
ROUTER_N = 256
ROUTED_N = 512
SHARED_N = 128
GATE_CLAMP = 2.5
UP_CLAMP = 1.0


def _concatenated_weights(
    device: str = "cuda",
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    torch.manual_seed(7)
    widths = (ROUTER_N, ROUTED_N, 2 * SHARED_N)
    fused = torch.randn(sum(widths), HIDDEN, dtype=torch.bfloat16, device=device) * 0.02
    offset = 0
    views = []
    for width in widths:
        views.append(fused.narrow(0, offset, width))
        offset += width
    return fused, views


def _reference(
    hidden: torch.Tensor, views: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    router = torch.nn.functional.linear(hidden.float(), views[0].float())
    routed = torch.nn.functional.linear(hidden, views[1])
    gate, up = torch.nn.functional.linear(hidden, views[2]).chunk(2, dim=-1)
    gated = (
        GATE_CLAMP * torch.tanh(gate.float() / GATE_CLAMP) * torch.sigmoid(gate.float())
    )
    shared = (gated * UP_CLAMP * torch.tanh(up.float() / UP_CLAMP)).to(hidden.dtype)
    return router, routed, shared


def test_fused_weight_view_requires_consecutive_rows() -> None:
    fused, views = _concatenated_weights()
    assert fused_weight_view(*views) is not None
    assert fused_weight_view(*views).shape == fused.shape
    assert fused_weight_view(views[1], views[0], views[2]) is None
    assert fused_weight_view(views[0], views[1], views[2].clone()) is None


@pytest.mark.parametrize("tokens", [1, 2, 7, 16, 33, 64, 129, 256, 512])
def test_fused_moe_input_projections_matches_composition(tokens: int) -> None:
    _, views = _concatenated_weights()
    hidden = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.05

    router, routed, shared = moe_input_projections(
        hidden,
        *views,
        gate_clamp=GATE_CLAMP,
        up_clamp=UP_CLAMP,
        override="triton_fused_moe_input_projections",
    )
    expected_router, expected_routed, expected_shared = _reference(hidden, views)

    # Expert selection reads the router in FP32, so a single GEMM over the
    # concatenated weight must not round those logits to the activation dtype.
    assert router.dtype == torch.float32
    torch.testing.assert_close(router, expected_router, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(routed, expected_routed, atol=8e-3, rtol=8e-3)
    torch.testing.assert_close(shared, expected_shared, atol=8e-3, rtol=8e-3)


def test_fused_moe_input_projections_without_up_clamp() -> None:
    _, views = _concatenated_weights()
    hidden = torch.randn(8, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.05

    _, _, shared = moe_input_projections(
        hidden,
        *views,
        gate_clamp=GATE_CLAMP,
        up_clamp=None,
        override="triton_fused_moe_input_projections",
    )

    gate, up = torch.nn.functional.linear(hidden, views[2]).chunk(2, dim=-1)
    expected = (
        GATE_CLAMP
        * torch.tanh(gate.float() / GATE_CLAMP)
        * torch.sigmoid(gate.float())
        * up.float()
    ).to(hidden.dtype)
    torch.testing.assert_close(shared, expected, atol=8e-3, rtol=8e-3)


def test_unfused_weights_do_not_select_the_fused_kernel() -> None:
    _, views = _concatenated_weights()
    separate = [view.clone() for view in views]
    hidden = torch.randn(4, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.05

    fused_result = moe_input_projections(
        hidden, *views, gate_clamp=GATE_CLAMP, up_clamp=UP_CLAMP
    )
    separate_result = moe_input_projections(
        hidden, *separate, gate_clamp=GATE_CLAMP, up_clamp=UP_CLAMP
    )
    for fused_tensor, separate_tensor in zip(fused_result, separate_result):
        torch.testing.assert_close(fused_tensor, separate_tensor, atol=8e-3, rtol=8e-3)
