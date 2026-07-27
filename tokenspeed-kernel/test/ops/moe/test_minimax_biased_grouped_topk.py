"""Fused-vs-reference tests for the biased grouped-topk router kernel.

Covers the original MiniMax envelope (<=256 experts, topk<=8) and the widened
Kimi-K3 envelope (896 experts, topk=16) with the degenerate single-group
config both models use.
"""

import pytest
import torch
from tokenspeed_kernel.thirdparty.triton import (
    _biased_grouped_topk_reference,
    minimax_biased_grouped_topk,
)

if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)


def _run_both(num_tokens, num_experts, topk, renormalize, scale, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    hidden = torch.empty(num_tokens, 8, device="cuda")  # only shape[0] is used
    logits = torch.randn(
        num_tokens, num_experts, device="cuda", generator=gen, dtype=torch.float32
    )
    bias = torch.randn(num_experts, device="cuda", generator=gen, dtype=torch.float32)

    fused_w, fused_i = minimax_biased_grouped_topk(
        hidden,
        logits,
        bias,
        topk=topk,
        renormalize=renormalize,
        num_expert_group=1,
        topk_group=1,
        routed_scaling_factor=scale,
    )
    ref_w, ref_i = _biased_grouped_topk_reference(
        hidden,
        logits,
        bias,
        topk=topk,
        renormalize=renormalize,
        num_expert_group=1,
        topk_group=1,
        routed_scaling_factor=scale,
    )
    return fused_w, fused_i, ref_w, ref_i


@pytest.mark.parametrize(
    "num_tokens,num_experts,topk",
    [
        (1, 256, 8),  # original MiniMax envelope
        (32, 256, 8),
        (1, 896, 16),  # Kimi-K3 envelope (widened)
        (32, 896, 16),
        (7, 896, 16),  # non-power-of-two token count
    ],
)
@pytest.mark.parametrize("renormalize", [True, False])
def test_fused_matches_reference(num_tokens, num_experts, topk, renormalize):
    fused_w, fused_i, ref_w, ref_i = _run_both(
        num_tokens, num_experts, topk, renormalize, scale=1.0, seed=1234
    )

    # Expert sets must match per token (order may differ: fused emits in
    # descending choice-score order, torch.topk is unsorted here).
    for t in range(num_tokens):
        assert set(fused_i[t].tolist()) == set(ref_i[t].tolist())

    # Weight for each chosen expert must match.
    for t in range(num_tokens):
        fused_map = dict(zip(fused_i[t].tolist(), fused_w[t].tolist()))
        ref_map = dict(zip(ref_i[t].tolist(), ref_w[t].tolist()))
        for e, w in ref_map.items():
            assert fused_map[e] == pytest.approx(w, rel=1e-5, abs=1e-6)


def test_routed_scaling_factor_applied():
    fused_w, _, ref_w, _ = _run_both(4, 896, 16, renormalize=True, scale=2.5, seed=99)
    assert fused_w.sum().item() == pytest.approx(ref_w.sum().item(), rel=1e-5)
    # renormalized weights times scale sum to scale per token
    assert fused_w.sum(dim=-1).cpu() == pytest.approx(torch.full((4,), 2.5), rel=1e-5)


@pytest.mark.parametrize("num_experts,topk", [(256, 8), (896, 16)])
def test_nan_padding_rows_stay_in_range(num_experts, topk):
    """Graph-padding rows can carry NaN logits; ids must stay in range and
    real rows must be unaffected (regression: OOB expert id 896 poisoned the
    downstream MoE kernel)."""
    gen = torch.Generator(device="cuda").manual_seed(7)
    logits = torch.randn(
        5, num_experts, device="cuda", generator=gen, dtype=torch.float32
    )
    logits[1] = float("nan")
    logits[3] = -float("inf")
    bias = torch.randn(num_experts, device="cuda", generator=gen)
    kw = dict(
        topk=topk,
        renormalize=True,
        num_expert_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
    )
    w, i = minimax_biased_grouped_topk(
        torch.empty(5, 8, device="cuda"), logits, bias, **kw
    )
    assert int(i.max()) < num_experts and int(i.min()) >= 0
    # real rows still match the reference
    ref_w, ref_i = _biased_grouped_topk_reference(
        torch.empty(5, 8, device="cuda"), logits, bias, **kw
    )
    for t in (0, 2, 4):
        assert set(i[t].tolist()) == set(ref_i[t].tolist())


def test_static_expert_map_applied_in_range():
    """EP static dispatch: ids are mapped logical->physical in-kernel; the
    OOB clamp must protect the map lookup, and mapped (physical) ids may
    legitimately exceed num_experts."""
    num_experts, topk = 896, 16
    gen = torch.Generator(device="cuda").manual_seed(3)
    logits = torch.randn(4, num_experts, device="cuda", generator=gen)
    logits[2] = float("nan")  # padding row exercises clamp + map together
    bias = torch.randn(num_experts, device="cuda", generator=gen)
    offset = 100  # physical = logical + 100 (mimics redundant-expert layout)
    l2p = torch.arange(offset, num_experts + offset, device="cuda", dtype=torch.int32)
    kw = dict(
        topk=topk,
        renormalize=True,
        num_expert_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
    )
    _, mapped = minimax_biased_grouped_topk(
        torch.empty(4, 8, device="cuda"),
        logits,
        bias,
        logical_to_physical_map=l2p,
        **kw
    )
    _, unmapped = minimax_biased_grouped_topk(
        torch.empty(4, 8, device="cuda"), logits, bias, **kw
    )
    assert torch.equal(mapped, unmapped + offset)
    assert int(unmapped.max()) < num_experts  # clamp held for the NaN row


def test_k3_shape_takes_fused_path():
    """896 experts / topk 16 must not silently fall back to the reference."""
    from unittest import mock

    import tokenspeed_kernel.thirdparty.triton as tt

    with mock.patch.object(
        tt, "_biased_grouped_topk_reference", side_effect=AssertionError("fell back")
    ):
        minimax_biased_grouped_topk(
            torch.empty(2, 8, device="cuda"),
            torch.randn(2, 896, device="cuda"),
            torch.randn(896, device="cuda"),
            topk=16,
            renormalize=True,
            num_expert_group=1,
            topk_group=1,
            routed_scaling_factor=1.0,
        )


@pytest.mark.parametrize("renormalize", [True, False])
def test_bf16_weights_output(renormalize):
    """weights_dtype=bf16 emits sidecar-ready weights: fp32 math in-kernel,
    one rounding at the store (matches .to(bf16) of the fp32 output)."""
    gen = torch.Generator(device="cuda").manual_seed(11)
    logits = torch.randn(4, 896, device="cuda", generator=gen)
    bias = torch.randn(896, device="cuda", generator=gen)
    kw = dict(
        topk=16,
        renormalize=renormalize,
        num_expert_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
    )
    w32, i32 = minimax_biased_grouped_topk(
        torch.empty(4, 8, device="cuda"), logits, bias, **kw
    )
    w16, i16 = minimax_biased_grouped_topk(
        torch.empty(4, 8, device="cuda"),
        logits,
        bias,
        weights_dtype=torch.bfloat16,
        **kw
    )
    assert w16.dtype == torch.bfloat16
    assert torch.equal(i32, i16)
    torch.testing.assert_close(w16, w32.to(torch.bfloat16))
