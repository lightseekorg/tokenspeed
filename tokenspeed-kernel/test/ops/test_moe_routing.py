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

from __future__ import annotations

import pytest
import torch

# TODO: cover the static expert_location_dispatch_info path; the MiniMax sibling
# kernel also leaves it untested.

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")


_NUM_EXPERTS = 256
_NUM_EXPERT_GROUP = 8
_TOPK_GROUP = 4
_TOPK = 8
_ROUTED_SCALING_FACTOR = 2.5


def _make_inputs(
    num_tokens: int,
    gating_dtype: torch.dtype,
    device: torch.device,
    seed: int = 0,
):
    torch.manual_seed(seed)
    hidden = torch.randn(num_tokens, 16, device=device, dtype=torch.bfloat16)
    gating = torch.randn(num_tokens, _NUM_EXPERTS, device=device, dtype=gating_dtype)
    bias = torch.randn(_NUM_EXPERTS, device=device, dtype=gating_dtype)
    return hidden, gating, bias


def _match_weights_by_id(
    triton_ids: torch.Tensor,
    triton_w: torch.Tensor,
    ref_ids: torch.Tensor,
    ref_w: torch.Tensor,
    routed_slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Permute triton weights to align with reference id order on routed slots."""
    aligned = torch.zeros_like(ref_w[:, :routed_slots])
    for row in range(triton_ids.shape[0]):
        for k in range(routed_slots):
            target = int(ref_ids[row, k].item())
            match = (triton_ids[row, :routed_slots] == target).nonzero(as_tuple=False)
            if match.numel() == 0:
                aligned[row, k] = float("nan")
            else:
                aligned[row, k] = triton_w[row, int(match[0].item())]
    return aligned, ref_w[:, :routed_slots]


@pytest.mark.parametrize("num_tokens", [1, 8, 32, 128])
@pytest.mark.parametrize("num_fused_shared_experts", [0, 1])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("apply_routed_scaling_factor_on_output", [True, False])
@pytest.mark.parametrize("with_padding", [False, True])
@pytest.mark.parametrize("gating_dtype", [torch.float32, torch.bfloat16])
def test_triton_biased_grouped_topk_matches_reference(
    num_tokens: int,
    num_fused_shared_experts: int,
    renormalize: bool,
    apply_routed_scaling_factor_on_output: bool,
    with_padding: bool,
    gating_dtype: torch.dtype,
) -> None:
    from tokenspeed_kernel.ops.moe.reference import biased_grouped_topk_gpu
    from tokenspeed_kernel.ops.moe.triton import triton_biased_grouped_topk

    if apply_routed_scaling_factor_on_output and not renormalize:
        pytest.skip("output scaling is only applied under renormalize=True")

    device = torch.device("cuda")
    hidden, gating, bias = _make_inputs(num_tokens, gating_dtype, device)

    num_token_non_padded = None
    if with_padding:
        if num_tokens < 2:
            pytest.skip("padding test requires num_tokens >= 2")
        num_token_non_padded = torch.tensor(
            num_tokens // 2, device=device, dtype=torch.int32
        )

    common_kwargs = dict(
        hidden_states=hidden,
        gating_output=gating,
        correction_bias=bias,
        topk=_TOPK,
        renormalize=renormalize,
        num_expert_group=_NUM_EXPERT_GROUP,
        topk_group=_TOPK_GROUP,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=_ROUTED_SCALING_FACTOR,
        num_token_non_padded=num_token_non_padded,
        expert_location_dispatch_info=None,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )

    torch.manual_seed(0)
    ref_w, ref_ids = biased_grouped_topk_gpu(**common_kwargs)
    torch.manual_seed(0)
    w, ids = triton_biased_grouped_topk(**common_kwargs)

    assert w.dtype == torch.float32
    assert ids.dtype == torch.int32
    assert w.shape == (num_tokens, _TOPK)
    assert ids.shape == (num_tokens, _TOPK)

    routed_slots = _TOPK - num_fused_shared_experts
    non_padded = (
        int(num_token_non_padded.item())
        if num_token_non_padded is not None
        else num_tokens
    )

    if non_padded > 0:
        ids_active = ids[:non_padded]
        ref_ids_active = ref_ids[:non_padded]
        w_active = w[:non_padded]
        ref_w_active = ref_w[:non_padded]

        # Routed slots: set equality (order is descending-by-score; ties may
        # reorder across implementations even if the set is identical).
        triton_set = ids_active[:, :routed_slots].sort(dim=-1).values
        ref_set = ref_ids_active[:, :routed_slots].sort(dim=-1).values
        torch.testing.assert_close(triton_set, ref_set)

        # Shared expert slot: only check the range, not the value (randint).
        if num_fused_shared_experts > 0:
            shared = ids_active[:, -1]
            assert torch.all(shared >= _NUM_EXPERTS)
            assert torch.all(shared < _NUM_EXPERTS + num_fused_shared_experts)

        aligned, ref_aligned = _match_weights_by_id(
            ids_active.cpu(),
            w_active.cpu(),
            ref_ids_active.cpu(),
            ref_w_active.cpu(),
            routed_slots,
        )
        if gating_dtype == torch.bfloat16:
            rtol, atol = 2e-3, 1e-3
        else:
            rtol, atol = 1e-4, 1e-5
        torch.testing.assert_close(aligned, ref_aligned, rtol=rtol, atol=atol)

        if num_fused_shared_experts > 0:
            # Shared-expert weight derivation must match the reference exactly
            # since both paths read the same routed weights.
            torch.testing.assert_close(
                w_active[:, -1],
                ref_w_active[:, -1],
                rtol=rtol if gating_dtype == torch.bfloat16 else 1e-5,
                atol=atol if gating_dtype == torch.bfloat16 else 1e-6,
            )

    if num_token_non_padded is not None:
        padded_ids = ids[non_padded:]
        assert torch.all(padded_ids == -1)


def test_triton_biased_grouped_topk_falls_back_when_traits_mismatch() -> None:
    from tokenspeed_kernel.ops.moe.triton import triton_biased_grouped_topk

    device = torch.device("cuda")
    hidden, gating, bias = _make_inputs(4, torch.float32, device)

    # topk != 8 should drop into the reference path without raising.
    w, ids = triton_biased_grouped_topk(
        hidden_states=hidden,
        gating_output=gating,
        correction_bias=bias,
        topk=4,
        renormalize=True,
        num_expert_group=_NUM_EXPERT_GROUP,
        topk_group=_TOPK_GROUP,
        num_fused_shared_experts=0,
        routed_scaling_factor=_ROUTED_SCALING_FACTOR,
    )
    assert ids.shape == (4, 4)
    assert w.shape == (4, 4)


def test_triton_biased_grouped_topk_zero_tokens() -> None:
    from tokenspeed_kernel.ops.moe.triton import triton_biased_grouped_topk

    device = torch.device("cuda")
    hidden = torch.empty(0, 16, device=device, dtype=torch.bfloat16)
    gating = torch.empty(0, _NUM_EXPERTS, device=device, dtype=torch.float32)
    bias = torch.randn(_NUM_EXPERTS, device=device, dtype=torch.float32)

    w, ids = triton_biased_grouped_topk(
        hidden_states=hidden,
        gating_output=gating,
        correction_bias=bias,
        topk=_TOPK,
        renormalize=True,
        num_expert_group=_NUM_EXPERT_GROUP,
        topk_group=_TOPK_GROUP,
        num_fused_shared_experts=1,
        routed_scaling_factor=_ROUTED_SCALING_FACTOR,
    )
    assert w.shape == (0, _TOPK)
    assert ids.shape == (0, _TOPK)
    assert w.dtype == torch.float32
    assert ids.dtype == torch.int32


def test_triton_biased_grouped_topk_bf16_falls_back_bitexact() -> None:
    """bf16 gating must round-trip through the reference and match it byte-for-byte."""
    from tokenspeed_kernel.ops.moe.reference import biased_grouped_topk_gpu
    from tokenspeed_kernel.ops.moe.triton import triton_biased_grouped_topk

    device = torch.device("cuda")
    hidden = torch.randn(16, 16, device=device, dtype=torch.bfloat16)
    gating = torch.randn(16, _NUM_EXPERTS, device=device, dtype=torch.bfloat16)
    bias = torch.randn(_NUM_EXPERTS, device=device, dtype=torch.bfloat16)
    kwargs = dict(
        hidden_states=hidden,
        gating_output=gating,
        correction_bias=bias,
        topk=_TOPK,
        renormalize=True,
        num_expert_group=_NUM_EXPERT_GROUP,
        topk_group=_TOPK_GROUP,
        num_fused_shared_experts=1,
        routed_scaling_factor=_ROUTED_SCALING_FACTOR,
    )
    torch.manual_seed(0)
    ref_w, ref_ids = biased_grouped_topk_gpu(**kwargs)
    torch.manual_seed(0)
    w, ids = triton_biased_grouped_topk(**kwargs)
    torch.testing.assert_close(w, ref_w, rtol=0, atol=0)
    torch.testing.assert_close(ids, ref_ids, rtol=0, atol=0)
