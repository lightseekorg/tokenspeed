# Copyright (c) 2026 LightSeek Foundation

"""Correctness tests for Kimi K3 prefill Gluon kernels."""

from __future__ import annotations

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 is required for Kimi K3 prefill Gluon tests",
        allow_module_level=True,
    )


from tokenspeed_kernel.ops.attn_res import (  # noqa: E402
    _specialized_shape_eligible,
    attn_res_fwd,
)
from tokenspeed_kernel.ops.moe import moe_sigmoid_bias_topk  # noqa: E402
from tokenspeed_kernel.ops.moe.sigmoid_topk import _gluon_eligible  # noqa: E402


def _attn_res_reference(
    layer: torch.Tensor,
    history: torch.Tensor,
    res_weight: torch.Tensor,
    score_weight: torch.Tensor,
    output_weight: torch.Tensor,
    valid_blocks: int,
) -> torch.Tensor:
    values = torch.cat((history[:, :valid_blocks], layer.unsqueeze(1)), dim=1).float()
    inverse_rms = torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
    logits = values.mul(inverse_rms) @ (score_weight * res_weight.float())
    mixed = torch.matmul(logits.softmax(-1).unsqueeze(1), values).squeeze(1)
    mixed = mixed.to(torch.bfloat16).float()
    return (
        mixed
        * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + 1e-6)
        * output_weight
    ).to(torch.bfloat16)


def test_attn_res_public_block_major_dispatch_matches_reference() -> None:
    tokens, valid_blocks = 256, 8
    generator = torch.Generator(device="cuda").manual_seed(91)
    layer = torch.randn(
        tokens, 7168, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    history = torch.randn(
        valid_blocks,
        tokens,
        7168,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    res_weight = torch.randn(
        7168, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    score_weight = torch.randn(7168, device="cuda", generator=generator)
    output_weight = torch.randn(7168, device="cuda", generator=generator)

    actual = attn_res_fwd(
        layer,
        history,
        res_weight,
        score_weight,
        eps=1e-6,
        out_norm_weight=output_weight,
    )
    expected = _attn_res_reference(
        layer,
        history.transpose(0, 1),
        res_weight,
        score_weight,
        output_weight,
        valid_blocks,
    )
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=1.6e-2)


def test_attn_res_large_prefill_dispatch_boundary() -> None:
    assert not _specialized_shape_eligible(0, 7168, 12, fused_output_norm=True)
    assert _specialized_shape_eligible(32768, 7168, 12, fused_output_norm=True)
    assert not _specialized_shape_eligible(32768, 7168, 12, fused_output_norm=False)
    assert not _specialized_shape_eligible(65537, 7168, 12, fused_output_norm=True)


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


def test_kimi_topk_prefill_ties_choose_smaller_expert_id() -> None:
    logits = torch.zeros(3, 896, device="cuda")
    bias = torch.zeros(896, device="cuda")
    weights, ids = moe_sigmoid_bias_topk(logits, bias, 16)
    expected_ids = torch.arange(16, device="cuda", dtype=torch.int32).expand(3, -1)
    assert torch.equal(ids, expected_ids)
    torch.testing.assert_close(weights, torch.full_like(weights, 1 / 16))
