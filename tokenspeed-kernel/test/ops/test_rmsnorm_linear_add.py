# Copyright (c) 2026 LightSeek Foundation

from __future__ import annotations

import tokenspeed_kernel
import torch


def test_rmsnorm_linear_add_composed_matches_torch() -> None:
    torch.manual_seed(19)
    hidden = torch.randn(2, 6, dtype=torch.bfloat16)
    norm_weight = torch.randn(6, dtype=torch.bfloat16)
    linear_weight = torch.randn(9, 6, dtype=torch.bfloat16)
    addend_a = torch.randn(2, 9, dtype=torch.bfloat16)
    addend_b = torch.randn(2, 9, dtype=torch.bfloat16)

    source = hidden.float()
    normalized = (
        source
        * torch.rsqrt(source.square().mean(dim=-1, keepdim=True) + 1e-5)
        * norm_weight.float()
    ).to(hidden.dtype)
    projected = torch.nn.functional.linear(normalized, linear_weight)
    expected = (projected.float() + addend_a.float() + addend_b.float()).to(
        hidden.dtype
    )

    actual = tokenspeed_kernel.rmsnorm_linear_add(
        hidden,
        norm_weight,
        linear_weight,
        addend_a,
        addend_b,
        eps=1e-5,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
