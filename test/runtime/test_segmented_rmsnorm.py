"""Tests for the engine-local segmented RMSNorm kernel."""

from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.layers.segmented_rmsnorm import segmented_rmsnorm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [1, 8, 127])
@pytest.mark.parametrize("num_segments", [1, 5])
@pytest.mark.parametrize("hidden_size", [128, 7168])
def test_segmented_rmsnorm_matches_independent_norms(
    dtype: torch.dtype,
    num_tokens: int,
    num_segments: int,
    hidden_size: int,
) -> None:
    eps = 1e-6
    x = torch.randn(
        num_tokens,
        num_segments,
        hidden_size,
        device="cuda",
        dtype=dtype,
    )
    weight = torch.randn(
        num_segments,
        hidden_size,
        device="cuda",
        dtype=torch.float32,
    )
    out = segmented_rmsnorm(x, weight, eps)
    x_float = x.float()
    ref = (
        x_float
        * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + eps)
        * weight
    ).to(dtype)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_segmented_rmsnorm_honors_out() -> None:
    x = torch.randn(7, 5, 7168, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(5, 7168, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    assert segmented_rmsnorm(x, weight, 1e-6, out=out) is out
