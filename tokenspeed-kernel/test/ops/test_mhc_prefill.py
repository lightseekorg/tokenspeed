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
import torch.nn.functional as F
from tokenspeed_kernel.ops.mhc.triton import triton_mhc_pre

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, hc_mult, hidden_size = residual.shape
    flat = residual.float().view(num_tokens, hc_mult * hidden_size)
    inv_rms = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + rms_eps)
    mixes = F.linear(flat, fn) * inv_rms
    pre_raw, post_raw, comb_raw = torch.split(
        mixes, [hc_mult, hc_mult, hc_mult * hc_mult], dim=-1
    )
    pre = torch.sigmoid(pre_raw * hc_scale[0] + hc_base[:hc_mult]) + hc_eps
    post = torch.sigmoid(post_raw * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]) * 2.0
    comb = torch.softmax(
        comb_raw.view(num_tokens, hc_mult, hc_mult) * hc_scale[2]
        + hc_base[2 * hc_mult :].view(1, hc_mult, hc_mult),
        dim=-1,
    )
    comb = comb + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(1, sinkhorn_iters):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    layer_input = (pre.unsqueeze(-1) * residual.float()).sum(dim=1)
    return layer_input.to(torch.bfloat16), post.unsqueeze(-1), comb


def test_tiled_hc4_prefill_matches_reference() -> None:
    generator = torch.Generator(device="cuda").manual_seed(123)
    residual = torch.randn(
        (257, 4, 64), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    fn = torch.randn((24, 256), device="cuda", dtype=torch.float32, generator=generator)
    hc_scale = torch.tensor([0.7, 1.1, 0.5], device="cuda", dtype=torch.float32)
    hc_base = torch.randn(24, device="cuda", dtype=torch.float32, generator=generator)
    args = (residual, fn, hc_scale, hc_base, 1e-6, 1e-5, 3)

    actual = triton_mhc_pre(*args, norm_weight=None, norm_eps=None)
    expected = _reference(*args)

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_tensor.float(), expected_tensor.float(), rtol=2e-2, atol=2e-2
        )
