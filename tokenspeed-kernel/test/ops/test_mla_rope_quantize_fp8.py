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
from tokenspeed_kernel.ops.embedding.flashinfer import mla_rope_quantize_fp8
from tokenspeed_kernel.platform import current_platform


def _rope_reference(
    x: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    is_neox: bool,
) -> torch.Tensor:
    rope_dim = x.shape[-1]
    half = rope_dim // 2
    cos = cos_sin_cache.index_select(0, positions.long())[:, :half].float()
    sin = cos_sin_cache.index_select(0, positions.long())[:, half:rope_dim].float()
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    x = x.float()
    if is_neox:
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).flatten(-2)


def test_mla_rope_quantize_fp8_triton_matches_reference(device: str) -> None:
    if not current_platform().is_cdna4:
        pytest.skip("Triton MLA RoPE FP8 fallback is currently exercised on AMD CDNA4")

    torch.manual_seed(5)
    tokens = 4
    q_heads = 3
    k_heads = 1
    nope_dim = 128
    rope_dim = 64
    q_full = torch.randn(
        (tokens, q_heads, nope_dim + rope_dim), device=device, dtype=torch.bfloat16
    )
    k_full = torch.randn(
        (tokens, k_heads, nope_dim + rope_dim), device=device, dtype=torch.bfloat16
    )
    q_nope = q_full[..., :nope_dim]
    q_rope = q_full[..., nope_dim:]
    k_nope = k_full[..., :nope_dim]
    k_rope = k_full[..., nope_dim:]
    positions = torch.tensor([0, 3, 5, 7], device=device, dtype=torch.int64)
    freqs = torch.randn((8, rope_dim // 2), device=device, dtype=torch.float32)
    cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)

    q_out = torch.empty_like(q_full, dtype=torch.float8_e4m3fn)
    k_out = torch.empty_like(k_full, dtype=torch.float8_e4m3fn)
    q_nope_out = q_out[..., :nope_dim]
    q_rope_out = q_out[..., nope_dim:]
    k_nope_out = k_out[..., :nope_dim]
    k_rope_out = k_out[..., nope_dim:]

    mla_rope_quantize_fp8(
        q_rope=q_rope,
        k_rope=k_rope,
        q_nope=q_nope,
        k_nope=k_nope,
        cos_sin_cache=cos_sin_cache,
        pos_ids=positions,
        is_neox=True,
        quantize_dtype=torch.float8_e4m3fn,
        q_rope_out=q_rope_out,
        k_rope_out=k_rope_out,
        q_nope_out=q_nope_out,
        k_nope_out=k_nope_out,
        quant_scale_q=1.0,
        quant_scale_kv=2.0,
        enable_pdl=False,
    )
    torch.cuda.synchronize()

    expected_q_rope = _rope_reference(q_rope, cos_sin_cache, positions, True).to(
        torch.float8_e4m3fn
    )
    expected_k_rope = (
        _rope_reference(k_rope, cos_sin_cache, positions, True) / 2.0
    ).to(torch.float8_e4m3fn)
    expected_q_nope = q_nope.to(torch.float8_e4m3fn)
    expected_k_nope = (k_nope.float() / 2.0).to(torch.float8_e4m3fn)

    torch.testing.assert_close(
        q_rope_out.float(), expected_q_rope.float(), atol=0.25, rtol=0
    )
    torch.testing.assert_close(
        k_rope_out.float(), expected_k_rope.float(), atol=0.25, rtol=0
    )
    torch.testing.assert_close(
        q_nope_out.float(), expected_q_nope.float(), atol=0, rtol=0
    )
    torch.testing.assert_close(
        k_nope_out.float(), expected_k_nope.float(), atol=0, rtol=0
    )
