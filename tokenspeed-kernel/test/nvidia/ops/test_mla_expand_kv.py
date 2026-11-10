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
from tokenspeed_kernel.ops.attention.mla import (
    mla_expand_kv_fp8,
    supports_mla_expand_kv_fp8,
)
from tokenspeed_kernel.ops.attention.mla._triton.kv_pack import (
    _mla_rope_broadcast_rows,
    triton_mla_kv_pack_quantize_fp8,
    triton_mla_rope_broadcast_fp8,
)
from tokenspeed_kernel.ops.gemm import mm
from tokenspeed_kernel.ops.quantization import quantize_fp8
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires Blackwell",
)

LATENT, NOPE, ROPE, VALUE = 512, 128, 64, 128


def _projection(heads):
    weight = torch.randn(heads * (NOPE + VALUE), LATENT, device="cuda") * 0.05
    return quantize_fp8(weight, granularity="block", block_size=(128, 128))


def _history(rows):
    latent = torch.randn(rows, LATENT, device="cuda", dtype=torch.bfloat16)
    k_pe = torch.randn(rows, 1, ROPE, device="cuda", dtype=torch.bfloat16) * 4
    values, scales = quantize_fp8(latent, granularity="token_group", group_size=128)
    return values, scales, k_pe


@pytest.mark.parametrize("heads", [8, 96])
@pytest.mark.parametrize("rows", [1, 63, 64, 128, 300, 4099])
def test_expand_kv_matches_bf16_projection_then_pack(heads, rows):
    torch.manual_seed(rows * 131 + heads)
    weight, weight_scales = _projection(heads)
    latent, latent_scales, k_pe = _history(rows)
    assert supports_mla_expand_kv_fp8(
        weight, weight_scales, nope_head_dim=NOPE, value_head_dim=VALUE
    )

    projected = mm(
        latent,
        weight,
        A_scales=latent_scales,
        B_scales=weight_scales,
        out_dtype=torch.bfloat16,
        quant="mxfp8",
        block_size=[128, 128],
        solution="trtllm_cutedsl",
    ).view(rows, heads, NOPE + VALUE)
    key_ref, value_ref = triton_mla_kv_pack_quantize_fp8(
        projected[..., :NOPE], k_pe, projected[..., NOPE:]
    )

    key = torch.full((rows, heads, NOPE + ROPE), float("nan"), device="cuda").to(
        torch.float8_e4m3fn
    )
    value = torch.full((rows, heads, VALUE), float("nan"), device="cuda").to(
        torch.float8_e4m3fn
    )
    mla_expand_kv_fp8(latent, latent_scales, weight, weight_scales, k_pe, key, value)

    assert torch.equal(key.view(torch.uint8), key_ref.view(torch.uint8))
    assert torch.equal(value.view(torch.uint8), value_ref.view(torch.uint8))


def test_expand_kv_support_needs_block_fp8_weights():
    weight, weight_scales = _projection(4)
    assert not supports_mla_expand_kv_fp8(
        weight.to(torch.bfloat16), None, nope_head_dim=NOPE, value_head_dim=VALUE
    )
    assert not supports_mla_expand_kv_fp8(
        weight, weight_scales[:, :1], nope_head_dim=NOPE, value_head_dim=VALUE
    )
    assert not supports_mla_expand_kv_fp8(
        weight, weight_scales, nope_head_dim=192, value_head_dim=VALUE
    )


def test_rope_broadcast_compiles_once_across_row_counts():
    heads = 96
    key = torch.empty(
        4099, heads, NOPE + ROPE, device="cuda", dtype=torch.float8_e4m3fn
    )
    k_pe = torch.randn(4099, ROPE, device="cuda", dtype=torch.bfloat16)
    triton_mla_rope_broadcast_fp8(k_pe[:64], key[:64, :, NOPE:])
    with assert_no_triton_compile(_mla_rope_broadcast_rows):
        for rows in (1, 17, 32, 1000, 4099):
            triton_mla_rope_broadcast_fp8(k_pe[:rows], key[:rows, :, NOPE:])
    torch.cuda.synchronize()
    expected = k_pe.float().to(torch.float8_e4m3fn)[:, None, :].expand(-1, heads, -1)
    assert torch.equal(
        key[..., NOPE:].view(torch.uint8), expected.contiguous().view(torch.uint8)
    )
