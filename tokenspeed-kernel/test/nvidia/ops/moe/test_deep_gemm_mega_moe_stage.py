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

"""MegaMoE input staging: UE8M0 FP8 hidden states and widened router ids."""

import pytest
import torch
from utils import is_nvidia

if not is_nvidia():
    pytest.skip("NVIDIA GPU required", allow_module_level=True)

from tokenspeed_kernel.ops.moe._deep_gemm.mega_moe_stage import (  # noqa: E402
    _stage_mxfp4_mega_moe_inputs_kernel,
    stage_mxfp4_mega_moe_inputs,
)
from utils import assert_no_triton_compile  # noqa: E402


def _stage(hidden, weights, ids):
    tokens, size = hidden.shape
    out = (
        torch.empty(tokens, size, device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty(tokens, size // 128, device="cuda", dtype=torch.int32),
        torch.full((tokens, ids.shape[1]), -1, device="cuda", dtype=torch.int64),
        torch.empty(tokens, ids.shape[1], device="cuda", dtype=torch.float32),
    )
    stage_mxfp4_mega_moe_inputs(hidden, weights, ids, *out)
    return out


@pytest.mark.parametrize("tokens", [1, 37])
def test_int32_ids_stage_like_int64_ids(tokens):
    torch.manual_seed(tokens)
    hidden = torch.randn(tokens, 512, device="cuda", dtype=torch.bfloat16)
    weights = torch.rand(tokens, 6, device="cuda", dtype=torch.float32)
    ids = torch.randint(0, 384, (tokens, 6), device="cuda", dtype=torch.int32)

    narrow = _stage(hidden, weights, ids)
    wide = _stage(hidden, weights, ids.long())
    assert torch.equal(narrow[2], ids.long())
    for got, want in zip(narrow, wide):
        assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))


def _reference(hidden, weights, ids):
    """Group-32 UE8M0 FP8 codes and per-128-column packed exponents."""
    tokens, size = hidden.shape
    groups = hidden.float().view(tokens, size // 32, 32)
    amax = groups.abs().amax(-1).clamp_min(1.0e-4)
    bits = (amax / 448.0).view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).int()
    exponent = exponent.clamp(1, 254)
    scale = (exponent << 23).view(torch.float32)
    codes = (groups * (1.0 / scale)[..., None]).view(tokens, size)
    packed = (
        exponent.view(tokens, size // 128, 4).long()
        << torch.arange(0, 32, 8, device=hidden.device)
    ).sum(-1)
    # Four bytes reinterpreted as one int32 word.
    packed = torch.where(packed >= 2**31, packed - 2**32, packed)
    return (
        codes.to(torch.float8_e4m3fn),
        packed.to(torch.int32),
        ids.long(),
        weights,
    )


@pytest.mark.parametrize("tokens", [1, 37, 63, 64, 65, 255, 1023, 1024, 1500])
@pytest.mark.parametrize("size", [512, 5120])
def test_stage_matches_reference(tokens, size):
    torch.manual_seed(tokens * size)
    hidden = torch.randn(tokens, size, device="cuda", dtype=torch.bfloat16)
    hidden[0, :32] = 0.0
    hidden[-1, 32:64] = 1.0e-7
    weights = torch.rand(tokens, 6, device="cuda", dtype=torch.float32)
    ids = torch.randint(0, 384, (tokens, 6), device="cuda", dtype=torch.int32)

    for got, want in zip(
        _stage(hidden, weights, ids), _reference(hidden, weights, ids)
    ):
        assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))


def test_stage_token_count_does_not_recompile():
    hidden = torch.randn(2048, 512, device="cuda", dtype=torch.bfloat16)
    weights = torch.rand(2048, 6, device="cuda", dtype=torch.float32)
    ids = torch.randint(0, 384, (2048, 6), device="cuda", dtype=torch.int32)
    for tokens in (1, 64, 1024):
        _stage(hidden[:tokens], weights[:tokens], ids[:tokens])
    with assert_no_triton_compile(_stage_mxfp4_mega_moe_inputs_kernel):
        for tokens in (2, 17, 63, 65, 200, 1023, 1025, 2048):
            _stage(hidden[:tokens], weights[:tokens], ids[:tokens])


def test_stage_reads_rows_past_int32_offsets():
    # Two rows 2**31 elements apart: the second row's offset needs int64.
    stride = 2**31
    base = torch.zeros(stride + 128, device="cuda", dtype=torch.bfloat16)
    hidden = base.as_strided((2, 128), (stride, 1))
    hidden[1].copy_(torch.randn(128, device="cuda", dtype=torch.bfloat16))
    weights = torch.rand(2, 6, device="cuda", dtype=torch.float32)
    ids = torch.randint(0, 384, (2, 6), device="cuda", dtype=torch.int32)

    got = _stage(hidden, weights, ids)
    want = _reference(hidden.contiguous(), weights, ids)
    for a, b in zip(got, want):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
