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

"""DSV4.1 wo_a in FP8: fused inverse RoPE + group-32 quantization feeding the
grouped FP8 projection."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsv41 import rope_quantize_fp8_grouped
from tokenspeed_kernel.ops.gemm import (
    grouped_fp8_projection,
    prepare_grouped_fp8_projection,
)
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and current_platform().is_blackwell),
    reason="The packed UE8M0 scale layout and the grouped FP8 GEMM target SM100.",
)

GROUPS, HEADS_PER_GROUP, OUTPUT_DIM = 8, 8, 1024


def _inverse_cache(max_positions):
    angles = torch.arange(max_positions, device="cuda")[:, None].float() * torch.exp(
        -torch.arange(32, device="cuda").float() / 4
    )
    return torch.cat((angles.cos(), -angles.sin()), dim=-1)


def _reference(values, positions, cache, groups):
    """Reference rounding: RoPE in FP32 products rounded to BF16, then
    group-32 UE8M0 quantization with a 1e-4 amax floor."""
    x = values.float()
    pairs = x[..., -64:].unflatten(-1, (-1, 2))
    cosine, sine = cache[positions.clamp_min(0)][:, None].chunk(2, dim=-1)
    even = pairs[..., 0] * cosine - pairs[..., 1] * sine
    odd = pairs[..., 1] * cosine + pairs[..., 0] * sine
    rotated = torch.stack((even, odd), dim=-1).flatten(-2)
    x = torch.cat((x[..., :-64], rotated), dim=-1).bfloat16().float()
    blocks = x.reshape(x.shape[0], groups, x.shape[1] // groups * 16, 32)
    raw = blocks.abs().amax(-1).clamp_min(1e-4) / 448
    bits = raw.view(torch.int32)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    scale = (exponent << 23).view(torch.float32)
    codes = (blocks / scale[..., None]).flatten(-2).to(torch.float8_e4m3fn)
    return codes, exponent


def _unpack(scales):
    words = scales.contiguous()
    return torch.stack([(words >> (8 * j)) & 255 for j in range(4)], -1).flatten(-2)


def _inputs(tokens, padded_heads=128, max_positions=4096):
    torch.manual_seed(tokens)
    storage = torch.randn(
        tokens, padded_heads, 512, device="cuda", dtype=torch.bfloat16
    )
    # The attention output is padded to the kernel's head count.
    values = storage[:, : GROUPS * HEADS_PER_GROUP]
    positions = torch.randint(-2, max_positions, (tokens,), device="cuda")
    return values, positions, _inverse_cache(max_positions)


@pytest.mark.parametrize("tokens", [1, 3, 32, 37, 64, 128, 256, 4099])
def test_rope_quantize_fp8_grouped_matches_reference(tokens):
    values, positions, cache = _inputs(tokens)
    before = values.clone()
    codes, scales = rope_quantize_fp8_grouped(values, positions, cache, GROUPS)
    expected_codes, expected_exponents = _reference(values, positions, cache, GROUPS)

    assert codes.shape == (tokens, GROUPS, HEADS_PER_GROUP * 512)
    assert torch.equal(codes.view(torch.uint8), expected_codes.view(torch.uint8))
    assert torch.equal(_unpack(scales), expected_exponents)
    torch.testing.assert_close(values, before, rtol=0, atol=0)
    # Scales are MN-major with the token count padded to 4; padding reads as 0.
    aligned = (tokens + 3) // 4 * 4
    assert scales.stride() == (1, scales.shape[2] * aligned, aligned)
    padded = scales.as_strided((aligned, *scales.shape[1:]), scales.stride())
    assert not padded[tokens:].any()


def test_rope_quantize_fp8_grouped_reuses_compilation_across_token_counts():
    from tokenspeed_kernel.ops.attention.dsv41 import triton as implementation
    from utils import assert_no_triton_compile

    values, positions, cache = _inputs(160)
    # Warm both integer specialization classes, then sweep serving counts.
    for tokens in (16, 7):
        rope_quantize_fp8_grouped(values[:tokens], positions[:tokens], cache, GROUPS)
    with assert_no_triton_compile(implementation._rope_quantize_fp8_grouped):
        for tokens in (1, 32, 37, 64, 128, 160):
            rope_quantize_fp8_grouped(
                values[:tokens], positions[:tokens], cache, GROUPS
            )


def test_rope_quantize_fp8_grouped_offsets_past_int32():
    # Both token * token_stride (input) and group * T * D (output) pass 2^31
    # elements at the far end; only the last token is checked.
    heads = GROUPS * HEADS_PER_GROUP
    tokens = (1 << 31) // ((GROUPS - 1) * HEADS_PER_GROUP * 512) + 4
    values = torch.empty(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)
    values[-1].copy_(torch.randn(heads, 512, device="cuda"))
    positions = torch.full((tokens,), 5, device="cuda")
    cache = _inverse_cache(16)
    codes, scales = rope_quantize_fp8_grouped(values, positions, cache, GROUPS)
    expected_codes, expected_exponents = _reference(
        values[-1:], positions[-1:], cache, GROUPS
    )
    assert torch.equal(codes[-1].view(torch.uint8), expected_codes[0].view(torch.uint8))
    assert torch.equal(_unpack(scales[-1:]), expected_exponents)


def _weights(groups, output_dim, input_dim):
    torch.manual_seed(7)
    codes = (torch.randn(groups * output_dim, input_dim, device="cuda") * 2).to(
        torch.float8_e4m3fn
    )
    # Checkpoint 32x32 blocks expand to one scale per 32 input channels.
    blocks = torch.randint(
        118, 124, (groups * output_dim // 32, input_dim // 32), device="cuda"
    )
    return codes, blocks.to(torch.uint8).repeat_interleave(32, dim=0)


def _dequantize(codes, exponents):
    scale = (exponents.to(torch.int32) << 23).view(torch.float32)
    return (codes.float().unflatten(-1, (-1, 32)) * scale[..., None]).flatten(-2)


@pytest.mark.parametrize("tokens", [0, 1, 5, 64, 256, 1031])
def test_grouped_fp8_projection_matches_dequantized_einsum(tokens):
    input_dim = HEADS_PER_GROUP * 512
    weight, weight_scales = _weights(GROUPS, OUTPUT_DIM, input_dim)
    before = weight_scales.clone()
    prepared = prepare_grouped_fp8_projection(weight, weight_scales, GROUPS)
    assert prepared is not None
    torch.testing.assert_close(weight_scales, before, rtol=0, atol=0)

    values, positions, cache = _inputs(tokens)
    codes, scales = rope_quantize_fp8_grouped(values, positions, cache, GROUPS)
    out = torch.empty(tokens, GROUPS, OUTPUT_DIM, device="cuda", dtype=torch.bfloat16)
    grouped = weight.view(GROUPS, OUTPUT_DIM, input_dim)
    assert grouped_fp8_projection(codes, scales, grouped, prepared, out) is out

    expected_codes, expected_exponents = _reference(values, positions, cache, GROUPS)
    x = _dequantize(expected_codes, expected_exponents).double()
    w = _dequantize(weight, weight_scales).view(GROUPS, OUTPUT_DIM, input_dim)
    expected = torch.einsum("tgk,gnk->tgn", x, w.double())
    # FP32 accumulation of exact products; only the BF16 store rounds.
    torch.testing.assert_close(out.double(), expected, rtol=2**-8, atol=1e-2)


def test_prepare_grouped_fp8_projection_declines_cpu_weights():
    weight = torch.zeros(2 * 64, 128, dtype=torch.float8_e4m3fn)
    scales = torch.full((2 * 64, 4), 127, dtype=torch.uint8)
    assert prepare_grouped_fp8_projection(weight, scales, 2) is None
