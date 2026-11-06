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

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsv41 import rope_quantize_fp8_grouped
from tokenspeed_kernel.ops.gemm import (
    fp8_projection,
    grouped_fp8_projection,
    grouped_fp8_projection_quantize_fp8,
    has_grouped_fp8_projection,
    prepare_fp8_projection,
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
    return _quantize_group32(x.reshape(x.shape[0], groups, x.shape[1] // groups * 512))


def _quantize_group32(x):
    """Group-32 UE8M0 quantization of FP32 values with a 1e-4 amax floor."""
    blocks = x.unflatten(-1, (x.shape[-1] // 32, 32))
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


def test_has_grouped_fp8_projection_agrees_with_prepare():
    input_dim = HEADS_PER_GROUP * 512
    weight, weight_scales = _weights(GROUPS, OUTPUT_DIM, input_dim)
    assert has_grouped_fp8_projection(weight.device, OUTPUT_DIM, input_dim)
    assert prepare_grouped_fp8_projection(weight, weight_scales, GROUPS) is not None
    weight, weight_scales = weight.cpu(), weight_scales.cpu()
    assert not has_grouped_fp8_projection(weight.device, OUTPUT_DIM, input_dim)
    assert prepare_grouped_fp8_projection(weight, weight_scales, GROUPS) is None


def _projection_inputs(tokens):
    input_dim = HEADS_PER_GROUP * 512
    weight, weight_scales = _weights(GROUPS, OUTPUT_DIM, input_dim)
    prepared = prepare_grouped_fp8_projection(weight, weight_scales, GROUPS)
    values, positions, cache = _inputs(tokens)
    codes, scales = rope_quantize_fp8_grouped(values, positions, cache, GROUPS)
    grouped = weight.view(GROUPS, OUTPUT_DIM, input_dim)
    return codes, scales, grouped, prepared


@pytest.mark.parametrize("tokens", [0, 1, 5, 64, 256, 1031])
def test_grouped_fp8_projection_quantize_fp8_matches_bf16_then_quantize(tokens):
    codes, scales, grouped, prepared = _projection_inputs(tokens)
    projected = grouped_fp8_projection(codes, scales, grouped, prepared, None)
    out_codes, out_scales = grouped_fp8_projection_quantize_fp8(
        codes, scales, grouped, prepared
    )
    expected_codes, expected_exponents = _quantize_group32(projected.flatten(1).float())
    assert out_codes.shape == (tokens, GROUPS * OUTPUT_DIM)
    assert torch.equal(out_codes.view(torch.uint8), expected_codes.view(torch.uint8))
    assert torch.equal(_unpack(out_scales), expected_exponents)
    aligned = (tokens + 3) // 4 * 4
    assert out_scales.stride() == (1, aligned)


@pytest.mark.parametrize("tokens", [0, 1, 5, 64, 256, 1031])
def test_fp8_projection_matches_dequantized_matmul(tokens):
    codes, scales, grouped, prepared = _projection_inputs(tokens)
    x, x_scales = grouped_fp8_projection_quantize_fp8(codes, scales, grouped, prepared)
    weight, weight_scales = _weights(1, 1024, GROUPS * OUTPUT_DIM)
    before = weight_scales.clone()
    weight_prepared = prepare_fp8_projection(weight, weight_scales)
    assert weight_prepared is not None
    torch.testing.assert_close(weight_scales, before, rtol=0, atol=0)

    out = torch.empty(tokens, 1024, device="cuda", dtype=torch.bfloat16)
    assert fp8_projection(x, x_scales, weight, weight_prepared, out) is out
    expected = _dequantize(x, _unpack(x_scales)).double() @ (
        _dequantize(weight, weight_scales).double().T
    )
    # FP32 accumulation of exact products; only the BF16 store rounds.
    torch.testing.assert_close(out.double(), expected, rtol=2**-8, atol=1e-2)

    # A destination off 16-byte alignment or with padded rows is written
    # through a temporary.
    storage = torch.full(
        (tokens * 1040 + 8,), -1.0, device="cuda", dtype=torch.bfloat16
    )
    for target in (
        storage[1 : tokens * 1024 + 1].view(tokens, 1024),
        storage[: tokens * 1040].view(tokens, 1040)[:, 8:1032],
    ):
        assert fp8_projection(x, x_scales, weight, weight_prepared, target) is target
        torch.testing.assert_close(target, out, rtol=0, atol=0)


def test_prepare_fp8_projection_declines_cpu_weights():
    weight = torch.zeros(64, 128, dtype=torch.float8_e4m3fn)
    scales = torch.full((64, 4), 127, dtype=torch.uint8)
    assert prepare_fp8_projection(weight, scales) is None


_WARMUP_SCRIPT = """
import sys
from pathlib import Path

import torch
from tokenspeed_kernel.ops.gemm import (
    fp8_projection,
    grouped_fp8_projection,
    grouped_fp8_projection_quantize_fp8,
    prepare_fp8_projection,
    prepare_grouped_fp8_projection,
    warmup_fp8_projections,
)

cache, warmed, fp8_output = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "1"
groups, rank, width, hidden = 2, 128, 512, 256
grouped = torch.zeros(groups * rank, width, device="cuda", dtype=torch.float8_e4m3fn)
grouped_scales = prepare_grouped_fp8_projection(
    grouped, torch.full((groups * rank, width // 32), 127, device="cuda", dtype=torch.uint8), groups
)
grouped = grouped.view(groups, rank, width)
weight = torch.zeros(hidden, groups * rank, device="cuda", dtype=torch.float8_e4m3fn)
weight_scales = prepare_fp8_projection(
    weight, torch.full((hidden, groups * rank // 32), 127, device="cuda", dtype=torch.uint8)
)
if not fp8_output:
    weight = weight_scales = None
warmup_fp8_projections(grouped, grouped_scales, weight, weight_scales, warmed)
compiled = sorted(str(path) for path in cache.rglob("*"))
assert compiled, "the warmup compiled nothing into the JIT cache"
for tokens in (1, 7, 16, 17, 100, 255, 256, 300):
    x = torch.zeros(groups, tokens, width, device="cuda", dtype=torch.float8_e4m3fn)
    aligned = (tokens + 3) // 4 * 4
    x_scales = torch.zeros(groups * width // 128 * aligned, device="cuda", dtype=torch.int32)
    x_scales = x_scales.as_strided((tokens, groups, width // 128), (1, width // 128 * aligned, aligned))
    x = x.transpose(0, 1)
    if fp8_output:
        z, z_scales = grouped_fp8_projection_quantize_fp8(x, x_scales, grouped, grouped_scales)
        fp8_projection(z, z_scales, weight, weight_scales, None)
    else:
        grouped_fp8_projection(x, x_scales, grouped, grouped_scales, None)
torch.cuda.synchronize()
new = sorted(set(str(path) for path in cache.rglob("*")) - set(compiled))
assert not new, f"serving-size launches compiled {new}"
print("no compilation after warmup")
"""


@pytest.mark.parametrize("fp8_output", [False, True])
def test_warmup_fp8_projections_leaves_nothing_to_compile(tmp_path, fp8_output):
    # A fresh DeepGEMM JIT cache records every kernel compiled in the child.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _WARMUP_SCRIPT,
            str(tmp_path / "jit"),
            "300",
            "1" if fp8_output else "0",
        ],
        env={**os.environ, "DG_JIT_CACHE_DIR": str(tmp_path / "jit")},
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout + result.stderr
