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

import pytest
import torch
import torch_npu  # noqa: F401 - registers the NPU device
from tokenspeed_kernel_npu.ops.dsv41_pointwise import (
    decode_rows,
    reference_rmsnorm,
    rope_inplace,
)


def _e4m3_table(device):
    return (
        torch.arange(256, dtype=torch.uint8)
        .view(torch.float8_e4m3fn)
        .float()
        .to(device)
    )


def _e8m0_table(device):
    return (
        torch.arange(256, dtype=torch.uint8)
        .view(torch.float8_e8m0fnu)
        .float()
        .to(device)
    )


def _signed_e2m1_table(device):
    return torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device=device
    )


def rotary_reference(x, p, table):
    width = table.shape[1]
    angles = table[p.clamp_min(0).long()]
    shape = (p.numel(),) + (1,) * (x.ndim - 2) + (width // 2,)
    c, s = (v.reshape(shape) for v in angles.chunk(2, -1))
    tail = x[..., -width:].float()
    a, b = tail[..., ::2], tail[..., 1::2]
    result = x.clone()
    result[..., -width:] = torch.stack((a * c - b * s, b * c + a * s), -1).flatten(-2)
    return result


@pytest.mark.parametrize(
    "tokens,heads,width,rotary,strided",
    [
        (0, 8, 512, 64, False),
        (1, 1, 128, 64, False),
        (8, 8, 512, 64, False),
        (16, 64, 512, 64, False),
        (32, 8, 512, 64, False),
        (129, 8, 512, 64, True),
        (32, 1, 128, 128, False),
        (32, 1, 512, 64, True),
    ],
)
def test_rotary(tokens, heads, width, rotary, strided):
    torch.manual_seed(91)
    base = torch.randn(
        tokens,
        heads * 2 if strided else heads,
        width,
        dtype=torch.bfloat16,
        device="npu",
    )
    x = base[:, ::2] if strided else base
    positions = torch.arange(tokens, device="npu", dtype=torch.int32) - 1
    table = torch.randn(max(tokens, 1), rotary, device="npu", dtype=torch.float32)
    expected = rotary_reference(x, positions, table)
    rope_inplace(x, positions, table)
    torch.npu.synchronize()
    torch.testing.assert_close(x, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "rows,width",
    [(0, 512), (1, 512), (8, 512), (32, 512), (32, 1024), (32, 5120), (129, 128)],
)
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_reference_norm(rows, width, weight_dtype):
    torch.manual_seed(92)
    base = torch.randn(rows, width * 2, device="npu", dtype=torch.bfloat16)
    x = base[:, :width]
    w = torch.randn(width, device="npu", dtype=weight_dtype)
    z = x.float()
    expected = (
        w.float() * (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-6))
    ).to(x.dtype)
    actual = reference_rmsnorm(x, w, 1e-6, None)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1 / 128, atol=1e-6)
    if rows:
        rel = (
            (actual.float() - expected.float()).norm() / expected.float().norm()
        ).item()
        assert rel < 1e-4, rel


@pytest.mark.parametrize("fp4", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_decode_all_codes_scales(fp4, dtype):
    # Cartesian coverage of every data byte and scale byte, including NaNs,
    # subnormals, infinities after scale multiplication, and signed zero.
    rows = 256
    codes = torch.arange(256, dtype=torch.uint8).repeat(rows, 1)
    if not fp4:
        codes = codes.repeat(1, 2)
    group = 16 if fp4 else 32
    scales = torch.arange(256, dtype=torch.uint8)[:, None].repeat(1, 512 // group)
    fp8_values = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    lut = (
        torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
        if fp4
        else fp8_values
    )
    sl = (
        fp8_values
        if fp4
        else torch.arange(256, dtype=torch.uint8).view(torch.float8_e8m0fnu).float()
    )
    if fp4:
        indices = torch.stack((codes.long() & 15, codes.long() >> 4), -1).flatten(-2)
    else:
        indices = codes.long()
    expected = (
        lut[indices].reshape(rows, -1, group) * sl[scales.long()][:, :, None]
    ).reshape(rows, 512)
    valid = (torch.arange(rows) % 3) != 0
    expected.masked_fill_(~valid[:, None], 0)
    expected = expected.to(dtype)
    out = torch.empty((rows, 512), dtype=dtype, device="npu")
    decode_rows(codes.npu(), scales.npu(), valid.npu(), out, fp4=fp4)
    actual = out.cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(
        torch.signbit(actual[~expected.isnan()]),
        torch.signbit(expected[~expected.isnan()]),
    )


def test_graph_replay_mutable_inputs():
    torch.manual_seed(93)
    x = torch.randn(32, 8, 512, device="npu", dtype=torch.bfloat16)
    positions = torch.arange(32, device="npu", dtype=torch.int32)
    table = torch.randn(256, 64, device="npu")
    weight = torch.randn(512, device="npu", dtype=torch.bfloat16)
    codes = torch.randint(0, 255, (32, 512), device="npu", dtype=torch.uint8)
    scales = torch.full((32, 16), 127, device="npu", dtype=torch.uint8)
    valid = torch.ones(32, device="npu", dtype=torch.bool)
    output = torch.empty((32, 512), device="npu", dtype=torch.bfloat16)
    vl = _e4m3_table(x.device)

    def forward():
        rope_inplace(x, positions, table)
        norm = reference_rmsnorm(x, weight, 1e-6, None)
        decode_rows(codes, scales, valid, output, fp4=False)
        return norm

    for _ in range(3):
        forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        norm = forward()
    for step in range(4):
        x.normal_()
        positions.add_(3)
        codes.random_(0, 254)
        valid.copy_((torch.arange(32, device="npu") + step) % 3 != 0)
        expected_rope = rotary_reference(x, positions, table)
        z = expected_rope.float()
        expected_norm = (
            weight.float() * (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-6))
        ).to(x.dtype)
        expected_decode = (
            vl[codes.long()].masked_fill(~valid[:, None], 0).to(output.dtype)
        )
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(x, expected_rope, rtol=0, atol=0)
        torch.testing.assert_close(norm, expected_norm, rtol=1 / 128, atol=1e-6)
        torch.testing.assert_close(
            output, expected_decode, rtol=0, atol=0, equal_nan=True
        )


@pytest.mark.parametrize("fp4", [False, True])
def test_full_c32_candidate_decode_launch_bound(fp4):
    # The real C32/2048 selection exceeds the old one-program-per-row limit.
    rows = 65537
    cpu_codes = torch.arange(256, dtype=torch.uint8).repeat(rows, 1)
    if not fp4:
        cpu_codes = cpu_codes.repeat(1, 2)
    cpu_scales = torch.full(
        (rows, 32 if fp4 else 16), 56 if fp4 else 127, dtype=torch.uint8
    )
    vl = (
        _signed_e2m1_table(torch.device("cpu"))
        if fp4
        else _e4m3_table(torch.device("cpu"))
    )
    sl = _e4m3_table(torch.device("cpu")) if fp4 else _e8m0_table(torch.device("cpu"))
    indices = (
        torch.stack((cpu_codes[0].long() & 15, cpu_codes[0].long() >> 4), -1).flatten()
        if fp4
        else cpu_codes[0].long()
    )
    expected_row = (vl[indices] * sl[cpu_scales[0, 0].long()]).to(torch.bfloat16)
    codes, scales = cpu_codes.npu(), cpu_scales.npu()
    out = torch.empty((rows, 512), device="npu", dtype=torch.bfloat16)
    decode_rows(codes, scales, None, out, fp4=fp4)
    torch.testing.assert_close(
        out.cpu(), expected_row.expand(rows, -1), atol=0, rtol=0, equal_nan=True
    )
