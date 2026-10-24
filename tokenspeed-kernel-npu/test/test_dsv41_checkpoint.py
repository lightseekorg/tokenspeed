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


"""CPU checkpoint decoding against independent float-format references."""

import pytest
import torch
from tokenspeed_kernel_npu.ops import dsv41_checkpoint as op


def test_e2m1_decode_is_dtype_independent():
    codes = torch.arange(16, dtype=torch.int32, device="cpu").to(torch.uint8)
    magnitude = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
    expected = torch.cat((magnitude, -magnitude))
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        decoded = op.decode_e2m1(codes)
    finally:
        torch.set_default_dtype(previous)
    assert decoded.dtype == torch.float32
    torch.testing.assert_close(decoded.cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(decoded.cpu().signbit(), expected.signbit())


def test_checkpoint_mxfp4_decoding():
    # All 16 codes, with independent explicit signed E2M1 levels.
    codes = torch.arange(16, dtype=torch.uint8).repeat(2).reshape(1, 32)
    packed = codes[:, ::2] | (codes[:, 1::2] << 4)
    scale = torch.tensor([[128]], dtype=torch.uint8)
    expected = (
        torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
        .repeat(2)
        .reshape(1, 32)
        * 2
    )
    torch.testing.assert_close(
        op.dequant_checkpoint(packed, scale, "mxfp4"),
        expected.bfloat16(),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("kind", ["e4m3", "e8m0"])
def test_float_byte_decoding(kind):
    codes = torch.arange(256, dtype=torch.uint8)
    dtype = torch.float8_e4m3fn if kind == "e4m3" else torch.float8_e8m0fnu
    decode = op.decode_e4m3 if kind == "e4m3" else op.decode_e8m0
    torch.testing.assert_close(
        decode(codes), codes.view(dtype).float(), rtol=0, atol=0, equal_nan=True
    )


def test_checkpoint_fp8_block_scales_and_partial_tiles():
    codes = torch.arange(35 * 67).remainder(256).to(torch.uint8).reshape(35, 67)
    scales = torch.tensor([[125, 126, 127], [128, 129, 130]], dtype=torch.uint8)
    values = codes.view(torch.float8_e4m3fn).float()
    expected = torch.empty_like(values)
    for row in range(35):
        for col in range(67):
            expected[row, col] = values[row, col] * 2.0 ** (
                int(scales[row // 32, col // 32]) - 127
            )
    actual = op.dequant_checkpoint(codes, scales, "fp8")
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(
        actual, expected.bfloat16(), rtol=0, atol=0, equal_nan=True
    )


def test_checkpoint_mxfp4_multiple_groups():
    codes = torch.arange(256, dtype=torch.uint8).reshape(4, 64)
    scales = torch.tensor([[125, 126, 127, 128]] * 4, dtype=torch.uint8)
    levels = (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6)
    expected = torch.empty((4, 128), dtype=torch.bfloat16)
    for row in range(4):
        for col in range(128):
            byte = int(codes[row, col // 2])
            code = (byte >> (4 * (col % 2))) & 15
            expected[row, col] = levels[code] * 2.0 ** (
                int(scales[row, col // 32]) - 127
            )
    actual = op.dequant_checkpoint(codes, scales, "mxfp4")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual.signbit(), expected.signbit())


def test_checkpoint_rejects_unknown_kind():
    with pytest.raises(ValueError, match="Unknown checkpoint kind"):
        op.dequant_checkpoint(
            torch.zeros(1, 16, dtype=torch.uint8),
            torch.ones(1, 1, dtype=torch.uint8),
            "int4",
        )


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_checkpoint_rejects_non_cpu_inputs(device):
    other = "meta" if device == "cpu" else "cpu"
    codes = torch.zeros(1, 16, dtype=torch.uint8, device=device)
    scales = torch.ones(1, 1, dtype=torch.uint8, device=other)
    with pytest.raises(ValueError, match="must run on CPU"):
        op.dequant_checkpoint(codes, scales, "mxfp4")
