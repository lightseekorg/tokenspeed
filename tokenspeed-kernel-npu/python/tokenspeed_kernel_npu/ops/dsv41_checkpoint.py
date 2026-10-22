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


"""Decode V4.1 checkpoint bytes on CPU before device weight preparation."""

from functools import cache

import torch


def power_of_two(exponent):
    """Construct FP32 powers of two directly, including the E8M0 subnormal."""
    exponent = exponent.to(torch.int32)
    bits = torch.where(exponent == -127, 1 << 22, (exponent + 127) << 23)
    return bits.contiguous().view(torch.float32)


def decode_e8m0(codes):
    values = power_of_two(codes.int() - 127)
    return torch.where(codes == 255, float("nan"), values)


def decode_e4m3(codes):
    codes = codes.to(torch.int32)
    magnitude = codes & 127
    exponent, mantissa = magnitude >> 3, magnitude & 7
    value = torch.where(
        exponent == 0,
        mantissa.float() * 2.0**-9,
        (1 + mantissa.float() / 8) * power_of_two(exponent - 7),
    )
    value = torch.where(magnitude == 127, float("nan"), value)
    return torch.where(codes >= 128, -value, value)


@cache
def _e2m1_table(device):
    # Reuse the immutable FP32 codebook across checkpoint shards.
    return torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32, device=device
    )


def decode_e2m1(codes):
    table = _e2m1_table(codes.device)
    codes = codes.int()
    value = table[(codes & 7).long()]
    return torch.where(codes >= 8, -value, value)


def dequant_checkpoint(codes, scales, kind):
    """Decode checkpoint bytes on CPU without requiring float8 device support."""
    if codes.device.type != "cpu" or scales.device.type != "cpu":
        raise ValueError("Checkpoint conversion must run on CPU")
    scale = decode_e8m0(scales)
    if kind == "fp8":
        values = decode_e4m3(codes)
        scale = scale.repeat_interleave(32, 0).repeat_interleave(32, 1)
        return (values * scale[: values.shape[0], : values.shape[1]]).to(torch.bfloat16)
    if kind == "mxfp4":
        unpacked = torch.stack((codes & 15, codes >> 4), -1).flatten(1)
        values = decode_e2m1(unpacked).reshape(codes.shape[0], -1, 32)
        return (values * scale.unsqueeze(-1)).flatten(1).to(torch.bfloat16)
    raise ValueError(f"Unknown checkpoint kind {kind}")
