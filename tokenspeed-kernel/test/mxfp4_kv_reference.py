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

"""Torch reference codec for MXFP4 KV-cache rows.

A row splits into blocks of 32 channels. Each block stores one UE8M0 scale (a
biased power-of-two exponent byte) and 32 E2M1 codes, two per byte with the
even channel in the low nibble. The scale is ``2**ceil(log2(max(amax, 6 *
2**-126) / 6))``, at most ``2**127``, so the block maximum never clips; codes
round to the nearest E2M1 value, ties to the even code. Bit-exact with
``tokenspeed_kernel.ops.quantization._triton_mxfp4._mxfp4_encode_rows``.
"""

from __future__ import annotations

import torch

MXFP4_BLOCK = 32


def _e2m1_codes(scaled: torch.Tensor) -> torch.Tensor:
    a = scaled.abs().clamp(max=6.0)
    code = sum(
        (a > t if strict else a >= t).to(torch.int32)
        for t, strict in (
            (0.25, True),
            (0.75, False),
            (1.25, True),
            (1.75, False),
            (2.5, True),
            (3.5, False),
            (5.0, True),
        )
    )
    return (code | (torch.signbit(scaled).to(torch.int32) << 3)).to(torch.uint8)


def _e2m1_values(codes: torch.Tensor) -> torch.Tensor:
    exponent = ((codes >> 1) & 3).float()
    mantissa = (codes & 1).float()
    magnitude = torch.where(
        exponent == 0, 0.5 * mantissa, torch.exp2(exponent - 1) * (1 + 0.5 * mantissa)
    )
    return torch.where((codes & 8) != 0, -magnitude, magnitude)


def _scale(exponent: torch.Tensor) -> torch.Tensor:
    return (exponent.to(torch.int32) << 23).view(torch.float32)


def mxfp4_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[..., D]`` rows -> uint8 ``[..., D // 2]`` E2M1 pairs and uint8
    ``[..., D // 32]`` biased exponents."""
    d = x.shape[-1]
    groups = x.float().reshape(*x.shape[:-1], d // MXFP4_BLOCK, MXFP4_BLOCK)
    scaled_max = groups.abs().amax(-1).clamp(min=6.0 * 2.0**-126) / 6.0
    bits = scaled_max.contiguous().view(torch.int32)
    exponent = (((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).int()).clamp(max=254)
    codes = _e2m1_codes(groups / _scale(exponent)[..., None])
    codes = codes.reshape(*x.shape[:-1], d // 2, 2)
    return codes[..., 0] | (codes[..., 1] << 4), exponent.to(torch.uint8)


def mxfp4_dequantize(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`mxfp4_quantize`, in fp32 ``[..., 2 * packed_D]``."""
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1)
    d = packed.shape[-1] * 2
    values = _e2m1_values(codes).reshape(
        *packed.shape[:-1], d // MXFP4_BLOCK, MXFP4_BLOCK
    )
    return (values * _scale(scales)[..., None]).reshape(*packed.shape[:-1], d)
