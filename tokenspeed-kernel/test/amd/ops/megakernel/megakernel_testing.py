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

"""Host helpers shared by the megakernel tests: MXFP reference
quantization and reading or filling a launch's mailbox regions."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd.ops.gfx950.megakernel.plan import Plan
from tokenspeed_kernel_amd.ops.gfx950.megakernel.weights import dequantize_mxfp4


def mxfp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP8 codes (uint8, the shape of ``x``) and per-32 exponents of ``x``
    with TokenSpeed's scale rule."""
    blocks = x.float().reshape(-1, 32)
    ratio = (blocks.abs().amax(-1) * (1.0 / 448.0)).view(torch.int32)
    exponent = (ratio >> 23) & 255
    exponent = exponent + ((exponent < 255) & ((ratio & 0x7FFFFF) != 0)).int()
    q = (blocks / torch.exp2(exponent.float() - 127)[:, None]).to(torch.float8_e4m3fn)
    return q.view(torch.uint8).reshape(x.shape), exponent


def dequantize_mxfp8(codes: torch.Tensor, exponents: torch.Tensor) -> torch.Tensor:
    """FP32 values of MXFP8 ``codes`` and ``exponents``, flattened."""
    values = codes.view(torch.float8_e4m3fn).float().reshape(-1, 32)
    return (values * torch.exp2(exponents.float() - 127)[:, None]).flatten()


def round_mxfp8(x: torch.Tensor) -> torch.Tensor:
    """``x`` through MXFP8 and back, in the shape of ``x``."""
    return dequantize_mxfp8(*mxfp8(x)).reshape(x.shape)


def random_mxfp4(
    shape: tuple[int, ...], generator: torch.Generator, lo: int = 116, hi: int = 121
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random MXFP4 codes ``[..., n, k / 2]`` and UE8M0 scales ``[..., n,
    k / 32]`` in ``[lo, hi)`` of a ``[..., n, k]`` weight, on the GPU."""
    *lead, n, k = shape
    codes = torch.randint(
        0, 256, (*lead, n, k // 2), generator=generator, dtype=torch.uint8
    )
    scales = torch.randint(
        lo, hi, (*lead, n, k // 32), generator=generator, dtype=torch.uint8
    )
    return codes.cuda(), scales.cuda()


def grid_weight(rows: int, cols: int, generator: torch.Generator) -> torch.Tensor:
    """A BF16 weight on the MXFP4 grid (as the checkpoint's mix weights)."""
    codes, scales = random_mxfp4((rows, cols), generator, 117, 123)
    return dequantize_mxfp4(codes, scales).bfloat16()


def softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 20, x, torch.log1p(torch.exp(x)))


def fill(
    scratch: torch.Tensor, plan: Plan, region: str, payload: torch.Tensor, tag: int
) -> None:
    """Publish ``payload`` (32-bit words) in row 0 of ``region`` with ``tag``,
    as the stage that writes the region would."""
    start = plan.offset(region)
    words = payload.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    scratch[start : start + words.numel()] = (tag << 32) | words


def payloads(
    scratch: torch.Tensor, plan: Plan, region: str, count: int, tag: int
) -> torch.Tensor:
    """The first ``count`` int32 payloads of row 0 of ``region``, after
    checking that every one carries ``tag``."""
    pairs = scratch[plan.offset(region) : plan.offset(region) + count]
    assert torch.all((pairs >> 32) == tag), region
    return (pairs & 0xFFFFFFFF).to(torch.int32)


__all__ = [
    "dequantize_mxfp8",
    "fill",
    "grid_weight",
    "mxfp8",
    "payloads",
    "random_mxfp4",
    "round_mxfp8",
    "softplus",
]
