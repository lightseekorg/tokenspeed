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
from tokenspeed_kernel.ops.quantization import quantize_mxfp4 as kernel_quantize_mxfp4
from tokenspeed_kernel_amd.ops.gfx950.megakernel.plan import Plan
from tokenspeed_kernel_amd.ops.gfx950.megakernel.weights import dequantize_mxfp4


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


def mxfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed E2M1 codes ``[N, K / 2]`` and UE8M0 exponents ``[N, K / 32]``
    of BF16 ``x [N, K]``, by TokenSpeed's activation quantizer."""
    return kernel_quantize_mxfp4(
        x.bfloat16().contiguous(), scale_layout="linear", solution="triton"
    )


def round_mxfp4(x: torch.Tensor) -> torch.Tensor:
    """``x [N, K]`` through TokenSpeed's MXFP4 activation quantizer and back."""
    return dequantize_mxfp4(*mxfp4(x))


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
    scratch: torch.Tensor, plan: Plan, region: str, count: int, tag: int, row: int = 0
) -> torch.Tensor:
    """The first ``count`` int32 payloads of row ``row`` of ``region``,
    after checking that every one carries ``tag``."""
    start = plan.offset(region) + row * plan.row_pairs
    pairs = scratch[start : start + count]
    assert torch.all((pairs >> 32) == tag), region
    return (pairs & 0xFFFFFFFF).to(torch.int32)


__all__ = [
    "fill",
    "mxfp4",
    "payloads",
    "random_mxfp4",
    "round_mxfp4",
    "softplus",
]
