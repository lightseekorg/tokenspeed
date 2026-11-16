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

"""Standalone block quantization of a BF16 matrix for the block-scaled tests.

Serving quantizes expert inputs inside the routing launch; these tests need
quantized weights and inputs on their own, on NVIDIA and gfx950 alike.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.moe._triton.block_scaled import (
    _CHUNK_PROGRAM_MAX_ROWS,
    _quantize_row,
    _quantized_buffers,
)


@triton.jit
def _block_quantize(
    X,
    Values,
    Scales,
    Amax,
    Dequant,
    width: tl.constexpr,
    NVFP4: tl.constexpr,
    BLOCK: tl.constexpr,
    ROW_PROGRAMS: tl.constexpr,
    STORE_DEQUANT: tl.constexpr,
):
    """One program per row (``ROW_PROGRAMS``) or per (row, column chunk)."""
    _quantize_row(
        X,
        Values,
        Scales,
        Amax,
        Dequant,
        tl.program_id(0).to(tl.int64),
        tl.program_id(1),
        width,
        NVFP4,
        BLOCK,
        ROW_PROGRAMS,
        STORE_DEQUANT,
    )


def triton_block_quantize(
    x: torch.Tensor, quant_format: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    values, scales, amax, _ = _block_quantize_rows(x, quant_format, None)
    return values, scales, amax


def triton_block_quantize_nvfp4_values(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """NVFP4 codes, scales and row maxima of ``x``, and the BF16 values the
    codes encode, as the serving quantizer stores them for BF16-input GEMMs."""
    return _block_quantize_rows(x, "nvfp4", torch.empty_like(x, dtype=torch.bfloat16))


def _block_quantize_rows(x, quant_format, dequant):
    rows, width = x.shape
    nvfp4 = quant_format == "nvfp4"
    values, scales, amax = _quantized_buffers(rows, width, nvfp4, x.device)
    if rows:
        row_programs = rows > _CHUNK_PROGRAM_MAX_ROWS
        _block_quantize[(rows, 1 if row_programs else triton.cdiv(width, 256))](
            x,
            values,
            scales,
            values if amax is None else amax,
            values if dequant is None else dequant,
            width,
            nvfp4,
            256,
            row_programs,
            dequant is not None,
            enable_fp_fusion=False,
        )
    return values, scales, amax, dequant
