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

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import error_fn

platform = current_platform()
_fp8_dtype = torch.float8_e4m3fn
_fp8_max = torch.finfo(_fp8_dtype).max
_fp8_min = torch.finfo(_fp8_dtype).min

_trtllm_per_token_group_quant_fp8 = error_fn

if platform.is_nvidia:
    from tokenspeed_kernel.ops.quantization.trtllm import (
        _per_token_group_quant_8bit as _trtllm_per_token_group_quant_fp8,
    )


@triton.jit
def _fp8_blockscale_quantize_prepacked_kernel(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    valid_rows,
    padded_rows,
    bit8_min,
    bit8_max,
    BLOCK: tl.constexpr,
):
    """Quantize valid rows and initialize the GEMM's M-padding in one launch."""
    group_id = tl.program_id(0)
    groups_per_row = y_num_columns // group_size
    row = group_id // groups_per_row
    scale_col = group_id % groups_per_row

    cols = tl.arange(0, BLOCK)
    col_mask = cols < group_size
    row_offset = row.to(tl.int64) * y_num_columns
    group_offset = scale_col.to(tl.int64) * group_size
    offsets = row_offset + group_offset + cols

    y = tl.load(y_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(y))
    # Match TRT-LLM's scale_1x128_kernel: an all-zero group uses a neutral
    # scale of one, while every other group uses amax / FP8_MAX.
    y_s_inv = tl.where(amax == 0.0, 1.0, bit8_max / amax)
    y_s = 1.0 / y_s_inv
    y_q = tl.clamp(y * y_s_inv, bit8_min, bit8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + offsets, y_q, mask=col_mask)
    tl.store(
        y_s_ptr + scale_col.to(tl.int64) * padded_rows + row,
        y_s,
    )

    # Only the first valid row's programs initialize the at-most-three tail
    # rows. Each program owns one 128-column group, so these stores do not race.
    for pad_offset in tl.static_range(0, 3):
        pad_row = valid_rows + pad_offset
        pad_mask = (row == 0) & (pad_row < padded_rows)
        pad_offsets = pad_row.to(tl.int64) * y_num_columns + group_offset + cols
        tl.store(y_q_ptr + pad_offsets, 0.0, mask=pad_mask & col_mask)
        tl.store(
            y_s_ptr + scale_col.to(tl.int64) * padded_rows + pad_row,
            1.0,
            mask=pad_mask,
        )


def flashinfer_fp8_blockscale_quantize_prepacked(
    x: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations into FlashInfer's native MN-major scale layout.

    For row counts already divisible by four, this exposes TRT-LLM's native
    ``[K / 128, M]`` scale output directly. Otherwise a fused Triton kernel
    writes the valid quantized rows plus zero/one padding directly into
    ``[round_up(M, 4), K]`` values and ``[K / 128, round_up(M, 4)]`` scales.

    Args:
        x: Contiguous BF16/FP16 activation matrix shaped ``[M, K]``.
        group_size: Number of K elements represented by one scale. FlashInfer's
            FP8 block-scale GEMM currently requires 128.

    Returns:
        A tuple containing padded FP8 values and contiguous MN-major FP32
        scales. The returned row count equals ``round_up(M, 4)``.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D, got shape {tuple(x.shape)}")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if x.shape[0] <= 0:
        raise ValueError(f"x must contain at least one row, got {x.shape[0]}")
    if group_size != 128:
        raise ValueError(
            "FlashInfer FP8 block-scale prepacking requires group_size=128, "
            f"got {group_size}"
        )
    if x.shape[1] % group_size:
        raise ValueError(
            f"x.shape[1] must be divisible by {group_size}, got {x.shape[1]}"
        )

    valid_rows, columns = x.shape
    padded_rows = (valid_rows + 3) // 4 * 4
    if (
        padded_rows == valid_rows
        and x.dtype == torch.bfloat16
        and _trtllm_per_token_group_quant_fp8 is not error_fn
    ):
        x_q, x_s = _trtllm_per_token_group_quant_fp8(x, group_size, False)
        expected_shape = (columns // group_size, valid_rows)
        if tuple(x_s.shape) != expected_shape or not x_s.is_contiguous():
            raise RuntimeError(
                "TRT-LLM FP8 quantizer returned unexpected prepared scales: "
                f"shape={tuple(x_s.shape)}, stride={tuple(x_s.stride())}, "
                f"expected contiguous {expected_shape}"
            )
        return x_q, x_s

    x_q = torch.empty(
        (padded_rows, columns),
        device=x.device,
        dtype=_fp8_dtype,
    )
    x_s = torch.empty(
        (columns // group_size, padded_rows),
        device=x.device,
        dtype=torch.float32,
    )
    groups = valid_rows * (columns // group_size)
    _fp8_blockscale_quantize_prepacked_kernel[(groups,)](
        x,
        x_q,
        x_s,
        group_size,
        columns,
        valid_rows,
        padded_rows,
        bit8_min=_fp8_min,
        bit8_max=_fp8_max,
        BLOCK=group_size,
        num_warps=1,
        num_stages=1,
    )
    return x_q, x_s
