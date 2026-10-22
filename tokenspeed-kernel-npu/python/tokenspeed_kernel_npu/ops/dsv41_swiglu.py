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

"""Clamped BF16 SwiGLU with FP32 intermediates, without routing weights."""

import math

import torch
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit
def _swiglu(
    X,
    Y,
    H: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    LIMIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(X + row * XS0 + col * XS1, col < H, other=0).to(tl.float32)
    up = tl.load(X + row * XS0 + (H + col) * XS1, col < H, other=0).to(tl.float32)
    # Preserve torch.clamp NaN propagation even on backends with minNum semantics.
    gate = tl.where(gate != gate, gate, tl.minimum(gate, LIMIT))  # noqa: PLR0124
    up = tl.where(
        up != up, up, tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
    )  # noqa: PLR0124
    silu = gate / (1.0 + tl.exp(-gate))
    tl.store(Y + row * H + col, silu * up, col < H)


@triton.jit(do_not_specialize=["rows"])
def _swiglu_batched(
    X,
    Y,
    rows,
    H: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    LIMIT: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = (row[:, None] < rows) & (col[None, :] < H)
    gate = tl.load(X + row[:, None] * XS0 + col[None, :] * XS1, valid, other=0).to(
        tl.float32
    )
    up = tl.load(X + row[:, None] * XS0 + (H + col[None, :]) * XS1, valid, other=0).to(
        tl.float32
    )
    gate = tl.where(gate != gate, gate, tl.minimum(gate, LIMIT))  # noqa: PLR0124
    up = tl.where(
        up != up, up, tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
    )  # noqa: PLR0124
    silu = gate / (1.0 + tl.exp(-gate))
    tl.store(Y + row[:, None] * H + col[None, :], silu * up, valid)


def _prepare_output(gate_up, limit, block_size, out):
    if gate_up.device.type != "npu" or gate_up.dtype != torch.bfloat16:
        raise ValueError("Clamped SwiGLU requires BF16 NPU activations")
    if gate_up.ndim != 2 or gate_up.shape[1] % 2:
        raise ValueError("Expected gate_up [rows, 2H]")
    rows, width = gate_up.shape
    if rows > 65535:
        raise ValueError("SwiGLU row count exceeds the Ascend launch grid limit")
    hidden = width // 2
    if hidden % 32:
        raise ValueError("Hidden width must be a multiple of 32 for aligned stores")
    limit = float(limit)
    if math.isnan(limit) or limit <= 0:
        raise ValueError("Clamped SwiGLU requires a positive clamp limit")
    if block_size < 32 or block_size & (block_size - 1):
        raise ValueError("block_size must be a power of two >= 32")
    if out is not None and (
        out.shape != (rows, hidden)
        or out.dtype != gate_up.dtype
        or out.device != gate_up.device
        or not out.is_contiguous()
    ):
        raise ValueError("out must be a contiguous BF16 matrix on the input device")
    result = (
        out
        if out is not None
        else torch.empty((rows, hidden), dtype=gate_up.dtype, device=gate_up.device)
    )
    return rows, hidden, limit, result


def swiglu(gate_up, limit, *, block_size, out=None):
    """Apply clamped SwiGLU, using a power-of-two column tile and optional out."""
    rows, hidden, limit, result = _prepare_output(gate_up, limit, block_size, out)
    if rows and hidden:
        _swiglu[(rows, triton.cdiv(hidden, block_size))](
            gate_up,
            result,
            hidden,
            *gate_up.stride(),
            limit,
            block_size,
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
    return result


def swiglu_batched(gate_up, limit, *, block_size, rows_per_program, out=None):
    """Batch independent rows without changing FP32 SwiGLU intermediates.

    block_size and rows_per_program explicitly select the column/row tile.
    The output is BF16; an optional contiguous out receives the result.
    """
    if rows_per_program not in (1, 2, 4) or rows_per_program * block_size > 4096:
        raise ValueError(
            "SwiGLU row/column tile exceeds the validated local-buffer size"
        )
    rows, hidden, limit, result = _prepare_output(gate_up, limit, block_size, out)
    if rows and hidden:
        _swiglu_batched[
            (triton.cdiv(rows, rows_per_program), triton.cdiv(hidden, block_size))
        ](
            gate_up,
            result,
            rows,
            hidden,
            *gate_up.stride(),
            limit,
            block_size,
            rows_per_program,
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
    return result
