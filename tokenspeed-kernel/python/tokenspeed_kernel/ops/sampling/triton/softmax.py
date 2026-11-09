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

"""Row softmax at a per-row temperature (the ``sampling`` ``softmax`` op)."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

__all__ = ["softmax_rows", "triton_softmax"]

_SOFTMAX_BLOCK = 4096


@triton.jit
def _row_inv_temperature(temperature_ptr, index_ptr, row, rows_per_index):
    """``1 / T`` of a row, 0 at ``T == 0``; ``T`` read at ``index[row //
    rows_per_index]`` when ``index_ptr`` is given, else at ``row``."""
    if index_ptr is None:
        temperature = tl.load(temperature_ptr + row)
    else:
        temperature = tl.load(
            temperature_ptr + tl.load(index_ptr + row // rows_per_index).to(tl.int64)
        )
    temperature = temperature.to(tl.float32)
    return tl.where(temperature == 0.0, 0.0, 1.0 / temperature)


@triton.jit(do_not_specialize=["rows_per_index", "logits_row_stride", "out_row_stride"])
def _softmax_rows_kernel(
    logits_ptr,
    temperature_ptr,
    index_ptr,
    out_ptr,
    rows_per_index,
    logits_row_stride,
    out_row_stride,
    vocab_size,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    inv_temp = _row_inv_temperature(temperature_ptr, index_ptr, row, rows_per_index)
    logits_row = logits_ptr + row.to(tl.int64) * logits_row_stride
    out_row = out_ptr + row.to(tl.int64) * out_row_stride
    cols = tl.arange(0, BLOCK)

    # Per-lane running max and sum of exp(x - max), merged at the end.
    run_max = tl.full((BLOCK,), float("-inf"), tl.float32)
    run_sum = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, vocab_size, BLOCK):
        mask = start + cols < vocab_size
        x = tl.load(logits_row + start + cols, mask=mask, other=float("-inf"))
        x = tl.where(mask, x.to(tl.float32) * inv_temp, float("-inf"))
        new_max = tl.maximum(run_max, x)
        safe = tl.where(new_max == float("-inf"), 0.0, new_max)
        run_sum = run_sum * tl.exp(run_max - safe) + tl.exp(x - safe)
        run_max = new_max
    row_max = tl.max(run_max, axis=0)
    safe_max = tl.where(row_max == float("-inf"), 0.0, row_max)
    total = tl.sum(run_sum * tl.exp(run_max - safe_max), axis=0)
    inv_total = 1.0 / total

    for start in range(0, vocab_size, BLOCK):
        mask = start + cols < vocab_size
        x = tl.load(logits_row + start + cols, mask=mask, other=float("-inf"))
        x = x.to(tl.float32) * inv_temp
        tl.store(out_row + start + cols, tl.exp(x - safe_max) * inv_total, mask=mask)


def softmax_rows(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    out: torch.Tensor,
    *,
    index: torch.Tensor | None,
    rows_per_index: int,
) -> torch.Tensor:
    """``out[r] = softmax(logits[r] / T_r)`` in fp32.

    Args:
        logits: ``[rows, V]`` fp32/fp16/bf16, unit column stride.
        temperature: fp32 temperatures: ``[rows]`` when ``index`` is None,
            else indexed by ``index``. ``T == 0`` scales the logits by 0, as
            the CUDA softmax does.
        out: ``[rows, V]`` fp32, unit column stride.
        index: ``[rows // rows_per_index]`` int32/int64 entry of
            ``temperature`` for each run of ``rows_per_index`` rows (pool
            slots of a verify window), or None.
        rows_per_index: Rows sharing one ``index`` entry; 1 without one.

    Returns:
        ``out``.
    """
    rows, vocab_size = logits.shape
    if logits.stride(1) != 1 or out.stride(1) != 1 or out.shape != logits.shape:
        raise ValueError(
            f"softmax_rows: logits {tuple(logits.shape)} and out "
            f"{tuple(out.shape)} must match with unit column strides"
        )
    if out.dtype != torch.float32 or temperature.dtype != torch.float32:
        raise ValueError("softmax_rows: out and temperature must be fp32")
    if index is None and temperature.numel() < rows:
        raise ValueError(
            f"softmax_rows: {temperature.numel()} temperatures for {rows} rows"
        )
    if rows == 0:
        return out
    _softmax_rows_kernel[(rows,)](
        logits,
        temperature,
        index,
        out,
        rows_per_index,
        logits.stride(0),
        out.stride(0),
        vocab_size,
        BLOCK=_SOFTMAX_BLOCK,
        num_warps=8,
    )
    return out


@register_kernel(
    "sampling",
    "softmax",
    name="triton_softmax",
    solution="triton",
    signatures=format_signatures(
        "logits", "dense", {torch.float32, torch.float16, torch.bfloat16}
    ),
    priority=Priority.PORTABLE,
)
def triton_softmax(
    logits: torch.Tensor,
    *,
    temperature: torch.Tensor,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """fp32 ``softmax(logits / temperature)`` per row; see the family op."""
    logits = logits.contiguous()
    if out is None:
        out = torch.empty(logits.shape, dtype=torch.float32, device=logits.device)
    return softmax_rows(
        logits, temperature.reshape(-1), out, index=None, rows_per_index=1
    )
