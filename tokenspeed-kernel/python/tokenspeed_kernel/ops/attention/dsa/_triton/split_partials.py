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

"""Merge of sparse attention partials over column splits of one top-k row."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _merge_split_partials_kernel(
    partial_out,
    partial_lse,
    split_lens,
    sink,
    out,
    lse_out,
    po_r,
    po_h,
    po_d,
    pl_r,
    pl_h,
    o_r,
    o_h,
    o_d,
    SPLITS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_SINK: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    heads = tl.num_programs(1)
    offsets = tl.arange(0, BLOCK).to(tl.int64)
    in_dim = offsets < DIM

    # A split with no live column, or one the kernel reports as having
    # attended to nothing (+inf), is the empty partial: weight zero. NaN stays
    # live so a corrupt key still poisons the row as it would unsplit.
    maximum = tl.full((), -float("inf"), tl.float32)
    for split in tl.static_range(SPLITS):
        part = row * SPLITS + split
        value = tl.load(partial_lse + part * pl_r + head * pl_h)
        live = (tl.load(split_lens + part) > 0) & (value != float("inf"))
        maximum = tl.maximum(maximum, tl.where(live, value, -float("inf")))
    safe_max = tl.where(maximum == -float("inf"), 0.0, maximum)

    denom = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((BLOCK,), tl.float32)
    for split in tl.static_range(SPLITS):
        part = row * SPLITS + split
        value = tl.load(partial_lse + part * pl_r + head * pl_h)
        live = (tl.load(split_lens + part) > 0) & (value != float("inf"))
        weight = tl.where(live, tl.exp(value - safe_max), 0.0)
        denom += weight
        values = tl.load(
            partial_out + part * po_r + head * po_h + offsets * po_d,
            mask=in_dim & live,
            other=0.0,
        ).to(tl.float32)
        acc += weight * values

    nonempty = denom != 0.0
    total = safe_max + tl.log(denom)
    scale = 1.0 / denom
    if HAS_SINK:
        # The sink is one extra softmax candidate without a value vector:
        # exp(total) / (exp(total) + exp(sink)).
        sink_logit = tl.load(sink + head).to(tl.float32)
        scale = scale / (1.0 + tl.exp(sink_logit - total))
    result = tl.where(nonempty, acc * scale, 0.0)
    tl.store(
        out + row * o_r + head * o_h + offsets * o_d,
        result.to(out.dtype.element_ty),
        mask=in_dim,
    )
    if RETURN_LSE:
        if HAS_SINK:
            sink_value = tl.load(sink + head).to(tl.float32)
            hi = tl.maximum(total, sink_value)
            total = hi + tl.log(tl.exp(total - hi) + tl.exp(sink_value - hi))
        tl.store(lse_out + row * heads + head, tl.where(nonempty, total, -float("inf")))


def merge_split_partials(
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
    split_lens: torch.Tensor,
    *,
    splits: int,
    heads: int,
    sink: torch.Tensor | None,
    out: torch.Tensor | None,
    return_lse: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Combine attention over consecutive column splits of every query row.

    Row ``r`` of the original launch was attended as the ``splits`` rows
    ``r * splits ... r * splits + splits - 1``, each over its own slice of the
    selected columns and without a sink. Their outputs are re-weighted by
    their LSE, then the sink is applied once to the merged softmax.

    Args:
        partial_out: CUDA no-sink outputs [rows * splits, padded_heads, dim],
            16-bit, possibly strided.
        partial_lse: Natural-log FP32 LSE [rows * splits, padded_heads].
        split_lens: Int32 live column count of every split [rows * splits].
        splits: Splits per original row.
        heads: Leading heads to merge; padded heads past it are ignored.
        sink: FP32 sink logits covering ``heads``, or explicitly None.
        out: Contiguous output buffer of ``rows * heads * dim`` elements in
            ``partial_out``'s dtype, written in place, or None.
        return_lse: Whether to also return the merged LSE.

    Returns:
        Output [rows, heads, dim] in ``partial_out``'s dtype (``out`` when
        given), and with ``return_lse`` the FP32 LSE [rows, heads], which
        counts the sink. Rows without a live key are zero with LSE ``-inf``.
    """
    if partial_out.ndim != 3 or partial_lse.shape != partial_out.shape[:2]:
        raise ValueError("split partial output and LSE shapes disagree")
    if splits <= 0 or partial_out.shape[0] % splits:
        raise ValueError("split partial rows are not a whole number of rows")
    if not 0 < heads <= partial_out.shape[1]:
        raise ValueError("split partial merge head count is out of range")
    if split_lens.shape != (partial_out.shape[0],) or split_lens.dtype != torch.int32:
        raise ValueError("split lengths must be int32 [rows * splits]")
    if partial_lse.dtype != torch.float32:
        raise TypeError("split partial LSE must be FP32")
    if sink is not None and (
        sink.dtype != torch.float32 or sink.numel() < heads or not sink.is_contiguous()
    ):
        raise ValueError("split partial sink must be contiguous FP32 over the heads")
    rows = partial_out.shape[0] // splits
    dim = partial_out.shape[2]
    if out is None:
        result = partial_out.new_empty((rows, heads, dim))
    else:
        if (
            out.numel() != rows * heads * dim
            or out.dtype != partial_out.dtype
            or not out.is_contiguous()
        ):
            raise ValueError(
                "split partial output buffer must be a contiguous "
                f"{partial_out.dtype} tensor of {rows * heads * dim} elements, "
                f"got {out.dtype} {tuple(out.shape)} with strides {out.stride()}"
            )
        result = out.view(rows, heads, dim)
    lse = (
        torch.empty((rows, heads), dtype=torch.float32, device=partial_out.device)
        if return_lse
        else None
    )
    if rows:
        _merge_split_partials_kernel[(rows, heads)](
            partial_out,
            partial_lse,
            split_lens.contiguous(),
            sink,
            result,
            lse,
            *partial_out.stride(),
            *partial_lse.stride(),
            *result.stride(),
            SPLITS=splits,
            DIM=dim,
            BLOCK=triton.next_power_of_2(dim),
            HAS_SINK=sink is not None,
            RETURN_LSE=return_lse,
            num_warps=4,
        )
    output = out if out is not None else result
    return (output, lse) if return_lse else output
