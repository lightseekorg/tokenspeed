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

"""GFX1250 length-128 Sylvester Hadamard transform."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton

cdna5 = gl.amd.cdna5

__all__ = [
    "gluon_hadamard_transform_128_gfx1250",
    "launch_gluon_hadamard_transform_128_gfx1250",
]

_BLOCK_M = 16
_ROW_BLOCK_MIN_ROWS = 256
_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _hadamard_launch_metadata(grid, kernel, args):
    """Report the Hadamard contraction and the row read plus write."""
    n_rows = args["n_rows"]
    terms = 3 if args["THREE_TERM"] else 1
    return {
        "name": kernel.name,
        "flops16": terms * 2 * n_rows * 128 * 128,
        "bytes": n_rows
        * 128
        * (args["x_ptr"].element_size() + args["out_ptr"].element_size()),
    }


@gluon.jit
def _accumulate_bf16(acc, lhs, rhs, a_layout, b_layout):
    return cdna5.wmma(
        gl.convert_layout(lhs, a_layout),
        gl.convert_layout(rhs, b_layout),
        acc,
    )


@gluon.jit
def _sylvester_signs(k0, n0, layout):
    """Return the ``[32, 16]`` Sylvester signs for one K/N tile.

    ``H[k, n] = (-1) ** popcount(k & n)``. Those entries are exact ``+-1``
    in BF16.
    """
    k_idx = k0 + gl.arange(0, 32, layout=gl.SliceLayout(1, layout))
    n_idx = n0 + gl.arange(0, 16, layout=gl.SliceLayout(0, layout))
    bits = k_idx[:, None] & n_idx[None, :]
    parity = bits ^ (bits >> 1)
    parity = parity ^ (parity >> 2)
    parity = parity ^ (parity >> 4)
    return gl.where((parity & 1) == 0, 1.0, -1.0).to(gl.bfloat16)


@gluon.jit(
    launch_metadata=_hadamard_launch_metadata,
    do_not_specialize=["n_rows"],
)
def gluon_hadamard_transform_128_gfx1250(
    x_ptr,
    out_ptr,
    n_rows,
    scale,
    THREE_TERM: gl.constexpr,
):
    """Apply ``x @ H * scale`` for one 16x16 output tile of a length-128 row.

    ``H`` is the Sylvester Hadamard. BF16 rows use one BF16 WMMA. FP16 and
    FP32 rows split each value into a BF16 rounding and two BF16 remainders
    that sum back to it, so the products match the FP32 value up to WMMA
    accumulation order. ``scale`` is a runtime scalar.
    """
    block_m: gl.constexpr = 16
    row0 = gl.program_id(0) * block_m
    n0 = gl.program_id(1) * block_m
    wmma_layout: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=[],
        reg_bases=[],
        instr_shape=[16, 16, 32],
    )
    a_layout: gl.constexpr = gl.DotOperandLayout(0, wmma_layout, k_width=8)
    b_layout: gl.constexpr = gl.DotOperandLayout(1, wmma_layout, k_width=8)
    a_blocked: gl.constexpr = gl.BlockedLayout([1, 16], [16, 2], [1, 1], [1, 0])
    b_blocked: gl.constexpr = gl.BlockedLayout([16, 1], [2, 16], [1, 1], [1, 0])

    rows = gl.arange(0, block_m, layout=gl.SliceLayout(1, a_blocked))
    row_mask = (row0 + rows)[:, None] < n_rows
    acc = gl.zeros([block_m, block_m], gl.float32, wmma_layout)
    for k_tile in gl.static_range(4):
        cols = k_tile * 32 + gl.arange(0, 32, layout=gl.SliceLayout(0, a_blocked))
        offsets = (row0 + rows)[:, None].to(gl.int64) * 128 + cols[None, :]
        values = gl.load(x_ptr + offsets, mask=row_mask, other=0.0).to(gl.float32)
        signs = _sylvester_signs(k_tile * 32, n0, b_blocked)
        if THREE_TERM:
            high = values.to(gl.bfloat16)
            rest = values - high.to(gl.float32)
            middle = rest.to(gl.bfloat16)
            low = (rest - middle.to(gl.float32)).to(gl.bfloat16)
            acc = _accumulate_bf16(acc, high, signs, a_layout, b_layout)
            acc = _accumulate_bf16(acc, middle, signs, a_layout, b_layout)
            acc = _accumulate_bf16(acc, low, signs, a_layout, b_layout)
        else:
            acc = _accumulate_bf16(
                acc, values.to(gl.bfloat16), signs, a_layout, b_layout
            )

    out_rows = row0 + gl.arange(0, block_m, layout=gl.SliceLayout(1, wmma_layout))
    out_cols = n0 + gl.arange(0, block_m, layout=gl.SliceLayout(0, wmma_layout))
    out_offsets = out_rows[:, None].to(gl.int64) * 128 + out_cols[None, :]
    stored = (acc * scale).to(out_ptr.dtype.element_ty)
    gl.store(out_ptr + out_offsets, stored, mask=out_rows[:, None] < n_rows)


def launch_gluon_hadamard_transform_128_gfx1250(
    x: torch.Tensor,
    *,
    scale: float = 1.0,
) -> torch.Tensor:
    """Apply a length-128 Sylvester Hadamard transform along the last dim.

    Batches below 256 rows use the portable one-row kernel. Larger batches
    use the WMMA kernel. The row count of that kernel is a runtime argument.

    Args:
        x: CUDA FP16, BF16, or FP32 tensor whose last dimension is 128.
        scale: Multiplicative output scale applied after the transform.

    Returns:
        Tensor with the same shape and dtype as ``x``.
    """
    if x.shape[-1] != 128:
        raise ValueError(
            "gluon_hadamard_transform_128_gfx1250 requires last dim 128, "
            f"got {x.shape[-1]}"
        )
    if not x.is_cuda:
        raise RuntimeError(
            "gluon_hadamard_transform_128_gfx1250 requires a CUDA tensor"
        )
    if x.dtype not in _DTYPES:
        raise TypeError(
            f"gluon_hadamard_transform_128_gfx1250 does not support dtype {x.dtype}"
        )

    shape = x.shape
    rows = x.reshape(-1, 128)
    if rows.shape[0] < _ROW_BLOCK_MIN_ROWS:
        from tokenspeed_kernel.ops.transform.triton import triton_hadamard_transform_128

        return triton_hadamard_transform_128(x, scale=scale)

    x_2d = rows.contiguous()
    out = torch.empty_like(x_2d)
    n_rows = x_2d.shape[0]
    gluon_hadamard_transform_128_gfx1250[
        (triton.cdiv(n_rows, _BLOCK_M), 128 // _BLOCK_M)
    ](
        x_2d,
        out,
        n_rows,
        float(scale),
        THREE_TERM=x_2d.dtype != torch.bfloat16,
        num_warps=1,
    )
    return out.reshape(shape)
