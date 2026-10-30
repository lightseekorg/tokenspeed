# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-FileCopyrightText: Copyright (c) 2024, Tri Dao
#
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

"""Triton kernel of the gated RMSNorm for the linear-attention blocks:
``norm(x) * gate(z)`` or ``norm(x * gate(z))`` in one launch, with ``silu`` or
``sigmoid`` as the gate. Inference only (no backward). Launched by
``tokenspeed_kernel.ops.layernorm.triton.triton_gated_rmsnorm``."""

from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.warmup import register_triton_warmup

__all__ = ["gated_rmsnorm_kernel"]


@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.jit
def gated_rmsnorm_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    Z,  # pointer to the other branch
    FP8_SCALE,  # static FP8 dequant scale of the consumer
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_z_row,
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    SIGMOID_GATE: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    WEIGHTS_INDEPENDENT: tl.constexpr,
    OUT_FP8: tl.constexpr,
    SHARED_WEIGHT: tl.constexpr,
):
    # Each program normalizes BLOCK_M rows of one group, one warp per row.
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    group = tl.program_id(1)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    mask = (rows < M)[:, None] & col_mask[None, :]
    # Row offsets in int64: z may be a view with a projection-wide row stride.
    rows = rows.to(tl.int64)
    offsets = group * N + cols[None, :]
    if not SHARED_WEIGHT:
        W += group * N
    if WEIGHTS_INDEPENDENT:
        w = tl.load(W + cols, mask=col_mask).to(tl.float32)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if not WEIGHTS_INDEPENDENT:
        w = tl.load(W + cols, mask=col_mask).to(tl.float32)
    x = tl.load(X + rows[:, None] * stride_x_row + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    if HAS_Z:
        z = tl.load(Z + rows[:, None] * stride_z_row + offsets, mask=mask).to(
            tl.float32
        )
    if HAS_Z and not NORM_BEFORE_GATE:
        x *= tl.sigmoid(z) if SIGMOID_GATE else z * tl.sigmoid(z)
    xbar = tl.where(mask, x, 0.0)
    var = tl.sum(xbar * xbar, axis=1) / N
    rstd = 1 / tl.sqrt(var + eps)
    y = x * rstd[:, None] * w[None, :]
    if HAS_Z and NORM_BEFORE_GATE:
        y *= tl.sigmoid(z) if SIGMOID_GATE else z * tl.sigmoid(z)
    if OUT_FP8:
        # Quantize the activation-dtype output, as the consumer would.
        y = y.to(X.dtype.element_ty).to(tl.float32)
        y = tl.clamp(y * (1.0 / tl.load(FP8_SCALE).to(tl.float32)), -448.0, 448.0)
    tl.store(
        Y + rows[:, None] * stride_y_row + offsets, y.to(Y.dtype.element_ty), mask=mask
    )


register_triton_warmup(gated_rmsnorm_kernel.fn, variants=({"M": (1, 16, 17)},))
