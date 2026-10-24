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

"""Portable Triton kernels next to the decode megakernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

# Rows of one launch of the MTP draft front: a decode step's rows.
MTP_FRONT_ROWS = 4
_STREAMS = 4  # hyper-connection streams of the hidden row


@triton.jit
def _qwen4_exp_mtp_front_kernel(
    ids,
    emb,
    emb_stride,
    hidden,
    hidden_stride,
    norm_e,
    norm_h,
    w_e,
    w_h,
    out,
    out_stride,
    rows,
    EPS: tl.constexpr,
    H: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """Columns ``BN pid ..`` of every row's MTP draft input.

    MFMA row ``m`` is stream ``m // 4`` of row ``m % 4``. Rows at or past
    ``rows`` are padding. Each row's sums and products do not depend on
    ``rows``, so a row's result equals its one-row launch.
    """
    pid = tl.program_id(0)
    n = pid * BN + tl.arange(0, BN)
    m = tl.arange(0, 16)
    r = m % 4
    s = m // 4
    live = r < rows
    token = tl.load(ids + r, mask=live, other=0).to(tl.int64)
    e_row = emb + token * emb_stride
    h_row = hidden + r.to(tl.int64) * hidden_stride + s * H
    kk = tl.arange(0, BK)

    # The rows' sums of squares: the embedding row, and each stream's part
    # of the hidden row (its norm spans all 4 streams).
    sum_e = tl.zeros([16], tl.float32)
    sum_h = tl.zeros([16], tl.float32)
    for k0 in range(0, H, BK):
        k = k0 + kk
        xe = tl.load(e_row[:, None] + k[None, :], mask=live[:, None], other=0.0)
        xh = tl.load(h_row[:, None] + k[None, :], mask=live[:, None], other=0.0)
        xe = xe.to(tl.float32)
        xh = xh.to(tl.float32)
        sum_e += tl.sum(xe * xe, 1)
        sum_h += tl.sum(xh * xh, 1)
    # Each row's total over its 4 streams, back on all 4 of its MFMA rows.
    per_row = tl.sum(tl.reshape(sum_h, [4, 4]), 0)
    total_h = tl.reshape(tl.broadcast_to(per_row[None, :], [4, 4]), [16])
    inv_e = tl.rsqrt(sum_e * (1.0 / H) + EPS)
    inv_h = tl.rsqrt(total_h * (1.0 / (4 * H)) + EPS)

    # Gemma RMSNorm (FP32 rows times rsqrt, times the FP32 1 + weight, one
    # BF16 cast), then the two BF16 linears with FP32 accumulation.
    acc_e = tl.zeros([16, BN], tl.float32)
    acc_h = tl.zeros([16, BN], tl.float32)
    for k0 in range(0, H, BK):
        k = k0 + kk
        xe = tl.load(e_row[:, None] + k[None, :], mask=live[:, None], other=0.0)
        xh = tl.load(h_row[:, None] + k[None, :], mask=live[:, None], other=0.0)
        ge = 1.0 + tl.load(norm_e + k).to(tl.float32)
        gh = 1.0 + tl.load(norm_h + s[:, None] * H + k[None, :]).to(tl.float32)
        xe = ((xe.to(tl.float32) * inv_e[:, None]) * ge[None, :]).to(tl.bfloat16)
        xh = ((xh.to(tl.float32) * inv_h[:, None]) * gh).to(tl.bfloat16)
        we = tl.load(w_e + n[None, :] * H + k[:, None])
        wh = tl.load(w_h + n[None, :] * H + k[:, None])
        acc_e = tl.dot(xe, we, acc_e)
        acc_h = tl.dot(xh, wh, acc_h)
    # Each linear's output is BF16; their sum rounds once more.
    y = acc_h.to(tl.bfloat16).to(tl.float32) + acc_e.to(tl.bfloat16).to(tl.float32)
    dst = out + r.to(tl.int64)[:, None] * out_stride + s[:, None] * H + n[None, :]
    tl.store(dst, y.to(tl.bfloat16), mask=live[:, None])


@register_kernel(
    "megakernel",
    "qwen4_exp_mtp_front",
    name="triton_qwen4_exp_mtp_front",
    solution="triton",
    signatures=format_signatures("hidden", "dense", {torch.bfloat16}),
    priority=Priority.PORTABLE,
    traits={"rows": frozenset(range(1, MTP_FRONT_ROWS + 1))},
)
def triton_qwen4_exp_mtp_front(
    input_ids: torch.Tensor,
    embedding: torch.Tensor,
    hidden: torch.Tensor,
    norm_embedding: torch.Tensor,
    norm_hidden: torch.Tensor,
    fc_embedding: torch.Tensor,
    fc_hidden: torch.Tensor,
    eps: float,
    out: torch.Tensor,
) -> None:
    """See :func:`tokenspeed_kernel.ops.megakernel.qwen4_exp_mtp_front`."""
    size = embedding.shape[1]
    rows = input_ids.shape[0]
    _qwen4_exp_mtp_front_kernel[(size // 16,)](
        input_ids,
        embedding,
        embedding.stride(0),
        hidden,
        hidden.stride(0),
        norm_embedding,
        norm_hidden,
        fc_embedding,
        fc_hidden,
        out,
        out.stride(0),
        rows,
        EPS=eps,
        H=size,
        BN=16,
        # 512 measured 4 us faster than 256 cold (fewer, longer weight runs).
        BK=512,
        num_warps=4,
    )


__all__ = ["MTP_FRONT_ROWS", "triton_qwen4_exp_mtp_front"]
