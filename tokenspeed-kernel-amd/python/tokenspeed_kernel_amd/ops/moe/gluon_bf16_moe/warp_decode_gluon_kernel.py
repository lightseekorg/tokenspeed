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

"""Pure-Gluon warp-reduce GEMV decode path for the bf16 MoE (small M).

At decode the activation is a vector, so both GEMMs are GEMVs and MFMA would
waste 15/16 of its tile. Instead each warp owns a few output elements and its
64 lanes stride the reduction dim, so ``gl.sum(axis=1)`` becomes a warp reduce.
No sort (the expert is read straight from ``topk_ids``), no MFMA, no LDS.

  Stage 1: gate/up GEMV + fused SwiGLU, one program per (routed slot, neuron block).
  Stage 2: down GEMV with the topk combine fused in, one program per (token, D block).
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton

_LANES = gl.constexpr(64)  # wavefront width (reduction lanes)


@gluon.jit
def _stage1_warp_gemv_gluon(
    x_ptr,  # hidden (num_tokens, D)         bf16
    w1_ptr,  # w1     (E, 2*I_r, D)           bf16
    out_ptr,  # inter  (num_tokens*topk, I_r)  bf16
    topk_ids_ptr,  # (num_tokens, topk) int32
    D,
    I_r,
    num_tokens,
    top_k,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    stride_tit,
    stride_tis,
    BLOCK_N: gl.constexpr,  # neurons per program
    BLOCK_K: gl.constexpr,  # D reduction tile
    NUM_WARPS: gl.constexpr,
):
    pid = gl.program_id(0)
    num_pid_n = gl.cdiv(I_r, BLOCK_N)
    slot = pid // num_pid_n
    pid_n = pid % num_pid_n
    token = slot // top_k
    e = gl.load(topk_ids_ptr + token * stride_tit + (slot % top_k) * stride_tis)

    # warps span the neurons, lanes span the D reduction -> gl.sum(axis=1) is a warp reduce
    blk: gl.constexpr = gl.BlockedLayout(
        [(BLOCK_N + NUM_WARPS - 1) // NUM_WARPS, BLOCK_K // _LANES],
        [1, _LANES],
        [NUM_WARPS, 1],
        [1, 0],
    )
    n_layout: gl.constexpr = gl.SliceLayout(1, blk)
    k_layout: gl.constexpr = gl.SliceLayout(0, blk)

    offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=n_layout)
    n_mask = offs_n < I_r
    x_row = x_ptr + token.to(gl.int64) * stride_xm
    we = w1_ptr + e.to(gl.int64) * stride_we
    gate_row = we + offs_n[:, None].to(gl.int64) * stride_wn  # gate at rows [0, I_r)
    up_row = (
        we + (I_r + offs_n)[:, None].to(gl.int64) * stride_wn
    )  # up   at rows [I_r, 2*I_r)

    acc_g = gl.zeros([BLOCK_N], gl.float32, n_layout)
    acc_u = gl.zeros([BLOCK_N], gl.float32, n_layout)
    for k0 in range(0, D, BLOCK_K):
        offs_k = k0 + gl.arange(0, BLOCK_K, layout=k_layout)
        k_mask = offs_k < D
        tile_mask = n_mask[:, None] & k_mask[None, :]
        wk = offs_k[None, :].to(gl.int64) * stride_wk
        x = gl.load(x_row + offs_k * stride_xk, mask=k_mask, other=0.0).to(gl.float32)
        wg = gl.load(gate_row + wk, mask=tile_mask, other=0.0).to(gl.float32)
        wu = gl.load(up_row + wk, mask=tile_mask, other=0.0).to(gl.float32)
        acc_g += gl.sum(wg * x[None, :], axis=1)
        acc_u += gl.sum(wu * x[None, :], axis=1)

    inter = acc_g * (1.0 / (1.0 + gl.exp(-acc_g))) * acc_u  # SwiGLU: silu(gate) * up
    gl.store(
        out_ptr + slot.to(gl.int64) * stride_om + offs_n * stride_on,
        inter.to(out_ptr.dtype.element_ty),
        mask=n_mask,
    )


def invoke_stage1_warp_decode_gluon(
    hidden_states,
    w1,
    topk_ids,
    out,
    topk,
    BLOCK_N: int = 4,
    BLOCK_K: int = 512,
    num_warps: int = 4,
):
    assert hidden_states.dtype == torch.bfloat16 and w1.dtype == torch.bfloat16
    assert out.dtype == torch.bfloat16
    num_tokens, D = hidden_states.shape
    E, two_I, Dw = w1.shape
    assert Dw == D and two_I % 2 == 0 and D % BLOCK_K == 0
    I_r = two_I // 2
    assert out.shape == (num_tokens * topk, I_r)
    topk_ids = topk_ids.to(torch.int32)
    grid = (num_tokens * topk * triton.cdiv(I_r, BLOCK_N),)
    _stage1_warp_gemv_gluon[grid](
        hidden_states,
        w1,
        out,
        topk_ids,
        D,
        I_r,
        num_tokens,
        topk,
        hidden_states.stride(0),
        hidden_states.stride(1),
        w1.stride(0),
        w1.stride(1),
        w1.stride(2),
        out.stride(0),
        out.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_WARPS=num_warps,
        num_warps=num_warps,
    )
    return out


@gluon.jit
def _stage2_warp_gemv_gluon(
    inter_ptr,  # inter (num_tokens*topk, I_r)  bf16
    w2_ptr,  # w2    (E, D, I_r)             bf16
    y_ptr,  # out   (num_tokens, D)         bf16
    topk_ids_ptr,  # (num_tokens, topk) int32
    topk_weights_ptr,  # (num_tokens, topk) float32
    D,
    I_r,
    num_tokens,
    top_k,
    stride_im,
    stride_ik,
    stride_we,
    stride_wd,
    stride_wk,
    stride_yt,
    stride_yd,
    stride_tit,
    stride_tis,
    stride_twt,
    stride_tws,
    BLOCK_D: gl.constexpr,  # output dims per program
    BLOCK_K: gl.constexpr,  # I_r reduction tile
    NUM_WARPS: gl.constexpr,
):
    pid = gl.program_id(0)
    num_pid_d = gl.cdiv(D, BLOCK_D)
    token = pid // num_pid_d
    pid_d = pid % num_pid_d

    # warps span the output dims, lanes span the I_r reduction -> gl.sum(axis=1) is a warp reduce
    blk: gl.constexpr = gl.BlockedLayout(
        [(BLOCK_D + NUM_WARPS - 1) // NUM_WARPS, BLOCK_K // _LANES],
        [1, _LANES],
        [NUM_WARPS, 1],
        [1, 0],
    )
    d_layout: gl.constexpr = gl.SliceLayout(1, blk)
    k_layout: gl.constexpr = gl.SliceLayout(0, blk)

    offs_d = pid_d * BLOCK_D + gl.arange(0, BLOCK_D, layout=d_layout)
    d_mask = offs_d < D

    acc = gl.zeros([BLOCK_D], gl.float32, d_layout)
    for s in range(0, top_k):  # fuse the topk combine
        e = gl.load(topk_ids_ptr + token * stride_tit + s * stride_tis)
        prob = gl.load(topk_weights_ptr + token * stride_twt + s * stride_tws).to(
            gl.float32
        )
        in_row = inter_ptr + (token * top_k + s).to(gl.int64) * stride_im
        w_row = (
            w2_ptr
            + e.to(gl.int64) * stride_we
            + offs_d[:, None].to(gl.int64) * stride_wd
        )
        dot = gl.zeros([BLOCK_D], gl.float32, d_layout)
        for k0 in range(0, I_r, BLOCK_K):
            offs_k = k0 + gl.arange(0, BLOCK_K, layout=k_layout)
            k_mask = offs_k < I_r
            tile_mask = d_mask[:, None] & k_mask[None, :]
            v = gl.load(in_row + offs_k * stride_ik, mask=k_mask, other=0.0).to(
                gl.float32
            )
            w = gl.load(
                w_row + offs_k[None, :].to(gl.int64) * stride_wk,
                mask=tile_mask,
                other=0.0,
            ).to(gl.float32)
            dot += gl.sum(w * v[None, :], axis=1)
        acc += prob * dot

    gl.store(
        y_ptr + token * stride_yt + offs_d * stride_yd,
        acc.to(y_ptr.dtype.element_ty),
        mask=d_mask,
    )


def invoke_stage2_warp_decode_gluon(
    inter_states,
    w2,
    topk_ids,
    topk_weights,
    out,
    topk,
    BLOCK_D: int = 64,
    BLOCK_K: int = 256,
    num_warps: int = 4,
):
    assert inter_states.dtype == torch.bfloat16 and w2.dtype == torch.bfloat16
    assert out.dtype == torch.bfloat16
    E, D, I_r = w2.shape
    num_tokens = out.shape[0]
    assert out.shape == (num_tokens, D)
    assert inter_states.shape == (num_tokens * topk, I_r) and I_r % BLOCK_K == 0
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = topk_weights.to(torch.float32)
    grid = (num_tokens * triton.cdiv(D, BLOCK_D),)
    _stage2_warp_gemv_gluon[grid](
        inter_states,
        w2,
        out,
        topk_ids,
        topk_weights,
        D,
        I_r,
        num_tokens,
        topk,
        inter_states.stride(0),
        inter_states.stride(1),
        w2.stride(0),
        w2.stride(1),
        w2.stride(2),
        out.stride(0),
        out.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
        NUM_WARPS=num_warps,
        num_warps=num_warps,
    )
    return out
