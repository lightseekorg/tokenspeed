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
from tokenspeed_kernel._triton import libdevice, tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_FP8_DTYPES = frozenset({torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz})
_MLA_PREFILL_DTYPES = frozenset({torch.float16, torch.bfloat16}) | _FP8_DTYPES


@triton.jit
def _mla_prefill_kernel(
    Q,
    K,
    V,
    O,
    LSE,
    cu_seqlens_q,
    cu_seqlens_kv,
    sm_scale,
    kv_group_num,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_lse_bs,
    stride_lse_h,
    logit_cap: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_LSE: tl.constexpr,
):
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    q_start = tl.load(cu_seqlens_q + cur_seq)
    q_len = tl.load(cu_seqlens_q + cur_seq + 1) - q_start
    kv_start = tl.load(cu_seqlens_kv + cur_seq)
    kv_len = tl.load(cu_seqlens_kv + cur_seq + 1) - kv_start
    q_causal_start = tl.maximum(kv_len - q_len, 0)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    q_offsets_m = cur_block_m * BLOCK_M + offs_m
    mask_m = q_offsets_m < q_len
    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv

    offs_q = (
        (q_start + q_offsets_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(Q + offs_q, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lq
        offs_qpe = (
            (q_start + q_offsets_m[:, None]) * stride_qbs
            + cur_head * stride_qh
            + offs_dpe[None, :]
        )
        qpe = tl.load(Q + offs_qpe, mask=mask_m[:, None] & mask_dpe[None, :], other=0.0)

    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    for start_n in range(0, kv_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        kv_offsets_n = start_n + offs_n
        mask_n = kv_offsets_n < kv_len
        final_mask = mask_m[:, None] & mask_n[None, :]

        if IS_CAUSAL:
            query_positions = q_causal_start + q_offsets_m[:, None]
            key_positions = kv_offsets_n[None, :]
            final_mask &= query_positions >= key_positions

        offs_k = (
            (kv_start + kv_offsets_n[None, :]) * stride_kbs
            + cur_kv_head * stride_kh
            + offs_d[:, None]
        )
        k = tl.load(K + offs_k, mask=mask_n[None, :] & mask_d[:, None], other=0.0)

        qk = tl.dot(q.to(k.dtype), k)

        if BLOCK_DPE > 0:
            offs_kpe = (
                (kv_start + kv_offsets_n[None, :]) * stride_kbs
                + cur_kv_head * stride_kh
                + offs_dpe[:, None]
            )
            kpe = tl.load(
                K + offs_kpe, mask=mask_n[None, :] & mask_dpe[:, None], other=0.0
            )
            qk += tl.dot(qpe.to(kpe.dtype), kpe)

        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * libdevice.tanh(qk / logit_cap)

        qk = tl.where(final_mask, qk, float("-inf"))

        row_max = tl.max(qk, 1)
        row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
        n_e_max = tl.maximum(row_max_fixed, e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        deno = deno * re_scale + tl.sum(p, 1)

        offs_v = (
            (kv_start + kv_offsets_n[:, None]) * stride_vbs
            + cur_kv_head * stride_vh
            + offs_dv[None, :]
        )
        v = tl.load(V + offs_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0)
        p = p.to(v.dtype)
        acc = acc * re_scale[:, None] + tl.dot(p, v)
        e_max = n_e_max

    safe_deno = tl.where(deno > 0.0, deno, 1.0)
    offs_o = (
        (q_start + q_offsets_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
    tl.store(
        O + offs_o,
        acc / safe_deno[:, None],
        mask=mask_m[:, None] & mask_dv[None, :],
    )

    if HAS_LSE:
        offs_lse = (q_start + q_offsets_m) * stride_lse_bs + cur_head * stride_lse_h
        lse = tl.where(deno > 0.0, tl.log(deno) + e_max, float("-inf"))
        tl.store(LSE + offs_lse, lse, mask=mask_m)


def mla_prefill_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    softmax_scale: float,
    *,
    is_causal: bool,
    logit_cap: float = 0.0,
    lse: torch.Tensor | None = None,
) -> None:
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(
            f"q/k head dims must match, got {q.shape[-1]} and {k.shape[-1]}"
        )
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError(
            "num_q_heads must be divisible by num_kv_heads, "
            f"got {q.shape[1]} and {k.shape[1]}"
        )
    if out.shape != (q.shape[0], q.shape[1], v.shape[-1]):
        raise ValueError(
            f"out shape must be {(q.shape[0], q.shape[1], v.shape[-1])}, "
            f"got {tuple(out.shape)}"
        )
    for name, tensor in (("q", q), ("k", k), ("v", v), ("out", out)):
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name} must have contiguous last dimension")
    if lse is not None and lse.shape != (q.shape[0], q.shape[1]):
        raise ValueError(
            f"lse shape must be {(q.shape[0], q.shape[1])}, got {tuple(lse.shape)}"
        )

    q_head_dim = q.shape[-1]
    v_head_dim = v.shape[-1]

    if q_head_dim == 576:
        block_dmodel = 512
        block_dpe = 64
    elif q_head_dim == 288:
        block_dmodel = 256
        block_dpe = 32
    elif q_head_dim == 192:
        block_dmodel = 128
        block_dpe = 64
    else:
        block_dmodel = triton.next_power_of_2(q_head_dim)
        block_dpe = 0
    block_dv = triton.next_power_of_2(v_head_dim)
    block_m, block_n = (64, 64)
    num_warps = 4

    lse_arg = lse if lse is not None else out
    grid = (cu_seqlens_q.shape[0] - 1, q.shape[1], triton.cdiv(max_seqlen_q, block_m))

    _mla_prefill_kernel[grid](
        q,
        k,
        v,
        out,
        lse_arg,
        cu_seqlens_q,
        cu_seqlens_kv,
        softmax_scale,
        q.shape[1] // k.shape[1],
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        out.stride(0),
        out.stride(1),
        lse_arg.stride(0),
        lse_arg.stride(1),
        logit_cap=logit_cap,
        Lq=q_head_dim,
        Lv=v_head_dim,
        BLOCK_DMODEL=block_dmodel,
        BLOCK_DPE=block_dpe,
        BLOCK_DV=block_dv,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        IS_CAUSAL=is_causal,
        HAS_LSE=lse is not None,
        num_warps=num_warps,
        num_stages=1,
    )


@register_kernel(
    "attention",
    "mla_prefill",
    name="triton_mla_prefill",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures(("q", "k", "v"), "dense", _MLA_PREFILL_DTYPES),
    priority=Priority.PORTABLE,
    traits={
        "is_causal": frozenset({False, True}),
        "support_logit_cap": frozenset({False, True}),
        "return_lse": frozenset({False, True}),
    },
    tags={"portability"},
)
def triton_mla_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    softmax_scale: float,
    *,
    is_causal: bool = True,
    logit_cap: float = 0.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
    seq_lens_kv: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        out_dtype = torch.bfloat16 if q.dtype in _FP8_DTYPES else q.dtype
        out = torch.empty(
            (q.shape[0], q.shape[1], v.shape[-1]),
            dtype=out_dtype,
            device=q.device,
        )

    lse = (
        torch.empty((q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    mla_prefill_fwd(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        softmax_scale,
        is_causal=is_causal,
        logit_cap=logit_cap,
        lse=lse,
    )
    if return_lse:
        return out, lse
    return out
