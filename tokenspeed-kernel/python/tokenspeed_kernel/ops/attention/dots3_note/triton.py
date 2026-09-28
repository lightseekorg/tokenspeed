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

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@triton.jit
def _swa_prefill_kernel(
    Q,
    K,
    V,
    O,
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
    WINDOW_LEFT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    q_start = tl.load(cu_seqlens_q + cur_seq)
    q_len = tl.load(cu_seqlens_q + cur_seq + 1) - q_start
    kv_start = tl.load(cu_seqlens_kv + cur_seq)
    kv_len = tl.load(cu_seqlens_kv + cur_seq + 1) - kv_start
    q_causal_start = kv_len - q_len

    offs_d = tl.arange(0, 256)
    offs_dv = tl.arange(0, 128)
    q_offsets_m = cur_block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = q_offsets_m < q_len
    offs_q = (
        (q_start + q_offsets_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(Q + offs_q, mask=mask_m[:, None], other=0.0)
    acc = tl.zeros([BLOCK_M, 128], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    query_positions = q_causal_start + q_offsets_m[:, None]
    key_begin = tl.maximum(q_causal_start + cur_block_m * BLOCK_M - WINDOW_LEFT, 0)
    key_end = tl.minimum(q_causal_start + (cur_block_m + 1) * BLOCK_M, kv_len)
    for start_n in range(key_begin // BLOCK_N * BLOCK_N, key_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        kv_offsets_n = start_n + offs_n
        mask_n = (kv_offsets_n >= key_begin) & (kv_offsets_n < key_end)
        key_positions = kv_offsets_n[None, :]
        final_mask = (
            mask_m[:, None]
            & mask_n[None, :]
            & (query_positions >= key_positions)
            & (key_positions >= query_positions - WINDOW_LEFT)
        )
        offs_k = (
            (kv_start + kv_offsets_n[None, :]) * stride_kbs
            + cur_kv_head * stride_kh
            + offs_d[:, None]
        )
        k = tl.load(K + offs_k, mask=mask_n[None, :], other=0.0)
        qk = tl.dot(q, k) * sm_scale
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
        v = tl.load(V + offs_v, mask=mask_n[:, None], other=0.0)
        p_hi = p.to(v.dtype)
        acc = acc * re_scale[:, None] + tl.dot(p_hi, v)
        if v.dtype == tl.bfloat16:
            # BF16 probability rounding loses small differences when values
            # cancel. Keep its residual without narrowing BF16 values to FP16.
            p_lo = (p - p_hi.to(tl.float32)).to(v.dtype)
            acc += tl.dot(p_lo, v)
        e_max = n_e_max

    safe_deno = tl.where(deno > 0.0, deno, 1.0)
    offs_o = (
        (q_start + q_offsets_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
    tl.store(O + offs_o, acc / safe_deno[:, None], mask=mask_m[:, None])


@register_kernel(
    "attention",
    "dots3_note_swa_prefill",
    name="triton_dots3_note_swa_prefill",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures(
        ("q", "k", "v"), "dense", frozenset({torch.float16, torch.bfloat16})
    ),
    priority=Priority.PORTABLE,
    traits={"head_dim": frozenset({256}), "value_head_dim": frozenset({128})},
)
def triton_dots3_note_swa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    softmax_scale: float,
    *,
    window_left: int,
) -> torch.Tensor:
    if window_left < 0 or max_seqlen_q < 0 or max_seqlen_kv < 0:
        raise ValueError(
            "window_left and maximum sequence lengths must be non-negative"
        )
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, v must be [tokens, heads, dim]")
    if q.shape[-1] != 256 or k.shape[-1] != 256 or v.shape[-1] != 128:
        raise ValueError("dots3 SWA prefill requires Q/K dim 256 and V dim 128")
    if k.shape[:2] != v.shape[:2] or k.shape[1] == 0 or q.shape[1] % k.shape[1]:
        raise ValueError("K/V token and head counts must match and divide query heads")
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if (
            tensor.dtype not in (torch.bfloat16, torch.float16)
            or tensor.dtype != q.dtype
            or tensor.device != q.device
            or not tensor.is_cuda
            or tensor.stride(-1) != 1
        ):
            raise ValueError(
                f"{name} must share Q's GPU and FP16/BF16 dtype with last stride 1"
            )
    if cu_seqlens_q.shape != cu_seqlens_kv.shape:
        raise ValueError("query and KV cumulative lengths must have the same shape")
    for cumulative in (cu_seqlens_q, cu_seqlens_kv):
        if (
            cumulative.ndim != 1
            or cumulative.numel() == 0
            or not cumulative.is_contiguous()
            or cumulative.device != q.device
            or cumulative.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "cumulative lengths must be contiguous GPU int32/int64 vectors"
            )
    out = torch.empty((q.shape[0], q.shape[1], 128), dtype=q.dtype, device=q.device)
    grid = (cu_seqlens_q.shape[0] - 1, q.shape[1], triton.cdiv(max_seqlen_q, 64))
    _swa_prefill_kernel[grid](
        q,
        k,
        v,
        out,
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
        WINDOW_LEFT=window_left,
        BLOCK_M=64,
        BLOCK_N=64,
        num_warps=4,
        num_stages=1,
    )
    return out
