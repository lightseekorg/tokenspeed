# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the LICENSE file
# of https://github.com/fla-org/flash-linear-attention (v0.5.1); vendored here
# with dual-index in-place state-pool addressing added for the flat KV cache.

"""Packed convolution producer for retained KDA verification.

Read committed convolution windows and candidate QKV rows without publishing
state. Packed query boundaries cover live rows and graph padding.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton


# The row and page strides stay specialized: they are fixed for a model, and
# their divisibility by 16 lets the loads and stores of each row vectorize.
@triton.jit(do_not_specialize=["query_stride", "read_stride", "total_tokens", "pages"])
def fused_kda_verify_conv_update_kernel(
    qkv_raw,
    conv_w,
    conv_bias,
    conv_pool,
    read_indices,
    query_start_loc,
    query_stride,
    read_stride,
    total_tokens,
    pages,
    out,
    original_qkv,
    stride_original_tok,
    stride_raw_tok,
    stride_out_tok,
    stride_conv_page,
    T: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_row = tl.program_id(0).to(tl.int64)
    i_c = tl.program_id(1).to(tl.int64)
    i_n = i_row // T
    i_t = i_row % T
    begin = i_n * T
    token_valid = True
    begin = tl.load(query_start_loc + i_n * query_stride).to(tl.int64)
    end = tl.load(query_start_loc + (i_n + 1) * query_stride).to(tl.int64)
    token_valid = (
        (begin >= 0) & (end <= total_tokens) & (end - begin <= T) & (i_t < end - begin)
    )
    i_row = begin + i_t
    o_c = i_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = (o_c < C) & token_valid
    b_read = tl.load(read_indices + i_n * read_stride).to(tl.int64)
    valid_page = (b_read > 0) & (b_read < pages)
    safe_read = tl.where(valid_page, b_read, 0)
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    for i_w in range(4):
        i_window = i_t + i_w
        from_raw = i_window >= 3
        i_raw = i_window - 3
        b_state = tl.load(
            conv_pool + safe_read * stride_conv_page + o_c * 3 + i_window,
            mask=mask_c & ~from_raw & valid_page,
            other=0.0,
        ).to(tl.float32)
        b_raw = tl.load(
            qkv_raw + (begin + i_raw) * stride_raw_tok + o_c,
            mask=mask_c & from_raw,
            other=0.0,
        )
        if original_qkv is not None:
            if i_w == 3:
                # The last tap reads this row before convolution and SiLU.
                tl.store(
                    original_qkv + i_row * stride_original_tok + o_c,
                    b_raw,
                    mask=mask_c,
                )
        b_w = tl.load(conv_w + o_c * 4 + i_w, mask=mask_c, other=0.0).to(tl.float32)
        acc += tl.where(from_raw, b_raw.to(tl.float32), b_state) * b_w
    if HAS_BIAS:
        acc += tl.load(conv_bias + o_c, mask=mask_c, other=0.0).to(tl.float32)
    acc = acc * tl.sigmoid(acc)
    tl.store(
        out + i_row * stride_out_tok + o_c,
        acc.to(out.dtype.element_ty),
        mask=mask_c,
    )


def fused_kda_verify_conv_update(
    qkv_raw: torch.Tensor,
    conv_w: torch.Tensor,
    conv_pool: torch.Tensor,
    read_indices: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    query_start_loc: torch.Tensor,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    original_qkv: torch.Tensor | None = None,
    block_c: int = 256,
    num_warps: int = 4,
) -> torch.Tensor:
    """Compute verify conv1d(+silu) rows from committed convolution state.

    Args:
        qkv_raw: ``[N*T, 3*P]`` FP16/BF16 pre-conv packed q|k|v, request-major.
        conv_w: ``[3*P, 4]`` contiguous convolution taps in the input dtype.
        conv_pool: ``[pages, 3*P, 3]`` committed state (read-only), in the
            input dtype or BF16 for FP16 inputs. History and current inputs
            are converted independently to FP32 for convolution.
        read_indices: ``[N]`` committed page per request (-1 = fresh).
        num_heads/head_dim: Per-rank head geometry (P = num_heads*head_dim).
        draft_token_num: Maximum draft positions per request (T).
        query_start_loc: Packed request boundaries, with one boundary per request.
        bias: Optional ``[3*P]`` convolution bias added before SiLU.
        out: Optional ``[N*T, 3*P]`` output tensor in the input dtype.
        original_qkv: Optional separate ``[N*T, 3*P]`` destination in the
            input dtype for pre-convolution rows. None skips retention.
        block_c: Number of feature channels handled by each program.
        num_warps: Number of warps used by each program.

    Returns:
        out: ``[N*T, 3*P]`` convolved and SiLU-activated q|k|v rows.
    """
    T = draft_token_num
    P = num_heads * head_dim
    C = 3 * P
    total = qkv_raw.shape[0]
    if T <= 0:
        raise ValueError(f"draft_token_num={T} must evenly divide {total} rows")
    N = read_indices.numel()
    if (
        query_start_loc.shape != (N + 1,)
        or query_start_loc.dtype not in (torch.int32, torch.int64)
        or query_start_loc.device != qkv_raw.device
    ):
        raise ValueError(
            "query_start_loc must be an integer [requests + 1] boundary vector"
        )
    if qkv_raw.shape != (total, C) or qkv_raw.stride(-1) != 1:
        raise ValueError(f"qkv_raw must have shape [{total}, {C}] and last stride 1")
    if qkv_raw.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"qkv_raw must have dtype FP16 or BF16, got {qkv_raw.dtype}")
    if conv_w.shape != (C, 4) or not conv_w.is_contiguous():
        raise ValueError(f"conv_w must be contiguous with shape [{C}, 4]")
    if conv_w.dtype != qkv_raw.dtype:
        raise ValueError(f"conv_w must have dtype {qkv_raw.dtype}, got {conv_w.dtype}")
    # A page may carry more channels; the kernel reads the leading C.
    if conv_pool.ndim != 3 or conv_pool.shape[1] < C or conv_pool.shape[2] != 3:
        raise ValueError(
            f"conv_pool must have shape [pages, >= {C}, 3], got "
            f"{tuple(conv_pool.shape)}"
        )
    if conv_pool.stride(-1) != 1 or conv_pool.stride(-2) != 3:
        raise ValueError("conv_pool feature and tap dimensions must be contiguous")
    if conv_pool.dtype not in (qkv_raw.dtype, torch.bfloat16):
        raise ValueError(
            f"conv_pool must have dtype {qkv_raw.dtype} or torch.bfloat16, "
            f"got {conv_pool.dtype}"
        )
    if read_indices.ndim != 1 or read_indices.numel() != N:
        raise ValueError(f"read_indices must have shape [{N}]")
    if read_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("read_indices must have dtype torch.int32 or torch.int64")
    if block_c <= 0 or block_c & (block_c - 1):
        raise ValueError(f"block_c={block_c} must be a positive power of two")
    if out is None:
        out = torch.empty_like(qkv_raw, memory_format=torch.contiguous_format)
    elif out.shape != (total, C) or out.stride(-1) != 1:
        raise ValueError(f"out must have shape [{total}, {C}] and last stride 1")
    elif out.dtype != qkv_raw.dtype:
        raise ValueError(f"out must have dtype {qkv_raw.dtype}, got {out.dtype}")
    if original_qkv is not None:
        if (
            original_qkv.shape != qkv_raw.shape
            or original_qkv.dtype != qkv_raw.dtype
            or original_qkv.device != qkv_raw.device
            or original_qkv.stride(-1) != 1
            or original_qkv.stride(0) < C
        ):
            raise ValueError("Original QKV destination must match the input rows")
        storage = original_qkv.untyped_storage()
        begin, end = storage.data_ptr(), storage.data_ptr() + storage.nbytes()
        for source in (qkv_raw, conv_w, conv_pool, bias, out):
            if source is None:
                continue
            source_storage = source.untyped_storage()
            if (
                begin < source_storage.data_ptr() + source_storage.nbytes()
                and source_storage.data_ptr() < end
            ):
                raise ValueError("Original QKV destination must use separate storage")
    grid = (N * T, triton.cdiv(C, block_c))
    fused_kda_verify_conv_update_kernel[grid](
        qkv_raw=qkv_raw,
        conv_w=conv_w,
        conv_bias=bias,
        conv_pool=conv_pool,
        read_indices=read_indices,
        query_start_loc=query_start_loc,
        query_stride=query_start_loc.stride(0),
        read_stride=read_indices.stride(0),
        total_tokens=total,
        pages=conv_pool.shape[0],
        out=out,
        original_qkv=original_qkv,
        stride_original_tok=0 if original_qkv is None else original_qkv.stride(0),
        stride_raw_tok=qkv_raw.stride(0),
        stride_out_tok=out.stride(0),
        stride_conv_page=conv_pool.stride(0),
        T=T,
        C=C,
        BLOCK_C=block_c,
        HAS_BIAS=bias is not None,
        num_warps=num_warps,
    )
    return out
