# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Sparse attention interface.

Current delivery scope:
    - head dimension is supported only for D=128

Public API:
    sparse_atten_func(...)
    sparse_decode_atten_func(...)
    SparseDecodePagedAttentionWrapper

Internal forward core:
    _sparse_atten_csr_varlen_forward(...)

Preprocessing (external, done once):
    q2k_indices [head_kv, total_q, topK]  ->  sparse_index_utils.build_k2q_csr()
        -> k2q_row_ptr   [head_kv, total_rows + 1]  int32
        -> k2q_q_indices [head_kv, total_q * topK]  int32
"""

import math
import os
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack
from src.common.cute_dsl_utils import to_cute_tensor as to_cute_tensor_kvouter
from src.common.tma_utils import (
    create_q_gather4_tma_desc,
)
from src.sm100.decode_schedule import (
    DecodeAttentionSchedule,
    prepare_decode_schedule,
)
from src.sm100.fwd.atten_fwd import SparseAttentionForwardSm100
from src.sm100.fwd.atten_fwd_nvfp4_kv import SparseAttentionForwardNvfp4KvSm100
from src.sm100.fwd.combine import combine
from src.sm100.prepare_scheduler import (
    SparseAttentionSchedule,
    prepare_sparse_fwd_schedule_and_split,
)

_compile_cache: dict = {}
_TEMPERATURE_LSE_FAST_PATH_ABS_TOL = 1e-12
_SUPPORTED_SPARSE_TOPK = (4, 8, 16, 32)
_SUPPORTED_FWD_DTYPES = (torch.bfloat16, torch.float8_e4m3fn)
_SUPPORTED_FWD_MMA_DTYPES = (torch.bfloat16, torch.float8_e4m3fn)
_SUPPORTED_DECODE_QHEAD_PER_KV = 16


def _to_cute_int32_metadata(t: torch.Tensor):
    return to_cute_tensor_kvouter(t, assumed_align=4)


def _normalize_partial_dtype(partial_dtype: torch.dtype) -> torch.dtype:
    supported = {torch.float32, torch.bfloat16, torch.float16, torch.float8_e4m3fn}
    if partial_dtype not in supported:
        raise TypeError(
            "partial_dtype must be one of torch.float32 / torch.bfloat16 / "
            "torch.float16 / torch.float8_e4m3fn, "
            f"got {partial_dtype}"
        )
    return partial_dtype


def _normalize_forward_mma_dtype(
    dtype: Optional[torch.dtype], fallback: torch.dtype, name: str
) -> torch.dtype:
    dtype = fallback if dtype is None else dtype
    if dtype not in _SUPPORTED_FWD_MMA_DTYPES:
        raise TypeError(
            f"{name} must be one of torch.bfloat16 / torch.float8_e4m3fn, got {dtype}"
        )
    return dtype


def _resolve_forward_mma_dtypes(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qk_dtype: Optional[torch.dtype],
    pv_dtype: Optional[torch.dtype],
) -> tuple[torch.dtype, torch.dtype]:
    qk_dtype = _normalize_forward_mma_dtype(qk_dtype, q.dtype, "qk_dtype")
    if pv_dtype is None:
        # Preserve the historical fp8 KV-cache path: BF16 Q with FP8 K/V
        # stages both K and V as BF16 compute operands.
        if (
            q.dtype == torch.bfloat16
            and k.dtype == torch.float8_e4m3fn
            and v.dtype == torch.float8_e4m3fn
        ):
            pv_dtype = torch.bfloat16
        else:
            pv_dtype = v.dtype
    pv_dtype = _normalize_forward_mma_dtype(pv_dtype, pv_dtype, "pv_dtype")

    if q.dtype != qk_dtype:
        raise ValueError(
            "qk_dtype must match q storage dtype; Q fp8->bf16 staging is not supported"
        )
    if k.dtype != qk_dtype:
        if not (k.dtype == torch.float8_e4m3fn and qk_dtype == torch.bfloat16):
            raise ValueError(
                "unsupported K storage/qk_dtype combination; only fp8 K -> bf16 QK staging is supported"
            )
    if v.dtype != pv_dtype:
        if not (v.dtype == torch.float8_e4m3fn and pv_dtype == torch.bfloat16):
            raise ValueError(
                "unsupported V storage/pv_dtype combination; only fp8 V -> bf16 PV staging is supported"
            )
    return qk_dtype, pv_dtype


def _to_cute_tensor_meta(t: torch.Tensor, assumed_align: int = 4):
    tensor = from_dlpack(t.detach(), assumed_align=assumed_align, enable_tvm_ffi=True)
    return tensor.mark_layout_dynamic(leading_dim=0)


def _torch_dtype_to_cutlass_dtype(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.float8_e4m3fn:
        return cutlass.Float8E4M3FN
    raise TypeError(
        f"Only torch.bfloat16, torch.float16, torch.float8_e4m3fn supported, got {dtype}"
    )


def _prepare_paged_kv_for_tma(k, v, blk_kv: int):
    page_size = int(k.shape[2])
    if page_size != blk_kv:
        raise ValueError(
            f"Sparse Page Attention requires page_size == blk_kv, got {page_size} vs {blk_kv}"
        )
    return k, v


def _validate_cu_seqlens(
    cu_seqlens: torch.Tensor,
    *,
    name: str,
    device: torch.device,
) -> None:
    if cu_seqlens.device != device:
        raise ValueError(f"{name} must be on the same device as q")
    if cu_seqlens.dtype != torch.int32:
        raise TypeError(f"{name} must be torch.int32")
    if cu_seqlens.ndim != 1:
        raise ValueError(f"{name} must have shape [B + 1]")
    if cu_seqlens.shape[0] < 1:
        raise ValueError(f"{name} must have at least one element")
    if not cu_seqlens.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _csr_row_capacity(k2q_row_ptr: torch.Tensor) -> int:
    return int(k2q_row_ptr.shape[1] - 1)


def _validate_csr_varlen_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    topK: int,
    blk_kv: int,
    page_table: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seqused_k: Optional[torch.Tensor],
) -> tuple[int, int]:
    if q.ndim != 3:
        raise ValueError("CSR sparse forward requires q to have shape [total_q, Hq, D]")
    if q.dtype not in _SUPPORTED_FWD_DTYPES:
        raise TypeError(
            "CSR sparse forward supports only torch.bfloat16 and "
            f"torch.float8_e4m3fn Q/K/V, got {q.dtype}"
        )
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, v must be on the same device")
    mixed_fp8_kv_bf16_q = (
        q.dtype == torch.bfloat16
        and k.dtype == torch.float8_e4m3fn
        and v.dtype == torch.float8_e4m3fn
    )
    if not mixed_fp8_kv_bf16_q and (q.dtype != k.dtype or q.dtype != v.dtype):
        raise ValueError(
            "q, k, v must have the same dtype, except q=bf16 with fp8_e4m3 K/V cache"
        )
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise ValueError("q, k, v must have the same head dimension")
    dim = q.shape[-1]
    if dim != 128:
        raise NotImplementedError(
            f"CSR sparse forward currently supports only D=128, got D={dim}"
        )
    if page_table is None:
        if k.shape[-2] != v.shape[-2] or k.shape[-1] != v.shape[-1]:
            raise ValueError("k and v must have the same [Hkv, D] tail dimensions")
        head_kv = k.shape[-2]
    else:
        if k.ndim != 4 or v.ndim != 4:
            raise ValueError(
                "Sparse Page Attention requires k and v to have shape "
                "[num_pages, Hkv, page_size, D]"
            )
        if k.shape[1] != v.shape[1] or k.shape[-1] != v.shape[-1]:
            raise ValueError(
                "Sparse Page Attention k and v must have the same Hkv and D"
            )
        head_kv = k.shape[1]
    if q.device != k2q_row_ptr.device or q.device != k2q_q_indices.device:
        raise ValueError("CSR metadata must be on the same device as q")
    if k2q_row_ptr.dtype != torch.int32 or k2q_q_indices.dtype != torch.int32:
        raise TypeError("CSR metadata tensors must be torch.int32")
    if k2q_row_ptr.ndim != 2 or k2q_q_indices.ndim != 2:
        raise ValueError("k2q_row_ptr and k2q_q_indices must be rank-2")

    _validate_cu_seqlens(cu_seqlens_q, name="cu_seqlens_q", device=q.device)
    _validate_cu_seqlens(cu_seqlens_k, name="cu_seqlens_k", device=q.device)
    if cu_seqlens_k.shape != cu_seqlens_q.shape:
        raise ValueError("cu_seqlens_k must have shape [B + 1] matching cu_seqlens_q")
    batch = int(cu_seqlens_q.shape[0] - 1)
    total_q = q.shape[0]

    head_q = q.shape[1]
    if head_q % head_kv != 0:
        raise ValueError("q.shape[1] must be divisible by Hkv")
    qhead_per_kv = head_q // head_kv
    if qhead_per_kv not in (1, 2, 4, 8, 16):
        raise NotImplementedError(
            "CSR forward is currently supported only for qhead_per_kv in {1, 2, 4, 8, 16}"
        )
    if k2q_row_ptr.shape[0] != head_kv or k2q_q_indices.shape[0] != head_kv:
        raise ValueError("CSR metadata head dimension must match KV head count")
    if k2q_q_indices.shape[1] < total_q * topK:
        raise ValueError(
            f"k2q_q_indices.shape[1] ({k2q_q_indices.shape[1]}) must be >= total_q * topK ({total_q * topK})"
        )
    if k2q_row_ptr.shape[1] < 1:
        raise ValueError("k2q_row_ptr must contain at least one row pointer column")

    if page_table is None:
        if seqused_k is not None:
            raise ValueError("seqused_k is only supported together with page_table")
        total_k = k.shape[0]
        if k.ndim != 3 or v.ndim != 3:
            raise ValueError(
                "Sparse Attention requires k and v to have shape [total_k, Hkv, D]"
            )
        if k.shape != (total_k, head_kv, q.shape[-1]) or v.shape != (
            total_k,
            head_kv,
            q.shape[-1],
        ):
            raise ValueError("Sparse Attention k and v must match [total_k, Hkv, D]")
    else:
        if page_table.device != q.device:
            raise ValueError("page_table must be on the same device as q")
        if page_table.dtype != torch.int32:
            raise TypeError("page_table must be torch.int32")
        if page_table.ndim != 2 or page_table.shape[0] != batch:
            raise ValueError("page_table must have shape [B, max_num_pages_per_seq]")
        if page_table.stride(-1) != 1:
            raise ValueError("page_table must be contiguous in the last dimension")
        if k.ndim != 4 or v.ndim != 4:
            raise ValueError(
                "Sparse Page Attention requires k and v to have shape "
                "[num_pages, Hkv, page_size, D]"
            )
        if k.shape != v.shape:
            raise ValueError(
                f"k and v must have the same shape, got {k.shape} and {v.shape}"
            )
        if k.shape[1] != head_kv or k.shape[3] != q.shape[-1]:
            raise ValueError(
                "Sparse Page Attention k and v must match "
                "[num_pages, Hkv, page_size, D]"
            )
        page_size = int(k.shape[2])
        if page_size != blk_kv:
            raise ValueError(
                f"Unsupported Sparse Page Attention page_size={page_size} for blk_kv={blk_kv}; "
                "require page_size == blk_kv"
            )
        if seqused_k is not None:
            if seqused_k.device != q.device:
                raise ValueError("seqused_k must be on the same device as q")
            if seqused_k.dtype != torch.int32:
                raise TypeError("seqused_k must be torch.int32")
            if seqused_k.shape != (batch,):
                raise ValueError("seqused_k must have shape [B]")
            if not seqused_k.is_contiguous():
                raise ValueError("seqused_k must be contiguous")
    if topK not in _SUPPORTED_SPARSE_TOPK:
        raise ValueError(
            f"CSR sparse forward supports topK in {_SUPPORTED_SPARSE_TOPK}, got {topK}"
        )
    return batch, head_kv


def _validate_csr_varlen_nvfp4_kv_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_scale_128x4: torch.Tensor,
    v_scale_128x4: torch.Tensor,
    k_global_scale: Optional[torch.Tensor],
    v_global_scale: Optional[torch.Tensor],
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    topK: int,
    blk_kv: int,
    page_table: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seqused_k: Optional[torch.Tensor],
) -> tuple[int, int]:
    if q.ndim != 3:
        raise ValueError(
            "KVFP4 CSR sparse forward requires q to have shape [total_q, Hq, D]"
        )
    if q.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise TypeError(
            f"KVFP4 CSR sparse forward requires BF16 or FP8 E4M3 q, got {q.dtype}"
        )
    if q.shape[-1] != 128:
        raise NotImplementedError(
            f"KVFP4 CSR sparse forward currently supports only D=128, got {q.shape[-1]}"
        )
    if k.dtype != torch.uint8 or v.dtype != torch.uint8:
        raise TypeError(f"KVFP4 k/v must be torch.uint8, got {k.dtype} and {v.dtype}")
    if k_scale_128x4.dtype != torch.uint8 or v_scale_128x4.dtype != torch.uint8:
        raise TypeError(
            "KVFP4 block scales must be torch.uint8 E4M3 tensors, got "
            f"{k_scale_128x4.dtype} and {v_scale_128x4.dtype}"
        )
    if k_global_scale is not None and k_global_scale.dtype != torch.float32:
        raise TypeError("KVFP4 K global scale must be a torch.float32 tensor or None")
    if v_global_scale is not None and v_global_scale.dtype != torch.float32:
        raise TypeError("KVFP4 V global scale must be a torch.float32 tensor or None")
    tensors = (
        k,
        v,
        k_scale_128x4,
        v_scale_128x4,
        k2q_row_ptr,
        k2q_q_indices,
        cu_seqlens_q,
        cu_seqlens_k,
    )
    optional_tensors = tuple(
        t for t in (k_global_scale, v_global_scale) if t is not None
    )
    if any(t.device != q.device for t in tensors + optional_tensors):
        raise ValueError("KVFP4 inputs and metadata must be on the same device as q")
    if k.shape != v.shape:
        raise ValueError(
            f"KVFP4 k and v must have the same shape, got {k.shape} and {v.shape}"
        )
    packed_dim = q.shape[-1] // 2
    scale_cols = q.shape[-1] // 16
    if k_scale_128x4.ndim != 2 or v_scale_128x4.ndim != 2:
        raise ValueError("KVFP4 block scales must be rank-2 128x4 tiled tensors")
    if k_scale_128x4.shape[1] < scale_cols or v_scale_128x4.shape[1] < scale_cols:
        raise ValueError(
            "KVFP4 block scales must have at least D/16 columns; "
            f"need {scale_cols}, got {k_scale_128x4.shape[1]} and {v_scale_128x4.shape[1]}"
        )
    if k_global_scale is not None and k_global_scale.numel() < 1:
        raise ValueError("KVFP4 K global scale must contain at least one element")
    if v_global_scale is not None and v_global_scale.numel() < 1:
        raise ValueError("KVFP4 V global scale must contain at least one element")

    if page_table is None:
        if seqused_k is not None:
            raise ValueError("seqused_k is only supported together with page_table")
        if k.ndim != 3:
            raise ValueError(
                "KVFP4 Sparse Attention requires k/v shape [total_k, Hkv, D/2]"
            )
        if k.shape[-1] != packed_dim:
            raise ValueError(
                f"KVFP4 packed K/V last dimension must be D/2={packed_dim}"
            )
        total_k = int(k.shape[0])
        head_kv = int(k.shape[1])
        required_scale_rows = total_k * head_kv
    else:
        if k.ndim != 4:
            raise ValueError(
                "KVFP4 Sparse Page Attention requires k/v shape "
                "[num_pages, Hkv, page_size, D/2]"
            )
        if k.shape[-1] != packed_dim:
            raise ValueError(
                f"KVFP4 packed K/V last dimension must be D/2={packed_dim}"
            )
        page_size = int(k.shape[2])
        if page_size != int(blk_kv):
            raise ValueError(
                f"KVFP4 Sparse Page Attention requires page_size == blk_kv, got {page_size} vs {blk_kv}"
            )
        head_kv = int(k.shape[1])
        required_scale_rows = int(k.shape[0]) * head_kv * page_size
        if page_table.device != q.device:
            raise ValueError("page_table must be on the same device as q")
        if page_table.dtype != torch.int32:
            raise TypeError("page_table must be torch.int32")
        if page_table.ndim != 2:
            raise ValueError("page_table must have shape [B, max_num_pages_per_seq]")
        if page_table.stride(-1) != 1:
            raise ValueError("page_table must be contiguous in the last dimension")
        if seqused_k is not None:
            if seqused_k.device != q.device:
                raise ValueError("seqused_k must be on the same device as q")
            if seqused_k.dtype != torch.int32:
                raise TypeError("seqused_k must be torch.int32")
            if not seqused_k.is_contiguous():
                raise ValueError("seqused_k must be contiguous")

    padded_scale_rows = ((required_scale_rows + 127) // 128) * 128
    padded_scale_cols = ((scale_cols + 3) // 4) * 4
    for name, scale in (
        ("k_scale_128x4", k_scale_128x4),
        ("v_scale_128x4", v_scale_128x4),
    ):
        if scale.shape[0] < padded_scale_rows or scale.shape[1] < padded_scale_cols:
            raise ValueError(
                f"{name} is too small for 128x4 layout: got {tuple(scale.shape)}, "
                f"need at least {(padded_scale_rows, padded_scale_cols)}"
            )

    if k2q_row_ptr.device != q.device or k2q_q_indices.device != q.device:
        raise ValueError("CSR metadata must be on the same device as q")
    if k2q_row_ptr.dtype != torch.int32 or k2q_q_indices.dtype != torch.int32:
        raise TypeError("CSR metadata tensors must be torch.int32")
    if k2q_row_ptr.ndim != 2 or k2q_q_indices.ndim != 2:
        raise ValueError("k2q_row_ptr and k2q_q_indices must be rank-2")
    _validate_cu_seqlens(cu_seqlens_q, name="cu_seqlens_q", device=q.device)
    _validate_cu_seqlens(cu_seqlens_k, name="cu_seqlens_k", device=q.device)
    if cu_seqlens_k.shape != cu_seqlens_q.shape:
        raise ValueError("cu_seqlens_k must have shape [B + 1] matching cu_seqlens_q")
    batch = int(cu_seqlens_q.shape[0] - 1)
    if page_table is not None and page_table.shape[0] != batch:
        raise ValueError("page_table must have shape [B, max_num_pages_per_seq]")
    if seqused_k is not None and seqused_k.shape != (batch,):
        raise ValueError("seqused_k must have shape [B]")
    head_q = int(q.shape[1])
    if head_q % head_kv != 0:
        raise ValueError("q.shape[1] must be divisible by Hkv")
    qhead_per_kv = head_q // head_kv
    if qhead_per_kv not in (1, 2, 4, 8, 16):
        raise NotImplementedError(
            "KVFP4 CSR forward is currently supported only for qhead_per_kv in {1, 2, 4, 8, 16}"
        )
    if k2q_row_ptr.shape[0] != head_kv or k2q_q_indices.shape[0] != head_kv:
        raise ValueError("CSR metadata head dimension must match KV head count")
    if k2q_q_indices.shape[1] < q.shape[0] * topK:
        raise ValueError(
            f"k2q_q_indices.shape[1] ({k2q_q_indices.shape[1]}) must be >= total_q * topK ({q.shape[0] * topK})"
        )
    if k2q_row_ptr.shape[1] < 1:
        raise ValueError("k2q_row_ptr must contain at least one row pointer column")
    if topK not in _SUPPORTED_SPARSE_TOPK:
        raise ValueError(
            f"KVFP4 CSR sparse forward supports topK in {_SUPPORTED_SPARSE_TOPK}, got {topK}"
        )
    return batch, head_kv


def _validate_sparse_decode_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: Optional[torch.Tensor],
    *,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    seqlen_q: int,
    max_seqlen_k: int,
    blk_kv: int,
    causal: bool,
) -> tuple[int, int]:
    if q.ndim != 3:
        raise ValueError("decode attention requires q to have shape [total_q, Hq, D]")
    if k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            "decode attention requires paged k/v with shape [num_pages, Hkv, page_size, D]"
        )
    if q.device != k.device or q.device != v.device:
        raise ValueError("decode q, k, and v must be on the same device")
    if q.dtype != torch.float8_e4m3fn or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError(
            "decode attention currently supports only torch.float8_e4m3fn Q/K/V"
        )
    if k.shape != v.shape:
        raise ValueError(
            f"decode k and v must have the same shape, got {k.shape} and {v.shape}"
        )
    if q.shape[-1] != 128 or k.shape[-1] != 128:
        raise NotImplementedError(
            f"decode attention currently supports only D=128, got q={q.shape[-1]} k={k.shape[-1]}"
        )
    if not bool(causal):
        raise NotImplementedError(
            "decode attention currently supports only causal=True"
        )
    page_size = int(k.shape[2])
    if page_size != int(blk_kv):
        raise ValueError(
            f"decode attention requires page_size == blk_kv, got {page_size} vs {blk_kv}"
        )

    head_kv = int(k.shape[1])
    head_q = int(q.shape[1])
    if head_q % head_kv != 0:
        raise ValueError("decode q.shape[1] must be divisible by Hkv")
    qhead_per_kv = head_q // head_kv
    if qhead_per_kv != _SUPPORTED_DECODE_QHEAD_PER_KV:
        raise NotImplementedError(
            "decode attention currently supports only "
            f"qhead_per_kv={_SUPPORTED_DECODE_QHEAD_PER_KV}, got {qhead_per_kv}"
        )

    if page_table is None:
        raise ValueError("decode attention requires page_table")
    if page_table.device != q.device:
        raise ValueError("decode page_table must be on the same device as q")
    if page_table.dtype != torch.int32:
        raise TypeError("decode page_table must be torch.int32")
    if page_table.ndim != 2:
        raise ValueError("decode page_table must have shape [B, max_num_pages_per_seq]")
    batch = int(page_table.shape[0])
    if page_table.stride(-1) != 1:
        raise ValueError("decode page_table must be contiguous in the last dimension")

    if seqused_k is None:
        raise ValueError("decode attention requires seqused_k")
    if seqused_k.device != q.device:
        raise ValueError("decode seqused_k must be on the same device as q")
    if seqused_k.dtype != torch.int32:
        raise TypeError("decode seqused_k must be torch.int32")
    if seqused_k.shape != (batch,):
        raise ValueError("decode seqused_k must have shape [B]")
    if not seqused_k.is_contiguous():
        raise ValueError("decode seqused_k must be contiguous")

    seqlen_q = int(seqlen_q)
    max_seqlen_k = int(max_seqlen_k)
    if seqlen_q <= 0 or max_seqlen_k <= 0:
        raise ValueError("decode seqlen_q and max_seqlen_k must be positive")
    if int(q.shape[0]) != batch * seqlen_q:
        raise ValueError("decode q.shape[0] must equal batch * seqlen_q")

    if q2k_indices is not None:
        if q2k_indices.device != q.device:
            raise ValueError("decode q2k_indices must be on the same device as q")
        if q2k_indices.dtype != torch.int32:
            raise TypeError("decode q2k_indices must be torch.int32")
        if q2k_indices.ndim != 3:
            raise ValueError("decode q2k_indices must have shape [Hkv, total_q, topK]")
        if q2k_indices.shape[0] != head_kv or q2k_indices.shape[1] != q.shape[0]:
            raise ValueError("decode q2k_indices must match [Hkv, total_q, topK]")
        if not q2k_indices.is_contiguous():
            raise ValueError("decode q2k_indices must be contiguous")
    return batch, head_kv


def _validate_schedule_common(
    schedule: SparseAttentionSchedule,
    *,
    device: torch.device,
) -> None:
    if schedule.scheduler_metadata is None:
        raise ValueError("schedule.scheduler_metadata is required")
    if schedule.work_count is None:
        raise ValueError("schedule.work_count is required")
    metadata = schedule.scheduler_metadata
    work_count = schedule.work_count
    if metadata.device != device or work_count.device != device:
        raise ValueError("schedule tensors must be on the same device as q")
    if metadata.dtype != torch.int32 or work_count.dtype != torch.int32:
        raise TypeError(
            "schedule.scheduler_metadata and schedule.work_count must be torch.int32"
        )
    if metadata.ndim != 2 or metadata.shape[1] != 6:
        raise ValueError("schedule.scheduler_metadata must have shape [capacity, 6]")
    if work_count.shape != (1,):
        raise ValueError("schedule.work_count must have shape [1]")
    if not metadata.is_contiguous() or not work_count.is_contiguous():
        raise ValueError(
            "schedule.scheduler_metadata and schedule.work_count must be contiguous"
        )


def _validate_fwd_schedule(
    schedule: SparseAttentionSchedule,
    *,
    q: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    head_kv: int,
) -> None:
    _validate_schedule_common(schedule, device=q.device)
    if schedule.qsplit_indices is None:
        raise ValueError("schedule.qsplit_indices is required for forward")
    if schedule.split_counts is None:
        raise ValueError("schedule.split_counts is required for forward")
    qsplit = schedule.qsplit_indices
    split_counts = schedule.split_counts
    if qsplit.device != q.device or split_counts.device != q.device:
        raise ValueError("forward schedule tensors must be on the same device as q")
    if qsplit.dtype != torch.int32 or split_counts.dtype != torch.int32:
        raise TypeError(
            "schedule.qsplit_indices and schedule.split_counts must be torch.int32"
        )
    if qsplit.shape != k2q_q_indices.shape:
        raise ValueError("schedule.qsplit_indices shape must match k2q_q_indices")
    total_q = q.shape[0]
    if split_counts.shape != (total_q, head_kv):
        raise ValueError(
            "schedule.split_counts must have shape "
            f"({total_q}, {head_kv}), got {tuple(split_counts.shape)}"
        )
    if not qsplit.is_contiguous() or not split_counts.is_contiguous():
        raise ValueError(
            "schedule.qsplit_indices and schedule.split_counts must be contiguous"
        )


def sparse_atten_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    topK: int,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    blk_kv: int = 128,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    lse_temperature_scale: float = 1.0,
    return_temperature_lse: bool = False,
    partial_dtype: torch.dtype = torch.bfloat16,
    return_softmax_lse: bool = False,
    page_table: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    schedule: Optional[SparseAttentionSchedule] = None,
    usable_SM_count: int = -1,
    qk_dtype: Optional[torch.dtype] = None,
    pv_dtype: Optional[torch.dtype] = None,
    output_scale: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
):
    """Run SM100 CSR block-sparse varlen attention.

    This is the public forward-only sparse attention API.  It consumes
    query-to-key block selections converted to CSR metadata by
    ``build_k2q_csr`` and supports both dense KV layout and paged KV layout.

    Parameters
    ----------
    q : torch.Tensor
        Shape ``[total_q, Hq, 128]`` on CUDA.  Supported dtypes are BF16 and
        FP8 E4M3.
    k : torch.Tensor
        Dense layout ``[total_k, Hkv, 128]`` or paged layout
        ``[num_pages, Hkv, blk_kv, 128]``.  For BF16 Q with FP8 K/V cache, K
        may be FP8 E4M3 while QK compute uses BF16 staging.
    v : torch.Tensor
        Same layout and head count as ``k``.
    k2q_row_ptr : torch.Tensor
        CSR row pointers with shape ``[Hkv, total_rows + 1]`` and dtype int32.
    k2q_q_indices : torch.Tensor
        CSR query indices with shape ``[Hkv, >= total_q * topK]`` and dtype
        int32.
    topK : int
        Number of selected KV blocks per query.  Supported values are
        ``4, 8, 16, 32``.
    cu_seqlens_q : torch.Tensor
        Shape ``[batch_size + 1]``, dtype int32.  Prefix sums of Q lengths.
    cu_seqlens_k : torch.Tensor
        Shape ``[batch_size + 1]``, dtype int32.  Prefix sums of KV lengths.
    max_seqlen_q : int
        Maximum Q sequence length in the batch.
    max_seqlen_k : int
        Maximum KV sequence length in the batch.
    blk_kv : int, optional
        KV block size.  Paged KV requires ``k.shape[2] == blk_kv``.
    causal : bool, optional
        Whether to apply causal masking.
    softmax_scale : float, optional
        Softmax scale.  Defaults to ``1 / sqrt(128)``.
    lse_temperature_scale : float, optional
        Extra divisor used only for temperature-scaled LSE output.
    return_temperature_lse : bool, optional
        If True, also return LSE computed with logits scaled by
        ``softmax_scale / lse_temperature_scale``.  Requires
        ``return_softmax_lse=True``.
    partial_dtype : torch.dtype, optional
        Accumulation dtype for per-block partial O.  Supported values are
        FP32, BF16, FP16, and FP8 E4M3.
    return_softmax_lse : bool, optional
        If True, return ``(out, softmax_lse)`` or
        ``(out, softmax_lse, temperature_lse)``.
    page_table : torch.Tensor, optional
        Paged-KV physical page table with shape
        ``[batch_size, max_num_pages_per_seq]`` and dtype int32.
    seqused_k : torch.Tensor, optional
        Shape ``[batch_size]``, dtype int32.  Effective KV length per request
        for paged causal attention.
    schedule : SparseAttentionSchedule, optional
        Prebuilt sparse forward schedule.  If omitted, the schedule is built
        during the call.
    usable_SM_count : int, optional
        Maximum number of SMs used by the scheduler.  ``-1`` uses all SMs.
    qk_dtype : torch.dtype, optional
        Compile-time MMA operand dtype for QK.  Defaults to Q storage dtype,
        except supported FP8 K/V cache staging modes.
    pv_dtype : torch.dtype, optional
        Compile-time MMA operand dtype for PV.  Defaults to V storage dtype,
        except supported FP8 K/V cache staging modes.
    output_scale : torch.Tensor, optional
        Optional FP32 device scalar applied in the combine stage before writing
        the final output.
    out : torch.Tensor, optional
        Optional BF16 output buffer with shape ``[total_q, Hq, 128]``.  When
        provided, the combine stage writes directly into this tensor.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, torch.Tensor]
        Output shape ``[total_q, Hq, 128]`` with BF16 dtype.  Optional LSE
        outputs have shape ``[total_q, Hq]`` and dtype float32.

    Notes
    -----
    ``Hq / Hkv`` must be one of ``1, 2, 4, 8, 16``.  Current kernels support
    head dimension 128 only.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    lse_temperature_scale = float(lse_temperature_scale)
    if not math.isfinite(lse_temperature_scale) or lse_temperature_scale <= 0.0:
        raise ValueError(
            f"lse_temperature_scale must be finite and > 0, got {lse_temperature_scale}"
        )
    return_temperature_lse = bool(return_temperature_lse)
    if return_temperature_lse and not return_softmax_lse:
        raise ValueError("return_temperature_lse=True requires return_softmax_lse=True")
    partial_dtype = _normalize_partial_dtype(partial_dtype)
    qk_dtype, pv_dtype = _resolve_forward_mma_dtypes(q, k, v, qk_dtype, pv_dtype)

    if cu_seqlens_q is None or cu_seqlens_k is None:
        raise ValueError(
            "sparse_atten_func requires CSR varlen metadata: pass cu_seqlens_q and cu_seqlens_k"
        )
    batch, head_kv = _validate_csr_varlen_inputs(
        q,
        k,
        v,
        k2q_row_ptr,
        k2q_q_indices,
        topK,
        blk_kv,
        page_table,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
    )
    max_seqlen_q = int(max_seqlen_q)
    max_seqlen_k = int(max_seqlen_k)

    return _sparse_atten_csr_varlen_forward(
        q,
        k,
        v,
        k2q_row_ptr,
        k2q_q_indices,
        int(topK),
        int(blk_kv),
        bool(causal),
        float(softmax_scale),
        lse_temperature_scale,
        return_temperature_lse,
        partial_dtype,
        bool(return_softmax_lse),
        cu_seqlens_q,
        cu_seqlens_k,
        page_table,
        seqused_k,
        schedule,
        int(usable_SM_count),
        int(batch),
        int(head_kv),
        int(max_seqlen_q),
        int(max_seqlen_k),
        qk_dtype,
        pv_dtype,
        output_scale,
        out,
    )


def sparse_atten_nvfp4_kv_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_scale_128x4: torch.Tensor,
    v_scale_128x4: torch.Tensor,
    k_global_scale: Optional[torch.Tensor],
    v_global_scale: Optional[torch.Tensor],
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    topK: int,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    blk_kv: int = 128,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    lse_temperature_scale: float = 1.0,
    return_temperature_lse: bool = False,
    partial_dtype: torch.dtype = torch.bfloat16,
    return_softmax_lse: bool = False,
    page_table: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    schedule: Optional[SparseAttentionSchedule] = None,
    out: Optional[torch.Tensor] = None,
):
    """Run SM100 CSR sparse attention with packed NVFP4 K/V.

    Parameters
    ----------
    q : torch.Tensor
        Shape ``[total_q, Hq, 128]`` on CUDA.  Supported dtypes are BF16 and
        FP8 E4M3.
    k : torch.Tensor
        Packed NVFP4 K data.  Dense layout is ``[total_k, Hkv, 64]``; paged
        layout is ``[num_pages, Hkv, blk_kv, 64]``.  Dtype must be uint8
        because each byte packs two FP4 values.
    v : torch.Tensor
        Packed NVFP4 V data with the same shape as ``k``.
    k_scale_128x4 : torch.Tensor
        K block scales in cuBLAS/cuDNN 128x4 tiled storage.  Dtype uint8
        containing FP8 E4M3 scale values.
    v_scale_128x4 : torch.Tensor
        V block scales in the same 128x4 tiled storage.
    k_global_scale : torch.Tensor, optional
        FP32 tensor/global dequant scale for K.  May be ``None``.
    v_global_scale : torch.Tensor, optional
        FP32 tensor/global dequant scale for V.  May be ``None``.  The V global
        scale is applied in the combine stage.
    k2q_row_ptr : torch.Tensor
        CSR row pointers with shape ``[Hkv, total_rows + 1]`` and dtype int32.
    k2q_q_indices : torch.Tensor
        CSR query indices with shape ``[Hkv, >= total_q * topK]`` and dtype
        int32.
    topK : int
        Number of selected KV blocks per query.  Supported values are
        ``4, 8, 16, 32``.
    cu_seqlens_q, cu_seqlens_k : torch.Tensor
        Shape ``[batch_size + 1]``, dtype int32.  Prefix sums of Q and KV
        lengths.
    max_seqlen_q, max_seqlen_k : int
        Maximum Q and KV sequence lengths in the batch.
    blk_kv : int, optional
        KV block/page size.  Paged KV requires ``k.shape[2] == blk_kv``.
    causal : bool, optional
        Whether to apply causal masking.
    softmax_scale : float, optional
        Softmax scale.  Defaults to ``1 / sqrt(128)``.
    lse_temperature_scale : float, optional
        Extra divisor used only for temperature-scaled LSE output.
    return_temperature_lse : bool, optional
        If True, also return temperature-scaled LSE.  Requires
        ``return_softmax_lse=True``.
    partial_dtype : torch.dtype, optional
        Accumulation dtype for per-block partial O.
    return_softmax_lse : bool, optional
        If True, return LSE together with the output.
    page_table : torch.Tensor, optional
        Paged-KV physical page table with shape
        ``[batch_size, max_num_pages_per_seq]`` and dtype int32.
    seqused_k : torch.Tensor, optional
        Effective KV length per request for paged causal attention.
    schedule : SparseAttentionSchedule, optional
        Prebuilt sparse forward schedule.
    out : torch.Tensor, optional
        Optional BF16 output buffer with shape ``[total_q, Hq, 128]``.  When
        provided, the combine stage writes directly into this tensor.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, torch.Tensor]
        Output shape ``[total_q, Hq, 128]`` with BF16 dtype.  Optional LSE
        outputs have shape ``[total_q, Hq]`` and dtype float32.
    """

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    lse_temperature_scale = float(lse_temperature_scale)
    if not math.isfinite(lse_temperature_scale) or lse_temperature_scale <= 0.0:
        raise ValueError(
            f"lse_temperature_scale must be finite and > 0, got {lse_temperature_scale}"
        )
    return_temperature_lse = bool(return_temperature_lse)
    if return_temperature_lse and not return_softmax_lse:
        raise ValueError("return_temperature_lse=True requires return_softmax_lse=True")
    partial_dtype = _normalize_partial_dtype(partial_dtype)

    if cu_seqlens_q is None or cu_seqlens_k is None:
        raise ValueError(
            "sparse_atten_nvfp4_kv_func requires CSR varlen metadata: pass cu_seqlens_q and cu_seqlens_k"
        )
    batch, head_kv = _validate_csr_varlen_nvfp4_kv_inputs(
        q,
        k,
        v,
        k_scale_128x4,
        v_scale_128x4,
        k_global_scale,
        v_global_scale,
        k2q_row_ptr,
        k2q_q_indices,
        topK,
        blk_kv,
        page_table,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
    )
    total_q, head_q, dim = q.shape
    max_num_kv_blocks = _csr_row_capacity(k2q_row_ptr)
    temperature_lse_fast_path = return_temperature_lse and math.isclose(
        float(lse_temperature_scale),
        1.0,
        rel_tol=0.0,
        abs_tol=_TEMPERATURE_LSE_FAST_PATH_ABS_TOL,
    )
    kernel_return_temperature_lse = (
        return_temperature_lse and not temperature_lse_fast_path
    )

    O_partial = torch.empty(
        topK, total_q, head_q, dim, dtype=partial_dtype, device=q.device
    )
    LSE_partial = torch.empty(
        topK, total_q, head_q, dtype=torch.float32, device=q.device
    )
    LSE_temperature_partial = (
        torch.empty(topK, total_q, head_q, dtype=torch.float32, device=q.device)
        if kernel_return_temperature_lse
        else None
    )
    O_out = (
        out
        if out is not None
        else torch.empty(total_q, head_q, dim, dtype=torch.bfloat16, device=q.device)
    )
    LSE_out = torch.empty(total_q, head_q, dtype=torch.float32, device=q.device)
    LSE_temperature_out = (
        torch.empty_like(LSE_out) if kernel_return_temperature_lse else None
    )
    if schedule is None:
        k2q_qsplit_indices = torch.empty_like(k2q_q_indices)
        split_counts = torch.zeros(
            (total_q, head_kv),
            dtype=torch.int32,
            device=q.device,
        )
    else:
        _validate_fwd_schedule(
            schedule,
            q=q,
            k2q_q_indices=k2q_q_indices,
            head_kv=head_kv,
        )
        k2q_qsplit_indices = schedule.qsplit_indices
        split_counts = schedule.split_counts

    schedule = _call_sparse_forward_sm100_csr_varlen_nvfp4_kv(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        k_scale_128x4.contiguous(),
        v_scale_128x4.contiguous(),
        None if k_global_scale is None else k_global_scale.contiguous(),
        None if v_global_scale is None else v_global_scale.contiguous(),
        k2q_row_ptr.contiguous(),
        k2q_q_indices.contiguous(),
        k2q_qsplit_indices.contiguous(),
        split_counts.contiguous(),
        cu_seqlens_q.contiguous(),
        cu_seqlens_k.contiguous(),
        None if page_table is None else page_table.contiguous(),
        None if seqused_k is None else seqused_k.contiguous(),
        O_partial,
        LSE_partial,
        LSE_temperature_partial,
        float(softmax_scale),
        lse_temperature_scale,
        kernel_return_temperature_lse,
        max_num_kv_blocks,
        int(blk_kv),
        head_kv,
        int(max_seqlen_q),
        causal=bool(causal),
        schedule=schedule,
    )

    combine(
        O_partial,
        LSE_partial,
        O_out,
        LSE_out,
        lse_temperature_partial=LSE_temperature_partial,
        lse_temperature_out=LSE_temperature_out,
        cu_seqlens=cu_seqlens_q,
        split_counts=split_counts,
        output_scale=v_global_scale,
        use_pdl=True,
    )
    if temperature_lse_fast_path:
        LSE_temperature_out = LSE_out

    if return_softmax_lse:
        if return_temperature_lse:
            return O_out, LSE_out, LSE_temperature_out
        return O_out, LSE_out
    return O_out


def sparse_decode_atten_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: Optional[torch.Tensor] = None,
    *,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    seqlen_q: int,
    max_seqlen_k: int,
    blk_kv: int = 128,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    return_softmax_lse: bool = False,
    schedule: Optional[DecodeAttentionSchedule] = None,
    O_partial: Optional[torch.Tensor] = None,
    LSE_partial: Optional[torch.Tensor] = None,
):
    """Run forward-only paged FP8 decode attention.

    Parameters
    ----------
    q : torch.Tensor
        Shape ``[batch_size * seqlen_q, Hq, 128]``.  Dtype must be FP8 E4M3.
    k : torch.Tensor
        Paged K cache with shape ``[num_pages, Hkv, blk_kv, 128]`` and FP8
        E4M3 dtype.
    v : torch.Tensor
        Paged V cache with the same shape and dtype as ``k``.
    q2k_indices : torch.Tensor, optional
        Sparse selected KV blocks with shape ``[Hkv, total_q, topK]`` and dtype
        int32.  ``None`` selects the dense all-KV decode path.
    page_table : torch.Tensor
        Physical page table with shape ``[batch_size, max_num_pages_per_seq]``
        and dtype int32.
    seqused_k : torch.Tensor
        Shape ``[batch_size]``, dtype int32.  Effective KV length per request.
    seqlen_q : int
        Uniform query length per request.  Ragged Q lengths should use prefill
        or append paths instead.
    max_seqlen_k : int
        Maximum KV sequence length in the batch.
    blk_kv : int, optional
        Page size.  Must match ``k.shape[2]``.
    causal : bool, optional
        Whether to apply causal masking.  Current decode kernel requires True.
    softmax_scale : float, optional
        Softmax scale.  Defaults to ``1 / sqrt(128)``.
    return_softmax_lse : bool, optional
        If True, return ``(out, lse)``.
    schedule : DecodeAttentionSchedule, optional
        Prebuilt decode schedule.
    O_partial, LSE_partial : torch.Tensor, optional
        Optional split-KV partial workspaces.  Normally owned by
        ``SparseDecodePagedAttentionWrapper``.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, torch.Tensor]
        BF16 output with shape ``q.shape``.  Optional LSE has shape
        ``[batch_size * seqlen_q, Hq]`` and dtype float32.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    batch, head_kv = _validate_sparse_decode_inputs(
        q,
        k,
        v,
        q2k_indices,
        page_table=page_table,
        seqused_k=seqused_k,
        seqlen_q=seqlen_q,
        max_seqlen_k=max_seqlen_k,
        blk_kv=blk_kv,
        causal=causal,
    )
    head_q = int(q.shape[1])
    head_dim = int(q.shape[2])
    if schedule is None:
        schedule = prepare_decode_schedule(
            seqused_k=seqused_k.contiguous(),
            page_size=int(blk_kv),
            seqlen_q=int(seqlen_q),
            num_qo_heads=head_q,
            num_kv_heads=head_kv,
            head_dim=head_dim,
            max_seqlen_k=int(max_seqlen_k),
        )
    if schedule.split_kv:
        if O_partial is None:
            O_partial = torch.empty(
                (schedule.partial_rows, head_q, head_dim),
                dtype=torch.float32,
                device=q.device,
            )
        if LSE_partial is None:
            LSE_partial = torch.empty(
                (schedule.partial_rows, head_q),
                dtype=torch.float32,
                device=q.device,
            )
    out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    lse = torch.empty(
        q.shape[:2] if (return_softmax_lse or schedule.split_kv) else (1, head_q),
        dtype=torch.float32,
        device=q.device,
    )
    _call_sparse_decode_forward_sm100_paged_fp8(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        None if q2k_indices is None else q2k_indices.contiguous(),
        page_table.contiguous(),
        seqused_k.contiguous(),
        out,
        lse,
        schedule,
        O_partial,
        LSE_partial,
        softmax_scale=float(softmax_scale),
        seqlen_q=int(seqlen_q),
        max_seqlen_k=int(max_seqlen_k),
        blk_kv=int(blk_kv),
        causal=bool(causal),
        return_lse=bool(return_softmax_lse),
    )
    if return_softmax_lse:
        return out, lse
    return out


class SparseDecodePagedAttentionWrapper:
    """Plan/run helper for paged FP8 decode attention.

    Use this wrapper when the same page table shape and sequence metadata are
    reused across multiple decode layers.  ``plan`` validates metadata and
    allocates persistent schedules/workspaces; ``run`` then launches the decode
    kernel with lower per-call overhead than ``sparse_decode_atten_func``.
    """

    def __init__(self, *, blk_kv: int = 128, causal: bool = True):
        self.blk_kv = int(blk_kv)
        self.causal = bool(causal)
        self.batch: Optional[int] = None
        self.num_qo_heads: Optional[int] = None
        self.num_kv_heads: Optional[int] = None
        self.head_dim: Optional[int] = None
        self.page_table: Optional[torch.Tensor] = None
        self.seqused_k: Optional[torch.Tensor] = None
        self.q2k_indices: Optional[torch.Tensor] = None
        self.seqlen_q: Optional[int] = None
        self.max_seqlen_k: Optional[int] = None
        self.is_sparse: bool = False
        self.decode_schedule: Optional[DecodeAttentionSchedule] = None
        self.request_indices: Optional[torch.Tensor] = None
        self.qo_tile_indices: Optional[torch.Tensor] = None
        self.kv_tile_indices: Optional[torch.Tensor] = None
        self.merge_indptr: Optional[torch.Tensor] = None
        self.o_indptr: Optional[torch.Tensor] = None
        self.block_valid_mask: Optional[torch.Tensor] = None
        self.kv_pages: Optional[torch.Tensor] = None
        self.split_counts: Optional[torch.Tensor] = None
        self.split_kv: bool = False
        self.cta_tile_q: int = 0
        self.num_q_tiles: int = 0
        self.kv_chunk_size_pages: int = 0
        self.kv_chunk_size_tokens: int = 0
        self.work_count: int = 0
        self.padded_work_count: int = 0
        self.O_partial: Optional[torch.Tensor] = None
        self.LSE_partial: Optional[torch.Tensor] = None
        # Cached dummy buffers used in non-split path to satisfy the kernel's
        # positional arg signature without per-call torch.empty (saves ~5us
        # on every run() for small kv).
        self._O_partial_dummy: Optional[torch.Tensor] = None
        self._LSE_partial_dummy: Optional[torch.Tensor] = None
        # When the caller doesn't ask for LSE, the kernel still needs a valid
        # tensor pointer to write to.  Cache a small placeholder so run() can
        # skip the per-call torch.empty for it as well.
        self._lse_dummy: Optional[torch.Tensor] = None

    def plan(
        self,
        *,
        page_table: torch.Tensor,
        seqused_k: torch.Tensor,
        seqlen_q: int,
        max_seqlen_k: int,
        q2k_indices: Optional[torch.Tensor] = None,
        num_qo_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = 128,
        enable_cuda_graph: bool = False,
        max_grid_size: Optional[int] = None,
        fixed_split_size: Optional[int] = None,
        disable_split_kv: bool = False,
    ) -> "SparseDecodePagedAttentionWrapper":
        """Prepare decode scheduling metadata and reusable workspaces.

        Parameters
        ----------
        page_table : torch.Tensor
            Shape ``[batch_size, max_num_pages_per_seq]``, dtype int32.  Maps
            logical pages to physical KV-cache pages.
        seqused_k : torch.Tensor
            Shape ``[batch_size]``, dtype int32.  Effective KV length per
            request.
        seqlen_q : int
            Uniform query length per request.
        max_seqlen_k : int
            Maximum KV sequence length in the batch.
        q2k_indices : torch.Tensor, optional
            Sparse selected KV blocks with shape ``[Hkv, total_q, topK]`` and
            dtype int32.  ``None`` selects the dense all-KV path.
        num_qo_heads : int
            Number of Q/O heads.
        num_kv_heads : int
            Number of KV heads.  Current decode kernel requires
            ``num_qo_heads / num_kv_heads == 16`` at run time.
        head_dim : int, optional
            Head dimension.  Must be 128.
        enable_cuda_graph : bool, optional
            Build schedule metadata compatible with CUDA graph capture.
        max_grid_size : int, optional
            Override maximum CTA count used by the scheduler.
        fixed_split_size : int, optional
            Force a fixed split-KV chunk size in pages.
        disable_split_kv : bool, optional
            Disable split-KV even for long KV sequences.

        Returns
        -------
        SparseDecodePagedAttentionWrapper
            ``self``, planned and ready for ``run``.
        """
        if page_table.ndim != 2:
            raise ValueError(
                "decode plan requires page_table with shape [B, max_num_pages_per_seq]"
            )
        if page_table.dtype != torch.int32:
            raise TypeError("decode plan requires page_table to be torch.int32")
        if seqused_k.dtype != torch.int32:
            raise TypeError("decode plan requires seqused_k to be torch.int32")
        if not page_table.is_cuda or not seqused_k.is_cuda:
            raise ValueError(
                "decode plan requires page_table and seqused_k to be CUDA tensors"
            )
        if page_table.device != seqused_k.device:
            raise ValueError(
                "decode plan requires page_table and seqused_k on the same device"
            )
        if page_table.stride(-1) != 1:
            raise ValueError(
                "decode plan requires page_table contiguous in the last dimension"
            )
        if seqused_k.shape != (int(page_table.shape[0]),):
            raise ValueError("decode plan requires seqused_k with shape [B]")
        if q2k_indices is not None and q2k_indices.dtype != torch.int32:
            raise TypeError("decode plan requires q2k_indices to be torch.int32")
        if int(seqlen_q) <= 0 or int(max_seqlen_k) <= 0:
            raise ValueError("decode plan requires positive seqlen_q and max_seqlen_k")
        if num_qo_heads is None or num_kv_heads is None or head_dim is None:
            raise ValueError(
                "decode plan requires num_qo_heads, num_kv_heads, and head_dim"
            )
        if head_dim is not None and int(head_dim) != 128:
            raise NotImplementedError(
                "decode plan currently supports only head_dim=128"
            )
        if int(num_qo_heads) % int(num_kv_heads) != 0:
            raise ValueError(
                "decode plan requires num_qo_heads divisible by num_kv_heads"
            )

        self.batch = int(page_table.shape[0])
        self.num_qo_heads = None if num_qo_heads is None else int(num_qo_heads)
        self.num_kv_heads = None if num_kv_heads is None else int(num_kv_heads)
        self.head_dim = None if head_dim is None else int(head_dim)
        self.page_table = page_table.contiguous()
        self.seqused_k = seqused_k.contiguous()
        self.q2k_indices = None if q2k_indices is None else q2k_indices.contiguous()
        self.seqlen_q = int(seqlen_q)
        self.max_seqlen_k = int(max_seqlen_k)
        self.is_sparse = q2k_indices is not None

        # max_grid_size is hardcoded to num_sms (1 CTA/SM) inside the C++
        # schedule launcher because the decode attn kernel always runs at
        # 1 CTA/SM (its register/smem budget saturates the SM).  Callers
        # can still override via the explicit max_grid_size kwarg.
        schedule = prepare_decode_schedule(
            seqused_k=self.seqused_k,
            page_size=self.blk_kv,
            seqlen_q=self.seqlen_q,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_seqlen_k=self.max_seqlen_k,
            enable_cuda_graph=bool(enable_cuda_graph),
            max_grid_size=max_grid_size,
            fixed_split_size=fixed_split_size,
            disable_split_kv=bool(disable_split_kv),
        )
        self.decode_schedule = schedule
        self.request_indices = schedule.request_indices
        self.qo_tile_indices = schedule.qo_tile_indices
        self.kv_tile_indices = schedule.kv_tile_indices
        self.merge_indptr = schedule.merge_indptr
        self.o_indptr = schedule.o_indptr
        self.block_valid_mask = schedule.block_valid_mask
        self.kv_pages = schedule.kv_pages
        self.split_counts = schedule.split_counts
        self.split_kv = schedule.split_kv
        self.cta_tile_q = schedule.cta_tile_q
        self.num_q_tiles = schedule.num_q_tiles
        self.kv_chunk_size_pages = schedule.kv_chunk_size_pages
        self.kv_chunk_size_tokens = schedule.kv_chunk_size_tokens
        self.work_count = schedule.work_count
        self.padded_work_count = schedule.padded_work_count
        if schedule.split_kv:
            self.O_partial = torch.empty(
                (schedule.partial_rows, self.num_qo_heads, self.head_dim),
                dtype=torch.float32,
                device=page_table.device,
            )
            self.LSE_partial = torch.empty(
                (schedule.partial_rows, self.num_qo_heads),
                dtype=torch.float32,
                device=page_table.device,
            )
            self._O_partial_dummy = None
            self._LSE_partial_dummy = None
        else:
            self.O_partial = None
            self.LSE_partial = None
            # decode_forward_paged_fp8 always wants non-None partial buffers
            # for the kernel's positional arg layout (compile keeps the slot
            # alive even when split_kv=False).  Allocate once here and reuse.
            self._O_partial_dummy = torch.empty(
                (1, self.head_dim),
                dtype=torch.float32,
                device=page_table.device,
            )
            self._LSE_partial_dummy = torch.empty(
                (1, self.num_qo_heads),
                dtype=torch.float32,
                device=page_table.device,
            )
        # LSE dummy is shape (1, head_q) — used when caller doesn't request
        # LSE and the schedule isn't split-KV (split-KV always writes LSE).
        self._lse_dummy = torch.empty(
            (1, self.num_qo_heads),
            dtype=torch.float32,
            device=page_table.device,
        )
        return self

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        softmax_scale: Optional[float] = None,
        return_softmax_lse: bool = False,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
    ):
        """Launch decode using metadata cached by ``plan``.

        Parameters
        ----------
        q : torch.Tensor
            Shape ``[batch_size * seqlen_q, Hq, 128]`` and dtype FP8 E4M3.
        k : torch.Tensor
            Paged K cache with shape ``[num_pages, Hkv, blk_kv, 128]``.
        v : torch.Tensor
            Paged V cache with the same shape as ``k``.
        softmax_scale : float, optional
            Softmax scale.  Defaults to ``1 / sqrt(128)``.
        return_softmax_lse : bool, optional
            If True, return ``(out, lse)``.
        out : torch.Tensor, optional
            Preallocated BF16 output buffer with shape ``q.shape``.
        lse : torch.Tensor, optional
            Preallocated float32 LSE buffer with shape ``[total_q, Hq]``.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]
            BF16 output, optionally with float32 LSE.
        """
        if self.decode_schedule is None:
            raise RuntimeError("decode wrapper must be planned before run")
        if self.is_sparse:
            # Sparse path still goes through the validating wrapper for now;
            # only the dense fast path is collapsed.
            return sparse_decode_atten_func(
                q,
                k,
                v,
                self.q2k_indices,
                page_table=self.page_table,
                seqused_k=self.seqused_k,
                seqlen_q=self.seqlen_q,
                max_seqlen_k=self.max_seqlen_k,
                blk_kv=self.blk_kv,
                causal=self.causal,
                softmax_scale=softmax_scale,
                return_softmax_lse=return_softmax_lse,
                schedule=self.decode_schedule,
                O_partial=self.O_partial,
                LSE_partial=self.LSE_partial,
            )

        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        if out is None:
            out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
        if lse is None:
            if return_softmax_lse or self.split_kv:
                # Real LSE needed — must allocate per-call (shape depends on q).
                lse = torch.empty(
                    q.shape[:2],
                    dtype=torch.float32,
                    device=q.device,
                )
            else:
                # Kernel only needs a valid pointer; reuse cached dummy.
                lse = self._lse_dummy
        from src.sm100.fwd_decode import decode_forward_paged_fp8

        schedule = self.decode_schedule
        decode_forward_paged_fp8(
            q,
            k,
            v,
            self.page_table,
            self.seqused_k,
            out,
            lse,
            schedule.request_indices,
            schedule.qo_tile_indices,
            schedule.kv_tile_indices,
            schedule.block_valid_mask,
            schedule.split_counts,
            schedule.o_indptr,
            schedule.merge_indptr,
            self.O_partial,
            self.LSE_partial,
            softmax_scale=float(softmax_scale),
            seqlen_q=self.seqlen_q,
            page_size=self.blk_kv,
            kv_chunk_size_pages=int(schedule.kv_chunk_size_pages),
            max_split_count=int(schedule.max_split_count),
            split_kv=bool(schedule.split_kv),
            causal=self.causal,
            return_lse=bool(return_softmax_lse),
            # cached dummies — avoid per-call torch.empty inside run_decode_attention
            O_partial_dummy=self._O_partial_dummy,
            LSE_partial_dummy=self._LSE_partial_dummy,
        )
        if return_softmax_lse:
            return out, lse
        return out


def _sparse_atten_csr_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    topK: int,
    blk_kv: int,
    causal: bool,
    softmax_scale: float,
    lse_temperature_scale: float,
    return_temperature_lse: bool,
    partial_dtype: torch.dtype,
    return_softmax_lse: bool,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    schedule: Optional[SparseAttentionSchedule],
    usable_SM_count: int,
    batch: int,
    head_kv: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    qk_dtype: torch.dtype,
    pv_dtype: torch.dtype,
    output_scale: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
):
    total_q, head_q, dim = q.shape
    if head_q % head_kv != 0:
        raise ValueError("q.shape[1] must be divisible by head_kv")
    max_num_kv_blocks = _csr_row_capacity(k2q_row_ptr)
    temperature_lse_fast_path = return_temperature_lse and math.isclose(
        float(lse_temperature_scale),
        1.0,
        rel_tol=0.0,
        abs_tol=_TEMPERATURE_LSE_FAST_PATH_ABS_TOL,
    )
    kernel_return_temperature_lse = (
        return_temperature_lse and not temperature_lse_fast_path
    )

    O_partial = torch.empty(
        topK, total_q, head_q, dim, dtype=partial_dtype, device=q.device
    )
    LSE_partial = torch.empty(
        topK, total_q, head_q, dtype=torch.float32, device=q.device
    )
    LSE_temperature_partial = (
        torch.empty(topK, total_q, head_q, dtype=torch.float32, device=q.device)
        if kernel_return_temperature_lse
        else None
    )
    O_out = (
        out
        if out is not None
        else torch.empty(total_q, head_q, dim, dtype=torch.bfloat16, device=q.device)
    )
    LSE_out = torch.empty(total_q, head_q, dtype=torch.float32, device=q.device)
    LSE_temperature_out = (
        torch.empty_like(LSE_out) if kernel_return_temperature_lse else None
    )
    if schedule is None:
        k2q_qsplit_indices = torch.empty_like(k2q_q_indices)
        split_counts = torch.zeros(
            (total_q, head_kv),
            dtype=torch.int32,
            device=q.device,
        )
    else:
        _validate_fwd_schedule(
            schedule,
            q=q,
            k2q_q_indices=k2q_q_indices,
            head_kv=head_kv,
        )
        k2q_qsplit_indices = schedule.qsplit_indices
        split_counts = schedule.split_counts
    schedule = _call_sparse_forward_sm100_csr_varlen(
        q,
        k,
        v,
        k2q_row_ptr,
        k2q_q_indices,
        k2q_qsplit_indices,
        split_counts,
        cu_seqlens_q,
        cu_seqlens_k,
        page_table,
        seqused_k,
        O_partial,
        LSE_partial,
        LSE_temperature_partial,
        softmax_scale,
        lse_temperature_scale,
        kernel_return_temperature_lse,
        max_num_kv_blocks,
        blk_kv,
        head_kv,
        max_seqlen_q,
        usable_SM_count,
        causal=causal,
        schedule=schedule,
        qk_dtype=qk_dtype,
        pv_dtype=pv_dtype,
    )
    # Sparse Attention and Sparse Page Attention both use the varlen-Q
    # combine path; the kernel-written LSE_out is the final contract.
    combine(
        O_partial,
        LSE_partial,
        O_out,
        LSE_out,
        lse_temperature_partial=LSE_temperature_partial,
        lse_temperature_out=LSE_temperature_out,
        cu_seqlens=cu_seqlens_q,
        split_counts=split_counts,
        use_pdl=True,
        output_scale=output_scale,
    )
    if temperature_lse_fast_path:
        LSE_temperature_out = LSE_out

    if return_softmax_lse:
        if return_temperature_lse:
            return O_out, LSE_out, LSE_temperature_out
        return O_out, LSE_out
    return O_out


def _call_sparse_decode_forward_sm100_paged_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: Optional[torch.Tensor],
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    schedule: DecodeAttentionSchedule,
    O_partial: Optional[torch.Tensor],
    LSE_partial: Optional[torch.Tensor],
    *,
    softmax_scale: float,
    seqlen_q: int,
    max_seqlen_k: int,
    blk_kv: int,
    causal: bool,
    return_lse: bool = True,
) -> None:
    """Compile and launch the SM100 paged fp8 decode forward kernel.

    Dense decode is selected by ``q2k_indices=None``.  Sparse decode will reuse
    the same schedule wrapper but needs a separate q2k gather path.
    """
    if q2k_indices is not None:
        raise NotImplementedError(
            "SM100 paged fp8 sparse decode forward is not implemented yet"
        )
    if schedule.cta_tile_q != 128:
        raise NotImplementedError(
            f"decode forward requires cta_tile_q=128, got {schedule.cta_tile_q}"
        )
    if schedule.split_kv:
        if O_partial is None or LSE_partial is None:
            raise ValueError("split decode forward requires O_partial and LSE_partial")
        if O_partial.dtype != torch.float32:
            raise TypeError(f"O_partial must be torch.float32, got {O_partial.dtype}")
        if LSE_partial.dtype != torch.float32:
            raise TypeError(
                f"LSE_partial must be torch.float32, got {LSE_partial.dtype}"
            )

    from src.sm100.fwd_decode import decode_forward_paged_fp8

    decode_forward_paged_fp8(
        q,
        k,
        v,
        page_table,
        seqused_k,
        out,
        lse,
        schedule.request_indices,
        schedule.qo_tile_indices,
        schedule.kv_tile_indices,
        schedule.block_valid_mask,
        schedule.split_counts,
        schedule.o_indptr,
        schedule.merge_indptr,
        O_partial,
        LSE_partial,
        softmax_scale=float(softmax_scale),
        seqlen_q=int(seqlen_q),
        page_size=int(blk_kv),
        kv_chunk_size_pages=int(schedule.kv_chunk_size_pages),
        max_split_count=int(schedule.max_split_count),
        split_kv=bool(schedule.split_kv),
        causal=bool(causal),
        return_lse=bool(return_lse),
    )


def _call_sparse_forward_sm100_csr_varlen(
    q,
    k,
    v,
    k2q_row_ptr,
    k2q_q_indices,
    k2q_qsplit_indices,
    split_counts,
    cu_seqlens_q,
    cu_seqlens_k,
    page_table,
    seqused_k,
    O_partial,
    LSE_partial,
    LSE_temperature_partial,
    softmax_scale,
    lse_temperature_scale,
    return_temperature_lse,
    max_num_kv_blocks,
    blk_kv,
    head_kv,
    max_seqlen_q,
    usable_SM_count=-1,
    *,
    causal=False,
    use_prepare_scheduler=True,
    schedule: Optional[SparseAttentionSchedule] = None,
    qk_dtype: torch.dtype,
    pv_dtype: torch.dtype,
):
    """Compile and launch the SM100 sparse forward K1 kernel on CSR metadata."""
    head_dim = q.shape[-1]
    dtype = q.dtype
    qk_dtype = _normalize_forward_mma_dtype(qk_dtype, q.dtype, "qk_dtype")
    pv_dtype = _normalize_forward_mma_dtype(pv_dtype, v.dtype, "pv_dtype")
    partial_dtype = O_partial.dtype
    return_temperature_lse = bool(return_temperature_lse)
    if return_temperature_lse != (LSE_temperature_partial is not None):
        raise ValueError(
            "return_temperature_lse must match LSE_temperature_partial presence"
        )
    lse_temperature_scale = float(lse_temperature_scale)
    if not math.isfinite(lse_temperature_scale) or lse_temperature_scale <= 0.0:
        raise ValueError(
            f"lse_temperature_scale must be finite and > 0, got {lse_temperature_scale}"
        )
    lse_temperature_inv_scale = 1.0 / lse_temperature_scale
    n_block_size = int(blk_kv)
    head_q = q.shape[1]
    qhead_per_kv = head_q // head_kv
    paged_kv = page_table is not None
    if not bool(use_prepare_scheduler):
        raise RuntimeError("sparse forward requires prepare scheduler")
    schedule_enabled = k2q_row_ptr.shape[1] > 1
    page_size = int(k.shape[2]) if paged_kv else None
    if paged_kv:
        k_kernel, v_kernel = _prepare_paged_kv_for_tma(k, v, n_block_size)
    else:
        k_kernel = k
        v_kernel = v
    O_partial_flat = O_partial.reshape(-1, head_dim).contiguous()
    Q_flat = q.reshape(-1, head_dim).contiguous()
    Q_gather4_desc = (
        create_q_gather4_tma_desc(
            Q_flat,
            box_x=128 if q.dtype == torch.float8_e4m3fn else 64,
        )
        if qhead_per_kv in (1, 2, 4)
        else None
    )
    if schedule is None:
        schedule = prepare_sparse_fwd_schedule_and_split(
            k2q_row_ptr=k2q_row_ptr,
            k2q_q_indices=k2q_q_indices,
            k2q_qsplit_indices=k2q_qsplit_indices,
            split_counts=split_counts,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            total_q=int(q.shape[0]),
            max_seqlen_q=max_seqlen_q,
            topk=int(O_partial.shape[0]),
            head_kv=head_kv,
            qhead_per_kv=qhead_per_kv,
            blk_kv=n_block_size,
            device=q.device,
            enabled=schedule_enabled,
        )
    use_prepare_scheduler = schedule.enabled
    scheduler_metadata = schedule.scheduler_metadata
    work_count = schedule.work_count
    work_capacity = schedule.work_capacity
    if (
        not use_prepare_scheduler
        or scheduler_metadata is None
        or work_count is None
        or work_capacity <= 0
    ):
        raise RuntimeError("sparse forward requires a non-empty prepared schedule")

    key = (
        "sparse_forward_sm100_csr_varlen",
        head_dim,
        n_block_size,
        qhead_per_kv,
        dtype,
        k.dtype,
        v.dtype,
        qk_dtype,
        pv_dtype,
        partial_dtype,
        bool(causal),
        bool(paged_kv),
        bool(use_prepare_scheduler),
        page_size,
        bool(seqused_k is not None),
        bool(return_temperature_lse),
    )
    if key not in _compile_cache:
        from src.common.aot_cache import save_aot, try_load_aot

        loaded = try_load_aot(key)
        if loaded is not None:
            _compile_cache[key] = loaded
        else:
            kernel = SparseAttentionForwardSm100(
                head_dim=head_dim,
                qheadperkv=qhead_per_kv,
                n_block_size=n_block_size,
                paged_kv=paged_kv,
                page_size=page_size,
                has_seqused_k=seqused_k is not None,
                causal=bool(causal),
                use_prepare_scheduler=use_prepare_scheduler,
                qk_dtype=_torch_dtype_to_cutlass_dtype(qk_dtype),
                pv_dtype=_torch_dtype_to_cutlass_dtype(pv_dtype),
            )
            _compile_cache[key] = cute.compile(
                kernel,
                to_cute_tensor_kvouter(k_kernel),
                to_cute_tensor_kvouter(v_kernel),
                to_cute_tensor_kvouter(k2q_q_indices),
                to_cute_tensor_kvouter(k2q_qsplit_indices),
                to_cute_tensor_kvouter(k2q_row_ptr),
                (
                    None
                    if scheduler_metadata is None
                    else to_cute_tensor_kvouter(scheduler_metadata)
                ),
                None if work_count is None else to_cute_tensor_kvouter(work_count),
                to_cute_tensor_kvouter(O_partial_flat),
                to_cute_tensor_kvouter(LSE_partial),
                (
                    None
                    if LSE_temperature_partial is None
                    else to_cute_tensor_kvouter(LSE_temperature_partial)
                ),
                to_cute_tensor_kvouter(Q_flat),
                (
                    None
                    if Q_gather4_desc is None
                    else to_cute_tensor_kvouter(Q_gather4_desc)
                ),
                None if page_table is None else _to_cute_int32_metadata(page_table),
                None if seqused_k is None else _to_cute_int32_metadata(seqused_k),
                _to_cute_int32_metadata(cu_seqlens_q),
                _to_cute_int32_metadata(cu_seqlens_k),
                Float32(softmax_scale),
                Float32(lse_temperature_inv_scale),
                Int32(max_num_kv_blocks),
                Int32(head_kv),
                Int32(max_seqlen_q),
                Int32(work_capacity),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )
            save_aot(key, _compile_cache[key])

    with torch.cuda.nvtx.range("Fwd_SparseAttn_Sm100_CsrVarlen"):
        _compile_cache[key](
            k_kernel,
            v_kernel,
            k2q_q_indices,
            k2q_qsplit_indices,
            k2q_row_ptr,
            scheduler_metadata,
            work_count,
            O_partial_flat,
            LSE_partial,
            LSE_temperature_partial,
            Q_flat,
            Q_gather4_desc,
            page_table,
            seqused_k,
            cu_seqlens_q,
            cu_seqlens_k,
            softmax_scale,
            lse_temperature_inv_scale,
            max_num_kv_blocks,
            head_kv,
            max_seqlen_q,
            work_capacity,
        )
    return schedule


def _call_sparse_forward_sm100_csr_varlen_nvfp4_kv(
    q,
    k,
    v,
    k_scale_128x4,
    v_scale_128x4,
    k_global_scale,
    v_global_scale,
    k2q_row_ptr,
    k2q_q_indices,
    k2q_qsplit_indices,
    split_counts,
    cu_seqlens_q,
    cu_seqlens_k,
    page_table,
    seqused_k,
    O_partial,
    LSE_partial,
    LSE_temperature_partial,
    softmax_scale,
    lse_temperature_scale,
    return_temperature_lse,
    max_num_kv_blocks,
    blk_kv,
    head_kv,
    max_seqlen_q,
    *,
    causal=False,
    use_prepare_scheduler=True,
    schedule: Optional[SparseAttentionSchedule] = None,
):
    """Compile and launch the SM100 sparse forward K1 kernel with NVFP4 K/V."""

    head_dim = q.shape[-1]
    dtype = q.dtype
    partial_dtype = O_partial.dtype
    return_temperature_lse = bool(return_temperature_lse)
    if return_temperature_lse != (LSE_temperature_partial is not None):
        raise ValueError(
            "return_temperature_lse must match LSE_temperature_partial presence"
        )
    lse_temperature_scale = float(lse_temperature_scale)
    if not math.isfinite(lse_temperature_scale) or lse_temperature_scale <= 0.0:
        raise ValueError(
            f"lse_temperature_scale must be finite and > 0, got {lse_temperature_scale}"
        )
    lse_temperature_inv_scale = 1.0 / lse_temperature_scale
    n_block_size = int(blk_kv)
    head_q = q.shape[1]
    qhead_per_kv = head_q // head_kv
    fp8_pair_dequant = os.environ.get("MINIMAX_KVFP4_FP8_PAIR_DEQUANT", "1") != "0"
    k_global_scale_kernel = k_global_scale
    # V global scale is linear in the final output. Keep K1 on block-scale-only V
    # and apply the tensor scale once in K2 combine.
    v_global_scale_kernel = None
    has_k_global_scale = k_global_scale_kernel is not None
    has_v_global_scale = v_global_scale_kernel is not None
    paged_kv = page_table is not None
    if not bool(use_prepare_scheduler):
        raise RuntimeError("KVFP4 sparse forward requires prepare scheduler")
    schedule_enabled = k2q_row_ptr.shape[1] > 1
    page_size = int(k.shape[2]) if paged_kv else None
    if paged_kv:
        _prepare_paged_kv_for_tma(k, v, n_block_size)
    k_kernel = k
    v_kernel = v
    O_partial_flat = O_partial.reshape(-1, head_dim).contiguous()
    Q_flat = q.reshape(-1, head_dim).contiguous()
    Q_gather4_desc = (
        create_q_gather4_tma_desc(
            Q_flat,
            box_x=128 if q.dtype == torch.float8_e4m3fn else 64,
        )
        if qhead_per_kv in (1, 2, 4)
        else None
    )
    if schedule is None:
        schedule = prepare_sparse_fwd_schedule_and_split(
            k2q_row_ptr=k2q_row_ptr,
            k2q_q_indices=k2q_q_indices,
            k2q_qsplit_indices=k2q_qsplit_indices,
            split_counts=split_counts,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            total_q=int(q.shape[0]),
            max_seqlen_q=max_seqlen_q,
            topk=int(O_partial.shape[0]),
            head_kv=head_kv,
            qhead_per_kv=qhead_per_kv,
            blk_kv=n_block_size,
            device=q.device,
            enabled=schedule_enabled,
        )
    use_prepare_scheduler = schedule.enabled
    scheduler_metadata = schedule.scheduler_metadata
    work_count = schedule.work_count
    work_capacity = schedule.work_capacity
    if (
        not use_prepare_scheduler
        or scheduler_metadata is None
        or work_count is None
        or work_capacity <= 0
    ):
        raise RuntimeError(
            "KVFP4 sparse forward requires a non-empty prepared schedule"
        )

    key = (
        "sparse_forward_sm100_csr_varlen_nvfp4_kv",
        head_dim,
        n_block_size,
        qhead_per_kv,
        dtype,
        partial_dtype,
        bool(causal),
        bool(paged_kv),
        bool(use_prepare_scheduler),
        page_size,
        bool(seqused_k is not None),
        bool(return_temperature_lse),
        bool(fp8_pair_dequant),
        bool(has_k_global_scale),
        bool(has_v_global_scale),
    )
    if key not in _compile_cache:
        from src.common.aot_cache import save_aot, try_load_aot

        loaded = try_load_aot(key)
        if loaded is not None:
            _compile_cache[key] = loaded
        else:
            kernel = SparseAttentionForwardNvfp4KvSm100(
                head_dim=head_dim,
                qheadperkv=qhead_per_kv,
                n_block_size=n_block_size,
                paged_kv=paged_kv,
                page_size=page_size,
                has_seqused_k=seqused_k is not None,
                causal=bool(causal),
                use_prepare_scheduler=use_prepare_scheduler,
                fp8_pair_dequant=bool(fp8_pair_dequant),
                has_k_global_scale=bool(has_k_global_scale),
                has_v_global_scale=bool(has_v_global_scale),
            )
            _compile_cache[key] = cute.compile(
                kernel,
                to_cute_tensor_kvouter(k_kernel),
                to_cute_tensor_kvouter(v_kernel),
                to_cute_tensor_kvouter(k_scale_128x4),
                to_cute_tensor_kvouter(v_scale_128x4),
                (
                    None
                    if k_global_scale_kernel is None
                    else to_cute_tensor_kvouter(k_global_scale_kernel)
                ),
                (
                    None
                    if v_global_scale_kernel is None
                    else to_cute_tensor_kvouter(v_global_scale_kernel)
                ),
                to_cute_tensor_kvouter(k2q_q_indices),
                to_cute_tensor_kvouter(k2q_qsplit_indices),
                to_cute_tensor_kvouter(k2q_row_ptr),
                (
                    None
                    if scheduler_metadata is None
                    else to_cute_tensor_kvouter(scheduler_metadata)
                ),
                None if work_count is None else to_cute_tensor_kvouter(work_count),
                to_cute_tensor_kvouter(O_partial_flat),
                to_cute_tensor_kvouter(LSE_partial),
                (
                    None
                    if LSE_temperature_partial is None
                    else to_cute_tensor_kvouter(LSE_temperature_partial)
                ),
                to_cute_tensor_kvouter(Q_flat),
                (
                    None
                    if Q_gather4_desc is None
                    else to_cute_tensor_kvouter(Q_gather4_desc)
                ),
                None if page_table is None else _to_cute_int32_metadata(page_table),
                None if seqused_k is None else _to_cute_int32_metadata(seqused_k),
                _to_cute_int32_metadata(cu_seqlens_q),
                _to_cute_int32_metadata(cu_seqlens_k),
                Float32(softmax_scale),
                Float32(lse_temperature_inv_scale),
                Int32(max_num_kv_blocks),
                Int32(head_kv),
                Int32(max_seqlen_q),
                Int32(work_capacity),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )
            save_aot(key, _compile_cache[key])

    with torch.cuda.nvtx.range("Fwd_SparseAttn_Sm100_CsrVarlen_KVFP4"):
        _compile_cache[key](
            k_kernel,
            v_kernel,
            k_scale_128x4,
            v_scale_128x4,
            k_global_scale_kernel,
            v_global_scale_kernel,
            k2q_q_indices,
            k2q_qsplit_indices,
            k2q_row_ptr,
            scheduler_metadata,
            work_count,
            O_partial_flat,
            LSE_partial,
            LSE_temperature_partial,
            Q_flat,
            Q_gather4_desc,
            page_table,
            seqused_k,
            cu_seqlens_q,
            cu_seqlens_k,
            softmax_scale,
            lse_temperature_inv_scale,
            max_num_kv_blocks,
            head_kv,
            max_seqlen_q,
            work_capacity,
        )
    return schedule
