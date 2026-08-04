# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""AOT CUDA extension for the q2k to k2q CSR builder."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.msa.cuda._extension import load_extension

_EXTENSION_NAME = "sparse_build_k2q_csr_ext"


def _current_stream_ptr(device: torch.device) -> int:
    return int(torch.cuda.current_stream(device).cuda_stream)


def run_build_k2q_csr(
    q2k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    row_ptr: torch.Tensor,
    q_idx: torch.Tensor,
    topk: int,
    blk_kv: int,
    total_rows: int,
    max_kv_blocks: int,
) -> None:
    """Fill the k2q CSR row pointers and query indices in place."""
    load_extension(_EXTENSION_NAME).run_build_k2q_csr(
        q2k,
        cu_seqlens_q,
        cu_seqlens_k,
        row_ptr,
        q_idx,
        int(topk),
        int(blk_kv),
        int(total_rows),
        int(max_kv_blocks),
        _current_stream_ptr(q2k.device),
    )


def run_build_k2q_csr_with_schedule(
    q2k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    row_ptr: torch.Tensor,
    q_idx: torch.Tensor,
    scheduler_metadata: torch.Tensor,
    work_count: torch.Tensor,
    qsplit_idx: torch.Tensor,
    split_counts: torch.Tensor,
    topk: int,
    blk_kv: int,
    total_rows: int,
    max_kv_blocks: int,
    target_q_per_cta: int,
    work_capacity: int,
    max_seqlen_q: int,
) -> None:
    """Fill CSR and fused sparse-attention schedule metadata in place."""
    load_extension(_EXTENSION_NAME).run_build_k2q_csr_with_schedule(
        q2k,
        cu_seqlens_q,
        cu_seqlens_k,
        row_ptr,
        q_idx,
        scheduler_metadata,
        work_count,
        qsplit_idx,
        split_counts,
        int(topk),
        int(blk_kv),
        int(total_rows),
        int(max_kv_blocks),
        int(target_q_per_cta),
        int(work_capacity),
        int(max_seqlen_q),
        _current_stream_ptr(q2k.device),
    )


def is_supported(topk: int, blk_kv: int) -> bool:
    return int(topk) in (4, 8, 16, 32) and int(blk_kv) == 128


__all__ = ["is_supported", "run_build_k2q_csr", "run_build_k2q_csr_with_schedule"]
