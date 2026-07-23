# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Sparse k2q CSR builder for SM100.

Thin dispatcher that calls the CUDA C++ kernel pipeline in
``src.sm100.build_k2q_csr``. Supports ``topK in {4, 8, 16, 32}`` and
``blk_kv == 128`` only — other shapes raise ``ValueError`` rather than
silently falling back to a torch-reference path.
"""

from __future__ import annotations

from typing import Optional

import torch
from src.sm100.prepare_scheduler import SPARSE_SCHEDULE_MODEL, SparseAttentionSchedule

_SUPPORTED_TOPK = (4, 8, 16, 32)
_SUPPORTED_BLK_KV = 128


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


class SparseK2qCsrBuilderSm100:
    """Build the k2q CSR reverse index for sparse attention on SM100.

    The public API matches the historical CUTE DSL builder so callers
    (``sparse_index_utils.build_k2q_csr``, attention kernels) need no
    changes. Internally the kernel pipeline runs five CUDA C++ kernels:
    ``build_row_map`` -> ``hist`` -> ``row_prefix`` -> ``tile_prefix_smem``
    -> ``scatter`` (5 kernels + 2 ``cudaMemsetAsync``).
    """

    def __init__(self) -> None:
        # No persistent state — the JIT-compiled extension is loaded
        # lazily by ``src.sm100.build_k2q_csr`` on first call.
        self._run = None
        self._run_with_schedule = None

    def _ensure_loaded(self) -> None:
        if self._run is None:
            from src.sm100.build_k2q_csr import (
                run_build_k2q_csr,
                run_build_k2q_csr_with_schedule,
            )

            self._run = run_build_k2q_csr
            self._run_with_schedule = run_build_k2q_csr_with_schedule

    def __call__(
        self,
        q2k_indices: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        *,
        total_k: int,
        blk_kv: int = 128,
        max_seqlen_k: Optional[int] = None,
        max_seqlen_q: Optional[int] = None,
        total_rows: Optional[int] = None,
        qhead_per_kv: int = 1,
        return_schedule: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, SparseAttentionSchedule]
    ):
        # ---- Validation ----------------------------------------------------
        if blk_kv != _SUPPORTED_BLK_KV:
            raise ValueError(
                f"SparseK2qCsrBuilderSm100 only supports blk_kv == "
                f"{_SUPPORTED_BLK_KV}, got {blk_kv}"
            )
        if q2k_indices.dtype != torch.int32:
            raise TypeError(f"q2k_indices must be torch.int32, got {q2k_indices.dtype}")
        if q2k_indices.ndim != 3:
            raise ValueError(
                f"q2k_indices must be rank-3 [head_kv, total_q, topK], "
                f"got shape {tuple(q2k_indices.shape)}"
            )
        # Kernel int4 loads require a contiguous topK dim, int4-aligned
        # (multiple-of-4) head/seq strides, and a 16B-aligned base. Enforce
        # strictly so a misaligned caller fails loudly instead of paying a
        # silent contiguous-copy on the hot path.
        if (
            q2k_indices.stride(2) != 1
            or q2k_indices.stride(0) % 4 != 0
            or q2k_indices.stride(1) % 4 != 0
            or q2k_indices.data_ptr() % 16 != 0
        ):
            raise ValueError(
                "q2k_indices must have a contiguous topK dim, int4-aligned "
                "(multiple-of-4) head/seq strides, and a 16-byte-aligned base; "
                f"got strides {tuple(q2k_indices.stride())} and "
                f"data_ptr % 16 == {q2k_indices.data_ptr() % 16}"
            )
        if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
            raise TypeError("cu_seqlens_q and cu_seqlens_k must be torch.int32")
        if cu_seqlens_q.ndim != 1 or cu_seqlens_k.ndim != 1:
            raise ValueError("cu_seqlens_q and cu_seqlens_k must be rank-1")
        if cu_seqlens_q.shape != cu_seqlens_k.shape:
            raise ValueError("cu_seqlens_q and cu_seqlens_k must share shape [B + 1]")
        if not (q2k_indices.is_cuda and cu_seqlens_q.is_cuda and cu_seqlens_k.is_cuda):
            raise ValueError("all inputs must be CUDA tensors")
        if (
            q2k_indices.device != cu_seqlens_q.device
            or q2k_indices.device != cu_seqlens_k.device
        ):
            raise ValueError("all inputs must share a device")
        if not cu_seqlens_q.is_contiguous() or not cu_seqlens_k.is_contiguous():
            raise ValueError("cu_seqlens_q and cu_seqlens_k must be contiguous")

        total_k = int(total_k)
        if total_k < 0:
            raise ValueError(f"total_k must be non-negative, got {total_k}")

        head_kv, total_q, topk = (int(v) for v in q2k_indices.shape)
        if topk not in _SUPPORTED_TOPK:
            raise ValueError(
                f"SparseK2qCsrBuilderSm100 only supports topK in "
                f"{_SUPPORTED_TOPK}, got {topk}"
            )

        batch = int(cu_seqlens_q.shape[0] - 1)
        if batch < 0:
            raise ValueError("cu_seqlens tensors must have shape [B + 1]")
        if return_schedule and max_seqlen_k is None:
            raise ValueError(
                "build_k2q_csr requires max_seqlen_k when return_schedule=True"
            )
        max_k_tokens = int(max_seqlen_k) if max_seqlen_k is not None else total_k
        max_kv_blocks = _ceil_div(max(max_k_tokens, blk_kv), blk_kv)
        if total_rows is not None:
            total_rows = int(total_rows)
        elif total_k % blk_kv == 0:
            total_rows = total_k // blk_kv
        else:
            total_rows = _ceil_div(total_k + batch * (blk_kv - 1), blk_kv)
        if total_rows < 0:
            raise ValueError(f"total_rows must be non-negative, got {total_rows}")
        total_rows = max(total_rows, 0)
        nnz_upper_bound = total_q * topk
        qhead_per_kv = int(qhead_per_kv)
        if qhead_per_kv <= 0:
            raise ValueError(f"qhead_per_kv must be positive, got {qhead_per_kv}")
        if return_schedule:
            if max_seqlen_q is None:
                raise ValueError(
                    "build_k2q_csr requires max_seqlen_q when return_schedule=True"
                )
            max_seqlen_q = int(max_seqlen_q)

        # ---- Output tensors ------------------------------------------------
        device = q2k_indices.device
        k2q_row_ptr = torch.empty(
            (head_kv, total_rows + 1),
            dtype=torch.int32,
            device=device,
        )
        k2q_q_indices = torch.empty(
            (head_kv, nnz_upper_bound),
            dtype=torch.int32,
            device=device,
        )
        schedule = None
        if return_schedule:
            target_q_per_cta = SPARSE_SCHEDULE_MODEL.balanced_target_q_per_cta(
                total_q=total_q,
                topk=topk,
                blk_kv=blk_kv,
                head_kv=head_kv,
                qhead_per_kv=qhead_per_kv,
                device=device,
            )
            scheduler_metadata_capacity = SPARSE_SCHEDULE_MODEL.flat_schedule_capacity(
                total_rows=total_rows,
                total_q=total_q,
                topk=topk,
                head_kv=head_kv,
                target_q_per_cta=target_q_per_cta,
            )
            scheduler_metadata = torch.empty(
                (scheduler_metadata_capacity, 6), dtype=torch.int32, device=device
            )
            work_count = torch.empty((1,), dtype=torch.int32, device=device)
            qsplit_indices = torch.empty_like(k2q_q_indices)
            split_counts = torch.empty(
                (total_q, head_kv), dtype=torch.int32, device=device
            )
            schedule = SparseAttentionSchedule(
                enabled=True,
                scheduler_metadata=scheduler_metadata,
                work_count=work_count,
                qsplit_indices=qsplit_indices,
                split_counts=split_counts,
                target_q_per_cta=target_q_per_cta,
            )

        # Empty workload short-circuit (the CUDA path also handles this,
        # but doing it here saves a JIT load for trivial calls).
        if total_rows == 0 or total_q == 0 or head_kv == 0 or topk == 0:
            k2q_row_ptr.zero_()
            k2q_q_indices.fill_(-1)
            if schedule is not None:
                schedule.work_count.zero_()
                schedule.split_counts.zero_()
                return k2q_row_ptr, k2q_q_indices, schedule
            return k2q_row_ptr, k2q_q_indices

        self._ensure_loaded()
        with torch.cuda.nvtx.range("SparseK2qCsr_Pipeline"):
            if schedule is None:
                self._run(
                    q2k_indices,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    k2q_row_ptr,
                    k2q_q_indices,
                    topk,
                    blk_kv,
                    total_rows,
                    max_kv_blocks,
                )
            else:
                self._run_with_schedule(
                    q2k_indices,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    k2q_row_ptr,
                    k2q_q_indices,
                    schedule.scheduler_metadata,
                    schedule.work_count,
                    schedule.qsplit_indices,
                    schedule.split_counts,
                    topk,
                    blk_kv,
                    total_rows,
                    max_kv_blocks,
                    schedule.target_q_per_cta,
                    schedule.work_capacity,
                    max_seqlen_q,
                )
        if schedule is not None:
            return k2q_row_ptr, k2q_q_indices, schedule
        return k2q_row_ptr, k2q_q_indices
