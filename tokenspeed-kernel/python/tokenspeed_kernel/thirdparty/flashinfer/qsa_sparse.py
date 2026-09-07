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

"""Stateful FlashInfer FA2 block-sparse runner for QSA attention."""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass

import torch

_DEFAULT_WORKSPACE_BYTES = 128 * 1024 * 1024
_BITS_PER_BYTE = 8
_MAX_PLANS = 256


@dataclass(frozen=True)
class _PlanKey:
    rows: int
    width: int
    cache_slots: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    softmax_scale: float


@dataclass
class _FlashInferQSAPlan:
    """Static FA2 schedule plus mutable GPU slot and mask views."""

    wrapper: object
    indices: torch.Tensor
    packed_mask: torch.Tensor
    captured: bool = False


class _FlashInferQSASparseRunner:
    """Run FlashInfer FA2 from one high-watermark metadata allocation.

    TokenSpeed executes QSA as an eager break between CUDA Graph segments, so
    all graph batch sizes on one device can reuse the same mutable plan and
    metadata storage. Plans retain only their compact FA2 schedule (normally
    kilobytes), not another rows-by-width slot buffer or FlashInfer's default
    8 MiB integer workspace. Directly captured standalone calls pin those
    compact schedules in the same hard-bounded cache.
    """

    def __init__(
        self,
        device: torch.device,
        *,
        workspace_bytes: int = _DEFAULT_WORKSPACE_BYTES,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("FlashInfer QSA sparse attention requires a CUDA device")
        from flashinfer.sparse import BlockSparseAttentionWrapper

        self.device = device
        self._wrapper_type = BlockSparseAttentionWrapper
        self.workspace = torch.empty(
            (int(workspace_bytes),), dtype=torch.uint8, device=device
        )
        self._planner = None
        self._plans: OrderedDict[_PlanKey, _FlashInferQSAPlan] = OrderedDict()
        self._row_capacity = 0
        self._buffer_width = 0
        self._indices: torch.Tensor | None = None
        self._initial_mask: torch.Tensor | None = None
        self._packed_mask: torch.Tensor | None = None
        self._indptr: torch.Tensor | None = None
        self._qo_indptr: torch.Tensor | None = None
        self._last_page_len: torch.Tensor | None = None
        self._mask_indptr: torch.Tensor | None = None
        self._scalar_scales: dict[tuple[int, float], torch.Tensor] = {}

    @staticmethod
    def _compact_workspace_bytes(wrapper: object, rows: int) -> int:
        """Return the used prefix of FlashInfer 0.6.x's FA2 int workspace."""

        info = [int(value) for value in wrapper._plan_info]
        if len(info) != 15:
            raise RuntimeError(
                "unsupported FlashInfer FA2 plan-info layout: "
                f"expected 15 values, got {len(info)}"
            )
        padded_batch, total_rows = info[0], info[1]
        request_offset, qo_tile_offset, kv_tile_offset = info[4:7]
        merge_offset, output_offset, chunk_size_offset = info[7:10]
        valid_mask_offset, split_kv = info[12], bool(info[14])
        ends = [
            request_offset + 4 * padded_batch,
            qo_tile_offset + 4 * padded_batch,
            kv_tile_offset + 4 * padded_batch,
            output_offset + 4 * (rows + 1),
            chunk_size_offset + 4,
        ]
        if split_kv:
            ends.extend(
                [
                    merge_offset + 4 * (total_rows + 1),
                    valid_mask_offset + padded_batch,
                ]
            )
        return ((max(ends) + 15) // 16) * 16

    def _metadata_views(self, rows: int) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        assert self._indices is not None
        assert self._packed_mask is not None
        assert self._indptr is not None
        assert self._qo_indptr is not None
        assert self._last_page_len is not None
        assert self._mask_indptr is not None
        return (
            self._indices[:rows],
            self._packed_mask[:rows].reshape(-1),
            self._indptr[: rows + 1],
            self._qo_indptr[: rows + 1],
            self._last_page_len[:rows],
            self._mask_indptr[: rows + 1],
        )

    def _bind_wrapper_metadata(
        self, key: _PlanKey, wrapper: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices, packed_mask, indptr, qo_indptr, last_page_len, mask_indptr = (
            self._metadata_views(key.rows)
        )
        wrapper._qo_indptr = qo_indptr
        wrapper._paged_kv_indptr_buf = indptr
        wrapper._paged_kv_indices_buf = indices.reshape(-1)
        wrapper._paged_kv_last_page_len = last_page_len
        wrapper._packed_mask_buf = packed_mask
        wrapper._mask_indptr_buf = mask_indptr
        return indices, packed_mask

    def _bind_plan_metadata(self, key: _PlanKey, plan: _FlashInferQSAPlan) -> None:
        indices, packed_mask = self._bind_wrapper_metadata(key, plan.wrapper)
        plan.indices = indices
        plan.packed_mask = packed_mask

    def _ensure_metadata_capacity(self, rows: int, width: int) -> None:
        if (
            self._indices is not None
            and self._buffer_width == width
            and self._row_capacity >= rows
        ):
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "FlashInfer QSA metadata must be allocated before CUDA Graph capture"
            )
        if any(plan.captured for plan in self._plans.values()):
            raise RuntimeError(
                "FlashInfer QSA metadata capacity cannot grow after direct CUDA "
                "Graph capture; warm with metadata_capacity_rows covering every graph"
            )

        if self._buffer_width not in (0, int(width)):
            self._plans.clear()
        capacity = max(int(rows), self._row_capacity)
        packed_width = (int(width) + _BITS_PER_BYTE - 1) // _BITS_PER_BYTE
        self._indices = torch.zeros(
            (capacity, width), dtype=torch.int32, device=self.device
        )
        self._initial_mask = torch.zeros(
            (capacity, width), dtype=torch.bool, device=self.device
        )
        self._packed_mask = torch.empty(
            (capacity, packed_width), dtype=torch.uint8, device=self.device
        )
        self._indptr = torch.arange(capacity + 1, dtype=torch.int32, device=self.device)
        self._indptr *= int(width)
        self._qo_indptr = torch.arange(
            capacity + 1, dtype=torch.int32, device=self.device
        )
        self._last_page_len = torch.ones(
            capacity, dtype=torch.int32, device=self.device
        )
        self._mask_indptr = torch.arange(
            capacity + 1, dtype=torch.int32, device=self.device
        )
        self._mask_indptr *= packed_width
        self._row_capacity = capacity
        self._buffer_width = int(width)

        for key, plan in self._plans.items():
            self._bind_plan_metadata(key, plan)

    def _active_metadata(
        self, rows: int, width: int, capacity_rows: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure_metadata_capacity(max(rows, capacity_rows), width)
        assert self._indices is not None
        assert self._initial_mask is not None
        indices, packed_mask, indptr, _, _, _ = self._metadata_views(rows)
        # FlashInfer validates indices before the dynamic slot-preparation
        # kernel runs, so erase values left by the preceding geometry.
        indices.zero_()
        initial_mask = self._initial_mask[:rows].view(rows * width, 1, 1)
        return indices, initial_mask, packed_mask, indptr

    def _cache_plan(self, key: _PlanKey, plan: _FlashInferQSAPlan) -> None:
        self._plans[key] = plan
        self._plans.move_to_end(key)
        while len(self._plans) > _MAX_PLANS:
            evicted = next(
                (
                    candidate
                    for candidate, cached in self._plans.items()
                    if not cached.captured
                ),
                None,
            )
            if evicted is None:
                raise RuntimeError(
                    f"all {_MAX_PLANS} FlashInfer QSA plans are CUDA Graph pinned"
                )
            del self._plans[evicted]

    def plan(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        selected_width: int,
        *,
        softmax_scale: float,
        metadata_capacity_rows: int | None,
    ) -> _FlashInferQSAPlan:
        """Return a cached FA2 plan for the supplied QSA tensor geometry."""

        rows, num_q_heads, head_dim = q.shape
        cache_slots, num_kv_heads, cache_head_dim = k_cache.shape
        if cache_head_dim != head_dim:
            raise ValueError("QSA query and key head dimensions must match")
        if k_cache.shape[:2] != v_cache.shape[:2]:
            raise ValueError("QSA key/value cache geometry must match")
        if v_cache.shape[-1] != head_dim:
            raise ValueError("FlashInfer FA2 requires equal Q/K/V head dimensions")
        if k_cache.dtype != v_cache.dtype:
            raise TypeError("QSA key and value caches must share one dtype")
        if q.device != k_cache.device or q.device != v_cache.device:
            raise ValueError("QSA query and caches must share one device")
        if num_q_heads % num_kv_heads:
            raise ValueError("QSA query heads must be divisible by KV heads")

        key = _PlanKey(
            rows=int(rows),
            width=int(selected_width),
            cache_slots=int(cache_slots),
            num_q_heads=int(num_q_heads),
            num_kv_heads=int(num_kv_heads),
            head_dim=int(head_dim),
            q_dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            softmax_scale=float(softmax_scale),
        )
        capacity_rows = max(int(rows), int(metadata_capacity_rows or rows))
        capturing = torch.cuda.is_current_stream_capturing()
        cached = self._plans.get(key)
        if cached is not None:
            self._ensure_metadata_capacity(capacity_rows, int(selected_width))
            if capturing:
                cached.captured = True
            self._plans.move_to_end(key)
            return cached
        if capturing:
            raise RuntimeError(
                "FlashInfer QSA plan geometry must be warmed before CUDA Graph capture"
            )

        indices, initial_mask, packed_mask, indptr = self._active_metadata(
            int(rows), int(selected_width), capacity_rows
        )
        if self._planner is None:
            self._planner = self._wrapper_type(self.workspace, backend="fa2")
        planner = self._planner
        planner.plan(
            indptr,
            indices.view(-1),
            int(rows),
            int(cache_slots),
            1,
            1,
            int(num_q_heads),
            int(num_kv_heads),
            int(head_dim),
            mask=initial_mask,
            causal=False,
            pos_encoding_mode="NONE",
            sm_scale=float(softmax_scale),
            q_data_type=q.dtype,
            kv_data_type=k_cache.dtype,
            o_data_type=q.dtype,
        )
        # FlashInfer 0.6.x converts custom-mask offsets from bits to bytes only
        # while planning from a bool mask. Plan from bool to obtain those byte
        # offsets, then replace the generated mask with the shared arena below.
        if planner._backend != "fa2":
            raise RuntimeError(
                f"FlashInfer QSA requested FA2 but planned {planner._backend!r}"
            )
        planned_packed_mask = planner._packed_mask_buf
        expected_mask_bytes = rows * (
            (selected_width + _BITS_PER_BYTE - 1) // _BITS_PER_BYTE
        )
        if (
            planned_packed_mask is None
            or planned_packed_mask.numel() != expected_mask_bytes
        ):
            raise RuntimeError("FlashInfer produced an invalid QSA packed-mask buffer")
        wrapper = copy.copy(planner)
        compact_bytes = self._compact_workspace_bytes(planner, int(rows))
        wrapper._int_workspace_buffer = planner._int_workspace_buffer[
            :compact_bytes
        ].clone()
        # The page-locked mirror participates in planning only. Keeping it on
        # each cached wrapper would recreate the original 8 MiB-per-shape leak.
        wrapper._pin_memory_int_workspace_buffer = torch.empty(
            0, dtype=torch.uint8, device="cpu"
        )
        cached = _FlashInferQSAPlan(
            wrapper=wrapper,
            indices=indices,
            packed_mask=packed_mask,
        )
        self._bind_plan_metadata(key, cached)
        # Drop the planner's shape-specific tensor references; cached wrappers
        # retain only their compact schedule and views of the shared arena.
        self._bind_wrapper_metadata(key, planner)
        self._cache_plan(key, cached)
        return cached

    def _scale_tensor(
        self,
        value: float | torch.Tensor | None,
        num_heads: int,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=self.device, dtype=torch.float32).reshape(-1)
            if tensor.numel() == num_heads:
                return tensor.contiguous()
            if tensor.numel() != 1:
                raise ValueError(
                    "FlashInfer QSA scale must be scalar or have one value per KV head"
                )
            if num_heads == 1:
                return tensor
            return tensor.expand(num_heads).contiguous()

        scalar = float(value)
        key = (int(num_heads), scalar)
        tensor = self._scalar_scales.get(key)
        if tensor is None:
            tensor = torch.full(
                (num_heads,), scalar, dtype=torch.float32, device=self.device
            )
            self._scalar_scales[key] = tensor
        return tensor

    def run(
        self,
        plan: _FlashInferQSAPlan,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        k_scale: float | torch.Tensor | None,
        v_scale: float | torch.Tensor | None,
        enable_pdl: bool,
    ) -> torch.Tensor:
        """Run a prepared FA2 plan and return the attention output."""

        if k_cache.dtype == torch.float8_e4m3fn and (
            k_scale is None or v_scale is None
        ):
            raise ValueError("FP8 FlashInfer QSA requires both K and V scales")
        out = q.new_empty((q.shape[0], q.shape[1], v_cache.shape[-1]))
        return plan.wrapper.run(
            q,
            k_cache,
            v_cache,
            scale_k=self._scale_tensor(k_scale, k_cache.shape[1]),
            scale_v=self._scale_tensor(v_scale, v_cache.shape[1]),
            out=out,
            enable_pdl=enable_pdl,
        )


_runners: dict[tuple[str, int | None], _FlashInferQSASparseRunner] = {}


def get_flashinfer_qsa_sparse_runner(
    device: torch.device | str,
) -> _FlashInferQSASparseRunner:
    """Return the process-local QSA FA2 runner for one CUDA device."""

    normalized = torch.device(device)
    index = normalized.index
    if index is None:
        index = torch.cuda.current_device()
        normalized = torch.device("cuda", index)
    key = (normalized.type, index)
    runner = _runners.get(key)
    if runner is None:
        runner = _FlashInferQSASparseRunner(normalized)
        _runners[key] = runner
    return runner


__all__ = ["get_flashinfer_qsa_sparse_runner"]
