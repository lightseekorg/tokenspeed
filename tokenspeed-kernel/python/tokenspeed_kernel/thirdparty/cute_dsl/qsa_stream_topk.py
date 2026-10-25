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

"""Blackwell exact QSA score/top-k dispatch and bounded workspaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_stream_topk.prefill import PrefillTopK
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_stream_topk.prefill_dense import (
    DensePrefillTopK,
)
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_stream_topk.prefill_overlap import (
    OverlapPrefillTopK,
)
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_stream_topk.prefill_split import (
    SplitPrefillTopK,
)
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_stream_topk.selector import (
    Merge32,
    PackedMerge,
)
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_stream_topk.small import (
    GuardMerge,
    GuardScore,
    WarpScore,
)

_QUERY_GROUP = 8
_HEADS = 4
_HEAD_DIM = 128
_TOPK = 512
_COMPILED = {}
_WARMED = set()


@dataclass(frozen=True)
class StreamConfig:
    algorithm: str
    splits: int
    keep: int
    compact_ids: bool
    linear_bins: bool
    partition_width: int
    group_layout: Literal["aligned", "uniform", "ragged"]


def _workspace_bytes(rows: int, config: StreamConfig) -> int:
    if config.algorithm == "split":
        return rows * config.splits * _TOPK * 8
    if config.algorithm != "small":
        return 0
    per_split = 64 * 4
    if config.keep:
        per_split += config.keep * (6 if config.compact_ids else 8)
        per_split += (64 // config.partition_width) * 4
    return rows * config.splits * per_split


def _configuration(
    rows: int, capacity: int, queries_per_request: int | None, max_partial_bytes: int
) -> StreamConfig | None:
    # Request widths only establish ownership. Geometry and power-of-two
    # capacity buckets specialize the JIT; live lengths and row counts do not.
    if rows < 1 or not 0 < capacity <= 131072:
        return None
    grouped = (
        queries_per_request is not None
        and queries_per_request >= 8
        and rows % queries_per_request == 0
    )
    group_layout = (
        "aligned"
        if grouped and queries_per_request % _QUERY_GROUP == 0
        else "uniform" if queries_per_request else "ragged"
    )
    if rows <= 128:
        if grouped and capacity > 8192:
            splits = 1 << (((capacity + 4095) // 4096) - 1).bit_length()
            config = StreamConfig("split", splits, 0, False, False, 64, group_layout)
        else:
            splits = 1 << (((capacity + 63) // 64) - 1).bit_length()
            keep = 0
            compact_ids = False
            linear_bins = False
            partition_width = 64
            if capacity == 4096 and 2 <= rows <= 8:
                linear_bins = True
            elif capacity == 8192 and rows <= 8:
                keep = 16
                if rows == 1:
                    linear_bins = True
                    partition_width = 32
            elif capacity > 8192 and rows <= 8:
                keep = 8
                compact_ids = True
            config = StreamConfig(
                "small",
                splits,
                keep,
                compact_ids,
                linear_bins,
                partition_width,
                "aligned",  # Unused by the independent-row small-grid kernels.
            )
        if _workspace_bytes(rows, config) <= max_partial_bytes:
            return config
    if capacity <= 2048:
        algorithm = "prefill"
    elif capacity <= 4096:
        algorithm = "dense"
    else:
        algorithm = "overlap"
    return StreamConfig(algorithm, 1, 0, False, False, 64, group_layout)


def _meta(tensor: torch.Tensor, leading_dim: int) -> cute.Tensor:
    return from_dlpack(tensor, assumed_align=16).mark_layout_dynamic(
        leading_dim=leading_dim
    )


def _compile(
    query,
    keys,
    page_table,
    request_indices,
    complete_blocks,
    selected,
    scratch,
    queries_per_request: int | None,
    config: StreamConfig,
    enable_pdl: bool,
):
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    options = "--enable-tvm-ffi --opt-level 3"
    if config.algorithm == "small":
        if config.keep:
            score_op = GuardScore(
                config.splits,
                64,
                enable_pdl,
                config.keep,
                config.compact_ids,
                config.partition_width,
            )
            merge_op = GuardMerge(
                config.splits,
                64,
                enable_pdl,
                config.keep,
                config.compact_ids,
                config.linear_bins,
                config.partition_width,
            )
        else:
            score_op = WarpScore(config.splits, 64, enable_pdl)
            merge_op = Merge32(config.splits, 64, enable_pdl, config.linear_bins)
        score = cute.compile(
            score_op,
            _meta(query, 2),
            _meta(keys, 2),
            _meta(page_table, 1),
            _meta(request_indices, 0),
            _meta(complete_blocks, 0),
            *(_meta(t, 1) for t in scratch),
            int(query.stride(0)),
            int(query.stride(1)),
            stream,
            options=options,
        )
        merge = cute.compile(
            merge_op,
            *(_meta(t, 1) for t in scratch),
            _meta(complete_blocks, 0),
            _meta(selected, 1),
            int(page_table.shape[1] * 64),
            stream,
            options=options,
        )
        return score, merge
    if config.algorithm == "split":
        op = SplitPrefillTopK(enable_pdl, config.splits, config.group_layout)
        target = scratch[0]
    else:
        if config.algorithm == "prefill":
            op = PrefillTopK(enable_pdl, config.group_layout)
        elif config.algorithm == "dense":
            op = DensePrefillTopK(enable_pdl, 1, config.group_layout)
        else:
            op = OverlapPrefillTopK(enable_pdl, config.group_layout)
        target = selected
    score = cute.compile(
        op,
        _meta(query, 2),
        _meta(keys, 2),
        _meta(page_table, 1),
        _meta(request_indices, 0),
        _meta(complete_blocks, 0),
        _meta(target, 1),
        int(query.stride(0)),
        int(query.stride(1)),
        int(queries_per_request or 0),
        64,
        stream,
        options=options,
    )
    if config.algorithm == "split":
        merge = cute.compile(
            PackedMerge(config.splits, _TOPK, enable_pdl),
            _meta(target, 1),
            _meta(complete_blocks, 0),
            _meta(selected, 1),
            int(page_table.shape[1] * 64),
            stream,
            options=options,
        )
        return score, merge
    return (score,)


def warmup(
    query, keys, request_indices, *, max_capacity, max_partial_bytes, enable_pdl
):
    """Warm algorithm/capacity buckets without reading request-owned cache rows.

    Args:
        query: BF16 query prototype supplying the CUDA device.
        keys: Existing compressed key cache, unchanged by warmup.
        request_indices: Prototype supplying the request-index dtype.
        max_capacity: Maximum compressed-block capacity; capped at 131072.
        max_partial_bytes: Caller budget for selection scratch.
        enable_pdl: Programmatic dependent launch setting to compile.

    Returns:
        None. Compiled configurations remain cached for subsequent calls.
    """
    max_capacity = min(max_capacity, 131072)
    key = (
        query.device.index,
        request_indices.dtype,
        max_capacity,
        max_partial_bytes,
        enable_pdl,
    )
    if key in _WARMED:
        return
    capacities = {max_capacity}
    capacity = 64
    while capacity <= max_capacity:
        capacities.add(capacity)
        capacity *= 2
    configurations = {}
    # Cover _configuration's row policies: singleton, both ends of 2..8,
    # the first row above 8, and the small-grid limit 128. Above that limit,
    # cover both aligned and partial query groups. Try ragged and
    # whole-request ownership for each row count.
    for capacity in sorted(capacities):
        for rows in (1, 2, 8, 9, 128, 135, 136):
            for width in (None, rows):
                config = _configuration(rows, capacity, width, max_partial_bytes)
                if config is not None:
                    configurations.setdefault(config, (rows, capacity, width))
    sample = query.new_zeros((136, _HEADS, _HEAD_DIM))
    table = torch.zeros((1, max_capacity // 64), dtype=torch.int32, device=query.device)
    requests = request_indices.new_zeros(136)
    complete = requests.new_zeros(136, dtype=torch.int32)
    for rows, capacity, width in configurations.values():
        kernel(
            sample[:rows],
            keys,
            table[:, : capacity // 64],
            requests[:rows],
            complete[:rows],
            page_size=64,
            block_topk=_TOPK,
            queries_per_request=width,
            max_partial_bytes=max_partial_bytes,
            enable_pdl=enable_pdl,
        )
    _WARMED.add(key)


def kernel(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    page_table: torch.Tensor,
    request_indices: torch.Tensor,
    complete_blocks: torch.Tensor,
    *,
    page_size: int,
    block_topk: int,
    queries_per_request: int | None,
    max_partial_bytes: int,
    enable_pdl: bool,
) -> torch.Tensor:
    """Select exact top-512 compressed blocks on Blackwell.

    Args:
        query: Aligned BF16 queries shaped ``[rows, 4, 128]``.
        key_cache: Contiguous BF16 keys shaped ``[slots, 1, 128]`` with
            complete 64-slot physical pages.
        page_table: Aligned int32 physical pages shaped ``[requests, pages]``
            with contiguous columns and a runtime row stride; at most 2048
            pages per request, including speculative context padding.
        request_indices: Contiguous int32 or int64 request id per query row.
        complete_blocks: Contiguous int32 valid compressed-block count per row.
            Counts are clamped to the page-table capacity on the device.
        page_size: Compressed blocks per page; must be 64.
        block_topk: Selected blocks per row; must be 512.
        queries_per_request: Uniform consecutive query-run width, or ``None``
            for ragged rows. Groups stop at request boundaries and mask tails.
        max_partial_bytes: Budget for global scratch, excluding the output.
            Grouped streaming needs none; split scoring respects this budget.
        enable_pdl: Explicit programmatic dependent launch ordering.

    Returns:
        Int32 logical block ids shaped ``[rows, 512]``, padded with ``-1``.
        Output order and equal-score block order are unspecified.
    """

    if query.ndim != 3 or key_cache.ndim != 3:
        raise ValueError("CuTe QSA stream kernel expects rank-three Q/K tensors")
    if page_table.ndim != 2:
        raise ValueError("CuTe QSA stream kernel expects a rank-two page table")
    rows = query.shape[0]
    if request_indices.shape != (rows,) or complete_blocks.shape != (rows,):
        raise ValueError("CuTe QSA stream metadata must have one entry per row")
    tensors = (query, key_cache, page_table, request_indices, complete_blocks)
    if not query.is_cuda or any(tensor.device != query.device for tensor in tensors):
        raise ValueError("CuTe QSA stream tensors must share one CUDA device")
    if block_topk != _TOPK or page_size != 64:
        raise ValueError("CuTe QSA stream kernel requires page_size=64 and topk=512")
    if queries_per_request is not None and (
        queries_per_request < 1 or rows % queries_per_request
    ):
        raise ValueError("QSA queries_per_request must be positive and divide rows")
    if query.shape[1:] != (_HEADS, _HEAD_DIM):
        raise ValueError("CuTe QSA stream kernel requires H=4 and D=128")
    if (
        key_cache.shape[0] < 64
        or key_cache.shape[0] % 64
        or key_cache.shape[1:] != (1, _HEAD_DIM)
    ):
        raise ValueError("CuTe QSA stream kernel requires K shape [slots, 1, 128]")
    if key_cache.dtype is not torch.bfloat16 or query.dtype is not torch.bfloat16:
        raise TypeError("CuTe QSA stream kernel requires BF16 Q/K")
    if page_table.dtype is not torch.int32:
        raise TypeError("CuTe QSA stream kernel requires an int32 page table")
    if request_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("CuTe QSA stream kernel requires int32 or int64 request ids")
    if complete_blocks.dtype is not torch.int32:
        raise TypeError("CuTe QSA stream kernel requires int32 complete-block counts")
    if query.stride(2) != 1 or query.stride(0) % 8 or query.stride(1) % 8:
        raise ValueError("CuTe QSA stream query vectors must be 16-byte aligned")
    if query.data_ptr() % 16:
        raise ValueError("CuTe QSA stream query storage must be 16-byte aligned")
    if not key_cache.is_contiguous() or key_cache.data_ptr() % 16:
        raise ValueError("CuTe QSA stream key cache must be contiguous and aligned")
    if (
        page_table.stride(1) != 1
        or page_table.stride(0) < page_table.shape[1]
        or page_table.data_ptr() % 16
    ):
        raise ValueError("CuTe QSA page-table columns must be contiguous and aligned")
    if not all(
        tensor.is_contiguous() and tensor.data_ptr() % 16 == 0
        for tensor in (request_indices, complete_blocks)
    ):
        raise ValueError("CuTe QSA stream metadata must be contiguous and aligned")
    selected = torch.empty((rows, _TOPK), dtype=torch.int32, device=query.device)
    if rows == 0:
        return selected
    capacity = page_table.shape[1] * page_size
    config = _configuration(rows, capacity, queries_per_request, max_partial_bytes)
    if config is None:
        raise ValueError("Unsupported CuTe QSA layout or insufficient workspace budget")
    scratch = []
    if config.algorithm == "split":
        scratch.append(
            torch.empty(
                (rows, config.splits * _TOPK), dtype=torch.int64, device=query.device
            )
        )
    elif config.algorithm == "small":
        scratch.append(
            torch.empty(
                (rows, config.splits * 64), dtype=torch.uint32, device=query.device
            )
        )
        if config.keep:
            dtype = torch.uint32 if config.compact_ids else torch.int64
            candidates = torch.empty(
                (rows, config.splits * config.keep), dtype=dtype, device=query.device
            )
            # Packed candidates already carry ids; their unused id argument
            # aliases the same tensor and does not require another allocation.
            candidate_ids = (
                torch.empty_like(candidates, dtype=torch.uint16)
                if config.compact_ids
                else candidates
            )
            discarded = torch.empty(
                (rows, config.splits * (64 // config.partition_width)),
                dtype=torch.uint32,
                device=query.device,
            )
            scratch.extend((candidates, candidate_ids, discarded))
    key = (query.device.index, request_indices.dtype, enable_pdl, config)
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = _compile(
            query,
            key_cache,
            page_table,
            request_indices,
            complete_blocks,
            selected,
            scratch,
            queries_per_request,
            config,
            enable_pdl,
        )
        _COMPILED[key] = compiled
    target = [selected] if not scratch else scratch
    compiled[0](
        query,
        key_cache,
        page_table,
        request_indices,
        complete_blocks,
        *target,
        int(query.stride(0)),
        int(query.stride(1)),
        *([] if config.algorithm == "small" else [int(queries_per_request or 0)]),
    )
    if len(compiled) > 1:
        compiled[1](*scratch, complete_blocks, selected, int(capacity))
    return selected
