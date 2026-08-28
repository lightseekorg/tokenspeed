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

import math
from dataclasses import dataclass

from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    cache_field_layer_id,
)
from tokenspeed.runtime.pd.cache_protocol import (
    CacheTransferContract,
    validate_cache_peer_layout,
)


class UnsupportedPDLayoutError(ValueError):
    pass


@dataclass(frozen=True)
class CacheTransferFragment:
    """One field-relative row fragment copied for every selected cache page.

    Arena bases, segment page-zero offsets, page bases, and page strides are
    deliberately resolved from the validated source/destination cache layouts
    at execution time. Keeping those peer-local addresses out of the route
    plan prevents the wire fragment from becoming a second, independently
    trusted cache ABI.
    """

    group_id: str
    field_id: str
    src_byte_offset: int
    dst_byte_offset: int
    src_row_stride_bytes: int
    dst_row_stride_bytes: int
    bytes_per_row: int
    rows_per_page: int


MAX_CACHE_TP_SIZE = 1024


@dataclass(frozen=True)
class RankTransferPlan:
    fragments_by_prefill_rank: dict[int, tuple[CacheTransferFragment, ...]]

    @property
    def target_prefill_ranks(self) -> tuple[int, ...]:
        return tuple(self.fragments_by_prefill_rank)


@dataclass(frozen=True)
class _Interval:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start

    def intersect(self, other: "_Interval") -> "_Interval | None":
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        if start >= end:
            return None
        return _Interval(start, end)


@dataclass(frozen=True)
class _RankPartition:
    interval: _Interval
    local_offset: int


class CacheTransferPlanner:
    """Plan model-neutral dense cache fields across unequal TP sizes."""

    def __init__(
        self,
        *,
        prefill_tp_size: int,
        decode_tp_size: int,
        prefill_layout: CacheTransferContract,
        decode_layout: CacheTransferContract,
        prefill_layer_window: tuple[int, int] | None = None,
    ):
        """Plan fragments between one Prefill rank set and one Decode rank set.

        Args:
            prefill_layer_window: With prefill chunk-pipeline parallelism, the
                ``[start, end)`` global layer window whose KV THIS planning
                context transfers. Fields owned by other pipeline stages are
                excluded from the route (each stage runs its own planner over
                its own window, and the union of stages covers the plan).
                None plans every field (no PP).
        """
        if prefill_tp_size <= 0 or decode_tp_size <= 0:
            raise UnsupportedPDLayoutError("Cache TP sizes must be positive")
        if prefill_tp_size > MAX_CACHE_TP_SIZE or decode_tp_size > MAX_CACHE_TP_SIZE:
            raise UnsupportedPDLayoutError(
                f"Cache TP sizes cannot exceed {MAX_CACHE_TP_SIZE}"
            )
        self.prefill_tp_size = prefill_tp_size
        self.decode_tp_size = decode_tp_size
        self._layer_window = prefill_layer_window
        validate_cache_peer_layout(prefill_layout, decode_layout)

        def in_window(field_id: str) -> bool:
            if prefill_layer_window is None:
                return True
            layer_id = cache_field_layer_id(field_id)
            return prefill_layer_window[0] <= layer_id < prefill_layer_window[1]

        self._partitions = {
            field.field_id: prefill_layout.transfer_schema.partition_for(field.field_id)
            for field in prefill_layout.plan.fields
        }
        self._segment_pairs = tuple(
            (prefill_spec.group_id, prefill_segment, decode_segment)
            for prefill_spec, decode_spec in zip(
                prefill_layout.group_specs,
                decode_layout.group_specs,
                strict=True,
            )
            for prefill_segment, decode_segment in zip(
                prefill_layout.fields_for_group(prefill_spec.group_id),
                decode_layout.fields_for_group(decode_spec.group_id),
                strict=True,
            )
            if in_window(prefill_segment.field_id)
        )
        for _, prefill_segment, decode_segment in self._segment_pairs:
            self._validate_tp_mapping(prefill_segment, decode_segment)
        self._decode_ranks_by_prefill_rank = self._calc_source_decode_ranks()

    @property
    def decode_ranks_by_prefill_rank(self) -> dict[int, frozenset[int]]:
        """Decode ranks served by each Prefill rank."""
        return dict(self._decode_ranks_by_prefill_rank)

    def plan_for_decode_rank(self, decode_tp_rank: int) -> RankTransferPlan:
        if not 0 <= decode_tp_rank < self.decode_tp_size:
            raise UnsupportedPDLayoutError(
                f"decode_tp_rank={decode_tp_rank} is out of range"
            )
        # Equal-TP fast path: empty fragments mean "copy every field whole".
        # A PP layer window cannot use it — only the window's fields may be
        # copied, so the explicit fragment route is mandatory.
        if self.prefill_tp_size == self.decode_tp_size and self._layer_window is None:
            return RankTransferPlan(
                fragments_by_prefill_rank={decode_tp_rank: ()},
            )

        fragments_by_rank = self._fragments_for_decode_rank(decode_tp_rank)
        target_ranks = tuple(fragments_by_rank)
        if not target_ranks:
            raise UnsupportedPDLayoutError(
                f"Cache-transfer decode TP rank {decode_tp_rank} has no source fragments"
            )
        return RankTransferPlan(
            fragments_by_prefill_rank=fragments_by_rank,
        )

    def _validate_tp_mapping(self, prefill_segment, decode_segment) -> None:
        field = prefill_segment.field_id
        if self.prefill_tp_size == self.decode_tp_size and (
            prefill_segment.shape != decode_segment.shape
            or prefill_segment.payload_bytes != decode_segment.payload_bytes
        ):
            raise UnsupportedPDLayoutError(
                f"equal-TP cache field {field!r} rank-local geometry differs"
            )
        partition = self._partitions[prefill_segment.field_id]
        if partition is None:
            return
        self._rank_partitions(prefill_segment, partition, self.prefill_tp_size, 0)
        self._rank_partitions(decode_segment, partition, self.decode_tp_size, 0)

    def _fragments_for_decode_rank(
        self, decode_tp_rank: int
    ) -> dict[int, tuple[CacheTransferFragment, ...]]:
        fragments: dict[int, list[CacheTransferFragment]] = {}
        for group_id, prefill_segment, decode_segment in self._segment_pairs:
            partition = self._partitions[prefill_segment.field_id]
            if partition is None:
                prefill_rank = self._replicated_source_tp_rank(
                    self.prefill_tp_size,
                    self.decode_tp_size,
                    decode_tp_rank,
                )
                fragment = self._make_fragment(
                    group_id=group_id,
                    prefill_segment=prefill_segment,
                    decode_segment=decode_segment,
                    partition=None,
                    intersection=None,
                    prefill_interval=None,
                    decode_interval=None,
                )
                fragments.setdefault(prefill_rank, []).append(fragment)
                continue

            decode_partitions = self._rank_partitions(
                decode_segment, partition, self.decode_tp_size, decode_tp_rank
            )
            for prefill_rank in range(self.prefill_tp_size):
                if not self._is_representative_rank(
                    prefill_segment,
                    partition,
                    self.prefill_tp_size,
                    prefill_rank,
                ):
                    continue
                prefill_partitions = self._rank_partitions(
                    prefill_segment,
                    partition,
                    self.prefill_tp_size,
                    prefill_rank,
                )
                for prefill_partition, decode_partition in zip(
                    prefill_partitions, decode_partitions, strict=True
                ):
                    intersection = prefill_partition.interval.intersect(
                        decode_partition.interval
                    )
                    if intersection is None:
                        continue
                    fragment = self._make_fragment(
                        group_id=group_id,
                        prefill_segment=prefill_segment,
                        decode_segment=decode_segment,
                        partition=partition,
                        intersection=intersection,
                        prefill_interval=prefill_partition.interval,
                        decode_interval=decode_partition.interval,
                        prefill_local_offset=prefill_partition.local_offset,
                        decode_local_offset=decode_partition.local_offset,
                    )
                    fragments.setdefault(prefill_rank, []).append(fragment)
        return {
            rank: tuple(rank_fragments)
            for rank, rank_fragments in sorted(fragments.items())
        }

    @staticmethod
    def _make_fragment(
        *,
        group_id,
        prefill_segment,
        decode_segment,
        partition,
        intersection,
        prefill_interval,
        decode_interval,
        prefill_local_offset=0,
        decode_local_offset=0,
    ) -> CacheTransferFragment:
        if partition is None:
            rows_per_page = 1
            src_row_stride = prefill_segment.payload_bytes
            dst_row_stride = decode_segment.payload_bytes
            bytes_per_row = prefill_segment.payload_bytes
            src_byte_offset = 0
            dst_byte_offset = 0
        else:
            axis = partition.axis
            inner_bytes = (
                math.prod(prefill_segment.shape[axis + 1 :])
                * prefill_segment.element_size
            )
            rows_per_page = math.prod(prefill_segment.shape[:axis])
            src_row_stride = prefill_segment.shape[axis] * inner_bytes
            dst_row_stride = decode_segment.shape[axis] * inner_bytes
            bytes_per_row = intersection.length * inner_bytes
            src_byte_offset = (
                prefill_local_offset + intersection.start - prefill_interval.start
            ) * inner_bytes
            dst_byte_offset = (
                decode_local_offset + intersection.start - decode_interval.start
            ) * inner_bytes

        if (
            rows_per_page > 1
            and src_row_stride == bytes_per_row
            and dst_row_stride == bytes_per_row
        ):
            bytes_per_row *= rows_per_page
            src_row_stride = bytes_per_row
            dst_row_stride = bytes_per_row
            rows_per_page = 1

        return CacheTransferFragment(
            group_id=group_id,
            field_id=prefill_segment.field_id,
            src_byte_offset=src_byte_offset,
            dst_byte_offset=dst_byte_offset,
            src_row_stride_bytes=src_row_stride,
            dst_row_stride_bytes=dst_row_stride,
            bytes_per_row=bytes_per_row,
            rows_per_page=rows_per_page,
        )

    @staticmethod
    def _rank_partitions(
        segment, partition, tp_size: int, tp_rank: int
    ) -> tuple[_RankPartition, ...]:
        axis = partition.axis
        local_extent = segment.shape[axis]
        global_extent = partition.global_extent
        distinct_shards = global_extent // local_extent
        if distinct_shards > tp_size or tp_size % distinct_shards:
            raise UnsupportedPDLayoutError(
                f"Cache field {segment.field_id!r} cannot map global "
                f"extent {global_extent} and local extent {local_extent} to TP={tp_size}"
            )
        replica_group_size = tp_size // distinct_shards
        shard_rank = tp_rank // replica_group_size
        global_parts = partition.global_parts or (global_extent,)
        partitions = []
        global_offset = 0
        local_offset = 0
        for global_part_extent in global_parts:
            local_part_extent = global_part_extent // distinct_shards
            start = global_offset + shard_rank * local_part_extent
            partitions.append(
                _RankPartition(
                    interval=_Interval(start, start + local_part_extent),
                    local_offset=local_offset,
                )
            )
            global_offset += global_part_extent
            local_offset += local_part_extent
        return tuple(partitions)

    @staticmethod
    def _is_representative_rank(segment, partition, tp_size: int, tp_rank: int) -> bool:
        local_extent = segment.shape[partition.axis]
        distinct_shards = partition.global_extent // local_extent
        replica_group_size = tp_size // distinct_shards
        return tp_rank % replica_group_size == 0

    @staticmethod
    def _replicated_source_tp_rank(
        prefill_tp_size: int, decode_tp_size: int, decode_tp_rank: int
    ) -> int:
        return (decode_tp_rank * prefill_tp_size) // decode_tp_size

    def _calc_source_decode_ranks(self) -> dict[int, frozenset[int]]:
        if self.prefill_tp_size == self.decode_tp_size:
            return {rank: frozenset({rank}) for rank in range(self.prefill_tp_size)}
        decode_ranks = {rank: set() for rank in range(self.prefill_tp_size)}
        for decode_tp_rank in range(self.decode_tp_size):
            for prefill_rank in self._fragments_for_decode_rank(decode_tp_rank):
                decode_ranks[prefill_rank].add(decode_tp_rank)
        return {rank: frozenset(ranks) for rank, ranks in decode_ranks.items()}
