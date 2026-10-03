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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec
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
class CachePageOwnerFilter:
    """Which of a sharded group's scheduler blocks one source rank holds.

    A DCP-sharded group deals virtual blocks cyclically to ``owner_count``
    ranks; the rank with ``owner_rank`` holds block ``v`` when
    ``(v - 1) % owner_count == owner_rank`` (block 0 is the null block). The
    sender keeps only those blocks of a manifest, translated to its local
    pages, and the matching destination subsequence. The same placement the
    runtime's zeroing and device translation use; see
    ``kv_cache/virtual_blocks.py``.
    """

    owner_rank: int
    owner_count: int

    def __post_init__(self) -> None:
        if self.owner_count < 2 or not 0 <= self.owner_rank < self.owner_count:
            raise ValueError("cache page owner filter needs 0 <= rank < count >= 2")


@dataclass(frozen=True)
class RankTransferPlan:
    fragments_by_prefill_rank: dict[int, tuple[CacheTransferFragment, ...]]
    # One decision per (source rank, sharded group): the filter selecting the
    # blocks that rank owns and sends, or None when its route carries no
    # fragment of the group and it sends nothing for it. Every source rank of
    # the route has an entry naming every sharded group of the layout (empty
    # when the layout has none). Replicated groups are never listed: their
    # blocks copy whole, as the rank's fragments say.
    owner_filters_by_prefill_rank: dict[int, dict[str, CachePageOwnerFilter | None]]

    def __post_init__(self) -> None:
        if set(self.owner_filters_by_prefill_rank) != set(
            self.fragments_by_prefill_rank
        ):
            raise ValueError(
                "owner-filter decisions must cover exactly the route's source ranks"
            )

    @property
    def target_prefill_ranks(self) -> tuple[int, ...]:
        return tuple(self.fragments_by_prefill_rank)


def validate_rank_owner_filters(
    *,
    group_specs: Sequence[CacheGroupSpec],
    fragments: tuple[CacheTransferFragment, ...],
    owner_filters: Mapping[str, CachePageOwnerFilter | None],
) -> None:
    """Check one source rank's owner-filter decisions against its cache groups.

    Every sharded group needs a decision and no replicated group may have
    one; a filter must match the group's shard count and accompany fragments
    of that group, while a None decision must not. The sender applies the
    decisions as given, so this runs once when the route is planned.
    """
    shard_counts = {
        spec.group_id: spec.shard_count for spec in group_specs if spec.shard_count != 1
    }
    routed_groups = {fragment.group_id for fragment in fragments}
    for group_id in owner_filters:
        if group_id not in shard_counts:
            raise ValueError(
                f"owner filter names cache group {group_id!r}, which is not sharded"
            )
    for group_id, shard_count in shard_counts.items():
        if group_id not in owner_filters:
            raise ValueError(
                f"cache group {group_id!r} is sharded but the registration route "
                "carries no owner-filter decision for it"
            )
        owner_filter = owner_filters[group_id]
        if owner_filter is None:
            if group_id in routed_groups:
                raise ValueError(
                    f"the registration route carries fragments of sharded cache "
                    f"group {group_id!r} but no owner filter"
                )
            continue
        if owner_filter.owner_count != shard_count:
            raise ValueError(
                f"cache group {group_id!r} owner filter ({owner_filter.owner_count} "
                f"owners) disagrees with its shard count {shard_count}"
            )
        if group_id not in routed_groups:
            raise ValueError(
                f"cache group {group_id!r} has an owner filter but the registration "
                "route carries no fragment of it"
            )


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
    """Plan model-neutral dense cache fields across unequal TP sizes.

    Two source geometries compose here: head partitions split a page's rows
    over TP ranks, and DCP page sharding (``CacheGroupSpec.shard_count``)
    deals whole pages over a consecutive TP subgroup. The destination is
    always unsharded.
    """

    def __init__(
        self,
        *,
        prefill_tp_size: int,
        decode_tp_size: int,
        prefill_layout: CacheTransferContract,
        decode_layout: CacheTransferContract,
        prefill_field_ids: frozenset[str] | None,
    ):
        """Plan fragments between one Prefill rank set and one Decode rank set.

        Args:
            prefill_field_ids: Explicit resident fields to transfer, or None
                for the complete plan. Model/cache setup determines placement.
        """
        if prefill_tp_size <= 0 or decode_tp_size <= 0:
            raise UnsupportedPDLayoutError("Cache TP sizes must be positive")
        if prefill_tp_size > MAX_CACHE_TP_SIZE or decode_tp_size > MAX_CACHE_TP_SIZE:
            raise UnsupportedPDLayoutError(
                f"Cache TP sizes cannot exceed {MAX_CACHE_TP_SIZE}"
            )
        self.prefill_tp_size = prefill_tp_size
        self.decode_tp_size = decode_tp_size
        all_fields = frozenset(field.field_id for field in prefill_layout.plan.fields)
        if prefill_field_ids is not None and not prefill_field_ids <= all_fields:
            raise UnsupportedPDLayoutError(
                "stage placement contains unknown cache fields"
            )
        self._field_ids = None if prefill_field_ids == all_fields else prefill_field_ids
        validate_cache_peer_layout(prefill_layout, decode_layout)

        self._partitions = {
            field.field_id: prefill_layout.transfer_schema.partition_for(field.field_id)
            for field in prefill_layout.plan.fields
        }
        # DCP page sharding on the source: a sharded group's virtual blocks are
        # dealt cyclically over a consecutive subgroup of shard_count Prefill
        # TP ranks, so every rank of the chosen subgroup is a source and sends
        # only the blocks it owns. The destination must hold every block
        # whole; landing a block on its Decode owner only has no receive path.
        self._shard_counts: dict[str, int] = {}
        for prefill_spec, decode_spec in zip(
            prefill_layout.group_specs, decode_layout.group_specs, strict=True
        ):
            if decode_spec.shard_count != 1:
                raise UnsupportedPDLayoutError(
                    f"cache group {decode_spec.group_id!r} is sharded on Decode; "
                    "PD transfer into a DCP-sharded destination is not supported"
                )
            if prefill_spec.shard_count == 1:
                continue
            if prefill_tp_size % prefill_spec.shard_count:
                raise UnsupportedPDLayoutError(
                    f"cache group {prefill_spec.group_id!r} shard count "
                    f"{prefill_spec.shard_count} does not divide Prefill "
                    f"TP={prefill_tp_size}"
                )
            self._shard_counts[prefill_spec.group_id] = prefill_spec.shard_count
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
            if prefill_field_ids is None
            or prefill_segment.field_id in prefill_field_ids
        )
        for group_id, prefill_segment, decode_segment in self._segment_pairs:
            self._validate_tp_mapping(prefill_segment, decode_segment)
            if (
                group_id in self._shard_counts
                and self._partitions[prefill_segment.field_id] is not None
            ):
                # Head partitions place a page's rows on distinct TP ranks and
                # page sharding places whole pages on distinct ranks; a field
                # under both would need rows no single rank holds.
                raise UnsupportedPDLayoutError(
                    f"cache field {prefill_segment.field_id!r} is both "
                    "head-partitioned and page-sharded on Prefill"
                )
        self._decode_ranks_by_prefill_rank = self._calc_source_decode_ranks()

    @property
    def decode_ranks_by_prefill_rank(self) -> dict[int, frozenset[int]]:
        """Decode ranks served by each Prefill rank."""
        return dict(self._decode_ranks_by_prefill_rank)

    @property
    def has_sharded_groups(self) -> bool:
        return bool(self._shard_counts)

    @property
    def sharded_group_ids(self) -> tuple[str, ...]:
        return tuple(self._shard_counts)

    @property
    def _uses_whole_copy_route(self) -> bool:
        """Equal TP, every field resident, no sharded group.

        Each Decode rank then reads its same-index Prefill rank whole, which
        empty fragments express. A stage owning only a subset of the fields
        needs its explicit fragment route, and so does a sharded source: its
        blocks come from a rank set, not from the one same-index rank.
        """
        return (
            self.prefill_tp_size == self.decode_tp_size
            and self._field_ids is None
            and not self.has_sharded_groups
        )

    def plan_for_decode_rank(self, decode_tp_rank: int) -> RankTransferPlan:
        if not 0 <= decode_tp_rank < self.decode_tp_size:
            raise UnsupportedPDLayoutError(
                f"decode_tp_rank={decode_tp_rank} is out of range"
            )
        if self._uses_whole_copy_route:
            fragments_by_rank = {decode_tp_rank: ()}
        else:
            fragments_by_rank = self._fragments_for_decode_rank(decode_tp_rank)
            if not fragments_by_rank:
                raise UnsupportedPDLayoutError(
                    f"Cache-transfer decode TP rank {decode_tp_rank} has no source "
                    "fragments"
                )
        return RankTransferPlan(
            fragments_by_prefill_rank=fragments_by_rank,
            owner_filters_by_prefill_rank={
                rank: self.owner_filters_for(rank, fragments)
                for rank, fragments in fragments_by_rank.items()
            },
        )

    def owner_filters_for(
        self, prefill_rank: int, fragments: tuple[CacheTransferFragment, ...]
    ) -> dict[str, CachePageOwnerFilter | None]:
        """Decide, for every sharded group, which blocks one source rank sends.

        A rank whose fragments name the group owns the blocks of its DCP
        subgroup position (``prefill_rank % shard_count``, subgroups being
        aligned runs of ``shard_count`` consecutive TP ranks); a rank whose
        fragments do not name the group sends nothing for it, recorded as
        None so the sender never has to infer that from an absent entry.
        """
        routed_groups = {fragment.group_id for fragment in fragments}
        return {
            group_id: (
                CachePageOwnerFilter(prefill_rank % shard_count, shard_count)
                if group_id in routed_groups
                else None
            )
            for group_id, shard_count in self._shard_counts.items()
        }

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
                fragment = self._make_fragment(
                    group_id=group_id,
                    prefill_segment=prefill_segment,
                    decode_segment=decode_segment,
                    partition=None,
                    intersection=None,
                    prefill_interval=None,
                    decode_interval=None,
                )
                replica_rank = self._replicated_source_tp_rank(
                    self.prefill_tp_size,
                    self.decode_tp_size,
                    decode_tp_rank,
                )
                shard_count = self._shard_counts.get(group_id, 1)
                if shard_count == 1:
                    fragments.setdefault(replica_rank, []).append(fragment)
                    continue
                # The replica this decode rank would read whole is spread over
                # its DCP subgroup (consecutive TP ranks); every member sends
                # the blocks it owns, which owner_filters_for records.
                subgroup_base = replica_rank - replica_rank % shard_count
                for prefill_rank in range(subgroup_base, subgroup_base + shard_count):
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
        if self._uses_whole_copy_route:
            return {rank: frozenset({rank}) for rank in range(self.prefill_tp_size)}
        decode_ranks = {rank: set() for rank in range(self.prefill_tp_size)}
        for decode_tp_rank in range(self.decode_tp_size):
            for prefill_rank in self._fragments_for_decode_rank(decode_tp_rank):
                decode_ranks[prefill_rank].add(decode_tp_rank)
        return {rank: frozenset(ranks) for rank, ranks in decode_ranks.items()}


def build_pipeline_transfer_plan(
    *,
    prefill_tp_size: int,
    decode_tp_size: int,
    decode_tp_rank: int,
    prefill_layout: CacheTransferContract,
    decode_layout: CacheTransferContract,
    cache_fields_by_stage: tuple[tuple[str, ...], ...],
) -> tuple[RankTransferPlan, tuple[int, ...]]:
    """Plan every Prefill stage's resident fields for one Decode TP rank.

    Args:
        prefill_tp_size: Attention TP width inside one Prefill stage.
        decode_tp_size: Attention TP width inside one Decode replica.
        decode_tp_rank: Receiving TP coordinate inside that replica.
        prefill_layout: Complete logical source cache contract.
        decode_layout: Complete destination cache contract.
        cache_fields_by_stage: Explicit resident field IDs for every Prefill stage.

    Returns:
        The stage-major source-rank plan and ranks joining completion with no data.
    """
    validate_cache_stage_fields(prefill_layout, cache_fields_by_stage)
    fragments: dict[int, tuple[CacheTransferFragment, ...]] = {}
    owner_filters: dict[int, dict[str, CachePageOwnerFilter | None]] = {}
    dummy_ranks: list[int] = []
    for stage, field_ids in enumerate(cache_fields_by_stage):
        planner = CacheTransferPlanner(
            prefill_tp_size=prefill_tp_size,
            decode_tp_size=decode_tp_size,
            prefill_layout=prefill_layout,
            decode_layout=decode_layout,
            prefill_field_ids=frozenset(field_ids),
        )
        stage_plan = planner.plan_for_decode_rank(decode_tp_rank)
        base = stage * prefill_tp_size
        for rank, stage_fragments in stage_plan.fragments_by_prefill_rank.items():
            fragments[base + rank] = stage_fragments
        for rank, filters in stage_plan.owner_filters_by_prefill_rank.items():
            owner_filters[base + rank] = filters
        if decode_tp_rank == 0:
            dummy_ranks.extend(
                base + rank
                for rank, decode_ranks in planner.decode_ranks_by_prefill_rank.items()
                if not decode_ranks
            )
    return RankTransferPlan(
        fragments_by_prefill_rank=fragments,
        owner_filters_by_prefill_rank=owner_filters,
    ), tuple(sorted(dummy_ranks))


def validate_cache_stage_fields(
    layout: CacheTransferContract, cache_fields_by_stage: tuple[tuple[str, ...], ...]
) -> None:
    """Require a nonempty stage list covering each logical field exactly once."""
    fields = [field for stage in cache_fields_by_stage for field in stage]
    if (
        not cache_fields_by_stage
        or any(not isinstance(field, str) for field in fields)
        or len(fields) != len(set(fields))
        or set(fields) != {field.field_id for field in layout.plan.fields}
    ):
        raise ValueError(
            "cache stage placement must cover every logical field exactly once"
        )
