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

import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from tokenspeed.runtime.pd.cache_protocol import (
    CacheTransferContract,
    validate_cache_peer_layout,
)


class UnsupportedPDLayoutError(ValueError):
    pass


class BufferKind(str, Enum):
    TARGET_K = "target_k"
    TARGET_V = "target_v"
    DRAFT_K = "draft_k"
    DRAFT_V = "draft_v"
    MAMBA_STATE = "mamba_state"


@dataclass(frozen=True)
class ParallelLayout:
    role: Literal["prefill", "decode"]
    world_size: int
    dp_size: int = 1

    def __post_init__(self):
        if self.world_size <= 0:
            raise UnsupportedPDLayoutError("world_size must be positive")
        if self.dp_size <= 0:
            raise UnsupportedPDLayoutError("dp_size must be positive")
        if self.world_size % self.dp_size != 0:
            raise UnsupportedPDLayoutError(
                f"world_size={self.world_size} must be divisible by dp_size={self.dp_size}"
            )

    @property
    def tp_size_per_dp(self) -> int:
        return self.world_size // self.dp_size


@dataclass(frozen=True)
class BufferLayout:
    """Logical layout for one cache/state buffer.

    ``tp_replica_group_size`` describes TP ranks that hold the same logical
    shard. It is used by GQA/MQA-style KV caches when the prefill TP size is
    larger than the number of distinct KV heads.
    """

    buffer_index: int
    buffer_kind: BufferKind
    logical_axis: Literal["kv_head", "state_channel", "replicated"]
    logical_size: int
    page_size: int
    bytes_per_logical_unit: int
    item_stride_bytes: int
    tp_replica_group_size: int = 1

    def __post_init__(self):
        if self.logical_size <= 0:
            raise UnsupportedPDLayoutError("logical_size must be positive")
        if self.page_size <= 0:
            raise UnsupportedPDLayoutError("page_size must be positive")
        if self.bytes_per_logical_unit <= 0:
            raise UnsupportedPDLayoutError("bytes_per_logical_unit must be positive")
        if self.item_stride_bytes <= 0:
            raise UnsupportedPDLayoutError("item_stride_bytes must be positive")
        if self.tp_replica_group_size <= 0:
            raise UnsupportedPDLayoutError("tp_replica_group_size must be positive")


@dataclass(frozen=True)
class TransferFragment:
    buffer_index: int
    buffer_kind: BufferKind
    src_rank: int
    dst_rank: int
    src_page_stride_bytes: int
    dst_page_stride_bytes: int
    src_byte_offset: int
    dst_byte_offset: int
    bytes_per_page: int
    page_count: int | None = None


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
    src_rank: int
    dst_rank: int
    src_byte_offset: int
    dst_byte_offset: int
    src_row_stride_bytes: int
    dst_row_stride_bytes: int
    bytes_per_row: int
    rows_per_page: int


TransferPlanFragment = TransferFragment | CacheTransferFragment


TRANSFER_PLAN_PROTOCOL_VERSION = 1
PAGED_TRANSFER_PLAN_PROTOCOL_VERSION = 2
MAX_PAGED_CACHE_TP_SIZE = 1024
_MAX_TRANSFER_PLAN_WIRE_BYTES = 2 << 20
_MAX_TRANSFER_FRAGMENTS = 1 << 16


def encode_transfer_fragments(
    fragments: tuple[TransferPlanFragment, ...],
) -> tuple[bytes, bytes]:
    if len(fragments) > _MAX_TRANSFER_FRAGMENTS:
        raise UnsupportedPDLayoutError("transfer plan contains too many fragments")
    if all(isinstance(fragment, TransferFragment) for fragment in fragments):
        payload = [
            {
                "buffer_index": fragment.buffer_index,
                "buffer_kind": fragment.buffer_kind.value,
                "src_rank": fragment.src_rank,
                "dst_rank": fragment.dst_rank,
                "src_page_stride_bytes": fragment.src_page_stride_bytes,
                "dst_page_stride_bytes": fragment.dst_page_stride_bytes,
                "src_byte_offset": fragment.src_byte_offset,
                "dst_byte_offset": fragment.dst_byte_offset,
                "bytes_per_page": fragment.bytes_per_page,
                "page_count": fragment.page_count,
            }
            for fragment in fragments
        ]
        version = TRANSFER_PLAN_PROTOCOL_VERSION
    elif all(isinstance(fragment, CacheTransferFragment) for fragment in fragments):
        payload = [
            {name: getattr(fragment, name) for name in fragment.__dataclass_fields__}
            for fragment in fragments
        ]
        version = PAGED_TRANSFER_PLAN_PROTOCOL_VERSION
    else:
        raise UnsupportedPDLayoutError(
            "legacy and Paged cache transfer fragments cannot be mixed"
        )
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if len(encoded) > _MAX_TRANSFER_PLAN_WIRE_BYTES:
        raise UnsupportedPDLayoutError("transfer plan exceeds the wire-size limit")
    return str(version).encode("ascii"), encoded


def decode_transfer_fragments(
    version_frame: bytes | None,
    payload_frame: bytes | None,
) -> tuple[TransferPlanFragment, ...]:
    if not version_frame and not payload_frame:
        return ()
    if not version_frame or not payload_frame:
        raise UnsupportedPDLayoutError("incomplete transfer plan frames")
    if len(version_frame) > 16:
        raise UnsupportedPDLayoutError("invalid transfer plan protocol version")

    try:
        version = int(version_frame.decode("ascii"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise UnsupportedPDLayoutError(
            "invalid transfer plan protocol version"
        ) from exc
    if version not in (
        TRANSFER_PLAN_PROTOCOL_VERSION,
        PAGED_TRANSFER_PLAN_PROTOCOL_VERSION,
    ):
        raise UnsupportedPDLayoutError(
            f"unsupported transfer plan protocol version={version}"
        )
    if version == PAGED_TRANSFER_PLAN_PROTOCOL_VERSION and version_frame != str(
        PAGED_TRANSFER_PLAN_PROTOCOL_VERSION
    ).encode("ascii"):
        raise UnsupportedPDLayoutError(
            "Paged cache transfer plan version must use canonical ASCII encoding"
        )
    if len(payload_frame) > _MAX_TRANSFER_PLAN_WIRE_BYTES:
        raise UnsupportedPDLayoutError("transfer plan exceeds the wire-size limit")
    try:
        raw_fragments = json.loads(payload_frame.decode("utf-8"))
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as exc:
        raise UnsupportedPDLayoutError("invalid transfer plan JSON") from exc
    if (
        not isinstance(raw_fragments, list)
        or len(raw_fragments) > _MAX_TRANSFER_FRAGMENTS
    ):
        raise UnsupportedPDLayoutError("transfer plan must be a bounded JSON list")
    fragments = []
    for fragment in raw_fragments:
        if not isinstance(fragment, dict):
            raise UnsupportedPDLayoutError("transfer fragment must be a JSON object")
        if version == TRANSFER_PLAN_PROTOCOL_VERSION:
            # Preserve the legacy v1 decoder's coercive schema during the
            # staged migration. Strict exact-schema decoding applies only to
            # the new Paged-cache v2 plan.
            try:
                fragments.append(
                    TransferFragment(
                        buffer_index=int(fragment["buffer_index"]),
                        buffer_kind=BufferKind(fragment["buffer_kind"]),
                        src_rank=int(fragment["src_rank"]),
                        dst_rank=int(fragment["dst_rank"]),
                        src_page_stride_bytes=int(fragment["src_page_stride_bytes"]),
                        dst_page_stride_bytes=int(fragment["dst_page_stride_bytes"]),
                        src_byte_offset=int(fragment["src_byte_offset"]),
                        dst_byte_offset=int(fragment["dst_byte_offset"]),
                        bytes_per_page=int(fragment["bytes_per_page"]),
                        page_count=(
                            None
                            if fragment["page_count"] is None
                            else int(fragment["page_count"])
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise UnsupportedPDLayoutError(
                    "invalid legacy transfer fragment"
                ) from exc
        else:
            expected_keys = set(CacheTransferFragment.__dataclass_fields__)
            if set(fragment) != expected_keys:
                raise UnsupportedPDLayoutError(
                    "Paged cache transfer fragment has unexpected fields"
                )
            if (
                not isinstance(fragment["group_id"], str)
                or not isinstance(fragment["field_id"], str)
                or any(
                    isinstance(fragment[name], bool)
                    or not isinstance(fragment[name], int)
                    for name in expected_keys - {"group_id", "field_id"}
                )
            ):
                raise UnsupportedPDLayoutError(
                    "Paged cache transfer fragment has invalid field types"
                )
            fragments.append(
                CacheTransferFragment(
                    **fragment,
                )
            )
    result = tuple(fragments)
    if (
        version == PAGED_TRANSFER_PLAN_PROTOCOL_VERSION
        and encode_transfer_fragments(result)[1] != payload_frame
    ):
        raise UnsupportedPDLayoutError(
            "Paged cache transfer plan must use canonical JSON encoding"
        )
    return result


@dataclass(frozen=True)
class RankTransferPlan:
    plan_kind: Literal["identity", "fragmented"]
    target_dp_group: int
    target_prefill_ranks: tuple[int, ...]
    required_prefill_response_num: int
    fragments_by_prefill_rank: dict[int, tuple[TransferPlanFragment, ...]]
    required_dst_info_num_by_prefill_rank: dict[int, int]

    def required_dst_info_num_for_prefill_rank(self, prefill_rank: int) -> int:
        return self.required_dst_info_num_by_prefill_rank[prefill_rank]


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


class PDTransferPlanner:
    def __init__(
        self,
        *,
        prefill_layout: ParallelLayout,
        decode_layout: ParallelLayout,
        prefill_buffers: tuple[BufferLayout, ...],
        decode_buffers: tuple[BufferLayout, ...],
    ):
        self.prefill_layout = prefill_layout
        self.decode_layout = decode_layout
        self.prefill_buffers = prefill_buffers
        self.decode_buffers = decode_buffers
        self._validate_buffers()
        self._validate_alignment()
        self._required_dst_info_num_by_prefill_rank = self._calc_source_fanout()

    def plan_for_decode_rank(self, decode_rank: int) -> RankTransferPlan:
        decode_tp_size = self.decode_layout.tp_size_per_dp
        if decode_rank < 0 or decode_rank >= self.decode_layout.world_size:
            raise UnsupportedPDLayoutError(f"decode_rank={decode_rank} is out of range")

        target_dp_group = decode_rank // decode_tp_size
        decode_tp_rank = decode_rank % decode_tp_size

        if self._can_use_identity_plan() and (
            self.prefill_layout.tp_size_per_dp == decode_tp_size
        ):
            prefill_rank = (
                target_dp_group * self.prefill_layout.tp_size_per_dp + decode_tp_rank
            )
            return RankTransferPlan(
                plan_kind="identity",
                target_dp_group=target_dp_group,
                target_prefill_ranks=(prefill_rank,),
                required_prefill_response_num=1,
                fragments_by_prefill_rank={},
                required_dst_info_num_by_prefill_rank={
                    prefill_rank: self._required_dst_info_num_by_prefill_rank[
                        prefill_rank
                    ]
                },
            )

        fragments: dict[int, list[TransferFragment]] = {}
        for prefill_buffer, decode_buffer in zip(
            self.prefill_buffers, self.decode_buffers
        ):
            if prefill_buffer.logical_axis == "replicated":
                prefill_tp_rank = self._replicated_source_tp_rank(
                    self.prefill_layout.tp_size_per_dp,
                    decode_tp_size,
                    decode_tp_rank,
                )
                prefill_rank = (
                    target_dp_group * self.prefill_layout.tp_size_per_dp
                    + prefill_tp_rank
                )
                fragment = TransferFragment(
                    buffer_index=prefill_buffer.buffer_index,
                    buffer_kind=prefill_buffer.buffer_kind,
                    src_rank=prefill_rank,
                    dst_rank=decode_rank,
                    src_page_stride_bytes=prefill_buffer.item_stride_bytes,
                    dst_page_stride_bytes=decode_buffer.item_stride_bytes,
                    src_byte_offset=0,
                    dst_byte_offset=0,
                    bytes_per_page=decode_buffer.item_stride_bytes,
                )
                fragments.setdefault(prefill_rank, []).append(fragment)
                continue

            decode_interval = self._rank_interval_for_buffer(
                decode_buffer,
                self.decode_layout,
                decode_tp_rank,
            )
            if decode_interval is None:
                continue
            for prefill_tp_rank in range(self.prefill_layout.tp_size_per_dp):
                prefill_rank = (
                    target_dp_group * self.prefill_layout.tp_size_per_dp
                    + prefill_tp_rank
                )
                prefill_interval = self._rank_interval_for_buffer(
                    prefill_buffer,
                    self.prefill_layout,
                    prefill_tp_rank,
                )
                if prefill_interval is None:
                    continue
                intersection = prefill_interval.intersect(decode_interval)
                if intersection is None:
                    continue

                fragment = TransferFragment(
                    buffer_index=prefill_buffer.buffer_index,
                    buffer_kind=prefill_buffer.buffer_kind,
                    src_rank=prefill_rank,
                    dst_rank=decode_rank,
                    src_page_stride_bytes=prefill_buffer.item_stride_bytes,
                    dst_page_stride_bytes=decode_buffer.item_stride_bytes,
                    src_byte_offset=(intersection.start - prefill_interval.start)
                    * prefill_buffer.bytes_per_logical_unit,
                    dst_byte_offset=(intersection.start - decode_interval.start)
                    * decode_buffer.bytes_per_logical_unit,
                    bytes_per_page=intersection.length
                    * prefill_buffer.bytes_per_logical_unit,
                )
                fragments.setdefault(prefill_rank, []).append(fragment)

        fragments_by_rank = {
            rank: tuple(rank_fragments)
            for rank, rank_fragments in sorted(fragments.items())
        }
        target_prefill_ranks = tuple(fragments_by_rank)
        return RankTransferPlan(
            plan_kind="fragmented",
            target_dp_group=target_dp_group,
            target_prefill_ranks=target_prefill_ranks,
            required_prefill_response_num=len(target_prefill_ranks),
            fragments_by_prefill_rank=fragments_by_rank,
            required_dst_info_num_by_prefill_rank={
                rank: self._required_dst_info_num_by_prefill_rank[rank]
                for rank in target_prefill_ranks
            },
        )

    def _validate_buffers(self) -> None:
        if len(self.prefill_buffers) != len(self.decode_buffers):
            raise UnsupportedPDLayoutError("prefill/decode buffer counts differ")
        for prefill_buffer, decode_buffer in zip(
            self.prefill_buffers, self.decode_buffers
        ):
            if prefill_buffer.buffer_index != decode_buffer.buffer_index:
                raise UnsupportedPDLayoutError("prefill/decode buffer indexes differ")
            if prefill_buffer.buffer_kind != decode_buffer.buffer_kind:
                raise UnsupportedPDLayoutError("prefill/decode buffer kinds differ")
            if prefill_buffer.logical_axis != decode_buffer.logical_axis:
                raise UnsupportedPDLayoutError("prefill/decode logical axes differ")
            if prefill_buffer.logical_size != decode_buffer.logical_size:
                raise UnsupportedPDLayoutError("prefill/decode logical sizes differ")
            if (
                prefill_buffer.bytes_per_logical_unit
                != decode_buffer.bytes_per_logical_unit
            ):
                raise UnsupportedPDLayoutError(
                    "prefill/decode logical unit sizes differ"
                )

    def _validate_alignment(self) -> None:
        for layout, buffers in (
            (self.prefill_layout, self.prefill_buffers),
            (self.decode_layout, self.decode_buffers),
        ):
            for buffer in buffers:
                if buffer.logical_axis == "replicated":
                    continue
                if layout.tp_size_per_dp % buffer.tp_replica_group_size != 0:
                    raise UnsupportedPDLayoutError(
                        "tp replica group must divide TP size for "
                        f"buffer_kind={buffer.buffer_kind.value}: "
                        f"tp_size_per_dp={layout.tp_size_per_dp}, "
                        f"tp_replica_group_size={buffer.tp_replica_group_size}"
                    )
                effective_tp_size = (
                    layout.tp_size_per_dp // buffer.tp_replica_group_size
                )
                if buffer.logical_size % effective_tp_size != 0:
                    raise UnsupportedPDLayoutError(
                        "non-aligned TP heterogeneous mapping for "
                        f"buffer_kind={buffer.buffer_kind.value}: logical_size="
                        f"{buffer.logical_size}, effective_tp_size={effective_tp_size}"
                    )
                item_units = buffer.item_stride_bytes // buffer.bytes_per_logical_unit
                required_units = buffer.logical_size // effective_tp_size
                if item_units < required_units:
                    raise UnsupportedPDLayoutError(
                        "buffer item is smaller than its logical shard for "
                        f"buffer_kind={buffer.buffer_kind.value}: item_units="
                        f"{item_units}, required_units={required_units}"
                    )

    def _calc_source_fanout(self) -> dict[int, int]:
        fanout = {rank: 0 for rank in range(self.prefill_layout.world_size)}
        for decode_rank in range(self.decode_layout.world_size):
            decode_tp_rank = decode_rank % self.decode_layout.tp_size_per_dp
            target_dp_group = decode_rank // self.decode_layout.tp_size_per_dp
            intersected_prefill_ranks = set()
            for prefill_buffer, decode_buffer in zip(
                self.prefill_buffers, self.decode_buffers
            ):
                if prefill_buffer.logical_axis == "replicated":
                    prefill_tp_rank = self._replicated_source_tp_rank(
                        self.prefill_layout.tp_size_per_dp,
                        self.decode_layout.tp_size_per_dp,
                        decode_tp_rank,
                    )
                    prefill_rank = (
                        target_dp_group * self.prefill_layout.tp_size_per_dp
                        + prefill_tp_rank
                    )
                    intersected_prefill_ranks.add(prefill_rank)
                    continue

                decode_interval = self._rank_interval_for_buffer(
                    decode_buffer,
                    self.decode_layout,
                    decode_tp_rank,
                )
                if decode_interval is None:
                    continue
                for prefill_tp_rank in range(self.prefill_layout.tp_size_per_dp):
                    prefill_interval = self._rank_interval_for_buffer(
                        prefill_buffer,
                        self.prefill_layout,
                        prefill_tp_rank,
                    )
                    if prefill_interval is None:
                        continue
                    if prefill_interval.intersect(decode_interval) is None:
                        continue
                    prefill_rank = (
                        target_dp_group * self.prefill_layout.tp_size_per_dp
                        + prefill_tp_rank
                    )
                    intersected_prefill_ranks.add(prefill_rank)
            for prefill_rank in intersected_prefill_ranks:
                fanout[prefill_rank] += 1
        return fanout

    def _can_use_identity_plan(self) -> bool:
        return all(
            prefill_buffer.tp_replica_group_size == 1
            and decode_buffer.tp_replica_group_size == 1
            for prefill_buffer, decode_buffer in zip(
                self.prefill_buffers, self.decode_buffers
            )
        )

    @staticmethod
    def _rank_interval(logical_size: int, tp_size: int, tp_rank: int) -> _Interval:
        local_size = logical_size // tp_size
        start = tp_rank * local_size
        return _Interval(start, start + local_size)

    @staticmethod
    def _rank_interval_for_buffer(
        buffer: BufferLayout, layout: ParallelLayout, tp_rank: int
    ) -> _Interval | None:
        replica_group_size = buffer.tp_replica_group_size
        if tp_rank % replica_group_size != 0:
            return None
        effective_tp_size = layout.tp_size_per_dp // replica_group_size
        effective_tp_rank = tp_rank // replica_group_size
        return PDTransferPlanner._rank_interval(
            buffer.logical_size, effective_tp_size, effective_tp_rank
        )

    @staticmethod
    def _replicated_source_tp_rank(
        prefill_tp_size: int, decode_tp_size: int, decode_tp_rank: int
    ) -> int:
        return (decode_tp_rank * prefill_tp_size) // decode_tp_size


class PagedCacheTransferPlanner:
    """Plan model-neutral dense Paged-cache fields across unequal TP sizes."""

    def __init__(
        self,
        *,
        prefill_tp_size: int,
        decode_tp_size: int,
        prefill_layout: CacheTransferContract,
        decode_layout: CacheTransferContract,
    ):
        if prefill_tp_size <= 0 or decode_tp_size <= 0:
            raise UnsupportedPDLayoutError("Paged cache TP sizes must be positive")
        if (
            prefill_tp_size > MAX_PAGED_CACHE_TP_SIZE
            or decode_tp_size > MAX_PAGED_CACHE_TP_SIZE
        ):
            raise UnsupportedPDLayoutError(
                f"Paged cache TP sizes cannot exceed {MAX_PAGED_CACHE_TP_SIZE}"
            )
        self.prefill_tp_size = prefill_tp_size
        self.decode_tp_size = decode_tp_size
        validate_cache_peer_layout(prefill_layout, decode_layout)
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
        if self.prefill_tp_size == self.decode_tp_size:
            return RankTransferPlan(
                plan_kind="identity",
                target_dp_group=0,
                target_prefill_ranks=(decode_tp_rank,),
                required_prefill_response_num=1,
                fragments_by_prefill_rank={decode_tp_rank: ()},
                required_dst_info_num_by_prefill_rank={
                    decode_tp_rank: len(
                        self._decode_ranks_by_prefill_rank[decode_tp_rank]
                    )
                },
            )

        fragments_by_rank = self._fragments_for_decode_rank(decode_tp_rank)
        target_ranks = tuple(fragments_by_rank)
        if not target_ranks:
            raise UnsupportedPDLayoutError(
                f"Paged cache decode TP rank {decode_tp_rank} has no source fragments"
            )
        return RankTransferPlan(
            plan_kind="fragmented",
            target_dp_group=0,
            target_prefill_ranks=target_ranks,
            required_prefill_response_num=len(target_ranks),
            fragments_by_prefill_rank=fragments_by_rank,
            required_dst_info_num_by_prefill_rank={
                rank: len(self._decode_ranks_by_prefill_rank[rank])
                for rank in target_ranks
            },
        )

    def _validate_tp_mapping(self, prefill_segment, decode_segment) -> None:
        field = prefill_segment.field_id
        if self.prefill_tp_size == self.decode_tp_size and (
            prefill_segment.shape != decode_segment.shape
            or prefill_segment.payload_bytes != decode_segment.payload_bytes
        ):
            raise UnsupportedPDLayoutError(
                f"equal-TP Paged cache field {field!r} rank-local geometry differs"
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
                    prefill_rank=prefill_rank,
                    decode_tp_rank=decode_tp_rank,
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
                    prefill_segment, partition, self.prefill_tp_size, prefill_rank
                ):
                    continue
                prefill_partitions = self._rank_partitions(
                    prefill_segment, partition, self.prefill_tp_size, prefill_rank
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
                        prefill_rank=prefill_rank,
                        decode_tp_rank=decode_tp_rank,
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
        prefill_rank,
        decode_tp_rank,
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
            src_rank=prefill_rank,
            dst_rank=decode_tp_rank,
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
                f"Paged cache field {segment.field_id!r} cannot map global "
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
