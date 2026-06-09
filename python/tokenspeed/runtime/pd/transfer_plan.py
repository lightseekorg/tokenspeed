from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Literal


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
    buffer_index: int
    buffer_kind: BufferKind
    logical_axis: Literal["kv_head", "state_channel", "replicated"]
    logical_size: int
    page_size: int
    bytes_per_logical_unit: int
    item_stride_bytes: int

    def __post_init__(self):
        if self.logical_size <= 0:
            raise UnsupportedPDLayoutError("logical_size must be positive")
        if self.page_size <= 0:
            raise UnsupportedPDLayoutError("page_size must be positive")
        if self.bytes_per_logical_unit <= 0:
            raise UnsupportedPDLayoutError("bytes_per_logical_unit must be positive")
        if self.item_stride_bytes <= 0:
            raise UnsupportedPDLayoutError("item_stride_bytes must be positive")


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


TRANSFER_PLAN_PROTOCOL_VERSION = 1


def encode_transfer_fragments(
    fragments: tuple[TransferFragment, ...],
) -> tuple[bytes, bytes]:
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
    return (
        str(TRANSFER_PLAN_PROTOCOL_VERSION).encode("ascii"),
        json.dumps(payload, separators=(",", ":")).encode("utf-8"),
    )


def decode_transfer_fragments(
    version_frame: bytes | None,
    payload_frame: bytes | None,
) -> tuple[TransferFragment, ...]:
    if not version_frame and not payload_frame:
        return ()
    if not version_frame or not payload_frame:
        raise UnsupportedPDLayoutError("incomplete transfer plan frames")

    try:
        version = int(version_frame.decode("ascii"))
    except ValueError as exc:
        raise UnsupportedPDLayoutError(
            "invalid transfer plan protocol version"
        ) from exc
    if version != TRANSFER_PLAN_PROTOCOL_VERSION:
        raise UnsupportedPDLayoutError(
            f"unsupported transfer plan protocol version={version}"
        )

    raw_fragments = json.loads(payload_frame.decode("utf-8"))
    return tuple(
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
                None if fragment["page_count"] is None else int(fragment["page_count"])
            ),
        )
        for fragment in raw_fragments
    )


@dataclass(frozen=True)
class RankTransferPlan:
    plan_kind: Literal["identity", "fragmented"]
    target_dp_group: int
    target_prefill_ranks: tuple[int, ...]
    required_prefill_response_num: int
    fragments_by_prefill_rank: dict[int, tuple[TransferFragment, ...]]
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

        if self.prefill_layout.tp_size_per_dp == decode_tp_size:
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

            decode_interval = self._rank_interval(
                decode_buffer.logical_size, decode_tp_size, decode_tp_rank
            )
            for prefill_tp_rank in range(self.prefill_layout.tp_size_per_dp):
                prefill_rank = (
                    target_dp_group * self.prefill_layout.tp_size_per_dp
                    + prefill_tp_rank
                )
                prefill_interval = self._rank_interval(
                    prefill_buffer.logical_size,
                    self.prefill_layout.tp_size_per_dp,
                    prefill_tp_rank,
                )
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
                if buffer.logical_size % layout.tp_size_per_dp != 0:
                    raise UnsupportedPDLayoutError(
                        "non-aligned TP heterogeneous mapping for "
                        f"buffer_kind={buffer.buffer_kind.value}: logical_size="
                        f"{buffer.logical_size}, tp_size_per_dp={layout.tp_size_per_dp}"
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

                decode_interval = self._rank_interval(
                    decode_buffer.logical_size,
                    self.decode_layout.tp_size_per_dp,
                    decode_tp_rank,
                )
                for prefill_tp_rank in range(self.prefill_layout.tp_size_per_dp):
                    prefill_interval = self._rank_interval(
                        prefill_buffer.logical_size,
                        self.prefill_layout.tp_size_per_dp,
                        prefill_tp_rank,
                    )
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

    @staticmethod
    def _rank_interval(logical_size: int, tp_size: int, tp_rank: int) -> _Interval:
        local_size = logical_size // tp_size
        start = tp_rank * local_size
        return _Interval(start, start + local_size)

    @staticmethod
    def _replicated_source_tp_rank(
        prefill_tp_size: int, decode_tp_size: int, decode_tp_rank: int
    ) -> int:
        return (decode_tp_rank * prefill_tp_size) // decode_tp_size
