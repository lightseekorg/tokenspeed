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

"""Vendor-neutral byte transfer boundary for compact Host cache storage."""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import torch
from tokenspeed_kernel.ops.kvcache.triton import (
    HOST_CACHE_TRANSFER_CHUNK_BYTES,
    STAGED_TRANSFER_BYTE_CHUNK_BYTES,
    STAGED_TRANSFER_WORD_CHUNK_BYTES,
)
from tokenspeed_kernel.ops.kvcache.triton import (
    stage_cache_blocks as _stage_cache_blocks_triton,
)
from tokenspeed_kernel.ops.kvcache.triton import (
    transfer_cache_blocks as _transfer_cache_blocks_triton,
)
from tokenspeed_kernel.ops.kvcache.triton import (
    wait_layer_ready as _wait_layer_ready_triton,
)
from tokenspeed_kernel.platform import current_platform

# Default per-stream HBM staging capacity for Ascend NPU staged SDMA transfers.
# Sized to saturate PCIe/SDMA while staying O(1) with respect to context length.
DEFAULT_NPU_STAGING_BYTES = int(
    os.environ.get("TOKENSPEED_NPU_HOST_STAGING_BYTES", str(32 * 1024 * 1024))
)


class StagingBuffer:
    """Reusable contiguous uint8 Device staging buffer shared by one stream.

    The runtime can supply an ``allocator`` callback so the staging tensor lives
    in a memory-saver discard region (for colocated RL sleep/wake) while
    ``tokenspeed-kernel`` stays independent of runtime lifecycle adapters.
    """

    def __init__(
        self,
        *,
        allocator: Callable[[int, torch.device], torch.Tensor] | None = None,
        default_bytes: int = DEFAULT_NPU_STAGING_BYTES,
    ) -> None:
        if default_bytes <= 0:
            raise ValueError("default_bytes must be positive")
        self._allocator: Callable[[int, torch.device], torch.Tensor] | None = allocator
        self._default_bytes: int = int(default_bytes)
        self._buffer: torch.Tensor | None = None

    def acquire(self, min_bytes: int, device: torch.device) -> torch.Tensor:
        """Return a contiguous uint8 tensor on ``device`` of at least ``min_bytes``."""
        if min_bytes <= 0:
            raise ValueError("min_bytes must be positive")
        needed = max(self._default_bytes, int(min_bytes))
        if (
            self._buffer is None
            or self._buffer.device != device
            or self._buffer.numel() < needed
        ):
            if self._allocator is not None:
                buffer = self._allocator(needed, device)
            else:
                buffer = torch.empty(needed, dtype=torch.uint8, device=device)
            if (
                buffer.device != device
                or buffer.dtype != torch.uint8
                or not buffer.is_contiguous()
                or buffer.numel() < needed
            ):
                raise ValueError(
                    "staging allocator must return a contiguous uint8 tensor on "
                    f"{device} with at least {needed} bytes"
                )
            self._buffer = buffer
        return self._buffer


@dataclass(frozen=True, slots=True)
class HostTransferMode:
    """Resolved transport and explicitly selected completion protocol.

    Attributes:
        backend: Transport used for this buffer binding: ``"triton"`` for
            mapped-Host GPU kernels, ``"staged_dma"`` for NPU HBM-staged
            Triton D2D gather/scatter + batched SDMA, or ``"dma"`` for direct
            DMA range copies.
        layer_ready: Whether H2D publishes per-layer ready flags. Currently
            supported only by NVIDIA mapped-Host Triton transfers; other
            backends use completion events instead.
    """

    backend: Literal["triton", "staged_dma", "dma"]
    layer_ready: bool

    def __post_init__(self) -> None:
        if self.backend != "triton" and self.layer_ready:
            raise ValueError(
                f"{self.backend} does not currently support layer_ready=True"
            )

    @property
    def uses_device_tables(self) -> bool:
        return self.backend in ("triton", "staged_dma")


@dataclass(frozen=True, slots=True)
class BlockTransferBatch:
    """Bucketed columnar block pairs ``(device_groups, host_groups)`` by cache group.

    Each entry in ``device_groups`` and ``host_groups`` is a 1-D ``int64`` NumPy
    array of 1-based local block IDs for that group. Implements ``__len__``,
    ``__bool__``, and ``__iter__`` so existing inspection/mock consumers continue
    to see ``(group_index, device_block_id, host_block_id)`` triples lazily while
    workspace staging copies directly without Python tuple boxing.
    """

    device_groups: tuple[np.ndarray, ...]
    host_groups: tuple[np.ndarray, ...]
    group_counts: tuple[int, ...]
    total_blocks: int

    @classmethod
    def from_group_pairs(
        cls,
        device_groups: Sequence[np.ndarray],
        host_groups: Sequence[np.ndarray],
    ) -> BlockTransferBatch:
        if len(device_groups) != len(host_groups):
            raise ValueError("device_groups and host_groups must cover the same groups")
        norm_device: list[np.ndarray] = []
        norm_host: list[np.ndarray] = []
        counts: list[int] = []
        total = 0
        for device_ids, host_ids in zip(device_groups, host_groups):
            d_arr = np.asarray(device_ids, dtype=np.int64)
            h_arr = np.asarray(host_ids, dtype=np.int64)
            if d_arr.ndim != 1 or h_arr.ndim != 1 or d_arr.shape != h_arr.shape:
                raise ValueError(
                    "per-group block IDs must be matching 1-D int64 arrays"
                )
            count = int(d_arr.size)
            norm_device.append(d_arr)
            norm_host.append(h_arr)
            counts.append(count)
            total += count
        return cls(
            device_groups=tuple(norm_device),
            host_groups=tuple(norm_host),
            group_counts=tuple(counts),
            total_blocks=total,
        )

    @property
    def num_groups(self) -> int:
        return len(self.device_groups)

    def __len__(self) -> int:
        return self.total_blocks

    def __bool__(self) -> bool:
        return self.total_blocks > 0

    def __iter__(self):
        for group_index, (device_ids, host_ids) in enumerate(
            zip(self.device_groups, self.host_groups)
        ):
            if device_ids.size == 0:
                continue
            for d_id, h_id in zip(device_ids.tolist(), host_ids.tolist()):
                yield (group_index, d_id, h_id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BlockTransferBatch):
            return self.group_counts == other.group_counts and all(
                np.array_equal(d_a, d_b) and np.array_equal(h_a, h_b)
                for d_a, h_a, d_b, h_b in zip(
                    self.device_groups,
                    self.host_groups,
                    other.device_groups,
                    other.host_groups,
                )
            )
        if isinstance(other, (list, tuple)):
            if len(other) != self.total_blocks:
                return False
            return list(self) == list(other)
        return NotImplemented


class _MappedHostUnavailable(RuntimeError):
    pass


def layer_ready_ptx_supported() -> bool:
    """Layer-ready flags use NVIDIA PTX acquire/release; AMD must not compile them."""

    return bool(current_platform().is_nvidia)


_GEOMETRY_ROW_WIDTH = 8
GeometryRow = tuple[int, int, int, int, int, int, int, int]
LayerSlice = tuple[int, int]


def _has_pinned_host_allocator() -> bool:
    return bool(
        torch.cuda.is_available()
        or (hasattr(torch, "npu") and torch.npu.is_available())
    )


def _pinned_host_int64(shape: tuple[int, ...]) -> torch.Tensor:
    if _has_pinned_host_allocator():
        return torch.empty(shape, dtype=torch.int64, pin_memory=True)
    return torch.empty(shape, dtype=torch.int64, device="cpu")


@dataclass(frozen=True, slots=True)
class _StagedGroupSlice:
    """Merged contiguous Host spans for one cache group in a geometry slice."""

    group_index: int
    span_base: int
    staged_block_bytes: int
    spans: tuple[tuple[int, int], ...]
    max_word_chunks: int
    max_byte_chunks: int
    word_aligned_geometry: bool
    buffer_indices: tuple[int, ...]
    packing: int
    host_block_bytes: int


def _merge_host_spans(
    intervals: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    ordered = sorted(intervals)
    merged: list[tuple[int, int]] = []
    cur_start, cur_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= cur_end:
            if end > cur_end:
                cur_end = end
        else:
            merged.append((cur_start, cur_end - cur_start))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end - cur_start))
    return tuple(merged)


@dataclass(frozen=True, slots=True)
class HostTransferGeometry:
    """Immutable field geometry for one executor lifetime.

    Device row and layer-slice tables must both be absent or both reside on
    the same device. Binding publishes the two tables together.
    """

    host_rows: torch.Tensor
    device_rows: torch.Tensor | None
    layer_slices: tuple[LayerSlice, ...]
    group_packing: tuple[int, ...]
    host_group_packing: tuple[int, ...]
    host_lcm_block_bytes: int
    num_host_lcm_blocks: int
    num_device_lcm_blocks: int
    row_work: tuple[tuple[int, int], ...]
    device_layer_slices: torch.Tensor | None
    _staged_slice_cache: dict[tuple[int, int], tuple[_StagedGroupSlice, ...]] = field(
        default_factory=dict,
        repr=False,
        compare=False,
        hash=False,
    )

    def staged_group_slices(
        self, geometry_offset: int, num_geometry_rows: int
    ) -> tuple[_StagedGroupSlice, ...]:
        """Return cached per-group merged Host spans for a geometry slice."""
        key = (int(geometry_offset), int(num_geometry_rows))
        cached = self._staged_slice_cache.get(key)
        if cached is not None:
            return cached
        if num_geometry_rows <= 0:
            self._staged_slice_cache[key] = ()
            return ()
        rows = self.host_rows[
            geometry_offset : geometry_offset + num_geometry_rows
        ].tolist()
        grouped: dict[int, list[tuple[int, int, int, int, int, int, int]]] = {}
        for row in rows:
            group_index = int(row[0])
            device_buffer_index = int(row[1])
            device_zero = int(row[2])
            device_stride = int(row[3])
            host_block_bytes = int(row[4])
            host_field_offset = int(row[5])
            packing = int(row[6])
            payload_bytes = int(row[7])
            grouped.setdefault(group_index, []).append(
                (
                    device_buffer_index,
                    device_zero,
                    device_stride,
                    host_field_offset,
                    payload_bytes,
                    packing,
                    host_block_bytes,
                )
            )
        slices: list[_StagedGroupSlice] = []
        for group_index, entries in grouped.items():
            spans = _merge_host_spans(
                [
                    (field_off, field_off + payload)
                    for _, _, _, field_off, payload, _, _ in entries
                ]
            )
            span_base = spans[0][0]
            span_end = spans[-1][0] + spans[-1][1]
            staged_block_bytes = span_end - span_base
            max_word_chunks = max(
                (payload + STAGED_TRANSFER_WORD_CHUNK_BYTES - 1)
                // STAGED_TRANSFER_WORD_CHUNK_BYTES
                for _, _, _, _, payload, _, _ in entries
            )
            max_byte_chunks = max(
                (payload + STAGED_TRANSFER_BYTE_CHUNK_BYTES - 1)
                // STAGED_TRANSFER_BYTE_CHUNK_BYTES
                for _, _, _, _, payload, _, _ in entries
            )
            align_bits = staged_block_bytes
            buffer_indices_seen: dict[int, None] = {}
            for (
                buf_idx,
                dev_zero,
                dev_stride,
                field_off,
                payload,
                _,
                _,
            ) in entries:
                buffer_indices_seen[buf_idx] = None
                align_bits |= dev_zero | dev_stride | (field_off - span_base) | payload
            packing = entries[0][5]
            host_block_bytes = entries[0][6]
            slices.append(
                _StagedGroupSlice(
                    group_index=group_index,
                    span_base=span_base,
                    staged_block_bytes=staged_block_bytes,
                    spans=spans,
                    max_word_chunks=max_word_chunks,
                    max_byte_chunks=max_byte_chunks,
                    word_aligned_geometry=(align_bits & 3) == 0,
                    buffer_indices=tuple(buffer_indices_seen),
                    packing=packing,
                    host_block_bytes=host_block_bytes,
                )
            )
        result = tuple(slices)
        self._staged_slice_cache[key] = result
        return result

    def __post_init__(self) -> None:
        if (self.device_rows is None) != (self.device_layer_slices is None):
            raise ValueError("geometry device tables must be bound together")
        if (
            self.device_rows is not None
            and self.device_layer_slices is not None
            and self.device_rows.device != self.device_layer_slices.device
        ):
            raise ValueError("geometry device tables must reside on the same device")

    @property
    def num_groups(self) -> int:
        return len(self.group_packing)

    @property
    def num_field_rows(self) -> int:
        return int(self.host_rows.shape[0])

    def max_device_block_id(self, group_index: int) -> int:
        return self.num_device_lcm_blocks * self.group_packing[group_index]

    def max_host_block_id(self, group_index: int) -> int:
        return self.num_host_lcm_blocks * self.host_group_packing[group_index]

    def bind(
        self,
        device: torch.device,
        *,
        non_blocking: bool,
    ) -> HostTransferGeometry:
        if self.device_rows is not None:
            if self.device_rows.device != device:
                raise ValueError("geometry is already bound to another device")
            return self
        device_rows = torch.empty_like(self.host_rows, device=device)
        device_rows.copy_(self.host_rows, non_blocking=non_blocking)
        layer_slice_host = torch.tensor(
            self.layer_slices, dtype=torch.int64, device="cpu"
        )
        device_layer_slices = torch.empty(
            layer_slice_host.shape, dtype=torch.int64, device=device
        )
        device_layer_slices.copy_(layer_slice_host, non_blocking=non_blocking)
        return replace(
            self,
            device_rows=device_rows,
            device_layer_slices=device_layer_slices,
        )


def _validate_geometry_row(
    row: Sequence[int],
    *,
    row_index: int,
    group_packing: Sequence[int],
    num_device_buffers: int,
    host_lcm_block_bytes: int,
) -> None:
    if len(row) != _GEOMETRY_ROW_WIDTH:
        raise ValueError(
            f"geometry row {row_index} must have {_GEOMETRY_ROW_WIDTH} columns, "
            f"got {len(row)}"
        )
    (
        group_index,
        device_buffer_index,
        device_block_zero_offset_bytes,
        device_block_stride_bytes,
        host_cache_block_bytes,
        host_field_offset_bytes,
        row_packing,
        payload_bytes,
    ) = (int(value) for value in row)
    if not 0 <= group_index < len(group_packing):
        raise IndexError(
            f"geometry row {row_index} group {group_index} outside "
            f"[0, {len(group_packing)})"
        )
    if not 0 <= device_buffer_index < num_device_buffers:
        raise IndexError(
            f"geometry row {row_index} device_buffer_index {device_buffer_index} "
            f"outside [0, {num_device_buffers})"
        )
    if device_block_zero_offset_bytes < 0:
        raise ValueError(
            f"geometry row {row_index} device_block_zero_offset_bytes must be "
            "non-negative"
        )
    if device_block_stride_bytes <= 0:
        raise ValueError(
            f"geometry row {row_index} device_block_stride_bytes must be positive"
        )
    if host_cache_block_bytes <= 0:
        raise ValueError(
            f"geometry row {row_index} host_cache_block_bytes must be positive"
        )
    if host_field_offset_bytes < 0:
        raise ValueError(
            f"geometry row {row_index} host_field_offset_bytes must be non-negative"
        )
    expected_packing = int(group_packing[group_index])
    if row_packing != expected_packing:
        raise ValueError(
            f"geometry row {row_index} cache_blocks_per_lcm_block {row_packing} "
            f"!= group packing {expected_packing}"
        )
    if payload_bytes <= 0:
        raise ValueError(f"geometry row {row_index} payload_bytes must be positive")
    if payload_bytes > device_block_stride_bytes:
        raise ValueError(
            f"geometry row {row_index} payload_bytes cannot exceed "
            "device_block_stride_bytes"
        )
    if host_field_offset_bytes + payload_bytes > host_cache_block_bytes:
        raise ValueError(
            f"geometry row {row_index} field payload lies outside host cache block"
        )
    if row_packing * host_cache_block_bytes > host_lcm_block_bytes:
        raise ValueError(
            f"geometry row {row_index} packed group lies outside host LCM block"
        )


def build_host_transfer_geometry(
    *,
    rows: Sequence[GeometryRow],
    layer_slices: Sequence[LayerSlice],
    group_packing: Sequence[int],
    host_group_packing: Sequence[int],
    host_lcm_block_bytes: int,
    num_host_lcm_blocks: int,
    num_device_lcm_blocks: int,
    num_device_buffers: int,
) -> HostTransferGeometry:
    """Validate and build Host-side geometry shared by transfer operations.

    Args:
        rows: Consumer-ordered field tuples: ``(group_index,
            device_buffer_index, device_block_zero_offset_bytes,
            device_block_stride_bytes, host_cache_block_bytes,
            host_field_offset_bytes, cache_blocks_per_lcm_block,
            payload_bytes)``. Group and buffer indices are zero-based;
            dynamic cache block IDs are one-based. Device block zero is
            reserved, whereas Host blocks are packed without a reserved slot.
        layer_slices: Contiguous ``(row_offset, row_count)`` slices covering
            all field rows in consumer order. Empty consumers have zero rows.
        group_packing: Device cache-block counts per LCM block, by group.
        host_group_packing: Compact Host cache-block counts per LCM block,
            in the same group order. Host packing can differ from Device.
        host_lcm_block_bytes: Byte stride of a packed Host LCM block.
        num_host_lcm_blocks: Host LCM capacity used to bound dynamic block IDs.
        num_device_lcm_blocks: Device LCM capacity, excluding reserved block
            zero, used with group packing to bound dynamic block IDs.
        num_device_buffers: Number of Device buffers referenced by field rows.

    Returns:
        Validated HostTransferGeometry with Host field rows, layer slices,
        bounds, and precomputed per-field work sizing. Device tables remain
        unbound until ``bind`` is called; DMA uses the Host geometry directly.
    """
    if not rows:
        raise ValueError("geometry must contain at least one field row")
    if not group_packing:
        raise ValueError("geometry must contain at least one cache group")
    if host_lcm_block_bytes <= 0:
        raise ValueError("host_lcm_block_bytes must be positive")
    if num_host_lcm_blocks <= 0:
        raise ValueError("num_host_lcm_blocks must be positive")
    if num_device_lcm_blocks <= 0:
        raise ValueError("num_device_lcm_blocks must be positive")
    if num_device_buffers <= 0:
        raise ValueError("num_device_buffers must be positive")
    if any(packing <= 0 for packing in group_packing):
        raise ValueError("group_packing entries must be positive")
    if len(host_group_packing) != len(group_packing):
        raise ValueError("Host and Device packing must cover the same groups")
    if any(packing <= 0 for packing in host_group_packing):
        raise ValueError("host_group_packing entries must be positive")

    for row_index, row in enumerate(rows):
        _validate_geometry_row(
            row,
            row_index=row_index,
            group_packing=host_group_packing,
            num_device_buffers=num_device_buffers,
            host_lcm_block_bytes=host_lcm_block_bytes,
        )

    row_offset = 0
    for layer_index, (slice_offset, num_rows) in enumerate(layer_slices):
        if slice_offset != row_offset:
            raise ValueError(
                f"layer slice {layer_index} offset {slice_offset} != {row_offset}"
            )
        if num_rows < 0:
            raise ValueError("layer slice row counts must be non-negative")
        row_offset += num_rows
    if row_offset != len(rows):
        raise ValueError("layer slices do not cover all geometry rows")

    host_rows = _pinned_host_int64((len(rows), _GEOMETRY_ROW_WIDTH))
    host_rows.copy_(torch.tensor(rows, dtype=torch.int64, device="cpu"))

    return HostTransferGeometry(
        host_rows=host_rows,
        device_rows=None,
        device_layer_slices=None,
        layer_slices=tuple(layer_slices),
        group_packing=tuple(group_packing),
        host_group_packing=tuple(host_group_packing),
        host_lcm_block_bytes=int(host_lcm_block_bytes),
        num_host_lcm_blocks=int(num_host_lcm_blocks),
        num_device_lcm_blocks=int(num_device_lcm_blocks),
        row_work=tuple(
            (
                int(row[0]),
                (int(row[7]) + HOST_CACHE_TRANSFER_CHUNK_BYTES - 1)
                // HOST_CACHE_TRANSFER_CHUNK_BYTES,
            )
            for row in rows
        ),
    )


class HostTransferWorkspace:
    """Reusable address and block tables for one Host-cache transfer stream."""

    def __init__(self) -> None:
        self._address_key: tuple[int, ...] | None = None
        self._address_host: torch.Tensor | None = None
        self._address_table: torch.Tensor | None = None
        self._block_host: torch.Tensor | None = None
        self._block_device: torch.Tensor | None = None
        self._block_group_offsets_host: torch.Tensor | None = None
        self._block_group_offsets_device: torch.Tensor | None = None
        self._num_loaded_blocks = 0
        self._num_committed_blocks = 0
        self._block_group_offsets_count = 0
        self._block_load_generation = 0
        self._block_commit_generation = -1
        self._layer_ready_flags: torch.Tensor | None = None
        self._layer_cta_counts: torch.Tensor | None = None
        self._num_layer_ready = 0
        self._backend_key = None
        self._mode: HostTransferMode | None = None
        self._staging_buffer: StagingBuffer | None = None

    def bind_staging_buffer(self, staging_buffer: StagingBuffer) -> None:
        """Attach a shared stream-scoped staging buffer to this workspace."""
        self._staging_buffer = staging_buffer

    def staging_buffer(self) -> StagingBuffer:
        """Return the attached stream-scoped staging buffer."""
        if self._staging_buffer is None:
            raise RuntimeError(
                "staging_buffer must be bound via bind_staging_buffer() "
                "before a staged DMA transfer"
            )
        return self._staging_buffer

    def prepare_backend(
        self,
        device_buffers: Sequence[torch.Tensor],
        host_buffer: torch.Tensor,
        *,
        backend: Literal["auto", "triton", "dma"],
    ) -> HostTransferMode:
        """Resolve capability before submitting payload or publishing waits.

        Args:
            device_buffers: Cache buffers on one device.
            host_buffer: Pinned Host allocation to map.
            backend: Requested transport; only auto may fall back to DMA.

        Returns:
            Buffer-scoped transport and layer completion capability. Call on
            the transfer stream so newly allocated address tables are ordered.
        """
        if backend not in ("auto", "triton", "dma"):
            raise ValueError(f"unknown cache transfer backend {backend!r}")
        device = device_buffers[0].device
        if backend == "dma":
            return HostTransferMode("dma", layer_ready=False)
        if device.type == "npu":
            if backend == "triton":
                raise RuntimeError("mapped Host Triton transfer is unavailable on NPU")
            key = (
                device,
                tuple(buffer.data_ptr() for buffer in device_buffers),
                host_buffer.data_ptr(),
                "staged_dma",
            )
            if key != self._backend_key:
                self._backend_key = key
                self._mode = None
            if self._mode is None:
                self._address_table = self.bind_device_addresses(device_buffers)
                self._mode = HostTransferMode("staged_dma", layer_ready=False)
            return self._mode
        key = (
            device,
            tuple(buffer.data_ptr() for buffer in device_buffers),
            host_buffer.data_ptr(),
        )
        if key != self._backend_key:
            self._backend_key = key
            self._mode = None
        if self._mode is None:
            try:
                self._address_table = self.bind_addresses(device_buffers, host_buffer)
            except _MappedHostUnavailable:
                if backend == "triton":
                    raise
                self._mode = HostTransferMode("dma", layer_ready=False)
                warnings.warn(
                    "Mapped Host access is unavailable; falling back to DMA",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                self._mode = HostTransferMode(
                    "triton",
                    layer_ready=device.type == "cuda" and layer_ready_ptx_supported(),
                )
        if backend == "triton" and not self._mode.uses_device_tables:
            raise _MappedHostUnavailable("mapped Host access is not available")
        return self._mode

    def _invalidate_block_commit(self) -> None:
        self._block_load_generation += 1
        self._num_committed_blocks = 0
        self._block_group_offsets_count = 0
        self._block_commit_generation = -1

    def _require_committed_block_state(self) -> None:
        if self._block_commit_generation != self._block_load_generation:
            raise ValueError("block transfers are not committed for the current load")

    def _require_loaded_block_state(self, num_blocks: int) -> None:
        if self._block_commit_generation == self._block_load_generation:
            raise ValueError("block transfers already committed for this load")
        if num_blocks != self._num_loaded_blocks:
            raise ValueError(
                "num_blocks must equal the number of rows loaded for this generation"
            )

    def bind_device_addresses(
        self,
        device_buffers: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Upload Device buffer base pointers without mapping Host memory."""
        device = device_buffers[0].device
        key = tuple(int(buffer.data_ptr()) for buffer in device_buffers) + (
            int(device.index if device.index is not None else 0),
        )
        if self._address_table is None or self._address_key != key:
            addresses = [int(buffer.data_ptr()) for buffer in device_buffers]
            address_host = _pinned_host_int64((len(addresses),)).view(torch.uint64)
            address_host.copy_(
                torch.tensor(addresses, dtype=torch.uint64, device="cpu")
            )
            self._address_table = address_host.to(device, non_blocking=True)
            self._address_host = address_host
            self._address_key = key
        return self._address_table

    def bind_addresses(
        self,
        device_buffers: Sequence[torch.Tensor],
        host_buffer: torch.Tensor,
    ) -> torch.Tensor:
        device = device_buffers[0].device
        try:
            host_ptr = int(current_platform().device_visible_data_ptr(host_buffer))
        except (AttributeError, RuntimeError) as error:
            if not _triton_is_unavailable(error):
                raise
            raise _MappedHostUnavailable(str(error)) from error
        key = tuple(int(buffer.data_ptr()) for buffer in device_buffers) + (
            host_ptr,
            int(device.index if device.index is not None else 0),
        )
        if self._address_table is None or self._address_key != key:
            addresses = [int(buffer.data_ptr()) for buffer in device_buffers]
            addresses.append(host_ptr)
            # A rebind must not overwrite a pinned source whose previous
            # asynchronous upload may still be in flight. The pinned allocator
            # tracks that copy's completion before recycling the old source.
            address_host = _pinned_host_int64((len(addresses),)).view(torch.uint64)
            address_host.copy_(
                torch.tensor(addresses, dtype=torch.uint64, device="cpu")
            )
            self._address_table = address_host.to(device, non_blocking=True)
            self._address_host = address_host
            self._address_key = key
        return self._address_table

    def _ensure_block_host(self, num_blocks: int) -> torch.Tensor:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if self._block_host is None or self._block_host.shape[0] < num_blocks:
            capacity = num_blocks
            if self._block_host is not None:
                capacity = max(capacity, self._block_host.shape[0] * 2)
            self._block_host = _pinned_host_int64((capacity, 2))
        return self._block_host

    def _ensure_block_group_offsets_host(self, num_groups: int) -> torch.Tensor:
        needed = num_groups + 1
        if (
            self._block_group_offsets_host is None
            or self._block_group_offsets_host.shape[0] < needed
        ):
            capacity = needed
            if self._block_group_offsets_host is not None:
                capacity = max(capacity, self._block_group_offsets_host.shape[0] * 2)
            self._block_group_offsets_host = _pinned_host_int64((capacity,))
        return self._block_group_offsets_host

    def _validate_block_transfer(
        self,
        group_index: int,
        device_block_id: int,
        host_block_id: int,
        *,
        geometry: HostTransferGeometry,
    ) -> None:
        if not 0 <= group_index < geometry.num_groups:
            raise IndexError(f"group {group_index} outside [0, {geometry.num_groups})")
        if device_block_id <= 0 or host_block_id <= 0:
            raise ValueError(
                "device_block_id and host_block_id must be 1-based positive integers"
            )
        if device_block_id > geometry.max_device_block_id(group_index):
            raise IndexError(
                f"device_block_id {device_block_id} outside "
                f"[1, {geometry.max_device_block_id(group_index)}] "
                f"for group {group_index}"
            )
        max_host_block_id = geometry.max_host_block_id(group_index)
        if host_block_id > max_host_block_id:
            raise IndexError(
                f"host_block_id {host_block_id} outside [1, {max_host_block_id}] "
                f"for group {group_index}"
            )

    def _reset_loaded_blocks(self, num_groups: int) -> tuple[int, tuple[int, ...]]:
        group_offset_count = num_groups + 1
        offsets = tuple(0 for _ in range(group_offset_count))
        self._num_loaded_blocks = 0
        self._block_group_offsets_count = group_offset_count
        host_offsets = self._ensure_block_group_offsets_host(num_groups)
        host_offsets[:group_offset_count].zero_()
        return 0, offsets

    def _load_columnar_block_transfers(
        self,
        transfers: BlockTransferBatch,
        *,
        geometry: HostTransferGeometry,
    ) -> tuple[int, tuple[int, ...]]:
        if transfers.num_groups != geometry.num_groups:
            raise ValueError(
                f"transfer batch groups {transfers.num_groups} != "
                f"geometry groups {geometry.num_groups}"
            )
        num_blocks = transfers.total_blocks
        if num_blocks == 0:
            return self._reset_loaded_blocks(geometry.num_groups)

        host = self._ensure_block_host(num_blocks)
        host_np = host[:num_blocks].numpy()
        offset = 0
        group_offsets: list[int] = [0]
        for group_index, (device_ids, host_ids) in enumerate(
            zip(transfers.device_groups, transfers.host_groups)
        ):
            count = int(device_ids.size)
            if count:
                if int(device_ids.min()) <= 0 or int(host_ids.min()) <= 0:
                    raise ValueError(
                        "device_block_id and host_block_id must be 1-based "
                        "positive integers"
                    )
                max_device = int(device_ids.max())
                max_device_limit = geometry.max_device_block_id(group_index)
                if max_device > max_device_limit:
                    raise IndexError(
                        f"device_block_id {max_device} outside "
                        f"[1, {max_device_limit}] for group {group_index}"
                    )
                max_host = int(host_ids.max())
                max_host_limit = geometry.max_host_block_id(group_index)
                if max_host > max_host_limit:
                    raise IndexError(
                        f"host_block_id {max_host} outside "
                        f"[1, {max_host_limit}] for group {group_index}"
                    )
                end = offset + count
                host_np[offset:end, 0] = device_ids
                host_np[offset:end, 1] = host_ids
                offset = end
            group_offsets.append(offset)

        group_offset_count = geometry.num_groups + 1
        offsets_tensor = self._ensure_block_group_offsets_host(geometry.num_groups)
        offsets_tensor[:group_offset_count].numpy()[:] = group_offsets
        self._num_loaded_blocks = num_blocks
        self._block_group_offsets_count = group_offset_count
        return num_blocks, tuple(group_offsets)

    def load_block_transfers(
        self,
        transfers: BlockTransferBatch | Sequence[tuple[int, int, int]],
        *,
        geometry: HostTransferGeometry,
    ) -> tuple[int, tuple[int, ...]]:
        """Load dynamic block mappings bucketed stably by cache group.

        Args:
            transfers: Either a columnar :class:`BlockTransferBatch` or legacy
                ``(group_index, device_block_id, host_block_id)`` rows.
            geometry: Static bounds used to validate 1-based block IDs.

        Returns:
            ``(num_blocks, group_offsets)`` where ``group_offsets`` has length
            ``num_groups + 1`` and indexes the flattened host/device tables.
        """

        self._invalidate_block_commit()
        if isinstance(transfers, BlockTransferBatch):
            return self._load_columnar_block_transfers(transfers, geometry=geometry)
        if not transfers:
            return self._reset_loaded_blocks(geometry.num_groups)

        buckets: list[list[tuple[int, int]]] = [[] for _ in range(geometry.num_groups)]
        for group_index, device_block_id, host_block_id in transfers:
            self._validate_block_transfer(
                int(group_index),
                int(device_block_id),
                int(host_block_id),
                geometry=geometry,
            )
            buckets[int(group_index)].append((int(device_block_id), int(host_block_id)))

        flat_rows: list[tuple[int, int]] = []
        group_offsets: list[int] = [0]
        for bucket in buckets:
            flat_rows.extend(bucket)
            group_offsets.append(len(flat_rows))

        group_offset_count = geometry.num_groups + 1
        num_blocks = len(flat_rows)
        host = self._ensure_block_host(num_blocks)
        host[:num_blocks].copy_(
            torch.tensor(flat_rows, dtype=torch.int64, device="cpu")
        )
        offsets = self._ensure_block_group_offsets_host(geometry.num_groups)
        offsets[:group_offset_count].copy_(
            torch.tensor(group_offsets, dtype=torch.int64, device="cpu")
        )
        self._num_loaded_blocks = num_blocks
        self._block_group_offsets_count = group_offset_count
        return num_blocks, tuple(group_offsets)

    def commit_block_transfers(
        self,
        num_blocks: int,
        device: torch.device,
        *,
        non_blocking: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_loaded_block_state(num_blocks)
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if self._block_host is None:
            raise ValueError("block host table is empty")
        if self._block_group_offsets_host is None:
            raise ValueError("block group offsets are empty")
        if self._block_group_offsets_count <= 0:
            raise ValueError("block group offsets are empty")

        if self._block_device is None or self._block_device.shape[0] < num_blocks:
            capacity = num_blocks
            if self._block_device is not None:
                capacity = max(capacity, self._block_device.shape[0] * 2)
            self._block_device = torch.empty(
                (capacity, 2), dtype=torch.int64, device=device
            )

        offset_count = self._block_group_offsets_count
        if (
            self._block_group_offsets_device is None
            or self._block_group_offsets_device.shape[0] < offset_count
        ):
            capacity = offset_count
            if self._block_group_offsets_device is not None:
                capacity = max(capacity, self._block_group_offsets_device.shape[0] * 2)
            self._block_group_offsets_device = torch.empty(
                (capacity,), dtype=torch.int64, device=device
            )

        self._block_device[:num_blocks].copy_(
            self._block_host[:num_blocks], non_blocking=non_blocking
        )
        self._block_group_offsets_device[:offset_count].copy_(
            self._block_group_offsets_host[:offset_count],
            non_blocking=non_blocking,
        )
        self._num_committed_blocks = num_blocks
        self._block_commit_generation = self._block_load_generation
        return (
            self._block_device[:num_blocks],
            self._block_group_offsets_device[:offset_count],
        )

    def block_group_offsets_device(self) -> torch.Tensor:
        self._require_committed_block_state()
        if self._block_group_offsets_device is None:
            raise ValueError("block group offsets are empty")
        if self._block_group_offsets_count <= 0:
            raise ValueError("block group offsets are empty")
        return self._block_group_offsets_device[: self._block_group_offsets_count]

    def committed_block_tables(
        self,
        num_blocks: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one complete committed generation's Device metadata."""

        self._require_committed_block_state()
        if num_blocks != self._num_committed_blocks:
            raise ValueError(
                f"num_blocks {num_blocks} must equal committed block count "
                f"{self._num_committed_blocks}"
            )
        if self._block_device is None:
            raise ValueError("block device table is empty")
        return (
            self._block_device[:num_blocks],
            self.block_group_offsets_device(),
        )

    def host_block_rows(self, num_blocks: int) -> torch.Tensor:
        """Return valid Host block-pair rows for the current load."""

        if num_blocks != self._num_loaded_blocks:
            raise ValueError(
                "num_blocks must equal the number of rows loaded for this generation"
            )
        if self._block_host is None:
            raise ValueError("block host table is empty")
        return self._block_host[:num_blocks]

    def block_group_offsets_host(self) -> torch.Tensor:
        """Return only valid Host group offsets for the current load."""

        if self._block_group_offsets_host is None:
            raise ValueError("block group offsets are empty")
        if self._block_group_offsets_count <= 0:
            raise ValueError("block group offsets are empty")
        return self._block_group_offsets_host[: self._block_group_offsets_count]

    def prepare_layer_ready(
        self,
        num_layers: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Allocate or reuse zeroed per-layer ready flags and CTA counters."""

        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        reusable = (
            self._layer_ready_flags is not None
            and self._layer_cta_counts is not None
            and self._layer_ready_flags.device == device
            and self._layer_ready_flags.shape[0] >= num_layers
            and self._layer_cta_counts.shape[0] >= num_layers
        )
        if not reusable:
            capacity = num_layers
            if self._layer_ready_flags is not None:
                capacity = max(capacity, self._layer_ready_flags.shape[0] * 2)
            self._layer_ready_flags = torch.zeros(
                capacity, dtype=torch.int32, device=device
            )
            self._layer_cta_counts = torch.zeros(
                capacity, dtype=torch.int32, device=device
            )
        else:
            self._layer_ready_flags[:num_layers].zero_()
            self._layer_cta_counts[:num_layers].zero_()
        self._num_layer_ready = num_layers
        return self._layer_ready_flags[:num_layers]

    def layer_cta_counts(self) -> torch.Tensor:
        if self._layer_cta_counts is None or self._num_layer_ready <= 0:
            raise ValueError("layer-ready state has not been prepared")
        return self._layer_cta_counts[: self._num_layer_ready]


def _triton_is_unavailable(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "hostgetdevicepointer",
            "mapped host access is not available",
        )
    )


def _make_block_ranges(
    geometry: HostTransferGeometry,
    workspace: HostTransferWorkspace,
    *,
    num_blocks: int,
    geometry_offset: int,
    num_geometry_rows: int,
) -> tuple[tuple[int, int, int, int], ...]:
    geometry_rows = geometry.host_rows[
        geometry_offset : geometry_offset + num_geometry_rows
    ].tolist()
    block_rows = workspace.host_block_rows(num_blocks)
    group_offsets = workspace.block_group_offsets_host()
    group_block_rows = {}
    ranges = []
    for row in geometry_rows:
        (
            group_index,
            device_buffer_index,
            device_zero,
            device_stride,
            host_block_bytes,
            host_field_offset,
            packing,
            payload_bytes,
        ) = (int(value) for value in row)
        if group_index not in group_block_rows:
            group_start = int(group_offsets[group_index].item())
            group_end = int(group_offsets[group_index + 1].item())
            group_block_rows[group_index] = block_rows[group_start:group_end].tolist()
        for device_block_id, host_block_id in group_block_rows[group_index]:
            device_block_id = int(device_block_id)
            host_block_id = int(host_block_id)
            host_zero_based = host_block_id - 1
            host_parent, host_child = divmod(host_zero_based, packing)
            ranges.append(
                (
                    device_buffer_index,
                    device_zero + device_block_id * device_stride,
                    host_parent * geometry.host_lcm_block_bytes
                    + host_child * host_block_bytes
                    + host_field_offset,
                    payload_bytes,
                )
            )
    return tuple(ranges)


def _block_work_items(
    geometry: HostTransferGeometry,
    workspace: HostTransferWorkspace,
    *,
    geometry_offset: int,
    num_geometry_rows: int,
) -> int:
    """Return the largest real block/chunk count in a geometry slice."""

    group_offsets = workspace.block_group_offsets_host().tolist()
    work_items = 0
    for group_index, num_chunks in geometry.row_work[
        geometry_offset : geometry_offset + num_geometry_rows
    ]:
        group_blocks = int(group_offsets[group_index + 1] - group_offsets[group_index])
        work_items = max(work_items, group_blocks * num_chunks)
    return work_items


def transfer_cache_blocks(
    direction: Literal["d2h", "h2d"],
    device_buffers: Sequence[torch.Tensor],
    host_buffer: torch.Tensor,
    geometry: HostTransferGeometry,
    workspace: HostTransferWorkspace,
    stream,
    *,
    num_blocks: int,
    geometry_offset: int,
    num_geometry_rows: int,
    backend: Literal["auto", "triton", "dma"],
    grid_cap: int | None,
    layer_ready_flags: torch.Tensor | None,
) -> None:
    """Copy compact Host blocks using static geometry and dynamic block IDs.

    Args:
        direction: ``"d2h"`` for writeback or ``"h2d"`` for restore.
        device_buffers: Device tensors referenced by static geometry rows.
        host_buffer: Contiguous pinned uint8 compact Host allocation.
        geometry: Immutable transfer geometry, Device-bound when using Triton.
        workspace: Current dynamic block mapping; Triton requires it committed.
        stream: Device stream ordering the asynchronous copies.
        num_blocks: Number of valid dynamic block-pair rows.
        geometry_offset: First static field row for this layer.
        num_geometry_rows: Number of static field rows for this layer.
        backend: Prefer mapped-Host Triton or lazily expand ranges for DMA.
        grid_cap: Max Triton CTAs, or None to use the configured grid cap.
        layer_ready_flags: Optional per-layer Device flags. On NVIDIA, Triton
            copies every layer slice in one grid and release-stores each flag.
            Other vendors keep the event path; ``auto`` falls back to DMA and
            fills the flags after the full range copy.

    Returns:
        None. Completion is observed by recording an event on ``stream``.
    """

    if direction not in ("d2h", "h2d"):
        raise ValueError(f"unknown cache transfer direction {direction!r}")
    if backend not in ("auto", "triton", "dma"):
        raise ValueError(f"unknown cache transfer backend {backend!r}")
    if geometry_offset < 0 or num_geometry_rows < 0:
        raise ValueError("geometry slice values must be non-negative")
    if geometry_offset + num_geometry_rows > geometry.num_field_rows:
        raise ValueError("geometry slice lies outside the static table")
    if num_blocks < 0:
        raise ValueError("num_blocks must be non-negative")
    if layer_ready_flags is not None:
        if (
            direction != "h2d"
            or geometry_offset != 0
            or num_geometry_rows != geometry.num_field_rows
        ):
            raise ValueError("layer-ready flags require a full-geometry H2D transfer")
        if layer_ready_flags.dtype != torch.int32 or layer_ready_flags.ndim != 1:
            raise ValueError("layer_ready_flags must be a 1-D int32 tensor")
        if layer_ready_flags.numel() != len(geometry.layer_slices):
            raise ValueError("layer_ready_flags must cover every geometry layer")
    if num_blocks == 0:
        return
    if layer_ready_flags is None and num_geometry_rows == 0:
        return

    def _publish_dma_flags() -> None:
        if layer_ready_flags is None:
            return
        device_module = torch.get_device_module(device_buffers[0].device)
        with device_module.stream(stream) if stream is not None else nullcontext():
            layer_ready_flags.fill_(1)

    def transfer_dma() -> None:
        ranges = _make_block_ranges(
            geometry,
            workspace,
            num_blocks=num_blocks,
            geometry_offset=geometry_offset,
            num_geometry_rows=num_geometry_rows,
        )
        _transfer_cache_ranges(
            direction,
            device_buffers,
            host_buffer,
            ranges,
            stream,
        )
        _publish_dma_flags()

    if backend == "dma":
        transfer_dma()
        return
    if layer_ready_flags is not None and not layer_ready_ptx_supported():
        if backend == "triton":
            raise ValueError("flagged Triton transfers require NVIDIA PTX")
        transfer_dma()
        return

    # An empty layer must not map buffers or touch the accelerator runtime.
    work_items = _block_work_items(
        geometry,
        workspace,
        geometry_offset=geometry_offset,
        num_geometry_rows=num_geometry_rows,
    )
    if work_items == 0 and layer_ready_flags is None:
        workspace.host_block_rows(num_blocks)
        return

    device_module = torch.get_device_module(device_buffers[0].device)
    with device_module.stream(stream) if stream is not None else nullcontext():
        mode = workspace.prepare_backend(device_buffers, host_buffer, backend=backend)
        if mode.backend == "dma":
            transfer_dma()
            return
        if geometry.device_rows is None:
            raise ValueError("geometry must be bound before a Triton block transfer")
        block_rows, group_offsets = workspace.committed_block_tables(num_blocks)
        if mode.backend == "staged_dma":
            _transfer_staged_dma(
                direction,
                device_buffers,
                host_buffer,
                geometry,
                workspace,
                block_rows,
                stream,
                num_blocks=num_blocks,
                geometry_offset=geometry_offset,
                num_geometry_rows=num_geometry_rows,
                grid_cap=grid_cap,
            )
            _publish_dma_flags()
            return
        triton_kwargs = {
            "geometry_offset": geometry_offset,
            "num_geometry_rows": num_geometry_rows,
            "host_lcm_block_bytes": geometry.host_lcm_block_bytes,
            "work_items": work_items,
            "num_device_buffers": len(device_buffers),
            "grid_cap": grid_cap,
            "layer_ready_flags": None,
            "layer_slices": None,
            "layer_cta_counts": None,
        }
        if layer_ready_flags is not None:
            if geometry.device_layer_slices is None:
                raise ValueError(
                    "geometry layer slices must be bound before a flagged Triton transfer"
                )
            triton_kwargs["layer_ready_flags"] = layer_ready_flags
            triton_kwargs["layer_slices"] = geometry.device_layer_slices
            triton_kwargs["layer_cta_counts"] = workspace.layer_cta_counts()
        _transfer_cache_blocks_triton(
            workspace._address_table,
            geometry.device_rows,
            block_rows,
            group_offsets,
            0 if direction == "d2h" else 1,
            **triton_kwargs,
        )


def _ascend_stream_and_device(device: torch.device, stream) -> tuple[int, int]:
    """Return ``(device_id, aclrtStream)`` after flushing torch_npu's task queue."""
    device_module = torch.get_device_module(device)
    active_stream = (
        stream if stream is not None else device_module.current_stream(device)
    )
    device_id = int(
        device.index if device.index is not None else device_module.current_device()
    )
    # Accessing ``.npu_stream`` drains torch_npu's host task queue before
    # submitting directly to ``libascendcl``, preserving stream FIFO order.
    return device_id, int(active_stream.npu_stream)


def _copy_staging_host(
    direction: Literal["d2h", "h2d"],
    staging: torch.Tensor,
    host_buffer: torch.Tensor,
    staging_offsets: Sequence[int],
    host_offsets: Sequence[int],
    sizes: Sequence[int],
    *,
    stream,
) -> None:
    if not sizes:
        return
    if staging.device.type == "npu":
        if not host_buffer.is_pinned():
            raise ValueError("host_buffer must be pinned CPU memory for NPU ACL memcpy")
        from tokenspeed_kernel.thirdparty.ascend.memcpy import (
            ascend_memcpy_batch_async,
        )

        staging_base = int(staging.data_ptr())
        host_base = int(host_buffer.data_ptr())
        if direction == "d2h":
            dst_addrs = [host_base + offset for offset in host_offsets]
            src_addrs = [staging_base + offset for offset in staging_offsets]
        else:
            dst_addrs = [staging_base + offset for offset in staging_offsets]
            src_addrs = [host_base + offset for offset in host_offsets]
        device_id, stream_ptr = _ascend_stream_and_device(staging.device, stream)
        ascend_memcpy_batch_async(
            dst_addrs=dst_addrs,
            src_addrs=src_addrs,
            sizes=sizes,
            direction=direction,
            device_id=device_id,
            stream_ptr=stream_ptr,
        )
        return

    for staging_offset, host_offset, num_bytes in zip(
        staging_offsets, host_offsets, sizes, strict=True
    ):
        staging_slice = staging[staging_offset : staging_offset + num_bytes]
        host_slice = host_buffer[host_offset : host_offset + num_bytes]
        if direction == "d2h":
            host_slice.copy_(staging_slice, non_blocking=True)
        else:
            staging_slice.copy_(host_slice, non_blocking=True)


def _transfer_staged_dma(
    direction: Literal["d2h", "h2d"],
    device_buffers: Sequence[torch.Tensor],
    host_buffer: torch.Tensor,
    geometry: HostTransferGeometry,
    workspace: HostTransferWorkspace,
    block_rows_device: torch.Tensor,
    stream,
    *,
    num_blocks: int,
    geometry_offset: int,
    num_geometry_rows: int,
    grid_cap: int | None,
) -> None:
    """Gather/scatter scattered Device fields via HBM staging and batched SDMA.

    For each cache group referenced by ``(geometry_offset, num_geometry_rows)``,
    fields are packed into the group's exact compact Host block layout inside a
    fixed-size HBM staging buffer and moved between staging and Host in one
    batched SDMA transfer per chunk. This collapses dozens of small per-layer /
    per-field copies (such as 4 KB TLS metadata fields) into one contiguous
    transfer per block while keeping HBM usage O(1) with respect to context
    length.
    """
    device = device_buffers[0].device
    if (
        host_buffer.device.type != "cpu"
        or host_buffer.dtype != torch.uint8
        or not host_buffer.is_contiguous()
        or (device.type != "cpu" and not host_buffer.is_pinned())
    ):
        raise ValueError("host_buffer must be contiguous pinned CPU uint8")
    group_slices = geometry.staged_group_slices(geometry_offset, num_geometry_rows)
    if not group_slices:
        return
    group_offsets = workspace.block_group_offsets_host().tolist()
    active_slices = [
        group_slice
        for group_slice in group_slices
        if int(group_offsets[group_slice.group_index + 1])
        > int(group_offsets[group_slice.group_index])
    ]
    if not active_slices:
        return

    host_block_rows = workspace.host_block_rows(num_blocks)
    max_block_bytes = max(
        group_slice.staged_block_bytes for group_slice in active_slices
    )
    staging = workspace.staging_buffer().acquire(max_block_bytes, device)
    staging_bytes = staging.numel()
    stage_direction = 0 if direction == "d2h" else 1
    host_lcm_bytes = geometry.host_lcm_block_bytes
    host_numel = host_buffer.numel()

    for group_slice in active_slices:
        group_start = int(group_offsets[group_slice.group_index])
        group_end = int(group_offsets[group_slice.group_index + 1])
        group_count = group_end - group_start
        host_ids = host_block_rows[group_start:group_end, 1].tolist()
        chunk_capacity = max(1, staging_bytes // group_slice.staged_block_bytes)
        span_base = group_slice.span_base
        spans = group_slice.spans
        packing = group_slice.packing
        host_block_bytes = group_slice.host_block_bytes
        use_words = (
            group_slice.word_aligned_geometry
            and (staging.data_ptr() & 3) == 0
            and all(
                (device_buffers[buf_idx].data_ptr() & 3) == 0
                for buf_idx in group_slice.buffer_indices
            )
        )
        max_chunks = (
            group_slice.max_word_chunks if use_words else group_slice.max_byte_chunks
        )

        for local_start in range(0, group_count, chunk_capacity):
            local_end = min(local_start + chunk_capacity, group_count)
            chunk_blocks = local_end - local_start
            staging_offsets: list[int] = []
            host_offsets: list[int] = []
            sizes: list[int] = []
            for local_index in range(chunk_blocks):
                host_zero_based = int(host_ids[local_start + local_index]) - 1
                host_parent, host_child = divmod(host_zero_based, packing)
                host_block_offset = (
                    host_parent * host_lcm_bytes + host_child * host_block_bytes
                )
                staging_block_offset = local_index * group_slice.staged_block_bytes
                for span_offset, span_bytes in spans:
                    target_host_offset = host_block_offset + span_offset
                    if target_host_offset + span_bytes > host_numel:
                        raise IndexError(
                            "staged cache transfer range lies outside Host buffer"
                        )
                    staging_offsets.append(
                        staging_block_offset + (span_offset - span_base)
                    )
                    host_offsets.append(target_host_offset)
                    sizes.append(span_bytes)

            if direction == "d2h":
                _stage_cache_blocks_triton(
                    workspace._address_table,
                    geometry.device_rows,
                    block_rows_device,
                    staging,
                    stage_direction,
                    geometry_offset=geometry_offset,
                    num_geometry_rows=num_geometry_rows,
                    target_group=group_slice.group_index,
                    chunk_start=group_start + local_start,
                    chunk_blocks=chunk_blocks,
                    span_base=span_base,
                    staged_block_bytes=group_slice.staged_block_bytes,
                    work_items=chunk_blocks * max_chunks,
                    use_words=use_words,
                    grid_cap=grid_cap,
                )
                _copy_staging_host(
                    direction,
                    staging,
                    host_buffer,
                    staging_offsets,
                    host_offsets,
                    sizes,
                    stream=stream,
                )
            else:
                _copy_staging_host(
                    direction,
                    staging,
                    host_buffer,
                    staging_offsets,
                    host_offsets,
                    sizes,
                    stream=stream,
                )
                _stage_cache_blocks_triton(
                    workspace._address_table,
                    geometry.device_rows,
                    block_rows_device,
                    staging,
                    stage_direction,
                    geometry_offset=geometry_offset,
                    num_geometry_rows=num_geometry_rows,
                    target_group=group_slice.group_index,
                    chunk_start=group_start + local_start,
                    chunk_blocks=chunk_blocks,
                    span_base=span_base,
                    staged_block_bytes=group_slice.staged_block_bytes,
                    work_items=chunk_blocks * max_chunks,
                    use_words=use_words,
                    grid_cap=grid_cap,
                )


def _validate_ranges(
    device_buffers: Sequence[torch.Tensor],
    host_buffer: torch.Tensor,
    ranges: Sequence[tuple[int, int, int, int]],
) -> None:
    if (
        host_buffer.device.type != "cpu"
        or host_buffer.dtype != torch.uint8
        or not host_buffer.is_contiguous()
        or not host_buffer.is_pinned()
    ):
        raise ValueError("host_buffer must be contiguous pinned CPU uint8")
    for buffer in device_buffers:
        if buffer.device.type == "cpu" or not buffer.is_contiguous():
            raise ValueError("device cache buffers must be contiguous device tensors")
    devices = {buffer.device for buffer in device_buffers}
    if len(devices) > 1:
        raise ValueError("device cache buffers must be on one device")
    for device_buffer_index, device_offset, host_offset, num_bytes in ranges:
        if not 0 <= device_buffer_index < len(device_buffers):
            raise IndexError(f"unknown device buffer {device_buffer_index}")
        if device_offset < 0 or host_offset < 0 or num_bytes <= 0:
            raise ValueError(
                "cache transfer offsets must be non-negative and non-empty"
            )
        device_bytes = (
            device_buffers[device_buffer_index].numel()
            * device_buffers[device_buffer_index].element_size()
        )
        if device_offset + num_bytes > device_bytes:
            raise IndexError("cache transfer range lies outside its device buffer")
        if host_offset + num_bytes > host_buffer.numel():
            raise IndexError("cache transfer range lies outside Host buffer")


def _transfer_dma(
    direction: Literal["d2h", "h2d"],
    device_buffers: Sequence[torch.Tensor],
    host_buffer: torch.Tensor,
    ranges: Sequence[tuple[int, int, int, int]],
    *,
    stream=None,
) -> None:
    if device_buffers and device_buffers[0].device.type == "npu":
        if not host_buffer.is_pinned():
            raise ValueError("host_buffer must be pinned CPU memory for NPU ACL memcpy")
        from tokenspeed_kernel.thirdparty.ascend.memcpy import (
            ascend_memcpy_batch_async,
        )

        device_bases = tuple(int(buffer.data_ptr()) for buffer in device_buffers)
        host_base = int(host_buffer.data_ptr())
        if direction == "d2h":
            dst_addrs = [host_base + host_offset for _, _, host_offset, _ in ranges]
            src_addrs = [
                device_bases[buf_idx] + dev_offset
                for buf_idx, dev_offset, _, _ in ranges
            ]
        else:
            dst_addrs = [
                device_bases[buf_idx] + dev_offset
                for buf_idx, dev_offset, _, _ in ranges
            ]
            src_addrs = [host_base + host_offset for _, _, host_offset, _ in ranges]
        sizes = [num_bytes for _, _, _, num_bytes in ranges]
        device_id, stream_ptr = _ascend_stream_and_device(
            device_buffers[0].device, stream
        )
        ascend_memcpy_batch_async(
            dst_addrs=dst_addrs,
            src_addrs=src_addrs,
            sizes=sizes,
            direction=direction,
            device_id=device_id,
            stream_ptr=stream_ptr,
        )
        return

    byte_buffers = tuple(
        buffer.view(torch.uint8).reshape(-1) for buffer in device_buffers
    )
    for device_buffer_index, device_offset, host_offset, num_bytes in ranges:
        device_slice = byte_buffers[device_buffer_index][
            device_offset : device_offset + num_bytes
        ]
        host_slice = host_buffer[host_offset : host_offset + num_bytes]
        destination, source = (
            (host_slice, device_slice)
            if direction == "d2h"
            else (device_slice, host_slice)
        )
        destination.copy_(source, non_blocking=True)


def _transfer_cache_ranges(
    direction: Literal["d2h", "h2d"],
    device_buffers: Sequence[torch.Tensor],
    host_buffer: torch.Tensor,
    ranges: Sequence[tuple[int, int, int, int]],
    stream,
) -> None:
    """Copy byte ranges with asynchronous DMA.

    Args:
        direction: ``"d2h"`` for snapshot/store or ``"h2d"`` for load/recover.
        device_buffers: Device tensors referenced by range buffer indices.
        host_buffer: Contiguous pinned uint8 Host allocation.
        ranges: ``(device_buffer_index, device_offset, host_offset, num_bytes)``
            rows.
        stream: Device stream that orders the asynchronous copies.

    Returns:
        None. Completion is observed by recording an event on ``stream``.
    """

    if direction not in ("d2h", "h2d"):
        raise ValueError(f"unknown cache transfer direction {direction!r}")
    _validate_ranges(device_buffers, host_buffer, ranges)
    if not ranges:
        return
    device_module = torch.get_device_module(device_buffers[0].device)
    with device_module.stream(stream):
        _transfer_dma(direction, device_buffers, host_buffer, ranges, stream=stream)


def wait_layer_ready(flags: torch.Tensor, layer_index: int) -> None:
    """Wait on the current stream until layer ``layer_index`` is released.

    NVIDIA-only: the waiter uses ``ld.acquire.gpu``. Other vendors wait on
    recorded CUDA/HIP events instead.

    Args:
        flags: Device int32 per-layer completion flags.
        layer_index: Consumer-local layer to wait for.
    """

    if not layer_ready_ptx_supported():
        raise RuntimeError("wait_layer_ready requires NVIDIA PTX")
    _wait_layer_ready_triton(flags, layer_index)
