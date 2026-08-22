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

"""Compact pinned Host storage for cache transfer entries."""

from __future__ import annotations

from tokenspeed.runtime.cache.transfer.layout import CacheTransferLayout


def _compute_host_layout(
    layout: CacheTransferLayout,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...], int]:
    host_cache_block_bytes = []
    host_field_offsets = []
    host_lcm_block_bytes = 0
    for group in layout.groups:
        offsets = []
        cursor = 0
        for field in group.fields:
            offsets.append(cursor)
            cursor += field.payload_bytes
        host_cache_block_bytes.append(cursor)
        host_field_offsets.append(tuple(offsets))
        host_lcm_block_bytes = max(
            host_lcm_block_bytes,
            group.cache_blocks_per_lcm_block * cursor,
        )
    return (
        tuple(host_cache_block_bytes),
        tuple(host_field_offsets),
        host_lcm_block_bytes,
    )


def compute_host_lcm_block_bytes(layout: CacheTransferLayout) -> int:
    """Return compact Host bytes required by one LCMBlock."""

    return _compute_host_layout(layout)[2]


class HostCacheStorage:
    """One compact Host allocation indexed by scheduler CacheBlock IDs."""

    def __init__(
        self,
        layout: CacheTransferLayout,
        *,
        num_host_lcm_blocks: int,
    ):
        if num_host_lcm_blocks <= 0:
            raise ValueError("num_host_lcm_blocks must be > 0")
        self.layout = layout
        (
            self.host_cache_block_bytes,
            self.host_field_offsets,
            self.host_lcm_block_bytes,
        ) = _compute_host_layout(layout)
        self.num_host_lcm_blocks = int(num_host_lcm_blocks)
        self.host_buffer_bytes = self.num_host_lcm_blocks * self.host_lcm_block_bytes
        import torch

        self.host_buffer = torch.empty(
            self.host_buffer_bytes,
            dtype=torch.uint8,
            pin_memory=True,
        )

    def host_block_offset(self, group_index: int, block_id: int) -> int:
        """Return the packed byte offset for one non-null CacheBlock."""

        try:
            group = self.layout.groups[group_index]
            host_cache_block_bytes = self.host_cache_block_bytes[group_index]
        except IndexError as exc:
            raise IndexError(f"unknown cache group index {group_index}") from exc
        packing = group.cache_blocks_per_lcm_block
        max_block_id = self.num_host_lcm_blocks * packing
        if block_id <= 0 or block_id > max_block_id:
            raise IndexError(
                f"block_id {block_id} outside [1, {max_block_id}] for "
                f"group {group.group_id!r}"
            )
        zero_based = block_id - 1
        parent_index, child_index = divmod(zero_based, packing)
        return (
            parent_index * self.host_lcm_block_bytes
            + child_index * host_cache_block_bytes
        )

    def host_block_range(self, group_index: int, block_id: int) -> tuple[int, int]:
        """Return ``(byte_offset, payload_bytes)`` for one Host CacheBlock."""

        return (
            self.host_block_offset(group_index, block_id),
            self.host_cache_block_bytes[group_index],
        )

    def host_field_offset(
        self, group_index: int, block_id: int, field_index: int
    ) -> int:
        try:
            host_field_offset = self.host_field_offsets[group_index][field_index]
        except IndexError as exc:
            raise IndexError(
                f"unknown field {field_index} for cache group {group_index}"
            ) from exc
        return self.host_block_offset(group_index, block_id) + host_field_offset
