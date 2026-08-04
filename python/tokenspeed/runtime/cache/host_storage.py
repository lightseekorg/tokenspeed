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

from tokenspeed.runtime.cache.layout import CacheTransferLayout


def _backing_bytes(backing: object) -> int:
    numel = getattr(backing, "numel", None)
    return int(numel()) if callable(numel) else len(backing)


class HostCacheStorage:
    """One compact Host allocation indexed by scheduler child page IDs."""

    def __init__(
        self,
        layout: CacheTransferLayout,
        *,
        num_lcm_blocks: int,
        backing: object | None = None,
    ):
        if num_lcm_blocks <= 0:
            raise ValueError("num_lcm_blocks must be > 0")
        self.layout = layout
        self.packed = layout.pack()
        self.num_lcm_blocks = int(num_lcm_blocks)
        self.num_bytes = self.num_lcm_blocks * self.packed.parent_bytes
        if backing is None:
            import torch

            backing = torch.empty(self.num_bytes, dtype=torch.uint8, pin_memory=True)
        if _backing_bytes(backing) != self.num_bytes:
            raise ValueError(
                f"Host cache backing has {_backing_bytes(backing)} bytes; "
                f"expected {self.num_bytes}"
            )
        self.backing = backing

    def child_offset(self, group_index: int, page_id: int) -> int:
        """Return the packed byte offset for one non-null child page."""

        try:
            group = self.layout.groups[group_index]
            child_bytes = self.packed.child_bytes[group_index]
        except IndexError as exc:
            raise IndexError(f"unknown cache group index {group_index}") from exc
        packing = group.cache_blocks_per_lcm_block
        max_page_id = self.num_lcm_blocks * packing
        if page_id <= 0 or page_id > max_page_id:
            raise IndexError(
                f"page_id {page_id} outside [1, {max_page_id}] for "
                f"group {group.group_id!r}"
            )
        zero_based = page_id - 1
        parent_index, child_index = divmod(zero_based, packing)
        return (
            parent_index * self.packed.parent_bytes
            + child_index * child_bytes
        )

    def segment_offset(
        self, group_index: int, page_id: int, segment_index: int
    ) -> int:
        try:
            segment_offset = self.packed.segment_offsets[group_index][segment_index]
        except IndexError as exc:
            raise IndexError(
                f"unknown segment {segment_index} for cache group {group_index}"
            ) from exc
        return self.child_offset(group_index, page_id) + segment_offset
