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

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from tokenspeed.runtime.cache.transfer.layout import CacheTransferLayout

if TYPE_CHECKING:
    from tokenspeed.runtime.cache.transfer.layout import CacheGroupLayout
    from tokenspeed.runtime.cache.transfer.ownership import BlockOwnerTranslation
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
        CacheMemoryPlan,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
        CacheGroupSpec,
    )

# The State density at which a Host parent rounded up to whole KV pages must
# still pay: one State checkpoint (a payload of every State group) per this
# many cached tokens. A request publishes only a few checkpoints (at its prompt
# end, a junction cut, its finish), so a prefix cache of long prompts holds far
# fewer; a denser guess only makes rounding up rarer.
_TOKENS_PER_STATE_CHECKPOINT = 8192


def _host_bytes_per_token(
    parent_bytes: int, groups: Sequence[tuple[CacheGroupSpec, int]]
) -> Fraction:
    """Host L2 bytes per cached token when each parent is ``parent_bytes``.

    A history group stores one block per ``page_size`` tokens and a State
    group one per checkpoint, each dealt among its ``shard_count`` owners;
    a replayable group never reaches Host L2.
    """
    parents_per_token = Fraction(0)
    for spec, payload_bytes in groups:
        if spec.replayable:
            continue
        tokens_per_block = (
            _TOKENS_PER_STATE_CHECKPOINT
            if spec.attn_kind == "state"
            else spec.page_size
        )
        parents_per_token += Fraction(
            1,
            tokens_per_block * spec.shard_count * (parent_bytes // payload_bytes),
        )
    return parent_bytes * parents_per_token


def _host_parent_bytes(groups: Sequence[tuple[CacheGroupSpec, int]]) -> int:
    """Bytes of the shared Host parent for ``(spec, payload_bytes)`` groups."""
    largest = max(payload_bytes for _, payload_bytes in groups)
    if any(
        spec.attn_kind != "state" and not spec.replayable and payload_bytes == largest
        for spec, payload_bytes in groups
    ):
        # A KV page Host L2 caches is the largest payload and fills its parent
        # exactly, as in every pure-attention model without a replayable window.
        return largest
    # Host bytes per token and page of each KV group Host L2 stores.
    cached_kv = [
        (Fraction(payload_bytes, spec.page_size * spec.shard_count), payload_bytes)
        for spec, payload_bytes in groups
        if spec.attn_kind != "state" and not spec.replayable
    ]
    if not cached_kv:
        return largest
    # The largest payload is a State checkpoint or a replayable window page,
    # which Host L2 never stores and which only sizes the parent. It leaves a
    # gap after the last whole KV page of every KV parent. Rounding the parent
    # up to whole pages of the KV group with the most Host bytes per token
    # closes that gap but adds the same growth to every parent of the largest
    # payload as slack, so it is taken only when it lowers the Host bytes per
    # cached token.
    _, page = max(cached_kv)
    rounded = -(-largest // page) * page
    if _host_bytes_per_token(rounded, groups) < _host_bytes_per_token(largest, groups):
        return rounded
    return largest


def _compute_host_layout(
    layout: CacheTransferLayout, *, group_specs: Sequence[CacheGroupSpec]
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...], int, tuple[int, ...]]:
    specs = {spec.group_id: spec for spec in group_specs}
    host_cache_block_bytes = []
    host_field_offsets = []
    for group in layout.groups:
        offsets = []
        cursor = 0
        for field in group.fields:
            offsets.append(cursor)
            cursor += field.payload_bytes
        host_cache_block_bytes.append(cursor)
        host_field_offsets.append(tuple(offsets))
    # Device packing satisfies model/kernel strides. Host bytes have no such
    # stride constraint: one shared parent fits the largest compact payload,
    # possibly rounded up to whole KV pages (``_host_parent_bytes``), and each
    # group fills it with as many complete payloads as it can hold.
    host_lcm_block_bytes = _host_parent_bytes(
        [
            (specs[group.group_id], size)
            for group, size in zip(layout.groups, host_cache_block_bytes)
        ]
    )
    host_group_packing = tuple(
        host_lcm_block_bytes // size for size in host_cache_block_bytes
    )
    return (
        tuple(host_cache_block_bytes),
        tuple(host_field_offsets),
        host_lcm_block_bytes,
        host_group_packing,
    )


def compute_host_lcm_block_bytes(
    layout: CacheTransferLayout, *, group_specs: Sequence[CacheGroupSpec]
) -> int:
    """Return compact Host bytes required by one LCMBlock.

    Args:
        layout: The transfer layout whose payloads the Host packs.
        group_specs: The scheduler's group specs. Their kinds, page sizes,
            shard counts and replay flags decide whether the parent rounds
            up from the largest payload to whole KV pages.
    """

    return _compute_host_layout(layout, group_specs=group_specs)[2]


def compute_host_group_packing(
    layout: CacheTransferLayout, *, group_specs: Sequence[CacheGroupSpec]
) -> tuple[int, ...]:
    """Return physical children per shared Host parent in layout group order.

    Args:
        layout: The transfer layout whose payloads the Host packs.
        group_specs: The scheduler's group specs, as for
            ``compute_host_lcm_block_bytes``.
    """

    return _compute_host_layout(layout, group_specs=group_specs)[3]


def validate_host_group_packing(packing: tuple[int, ...], *, cpu_group) -> None:
    """Require identical Host block IDs on every mirrored scheduler rank.

    Args:
        packing: Virtual Host children per parent in scheduler group order.
        cpu_group: The ranks' Gloo group, or None for one rank.
    """
    if cpu_group is None:
        return
    import torch
    import torch.distributed as dist

    # Compare lengths first so every rank joins the same number of collectives.
    for value in (len(packing), *packing):
        minimum = torch.tensor(value, dtype=torch.int64)
        negative_maximum = torch.tensor(-value, dtype=torch.int64)
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN, group=cpu_group)
        dist.all_reduce(negative_maximum, op=dist.ReduceOp.MIN, group=cpu_group)
        if minimum.item() != -negative_maximum.item():
            raise ValueError(
                "Host packing must match across scheduler ranks; their compact "
                f"Host layouts differ (this rank's packing is {packing})"
            )


class HostCacheStorage:
    """Shared Host allocation indexed by scheduler CacheBlock IDs.

    L2 and L3 retain this resource independently. Construction consumes the
    transfer layout but never retains its device buffers.
    """

    def __init__(
        self,
        layout: CacheTransferLayout,
        *,
        group_specs: Sequence[CacheGroupSpec],
        num_host_lcm_blocks: int,
    ):
        """
        Args:
            layout: The transfer layout whose payloads the Host packs.
            group_specs: The scheduler's group specs, which size the shared
                parent (``compute_host_lcm_block_bytes``).
            num_host_lcm_blocks: Parents to allocate, the null block excluded.
        """
        if num_host_lcm_blocks <= 0:
            raise ValueError("num_host_lcm_blocks must be > 0")
        self.group_ids = tuple(group.group_id for group in layout.groups)
        (
            self.host_cache_block_bytes,
            self.host_field_offsets,
            self.host_lcm_block_bytes,
            self.group_packing,
        ) = _compute_host_layout(layout, group_specs=group_specs)
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
            packing = self.group_packing[group_index]
            host_cache_block_bytes = self.host_cache_block_bytes[group_index]
        except IndexError as exc:
            raise IndexError(f"unknown cache group index {group_index}") from exc
        max_block_id = self.num_host_lcm_blocks * packing
        if block_id <= 0 or block_id > max_block_id:
            raise IndexError(
                f"block_id {block_id} outside [1, {max_block_id}] for "
                f"group {self.group_ids[group_index]!r}"
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


@dataclass(frozen=True, slots=True)
class HostCacheResources:
    """Startup-only Host resources and resolved CPU cache facts.

    The memory plan and group descriptors carry no Device buffers. Consumers
    derive their own compatibility schema without retaining a Device executor.
    """

    storage: HostCacheStorage
    ownership: BlockOwnerTranslation
    groups: tuple[CacheGroupLayout, ...]
    memory_plan: CacheMemoryPlan
    attention_backend_name: str
    draft_attention_backend_name: str
    checkpoint_granularities: tuple[tuple[str, int], ...] = ()
