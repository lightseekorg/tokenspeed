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

"""The one cache allocation every compute view is a window onto."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from tokenspeed_kernel.ops.kvcache.triton import zero_byte_ranges

from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    CacheRuntimeContract,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import CacheMemoryPlan
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    CacheGroupSpec,
)
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = get_colorful_logger(__name__)


class CacheArena:
    """Own the cache allocation, its typed field views, and its contract.

    One model, one arena. Every ``CachePool`` is a compute view onto an
    arena and owns no memory of its own: target and draft views of one
    merged plan share this object, so "who allocates" and "who may clear"
    stop being per-pool modes and become properties of the single owner.

    The arena is the sole publisher of the scheduler-facing runtime
    contract, so a secondary view cannot republish or diverge from it.
    Publication is unconditional: an arena without group specs would be an
    arena no scheduler can address, and ModelExecutor rejects it anyway.
    """

    def __init__(
        self,
        plan: CacheMemoryPlan,
        device: str,
        *,
        cache_group_specs: tuple[CacheGroupSpec, ...],
        token_capacity: int | None = None,
        enable_memory_saver: bool = False,
    ):
        if not cache_group_specs:
            raise ValueError(
                "cache arena requires at least one cache group spec to publish"
            )
        self.plan = plan
        self.device = device
        # One adapter for the whole arena. The allocation happens inside its
        # region, so the sleep/wake lifetime is a property of the arena.
        # Tag as "kv_cache", no CPU backup: cache bytes are discarded on
        # sleep and rebuilt after wake (paging overwrites; clear() zeros the
        # remapped pages).
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        with self.memory_saver_adapter.region(tag="kv_cache", enable_cpu_backup=False):
            self.buffer = torch.zeros(
                plan.arena_bytes,
                dtype=torch.uint8,
                device=device,
            )

        # The plan names every field and its dtype
        self._fields: dict[str, torch.Tensor] = {
            field.field_id: self._bind(field) for field in plan.fields
        }
        # Publish the contract: the recipe's logical specs joined with the
        # plan's physical facts for the same groups. Nothing is copied into
        # the specs -- the plan owns page counts and LCM packing, the contract
        # carries them beside the specs, so the two never need reconciling.
        # That the two name the same groups needs no assertion here: the
        # recipe packs the plan from the very (spec, fields) pairs it publishes
        # these specs from, so one group id cannot reach only one of them.
        plan_groups = {group.group_id: group for group in plan.groups}
        self.runtime_contract = CacheRuntimeContract(
            # The identity axis comes from the plan, never read back out of
            # view state. Per-group CacheBlock spans live in the group specs
            # as block_granularity.
            prefix_granularity=plan.prefix_granularity,
            num_lcm_blocks=plan.num_lcm_blocks,
            token_capacity=(
                token_capacity if token_capacity is not None else self.size
            ),
            group_specs=tuple(cache_group_specs),
            group_page_counts={
                spec.group_id: plan_groups[spec.group_id].page_count
                for spec in cache_group_specs
            },
            group_packing={
                spec.group_id: plan_groups[spec.group_id].cache_blocks_per_lcm_block
                for spec in cache_group_specs
            },
        )
        logger.info(
            "Allocated cache arena: %d bytes, prefix_granularity=%d, num_lcm_blocks=%d, device %s",
            plan.arena_bytes,
            plan.prefix_granularity,
            plan.num_lcm_blocks,
            device,
        )

    @property
    def cache_group_specs(self) -> tuple[CacheGroupSpec, ...]:
        """The published group specs. Stored once, in the contract."""
        return self.runtime_contract.group_specs

    @property
    def cache_group_page_counts(self) -> Mapping[str, int]:
        """Per-group CacheBlock counts. Stored once, in the contract."""
        return self.runtime_contract.group_page_counts

    @property
    def prefix_granularity(self) -> int:
        """The identity grain, from the plan that defined this arena."""
        return self.plan.prefix_granularity

    @property
    def kv_page_size(self) -> int:
        """KV arena page span for paged consumers.

        Pinned 1:1 to the identity grain -- this property is the single point
        of the prefix-page <-> KV-page convention, and geometry math must read
        it rather than ``prefix_granularity``.
        """
        return self.plan.prefix_granularity

    @property
    def size(self) -> int:
        """Token slots the most finely packed group addresses.

        The plan's parent count times the tightest packing. One definition
        here, so compute views no longer re-derive it and disagree.
        """
        max_packing = max(
            group.cache_blocks_per_lcm_block for group in self.plan.groups
        )
        return self.plan.num_lcm_blocks * max_packing * self.plan.prefix_granularity

    def _bind(self, field) -> torch.Tensor:
        """Stride one planned field's view over the arena bytes.

        Pages are the physical unit, so the stride math is page-major; a
        per-token field then folds that axis into its entries (see
        :meth:`field` for why the planned shape decides).
        """
        dtype = getattr(torch, field.dtype)
        group = self.plan.group(field.group_id)
        element_strides = []
        stride = 1
        for extent in reversed(field.shape):
            element_strides.append(stride)
            stride *= extent
        pages = self.buffer.view(dtype).as_strided(
            (group.page_count, *field.shape),
            (
                field.page_stride_bytes // field.element_size,
                *reversed(element_strides),
            ),
            self.field_block_byte_offset(field.field_id, 0) // field.element_size,
        )
        if field.shape[0] != self.plan.prefix_granularity:
            return pages
        return pages.view(-1, *field.shape[1:])

    def field(self, field_id: str) -> torch.Tensor:
        """Return one planned field's view into the arena.

        The view is shaped the way the field is addressed. A field whose
        planned shape leads with the identity grain stores one entry per
        token, so its page axis is folded away and consumers index entries
        directly; every other field (recurrent state, block-scaled scale
        planes, V4's byte-shaped planes) is addressed per page and keeps its
        planned shape. The plan decides which, so no compute view restates
        its kernel's geometry, and the dtype likewise comes from the plan.

        Args:
            field_id: A field id named by the memory plan.

        Returns:
            The field's view over its arena bytes.

        Raises:
            ValueError: The plan does not name this field.
        """
        try:
            return self._fields[field_id]
        except KeyError:
            raise ValueError(f"cache field {field_id!r} is not planned") from None

    def field_ids(self) -> frozenset[str]:
        """Every field the plan names, all of them materialized."""
        return frozenset(self._fields)

    def field_block_byte_offset(self, field_id: str, block_id: int) -> int:
        return self.plan.field_page_byte_offset(field_id, block_id)

    def zero_blocks(self, block_ids_by_group: dict[str, list[int]]) -> None:
        """Clear selected CacheBlocks without interpreting their field types."""
        segments = [
            segment
            for group_id, block_ids in block_ids_by_group.items()
            for segment in self.block_byte_segments(group_id, block_ids)
        ]
        if segments:
            zero_byte_ranges(self.buffer, segments)

    def block_byte_segments(
        self, group_id: str, block_ids: list[int]
    ) -> list[tuple[int, int]]:
        self.plan.group(group_id)
        fields = [field for field in self.plan.fields if field.group_id == group_id]
        return [
            (
                self.field_block_byte_offset(field.field_id, block_id),
                field.payload_bytes,
            )
            for block_id in block_ids
            for field in fields
        ]

    @property
    def supports_disaggregation(self) -> bool:
        """Whether the arena exposes complete inputs for cache transfer."""
        return all(spec.transfer_policy is not None for spec in self.cache_group_specs)

    def contract_binding(self) -> torch.Tensor:
        """Return the arena buffer the cache contract is bound to.

        Field dtypes are not returned: the plan carries them, and the PD
        contract reads them from there.
        """
        if not self.supports_disaggregation:
            raise RuntimeError(
                "cache arena has no transfer policy in its runtime contract"
            )
        return self.buffer

    @torch.no_grad()
    def clear(self) -> None:
        """Zero the arena after sleep/wake remaps its storage.

        Exactly one owner, so callers may fan this out over every compute
        view without double-clearing or needing a "views skip" rule.
        """
        self.buffer.zero_()
