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

"""The cache pipeline every model family runs: layers, group, pack, bind."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import TYPE_CHECKING

from tokenspeed.runtime.layers.attention.kv_cache.recipes import (
    configured_token_limit,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    CacheLayout,
    pack,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    CacheGroupSpec,
    compute_cache_group_page_counts,
    group,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import (
        CacheModelFamily,
        CachePlacement,
        CacheSetup,
    )

# One declared cache group: what the scheduler is told, and the bytes it costs.
CacheGroupDeclaration = tuple[CacheGroupSpec, tuple[CacheFieldSpec, ...]]


class CacheRecipe(ABC):
    """One model family's cache recipe.

    The pipeline is the same for every family and lives once, in :meth:`setup`:

    ``layers -> group -> pack -> bind``

    A family fills in the seams -- which layers exist and how they collapse
    into groups, what bytes each layer costs, how tightly groups pack into a
    physical parent, and how many parents the budget affords. It never
    restates the order of the stages.
    """

    # Set as a class attribute by every subclass; OrdinaryRecipe takes it per
    # instance because its four families differ by nothing else.
    family: CacheModelFamily

    def __init__(
        self,
        *,
        server_args,
        model_config,
        attn_config,
        draft_model_config,
        draft_attn_config,
        cache_budget_bytes: int,
        decode_input_tokens: int,
        overlap_schedule_depth: int,
    ) -> None:
        self.server_args = server_args
        self.model_config = model_config
        self.attn_config = attn_config
        self.draft_model_config = draft_model_config
        self.draft_attn_config = draft_attn_config
        self.cache_budget_bytes = cache_budget_bytes
        self.decode_input_tokens = decode_input_tokens
        self.overlap_schedule_depth = overlap_schedule_depth

    # ------------------------------------------------------------------
    # The pipeline. One place, one order.
    # ------------------------------------------------------------------

    def setup(self) -> CacheSetup:
        """Run the pipeline for this family and bind it to the budget."""
        from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import (
            CachePlacementContract,
            CachePoolSpec,
            CacheSetup,
        )

        groups = self.groups()
        layout = pack(
            groups,
            prefix_granularity=self.prefix_granularity,
            cache_blocks_per_lcm_block=self.packing(groups),
            alignment=self.alignment,
            max_padding_fraction=self.max_padding_fraction,
        )
        self.check_layout(layout)
        num_lcm_blocks = self.num_lcm_blocks(layout)
        return CacheSetup(
            spec=CachePoolSpec(
                family=self.family,
                memory_plan=layout.bind(num_lcm_blocks),
                layer_types=self.layer_types
                or (FULL_ATTENTION,) * (self.num_target_layers + self.num_draft_layers),
                # The same declarations the layout was packed from, so plan and
                # specs cannot name different groups.
                cache_group_specs=tuple(spec for spec, _ in groups),
                token_capacity=self.token_capacity(layout, num_lcm_blocks),
                layer_kv_head_counts=self.layer_kv_head_counts,
                pool_options=self.pool_options(),
                placement_contract=CachePlacementContract(
                    dcp_size=int(getattr(self.attn_config, "dcp_size", 1)),
                    dcp_rank=int(getattr(self.attn_config, "dcp_rank", 0)),
                    layer_placements=self.layer_placements,
                ),
            ),
            num_draft_layers=self.num_draft_layers,
            cache_budget_bytes=self.cache_budget_bytes,
            fixed_workspace_bytes=self.workspace_bytes(),
        )

    # ------------------------------------------------------------------
    # Seams: the layer vocabulary
    #
    # Subclasses mark every seam they fill with @override, so a renamed or
    # mistyped seam fails type checking instead of silently falling back to
    # the default below. The abstract seam is the exception: ABC already
    # refuses to instantiate a subclass that misses it.
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def layer_types(self) -> tuple[str, ...]:
        """Per-layer cache-group label, target layers then draft layers."""

    @property
    def group_ids(self) -> tuple[str, ...]:
        """Per-layer cache group id, target layers then draft layers.

        Only the default per-layer :meth:`groups` walk consumes this seam:
        it is the storage policy "layer i's fields go to group_ids[i]".
        Families that override :meth:`groups` wholesale express that policy
        directly in their field declarations and need not implement it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} uses the default per-layer groups() walk "
            "but does not define group_ids"
        )

    @property
    def layer_placements(self) -> tuple[CachePlacement, ...]:
        """Per-layer DCP storage placement, target layers then draft layers.

        The default follows the per-layer group vocabulary. Families that
        declare their groups wholesale can override this seam independently
        without manufacturing ``group_ids`` that do not describe their
        storage layout.
        """
        dcp_size = int(getattr(self.attn_config, "dcp_size", 1))
        return tuple(
            (
                "cyclic_history"
                if (
                    layer_id < self.num_target_layers
                    and dcp_size > 1
                    and group_id == FULL_ATTENTION
                )
                else "replicated"
            )
            for layer_id, group_id in enumerate(self.group_ids)
        )

    @property
    def num_target_layers(self) -> int:
        """Leading layers that belong to the target model."""
        configured = getattr(self.model_config, "num_attention_layers", None)
        if configured is not None:
            return int(configured)
        # Lightweight recipe wrappers and synthetic layouts may not carry a
        # full ModelConfig.  Their merged layer list still defines the target
        # prefix exactly: target layers first, then draft continuation layers.
        return len(self.layer_types) - self.num_draft_layers

    @property
    def num_draft_layers(self) -> int:
        """Trailing layers that belong to the draft model, zero without one.

        A draft's layers are continuation layers of the merged plan, so their
        count is a fact of the draft config -- override this only to constrain
        it further, never to restate it.
        """
        if self.draft_attn_config is None:
            return 0
        return self.draft_model_config.num_attention_layers

    @property
    def layer_kv_head_counts(self) -> tuple[int, ...] | None:
        """Per-layer KV head count, when a family's layers differ."""
        return None

    # ------------------------------------------------------------------
    # Seams: geometry knobs
    # ------------------------------------------------------------------

    @property
    def prefix_granularity(self) -> int:
        """Scheduler-wide identity grain in tokens."""
        configured = int(self.attn_config.prefix_granularity)
        dcp_size = int(getattr(self.attn_config, "dcp_size", 1))
        if dcp_size == 1:
            return configured
        kernel_page_size = int(
            getattr(self.attn_config, "kernel_page_size", None) or 64
        )
        return math.lcm(configured, kernel_page_size * dcp_size)

    @property
    def alignment(self) -> int:
        """Byte alignment for plane sizes."""
        return 256

    @property
    def max_padding_fraction(self) -> float:
        """Padding budget a group may waste inside its parent."""
        return 0.25

    @cached_property
    def token_limit(self) -> int | None:
        """The configured token cap, if any: --max-total-tokens or the CI cap."""
        return configured_token_limit(self.server_args)

    @property
    def pd_disaggregation_enabled(self) -> bool:
        return bool(getattr(self.attn_config, "pd_disaggregation_enabled", False))

    # ------------------------------------------------------------------
    # Seams: groups
    # ------------------------------------------------------------------

    def fields_for_layer(
        self, layer_id: int, group_id: str, occurrence: int
    ) -> tuple[CacheFieldSpec, ...]:
        """The bytes this layer costs its group.

        ``occurrence`` is the layer's slot in its group's plane numbering.
        Families whose groups are not per-layer override :meth:`groups`
        instead and need not implement this.
        """
        raise NotImplementedError

    def groups(self) -> tuple[CacheGroupDeclaration, ...]:
        """Declare this family's cache groups, spec and fields together.

        The default walks the layers once (:func:`spec.group`). Families with
        groups that are not per-layer either append whole-group declarations
        here or replace the walk entirely.
        """
        return group(
            layer_types=self.layer_types,
            group_ids=self.group_ids,
            sliding_window_tokens=getattr(
                self.attn_config, "sliding_window_tokens", None
            ),
            prefix_granularity=self.prefix_granularity,
            fields_for_layer=self.fields_for_layer,
            pd_disaggregation_enabled=self.pd_disaggregation_enabled,
        )

    def check_layout(self, layout: CacheLayout) -> None:
        """Assert whatever this family requires of the packed parent.

        Runs once, between pack and bind: the layout is final but nothing has
        been sized or allocated yet.
        """

    def packing(
        self, groups: Sequence[CacheGroupDeclaration]
    ) -> Mapping[str, int] | None:
        """How many of each group's CacheBlocks share one physical parent.

        ``None`` lets :func:`plan.pack` derive it from byte ratios and the
        exact-page-stride constraints the fields impose.
        """
        return None

    # ------------------------------------------------------------------
    # Seams: capacity
    # ------------------------------------------------------------------

    def num_lcm_blocks(self, layout: CacheLayout) -> int:
        """Physical parents to allocate, excluding the reserved null parent.

        The budget minus this family's fixed workspace, capped by the
        configured token limit.
        """
        usable_bytes = self.cache_budget_bytes - self.workspace_bytes()
        return self._capped_parents(
            usable_bytes // layout.lcm_block_bytes - 1,
            parent_tokens=self._max_packing(layout) * layout.prefix_granularity,
        )

    def _capped_parents(self, budgeted: int, *, parent_tokens: int) -> int:
        """Trim a budget-derived parent count to the configured token limit.

        The one place the token limit turns into a parent count, so a family
        that sizes its budget differently still reads the limit the same way.
        """
        if budgeted < 1:
            raise ValueError(
                f"{self.family} cache budget must hold a null parent and one "
                "usable LCM parent"
            )
        if self.token_limit is None:
            return budgeted
        requested = self.token_limit // parent_tokens
        if requested < 1:
            raise ValueError(
                "the configured token limit must hold at least one LCM parent "
                f"({parent_tokens} child tokens)"
            )
        return min(budgeted, requested)

    def token_capacity(self, layout: CacheLayout, num_lcm_blocks: int) -> int:
        """Tokens the scheduler may admit against these parents.

        The one definition of child-token capacity: parents times the tightest
        packing times the identity grain.
        """
        return num_lcm_blocks * self._max_packing(layout) * layout.prefix_granularity

    @cached_property
    def scheduler_limits(self) -> dict[str, int]:
        """The concurrency the cache has to hold at once.

        The one place a recipe reads the scheduler's limits, so per-group page
        demand and the capacity search cannot size against different numbers.
        """
        return {
            "max_live_requests": self.attn_config.max_bs,
            "max_scheduled_tokens": max(0, int(self.server_args.chunked_prefill_size)),
            "max_context_len": self.attn_config.context_len,
            "decode_input_tokens": self.decode_input_tokens,
            "overlap_schedule_depth": self.overlap_schedule_depth,
        }

    def parents_needed(self, layout: CacheLayout, token_capacity: int) -> int:
        """Physical parents this capacity needs at the configured concurrency.

        Reads only capacity-independent facts of the layout, so it can probe a
        layout directly instead of binding a stand-in plan first. Families
        whose per-group demand is not the contract's own page-count formula
        override this.
        """
        counts = compute_cache_group_page_counts(
            self._group_specs,
            max_total_tokens=token_capacity,
            **self.scheduler_limits,
        )
        parents = 0
        for group_id, packing in layout.group_packing:
            child_pages = counts[group_id] - 1  # page 0 is the reserved null page
            parents += (child_pages + packing - 1) // packing
        return parents

    def _capacity_from_parents(
        self, layout: CacheLayout, num_lcm_blocks: int, *, upper_bound: int
    ) -> int:
        """Largest capacity these parents admit, by monotonic binary search.

        The inverse of :meth:`parents_needed`, for families that size parents
        from per-group demand instead of the flat packing product.
        """
        if num_lcm_blocks <= 0:
            raise ValueError("num_lcm_blocks must be positive")
        if upper_bound <= 0:
            raise ValueError("upper_bound must be positive")
        low, high = 0, upper_bound
        while low < high:
            candidate = (low + high + 1) // 2
            if self.parents_needed(layout, candidate) <= num_lcm_blocks:
                low = candidate
            else:
                high = candidate - 1
        if low == 0:
            raise ValueError(
                f"num_lcm_blocks={num_lcm_blocks} cannot admit one token with "
                f"the configured {self.family} cache scheduler limits"
            )
        return low

    @cached_property
    def _group_specs(self) -> tuple[CacheGroupSpec, ...]:
        return tuple(spec for spec, _ in self.groups())

    @staticmethod
    def _max_packing(layout: CacheLayout) -> int:
        """CacheBlocks per parent of the most finely packed group."""
        return max(count for _, count in layout.group_packing)

    # ------------------------------------------------------------------
    # Seams: extras
    # ------------------------------------------------------------------

    def workspace_bytes(self) -> int:
        """Cache-adjacent fixed allocation this family also needs."""
        return 0

    def pool_options(self) -> object | None:
        """Family-specific options the pool constructor needs."""
        return None
