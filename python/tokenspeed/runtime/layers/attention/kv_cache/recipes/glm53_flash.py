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

"""GLM-5.3-Flash cache recipe: DSA history plus paged KDA checkpoints."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property

import torch
from typing_extensions import override

from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.base import (
    CacheGroupDeclaration,
    CacheRecipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    require_positive_int,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    CacheLayout,
    cache_dtype_name,
    scatter_stored_dtype_name,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    group,
    split_recurrent_state_groups,
)

GLM53_FLASH_LOGICAL_BLOCK_TOKENS = 64

# Full-attention pages per LCM parent, keyed by (attn tp size, MLA cache
# element size). The compressed KPool index is a companion field of the same
# 64-token CacheBlock and therefore shares this packing and block table. KDA
# state groups always pack one checkpoint per parent; the bounded raw KPool
# tail is request-local workspace and has no scheduler-facing packing.
_PACKING = {
    (1, 2): 72,
    (1, 1): 144,
    (2, 2): 36,
    (2, 1): 72,
    (4, 2): 18,
    (4, 1): 36,
    (8, 2): 9,
    (8, 1): 18,
}


@dataclass(frozen=True)
class Glm53FlashPoolOptions:
    """Fixed request-local KPool tail geometry owned by the GLM cache pool."""

    index_kpool: int
    tail_extra_slots: int
    index_head_dim: int
    num_request_slots: int
    dsa_layer_ids: tuple[int, ...]

    @property
    def tail_width(self) -> int:
        return self.index_kpool + self.tail_extra_slots

    @property
    def workspace_bytes(self) -> int:
        return (
            len(self.dsa_layer_ids)
            * 2
            * self.num_request_slots
            * self.tail_width
            * self.index_head_dim
            * torch.bfloat16.itemsize
        )


def _require_non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def glm53_flash_packing_counts(
    *,
    tp_size: int,
    mla_element_size: int,
    state_group_ids: Sequence[str] = (),
) -> dict[str, int]:
    """Look up the LCM packing for one GLM-5.3-Flash parallel/dtype shape."""
    try:
        full = _PACKING[(int(tp_size), int(mla_element_size))]
    except KeyError:
        raise NotImplementedError(
            f"GLM-5.3-Flash has no verified LCM packing for tp_size={tp_size} "
            f"with a {mla_element_size}-byte MLA cache; supported shapes are "
            f"{sorted(_PACKING)}"
        ) from None
    packing = {FULL_ATTENTION: full}
    packing.update(dict.fromkeys(state_group_ids, 1))
    return packing


def glm53_flash_parents_needed(
    layout: CacheLayout,
    *,
    token_capacity: int,
    max_scheduled_tokens: int,
    max_live_requests: int,
    decode_input_tokens: int = 1,
    overlap_schedule_depth: int = 0,
) -> int:
    """Physical parents needed at the configured concurrency."""
    require_positive_int("token_capacity", token_capacity)
    _require_non_negative_int("max_scheduled_tokens", max_scheduled_tokens)
    require_positive_int("max_live_requests", max_live_requests)
    _require_non_negative_int("decode_input_tokens", decode_input_tokens)
    if overlap_schedule_depth not in (0, 1):
        raise ValueError(
            f"overlap_schedule_depth must be 0 or 1, got {overlap_schedule_depth}"
        )
    if overlap_schedule_depth and decode_input_tokens == 0:
        raise ValueError("overlapped cache sizing requires decode_input_tokens > 0")

    page_tokens = layout.prefix_granularity
    protected_pages = max_live_requests * math.ceil(
        overlap_schedule_depth * decode_input_tokens / page_tokens
    )
    scheduled_pages = math.ceil(min(max_scheduled_tokens, token_capacity) / page_tokens)
    parents = 0
    for group_id, packing in layout.group_packing:
        if group_id == FULL_ATTENTION:
            child_pages = (
                math.ceil(token_capacity / page_tokens)
                + max_live_requests
                - 1
                + protected_pages
            )
        else:
            child_pages = max_live_requests + scheduled_pages + protected_pages
        parents += math.ceil(child_pages / packing)
    return parents


def _require_dsa_config(config: AttnConfig, role: str) -> DSAConfig:
    spec = config.component(DSAConfig)
    if spec is None:
        raise TypeError(f"GLM-5.3-Flash {role} cache requires a DSA component")
    return spec


def _require_linear_config(config: AttnConfig) -> LinearAttnConfig:
    spec = config.component(LinearAttnConfig)
    if spec is None:
        raise TypeError(
            "GLM-5.3-Flash target cache requires a linear-attention component"
        )
    return spec


def _target_layer_types(config: AttnConfig, num_layers: int) -> tuple[str, ...]:
    num_layers = require_positive_int("num_attention_layers", num_layers)
    linear_layer_ids = _require_linear_config(config).layer_ids
    if len(linear_layer_ids) != len(set(linear_layer_ids)) or any(
        layer_id < 0 or layer_id >= num_layers for layer_id in linear_layer_ids
    ):
        raise ValueError(
            "GLM-5.3-Flash linear-attention layer ids must be unique and "
            f"inside [0, {num_layers})"
        )
    linear_layer_id_set = set(linear_layer_ids)
    return tuple(
        LINEAR_ATTENTION if layer_id in linear_layer_id_set else FULL_ATTENTION
        for layer_id in range(num_layers)
    )


def declare_glm53_flash_groups(
    num_target_layers: int,
    *,
    attn_config: AttnConfig,
    draft_attn_config: AttnConfig | None = None,
    draft_layers: int = 0,
) -> tuple[CacheGroupDeclaration, ...]:
    """Declare GLM cache groups from its composite attention components."""
    if attn_config.pd_disaggregation_enabled:
        raise NotImplementedError(
            "GLM-5.3-Flash disaggregated serving requires an explicit transfer "
            "bridge for its request-local KPool tail"
        )
    if draft_layers and draft_attn_config is None:
        raise TypeError("GLM-5.3-Flash draft layers require a draft attention config")

    target_dsa = _require_dsa_config(attn_config, "target")
    target_linear = _require_linear_config(attn_config)
    target_layer_types = _target_layer_types(attn_config, num_target_layers)
    group_ids = (
        tuple(split_recurrent_state_groups(target_layer_types))
        + (FULL_ATTENTION,) * draft_layers
    )
    layer_types = tuple(
        FULL_ATTENTION if group_id == FULL_ATTENTION else LINEAR_ATTENTION
        for group_id in group_ids
    )
    target_layers = len(target_layer_types)

    def fields_for_layer(
        layer_id: int, group_id: str, occurrence: int
    ) -> tuple[CacheFieldSpec, ...]:
        plane_id = f"slot.{occurrence}"
        if group_id == FULL_ATTENTION:
            config = attn_config if layer_id < target_layers else draft_attn_config
            if config is None:
                raise TypeError(
                    "GLM-5.3-Flash draft cache field requires a draft attention config"
                )
            dsa = _require_dsa_config(
                config, "target" if layer_id < target_layers else "draft"
            )
            return (
                CacheFieldSpec(
                    f"layer.{layer_id}.latent_kv",
                    plane_id,
                    (
                        GLM53_FLASH_LOGICAL_BLOCK_TOKENS,
                        1,
                        dsa.kv_lora_rank + dsa.qk_rope_head_dim,
                    ),
                    scatter_stored_dtype_name(config.kv_cache_dtype),
                ),
            )
        return (
            CacheFieldSpec(
                f"layer.{layer_id}.conv_state",
                plane_id,
                target_linear.conv_state_shape,
                cache_dtype_name(torch.bfloat16),
                exact_page_stride=False,
            ),
            CacheFieldSpec(
                f"layer.{layer_id}.recurrent_state",
                plane_id,
                target_linear.temporal_state_shape,
                cache_dtype_name(torch.float32),
                exact_page_stride=False,
            ),
        )

    base = group(
        layer_types=layer_types,
        group_ids=group_ids,
        sliding_window_tokens=target_dsa.sliding_window_tokens,
        prefix_granularity=GLM53_FLASH_LOGICAL_BLOCK_TOKENS,
        fields_for_layer=fields_for_layer,
        pd_disaggregation_enabled=attn_config.pd_disaggregation_enabled,
    )
    kpool = require_positive_int("index_kpool", target_dsa.index_kpool)
    index_head_dim = require_positive_int("index_head_dim", target_dsa.index_head_dim)
    pooled_rows = GLM53_FLASH_LOGICAL_BLOCK_TOKENS // kpool
    index_fields = []
    index_plane_id = f"slot.{sum(gid == FULL_ATTENTION for gid in group_ids)}"
    for layer_id, group_id in enumerate(group_ids):
        if group_id != FULL_ATTENTION:
            continue
        index_fields.append(
            CacheFieldSpec(
                f"layer.{layer_id}.index_k",
                # Keep all flexible-stride index fields in the first plane
                # not occupied by this group's exact-stride MLA pages. On the
                # target-only topology that plane aliases otherwise-unused KDA
                # slab space; a merged target+draft plan adds it explicitly.
                index_plane_id,
                (pooled_rows, index_head_dim + 4),
                cache_dtype_name(torch.uint8),
                exact_page_stride=False,
            )
        )
    return tuple(
        (
            spec,
            fields + tuple(index_fields) if spec.group_id == FULL_ATTENTION else fields,
        )
        for spec, fields in base
    )


class Glm53FlashRecipe(CacheRecipe):
    """DSA history plus KDA checkpoints and request-local KPool tails."""

    family = "glm53_flash"

    @cached_property
    def _dsa_config(self) -> DSAConfig:
        return _require_dsa_config(self.attn_config, "target")

    @cached_property
    def target_layer_types(self) -> tuple[str, ...]:
        return _target_layer_types(
            self.attn_config, self.model_config.num_attention_layers
        )

    @cached_property
    def target_group_ids(self) -> tuple[str, ...]:
        return tuple(split_recurrent_state_groups(self.target_layer_types))

    @cached_property
    def group_ids(self) -> tuple[str, ...]:
        return self.target_group_ids + (FULL_ATTENTION,) * self.num_draft_layers

    @cached_property
    def layer_types(self) -> tuple[str, ...]:
        return tuple(
            FULL_ATTENTION if group_id == FULL_ATTENTION else LINEAR_ATTENTION
            for group_id in self.group_ids
        )

    @property
    @override
    def prefix_granularity(self) -> int:
        return GLM53_FLASH_LOGICAL_BLOCK_TOKENS

    @property
    @override
    def max_padding_fraction(self) -> float:
        return float("inf") if self.num_draft_layers else 0.25

    @cached_property
    def tail_extra_slots(self) -> int:
        if getattr(self.server_args, "speculative_algorithm", None) is None:
            return 0
        return _require_non_negative_int(
            "speculative_num_draft_tokens",
            int(getattr(self.server_args, "speculative_num_draft_tokens", 0) or 0),
        )

    @override
    def groups(self) -> tuple[CacheGroupDeclaration, ...]:
        return declare_glm53_flash_groups(
            self.num_target_layers,
            attn_config=self.attn_config,
            draft_attn_config=self.draft_attn_config,
            draft_layers=self.num_draft_layers,
        )

    @override
    def packing(self, groups: tuple[CacheGroupDeclaration, ...]) -> Mapping[str, int]:
        state_group_ids = [
            spec.group_id
            for spec, _ in groups
            if spec.group_id.startswith(LINEAR_ATTENTION)
        ]
        return glm53_flash_packing_counts(
            tp_size=self._dsa_config.attn_tp_size,
            mla_element_size=self.attn_config.kv_cache_dtype.itemsize,
            state_group_ids=state_group_ids,
        )

    @override
    def workspace_bytes(self) -> int:
        """Bounded raw KPool tails kept once per request and DSA layer."""
        return self.pool_options().workspace_bytes

    @override
    def pool_options(self) -> Glm53FlashPoolOptions:
        max_bs = self.attn_config.max_bs
        if self.draft_attn_config is not None:
            max_bs = max(max_bs, self.draft_attn_config.max_bs)
        return Glm53FlashPoolOptions(
            index_kpool=require_positive_int(
                "index_kpool", self._dsa_config.index_kpool
            ),
            tail_extra_slots=self.tail_extra_slots,
            index_head_dim=require_positive_int(
                "index_head_dim", self._dsa_config.index_head_dim
            ),
            # Scheduler request IDs are 1-based; row 0 and one graph-padding
            # sentinel sit outside the live range.
            num_request_slots=max_bs + 2,
            dsa_layer_ids=tuple(
                layer_id
                for layer_id, group_id in enumerate(self.group_ids)
                if group_id == FULL_ATTENTION
            ),
        )

    @override
    def check_layout(self, layout: CacheLayout) -> None:
        group_counts = {
            label: sum(1 for group_id in self.group_ids if group_id == label)
            for label in dict.fromkeys(self.group_ids)
        }
        # Index fields use the slot immediately after the last MLA field.  A
        # target-only layout aliases that slot with the wider KDA state slab;
        # adding a draft MLA layer consumes the alias and needs one more plane.
        expected = max(max(group_counts.values()), group_counts[FULL_ATTENTION] + 1)
        if len(layout.plane_bytes) != expected:
            raise ValueError(
                f"GLM-5.3-Flash LCM requires {expected} planes, got "
                f"{len(layout.plane_bytes)}"
            )

    @override
    def num_lcm_blocks(self, layout: CacheLayout) -> int:
        usable_bytes = self.cache_budget_bytes - self.workspace_bytes()
        num_lcm_blocks = usable_bytes // layout.lcm_block_bytes - 1
        if num_lcm_blocks < 1:
            raise ValueError(
                f"{self.family} cache budget must hold a null parent and one "
                "usable LCM parent"
            )
        token_limit = self.token_limit
        if token_limit is None:
            return num_lcm_blocks
        return min(num_lcm_blocks, self.parents_needed(layout, token_limit))

    @override
    def token_capacity(self, layout: CacheLayout, num_lcm_blocks: int) -> int:
        upper = self.token_limit
        if upper is None:
            full_packing = dict(layout.group_packing)[FULL_ATTENTION]
            upper = num_lcm_blocks * full_packing * layout.prefix_granularity
        return self._capacity_from_parents(layout, num_lcm_blocks, upper_bound=upper)

    @override
    def parents_needed(self, layout: CacheLayout, token_capacity: int) -> int:
        limits = self.scheduler_limits
        return glm53_flash_parents_needed(
            layout,
            token_capacity=token_capacity,
            max_scheduled_tokens=limits["max_scheduled_tokens"],
            max_live_requests=limits["max_live_requests"],
            decode_input_tokens=limits["decode_input_tokens"],
            overlap_schedule_depth=limits["overlap_schedule_depth"],
        )
