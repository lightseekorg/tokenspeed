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

"""Cache setup results and model-recipe dispatch."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch

from tokenspeed.runtime.layers.attention.configs.base import BaseAttnConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.deepseek_v4 import (
    prepare_deepseek_v4_cache,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.inkling import (
    prepare_inkling_cache,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.kimi_k3 import (
    prepare_kimi_k3_cache,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import (
    prepare_ordinary_cache,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import CacheMemoryPlan
from tokenspeed.runtime.layers.attention.kv_cache.recipes.qwen35 import (
    prepare_qwen35_cache,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    PagedCacheGroupSpec,
)

CacheModelFamily = Literal[
    "mha",
    "mla",
    "dsa",
    "msa",
    "qwen_gdn",
    "inkling",
    "kimi_k3",
    "deepseek_v4",
]


@dataclass(frozen=True)
class CachePoolSpec:
    """Everything needed to bind one model's compute views to a cache buffer."""

    family: CacheModelFamily
    memory_plan: CacheMemoryPlan
    layer_types: tuple[str, ...]
    layer_group_ids: tuple[str, ...]
    # Scheduler group specs, computed once by the recipe. The pool aligns
    # their physical fields (packing) with the memory plan and publishes the
    # runtime contract from the pair.
    paged_cache_group_specs: tuple[PagedCacheGroupSpec, ...]
    state_field_dtypes: Mapping[str, torch.dtype]
    token_capacity: int
    layer_kv_head_counts: tuple[int, ...] | None = None
    pool_options: object | None = None

    @property
    def pool_size(self) -> int:
        max_packing = max(
            group.cache_blocks_per_lcm_block for group in self.memory_plan.groups
        )
        return (
            self.memory_plan.num_lcm_blocks
            * max_packing
            * self.memory_plan.logical_block_tokens
        )


@dataclass(frozen=True)
class CacheSetup:
    """One big model, one spec: target and draft layers share everything.

    Draft layers are continuation layers of the one merged model (global
    layer ids ``num_target_layers..``, the DeepSeek-V4 MTP convention
    generalized): one plan, one arena, one contract, one pool.
    ``num_draft_layers`` is the only draft-specific fact, consumed at
    model-runner wiring time (a draft model's local layer ``i`` maps to
    global layer ``num_target_layers + i``); the spec itself is
    draft-oblivious.
    """

    spec: CachePoolSpec
    num_draft_layers: int
    cache_budget_bytes: int
    fixed_workspace_bytes: int

    @property
    def num_target_layers(self) -> int:
        return len(self.spec.layer_group_ids) - self.num_draft_layers


_PREPARE_CACHE = {
    "mha": partial(prepare_ordinary_cache, family="mha"),
    "mla": partial(prepare_ordinary_cache, family="mla"),
    "dsa": partial(prepare_ordinary_cache, family="dsa"),
    "msa": partial(prepare_ordinary_cache, family="msa"),
    "qwen_gdn": prepare_qwen35_cache,
    "inkling": prepare_inkling_cache,
    "kimi_k3": prepare_kimi_k3_cache,
    "deepseek_v4": prepare_deepseek_v4_cache,
}


def prepare_cache_setup(
    *,
    family: CacheModelFamily,
    server_args,
    model_config,
    attn_config: BaseAttnConfig,
    draft_model_config,
    draft_attn_config: BaseAttnConfig | None,
    cache_budget_bytes: int,
    decode_input_tokens: int,
    overlap_schedule_depth: int,
) -> CacheSetup:
    """Apply one model recipe and size target/draft arenas from one budget."""
    prepare = _PREPARE_CACHE.get(family)
    if prepare is None:
        raise ValueError(f"unsupported cache model family: {family}")
    return prepare(
        server_args=server_args,
        model_config=model_config,
        attn_config=attn_config,
        draft_model_config=draft_model_config,
        draft_attn_config=draft_attn_config,
        cache_budget_bytes=cache_budget_bytes,
        decode_input_tokens=decode_input_tokens,
        overlap_schedule_depth=overlap_schedule_depth,
    )
