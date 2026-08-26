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

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from typing import Literal

from tokenspeed.runtime.layers.attention.configs.base import BaseAttnConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.base import CacheRecipe
from tokenspeed.runtime.layers.attention.kv_cache.recipes.deepseek_v4 import (
    DeepseekV4Recipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.glm53_flash import (
    Glm53FlashRecipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.inkling import (
    InklingRecipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.kimi_k3 import (
    KimiK3Recipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import (
    OrdinaryRecipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import CacheMemoryPlan
from tokenspeed.runtime.layers.attention.kv_cache.recipes.qwen35 import (
    QwenGDNRecipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    CacheGroupSpec,
)

CacheModelFamily = Literal[
    "mha",
    "mla",
    "dsa",
    "msa",
    "qwen_gdn",
    "inkling",
    "kimi_k3",
    "glm53_flash",
    "deepseek_v4",
]


@dataclass(frozen=True)
class CachePoolSpec:
    """Everything needed to bind one model's compute views to a cache buffer."""

    family: CacheModelFamily
    memory_plan: CacheMemoryPlan
    layer_types: tuple[str, ...]
    # Scheduler group specs, declared next to the fields the plan was packed
    # from (CacheRecipe.groups), so the plan and the specs name one group set.
    cache_group_specs: tuple[CacheGroupSpec, ...]
    token_capacity: int
    layer_kv_head_counts: tuple[int, ...] | None = None
    pool_options: object | None = None

    def layer_view(
        self,
        *,
        first_layer: int,
        num_layers: int,
        family: CacheModelFamily | None = None,
    ) -> CachePoolSpec:
        """Describe one concrete compute view over this spec's shared arena.

        The memory plan and scheduler geometry stay merged, group specs
        included: publication is not a view's concern. Only per-layer compute
        metadata is sliced. The arena publishes the contract once, from the
        merged spec, so no view can republish or diverge from it.
        """
        total_layers = len(self.layer_types)
        if first_layer < 0 or num_layers < 0:
            raise ValueError("cache layer view bounds must be non-negative")
        last_layer = first_layer + num_layers
        if last_layer > total_layers:
            raise ValueError(
                f"cache layer view [{first_layer}, {last_layer}) exceeds "
                f"the merged {total_layers}-layer spec"
            )
        if self.layer_kv_head_counts is not None and (
            len(self.layer_kv_head_counts) != total_layers
        ):
            raise ValueError("cache KV head counts must cover every layer")
        # A family whose pool options are themselves per-layer (V4's kernel
        # cache layout) narrows them to the same window; anything else is a
        # whole-pool fact and travels unchanged.
        narrow_options = getattr(self.pool_options, "layer_view", None)
        return replace(
            self,
            family=family or self.family,
            layer_types=self.layer_types[first_layer:last_layer],
            layer_kv_head_counts=(
                self.layer_kv_head_counts[first_layer:last_layer]
                if self.layer_kv_head_counts is not None
                else None
            ),
            pool_options=(
                narrow_options(first_layer=first_layer, num_layers=num_layers)
                if narrow_options is not None
                else self.pool_options
            ),
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
        return len(self.spec.layer_types) - self.num_draft_layers


# family -> how to build its recipe. Every family runs the one pipeline in
# CacheRecipe.setup(); the class only fills in that family's seams, and the
# four ordinary families differ by nothing but the family label.
_RECIPES: dict[CacheModelFamily, Callable[..., CacheRecipe]] = {
    "mha": partial(OrdinaryRecipe, family="mha"),
    "mla": partial(OrdinaryRecipe, family="mla"),
    "dsa": partial(OrdinaryRecipe, family="dsa"),
    "msa": partial(OrdinaryRecipe, family="msa"),
    "qwen_gdn": QwenGDNRecipe,
    "inkling": InklingRecipe,
    "kimi_k3": KimiK3Recipe,
    "glm53_flash": Glm53FlashRecipe,
    "deepseek_v4": DeepseekV4Recipe,
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
    recipe = _RECIPES.get(family)
    if recipe is None:
        raise ValueError(f"unsupported cache model family: {family}")
    return recipe(
        server_args=server_args,
        model_config=model_config,
        attn_config=attn_config,
        draft_model_config=draft_model_config,
        draft_attn_config=draft_attn_config,
        cache_budget_bytes=cache_budget_bytes,
        decode_input_tokens=decode_input_tokens,
        overlap_schedule_depth=overlap_schedule_depth,
    ).setup()
