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

"""Mamba2 hybrid cache recipe: attention KV and SSD recurrent state."""

from __future__ import annotations

import math
from functools import cached_property

import torch
from typing_extensions import override

from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    scatter_stored_dtype_name,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.qwen35 import (
    QwenGDNRecipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    STATE_LAYER_TYPES,
)

# Two KV groups give each as many K and V pages as a state group has SSM states.
_KV_GROUPS = 2


class Mamba2Recipe(QwenGDNRecipe):
    """Mamba2 blocks interleaved with MHA, sharing one cache arena.

    Every group's page spans the whole parent block, so a parent is only fully
    used when all groups fill the same segments. A state layer brings one large
    SSM state and a small conv window; an attention layer brings two
    equal-sized K and V pages. The layout matches them:

    * state layers are dealt round-robin into groups of about ``2 * n``
      layers, and the attention layers into ``_KV_GROUPS`` groups of ``n``;
    * segment ``unit.i`` holds each state group's ``i``-th SSM state and each
      KV group's ``i``-th K or V field, K and V of one layer in adjacent
      segments, so K/V pages pack exactly to the SSM state's size;
    * each conv window has a segment of its own, which the KV groups skip.

    State pages then carry no padding and KV pages only the conv segments.
    Grouping state layers by their position between attention layers, as Qwen
    does, fails here: Nemotron-3 Super's Mamba2 runs are four or five long.
    Draft (MTP) attention layers continue the KV round-robin, so they pad
    every other group by their K/V segments, as Qwen's drafts do.
    """

    family = "mamba2"

    @cached_property
    def group_ids(self) -> tuple[str, ...]:
        labels = self.layer_types
        num_full = sum(label == FULL_ATTENTION for label in self.target_layer_types)
        num_state = sum(label in STATE_LAYER_TYPES for label in labels)
        num_state_groups = max(1, math.ceil(num_state / max(num_full, 1)))
        group_ids = []
        state_index = full_index = 0
        for label in labels:
            if label in STATE_LAYER_TYPES:
                group_ids.append(f"{label}_{state_index % num_state_groups}")
                state_index += 1
            else:
                group_ids.append(f"{label}_{full_index % _KV_GROUPS}")
                full_index += 1
        return tuple(group_ids)

    def _replay_commit_supported(self, dtype: torch.dtype) -> bool:
        from tokenspeed_kernel.ops.attention.mamba2 import (
            mamba2_replay_commit_supported,
        )

        return mamba2_replay_commit_supported(dtype)

    @override
    def fields_for_layer(
        self, layer_id: int, group_id: str, occurrence: int
    ) -> tuple[CacheFieldSpec, ...]:
        if self.layer_types[layer_id] == LINEAR_ATTENTION:
            conv_shape, conv_dtype, ssm_shape, ssm_dtype = self._state_shapes
            return (
                CacheFieldSpec(
                    f"layer.{layer_id}.ssm", f"unit.{occurrence}", ssm_shape, ssm_dtype
                ),
                CacheFieldSpec(
                    f"layer.{layer_id}.conv",
                    f"conv.{occurrence}",
                    conv_shape,
                    conv_dtype,
                    exact_page_stride=False,
                ),
            )
        is_draft = layer_id >= len(self.target_layer_types)
        config = self.draft_attn_config if is_draft else self.attn_config
        if config.kv_cache_mxfp8:
            raise NotImplementedError("Mamba2 hybrids do not store MXFP8 KV scales")
        kv_shape = self._draft_kv_shape if is_draft else self._kv_shape
        kv_dtype = scatter_stored_dtype_name(config.kv_cache_dtype)
        return (
            CacheFieldSpec(
                f"layer.{layer_id}.k", f"unit.{2 * occurrence}", kv_shape, kv_dtype
            ),
            CacheFieldSpec(
                f"layer.{layer_id}.v", f"unit.{2 * occurrence + 1}", kv_shape, kv_dtype
            ),
        )
