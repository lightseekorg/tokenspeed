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

"""Kimi-K3 cache recipe: MLA latents plus KDA recurrent state groups.

The MLA layers and the KDA state layers live in one arena at P=128. MLA pages
are dense and pin the smallest packing whose plane width covers one per-layer
KDA state (12 at attn tp=8 with the default FP32 state, 6 with BF16); the state
groups pack to match the MLA plane's byte width so no parent is wasted.

Kimi-K3.1 (a DSA target config) adds one index-K plane per MLA target layer
in the full-attention group, beside its latent plane: the index keys share
the latent rows' pages, prefix matches, transfers and lifetime. Index-K
planes hold no KDA state, so they do not change the packing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import cached_property

import torch
from typing_extensions import override

from tokenspeed.runtime.layers.attention.configs.dsa import (
    DSAConfig,
    dsa_history_gather_workspace_bytes,
    dsa_kvp_history_prefill,
)
from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.base import (
    CacheRecipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    require_positive_int,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.kda import (
    kda_recurrent_state_dtype,
    kda_replay_supported,
    kda_verify_workspace_bytes,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import (
    index_k_field,
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
    CacheGroupDeclaration,
)

_KIMI_K3_LAYERS = 93
_KIMI_K3_KDA_LAYERS = 69
_KIMI_K3_MLA_LAYERS = 24
_KIMI_K3_PREFIX_GRANULARITY = 128
_KIMI_K3_STATE_GROUPS = 3


def _one_based_layers(value: object, name: str, num_layers: int) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple of 1-based layer numbers")
    layers = tuple(value)
    if any(isinstance(layer, bool) or not isinstance(layer, int) for layer in layers):
        raise ValueError(f"{name} must contain integer layer numbers")
    if len(layers) != len(set(layers)):
        raise ValueError(f"{name} contains duplicate layer numbers")
    if any(layer < 1 or layer > num_layers for layer in layers):
        raise ValueError(f"{name} contains a layer outside 1..{num_layers}")
    return tuple(sorted(layers))


class KimiK3Recipe(CacheRecipe):
    """Kimi-K3: MLA full-attention layers plus three KDA state groups.

    Draft MLA layers are continuation layers of the one big model and join the
    target's full-attention group, sharing its packing and page-id space.
    """

    family = "kimi_k3"
    uses_paged_state_verify = True

    # ---- layer vocabulary ----

    @property
    def _text_config(self):
        return self.model_config.hf_text_config

    @cached_property
    def target_group_ids(self) -> tuple[str, ...]:
        """Map every target layer to one full-attention or KDA cache group."""
        num_layers = require_positive_int(
            "num_hidden_layers", self._text_config.num_hidden_layers
        )
        linear = self._text_config.linear_attn_config
        if not isinstance(linear, Mapping):
            raise TypeError("linear_attn_config must be a mapping")
        kda_layers = _one_based_layers(
            linear.get("kda_layers"), "linear_attn_config.kda_layers", num_layers
        )
        kda_layer_ids = tuple(layer - 1 for layer in kda_layers)
        kda_layer_id_set = set(kda_layer_ids)
        full_layer_ids = tuple(
            layer_id
            for layer_id in range(num_layers)
            if layer_id not in kda_layer_id_set
        )
        if not kda_layer_ids or not full_layer_ids:
            raise ValueError(
                "Kimi-K3 cache requires both KDA and full-attention layers, got "
                f"{len(kda_layer_ids)} and {len(full_layer_ids)}"
            )
        if num_layers == _KIMI_K3_LAYERS and (
            len(kda_layer_ids) != _KIMI_K3_KDA_LAYERS
            or len(full_layer_ids) != _KIMI_K3_MLA_LAYERS
        ):
            raise ValueError(
                f"{_KIMI_K3_LAYERS}-layer Kimi-K3 requires "
                f"{_KIMI_K3_KDA_LAYERS} KDA and {_KIMI_K3_MLA_LAYERS} MLA "
                f"layers, got {len(kda_layer_ids)} and {len(full_layer_ids)}"
            )
        if len(kda_layer_ids) % _KIMI_K3_STATE_GROUPS:
            raise ValueError(
                "Kimi-K3 cache requires the KDA layer count to divide into "
                f"{_KIMI_K3_STATE_GROUPS} state groups, got {len(kda_layer_ids)}"
            )
        if "full_attn_layers" in linear:
            declared = [
                layer
                for layer in linear["full_attn_layers"]
                if not (isinstance(layer, int) and layer > num_layers)
            ]
            declared_full = _one_based_layers(
                declared,
                "linear_attn_config.full_attn_layers",
                num_layers,
            )
            if declared_full != tuple(layer_id + 1 for layer_id in full_layer_ids):
                raise ValueError(
                    "linear_attn_config.full_attn_layers must equal the "
                    "kda_layers complement"
                )

        group_ids = [FULL_ATTENTION] * num_layers
        per_group = len(kda_layer_ids) // _KIMI_K3_STATE_GROUPS
        for index, layer_id in enumerate(kda_layer_ids):
            group_ids[layer_id] = f"{LINEAR_ATTENTION}_{index // per_group}"
        return tuple(group_ids)

    @property
    @override
    def num_target_layers(self) -> int:
        return len(self.target_group_ids)

    @cached_property
    def group_ids(self) -> tuple[str, ...]:
        return self.target_group_ids + (FULL_ATTENTION,) * self.num_draft_layers

    @cached_property
    def layer_types(self) -> tuple[str, ...]:
        return tuple(
            FULL_ATTENTION if group_id == FULL_ATTENTION else LINEAR_ATTENTION
            for group_id in self.group_ids
        )

    @override
    def groups(self) -> tuple[CacheGroupDeclaration, ...]:
        shard_count = self.attn_config.kvp_size
        return tuple(
            (
                (
                    replace(spec, shard_count=shard_count)
                    if spec.group_id == FULL_ATTENTION
                    else spec
                ),
                fields,
            )
            for spec, fields in super().groups()
        )

    # ---- geometry ----

    @property
    @override
    def prefix_granularity(self) -> int:
        return _KIMI_K3_PREFIX_GRANULARITY

    @property
    @override
    def max_padding_fraction(self) -> float:
        # Draft MLA and K3.1 index-K planes widen the arena past the target-only bound.
        if self.num_draft_layers or self.num_index_k_planes:
            return float("inf")
        return 0.25

    # ---- fields ----

    @cached_property
    def num_index_k_planes(self) -> int:
        """Index-K planes: one per target MLA layer under a DSA target
        config (Kimi-K3.1), none for Kimi-K3. Draft layers never get one."""
        if not isinstance(self.attn_config.component(MLAConfig), DSAConfig):
            return 0
        return sum(group_id == FULL_ATTENTION for group_id in self.target_group_ids)

    @cached_property
    def _kda_shapes(self):
        """(conv shape, recurrent shape) per rank, from the linear component.

        Geometry, TP width, and divisibility validation live on
        ``LinearAttnConfig`` (sourced from ``mapping.linear_attn``); the
        recipe only validates the cache-dtype contract around them.
        """
        cache_dtype = self.attn_config.kv_cache_dtype
        if cache_dtype not in (torch.float8_e4m3fn, torch.bfloat16):
            raise ValueError(
                "Kimi-K3 cache requires mla_cache_dtype in "
                f"{{torch.float8_e4m3fn, torch.bfloat16}}, got {cache_dtype}"
            )
        if self.attn_config.kv_cache_quant_method == "per_token_head":
            raise ValueError("Kimi-K3 cache does not support per_token_head MLA cache")
        if getattr(self._text_config, "mla_use_nope", None) is not True:
            raise ValueError("Kimi-K3 cache requires mla_use_nope=True")
        linear_attn = self.attn_config.component(LinearAttnConfig)
        if linear_attn is None:
            raise ValueError("Kimi-K3 cache requires a linear-attention component")
        return (linear_attn.conv_state_shape, linear_attn.temporal_state_shape)

    @cached_property
    def _recurrent_dtype(self) -> torch.dtype:
        return kda_recurrent_state_dtype(self.server_args)

    @override
    def fields_for_layer(
        self, layer_id: int, group_id: str, occurrence: int
    ) -> tuple[CacheFieldSpec, ...]:
        plane_id = f"slot.{occurrence}"
        if group_id == FULL_ATTENTION:
            config = (
                self.attn_config
                if layer_id < self.num_target_layers
                else self.draft_attn_config
            )
            spec = config.component(MLAConfig)
            latent_width = spec.kv_lora_rank + spec.qk_rope_head_dim
            latent = CacheFieldSpec(
                f"layer.{layer_id}.latent_kv",
                plane_id,
                (self.prefix_granularity, 1, latent_width),
                scatter_stored_dtype_name(config.kv_cache_dtype),
            )
            if isinstance(spec, DSAConfig):
                # Own plane (latent stride = payload), same group: moves with latent.
                return (
                    latent,
                    index_k_field(config, self.prefix_granularity, layer_id),
                )
            return (latent,)
        conv_shape, recurrent_shape = self._kda_shapes
        return (
            CacheFieldSpec(
                f"layer.{layer_id}.conv_state",
                plane_id,
                conv_shape,
                cache_dtype_name(torch.bfloat16),
                exact_page_stride=False,
            ),
            CacheFieldSpec(
                f"layer.{layer_id}.recurrent_state",
                plane_id,
                recurrent_shape,
                cache_dtype_name(self._recurrent_dtype),
                exact_page_stride=False,
            ),
        )

    # ---- packing: state groups match the MLA plane's byte width ----

    @override
    def packing(self, groups: tuple[CacheGroupDeclaration, ...]) -> Mapping[str, int]:
        fields_by_group = {spec.group_id: fields for spec, fields in groups}
        # First MLA layer's latent page: KDA state rides beside it in plane slot.0.
        mla_page_bytes = sum(
            field.payload_bytes
            for field in fields_by_group[FULL_ATTENTION]
            if field.plane_id == "slot.0"
        )
        first_state = f"{LINEAR_ATTENTION}_0"
        linear_plane_bytes = sum(
            field.payload_bytes
            for field in fields_by_group[first_state]
            if field.plane_id == "slot.0"
        )
        # The smallest packing whose plane width covers one per-layer KDA
        # state, so the state page rides inside the MLA plane at the least
        # parent granularity. Attn tp=8 gives the historical 12; attention-DP
        # (tp < 8) grows the state 8/tp-fold and the packing follows it up,
        # while larger tp shrinks the state and the parent with it.
        mla_packing = max(1, -(-linear_plane_bytes // mla_page_bytes))
        linear_packing = max(1, mla_packing * mla_page_bytes // linear_plane_bytes)
        return {
            spec.group_id: (
                mla_packing if spec.group_id == FULL_ATTENTION else linear_packing
            )
            for spec, _ in groups
        }

    @override
    def check_layout(self, layout: CacheLayout) -> None:
        """One plane per MLA layer, the draft's own, and one index-K plane
        per DSA layer -- no more.

        The KDA state groups pack to the MLA plane's byte width precisely so
        they ride inside those planes; a plane of their own means the packing
        did not divide and the parent grew.

        A draft layer costs one plane when it plans the target's attention (an
        MTP depth: one latent) and two when it plans its own family (a DFlash
        GQA stack: K and V head rows), because the planes a layer occupies are
        the fields its config declares.
        """
        num_mla_layers = sum(
            group_id == FULL_ATTENTION for group_id in self.target_group_ids
        )
        planes_per_draft_layer = (
            1
            if self.draft_attn_config is None
            or self.draft_attn_config.component(MLAConfig) is not None
            else 2
        )
        num_draft_planes = planes_per_draft_layer * self.num_draft_layers
        expected = num_mla_layers + num_draft_planes + self.num_index_k_planes
        if len(layout.plane_bytes) != expected:
            raise ValueError(
                f"Kimi-K3 LCM requires {num_mla_layers} target planes (one per "
                f"MLA layer) plus {num_draft_planes} draft planes and "
                f"{self.num_index_k_planes} index-K planes, got "
                f"{len(layout.plane_bytes)}"
            )

    # ---- extras ----

    @cached_property
    def replay_kda(self) -> bool:
        """Whether target decode uses replay, including the one-token window."""
        return kda_replay_supported(self.attn_config)

    @override
    def workspace_bytes(self) -> int:
        """KDA verify staging reserved outside the cache arena, plus the DSA
        history gather workspace of a KVP prefill over gathered histories
        (``dsa_kvp_history_prefill``)."""
        history = 0
        if dsa_kvp_history_prefill(self.attn_config):
            history = dsa_history_gather_workspace_bytes(
                self.attn_config, max_model_len=self.attn_config.context_len
            )
        if self.server_args.disaggregation_mode == "prefill":
            return history
        width = max(int(self.attn_config.speculative_num_draft_tokens or 1), 1)
        if width == 1 and not self.replay_kda:
            return history
        return history + kda_verify_workspace_bytes(
            self.attn_config,
            self.groups(),
            draft_token_num=width,
            replay_kda=self.replay_kda,
        )

    # ---- capacity: the scheduler's concurrency decides, then a search ----

    @override
    def select_num_lcm_blocks(self, layout: CacheLayout, *, max_lcm_blocks: int) -> int:
        if self.token_limit is None:
            return max_lcm_blocks
        # A token limit caps history, not the unsharded KDA working set.
        # Use the same per-group demand as the inverse capacity calculation.
        return min(max_lcm_blocks, self.parents_needed(layout, self.token_limit))

    @override
    def token_capacity(self, layout: CacheLayout, num_lcm_blocks: int) -> int:
        upper = self.token_limit
        if upper is None:
            # Search bound in logical tokens, not per-rank physical rows.
            # parents_needed still accounts for unsharded KDA state/reservations.
            full_packing = dict(layout.group_packing)[FULL_ATTENTION]
            upper = (
                num_lcm_blocks
                * full_packing
                * self._shard_counts[FULL_ATTENTION]
                * layout.prefix_granularity
            )
        return self._capacity_from_parents(layout, num_lcm_blocks, upper_bound=upper)
