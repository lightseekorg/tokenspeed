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
KDA state (12 at attn tp=8). DCP raises that packing as needed so the replicated
KDA tenants still fit in each sharded MLA plane without wasting a parent.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from functools import cached_property

import torch
from typing_extensions import override

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
    cyclic_history_spec,
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

    # ---- layer vocabulary ----

    @cached_property
    def _text_config(self):
        hf_config = self.model_config.hf_config
        return getattr(hf_config, "text_config", hf_config)

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
                "93-layer Kimi-K3 requires 69 KDA and 24 MLA layers, got "
                f"{len(kda_layer_ids)} and {len(full_layer_ids)}"
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

    # ---- geometry ----

    @property
    @override
    def prefix_granularity(self) -> int:
        dcp_size = int(getattr(self.attn_config, "dcp_size", 1))
        if dcp_size == 1:
            return _KIMI_K3_PREFIX_GRANULARITY
        kernel_page_size = int(
            getattr(self.attn_config, "kernel_page_size", None) or 64
        )
        return math.lcm(_KIMI_K3_PREFIX_GRANULARITY, kernel_page_size * dcp_size)

    @override
    def groups(self) -> tuple[CacheGroupDeclaration, ...]:
        from dataclasses import replace

        dcp_size = int(getattr(self.attn_config, "dcp_size", 1))
        groups = super().groups()
        if dcp_size == 1:
            return groups
        resolved = []
        for spec, fields in groups:
            if spec.family == "state":
                spec = replace(
                    spec,
                    checkpoint_granularity=_KIMI_K3_PREFIX_GRANULARITY,
                )
            else:
                spec = cyclic_history_spec(spec, dcp_size=dcp_size)
            resolved.append((spec, fields))
        return tuple(resolved)

    @property
    @override
    def max_padding_fraction(self) -> float:
        # A draft adds MLA planes of its own; the target-only bound would
        # reject the wider arena they imply.
        return float("inf") if self.num_draft_layers else 0.25

    # ---- fields ----

    @cached_property
    def _kda_shapes(self):
        """(conv shape, recurrent shape) per rank, validated against tp size."""
        tp_size = self.attn_config.attn_tp_size
        if tp_size <= 0:
            raise ValueError(f"tp_size must be positive, got {tp_size}")
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
        linear = self._text_config.linear_attn_config
        num_heads = int(linear["num_heads"])
        head_dim = int(linear["head_dim"])
        kernel_size = int(linear["short_conv_kernel_size"])
        if num_heads % tp_size:
            raise ValueError(
                f"KDA num_heads={num_heads} must be divisible by tp_size={tp_size}"
            )
        return (
            (3 * num_heads * head_dim // tp_size, kernel_size - 1),
            (num_heads // tp_size, head_dim, head_dim),
        )

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
            latent_width = config.kv_lora_rank + config.qk_rope_head_dim
            dcp_size = int(getattr(config, "dcp_size", 1))
            if self.prefix_granularity % dcp_size:
                raise ValueError(
                    f"Kimi-K3 prefix grain {self.prefix_granularity} is not "
                    f"divisible by DCP size {dcp_size}"
                )
            return (
                CacheFieldSpec(
                    f"layer.{layer_id}.latent_kv",
                    plane_id,
                    (self.prefix_granularity // dcp_size, 1, latent_width),
                    scatter_stored_dtype_name(config.kv_cache_dtype),
                    row_addressed=True,
                ),
            )
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
                cache_dtype_name(torch.float32),
                exact_page_stride=False,
            ),
        )

    # ---- packing: state groups match the MLA plane's byte width ----

    @override
    def packing(self, groups: tuple[CacheGroupDeclaration, ...]) -> Mapping[str, int]:
        fields_by_group = {spec.group_id: fields for spec, fields in groups}
        mla_page_bytes = next(
            field.payload_bytes for field in fields_by_group[FULL_ATTENTION]
        )
        first_state = f"{LINEAR_ATTENTION}_0"
        linear_plane_bytes = sum(
            field.payload_bytes
            for field in fields_by_group[first_state]
            if field.plane_id == "slot.0"
        )
        # Fit one replicated KDA state inside each DCP-sharded MLA plane.
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
        """One plane per MLA layer, plus one per draft layer -- no more.

        The KDA state groups pack to the MLA plane's byte width precisely so
        they ride inside those planes; a plane of their own means the packing
        did not divide and the parent grew.
        """
        num_mla_layers = sum(
            group_id == FULL_ATTENTION for group_id in self.target_group_ids
        )
        expected = num_mla_layers + self.num_draft_layers
        if len(layout.plane_bytes) != expected:
            raise ValueError(
                f"Kimi-K3 LCM requires {num_mla_layers} target planes (one per "
                f"MLA layer) plus {self.num_draft_layers} draft planes, got "
                f"{len(layout.plane_bytes)}"
            )

    # ---- extras ----

    @cached_property
    def replay_kda(self) -> bool:
        """Whether verify commits by replaying from one conv checkpoint row."""
        if self.server_args.speculative_algorithm is None:
            return False
        from tokenspeed_kernel.ops.attention import (
            kda_recurrent_layout,
            kda_replay_commit_supported,
        )

        return bool(
            kda_replay_commit_supported(
                self.attn_config.dtype, recurrent_layout=kda_recurrent_layout()
            )
        )

    @override
    def workspace_bytes(self) -> int:
        """KDA verify staging reserved outside the cache arena."""
        if self.server_args.speculative_algorithm is None:
            return 0
        if self.replay_kda:
            # Replay starts from the committed convolution checkpoint and
            # reconstructs the accepted recurrent state.
            from tokenspeed_kernel.ops.attention import (
                kda_batched_replay_uses_raw_gate,
            )

            replay_uses_raw_gate = kda_batched_replay_uses_raw_gate(
                self.attn_config.dtype
            )
            # Raw-g replay reuses the committed convolution pool as verify scratch.
            conv_bytes = (
                0
                if replay_uses_raw_gate
                else self.attn_config.max_bs
                * sum(
                    field.payload_bytes
                    for spec, fields in self.groups()
                    if spec.group_id != FULL_ATTENTION
                    for field in fields
                    if field.field_id.endswith(".conv_state")
                )
            )
            conv_shape, recurrent_shape = self._kda_shapes
            heads, head_dim, _ = recurrent_shape
            rows = self.attn_config.max_bs * int(
                self.server_args.speculative_num_draft_tokens
            )
            layer_count = sum(
                field.field_id.endswith(".conv_state")
                for spec, fields in self.groups()
                if spec.group_id != FULL_ATTENTION
                for field in fields
            )
            payload_bytes_per_row = (
                conv_shape[0] + head_dim + heads
            ) * torch.bfloat16.itemsize
            # Fused raw-g capture stores BF16; other replay paths need FP32 scratch.
            gate_itemsize = (
                torch.bfloat16.itemsize
                if replay_uses_raw_gate
                else torch.float32.itemsize
            )
            payload_bytes_per_row += heads * head_dim * gate_itemsize
            return conv_bytes + layer_count * rows * payload_bytes_per_row
        verify_rows = self.attn_config.max_bs * (
            int(self.server_args.speculative_num_draft_tokens) + 1
        )
        return verify_rows * sum(
            field.payload_bytes
            for spec, fields in self.groups()
            if spec.group_id != FULL_ATTENTION
            for field in fields
        )

    # ---- capacity: the scheduler's concurrency decides, then a search ----

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
        # K3 mixes persistent MLA history with a fixed two-page working set
        # for each KDA group.  The generic flat-packing cap is therefore not
        # its inverse and can underallocate when DCP changes MLA packing.
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
        """Physical parents this capacity needs at the configured concurrency.

        K3's KDA state rides inside the MLA planes, so its demand is a pair of
        closed forms (history pages for MLA, a fixed working set for the state
        groups) rather than the contract's per-retention page-count formula.
        """
        specs = {spec.group_id: spec for spec in self._group_specs}
        limits = self.scheduler_limits
        max_live_requests = limits["max_live_requests"]
        depth = limits["overlap_schedule_depth"]
        if depth not in (0, 1):
            raise ValueError(f"overlap_schedule_depth must be 0 or 1, got {depth}")
        if depth and limits["decode_input_tokens"] == 0:
            raise ValueError("overlapped cache sizing requires decode_input_tokens > 0")
        parents = 0
        for group_id, packing in layout.group_packing:
            block_tokens = specs[group_id].block_granularity
            if group_id == FULL_ATTENTION:
                protected_pages = max_live_requests * math.ceil(
                    depth * limits["decode_input_tokens"] / block_tokens
                )
                child_pages = (
                    math.ceil(token_capacity / block_tokens)
                    + max_live_requests
                    - 1
                    + protected_pages
                )
            else:
                # Snapshot state rolls between two pages per live request.
                child_pages = 2 * max_live_requests
            parents += math.ceil(child_pages / packing)
        return parents
