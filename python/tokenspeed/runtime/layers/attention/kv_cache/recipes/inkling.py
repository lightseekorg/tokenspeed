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

"""Inkling cache recipe: attention pages plus paged ShortConv checkpoints.

Every attention layer costs bytes in three groups -- its own attention group,
the kvconv column, and its attention group's hiddenconv column -- so the conv
groups are declared whole (spec and fields together) rather than derived from
a layer label. An ordinary MHA draft adds only attention pages to the same
arena.
"""

from __future__ import annotations

import os
from functools import cached_property

import torch
from typing_extensions import override

from tokenspeed.runtime.configs.inkling_config import InklingModelConfig
from tokenspeed.runtime.layers.attention.configs.base import (
    SoftmaxAttnConfig,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.base import (
    CacheRecipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    CacheLayout,
    cache_dtype_bytes,
    cache_dtype_name,
    mxfp8_kv_scale_fields,
    scatter_stored_dtype_name,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    CacheGroupDeclaration,
    CacheGroupSpec,
    apply_pd_transfer_policies,
)

_INKLING_PREFIX_GRANULARITY = 128

# The ShortConv checkpoint groups. Named once: the field builder and the
# whole-group declaration below both reference these.
KVCONV_GROUP_ID = "kvconv"
HIDDENCONV_GROUP_ID = "hiddenconv"


class InklingRecipe(CacheRecipe):
    """Inkling: MHA pages plus per-layer ShortConv state checkpoints."""

    family = "inkling"

    # ---- layer vocabulary ----

    @property
    @override
    def num_target_layers(self) -> int:
        return len(self.attn_config.component(SoftmaxAttnConfig).cache_layer_types)

    @property
    def draft_has_conv(self) -> bool:
        """Whether the draft owns ShortConv state, independently of the target."""
        return self.draft_attn_config is not None and isinstance(
            self.draft_model_config.hf_text_config, InklingModelConfig
        )

    @property
    @override
    def num_draft_layers(self) -> int:
        """The base count, plus the one constraint Inkling's depths impose."""
        num_layers = super().num_draft_layers
        num_steps = self.server_args.speculative_num_steps
        if self.draft_has_conv and num_steps > num_layers:
            raise ValueError(
                f"Inkling MTP has {num_layers} depth layers; "
                f"--speculative-num-steps {num_steps} would wrap depths "
                "with no trained meaning."
            )
        return num_layers

    @cached_property
    def layer_types(self) -> tuple[str, ...]:
        target = tuple(self.attn_config.component(SoftmaxAttnConfig).cache_layer_types)
        if self.draft_attn_config is None:
            return target
        return target + tuple(
            self.draft_attn_config.component(SoftmaxAttnConfig).cache_layer_types
        )

    @property
    def group_ids(self) -> tuple[str, ...]:
        """Inkling groups by label: the label *is* the attention group."""
        return self.layer_types

    @cached_property
    @override
    def layer_kv_head_counts(self) -> tuple[int, ...]:
        counts = inkling_layer_kv_head_counts(self.model_config)
        if self.draft_attn_config is None:
            return counts
        if self.draft_has_conv:
            return counts + inkling_layer_kv_head_counts(self.draft_model_config)
        spec = self.draft_attn_config.component(SoftmaxAttnConfig)
        return counts + (spec.num_kv_heads,) * self.num_draft_layers

    # ---- geometry ----

    @property
    @override
    def prefix_granularity(self) -> int:
        return _INKLING_PREFIX_GRANULARITY

    @property
    @override
    def max_padding_fraction(self) -> float:
        # Conv columns borrow KV planes without filling them; KV pages size the parent.
        return float("inf")

    # ---- capacity: history and conv checkpoints share the parent budget ----

    @override
    def select_num_lcm_blocks(self, layout: CacheLayout, *, max_lcm_blocks: int) -> int:
        if self.token_limit is None:
            return max_lcm_blocks
        # Conv packing is not history capacity; size every group with the scheduler's reservations.
        return min(max_lcm_blocks, self.parents_needed(layout, self.token_limit))

    @override
    def token_capacity(self, layout: CacheLayout, num_lcm_blocks: int) -> int:
        upper = self.token_limit
        if upper is None:
            upper = super().token_capacity(layout, num_lcm_blocks)
        return self._capacity_from_parents(layout, num_lcm_blocks, upper_bound=upper)

    # ---- groups: the layer walk, plus the conv checkpoint columns ----

    @override
    def groups(self) -> tuple[CacheGroupDeclaration, ...]:
        """The attention groups from the layer walk, plus the conv columns."""
        attention = super().groups()
        conv = tuple(
            (
                CacheGroupSpec(
                    group_id=group_id,
                    attn_kind="state",
                    sliding_window_tokens=None,
                    replayable=False,
                ),
                fields,
            )
            for group_id, fields in self._conv_columns(attention).items()
        )
        if self.pd_disaggregation_enabled:
            # The layer walk already stamped the attention groups.
            policies = apply_pd_transfer_policies(tuple(spec for spec, _ in conv))
            conv = tuple(
                (spec, fields) for spec, (_, fields) in zip(policies, conv, strict=True)
            )
        return attention + conv

    @override
    def fields_for_layer(
        self, layer_id: int, group_id: str, occurrence: int
    ) -> tuple[CacheFieldSpec, ...]:
        """This layer's attention pages. Its conv checkpoints are columns."""
        config = self._layer_config(layer_id)
        spec = self._layer_spec(layer_id)
        # mxfp8_kv_scale_fields owns the page-span and head-dim constraints the
        # interleaved scale layout imposes.
        mxfp8 = bool(config.kv_cache_mxfp8)
        kv_heads = self._layer_kv_heads(layer_id)
        kv_shape = (self.prefix_granularity, kv_heads, spec.head_dim)
        kv_dtype = (
            cache_dtype_name(torch.float8_e4m3fn)
            if mxfp8
            else scatter_stored_dtype_name(config.kv_cache_dtype)
        )
        fields = (
            CacheFieldSpec(
                f"layer.{layer_id}.k",
                f"unit.{occurrence}.k",
                kv_shape,
                kv_dtype,
            ),
            CacheFieldSpec(
                f"layer.{layer_id}.v",
                f"unit.{occurrence}.v",
                kv_shape,
                kv_dtype,
            ),
        )
        if not mxfp8:
            return fields
        return fields + mxfp8_kv_scale_fields(
            layer_id=layer_id,
            occurrence=occurrence,
            kv_heads=kv_heads,
            head_dim=spec.head_dim,
            prefix_granularity=self.prefix_granularity,
        )

    def _conv_columns(
        self, attention: tuple[CacheGroupDeclaration, ...]
    ) -> dict[str, tuple[CacheFieldSpec, ...]]:
        """The ShortConv checkpoint columns, one entry per attention layer.

        The checkpoints alias their layer's KV planes, so they reuse the plane
        ids -- which means reproducing the group's plane numbering here. It is
        the same rule the layer walk applies (a group's Nth layer takes plane
        N) and Inkling's layers all declare fields, so the two agree.

        The hidden-stream rows are not TP-sharded, so one parent's KV planes
        cannot hold every layer's: each attention group's rows form their own
        hiddenconv column instead. A parent then stays exactly its KV pages,
        and the columns do not depend on TP, which P/D peers of different TP
        widths must agree on. A row wider than its K/V plane (narrow KV heads
        under wide TP or an FP8 KV cache) takes a plane of its own.
        """
        plane_bytes: dict[str, int] = {}
        for _, fields in attention:
            for field in fields:
                plane_bytes[field.plane_id] = max(
                    plane_bytes.get(field.plane_id, 0), field.payload_bytes
                )
        columns = {KVCONV_GROUP_ID: ()}
        occurrences: dict[str, int] = {}
        hiddenconv_dtype = cache_dtype_name(
            torch.float8_e5m2 if _fp8_sconv() else torch.bfloat16
        )
        for layer_id, group_id in enumerate(self.group_ids):
            occurrence = occurrences.get(group_id, 0)
            occurrences[group_id] = occurrence + 1
            if layer_id >= self.num_target_layers and not self.draft_has_conv:
                continue
            text_config = self._layer_model_config(layer_id).hf_config.get_text_config()
            checkpoint_rows = text_config.sconv_kernel_size - 1
            kv_row = (
                self._layer_kv_heads(layer_id) * self._layer_spec(layer_id).head_dim
            )
            k_plane = f"unit.{occurrence}.k"
            v_plane = f"unit.{occurrence}.v"
            wide_hidden = checkpoint_rows * text_config.hidden_size * cache_dtype_bytes(
                hiddenconv_dtype
            ) > min(plane_bytes[k_plane], plane_bytes[v_plane])
            hidden_group = f"{HIDDENCONV_GROUP_ID}_{group_id}"
            columns[KVCONV_GROUP_ID] += (
                CacheFieldSpec(
                    f"layer.{layer_id}.kvconv_k",
                    k_plane,
                    (checkpoint_rows, kv_row),
                    cache_dtype_name(torch.bfloat16),
                    exact_page_stride=False,
                ),
                CacheFieldSpec(
                    f"layer.{layer_id}.kvconv_v",
                    v_plane,
                    (checkpoint_rows, kv_row),
                    cache_dtype_name(torch.bfloat16),
                    exact_page_stride=False,
                ),
            )
            columns[hidden_group] = columns.get(hidden_group, ()) + (
                CacheFieldSpec(
                    f"layer.{layer_id}.attnconv",
                    f"unit.{occurrence}.hidden_k" if wide_hidden else k_plane,
                    (checkpoint_rows, text_config.hidden_size),
                    hiddenconv_dtype,
                    exact_page_stride=False,
                ),
                CacheFieldSpec(
                    f"layer.{layer_id}.mlpconv",
                    f"unit.{occurrence}.hidden_v" if wide_hidden else v_plane,
                    (checkpoint_rows, text_config.hidden_size),
                    hiddenconv_dtype,
                    exact_page_stride=False,
                ),
            )
        return columns

    def _layer_config(self, layer_id: int):
        return (
            self.attn_config
            if layer_id < self.num_target_layers
            else self.draft_attn_config
        )

    def _layer_spec(self, layer_id: int):
        """The owning attn config's softmax (MHA) component spec."""
        return self._layer_config(layer_id).component(SoftmaxAttnConfig)

    def _layer_model_config(self, layer_id: int):
        return (
            self.model_config
            if layer_id < self.num_target_layers
            else self.draft_model_config
        )

    def _layer_kv_heads(self, layer_id: int) -> int:
        spec = self._layer_spec(layer_id)
        return max(1, self.layer_kv_head_counts[layer_id] // spec.attn_tp_size)

    # ---- extras ----

    @override
    def workspace_bytes(self) -> int:
        """ShortConv rings, including the target's speculative verify rows,
        and the target's MXFP8 KV dequant scratch."""
        draft_tokens = (
            int(self.server_args.speculative_num_draft_tokens)
            if self.draft_attn_config is not None
            else 0
        )
        text_config = self.model_config.hf_config.get_text_config()
        total = _conv_ring_bytes(
            text_config=text_config,
            attn_config=self.attn_config,
            num_layers=text_config.num_hidden_layers,
            spec_tokens=draft_tokens,
        )
        if self.draft_has_conv:
            total += _conv_ring_bytes(
                text_config=self.draft_model_config.hf_config.get_text_config(),
                attn_config=self.draft_attn_config,
                num_layers=self.num_draft_layers,
                spec_tokens=draft_tokens,
            )
        scratch = inkling_kv_dequant_scratch_shape(text_config, self.attn_config)
        if scratch is not None:
            rows, width = scratch
            total += 2 * rows * width * torch.bfloat16.itemsize
        return total


def _fp8_sconv() -> bool:
    return os.environ.get("INKLING_FP8_SCONV", "0") != "0"


def inkling_layer_kv_head_counts(model_config) -> tuple[int, ...]:
    """Served KV heads per layer, before TP sharding.

    Each kind's native checkpoint count (hetero byte-uniform slots, #647), so
    page sizes derive from these. Also read by the PD field-partition logic,
    which needs the global width.
    """
    text_config = model_config.hf_config.get_text_config()
    local = set(text_config.local_layer_ids)
    return tuple(
        (
            text_config.swa_num_key_value_heads
            if layer_id in local
            else text_config.ckpt_num_key_value_heads
        )
        for layer_id in range(text_config.num_hidden_layers)
    )


def inkling_kv_dequant_scratch_shape(
    text_config, attn_config
) -> tuple[int, int] | None:
    """``(rows, row_elements)`` of the BF16 key and value scratch a prefix
    extend over an MXFP8 KV cache dequantizes its pages into, or None for
    other caches.

    The extend kernels read BF16 pages, so each one dequantizes the pages it
    reads first. The scratch holds one request at the longest context; a
    batch that needs more runs in request groups. A row holds the most KV
    heads a layer serves at this TP.
    """
    if not attn_config.kv_cache_mxfp8:
        return None
    spec = attn_config.component(SoftmaxAttnConfig)
    heads = max(
        max(1, int(count) // spec.attn_tp_size)
        for count in (
            text_config.ckpt_num_key_value_heads,
            text_config.swa_num_key_value_heads,
        )
    )
    page = attn_config.kernel_page_size or _INKLING_PREFIX_GRANULARITY
    rows = -(-int(attn_config.context_len) // page) * page
    return rows, heads * spec.head_dim


def _conv_ring_bytes(*, text_config, attn_config, num_layers: int, spec_tokens: int):
    from tokenspeed.runtime.configs.inkling_config import inkling_conv_total_dim

    rows = int(attn_config.max_resident_requests) + 2
    # Match InklingAttnBackend.from_config: (W-1) taps + K chunk rows.
    spec_tokens = max(1, int(spec_tokens))
    ring_rows = int(text_config.sconv_kernel_size) - 1 + spec_tokens
    conv_dim = inkling_conv_total_dim(
        text_config, attn_config.component(SoftmaxAttnConfig).attn_tp_size
    )
    pending_bytes = rows * torch.bool.itemsize
    return (
        num_layers * rows * ring_rows * conv_dim * torch.bfloat16.itemsize
        + pending_bytes
    )
