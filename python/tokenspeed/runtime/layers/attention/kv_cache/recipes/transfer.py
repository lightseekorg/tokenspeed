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

"""Cache-recipe transfer distribution schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldLayout,
    CacheMemoryPlan,
    cache_field_layer_id,
    cache_field_plane,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.configs.model_config import ModelConfig


@dataclass(frozen=True, slots=True)
class CacheFieldPartition:
    """Dense global-to-rank-local partitioning for one PD cache field."""

    axis: int
    global_extent: int
    global_parts: tuple[int, ...] = ()

    def validate(self, *, field_id: str, local_shape: tuple[int, ...]) -> None:
        if (
            isinstance(self.axis, bool)
            or not isinstance(self.axis, int)
            or not 0 <= self.axis < len(local_shape)
            or isinstance(self.global_extent, bool)
            or not isinstance(self.global_extent, int)
            or self.global_extent < local_shape[self.axis]
        ):
            raise ValueError(f"cache field {field_id!r} has invalid partition geometry")
        local_extent = local_shape[self.axis]
        if self.global_extent % local_extent:
            raise ValueError(
                f"cache field {field_id!r} global partition extent is not "
                "divisible by its local extent"
            )
        distinct_shards = self.global_extent // local_extent
        if self.global_parts and (
            len(self.global_parts) < 2
            or len(self.global_parts) > 16
            or any(
                isinstance(part, bool)
                or not isinstance(part, int)
                or part <= 0
                or part % distinct_shards
                for part in self.global_parts
            )
            or sum(self.global_parts) != self.global_extent
        ):
            raise ValueError(
                f"cache field {field_id!r} has invalid global partition parts"
            )


@dataclass(frozen=True, slots=True)
class CacheFieldTransferSpec:
    field_id: str
    partition: CacheFieldPartition

    def __post_init__(self) -> None:
        if not self.field_id:
            raise ValueError("cache transfer field_id must be non-empty")


@dataclass(frozen=True, slots=True)
class CacheTransferSchema:
    """PD wire semantics omitted from the rank-local cache memory plan."""

    fields: tuple[CacheFieldTransferSpec, ...] = ()

    def __post_init__(self) -> None:
        field_ids = tuple(field.field_id for field in self.fields)
        if len(field_ids) != len(set(field_ids)):
            raise ValueError("cache transfer schema contains a duplicate field")

    def partition_for(self, field_id: str) -> CacheFieldPartition | None:
        for field in self.fields:
            if field.field_id == field_id:
                return field.partition
        return None

    def validate(self, plan: CacheMemoryPlan) -> None:
        planned_fields = {field.field_id: field for field in plan.fields}
        unknown = {field.field_id for field in self.fields} - set(planned_fields)
        if unknown:
            raise ValueError(
                f"cache transfer schema references unknown fields {sorted(unknown)}"
            )
        for transfer_field in self.fields:
            field = planned_fields[transfer_field.field_id]
            transfer_field.partition.validate(
                field_id=field.field_id,
                local_shape=field.shape,
            )


def _source_model(
    layer_id: int,
    *,
    model_config: ModelConfig,
    draft_model_config: ModelConfig | None,
) -> tuple[ModelConfig, int]:
    target_layers = model_config.num_attention_layers
    if layer_id < target_layers:
        return model_config, layer_id
    if draft_model_config is None:
        raise ValueError(
            f"PD cache field layer {layer_id} exceeds the target model without a draft"
        )
    draft_layer_id = layer_id - target_layers
    if draft_layer_id >= draft_model_config.num_attention_layers:
        raise ValueError(f"PD cache field layer {layer_id} exceeds the merged model")
    return draft_model_config, draft_layer_id


def _partition_for_field(
    field: CacheFieldLayout,
    *,
    model_config: ModelConfig,
    draft_model_config: ModelConfig | None,
    inkling_layers: frozenset[int],
) -> CacheFieldPartition | None:
    layer_id = cache_field_layer_id(field.field_id)
    source_model, source_layer_id = _source_model(
        layer_id,
        model_config=model_config,
        draft_model_config=draft_model_config,
    )
    suffix = cache_field_plane(field.field_id)

    if layer_id in inkling_layers:
        from tokenspeed.runtime.layers.attention.kv_cache.recipes.inkling import (
            inkling_layer_kv_head_counts,
        )

        global_heads = inkling_layer_kv_head_counts(source_model)[source_layer_id]
        if suffix in ("k", "v"):
            return CacheFieldPartition(1, global_heads)
        if suffix in ("k_scale", "v_scale"):
            return CacheFieldPartition(0, global_heads)
        if suffix.startswith("kvconv_"):
            return CacheFieldPartition(1, global_heads * source_model.head_dim)
        return None

    if suffix in ("k", "v"):
        return CacheFieldPartition(1, source_model.num_key_value_heads)
    if suffix in ("k_scale", "v_scale"):
        return CacheFieldPartition(0, source_model.num_key_value_heads)

    text_config = source_model.hf_text_config
    if suffix in ("conv", "ssm") and hasattr(text_config, "linear_num_key_heads"):
        key_width = text_config.linear_key_head_dim * text_config.linear_num_key_heads
        value_width = (
            text_config.linear_value_head_dim * text_config.linear_num_value_heads
        )
        if suffix == "conv":
            return CacheFieldPartition(
                0,
                2 * key_width + value_width,
                (key_width, key_width, value_width),
            )
        return CacheFieldPartition(0, text_config.linear_num_value_heads)

    linear_config = getattr(text_config, "linear_attn_config", None)
    if suffix in ("conv_state", "recurrent_state") and linear_config is not None:
        num_heads = int(linear_config["num_heads"])
        head_dim = int(linear_config["head_dim"])
        if suffix == "conv_state":
            width = num_heads * head_dim
            return CacheFieldPartition(0, 3 * width, (width, width, width))
        return CacheFieldPartition(0, num_heads)

    return None


def build_cache_transfer_schema(
    plan: CacheMemoryPlan,
    *,
    model_config: ModelConfig,
    draft_model_config: ModelConfig | None = None,
) -> CacheTransferSchema:
    """Compile model-specific PD distribution semantics beside a pure plan."""

    inkling_layers = frozenset(
        cache_field_layer_id(field.field_id)
        for field in plan.fields
        if cache_field_plane(field.field_id).startswith("kvconv_")
    )
    fields = []
    for field in plan.fields:
        partition = _partition_for_field(
            field,
            model_config=model_config,
            draft_model_config=draft_model_config,
            inkling_layers=inkling_layers,
        )
        if partition is not None:
            fields.append(CacheFieldTransferSpec(field.field_id, partition))
    return CacheTransferSchema(tuple(fields))


__all__ = [
    "CacheFieldPartition",
    "CacheFieldTransferSpec",
    "CacheTransferSchema",
    "build_cache_transfer_schema",
]
