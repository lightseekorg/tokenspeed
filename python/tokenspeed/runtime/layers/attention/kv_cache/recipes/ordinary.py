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

"""The four ordinary cache families: MHA, MLA, DSA, MSA.

One recipe serves all four, and a heterogeneous draft too: what a layer costs
is dispatched on the attention config that owns it, so an MLA target with an
MHA draft is just layers with two different geometries in one plan.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cached_property

import torch
from typing_extensions import override

from tokenspeed.runtime.layers.attention.kv_cache.recipes.base import (
    CacheGroupDeclaration,
    CacheRecipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    CacheLayout,
    cache_dtype_name,
    mxfp8_kv_scale_fields,
    scatter_stored_dtype_name,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    MXFP8_KV_SCALE_TILE_TOKENS,
    cyclic_history_spec,
    hybrid_slab_group_size,
    layer_group_ids,
)


class OrdinaryRecipe(CacheRecipe):
    """MHA / MLA / DSA / MSA: one cache group per attention structure.

    Capacity comes from the profiled bytes-per-token rather than the parent
    size, and every group packs one CacheBlock per parent -- the identity
    grain is the block span.
    """

    def __init__(self, *, family, **kwargs) -> None:
        super().__init__(**kwargs)
        self.family = family

    # ---- layer vocabulary ----

    @cached_property
    def group_ids(self) -> tuple[str, ...]:
        ids = _config_group_ids(self.attn_config, self.num_target_layers)
        if self.draft_attn_config is None:
            return ids
        if self.prefix_granularity % self.draft_attn_config.prefix_granularity:
            raise ValueError(
                "the widened target prefix grain must contain whole draft pages"
            )
        return ids + _config_group_ids(self.draft_attn_config, self.num_draft_layers)

    @cached_property
    def layer_types(self) -> tuple[str, ...]:
        """Merged labels, or empty when they cannot align per layer.

        A NextN draft inherits the target hf_config's ``layer_types`` (one
        draft layer against 61 target labels), so misaligned labels degrade to
        full-history rather than mislabeling a group.
        """
        target = tuple(getattr(self.attn_config, "layer_types", ()))
        if len(target) != self.num_target_layers:
            target = ()
        if self.draft_attn_config is None:
            return target if target else ()
        draft = tuple(getattr(self.draft_attn_config, "layer_types", ()))
        if len(draft) != self.num_draft_layers:
            draft = (FULL_ATTENTION,) * self.num_draft_layers
        merged = target + draft
        return merged if len(merged) == len(self.group_ids) else ()

    # ---- geometry ----

    @property
    @override
    def alignment(self) -> int:
        return 1

    @property
    @override
    def max_padding_fraction(self) -> float:
        return 1.0

    @override
    def packing(self, groups: tuple[CacheGroupDeclaration, ...]) -> Mapping[str, int]:
        """One CacheBlock per parent: the block span is the identity grain."""
        return {spec.group_id: 1 for spec, _ in groups}

    # ---- fields ----

    @override
    def fields_for_layer(
        self, layer_id: int, group_id: str, occurrence: int
    ) -> tuple[CacheFieldSpec, ...]:
        if layer_id < self.num_target_layers:
            config, local_layer_id = self.attn_config, layer_id
        else:
            config = self.draft_attn_config
            local_layer_id = layer_id - self.num_target_layers
        return _config_layer_fields(
            config,
            layer_id=layer_id,
            local_layer_id=local_layer_id,
            occurrence=occurrence,
            block_span=self.prefix_granularity,
        )

    @override
    def groups(self) -> tuple[CacheGroupDeclaration, ...]:
        groups = super().groups()
        dcp_size = int(getattr(self.attn_config, "dcp_size", 1))
        if dcp_size == 1:
            return groups
        return tuple(
            (cyclic_history_spec(spec, dcp_size=dcp_size), fields)
            for spec, fields in groups
        )

    # ---- capacity: profiled bytes per token, not parent size ----

    @override
    def num_lcm_blocks(self, layout: CacheLayout) -> int:
        parent_bytes = layout.lcm_block_bytes
        if not layout.group_packing:
            # Capacity-only probes may not carry a packed layout.  Preserve
            # ordinary cache sizing from the declared per-token payload while
            # production plans use the exact (DCP-aware) packed parent bytes.
            bytes_per_token = self.attn_config.cache_cell_size() * _storage_layers(
                self.attn_config, self.num_target_layers
            )
            if self.draft_attn_config is not None:
                bytes_per_token += (
                    self.draft_attn_config.cache_cell_size()
                    * _storage_layers(self.draft_attn_config, self.num_draft_layers)
                )
            parent_bytes = bytes_per_token * self.prefix_granularity
        if parent_bytes <= 0:
            raise ValueError(
                f"KV cache LCM block size must be positive, got {parent_bytes}"
            )
        return self._capped_parents(
            self.cache_budget_bytes // parent_bytes - 1,
            parent_tokens=self.prefix_granularity,
        )


def _storage_layers(config, num_layers: int) -> int:
    group_size = hybrid_slab_group_size(
        getattr(config, "layer_types", None),
        sliding_window_tokens=getattr(config, "sliding_window_tokens", None),
    )
    return group_size if group_size is not None else num_layers


def _config_group_ids(config, num_layers: int) -> tuple[str, ...]:
    """Per-layer group ids for one ordinary attention config."""
    from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
    from tokenspeed.runtime.layers.attention.configs.msa import MSAConfig

    if isinstance(config, MHAConfig | MSAConfig):
        layer_types = tuple(config.layer_types)
        if layer_types:
            ids = tuple(
                layer_group_ids(
                    layer_types=layer_types,
                    sliding_window_tokens=config.sliding_window_tokens,
                )
            )
            if len(ids) != num_layers:
                raise ValueError("cache group ids must cover every layer")
            return ids
    return (FULL_ATTENTION,) * num_layers


def _config_layer_fields(
    config,
    *,
    layer_id: int,
    local_layer_id: int,
    occurrence: int,
    block_span: int,
) -> tuple[CacheFieldSpec, ...]:
    """What one layer costs, dispatched on the config that owns the layer."""
    from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
    from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
    from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
    from tokenspeed.runtime.layers.attention.configs.msa import MSAConfig

    if isinstance(config, DSAConfig):
        return _mla_layer_fields(config, layer_id, occurrence, block_span) + (
            _index_k_field(config, layer_id),
        )
    if isinstance(config, MSAConfig):
        fields = _mha_layer_fields(config, layer_id, occurrence, block_span)
        if local_layer_id in config.sparse_layer_ids:
            fields += (_index_k_field(config, layer_id),)
        return fields
    if isinstance(config, MLAConfig):
        return _mla_layer_fields(config, layer_id, occurrence, block_span)
    if isinstance(config, MHAConfig):
        return _mha_layer_fields(config, layer_id, occurrence, block_span)
    raise TypeError(f"no ordinary cache recipe for {type(config).__name__}")


def _mha_layer_fields(config, layer_id: int, occurrence: int, block_span: int):
    """One MHA layer's K/V pages, with mxfp8 scale planes when enabled."""
    mxfp8 = bool(config.kv_cache_mxfp8)
    if mxfp8 and config.prefix_granularity != MXFP8_KV_SCALE_TILE_TOKENS:
        raise AssertionError(
            "mxfp8 KV cache requires --prefix-granularity "
            f"{MXFP8_KV_SCALE_TILE_TOKENS} (the attention kernel consumes "
            "the interleaved paged scale layout)"
        )
    kv_heads = max(config.num_kv_heads // config.attn_tp_size, 1)
    head_dim = config.head_dim
    if config.prefix_granularity <= 0 or kv_heads <= 0 or head_dim <= 0:
        raise ValueError("MHA full-attention geometry must be positive")
    shape = (block_span, kv_heads, head_dim)
    kv_dtype = (
        # MXFP8 writes go through dtype-aware kernels, so the arena keeps the
        # fp8 view; the scatter-written paths fall back to uint8.
        cache_dtype_name(torch.float8_e4m3fn)
        if mxfp8
        else scatter_stored_dtype_name(config.kv_cache_dtype)
    )
    fields = (
        CacheFieldSpec(f"layer.{layer_id}.k", f"unit.{occurrence}.k", shape, kv_dtype),
        CacheFieldSpec(f"layer.{layer_id}.v", f"unit.{occurrence}.v", shape, kv_dtype),
    )
    if not mxfp8:
        return fields
    return fields + mxfp8_kv_scale_fields(
        layer_id=layer_id,
        occurrence=occurrence,
        kv_heads=kv_heads,
        head_dim=head_dim,
        prefix_granularity=block_span,
    )


def _mla_layer_fields(config, layer_id: int, occurrence: int, block_span: int):
    """One MLA layer's latent page, split into planes when quantized."""
    dcp_size = int(getattr(config, "dcp_size", 1))
    if block_span <= 0 or block_span % dcp_size:
        raise ValueError("MLA full-attention geometry must be positive")
    physical_rows = block_span // dcp_size
    if config.kv_cache_quant_method != "per_token_head":
        latent_width = config.kv_lora_rank + config.qk_rope_head_dim
        return (
            CacheFieldSpec(
                f"layer.{layer_id}.latent_kv",
                f"slot.{occurrence}",
                (physical_rows, 1, latent_width),
                scatter_stored_dtype_name(config.kv_cache_dtype),
                row_addressed=True,
            ),
        )
    return tuple(
        CacheFieldSpec(
            f"layer.{layer_id}.{name}",
            f"layer.{layer_id}.{name}",
            shape,
            dtype,
            row_addressed=True,
        )
        for name, shape, dtype in (
            (
                "latent_kv",
                (physical_rows, 1, config.kv_lora_rank),
                scatter_stored_dtype_name(config.kv_cache_dtype),
            ),
            (
                "latent_scale",
                (physical_rows, 1, 1),
                cache_dtype_name(torch.float32),
            ),
            (
                "rope_k",
                (physical_rows, 1, config.qk_rope_head_dim),
                cache_dtype_name(config.dtype),
            ),
        )
    )


def _index_k_field(config, layer_id: int) -> CacheFieldSpec:
    """The sparse indexer's key row for one layer (DSA bytes, MSA elements)."""
    from tokenspeed.runtime.layers.attention.configs.dsa import (
        DSAConfig,
        dsa_index_k_row_bytes,
    )

    if isinstance(config, DSAConfig):
        return CacheFieldSpec(
            f"layer.{layer_id}.index_k",
            f"layer.{layer_id}.index_k",
            (config.prefix_granularity, dsa_index_k_row_bytes(config.index_head_dim)),
            "uint8",
        )
    return CacheFieldSpec(
        f"layer.{layer_id}.index_k",
        f"layer.{layer_id}.index_k",
        (config.prefix_granularity, config.index_head_dim),
        cache_dtype_name(config.dtype),
    )
