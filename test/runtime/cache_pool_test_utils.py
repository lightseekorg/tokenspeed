from __future__ import annotations

import torch

from tokenspeed.runtime.layers.attention.kv_cache.recipes import spec
from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import (
    mha_cache_fields,
    mla_cache_fields,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import solve_cache_layout


def plan_fields(
    fields,
    *,
    prefix_granularity,
    budget_bytes=None,
    num_lcm_blocks=None,
    **kwargs,
):
    """Solve a layout and bind capacity the way the recipes do."""
    layout = solve_cache_layout(
        fields,
        prefix_granularity=prefix_granularity,
        **kwargs,
    )
    if budget_bytes is not None:
        # Parent 0 backs logical null page 0 and is never schedulable.
        num_lcm_blocks = budget_bytes // layout.lcm_block_bytes - 1
    return layout.with_num_lcm_blocks(num_lcm_blocks)


def make_layer_group_ids(
    *,
    layer_num: int,
    layer_types: tuple[str, ...] = (),
    sliding_window_tokens: int | tuple[int | None, ...] | None = None,
) -> tuple[str, ...]:
    """Derive per-layer cache group ids the way the recipes do."""
    if not layer_types:
        return ("full_attention",) * layer_num
    return tuple(
        spec.layer_group_ids(
            layer_types=layer_types,
            sliding_window_tokens=sliding_window_tokens,
        )
    )


def make_mha_memory_plan(
    *,
    size: int,
    page_size: int,
    layer_num: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    layer_types: tuple[str, ...] = (),
    sliding_window_tokens: int | tuple[int | None, ...] | None = None,
    mxfp8: bool = False,
):
    if size % page_size:
        raise ValueError("test pool size must be divisible by page_size")
    group_ids = make_layer_group_ids(
        layer_num=layer_num,
        layer_types=layer_types,
        sliding_window_tokens=sliding_window_tokens,
    )
    fields = mha_cache_fields(
        layer_group_ids=group_ids,
        prefix_granularity=page_size,
        kv_heads=kv_heads,
        head_dim=head_dim,
        kv_element_size=(1 if mxfp8 else torch.empty((), dtype=dtype).element_size()),
        kv_scale_block_size=(32 if mxfp8 else 0),
        kv_scale_element_size=(1 if mxfp8 else 0),
    )
    layout = solve_cache_layout(
        fields,
        prefix_granularity=page_size,
        cache_blocks_per_lcm_block={group_id: 1 for group_id in group_ids},
        alignment=1,
        max_padding_fraction=1.0,
    )
    return layout.with_num_lcm_blocks(size // page_size)


def make_mla_memory_plan(
    *,
    size: int,
    page_size: int,
    layer_num: int,
    latent_width: int,
    dtype: torch.dtype,
):
    if size % page_size:
        raise ValueError("test pool size must be divisible by page_size")
    fields = mla_cache_fields(
        layer_group_ids=("full_attention",) * layer_num,
        prefix_granularity=page_size,
        latent_width=latent_width,
        element_size=torch.empty((), dtype=dtype).element_size(),
    )
    layout = solve_cache_layout(
        fields,
        prefix_granularity=page_size,
        cache_blocks_per_lcm_block={"full_attention": 1},
        alignment=1,
        max_padding_fraction=1.0,
    )
    return layout.with_num_lcm_blocks(size // page_size)
