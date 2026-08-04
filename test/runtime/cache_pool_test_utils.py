from __future__ import annotations

import torch

from tokenspeed.runtime.configs import paged_cache_spec
from tokenspeed.runtime.layers.attention.kv_cache.plan import solve_cache_layout
from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import (
    mha_cache_fields,
    mla_cache_fields,
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
    group_ids = (
        tuple(
            paged_cache_spec.layer_group_ids(
                layer_types=layer_types,
                sliding_window_tokens=sliding_window_tokens,
            )
        )
        if layer_types
        else ("full_attention",) * layer_num
    )
    fields = mha_cache_fields(
        layer_group_ids=group_ids,
        logical_block_tokens=page_size,
        kv_heads=kv_heads,
        head_dim=head_dim,
        kv_element_size=(1 if mxfp8 else torch.empty((), dtype=dtype).element_size()),
        kv_scale_block_size=(32 if mxfp8 else 0),
        kv_scale_element_size=(1 if mxfp8 else 0),
    )
    layout = solve_cache_layout(
        fields,
        logical_block_tokens=page_size,
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
        logical_block_tokens=page_size,
        latent_width=latent_width,
        element_size=torch.empty((), dtype=dtype).element_size(),
    )
    layout = solve_cache_layout(
        fields,
        logical_block_tokens=page_size,
        cache_blocks_per_lcm_block={"full_attention": 1},
        alignment=1,
        max_padding_fraction=1.0,
    )
    return layout.with_num_lcm_blocks(size // page_size)
