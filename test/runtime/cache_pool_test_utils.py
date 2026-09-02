from __future__ import annotations

import torch

from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
from tokenspeed.runtime.layers.attention.kv_cache.recipes import spec
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    cache_dtype_name,
    mxfp8_kv_scale_fields,
    pack,
    scatter_stored_dtype_name,
)


def specs_for_layers(
    *,
    layer_types,
    group_ids,
    prefix_granularity,
    sliding_window_tokens=None,
    page_sizes=None,
    pd_disaggregation_enabled=False,
):
    """The group specs a layer vocabulary produces.

    A group must declare fields, so this supplies a one-byte placeholder per
    layer: the subject here is the scheduler semantics (retention, family,
    block span), not the bytes.
    """
    return tuple(
        group_spec
        for group_spec, _ in spec.group(
            layer_types=layer_types,
            group_ids=group_ids,
            sliding_window_tokens=sliding_window_tokens,
            prefix_granularity=prefix_granularity,
            page_sizes=page_sizes,
            pd_disaggregation_enabled=pd_disaggregation_enabled,
            fields_for_layer=lambda layer_id, group_id, occurrence: (
                CacheFieldSpec(
                    f"layer.{layer_id}.probe", f"unit.{occurrence}", (1,), "uint8"
                ),
            ),
        )
    )


def one_group(group_id: str, *fields, **spec_kwargs):
    """One ``(spec, fields)`` pair for a plan under test.

    A whole-group declaration, the way a recipe declares a group that is not
    per-layer: the id is spelled once, in its spec, and its fields hang off
    it. Row geometry defaults so a byte-layout test need not restate
    scheduler semantics. ``group`` (the pipeline stage) is the other way in.
    """
    spec_kwargs.setdefault("retention", "full_history")
    if "checkpoint_granularity" not in spec_kwargs:
        spec_kwargs.setdefault("rows_per_page", spec_kwargs.pop("page_size", 1))
        spec_kwargs.setdefault("entry_stride_tokens", 1)
    return spec.CacheGroupSpec(group_id=group_id, **spec_kwargs), tuple(fields)


def plan_group_specs(plan) -> tuple[spec.CacheGroupSpec, ...]:
    """Row-geometry specs for every group the plan names.

    The arena publishes a contract unconditionally, so a test that only
    exercises field geometry still needs specs. Derive them from the plan
    rather than restating numbers: a group's CacheBlock spans
    ``prefix_granularity / cache_blocks_per_lcm_block`` tokens, declared here
    as one row per token. Tests whose subject *is* the geometry or the
    retention/transfer policy pass their own specs instead.
    """
    return tuple(
        spec.CacheGroupSpec(
            group_id=group.group_id,
            retention="full_history",
            rows_per_page=plan.prefix_granularity // group.cache_blocks_per_lcm_block,
            entry_stride_tokens=1,
        )
        for group in plan.groups
    )


class MinimalCacheView(CachePool):
    """The smallest constructible cache view.

    ``CachePool`` is abstract in exactly the four accessors a view owes its
    kernels, so this stub doubles as the list of what a real pool must add.
    Use it to exercise view-layer behaviour the base class owns (arena
    ownership, dtype, layer window, the scheduler bridge) without dragging in
    a family's kernel buffers.
    """

    layer_num = 0

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise AssertionError("not exercised")

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise AssertionError("not exercised")

    def get_kv_buffer(self, layer_id: int):
        raise AssertionError("not exercised")

    def set_kv_buffer(self, layer, loc, cache_k, cache_v) -> None:
        raise AssertionError("not exercised")


def make_arena(plan, device: str = "cuda", **kwargs) -> CacheArena:
    """Allocate the arena the pool(s) under test are compute views onto."""
    if "cache_group_specs" not in kwargs:
        kwargs["cache_group_specs"] = plan_group_specs(plan)
    return CacheArena(plan, device, **kwargs)


def make_pool(pool_cls, plan, *, device="cpu", arena=None, **kwargs):
    """Build one pool as a compute view over ``plan``'s arena.

    Tests name a plan and the view's compute parameters; the arena is
    created here (or shared, when several views of one arena are under
    test) so no test has to restate the allocation contract.
    """
    arena_kwargs = {
        key: kwargs.pop(key)
        for key in ("cache_group_specs", "token_capacity")
        if key in kwargs
    }
    if arena is None:
        arena = make_arena(plan, device, **arena_kwargs)
    elif arena_kwargs:
        raise TypeError("arena parameters cannot be passed alongside a shared arena")
    return arena, pool_cls(arena=arena, **kwargs)


def plan_fields(
    fields,
    *,
    prefix_granularity,
    budget_bytes=None,
    num_lcm_blocks=None,
    **kwargs,
):
    """Solve a layout and bind capacity the way the recipes do.

    ``fields`` is the ``{group_id: fields}`` map a recipe field builder
    returns; declarations are formed here, the same join the recipes do.
    """
    layout = pack(
        tuple(
            one_group(group_id, *declared, rows_per_page=prefix_granularity)
            for group_id, declared in fields.items()
        ),
        prefix_granularity=prefix_granularity,
        **kwargs,
    )
    if budget_bytes is not None:
        # Parent 0 backs logical null page 0 and is never schedulable.
        num_lcm_blocks = budget_bytes // layout.lcm_block_bytes - 1
    return layout.bind(num_lcm_blocks)


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
    prefix_granularity: int,
    layer_num: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    layer_types: tuple[str, ...] = (),
    sliding_window_tokens: int | tuple[int | None, ...] | None = None,
    mxfp8: bool = False,
):
    """An MHA plan the way the recipe builds one: group, pack, bind."""
    if size % prefix_granularity:
        raise ValueError("test pool size must be divisible by prefix_granularity")
    # Resolve the labels the way a recipe does: one per layer, full-history
    # when the caller declares none.
    if not layer_types:
        layer_types = ("full_attention",) * layer_num
    group_ids = make_layer_group_ids(
        layer_num=layer_num,
        layer_types=layer_types,
        sliding_window_tokens=sliding_window_tokens,
    )
    kv_dtype = (
        cache_dtype_name(torch.float8_e4m3fn)
        if mxfp8
        else scatter_stored_dtype_name(dtype)
    )
    shape = (prefix_granularity, kv_heads, head_dim)

    def fields_for_layer(layer_id, group_id, occurrence):
        fields = (
            CacheFieldSpec(
                f"layer.{layer_id}.k", f"unit.{occurrence}.k", shape, kv_dtype
            ),
            CacheFieldSpec(
                f"layer.{layer_id}.v", f"unit.{occurrence}.v", shape, kv_dtype
            ),
        )
        if not mxfp8:
            return fields
        # The production builder, so a test plan cannot declare a scale shape
        # the recipes would never emit.
        return fields + mxfp8_kv_scale_fields(
            layer_id=layer_id,
            occurrence=occurrence,
            kv_heads=kv_heads,
            head_dim=head_dim,
            prefix_granularity=prefix_granularity,
        )

    groups = spec.group(
        layer_types=layer_types,
        group_ids=group_ids,
        sliding_window_tokens=sliding_window_tokens,
        prefix_granularity=prefix_granularity,
        fields_for_layer=fields_for_layer,
    )
    layout = pack(
        groups,
        prefix_granularity=prefix_granularity,
        cache_blocks_per_lcm_block={gid: 1 for gid in set(group_ids)},
        alignment=1,
        max_padding_fraction=1.0,
    )
    return layout.bind(size // prefix_granularity)


def make_mla_memory_plan(
    *,
    size: int,
    prefix_granularity: int,
    layer_num: int,
    latent_width: int,
    dtype: torch.dtype,
):
    """An MLA plan: one full-attention group, one latent field per layer."""
    if size % prefix_granularity:
        raise ValueError("test pool size must be divisible by prefix_granularity")
    latent_dtype = scatter_stored_dtype_name(dtype)
    groups = spec.group(
        layer_types=("full_attention",) * layer_num,
        group_ids=("full_attention",) * layer_num,
        sliding_window_tokens=None,
        prefix_granularity=prefix_granularity,
        fields_for_layer=lambda layer_id, group_id, occurrence: (
            CacheFieldSpec(
                f"layer.{layer_id}.latent_kv",
                f"slot.{occurrence}",
                (prefix_granularity, 1, latent_width),
                latent_dtype,
            ),
        ),
    )
    layout = pack(
        groups,
        prefix_granularity=prefix_granularity,
        cache_blocks_per_lcm_block={"full_attention": 1},
        alignment=1,
        max_padding_fraction=1.0,
    )
    return layout.bind(size // prefix_granularity)
