"""Full-attention cache recipes shared by MHA and MLA models."""

from __future__ import annotations

from tokenspeed.runtime.layers.attention.kv_cache.recipes import (
    configured_token_limit,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    merge_continuation_layers,
    solve_cache_layout,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    MXFP8_KV_SCALE_TILE_TOKENS,
)


def mla_cache_fields(
    *,
    layer_group_ids,
    prefix_granularity,
    latent_width,
    element_size,
) -> tuple[CacheFieldSpec, ...]:
    """Describe MLA full-attention cache fields."""
    if prefix_granularity <= 0 or latent_width <= 0 or element_size <= 0:
        raise ValueError("MLA full-attention geometry must be positive")
    occurrences: dict[str, int] = {}
    fields = []
    for layer_id, group_id in enumerate(layer_group_ids):
        slot = occurrences.get(group_id, 0)
        occurrences[group_id] = slot + 1
        fields.append(
            CacheFieldSpec(
                group_id,
                f"layer.{layer_id}.latent_kv",
                f"slot.{slot}",
                (prefix_granularity, 1, latent_width),
                element_size,
            )
        )
    return tuple(fields)


def mha_cache_fields(
    *,
    layer_group_ids,
    prefix_granularity,
    kv_heads,
    head_dim,
    kv_element_size,
    kv_scale_block_size=0,
    kv_scale_element_size=0,
) -> tuple[CacheFieldSpec, ...]:
    """Describe fixed per-layer MHA views and their physical placement."""
    if prefix_granularity <= 0 or kv_heads <= 0 or head_dim <= 0:
        raise ValueError("MHA full-attention geometry must be positive")
    if kv_element_size <= 0:
        raise ValueError("MHA KV element size must be positive")
    if bool(kv_scale_block_size) != bool(kv_scale_element_size):
        raise ValueError(
            "kv_scale_block_size and kv_scale_element_size must both be zero "
            "or both be positive"
        )
    if kv_scale_block_size and head_dim % kv_scale_block_size:
        raise ValueError("head_dim must be divisible by kv_scale_block_size")

    occurrences: dict[str, int] = {}
    fields = []
    for layer_id, group_id in enumerate(layer_group_ids):
        unit = occurrences.get(group_id, 0)
        occurrences[group_id] = unit + 1
        shape = (prefix_granularity, kv_heads, head_dim)
        fields.extend(
            (
                CacheFieldSpec(
                    group_id,
                    f"layer.{layer_id}.k",
                    f"unit.{unit}.k",
                    shape,
                    kv_element_size,
                ),
                CacheFieldSpec(
                    group_id,
                    f"layer.{layer_id}.v",
                    f"unit.{unit}.v",
                    shape,
                    kv_element_size,
                ),
            )
        )
        if kv_scale_block_size:
            scale_dim = head_dim // kv_scale_block_size
            scale_shape = (
                (kv_heads, 32, scale_dim, scale_dim)
                if prefix_granularity == MXFP8_KV_SCALE_TILE_TOKENS
                else (prefix_granularity, kv_heads, scale_dim)
            )
            fields.extend(
                (
                    CacheFieldSpec(
                        group_id,
                        f"layer.{layer_id}.k_scale",
                        f"unit.{unit}.k_scale",
                        scale_shape,
                        kv_scale_element_size,
                    ),
                    CacheFieldSpec(
                        group_id,
                        f"layer.{layer_id}.v_scale",
                        f"unit.{unit}.v_scale",
                        scale_shape,
                        kv_scale_element_size,
                    ),
                )
            )
    return tuple(fields)


def draft_cache_fields(
    *,
    layer_group_ids,
    enabled_layer_ids,
    prefix_granularity,
    layer_kv_heads,
    head_dim,
    kv_element_size,
    kv_scale_block_size=0,
    kv_scale_element_size=0,
) -> tuple[CacheFieldSpec, ...]:
    """Describe the enabled draft model's full-attention cache fields."""
    if len(layer_group_ids) != len(layer_kv_heads):
        raise ValueError(
            f"layer_group_ids has {len(layer_group_ids)} entries but "
            f"layer_kv_heads has {len(layer_kv_heads)}"
        )
    if prefix_granularity <= 0 or head_dim <= 0 or kv_element_size <= 0:
        raise ValueError("draft full-attention geometry must be positive")
    if bool(kv_scale_block_size) != bool(kv_scale_element_size):
        raise ValueError(
            "kv_scale_block_size and kv_scale_element_size must both be zero "
            "or both be positive"
        )
    if kv_scale_block_size and (
        prefix_granularity % MXFP8_KV_SCALE_TILE_TOKENS
        or head_dim % kv_scale_block_size
        or kv_scale_block_size <= 0
    ):
        raise ValueError("draft scale geometry is incompatible with the KV shape")

    enabled_layer_ids = tuple(enabled_layer_ids)
    if len(set(enabled_layer_ids)) != len(enabled_layer_ids) or any(
        isinstance(layer_id, bool)
        or not isinstance(layer_id, int)
        or layer_id < 0
        or layer_id >= len(layer_group_ids)
        for layer_id in enabled_layer_ids
    ):
        raise ValueError("enabled draft layer ids must be unique valid layer ids")

    occurrences: dict[str, int] = {}
    fields = []
    for layer_id in enabled_layer_ids:
        group_id = layer_group_ids[layer_id]
        kv_heads = layer_kv_heads[layer_id]
        if kv_heads <= 0:
            raise ValueError("draft layer KV heads must be positive")
        unit = occurrences.get(group_id, 0)
        occurrences[group_id] = unit + 1
        kv_shape = (prefix_granularity, kv_heads, head_dim)
        fields.extend(
            (
                CacheFieldSpec(
                    group_id,
                    f"layer.{layer_id}.k",
                    f"unit.{unit}.k",
                    kv_shape,
                    kv_element_size,
                ),
                CacheFieldSpec(
                    group_id,
                    f"layer.{layer_id}.v",
                    f"unit.{unit}.v",
                    kv_shape,
                    kv_element_size,
                ),
            )
        )
        if kv_scale_block_size:
            scale_dim = head_dim // kv_scale_block_size
            scale_shape = (
                kv_heads,
                prefix_granularity // MXFP8_KV_SCALE_TILE_TOKENS,
                32,
                scale_dim,
                scale_dim,
            )
            fields.extend(
                (
                    CacheFieldSpec(
                        group_id,
                        f"layer.{layer_id}.k_scale",
                        f"unit.{unit}.k_scale",
                        scale_shape,
                        kv_scale_element_size,
                    ),
                    CacheFieldSpec(
                        group_id,
                        f"layer.{layer_id}.v_scale",
                        f"unit.{unit}.v_scale",
                        scale_shape,
                        kv_scale_element_size,
                    ),
                )
            )
    return tuple(fields)


def build_hybrid_cache_setup(
    *,
    family: str,
    server_args,
    fields,
    layer_types: tuple[str, ...],
    group_ids: tuple[str, ...],
    group_specs: tuple,
    state_dtypes,
    layer_kv_head_counts: tuple[int, ...] | None,
    num_draft_layers: int,
    cache_budget_bytes: int,
    fixed_workspace_bytes: int,
    prefix_granularity: int,
    max_padding_fraction: float,
):
    """Bind one hybrid cache layout to a capacity.

    One big model: every parameter is already merged over target AND draft
    layers (see ``merge_continuation_layers``) — the builder itself is
    draft-oblivious. The recipe owns the complete published specs
    (``group_specs``, incl. groups outside the layer-type vocabulary and
    PD transfer policies). ``num_draft_layers`` is carried through to the
    setup for wiring time.
    """
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import (
        CachePoolSpec,
        CacheSetup,
    )

    merged_fields = tuple(fields)
    merged_layer_types = tuple(layer_types)
    merged_group_ids = tuple(group_ids)
    merged_specs = tuple(group_specs)
    merged_head_counts = layer_kv_head_counts
    merged_layout = solve_cache_layout(
        merged_fields,
        prefix_granularity=prefix_granularity,
        alignment=256,
        max_padding_fraction=max_padding_fraction,
    )
    packing = dict(merged_layout.group_packing)

    usable_cache_bytes = cache_budget_bytes - fixed_workspace_bytes
    num_lcm_blocks = usable_cache_bytes // merged_layout.lcm_block_bytes - 1
    if num_lcm_blocks < 1:
        raise ValueError(
            "cache budget must hold a null parent and one usable LCM parent"
        )

    max_packing = max(packing.values())
    token_limit = configured_token_limit(server_args)
    if token_limit is not None:
        requested = token_limit // prefix_granularity // max_packing
        if requested < 1:
            raise ValueError(
                "the configured token limit must hold at least one LCM parent "
                f"({prefix_granularity * max_packing} child tokens)"
            )
        num_lcm_blocks = min(num_lcm_blocks, requested)

    return CacheSetup(
        spec=CachePoolSpec(
            family=family,
            memory_plan=merged_layout.with_num_lcm_blocks(num_lcm_blocks),
            layer_types=merged_layer_types,
            layer_group_ids=merged_group_ids,
            paged_cache_group_specs=merged_specs,
            state_field_dtypes=state_dtypes,
            token_capacity=(num_lcm_blocks * max_packing * prefix_granularity),
            layer_kv_head_counts=merged_head_counts,
        ),
        num_draft_layers=num_draft_layers,
        cache_budget_bytes=cache_budget_bytes,
        fixed_workspace_bytes=fixed_workspace_bytes,
    )


def _profiled_pages(
    *,
    cache_budget_bytes: int,
    bytes_per_token: int,
    page_size: int,
    max_total_tokens: int | None,
) -> int:
    """Return usable pages while keeping the reserved null page in budget."""
    if bytes_per_token <= 0:
        raise ValueError(f"KV cache cell size must be positive, got {bytes_per_token}")
    bytes_per_page = bytes_per_token * page_size
    num_pages = cache_budget_bytes // bytes_per_page - 1
    if max_total_tokens is not None:
        requested_pages = max_total_tokens // page_size
        if requested_pages < 1:
            raise ValueError(
                f"max_total_tokens={max_total_tokens} must contain at least "
                f"one full page (page_size={page_size})"
            )
        num_pages = min(num_pages, requested_pages)

    from tokenspeed.runtime.utils.env import envs

    ci_size = envs.TOKENSPEED_CI_SMALL_KV_SIZE.get_set_value_or(None)
    if ci_size is not None and int(ci_size) > 0:
        ci_tokens = int(ci_size)
        if ci_tokens % page_size:
            raise ValueError(
                "TOKENSPEED_CI_SMALL_KV_SIZE must be divisible by page_size"
            )
        num_pages = min(num_pages, ci_tokens // page_size)
    if num_pages < 1:
        raise ValueError("KV cache token pool size must be positive")
    return num_pages


def _storage_layers(config, num_layers: int) -> int:
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
        hybrid_slab_group_size,
    )

    group_size = hybrid_slab_group_size(
        getattr(config, "layer_types", None),
        sliding_window_tokens=getattr(config, "sliding_window_tokens", None),
    )
    return group_size if group_size is not None else num_layers


def _ordinary_setup(
    *,
    family: str,
    server_args,
    model_config,
    attn_config,
    target_fields,
    target_group_ids: tuple[str, ...],
    draft_model_config,
    draft_attn_config,
    draft_fields,
    draft_group_ids: tuple[str, ...],
    cache_budget_bytes: int,
):
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import (
        CachePoolSpec,
        CacheSetup,
    )

    page_size = int(attn_config.page_size)
    target_layers = model_config.num_attention_layers
    bytes_per_token = attn_config.cache_cell_size() * _storage_layers(
        attn_config, target_layers
    )
    if draft_attn_config is not None:
        if draft_attn_config.page_size != page_size:
            raise ValueError("target and draft cache page sizes must match")
        bytes_per_token += draft_attn_config.cache_cell_size() * _storage_layers(
            draft_attn_config,
            draft_model_config.num_attention_layers,
        )
    num_pages = _profiled_pages(
        cache_budget_bytes=cache_budget_bytes,
        bytes_per_token=bytes_per_token,
        page_size=page_size,
        max_total_tokens=server_args.max_total_tokens,
    )
    # One merged solve, one spec (see build_hybrid_cache_setup): draft
    # layers continue the target's numbering; packing is uniformly 1 on
    # this profiled path. Labels not aligned per-layer degrade to
    # full-history: a NextN draft inherits the target hf_config's
    # layer_types (1 draft layer vs 61 target labels).
    target_layer_types = tuple(getattr(attn_config, "layer_types", ()))
    if len(target_layer_types) != len(target_group_ids):
        target_layer_types = ()
    draft_layer_types = ()
    if draft_fields is not None:
        draft_layer_types = tuple(getattr(draft_attn_config, "layer_types", ()))
        if len(draft_layer_types) != len(draft_group_ids):
            draft_layer_types = ("full_attention",) * len(draft_group_ids)
    (
        merged_fields,
        merged_layer_types,
        merged_group_ids,
        _,
        num_draft_layers,
    ) = merge_continuation_layers(
        fields=target_fields,
        layer_types=target_layer_types,
        group_ids=target_group_ids,
        draft_fields=draft_fields,
        draft_layer_types=draft_layer_types,
        draft_group_ids=draft_group_ids,
    )
    # Empty-or-aligned invariant: with an unlabeled target the merged
    # labels cannot align; every merged group publishes full-history.
    if len(merged_layer_types) != len(merged_group_ids):
        merged_layer_types = ()
    merged_layout = solve_cache_layout(
        merged_fields,
        prefix_granularity=page_size,
        cache_blocks_per_lcm_block={field.group_id: 1 for field in merged_fields},
        alignment=1,
        max_padding_fraction=1.0,
    )
    token_capacity = num_pages * page_size

    from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
        build_paged_cache_group_specs,
    )

    return CacheSetup(
        spec=CachePoolSpec(
            family=family,
            memory_plan=merged_layout.with_num_lcm_blocks(num_pages),
            layer_types=merged_layer_types,
            layer_group_ids=merged_group_ids,
            # ONE spec derivation over the merged layers: a draft-only
            # group is planned AND published (planned-but-unallocated is
            # not a state the scheduler can represent).
            paged_cache_group_specs=build_paged_cache_group_specs(
                layer_types=merged_layer_types,
                group_ids=merged_group_ids,
                sliding_window_tokens=getattr(
                    attn_config, "sliding_window_tokens", None
                ),
                prefix_granularity=page_size,
                pd_disaggregation_enabled=getattr(
                    attn_config, "pd_disaggregation_enabled", False
                ),
            ),
            state_field_dtypes={},
            token_capacity=token_capacity,
        ),
        num_draft_layers=num_draft_layers,
        cache_budget_bytes=cache_budget_bytes,
        fixed_workspace_bytes=0,
    )


def _ordinary_fields(config, num_layers: int):
    """Return the fields and group ids for one ordinary attention config."""
    from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
    from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
    from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
    from tokenspeed.runtime.layers.attention.configs.msa import MSAConfig

    if isinstance(config, DSAConfig):
        return _dsa_fields(config, num_layers), ("full_attention",) * num_layers
    if isinstance(config, MSAConfig):
        return _msa_fields(config, num_layers)
    if isinstance(config, MLAConfig):
        return _mla_fields(config, num_layers), ("full_attention",) * num_layers
    if isinstance(config, MHAConfig):
        return _mha_fields(config, num_layers)
    raise TypeError(f"no ordinary cache recipe for {type(config).__name__}")


def _mha_fields(config, num_layers: int):
    import torch

    from tokenspeed.runtime.layers.attention.kv_cache.recipes import spec

    if config.kv_cache_mxfp8:
        assert config.page_size == MXFP8_KV_SCALE_TILE_TOKENS, (
            "mxfp8 KV cache requires --block-size "
            f"{MXFP8_KV_SCALE_TILE_TOKENS} (the attention kernel consumes "
            "the interleaved paged scale layout)"
        )
    layer_types = tuple(config.layer_types)
    group_ids = (
        tuple(
            spec.layer_group_ids(
                layer_types=layer_types,
                sliding_window_tokens=config.sliding_window_tokens,
            )
        )
        if layer_types
        else ("full_attention",) * num_layers
    )
    if len(group_ids) != num_layers:
        raise ValueError("cache group ids must cover every MHA layer")
    fields = mha_cache_fields(
        layer_group_ids=group_ids,
        prefix_granularity=config.page_size,
        kv_heads=max(config.num_kv_heads // config.attn_tp_size, 1),
        head_dim=config.head_dim,
        kv_element_size=(
            1
            if config.kv_cache_mxfp8
            else torch.empty((), dtype=config.kv_cache_dtype).element_size()
        ),
        kv_scale_block_size=32 if config.kv_cache_mxfp8 else 0,
        kv_scale_element_size=1 if config.kv_cache_mxfp8 else 0,
    )
    return fields, group_ids


def prepare_ordinary_cache(
    *,
    family: str,
    server_args,
    model_config,
    attn_config,
    draft_model_config,
    draft_attn_config,
    cache_budget_bytes: int,
    decode_input_tokens: int,
    overlap_schedule_depth: int,
):
    """Build an ordinary-family setup using the legacy capacity formula.

    ``_ordinary_fields`` dispatches on the config type for target and draft
    alike, so the four ordinary families (mha/mla/dsa/msa) share this one
    entry point.
    """
    target_fields, target_group_ids = _ordinary_fields(
        attn_config, model_config.num_attention_layers
    )
    draft_fields = None
    draft_group_ids = ()
    if draft_attn_config is not None:
        draft_fields, draft_group_ids = _ordinary_fields(
            draft_attn_config, draft_model_config.num_attention_layers
        )
    return _ordinary_setup(
        family=family,
        server_args=server_args,
        model_config=model_config,
        attn_config=attn_config,
        target_fields=target_fields,
        target_group_ids=target_group_ids,
        draft_model_config=draft_model_config,
        draft_attn_config=draft_attn_config,
        draft_fields=draft_fields,
        draft_group_ids=draft_group_ids,
        cache_budget_bytes=cache_budget_bytes,
    )


def _mla_fields(config, num_layers: int):
    import torch

    if config.kv_cache_quant_method == "per_token_head":
        fields = []
        for layer_id in range(num_layers):
            for name, shape, dtype in (
                (
                    "latent_kv",
                    (config.page_size, 1, config.kv_lora_rank),
                    config.kv_cache_dtype,
                ),
                ("latent_scale", (config.page_size, 1, 1), torch.float32),
                (
                    "rope_k",
                    (config.page_size, 1, config.qk_rope_head_dim),
                    config.dtype,
                ),
            ):
                fields.append(
                    CacheFieldSpec(
                        "full_attention",
                        f"layer.{layer_id}.{name}",
                        f"layer.{layer_id}.{name}",
                        shape,
                        torch.empty((), dtype=dtype).element_size(),
                    )
                )
        return tuple(fields)
    return mla_cache_fields(
        layer_group_ids=("full_attention",) * num_layers,
        prefix_granularity=config.page_size,
        latent_width=config.kv_lora_rank + config.qk_rope_head_dim,
        element_size=torch.empty((), dtype=config.kv_cache_dtype).element_size(),
    )


def _dsa_fields(config, num_layers: int):
    fields = list(_mla_fields(config, num_layers))
    from tokenspeed.runtime.layers.attention.configs.dsa import (
        dsa_index_k_row_bytes,
    )

    index_row_bytes = dsa_index_k_row_bytes(config.index_head_dim)
    fields.extend(
        CacheFieldSpec(
            "full_attention",
            f"layer.{layer_id}.index_k",
            f"layer.{layer_id}.index_k",
            (config.page_size, index_row_bytes),
            1,
        )
        for layer_id in range(num_layers)
    )
    return tuple(fields)


def _msa_fields(config, num_layers: int):
    import torch

    fields, group_ids = _mha_fields(config, num_layers)
    fields = list(fields)
    element_size = torch.empty((), dtype=config.dtype).element_size()
    fields.extend(
        CacheFieldSpec(
            "full_attention",
            f"layer.{layer_id}.index_k",
            f"layer.{layer_id}.index_k",
            (config.page_size, config.index_head_dim),
            element_size,
        )
        for layer_id in sorted(config.sparse_layer_ids)
    )
    return tuple(fields), group_ids
