"""Full-attention cache recipes shared by MHA and MLA models."""

from __future__ import annotations

from tokenspeed.runtime.layers.attention.kv_cache.plan import (
    CacheFieldSpec,
    solve_cache_layout,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes import (
    configured_token_limit,
)


def mla_cache_fields(
    *,
    layer_group_ids,
    logical_block_tokens,
    latent_width,
    element_size,
) -> tuple[CacheFieldSpec, ...]:
    """Describe MLA full-attention cache fields."""
    if logical_block_tokens <= 0 or latent_width <= 0 or element_size <= 0:
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
                (logical_block_tokens, 1, latent_width),
                element_size,
            )
        )
    return tuple(fields)


def mha_cache_fields(
    *,
    layer_group_ids,
    logical_block_tokens,
    kv_heads,
    head_dim,
    kv_element_size,
    kv_scale_block_size=0,
    kv_scale_element_size=0,
) -> tuple[CacheFieldSpec, ...]:
    """Describe fixed per-layer MHA views and their physical placement."""
    if logical_block_tokens <= 0 or kv_heads <= 0 or head_dim <= 0:
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
        shape = (logical_block_tokens, kv_heads, head_dim)
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
                if logical_block_tokens == 128
                else (logical_block_tokens, kv_heads, scale_dim)
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
    logical_block_tokens,
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
    if logical_block_tokens <= 0 or head_dim <= 0 or kv_element_size <= 0:
        raise ValueError("draft full-attention geometry must be positive")
    if bool(kv_scale_block_size) != bool(kv_scale_element_size):
        raise ValueError(
            "kv_scale_block_size and kv_scale_element_size must both be zero "
            "or both be positive"
        )
    if kv_scale_block_size and (
        logical_block_tokens % 128
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
        kv_shape = (logical_block_tokens, kv_heads, head_dim)
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
                logical_block_tokens // 128,
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
    state_dtypes,
    layer_kv_head_counts: tuple[int, ...] | None,
    draft_fields,
    draft_layer_types: tuple[str, ...],
    draft_group_ids: tuple[str, ...],
    draft_layer_kv_head_counts: tuple[int, ...] | None,
    cache_budget_bytes: int,
    fixed_workspace_bytes: int,
    logical_block_tokens: int,
    max_padding_fraction: float,
):
    """Bind one hybrid cache layout and its draft to a common capacity."""
    from tokenspeed.runtime.layers.attention.kv_cache.setup import (
        CachePoolSpec,
        CacheSetup,
    )

    target_layout = solve_cache_layout(
        fields,
        logical_block_tokens=logical_block_tokens,
        alignment=256,
        max_padding_fraction=max_padding_fraction,
    )
    target_packing = dict(target_layout.group_packing)
    draft_group_packing = {}
    draft_layout = None
    draft_parent_bytes = 0
    if draft_fields is not None:
        unknown = set(draft_group_ids) - set(target_packing)
        if unknown:
            raise ValueError(
                "draft cache groups are not present in the target plan: "
                f"{sorted(unknown)}"
            )
        draft_group_packing = {
            group_id: target_packing[group_id]
            for group_id in {field.group_id for field in draft_fields}
        }
        draft_layout = solve_cache_layout(
            draft_fields,
            logical_block_tokens=logical_block_tokens,
            cache_blocks_per_lcm_block=draft_group_packing,
            alignment=256,
            max_padding_fraction=max_padding_fraction,
        )
        draft_parent_bytes = draft_layout.lcm_block_bytes

    usable_cache_bytes = cache_budget_bytes - fixed_workspace_bytes
    parent_bytes = target_layout.lcm_block_bytes + draft_parent_bytes
    num_lcm_blocks = usable_cache_bytes // parent_bytes - 1
    if num_lcm_blocks < 1:
        raise ValueError(
            "cache budget must hold a null parent and one usable LCM parent"
        )

    max_packing = max(target_packing.values())
    token_limit = configured_token_limit(server_args)
    if token_limit is not None:
        requested = token_limit // logical_block_tokens // max_packing
        if requested < 1:
            raise ValueError(
                "the configured token limit must hold at least one LCM parent "
                f"({logical_block_tokens * max_packing} child tokens)"
            )
        num_lcm_blocks = min(num_lcm_blocks, requested)

    target_plan = target_layout.with_num_lcm_blocks(num_lcm_blocks)
    target_spec = CachePoolSpec(
        family=family,
        memory_plan=target_plan,
        layer_types=layer_types,
        layer_group_ids=group_ids,
        state_field_dtypes=state_dtypes,
        token_capacity=(num_lcm_blocks * max_packing * logical_block_tokens),
        layer_kv_head_counts=layer_kv_head_counts,
    )
    draft_spec = None
    if draft_layout is not None:
        draft_plan = draft_layout.with_num_lcm_blocks(num_lcm_blocks)
        draft_spec = CachePoolSpec(
            family=family,
            memory_plan=draft_plan,
            layer_types=draft_layer_types,
            layer_group_ids=draft_group_ids,
            state_field_dtypes={},
            token_capacity=target_spec.token_capacity,
            layer_kv_head_counts=draft_layer_kv_head_counts,
        )
    return CacheSetup(
        target=target_spec,
        draft=draft_spec,
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
    from tokenspeed.runtime.layers.attention.kv_cache.publish import (
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
    from tokenspeed.runtime.layers.attention.kv_cache.setup import (
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
    target_layout = solve_cache_layout(
        target_fields,
        logical_block_tokens=page_size,
        cache_blocks_per_lcm_block={field.group_id: 1 for field in target_fields},
        alignment=1,
        max_padding_fraction=1.0,
    )
    target_plan = target_layout.with_num_lcm_blocks(num_pages)
    token_capacity = num_pages * page_size
    target_spec = CachePoolSpec(
        family=family,
        memory_plan=target_plan,
        layer_types=tuple(getattr(attn_config, "layer_types", ())),
        layer_group_ids=target_group_ids,
        state_field_dtypes={},
        token_capacity=token_capacity,
        extra_paged_groups=tuple(getattr(attn_config, "extra_paged_groups", ())),
    )
    draft_spec = None
    if draft_fields is not None:
        draft_layout = solve_cache_layout(
            draft_fields,
            logical_block_tokens=page_size,
            cache_blocks_per_lcm_block={field.group_id: 1 for field in draft_fields},
            alignment=1,
            max_padding_fraction=1.0,
        )
        draft_plan = draft_layout.with_num_lcm_blocks(num_pages)
        draft_spec = CachePoolSpec(
            family=family,
            memory_plan=draft_plan,
            layer_types=tuple(getattr(draft_attn_config, "layer_types", ())),
            layer_group_ids=draft_group_ids,
            state_field_dtypes={},
            token_capacity=token_capacity,
            extra_paged_groups=tuple(
                getattr(draft_attn_config, "extra_paged_groups", ())
            ),
        )
    return CacheSetup(
        target=target_spec,
        draft=draft_spec,
        cache_budget_bytes=cache_budget_bytes,
        fixed_workspace_bytes=0,
    )


def _mha_fields(config, num_layers: int):
    import torch

    from tokenspeed.runtime.layers.attention.kv_cache import publish

    if config.kv_cache_mxfp8:
        assert config.page_size == 128, (
            "mxfp8 KV cache requires --block-size 128 (the attention "
            "kernel consumes the interleaved paged scale layout)"
        )
    layer_types = tuple(config.layer_types)
    group_ids = (
        tuple(
            publish.layer_group_ids(
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
        logical_block_tokens=config.page_size,
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


def prepare_mha_cache(
    *,
    server_args,
    model_config,
    attn_config,
    draft_model_config,
    draft_attn_config,
    cache_budget_bytes: int,
    decode_input_tokens: int,
    overlap_schedule_depth: int,
):
    """Build the ordinary MHA setup using the legacy capacity formula."""
    target_fields, target_group_ids = _mha_fields(
        attn_config, model_config.num_attention_layers
    )
    draft_fields = None
    draft_group_ids = ()
    if draft_attn_config is not None:
        draft_fields, draft_group_ids = _mha_fields(
            draft_attn_config, draft_model_config.num_attention_layers
        )
    return _ordinary_setup(
        family="mha",
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
        logical_block_tokens=config.page_size,
        latent_width=config.kv_lora_rank + config.qk_rope_head_dim,
        element_size=torch.empty((), dtype=config.kv_cache_dtype).element_size(),
    )


def prepare_mla_cache(
    *,
    server_args,
    model_config,
    attn_config,
    draft_model_config,
    draft_attn_config,
    cache_budget_bytes: int,
    decode_input_tokens: int,
    overlap_schedule_depth: int,
):
    """Build the ordinary MLA setup using the legacy capacity formula."""
    target_group_ids = ("full_attention",) * model_config.num_attention_layers
    target_fields = _mla_fields(attn_config, model_config.num_attention_layers)
    draft_fields = None
    draft_group_ids = ()
    if draft_attn_config is not None:
        draft_group_ids = ("full_attention",) * draft_model_config.num_attention_layers
        draft_fields = _mla_fields(
            draft_attn_config, draft_model_config.num_attention_layers
        )
    return _ordinary_setup(
        family="mla",
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


def prepare_dsa_cache(
    *,
    server_args,
    model_config,
    attn_config,
    draft_model_config,
    draft_attn_config,
    cache_budget_bytes: int,
    decode_input_tokens: int,
    overlap_schedule_depth: int,
):
    """Build the ordinary DSA setup with its index-key side cache."""
    target_group_ids = ("full_attention",) * model_config.num_attention_layers
    target_fields = _dsa_fields(attn_config, model_config.num_attention_layers)
    draft_fields = None
    draft_group_ids = ()
    if draft_attn_config is not None:
        draft_group_ids = ("full_attention",) * draft_model_config.num_attention_layers
        draft_fields = _dsa_fields(
            draft_attn_config, draft_model_config.num_attention_layers
        )
    return _ordinary_setup(
        family="dsa",
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


def prepare_msa_cache(
    *,
    server_args,
    model_config,
    attn_config,
    draft_model_config,
    draft_attn_config,
    cache_budget_bytes: int,
    decode_input_tokens: int,
    overlap_schedule_depth: int,
):
    """Build the ordinary MSA setup with sparse-layer index keys."""
    target_fields, target_group_ids = _msa_fields(
        attn_config, model_config.num_attention_layers
    )
    draft_fields = None
    draft_group_ids = ()
    if draft_attn_config is not None:
        draft_fields, draft_group_ids = _msa_fields(
            draft_attn_config, draft_model_config.num_attention_layers
        )
    return _ordinary_setup(
        family="msa",
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
