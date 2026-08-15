"""Inkling cache recipe."""

import os

from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import (
    build_hybrid_cache_setup,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    merge_continuation_layers,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    MXFP8_KV_SCALE_TILE_TOKENS,
)


def inkling_cache_fields(
    *,
    layer_group_ids,
    prefix_granularity,
    layer_kv_heads,
    head_dim,
    kv_element_size,
    hidden_size,
    checkpoint_rows,
    kvconv_element_size,
    hiddenconv_element_size,
    kv_scale_block_size=0,
    kv_scale_element_size=0,
) -> tuple[CacheFieldSpec, ...]:
    """Describe Inkling attention pages and ShortConv checkpoints."""
    if len(layer_group_ids) != len(layer_kv_heads):
        raise ValueError(
            f"layer_group_ids has {len(layer_group_ids)} entries but "
            f"layer_kv_heads has {len(layer_kv_heads)}"
        )
    if bool(kv_scale_block_size) != bool(kv_scale_element_size):
        raise ValueError(
            "kv_scale_block_size and kv_scale_element_size must both be zero "
            "or both be positive"
        )
    if kv_scale_block_size and head_dim % kv_scale_block_size:
        raise ValueError("head_dim must be divisible by kv_scale_block_size")
    if kv_scale_block_size and (
        prefix_granularity % 128 or head_dim != 128 or kv_scale_block_size != 32
    ):
        raise ValueError(
            "Inkling MXFP8 scale fields require P divisible by 128, "
            "head_dim 128 and scale block size 32"
        )

    occurrences: dict[str, int] = {}
    fields = []
    for layer_id, (group_id, kv_heads) in enumerate(
        zip(layer_group_ids, layer_kv_heads)
    ):
        unit = occurrences.get(group_id, 0)
        occurrences[group_id] = unit + 1
        k_plane = f"unit.{unit}.k"
        v_plane = f"unit.{unit}.v"
        if hiddenconv_element_size > 1:
            hidden_k_plane = f"unit.{unit}.hidden_k"
            hidden_v_plane = f"unit.{unit}.hidden_v"
        else:
            hidden_k_plane = k_plane
            hidden_v_plane = v_plane
        kv_shape = (prefix_granularity, kv_heads, head_dim)
        kvconv_shape = (checkpoint_rows, kv_heads * head_dim)
        hiddenconv_shape = (checkpoint_rows, hidden_size)
        fields.extend(
            (
                CacheFieldSpec(
                    group_id,
                    f"layer.{layer_id}.k",
                    k_plane,
                    kv_shape,
                    kv_element_size,
                ),
                CacheFieldSpec(
                    group_id,
                    f"layer.{layer_id}.v",
                    v_plane,
                    kv_shape,
                    kv_element_size,
                ),
                CacheFieldSpec(
                    "kvconv",
                    f"layer.{layer_id}.kvconv_k",
                    k_plane,
                    kvconv_shape,
                    kvconv_element_size,
                    exact_page_stride=False,
                ),
                CacheFieldSpec(
                    "kvconv",
                    f"layer.{layer_id}.kvconv_v",
                    v_plane,
                    kvconv_shape,
                    kvconv_element_size,
                    exact_page_stride=False,
                ),
                CacheFieldSpec(
                    "hiddenconv",
                    f"layer.{layer_id}.attnconv",
                    hidden_k_plane,
                    hiddenconv_shape,
                    hiddenconv_element_size,
                    exact_page_stride=False,
                ),
                CacheFieldSpec(
                    "hiddenconv",
                    f"layer.{layer_id}.mlpconv",
                    hidden_v_plane,
                    hiddenconv_shape,
                    hiddenconv_element_size,
                    exact_page_stride=False,
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


def inkling_layer_kv_head_counts(model_config) -> tuple[int, ...]:
    """Served KV heads per layer: each kind's native checkpoint count
    (hetero byte-uniform slots, #647), so page sizes derive from these."""
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


def _inkling_fields(attn_config, model_config, prefix_granularity: int):
    text_config = model_config.hf_config.get_text_config()
    layer_kv_head_counts = inkling_layer_kv_head_counts(model_config)
    per_rank_heads = tuple(
        max(1, heads // attn_config.attn_tp_size) for heads in layer_kv_head_counts
    )
    hiddenconv_element_size = (
        2 if os.environ.get("INKLING_FP8_SCONV", "0") == "0" else 1
    )
    fields = inkling_cache_fields(
        layer_group_ids=attn_config.layer_types,
        prefix_granularity=prefix_granularity,
        layer_kv_heads=per_rank_heads,
        head_dim=attn_config.head_dim,
        kv_element_size=attn_config.kv_cache_dtype.itemsize,
        hidden_size=text_config.hidden_size,
        checkpoint_rows=text_config.sconv_kernel_size - 1,
        kvconv_element_size=2,
        hiddenconv_element_size=hiddenconv_element_size,
        kv_scale_block_size=32 if attn_config.kv_cache_mxfp8 else 0,
        kv_scale_element_size=1 if attn_config.kv_cache_mxfp8 else 0,
    )
    return fields, layer_kv_head_counts


def _checkpoint_groups(prefix_granularity: int):
    # Physical packing is aligned with the memory plan by the pool at
    # construction; the recipe only declares the scheduler semantics.
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
        PagedCacheGroupSpec,
    )

    return tuple(
        PagedCacheGroupSpec(
            group_id=group_id,
            retention="full_history",
            sliding_window_tokens=None,
            family="state",
            checkpoint_granularity=prefix_granularity,
        )
        for group_id in ("kvconv", "hiddenconv")
    )


def _workspace_bytes(
    *,
    text_config,
    attn_config,
    num_layers: int,
    spec_tokens: int = 1,
) -> int:
    import torch

    from tokenspeed.runtime.configs.inkling_config import inkling_conv_total_dim

    rows = int(attn_config.max_bs) + 2
    # Must match _wrap_inkling_backend's ring sizing: (W-1) taps + K chunk rows.
    spec_tokens = max(1, int(spec_tokens))
    ring_rows = int(text_config.sconv_kernel_size) - 1 + spec_tokens
    conv_dim = inkling_conv_total_dim(text_config, attn_config.attn_tp_size)
    element_size = torch.bfloat16.itemsize
    pending_bytes = rows * torch.bool.itemsize
    return num_layers * rows * ring_rows * conv_dim * element_size + pending_bytes


def prepare_inkling_cache(
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
    """Build target and optional draft cache specs for Inkling."""
    prefix_granularity = 128
    fields, layer_kv_head_counts = _inkling_fields(
        attn_config, model_config, prefix_granularity
    )
    layer_types = tuple(attn_config.layer_types)
    text_config = model_config.hf_config.get_text_config()
    draft_tokens = (
        int(server_args.speculative_num_draft_tokens)
        if draft_attn_config is not None
        else 0
    )
    fixed_workspace_bytes = _workspace_bytes(
        text_config=text_config,
        attn_config=attn_config,
        num_layers=text_config.num_hidden_layers,
        spec_tokens=draft_tokens,
    )

    draft_fields = None
    draft_layer_types = ()
    draft_group_ids = ()
    draft_layer_kv_head_counts = None
    if draft_attn_config is not None:
        draft_num_layers = draft_model_config.num_attention_layers
        num_steps = server_args.speculative_num_steps
        if num_steps > draft_num_layers:
            raise ValueError(
                f"Inkling MTP has {draft_num_layers} depth layers; "
                f"--speculative-num-steps {num_steps} would wrap depths "
                "with no trained meaning."
            )
        draft_layer_types = tuple(draft_attn_config.layer_types)
        draft_group_ids = draft_layer_types
        # Same recipe as the target: the depth layers get kvconv/hiddenconv
        # checkpoint fields as continuation tenants of the existing groups,
        # so the draft conv ring gains the same publish/restore bridges.
        draft_fields, draft_layer_kv_head_counts = _inkling_fields(
            draft_attn_config, draft_model_config, prefix_granularity
        )
        fixed_workspace_bytes += _workspace_bytes(
            text_config=draft_model_config.hf_config.get_text_config(),
            attn_config=draft_attn_config,
            num_layers=draft_num_layers,
            spec_tokens=draft_tokens,
        )

    max_padding_fraction = (
        float("inf") if os.environ.get("INKLING_FP8_SCONV", "0") == "0" else 1.0
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
        apply_pd_transfer_policies,
        build_paged_cache_group_specs,
    )

    (
        merged_fields,
        merged_layer_types,
        merged_group_ids,
        merged_head_counts,
        num_draft_layers,
    ) = merge_continuation_layers(
        fields=fields,
        layer_types=layer_types,
        group_ids=layer_types,
        layer_kv_head_counts=layer_kv_head_counts,
        draft_fields=draft_fields,
        draft_layer_types=draft_layer_types,
        draft_group_ids=draft_group_ids,
        draft_layer_kv_head_counts=draft_layer_kv_head_counts,
    )
    # ONE spec derivation over the merged layers (shared groups validate
    # for consistent policy), plus the paged sconv checkpoint groups
    # (per-layer state columns outside the layer-type vocabulary); PD
    # policies stamp the complete tuple. Target and draft share the
    # sliding window width by construction (same architecture family).
    group_specs = (
        *build_paged_cache_group_specs(
            layer_types=merged_layer_types,
            group_ids=merged_group_ids,
            sliding_window_tokens=attn_config.sliding_window_tokens,
            prefix_granularity=prefix_granularity,
        ),
        *_checkpoint_groups(prefix_granularity),
    )
    if attn_config.pd_disaggregation_enabled:
        group_specs = tuple(apply_pd_transfer_policies(group_specs))
    return build_hybrid_cache_setup(
        family="inkling",
        server_args=server_args,
        fields=merged_fields,
        layer_types=merged_layer_types,
        group_ids=merged_group_ids,
        group_specs=group_specs,
        state_dtypes={},
        layer_kv_head_counts=merged_head_counts,
        num_draft_layers=num_draft_layers,
        cache_budget_bytes=cache_budget_bytes,
        fixed_workspace_bytes=fixed_workspace_bytes,
        prefix_granularity=prefix_granularity,
        max_padding_fraction=max_padding_fraction,
    )
