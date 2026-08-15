"""Qwen3.5 cache field recipe."""

import torch

from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import (
    build_hybrid_cache_setup,
    draft_cache_fields,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    merge_continuation_layers,
)


def qwen_gdn_max_padding_fraction(*, layer_types, num_draft_layers: int) -> float:
    """Allow the structural K/V planes added by a Qwen MTP draft.

    Args:
        layer_types: Target-only attention labels, before merging draft layers.
        num_draft_layers: Number of full-attention MTP layers being merged.

    Returns:
        The Qwen-specific upper bound on unified-arena padding.

    Raises:
        ValueError: If the target has no full-attention layers.
    """
    num_full_attention_layers = sum(
        layer_type == "full_attention" for layer_type in layer_types
    )
    if num_full_attention_layers == 0:
        raise ValueError("Qwen3.5 cache requires at least one full-attention layer")

    # If target-only recurrent padding is p <= 1, each mirrored draft layer's
    # K/V planes increase it by (1 + p) / num_full_attention_layers.  Bound
    # that increase by 2 without relaxing the original limit when no draft is
    # present.  The derivation margin is intentionally the only headroom: if
    # future cache geometry trips the guard, re-derive this bound rather than
    # adding an epsilon or silently accepting an unbounded binding hole.
    return 1.0 + 2.0 * num_draft_layers / num_full_attention_layers


def qwen_gdn_cache_fields(
    *,
    layer_types,
    layer_group_ids,
    prefix_granularity,
    kv_shape,
    kv_element_size,
    conv_shape,
    conv_element_size,
    ssm_shape,
    ssm_element_size,
) -> tuple[CacheFieldSpec, ...]:
    """Describe Qwen3.5 full-attention KV and recurrent checkpoints."""
    if len(layer_types) != len(layer_group_ids):
        raise ValueError(
            f"layer_types has {len(layer_types)} entries but layer_group_ids "
            f"has {len(layer_group_ids)}"
        )
    if next(iter(kv_shape)) != prefix_granularity:
        raise ValueError("kv_shape must start with prefix_granularity")
    occurrences: dict[str, int] = {}
    fields = []
    for layer_id, (label, group_id) in enumerate(zip(layer_types, layer_group_ids)):
        unit = occurrences.get(group_id, 0)
        occurrences[group_id] = unit + 1
        if label == "linear_attention":
            fields.extend(
                (
                    CacheFieldSpec(
                        group_id,
                        f"layer.{layer_id}.ssm",
                        f"unit.{unit}.a",
                        tuple(ssm_shape),
                        ssm_element_size,
                    ),
                    CacheFieldSpec(
                        group_id,
                        f"layer.{layer_id}.conv",
                        f"unit.{unit}.b",
                        tuple(conv_shape),
                        conv_element_size,
                        exact_page_stride=False,
                    ),
                )
            )
        else:
            fields.extend(
                (
                    CacheFieldSpec(
                        group_id,
                        f"layer.{layer_id}.k",
                        f"unit.{unit}.a",
                        tuple(kv_shape),
                        kv_element_size,
                    ),
                    CacheFieldSpec(
                        group_id,
                        f"layer.{layer_id}.v",
                        f"unit.{unit}.b",
                        tuple(kv_shape),
                        kv_element_size,
                    ),
                )
            )
    return tuple(fields)


def prepare_qwen35_cache(
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
    """Build target and optional draft cache specs for Qwen3.5."""
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
        FULL_ATTENTION,
        LINEAR_ATTENTION,
        build_paged_cache_group_specs,
        split_recurrent_state_groups,
    )

    if attn_config.kv_cache_mxfp8:
        raise RuntimeError(
            "Qwen cache buffer does not yet support the MXFP8 interleaved scale layout"
        )
    prefix_granularity = 128
    text_config = getattr(model_config.hf_config, "text_config", model_config.hf_config)
    conv_shape, ssm_shape, conv_dtype, ssm_dtype, _ = text_config.mamba2_cache_params
    layer_types = tuple(attn_config.layer_types)
    group_ids = tuple(split_recurrent_state_groups(layer_types))
    fields = qwen_gdn_cache_fields(
        layer_types=layer_types,
        layer_group_ids=group_ids,
        prefix_granularity=prefix_granularity,
        kv_shape=(
            prefix_granularity,
            max(attn_config.num_kv_heads // attn_config.attn_tp_size, 1),
            attn_config.head_dim,
        ),
        kv_element_size=attn_config.kv_cache_dtype.itemsize,
        conv_shape=conv_shape,
        conv_element_size=conv_dtype.itemsize,
        ssm_shape=ssm_shape,
        ssm_element_size=ssm_dtype.itemsize,
    )
    state_dtypes = {
        f"layer.{layer_id}.conv": conv_dtype
        for layer_id, layer_type in enumerate(layer_types)
        if layer_type == LINEAR_ATTENTION
    } | {
        f"layer.{layer_id}.ssm": ssm_dtype
        for layer_id, layer_type in enumerate(layer_types)
        if layer_type == LINEAR_ATTENTION
    }

    draft_fields = None
    draft_layer_types = ()
    draft_group_ids = ()
    fixed_workspace_bytes = 0
    replay_ssm = False
    if draft_attn_config is not None:
        draft_num_layers = draft_model_config.num_attention_layers
        draft_layer_types = (FULL_ATTENTION,) * draft_num_layers
        draft_group_ids = draft_layer_types
        per_rank_heads = (
            max(
                draft_attn_config.num_kv_heads // draft_attn_config.attn_tp_size,
                1,
            ),
        ) * draft_num_layers
        draft_fields = draft_cache_fields(
            layer_group_ids=draft_group_ids,
            enabled_layer_ids=range(draft_num_layers),
            prefix_granularity=prefix_granularity,
            layer_kv_heads=per_rank_heads,
            head_dim=draft_attn_config.head_dim,
            kv_element_size=draft_attn_config.kv_cache_dtype.itemsize,
            kv_scale_block_size=32 if draft_attn_config.kv_cache_mxfp8 else 0,
            kv_scale_element_size=1 if draft_attn_config.kv_cache_mxfp8 else 0,
        )
        verify_rows = attn_config.max_bs * (
            int(server_args.speculative_num_draft_tokens) + 1
        )
        replay_ssm = (
            getattr(server_args, "enable_replay_ssm", False)
            and int(server_args.speculative_num_draft_tokens) > 1
            and torch.device(attn_config.device).type == "cuda"
        )
        if replay_ssm:
            from tokenspeed_kernel.ops.attention import gdn_replay_commit_supported

            replay_ssm = gdn_replay_commit_supported(attn_config.dtype)
        workspace_suffixes = (".conv",) if replay_ssm else (".conv", ".ssm")
        fixed_workspace_bytes = verify_rows * sum(
            field.payload_bytes
            for field in fields
            if field.field_id.endswith(workspace_suffixes)
        )
    attn_config.replay_ssm = replay_ssm

    (
        merged_fields,
        merged_layer_types,
        merged_group_ids,
        _,
        num_draft_layers,
    ) = merge_continuation_layers(
        fields=fields,
        layer_types=layer_types,
        group_ids=group_ids,
        draft_fields=draft_fields,
        draft_layer_types=draft_layer_types,
        draft_group_ids=draft_group_ids,
    )
    # ONE spec derivation over the merged layers: shared groups validate
    # for consistent policy instead of the draft's being silently dropped.
    group_specs = build_paged_cache_group_specs(
        layer_types=merged_layer_types,
        group_ids=merged_group_ids,
        sliding_window_tokens=None,
        prefix_granularity=prefix_granularity,
        pd_disaggregation_enabled=attn_config.pd_disaggregation_enabled,
    )
    return build_hybrid_cache_setup(
        family="qwen_gdn",
        server_args=server_args,
        fields=merged_fields,
        layer_types=merged_layer_types,
        group_ids=merged_group_ids,
        group_specs=group_specs,
        state_dtypes=state_dtypes,
        layer_kv_head_counts=None,
        num_draft_layers=num_draft_layers,
        cache_budget_bytes=cache_budget_bytes,
        fixed_workspace_bytes=fixed_workspace_bytes,
        prefix_granularity=prefix_granularity,
        max_padding_fraction=qwen_gdn_max_padding_fraction(
            layer_types=layer_types,
            num_draft_layers=num_draft_layers,
        ),
    )
