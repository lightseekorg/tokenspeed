"""DeepSeek V4 cache field recipe."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tokenspeed.runtime.layers.attention.kv_cache.recipes import (
    configured_token_limit,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.deepseek_v4_cache_spec import (
    DEEPSEEK_V4_COMPRESSED_LOGICAL_BLOCK_SIZE,
    V4_KERNEL_BLOCK_ROWS,
    DeepseekV4CacheLayout,
    build_v4_cache_specs,
    deepseek_v4_cache_layout_from_config,
    deepseek_v4_lcm_blocks_needed,
    deepseek_v4_token_capacity_for_cache_pool,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    solve_cache_layout,
)

_LOGICAL_BLOCK_TOKENS = 256
_MAX_PADDING_FRACTION = 2.0


@dataclass(frozen=True)
class DeepseekV4PoolOptions:
    layout: DeepseekV4CacheLayout


def build_deepseek_v4_cache_fields(
    layout: DeepseekV4CacheLayout,
    *,
    sliding_window: int,
    logical_block_tokens: int = DEEPSEEK_V4_COMPRESSED_LOGICAL_BLOCK_SIZE,
):
    """Build DSV4 fields for one scheduler-wide logical page size."""
    if sliding_window <= 0 or logical_block_tokens % sliding_window:
        raise ValueError(
            "DeepSeek V4 sliding_window must divide the logical block size"
        )
    if sliding_window % V4_KERNEL_BLOCK_ROWS:
        raise ValueError(
            "DeepSeek V4 sliding_window must be divisible by the SWA kernel page"
        )

    compressed_shapes = {
        ratio: (layout.swa_block_bytes(layout.storage_block_size(ratio)),)
        for ratio in set(layout.layer_ratio)
        if ratio > 1
    }
    compressor_state_shapes = {
        ratio: (
            layout.compressor_state_block_size(ratio),
            layout.head_dim * (2 if ratio == 4 else 1) * 2,
        )
        for ratio in set(layout.layer_ratio)
        if ratio > 1
    }
    return deepseek_v4_cache_fields(
        layer_ratios=layout.layer_ratio,
        logical_block_tokens=logical_block_tokens,
        swa_shape=(layout.swa_block_bytes(V4_KERNEL_BLOCK_ROWS),),
        compressed_shapes=compressed_shapes,
        compressor_state_shapes=compressor_state_shapes,
        indexer_kv_shape=(V4_KERNEL_BLOCK_ROWS, layout.indexer_row_bytes),
        indexer_state_shape=(
            layout.compressor_state_block_size(4),
            layout.index_head_dim * 2 * 2,
        ),
        kv_page_stride_alignment_bytes=layout.swa_token_stride,
    )


def solve_deepseek_v4_memory_layout(fields):
    """Solve DSV4's power-of-two group packing for one physical parent."""
    raw_bytes: dict[str, int] = {}
    for field in fields:
        raw_bytes[field.group_id] = (
            raw_bytes.get(field.group_id, 0) + field.payload_bytes
        )
    largest_group = max(raw_bytes.values())
    # Power-of-two packing keeps every field stride naturally aligned. Using
    # exact byte ratios would inflate a parent through their large common LCM.
    packing = {
        group_id: 1 << max(0, (largest_group // group_bytes).bit_length() - 1)
        for group_id, group_bytes in raw_bytes.items()
    }
    return solve_cache_layout(
        fields,
        logical_block_tokens=_LOGICAL_BLOCK_TOKENS,
        cache_blocks_per_lcm_block=packing,
        alignment=256,
        max_padding_fraction=_MAX_PADDING_FRACTION,
    )


def deepseek_v4_cache_fields(
    *,
    layer_ratios: Sequence[int],
    logical_block_tokens: int,
    swa_shape: Sequence[int],
    compressed_shapes: Mapping[int, tuple[int, ...]],
    compressor_state_shapes: Mapping[int, tuple[int, ...]],
    indexer_kv_shape: Sequence[int],
    indexer_state_shape: Sequence[int],
    kv_page_stride_alignment_bytes: int,
) -> tuple[CacheFieldSpec, ...]:
    """Describe DSV4 history pages and bounded live-tail state."""
    if logical_block_tokens != 256:
        raise ValueError("DeepSeek V4 LCM scheduling requires P=256")
    ratios = tuple(int(ratio) for ratio in layer_ratios)
    if any(ratio not in (1, 4, 128) for ratio in ratios):
        raise ValueError("DeepSeek V4 layer ratios must be 1, 4, or 128")

    fields: list[CacheFieldSpec] = []
    ratio_counts = Counter(ratios)
    occurrences: dict[str, int] = {}
    for layer_id, ratio in enumerate(ratios):
        swa_group = "v4.swa_kv"
        swa_slot = occurrences.get(swa_group, 0)
        occurrences[swa_group] = swa_slot + 1
        fields.append(
            CacheFieldSpec(
                swa_group,
                f"layer.{layer_id}.swa",
                f"unit.{swa_slot}",
                tuple(swa_shape),
                1,
                exact_page_stride=False,
                page_stride_alignment_bytes=kv_page_stride_alignment_bytes,
            )
        )
        if ratio == 1:
            continue

        compressed_group = f"v4.c{ratio}a.compressed_kv"
        state_group = f"v4.c{ratio}a.compressor_state"
        try:
            compressed_shape = tuple(compressed_shapes[ratio])
            state_shape = tuple(compressor_state_shapes[ratio])
        except KeyError as exc:
            raise ValueError(f"missing DeepSeek V4 ratio-{ratio} geometry") from exc
        compressed_slot = occurrences.get(compressed_group, 0)
        occurrences[compressed_group] = compressed_slot + 1
        state_slot = occurrences.get(state_group, 0)
        occurrences[state_group] = state_slot + 1
        fields.extend(
            (
                CacheFieldSpec(
                    compressed_group,
                    f"layer.{layer_id}.compressed_kv",
                    f"unit.{compressed_slot}",
                    compressed_shape,
                    1,
                    exact_page_stride=False,
                    page_stride_alignment_bytes=kv_page_stride_alignment_bytes,
                ),
                CacheFieldSpec(
                    state_group,
                    f"layer.{layer_id}.compressor_state",
                    f"unit.{state_slot}",
                    state_shape,
                    4,
                    exact_page_stride=False,
                ),
            )
        )
        if ratio != 4:
            continue
        indexer_state_group = "v4.c4a.indexer_compressor_state"
        indexer_state_slot = occurrences.get(indexer_state_group, 0)
        occurrences[indexer_state_group] = indexer_state_slot + 1
        fields.extend(
            (
                CacheFieldSpec(
                    compressed_group,
                    f"layer.{layer_id}.indexer_kv",
                    f"unit.{ratio_counts[4] + compressed_slot}",
                    tuple(indexer_kv_shape),
                    1,
                    exact_page_stride=False,
                ),
                CacheFieldSpec(
                    indexer_state_group,
                    f"layer.{layer_id}.indexer_state",
                    f"unit.{indexer_state_slot}",
                    tuple(indexer_state_shape),
                    4,
                    exact_page_stride=False,
                ),
            )
        )
    return tuple(fields)


def prepare_deepseek_v4_cache(
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
    """Build target and draft cache specs for DeepSeek V4."""
    # Deferred: setup.py imports this recipe at module load.
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import (
        CachePoolSpec,
        CacheSetup,
    )

    def use_fp4_indexer(hf_config) -> bool:
        override = getattr(server_args, "attention_use_fp4_indexer_cache", None)
        if override is not None:
            return bool(override)
        attention_config = getattr(hf_config, "attention_config", None)
        if isinstance(attention_config, dict):
            return bool(attention_config.get("use_fp4_indexer_cache", False))
        return bool(getattr(attention_config, "use_fp4_indexer_cache", False))

    def layout_and_fields(config, runtime_config, *, layer_indices=None):
        layout = deepseek_v4_cache_layout_from_config(
            config,
            page_size=runtime_config.page_size,
            use_fp4_indexer_cache=use_fp4_indexer(config),
            layer_indices=layer_indices,
        )
        fields = build_deepseek_v4_cache_fields(
            layout,
            sliding_window=int(config.sliding_window),
            logical_block_tokens=_LOGICAL_BLOCK_TOKENS,
        )
        return layout, fields

    # One big model, one solve, one spec: V4's MTP layer already IS a
    # continuation layer of the target config (compress_ratios carries
    # num_hidden_layers + num_nextn entries, the draft layer indexed after
    # the target's) — build the merged fields in one pass over the full
    # layer range.
    num_target_layers = model_config.num_attention_layers
    num_draft_layers = (
        draft_model_config.num_attention_layers if draft_attn_config is not None else 0
    )
    merged_source = (
        draft_model_config.hf_config
        if draft_attn_config is not None
        else model_config.hf_config
    )
    merged_layout, merged_fields = layout_and_fields(
        merged_source,
        attn_config,
        layer_indices=range(num_target_layers + num_draft_layers),
    )
    merged_layout_plan = solve_deepseek_v4_memory_layout(merged_fields)
    packing = dict(merged_layout_plan.group_packing)

    num_lcm_blocks = cache_budget_bytes // merged_layout_plan.lcm_block_bytes - 1
    if num_lcm_blocks < 1:
        raise ValueError(
            "DeepSeek V4 cache budget must hold a null parent and one usable parent"
        )

    specs = tuple(
        build_v4_cache_specs(
            model_config.hf_config,
            layer_ratio=merged_layout.layer_ratio,
            cache_blocks_per_lcm_block=packing,
            decode_input_tokens=decode_input_tokens,
        )
    )
    max_packing = max(packing.values())
    token_limit = configured_token_limit(server_args)
    upper_bound = (
        token_limit
        if token_limit is not None
        else num_lcm_blocks * max_packing * _LOGICAL_BLOCK_TOKENS
    )
    sizing = {
        "logical_block_tokens": _LOGICAL_BLOCK_TOKENS,
        "max_live_requests": attn_config.max_bs,
        "max_scheduled_tokens": max(0, int(server_args.chunked_prefill_size)),
        "max_context_len": attn_config.context_len,
        "decode_input_tokens": decode_input_tokens,
        "overlap_schedule_depth": overlap_schedule_depth,
    }
    token_capacity = deepseek_v4_token_capacity_for_cache_pool(
        specs,
        num_lcm_blocks=num_lcm_blocks,
        upper_bound_tokens=upper_bound,
        **sizing,
    )
    num_lcm_blocks = deepseek_v4_lcm_blocks_needed(
        specs,
        token_capacity=token_capacity,
        **sizing,
    )

    return CacheSetup(
        spec=CachePoolSpec(
            family="deepseek_v4",
            memory_plan=merged_layout_plan.with_num_lcm_blocks(num_lcm_blocks),
            layer_types=tuple(str(ratio) for ratio in merged_layout.layer_ratio),
            layer_group_ids=tuple(
                (f"v4.c{ratio}a.compressed_kv" if ratio > 1 else "v4.swa_kv")
                for ratio in merged_layout.layer_ratio
            ),
            paged_cache_group_specs=specs,
            state_field_dtypes={},
            token_capacity=token_capacity,
            pool_options=DeepseekV4PoolOptions(layout=merged_layout),
        ),
        num_draft_layers=num_draft_layers,
        cache_budget_bytes=cache_budget_bytes,
        fixed_workspace_bytes=0,
    )
