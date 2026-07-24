# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Any

import numpy as np
import torch

from tokenspeed.runtime.configs.deepseek_v4_cache_spec import (
    CACHE_OWNER_DRAFT,
    CACHE_OWNER_TARGET,
    DEEPSEEK_V4_COMPRESSED_LOGICAL_BLOCK_SIZE,
    V4_INDEXER_COMPRESSOR_STATE_GROUP_ID,
    V4_KERNEL_BLOCK_ROWS,
    V4_SWA_KV_GROUP_ID,
    build_v4_cache_specs,
    deepseek_v4_indexer_fp8_row_bytes,
    deepseek_v4_indexer_mxfp4_row_bytes,
    deepseek_v4_swa_scale_dim,
    deepseek_v4_swa_token_stride,
    parse_v4_compressor_state_group_id,
    v4_compressed_kv_group_id,
    v4_compressor_state_group_id,
)
from tokenspeed.runtime.configs.deepseek_v4_flat_memory_plan import (
    V4FlatMemoryPlan,
)
from tokenspeed.runtime.configs.deepseek_v4_flat_memory_plan import (
    build_v4_flat_memory_plan as _build_v4_flat_memory_plan,
)
from tokenspeed.runtime.configs.flat_memory_plan import (
    FlatCacheOwner,
    FlatComponentTensorPlan,
    FlatGroupTablePlan,
    FlatRuntimeMetadataPlan,
)
from tokenspeed.runtime.configs.paged_cache_spec import (
    PagedCacheGroupSpec,
    compute_flat_capture_cols,
    compute_flat_export_cols,
    compute_paged_cache_group_page_counts,
)
from tokenspeed.runtime.layers.attention.deepseek_v4_ops import (
    deepseek_v4_compressed_slot_mapping,
)
from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.common import ceil_div
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = get_colorful_logger(__name__)


@dataclass(frozen=True)
class DeepseekV4CacheLayout:
    layer_ratio: tuple[int, ...]
    head_dim: int
    rope_head_dim: int
    page_size: int
    use_fp4_indexer_cache: bool
    index_head_dim: int = 128

    @property
    def swa_token_stride(self) -> int:
        return deepseek_v4_swa_token_stride(self.head_dim, self.rope_head_dim)

    @property
    def swa_scale_dim(self) -> int:
        return deepseek_v4_swa_scale_dim(self.head_dim, self.rope_head_dim)

    @property
    def swa_row_bytes(self) -> int:
        return self.swa_token_stride + self.swa_scale_dim

    def swa_block_bytes(self, rows_per_page: int | None = None) -> int:
        if rows_per_page is None:
            rows_per_page = self.page_size
        block_bytes = rows_per_page * self.swa_row_bytes
        alignment = self.swa_token_stride
        return ((block_bytes + alignment - 1) // alignment) * alignment

    def swa_cell_bytes(self) -> int:
        block_bytes = self.swa_block_bytes()
        return (block_bytes + self.page_size - 1) // self.page_size

    def storage_block_size(self, compress_ratio: int) -> int:
        if compress_ratio > 1:
            return max(1, DEEPSEEK_V4_COMPRESSED_LOGICAL_BLOCK_SIZE // compress_ratio)
        return self.page_size

    def compressed_cell_bytes(self, compress_ratio: int) -> int:
        block_bytes = self.swa_block_bytes(self.storage_block_size(compress_ratio))
        return (block_bytes + self.page_size - 1) // self.page_size

    @property
    def indexer_row_bytes(self) -> int:
        if self.use_fp4_indexer_cache:
            return deepseek_v4_indexer_mxfp4_row_bytes(self.index_head_dim)
        return deepseek_v4_indexer_fp8_row_bytes(self.index_head_dim)

    def state_width(self, layer_id: int, *, indexer: bool = False) -> int:
        if indexer:
            return self.index_head_dim * 2
        return self.head_dim * (2 if self.layer_ratio[layer_id] == 4 else 1)

    def cache_cell_size(self, layer_num: int | None = None) -> int:
        """Return bytes per token for the current V4 cache allocation layout."""
        if layer_num is None:
            layer_num = len(self.layer_ratio)
        if layer_num > len(self.layer_ratio):
            raise ValueError(
                "DeepSeek V4 cache layout has fewer layer ratios "
                f"({len(self.layer_ratio)}) than requested layers ({layer_num})"
            )

        fp32_size = torch._utils._element_size(torch.float32)
        cell_size = 0
        for layer_id in range(layer_num):
            ratio = self.layer_ratio[layer_id]
            cell_size += self.swa_cell_bytes()
            if ratio > 1:
                cell_size += self.compressed_cell_bytes(ratio)
                cell_size += self.state_width(layer_id) * 2 * fp32_size
            if ratio == 4:
                indexer_block_bytes = (
                    self.storage_block_size(ratio) * self.indexer_row_bytes
                )
                cell_size += (
                    indexer_block_bytes + self.page_size - 1
                ) // self.page_size
                cell_size += self.state_width(layer_id, indexer=True) * 2 * fp32_size
        return cell_size


_V4_SWA_COMPONENT = "swa_kv"
_V4_COMPRESSED_COMPONENT = "compressed_kv"
_V4_COMPRESSOR_STATE_COMPONENT = "compressor_state"
_V4_INDEXER_COMPONENT = "indexer_kv"
_V4_INDEXER_STATE_COMPONENT = "indexer_state"
_V4_FLAT_DTYPE_BYTES = {"uint8": 1, "float32": 4}


def _contiguous_stride_bytes(
    shape: tuple[int, ...],
    itemsize: int,
) -> tuple[int, ...]:
    strides: list[int] = []
    running = itemsize
    for dimension in reversed(shape):
        strides.append(running)
        running *= dimension
    return tuple(reversed(strides))


def _flat_component_plan(
    *,
    owner: FlatCacheOwner,
    group_id: str,
    layer: int,
    component: str,
    dtype: str,
    shape_per_block: tuple[int, ...],
) -> FlatComponentTensorPlan:
    itemsize = _V4_FLAT_DTYPE_BYTES[dtype]
    bytes_per_block = itemsize
    for dimension in shape_per_block:
        bytes_per_block *= dimension
    return FlatComponentTensorPlan(
        owner=owner,
        group_id=group_id,
        layer=layer,
        component=component,
        dtype=dtype,
        shape_per_block=shape_per_block,
        stride_bytes=_contiguous_stride_bytes(shape_per_block, itemsize),
        alignment_bytes=itemsize,
        bytes_per_block=bytes_per_block,
    )


def deepseek_v4_flat_component_plans(
    *,
    layout: DeepseekV4CacheLayout,
    specs: Sequence[PagedCacheGroupSpec],
    layer_num: int,
    owner: FlatCacheOwner,
) -> tuple[FlatComponentTensorPlan, ...]:
    """Describe every independently allocated V4 component plane.

    Args:
        layout: Owner-local ratios and packed per-row storage schema.
        specs: Owner-local groups; their ``rows_per_page`` is the sole page-row
            geometry used for every component shape.
        layer_num: Number of owner-local attention layers.
        owner: Target or draft tensor namespace.

    Returns:
        Canonically sortable component schemas with trailing per-block shape,
        byte strides, alignment, and exact bytes per physical block.
    """
    if layer_num != len(layout.layer_ratio):
        raise ValueError(
            "DeepSeek V4 flat component layer_num must match layout ratios: "
            f"layer_num={layer_num}, ratios={len(layout.layer_ratio)}"
        )
    specs_by_id = {spec.group_id: spec for spec in specs}
    if len(specs_by_id) != len(specs):
        raise ValueError("DeepSeek V4 flat component specs contain duplicate groups")

    def _group_rows(group_id: str) -> int:
        spec = specs_by_id.get(group_id)
        if spec is None:
            raise ValueError(
                f"DeepSeek V4 flat component is missing group spec {group_id!r}"
            )
        rows_per_page = int(spec.rows_per_page)
        if rows_per_page <= 0:
            raise ValueError(
                f"DeepSeek V4 flat group {group_id!r} rows_per_page must be positive"
            )
        return rows_per_page

    swa_rows_per_page = _group_rows(V4_SWA_KV_GROUP_ID)
    components: list[FlatComponentTensorPlan] = []
    for layer_id, ratio in enumerate(layout.layer_ratio):
        components.append(
            _flat_component_plan(
                owner=owner,
                group_id=V4_SWA_KV_GROUP_ID,
                layer=layer_id,
                component=_V4_SWA_COMPONENT,
                dtype="uint8",
                shape_per_block=(layout.swa_block_bytes(swa_rows_per_page),),
            )
        )
        if ratio <= 1:
            continue

        compressed_group_id = v4_compressed_kv_group_id(ratio)
        compressor_state_group_id = v4_compressor_state_group_id(ratio)
        compressed_rows_per_page = _group_rows(compressed_group_id)
        compressor_state_rows_per_page = _group_rows(compressor_state_group_id)
        components.extend(
            (
                _flat_component_plan(
                    owner=owner,
                    group_id=compressed_group_id,
                    layer=layer_id,
                    component=_V4_COMPRESSED_COMPONENT,
                    dtype="uint8",
                    shape_per_block=(layout.swa_block_bytes(compressed_rows_per_page),),
                ),
                _flat_component_plan(
                    owner=owner,
                    group_id=compressor_state_group_id,
                    layer=layer_id,
                    component=_V4_COMPRESSOR_STATE_COMPONENT,
                    dtype="float32",
                    shape_per_block=(
                        compressor_state_rows_per_page,
                        layout.state_width(layer_id) * 2,
                    ),
                ),
            )
        )
        if ratio != 4:
            continue

        indexer_state_rows_per_page = _group_rows(V4_INDEXER_COMPRESSOR_STATE_GROUP_ID)
        components.extend(
            (
                _flat_component_plan(
                    owner=owner,
                    group_id=compressed_group_id,
                    layer=layer_id,
                    component=_V4_INDEXER_COMPONENT,
                    dtype="uint8",
                    shape_per_block=(
                        compressed_rows_per_page * layout.indexer_row_bytes,
                    ),
                ),
                _flat_component_plan(
                    owner=owner,
                    group_id=V4_INDEXER_COMPRESSOR_STATE_GROUP_ID,
                    layer=layer_id,
                    component=_V4_INDEXER_STATE_COMPONENT,
                    dtype="float32",
                    shape_per_block=(
                        indexer_state_rows_per_page,
                        layout.state_width(layer_id, indexer=True) * 2,
                    ),
                ),
            )
        )
    return tuple(components)


def _validate_v4_flat_plan_components(
    plan: V4FlatMemoryPlan,
    *,
    specs: Sequence[PagedCacheGroupSpec],
    expected_components: Sequence[FlatComponentTensorPlan],
    owner: FlatCacheOwner,
) -> None:
    """Fail closed if a plan drifts from the spec-derived storage schema."""
    pool_by_group = {spec.group_id: spec.pool_id for spec in specs}
    expected = {
        (
            pool_by_group[component.group_id],
            component.group_id,
            component.layer,
            component.component,
        ): component
        for component in expected_components
    }
    actual = {
        (
            pool.pool_id,
            component.group_id,
            component.layer,
            component.component,
        ): component
        for pool in plan.pools
        for component in pool.tensors
        if component.owner == owner
    }
    if actual == expected:
        return

    missing = sorted(expected.keys() - actual.keys())
    extra = sorted(actual.keys() - expected.keys())
    if missing or extra:
        raise ValueError(
            f"DeepSeek V4 flat {owner} component geometry disagrees with specs: "
            f"missing={missing}, extra={extra}"
        )
    for identity in sorted(expected):
        if actual[identity] != expected[identity]:
            raise ValueError(
                f"DeepSeek V4 flat {owner} component geometry disagrees with "
                f"specs for {identity}: expected={expected[identity]!r}, "
                f"actual={actual[identity]!r}"
            )
    raise AssertionError("unreachable component geometry comparison")


def _deepseek_v4_cache_group_page_bytes(
    layout: DeepseekV4CacheLayout,
    specs: Sequence[PagedCacheGroupSpec],
    layer_num: int,
) -> dict[str, int]:
    """Sum canonical component-plane bytes for each physical group page."""
    page_bytes = {spec.group_id: 0 for spec in specs}
    components = deepseek_v4_flat_component_plans(
        layout=layout,
        specs=specs,
        layer_num=layer_num,
        owner="target",
    )
    for component in components:
        page_bytes[component.group_id] += component.bytes_per_block

    return page_bytes


def _estimate_deepseek_v4_cache_bytes(
    *,
    layout: DeepseekV4CacheLayout,
    hf_config: Any,
    layer_num: int,
    max_total_tokens: int,
    max_live_requests: int,
    max_scheduled_tokens: int,
    max_context_len: int,
) -> int:
    """Estimate bytes allocated by DeepseekV4TokenToKVPool for a token budget."""
    if layer_num > len(layout.layer_ratio):
        raise ValueError(
            "DeepSeek V4 cache layout has fewer layer ratios "
            f"({len(layout.layer_ratio)}) than requested layers ({layer_num})"
        )
    if max_total_tokens < 0:
        raise ValueError(f"max_total_tokens must be >= 0, got {max_total_tokens}")

    specs = tuple(build_v4_cache_specs(hf_config, layer_ratio=layout.layer_ratio))
    page_bytes = _deepseek_v4_cache_group_page_bytes(layout, specs, layer_num)
    counts = compute_paged_cache_group_page_counts(
        specs,
        max_live_requests=max_live_requests,
        max_scheduled_tokens=max(0, int(max_scheduled_tokens)),
        max_total_tokens=max_total_tokens,
        max_context_len=max_context_len,
    )
    return int(
        sum(
            int(counts[gid]) * bytes_per_page
            for gid, bytes_per_page in page_bytes.items()
        )
    )


def _resolve_flat_graph_batch_rows(
    value: int | None,
    *,
    fallback: int,
    owner: FlatCacheOwner,
) -> int:
    resolved = fallback if value is None else value
    if isinstance(resolved, bool) or not isinstance(resolved, int) or resolved < 0:
        raise ValueError(
            f"DeepSeek V4 {owner} max_graph_bs must be an integer >= 0, "
            f"got {resolved!r}"
        )
    return resolved


def _build_v4_flat_runtime_metadata(
    *,
    target_specs: Sequence[PagedCacheGroupSpec],
    target_max_context_len: int,
    target_max_live_requests: int,
    target_max_graph_bs: int,
    max_scheduled_tokens: int,
    decode_input_tokens: int,
    overlap_schedule_depth: int,
    draft_specs: Sequence[PagedCacheGroupSpec] = (),
    draft_max_context_len: int | None = None,
    draft_max_live_requests: int | None = None,
    draft_max_graph_bs: int = 0,
) -> FlatRuntimeMetadataPlan:
    """Build the exact non-payload accounting for one owner-union plan."""
    target_by_id = {spec.group_id: spec for spec in target_specs}
    draft_by_id = {spec.group_id: spec for spec in draft_specs}
    if draft_specs and (
        draft_max_context_len is None or draft_max_live_requests is None
    ):
        raise ValueError("DeepSeek V4 flat draft metadata limits are required")
    if target_max_graph_bs > target_max_live_requests:
        raise ValueError(
            "DeepSeek V4 target graph rows exceed live request rows: "
            f"graph={target_max_graph_bs}, live={target_max_live_requests}"
        )
    if draft_specs and draft_max_graph_bs > int(draft_max_live_requests):
        raise ValueError(
            "DeepSeek V4 draft graph rows exceed live request rows: "
            f"graph={draft_max_graph_bs}, live={draft_max_live_requests}"
        )
    if draft_specs and draft_max_graph_bs != target_max_graph_bs:
        raise ValueError(
            "DeepSeek V4 target/draft graph rows must match the shared executor "
            f"graph: target={target_max_graph_bs}, draft={draft_max_graph_bs}"
        )

    max_new_tokens_per_req = max(0, int(max_scheduled_tokens))
    group_table_plans: list[FlatGroupTablePlan] = []
    for group_id in sorted(target_by_id.keys() | draft_by_id.keys()):
        target_spec = target_by_id.get(group_id)
        draft_spec = draft_by_id.get(group_id)
        target_capture_cols = (
            compute_flat_capture_cols(
                target_spec,
                max_context_len=target_max_context_len,
                max_tokens_per_req=decode_input_tokens,
                max_prefill_tokens_per_req=max_new_tokens_per_req,
                overlap_schedule_depth=overlap_schedule_depth,
            )
            if target_spec is not None
            else 0
        )
        draft_capture_cols = (
            compute_flat_capture_cols(
                draft_spec,
                max_context_len=int(draft_max_context_len),
                max_tokens_per_req=decode_input_tokens,
                max_prefill_tokens_per_req=max_new_tokens_per_req,
                overlap_schedule_depth=overlap_schedule_depth,
            )
            if draft_spec is not None
            else 0
        )
        if target_spec is not None and draft_spec is not None:
            shared_capture_cols = max(target_capture_cols, draft_capture_cols)
            target_capture_cols = shared_capture_cols
            draft_capture_cols = shared_capture_cols
        export_cols = []
        if target_spec is not None:
            export_cols.append(
                compute_flat_export_cols(
                    target_spec,
                    max_context_len=target_max_context_len,
                    max_new_tokens_per_req=max_new_tokens_per_req,
                    max_tokens_per_req=decode_input_tokens,
                    overlap_schedule_depth=overlap_schedule_depth,
                )
            )
        if draft_spec is not None:
            export_cols.append(
                compute_flat_export_cols(
                    draft_spec,
                    max_context_len=int(draft_max_context_len),
                    max_new_tokens_per_req=max_new_tokens_per_req,
                    max_tokens_per_req=decode_input_tokens,
                    overlap_schedule_depth=overlap_schedule_depth,
                )
            )
        max_export_cols = max(export_cols)
        group_table_plans.append(
            FlatGroupTablePlan(
                group_id=group_id,
                target_capture_cols=target_capture_cols,
                draft_capture_cols=draft_capture_cols,
                max_export_cols=max_export_cols,
            )
        )

    forward_buffer_depth = overlap_schedule_depth + 1
    max_scheduled_batch_rows = max(
        target_max_live_requests,
        int(draft_max_live_requests or 0),
    )
    return FlatRuntimeMetadataPlan(
        group_table_plans=tuple(group_table_plans),
        forward_buffer_depth=forward_buffer_depth,
        graph_batch_rows=target_max_graph_bs,
        max_scheduled_batch_rows=max_scheduled_batch_rows,
    )


def build_deepseek_v4_flat_memory_plan(
    *,
    target_layout: DeepseekV4CacheLayout,
    target_hf_config: Any,
    target_layer_num: int,
    target_max_live_requests: int,
    target_max_context_len: int,
    max_scheduled_tokens: int,
    max_total_tokens: int,
    target_max_graph_bs: int | None = None,
    draft_layout: DeepseekV4CacheLayout | None = None,
    draft_hf_config: Any | None = None,
    draft_layer_num: int = 0,
    draft_max_live_requests: int | None = None,
    draft_max_context_len: int | None = None,
    draft_max_graph_bs: int | None = None,
    decode_input_tokens: int = 1,
    overlap_schedule_depth: int = 0,
) -> V4FlatMemoryPlan:
    """Build the exact owner-union plan for one V4 token budget.

    Target and draft share scheduler page IDs. Their per-group capacity demand
    is therefore combined with ``max`` while their component-plane bytes are
    summed under separate owner namespaces.
    """
    if max_total_tokens <= 0:
        raise ValueError(
            f"DeepSeek V4 flat max_total_tokens must be positive, got "
            f"{max_total_tokens}"
        )
    page_size = int(target_layout.page_size)
    if page_size <= 0:
        raise ValueError(
            f"DeepSeek V4 flat page_size must be positive, got {page_size}"
        )
    if max_total_tokens % page_size != 0:
        raise ValueError(
            "DeepSeek V4 flat max_total_tokens must be page-aligned: "
            f"tokens={max_total_tokens}, page_size={page_size}"
        )
    if target_layer_num != len(target_layout.layer_ratio):
        raise ValueError(
            "DeepSeek V4 target layer count does not match cache layout: "
            f"layers={target_layer_num}, ratios={len(target_layout.layer_ratio)}"
        )
    resolved_target_max_graph_bs = _resolve_flat_graph_batch_rows(
        target_max_graph_bs,
        fallback=target_max_live_requests,
        owner="target",
    )
    target_specs = tuple(
        build_v4_cache_specs(
            target_hf_config,
            layer_ratio=target_layout.layer_ratio,
            owner_mask=CACHE_OWNER_TARGET,
        )
    )
    target_counts = compute_paged_cache_group_page_counts(
        target_specs,
        max_live_requests=target_max_live_requests,
        max_scheduled_tokens=max(0, int(max_scheduled_tokens)),
        max_total_tokens=max_total_tokens,
        max_context_len=target_max_context_len,
        decode_input_tokens=decode_input_tokens,
        overlap_schedule_depth=overlap_schedule_depth,
    )
    target_components = deepseek_v4_flat_component_plans(
        layout=target_layout,
        specs=target_specs,
        layer_num=target_layer_num,
        owner="target",
    )

    draft_specs: tuple[PagedCacheGroupSpec, ...] = ()
    draft_counts: dict[str, int] | None = None
    draft_components: tuple[FlatComponentTensorPlan, ...] = ()
    resolved_draft_max_graph_bs = 0
    if draft_layout is not None:
        if draft_hf_config is None:
            raise ValueError("DeepSeek V4 flat draft_hf_config is required")
        if draft_layer_num != len(draft_layout.layer_ratio):
            raise ValueError(
                "DeepSeek V4 draft layer count does not match cache layout: "
                f"layers={draft_layer_num}, ratios={len(draft_layout.layer_ratio)}"
            )
        if draft_layout.page_size != target_layout.page_size:
            raise ValueError(
                "DeepSeek V4 target/draft flat layouts must use the same "
                f"page_size, got {target_layout.page_size} and "
                f"{draft_layout.page_size}"
            )
        if draft_max_live_requests is None or draft_max_context_len is None:
            raise ValueError(
                "DeepSeek V4 flat draft request/context limits are required"
            )
        resolved_draft_max_graph_bs = _resolve_flat_graph_batch_rows(
            draft_max_graph_bs,
            fallback=draft_max_live_requests,
            owner="draft",
        )
        draft_specs = tuple(
            build_v4_cache_specs(
                draft_hf_config,
                layer_ratio=draft_layout.layer_ratio,
                owner_mask=CACHE_OWNER_DRAFT,
            )
        )
        draft_counts = compute_paged_cache_group_page_counts(
            draft_specs,
            max_live_requests=draft_max_live_requests,
            max_scheduled_tokens=max(0, int(max_scheduled_tokens)),
            max_total_tokens=max_total_tokens,
            max_context_len=draft_max_context_len,
            decode_input_tokens=decode_input_tokens,
            overlap_schedule_depth=overlap_schedule_depth,
        )
        draft_components = deepseek_v4_flat_component_plans(
            layout=draft_layout,
            specs=draft_specs,
            layer_num=draft_layer_num,
            owner="draft",
        )
    elif (
        any(
            value is not None
            for value in (
                draft_hf_config,
                draft_max_live_requests,
                draft_max_context_len,
                draft_max_graph_bs,
            )
        )
        or draft_layer_num != 0
    ):
        raise ValueError("DeepSeek V4 flat draft arguments require a draft_layout")

    runtime_metadata = _build_v4_flat_runtime_metadata(
        target_specs=target_specs,
        target_max_context_len=target_max_context_len,
        target_max_live_requests=target_max_live_requests,
        target_max_graph_bs=resolved_target_max_graph_bs,
        max_scheduled_tokens=max_scheduled_tokens,
        decode_input_tokens=decode_input_tokens,
        overlap_schedule_depth=overlap_schedule_depth,
        draft_specs=draft_specs,
        draft_max_context_len=draft_max_context_len,
        draft_max_live_requests=draft_max_live_requests,
        draft_max_graph_bs=resolved_draft_max_graph_bs,
    )

    plan = _build_v4_flat_memory_plan(
        max_total_tokens=max_total_tokens,
        target_group_specs=target_specs,
        target_group_page_counts=target_counts,
        target_components=target_components,
        draft_group_specs=draft_specs,
        draft_group_page_counts=draft_counts,
        draft_components=draft_components,
        runtime_metadata=runtime_metadata,
    )
    _validate_v4_flat_plan_components(
        plan,
        specs=target_specs,
        expected_components=target_components,
        owner="target",
    )
    if draft_specs:
        _validate_v4_flat_plan_components(
            plan,
            specs=draft_specs,
            expected_components=draft_components,
            owner="draft",
        )
    return plan


def profile_deepseek_v4_flat_memory_plan(
    *,
    target_layout: DeepseekV4CacheLayout,
    target_hf_config: Any,
    target_layer_num: int,
    target_max_live_requests: int,
    target_max_context_len: int,
    max_scheduled_tokens: int,
    available_cache_memory_bytes: int,
    target_max_graph_bs: int | None = None,
    draft_layout: DeepseekV4CacheLayout | None = None,
    draft_hf_config: Any | None = None,
    draft_layer_num: int = 0,
    draft_max_live_requests: int | None = None,
    draft_max_context_len: int | None = None,
    draft_max_graph_bs: int | None = None,
    max_total_tokens_cap: int | None = None,
    decode_input_tokens: int = 1,
    overlap_schedule_depth: int = 0,
) -> V4FlatMemoryPlan:
    """Return the largest exact V4 flat plan that fits the device budget.

    Profiling evaluates the same immutable plan later consumed by the arena and
    scheduler bridge. The binary search runs only during startup and performs
    no tensor allocation.
    """
    page_size = int(target_layout.page_size)
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if available_cache_memory_bytes <= 0:
        raise ValueError("DeepSeek V4 flat cache memory budget must be positive")
    natural_token_cap = int(target_max_live_requests) * int(target_max_context_len)
    if draft_layout is not None:
        if draft_max_live_requests is None or draft_max_context_len is None:
            raise ValueError(
                "DeepSeek V4 flat draft request/context limits are required"
            )
        natural_token_cap = max(
            natural_token_cap,
            int(draft_max_live_requests) * int(draft_max_context_len),
        )
    if max_total_tokens_cap is not None:
        if max_total_tokens_cap < page_size:
            raise ValueError(
                f"max_total_tokens={max_total_tokens_cap} must contain at least "
                f"one full page (page_size={page_size})"
            )
        natural_token_cap = min(natural_token_cap, int(max_total_tokens_cap))
    max_pages = natural_token_cap // page_size
    if max_pages < 1:
        raise ValueError("DeepSeek V4 flat request/context limits contain no full page")

    common = dict(
        target_layout=target_layout,
        target_hf_config=target_hf_config,
        target_layer_num=target_layer_num,
        target_max_live_requests=target_max_live_requests,
        target_max_context_len=target_max_context_len,
        target_max_graph_bs=target_max_graph_bs,
        max_scheduled_tokens=max_scheduled_tokens,
        draft_layout=draft_layout,
        draft_hf_config=draft_hf_config,
        draft_layer_num=draft_layer_num,
        draft_max_live_requests=draft_max_live_requests,
        draft_max_context_len=draft_max_context_len,
        draft_max_graph_bs=draft_max_graph_bs,
        decode_input_tokens=decode_input_tokens,
        overlap_schedule_depth=overlap_schedule_depth,
    )
    low = 1
    high = max_pages
    best: V4FlatMemoryPlan | None = None
    while low <= high:
        candidate_pages = (low + high) // 2
        candidate = build_deepseek_v4_flat_memory_plan(
            max_total_tokens=candidate_pages * page_size,
            **common,
        )
        if candidate.device_cache_total_bytes <= available_cache_memory_bytes:
            best = candidate
            low = candidate_pages + 1
        else:
            high = candidate_pages - 1
    if best is None:
        one_page_plan = build_deepseek_v4_flat_memory_plan(
            max_total_tokens=page_size,
            **common,
        )
        raise ValueError(
            "DeepSeek V4 flat cache budget cannot fit one plan page: "
            f"required={one_page_plan.device_cache_total_bytes}, "
            f"available={available_cache_memory_bytes}"
        )
    return best


class V4FlatArenaSet:
    """Sole allocator and lifecycle owner of all V4 device component planes."""

    def __init__(
        self,
        plan: V4FlatMemoryPlan,
        *,
        device: str,
        enable_memory_saver: bool,
    ) -> None:
        if plan.max_total_tokens <= 0:
            raise ValueError(
                "V4FlatArenaSet requires a plan with positive max_total_tokens"
            )
        self.plan = plan
        self.device = device
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        self._arena_generation = 0
        self._tensors: dict[tuple[str, str, int, str], torch.Tensor] = {}
        dtype_by_name = {"uint8": torch.uint8, "float32": torch.float32}
        with self.memory_saver_adapter.region(
            tag="kv_cache",
            enable_cpu_backup=False,
        ):
            for pool in plan.pools:
                for component in pool.tensors:
                    try:
                        dtype = dtype_by_name[component.dtype]
                    except KeyError as exc:
                        raise ValueError(
                            "DeepSeek V4 flat arena has unsupported dtype "
                            f"{component.dtype!r}"
                        ) from exc
                    tensor = torch.empty(
                        (pool.total_blocks, *component.shape_per_block),
                        dtype=dtype,
                        device=device,
                    )
                    itemsize = torch._utils._element_size(dtype)
                    actual_strides = tuple(
                        int(stride) * itemsize for stride in tensor.stride()[1:]
                    )
                    if actual_strides != component.stride_bytes:
                        raise RuntimeError(
                            "DeepSeek V4 flat tensor stride mismatch for "
                            f"{component.owner}/{component.group_id}/"
                            f"{component.layer}/{component.component}: "
                            f"expected={component.stride_bytes}, "
                            f"actual={actual_strides}"
                        )
                    if int(tensor[0].nbytes) != component.bytes_per_block:
                        raise RuntimeError(
                            "DeepSeek V4 flat tensor bytes-per-block mismatch for "
                            f"{component.owner}/{component.group_id}/"
                            f"{component.layer}/{component.component}"
                        )
                    if tensor.data_ptr() % component.alignment_bytes != 0:
                        raise RuntimeError(
                            "DeepSeek V4 flat tensor base alignment mismatch for "
                            f"{component.owner}/{component.group_id}/"
                            f"{component.layer}/{component.component}"
                        )
                    identity = (
                        component.owner,
                        component.group_id,
                        component.layer,
                        component.component,
                    )
                    self._tensors[identity] = tensor

        actual_bytes = sum(int(tensor.nbytes) for tensor in self._tensors.values())
        if actual_bytes != plan.payload_bytes:
            raise RuntimeError(
                "DeepSeek V4 flat arena allocation size disagrees with plan: "
                f"planned={plan.payload_bytes}, actual={actual_bytes}"
            )
        self._reset_null_pages()

    @property
    def plan_fingerprint(self) -> str:
        return self.plan.plan_fingerprint

    @property
    def arena_generation(self) -> int:
        """Current owner-level generation shared by target and draft views."""
        return self._arena_generation

    def tensor(
        self,
        owner: FlatCacheOwner,
        group_id: str,
        layer: int,
        component: str,
    ) -> torch.Tensor:
        """Return one non-owning tensor view by canonical component identity."""
        identity = (owner, group_id, layer, component)
        try:
            return self._tensors[identity]
        except KeyError as exc:
            raise KeyError(
                f"DeepSeek V4 flat arena has no tensor {identity!r}"
            ) from exc

    def get_kv_size_bytes(self) -> int:
        return self.plan.payload_bytes

    def _reset_null_pages(self) -> None:
        for tensor in self._tensors.values():
            tensor[0].zero_()

    def repair_after_wake(self, *, expected_generation: int | None = None) -> int:
        """Repair all null pages and atomically advance the arena generation.

        The event loop must invoke this once per shared arena after its device
        mappings are restored. Target and draft pools are non-owning views and
        must not repair their component planes independently.

        Args:
            expected_generation: Generation returned by the scheduler's
                quiescent pool-set reset. When supplied, it must be exactly the
                next arena generation; a mismatch fails before touching page 0.

        Returns:
            The new owner-level arena generation.
        """
        next_generation = self._arena_generation + 1
        if expected_generation is not None and (
            isinstance(expected_generation, bool)
            or not isinstance(expected_generation, int)
            or expected_generation != next_generation
        ):
            raise RuntimeError(
                "DeepSeek V4 flat arena/scheduler generation mismatch: "
                f"arena_next={next_generation}, scheduler={expected_generation!r}"
            )
        self._reset_null_pages()
        self._arena_generation = next_generation
        return self._arena_generation


def profile_deepseek_v4_max_num_pages(
    *,
    layout: DeepseekV4CacheLayout,
    hf_config: Any,
    layer_num: int,
    max_live_requests: int,
    max_scheduled_tokens: int,
    max_context_len: int,
    available_cache_memory_bytes: int,
    draft_cache_cell_size: int = 0,
    decode_input_tokens: int = 1,
    overlap_schedule_depth: int = 0,
) -> int:
    """Return the largest scheduler page budget that fits V4 grouped caches."""
    page_size = int(layout.page_size)
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if available_cache_memory_bytes <= 0:
        return 0
    if draft_cache_cell_size < 0:
        raise ValueError(
            f"draft_cache_cell_size must be >= 0, got {draft_cache_cell_size}"
        )

    draft_cache_cell_size = int(draft_cache_cell_size)
    max_live_requests = int(max_live_requests)
    max_scheduled_tokens = max(0, int(max_scheduled_tokens))
    max_context_len = int(max_context_len)
    specs = tuple(build_v4_cache_specs(hf_config, layer_ratio=layout.layer_ratio))
    page_bytes = _deepseek_v4_cache_group_page_bytes(layout, specs, layer_num)

    def _bytes_for_pages(num_pages: int) -> int:
        num_tokens = int(num_pages) * page_size
        counts = compute_paged_cache_group_page_counts(
            specs,
            max_live_requests=max_live_requests,
            max_scheduled_tokens=max_scheduled_tokens,
            max_total_tokens=num_tokens,
            max_context_len=max_context_len,
            decode_input_tokens=decode_input_tokens,
            overlap_schedule_depth=overlap_schedule_depth,
        )
        cache_bytes = sum(
            int(counts[gid]) * bytes_per_page
            for gid, bytes_per_page in page_bytes.items()
        )
        return int(cache_bytes + num_tokens * draft_cache_cell_size)

    if _bytes_for_pages(1) > available_cache_memory_bytes:
        return 0

    if not any(int(ratio) > 1 for ratio in layout.layer_ratio[:layer_num]):
        return max(
            1,
            (int(max_live_requests) * int(max_context_len) + page_size - 1)
            // page_size,
        )

    # Fixed bytes cover resident sliding windows, request fragments, and dummy
    # pages. Variable bytes are piecewise linear before and after the global
    # scheduled-token write budget is capped.
    fixed_counts = compute_paged_cache_group_page_counts(
        specs,
        max_live_requests=max_live_requests,
        max_scheduled_tokens=max_scheduled_tokens,
        max_total_tokens=0,
        max_context_len=max_context_len,
        decode_input_tokens=decode_input_tokens,
        overlap_schedule_depth=overlap_schedule_depth,
    )
    fixed_bytes = sum(
        int(fixed_counts[gid]) * bytes_per_page
        for gid, bytes_per_page in page_bytes.items()
    )
    full_history_slope = Fraction(page_size * draft_cache_cell_size, 1)
    scheduled_slope = Fraction(0, 1)
    scheduled_cap_bytes = 0
    for spec in specs:
        bytes_per_page = page_bytes[spec.group_id]
        if bytes_per_page == 0:
            continue
        raw_per_page = int(spec.block_size)
        if spec.retention == "full_history":
            full_history_slope += Fraction(page_size * bytes_per_page, raw_per_page)
        elif spec.retention == "sliding_window":
            scheduled_slope += Fraction(page_size * bytes_per_page, raw_per_page)
            scheduled_cap_bytes += (
                ceil_div(max_scheduled_tokens, raw_per_page) * bytes_per_page
            )

    def _pages_from_budget(extra_bytes: int, slope: Fraction) -> int:
        if extra_bytes <= 0 or slope <= 0:
            return 0
        return int(extra_bytes * slope.denominator // slope.numerator)

    cap_pages = ceil_div(max_scheduled_tokens, page_size)
    candidate = 0
    pre_cap_slope = full_history_slope + scheduled_slope
    if cap_pages > 0:
        pre_cap_pages = _pages_from_budget(
            available_cache_memory_bytes - fixed_bytes,
            pre_cap_slope,
        )
        candidate = min(pre_cap_pages, cap_pages - 1)

    post_cap_fixed_bytes = fixed_bytes + scheduled_cap_bytes
    post_cap_pages = _pages_from_budget(
        available_cache_memory_bytes - post_cap_fixed_bytes,
        full_history_slope,
    )
    if post_cap_pages >= cap_pages:
        candidate = max(candidate, post_cap_pages)
    candidate = max(1, candidate)

    while candidate > 0 and _bytes_for_pages(candidate) > available_cache_memory_bytes:
        candidate -= 1
    while _bytes_for_pages(candidate + 1) <= available_cache_memory_bytes:
        candidate += 1
    return int(candidate)


def _split_paged_cache_block_tables_into_v4_metadata(
    paged_cache_block_tables: dict[str, torch.Tensor],
    paged_cache_block_table_base_offsets: dict[str, torch.Tensor] | None = None,
) -> tuple[
    torch.Tensor | None,
    dict[int, torch.Tensor],
    torch.Tensor | None,
    torch.Tensor | None,
    dict[int, torch.Tensor],
    torch.Tensor | None,
]:
    """Split paged-cache dict into V4-named tables + per-sliding-group offsets.

    Returns (swa, {ratio: compressor_state}, indexer_state, swa_base,
    {ratio: compressor_state_base}, indexer_state_base). Unknown group ids
    are ignored. Base offsets are None / missing when the input lacks them.
    """
    offsets = paged_cache_block_table_base_offsets or {}
    swa = paged_cache_block_tables.get(V4_SWA_KV_GROUP_ID)
    indexer_state = paged_cache_block_tables.get(V4_INDEXER_COMPRESSOR_STATE_GROUP_ID)
    swa_base = offsets.get(V4_SWA_KV_GROUP_ID)
    indexer_state_base = offsets.get(V4_INDEXER_COMPRESSOR_STATE_GROUP_ID)
    compressor_state: dict[int, torch.Tensor] = {}
    compressor_state_base: dict[int, torch.Tensor] = {}
    for gid, table in paged_cache_block_tables.items():
        ratio = parse_v4_compressor_state_group_id(gid)
        if ratio is None:
            continue
        compressor_state[ratio] = table
        base = offsets.get(gid)
        if base is not None:
            compressor_state_base[ratio] = base
    return (
        swa,
        compressor_state,
        indexer_state,
        swa_base,
        compressor_state_base,
        indexer_state_base,
    )


def _safe_page_ids(
    block_table: torch.Tensor,
    req_indices: torch.Tensor,
    page_indices: torch.Tensor,
) -> torch.Tensor:
    req_i64 = req_indices.to(torch.int64)
    page_i64 = page_indices.to(torch.int64)
    sentinel = torch.full_like(page_i64, -1, dtype=torch.int64)
    rows = int(block_table.shape[0]) if block_table.ndim >= 1 else 0
    cols = int(block_table.shape[1]) if block_table.ndim >= 2 else 0
    if rows <= 0 or cols <= 0:
        return sentinel
    valid = (req_i64 >= 0) & (req_i64 < rows) & (page_i64 >= 0) & (page_i64 < cols)
    safe_req = req_i64.clamp(0, rows - 1)
    safe_page = page_i64.clamp(0, cols - 1)
    page_ids = block_table[safe_req, safe_page].to(torch.int64)
    return torch.where(valid, page_ids, sentinel)


def _expand_group_values_for_tokens(
    values: torch.Tensor,
    num_tokens: int,
    name: str,
) -> torch.Tensor:
    if values.numel() == num_tokens:
        return values
    if values.numel() <= 0 or num_tokens % values.numel() != 0:
        raise RuntimeError(
            f"DeepSeek V4 {name} has incompatible shape for packed tokens: "
            f"{values.numel()} entries for {num_tokens} tokens"
        )
    return values.repeat_interleave(num_tokens // values.numel())


def _group_slot_mapping_from_raw(
    positions: torch.Tensor,
    req_indices: torch.Tensor,
    block_table: torch.Tensor,
    rows_per_page: int,
    entry_stride_tokens: int = 1,
    base_offsets: torch.Tensor | None = None,
    capacity_pages: int | None = None,
) -> torch.Tensor:
    if rows_per_page <= 0:
        raise ValueError(f"rows_per_page must be > 0, got {rows_per_page}")
    if entry_stride_tokens <= 0:
        raise ValueError(f"entry_stride_tokens must be > 0, got {entry_stride_tokens}")
    if capacity_pages is not None and capacity_pages < 0:
        raise ValueError(f"capacity_pages must be >= 0, got {capacity_pages}")
    pos_i64 = positions.to(torch.int64)
    logical_row = torch.div(pos_i64, entry_stride_tokens, rounding_mode="floor")
    logical_page = torch.div(logical_row, rows_per_page, rounding_mode="floor")
    offsets = logical_row % rows_per_page
    req_indices = _expand_group_values_for_tokens(
        req_indices,
        positions.numel(),
        "request indices",
    )
    table_page = logical_page
    if base_offsets is not None:
        req_i64 = req_indices.to(torch.int64)
        rows = int(base_offsets.shape[0])
        if rows <= 0:
            table_page = logical_page.new_full(logical_page.shape, -1)
        else:
            valid_req = (req_i64 >= 0) & (req_i64 < rows)
            safe_req = req_i64.clamp(0, rows - 1)
            base = base_offsets.to(
                device=logical_page.device,
                dtype=torch.int64,
            )[safe_req]
            table_page = torch.where(valid_req, logical_page - base, -1)
    page_ids = _safe_page_ids(block_table, req_indices, table_page)
    slots = page_ids * rows_per_page + offsets
    # Page 0 is the zero-initialized null page. It may be read by padded rows,
    # but no V4 producer is ever allowed to write through it. The upper bound is
    # the actual owner/component allocation, not a scheduler-global page count.
    # Flat CPU staging already rejects out-of-pool IDs before H2D; this mask is
    # the device-side defense for stale graph buffers and direct/legacy metadata,
    # and keeps every writer kernel safe without a hot-path host sync.
    valid_pages = page_ids > 0
    if capacity_pages is not None:
        valid_pages &= page_ids < capacity_pages
    return torch.where(valid_pages, slots, torch.full_like(slots, -1))


def _mask_invalid_graph_tokens(
    slot_mapping: torch.Tensor,
    is_valid_token: torch.Tensor | None,
) -> torch.Tensor:
    if is_valid_token is None:
        return slot_mapping
    valid = _expand_group_values_for_tokens(
        is_valid_token,
        slot_mapping.numel(),
        "slot validity mask",
    ).to(
        device=slot_mapping.device,
        dtype=torch.bool,
    )
    return torch.where(valid, slot_mapping, torch.full_like(slot_mapping, -1))


def _compressed_boundary_mask(
    positions: torch.Tensor,
    compress_ratio: int,
) -> torch.Tensor:
    if compress_ratio <= 1:
        return torch.ones_like(positions, dtype=torch.bool)
    return ((positions.to(torch.int64) + 1) % compress_ratio) == 0


@dataclass
class DeepseekV4CacheMetadata:
    page_size: int
    block_table: torch.Tensor
    paged_cache_block_tables: dict[str, torch.Tensor] = field(default_factory=dict)
    # Per-sliding-group [num_reqs] int32 base logical-page offset that
    # accompanies each compact block table. Consumers index sliding tables as
    # logical_page - base_offset; full-history groups omit the key (base 0).
    paged_cache_block_table_base_offsets: dict[str, torch.Tensor] = field(
        default_factory=dict
    )
    swa_block_table: torch.Tensor | None = None
    swa_base_logical_page: torch.Tensor | None = None
    compressor_state_block_tables: dict[int, torch.Tensor] = field(default_factory=dict)
    compressor_state_base_logical_pages: dict[int, torch.Tensor] = field(
        default_factory=dict
    )
    indexer_state_block_table: torch.Tensor | None = None
    indexer_state_base_logical_page: torch.Tensor | None = None
    decode_compressed_slot_mappings: dict[tuple[int, int], torch.Tensor] = field(
        default_factory=dict
    )
    decode_compressed_capacity_pages: dict[tuple[int, int], int] = field(
        default_factory=dict
    )

    @property
    def has_legacy_block_table(self) -> bool:
        """Whether the radix-compatible single-table fallback is available."""
        return self.block_table.ndim == 2 and self.block_table.shape[1] > 0

    def compressed_block_table(
        self,
        compress_ratio: int,
        kv_cache_block_size: int | None = None,
    ) -> torch.Tensor:
        del kv_cache_block_size
        if compress_ratio <= 1:
            if not self.has_legacy_block_table:
                raise RuntimeError(
                    "DeepSeek V4 flat cache metadata cannot use the legacy "
                    "single block_table fallback for an uncompressed group"
                )
            return self.block_table
        table = self.paged_cache_block_tables.get(
            v4_compressed_kv_group_id(compress_ratio)
        )
        if table is None:
            raise RuntimeError(
                "DeepSeek V4 missing paged-cache block table for compressed "
                f"KV group {v4_compressed_kv_group_id(compress_ratio)!r}"
            )
        return table

    @staticmethod
    def safe_page_ids(
        block_table: torch.Tensor,
        req_indices: torch.Tensor,
        page_indices: torch.Tensor,
    ) -> torch.Tensor:
        return _safe_page_ids(block_table, req_indices, page_indices)

    def _update_decode_compressed_slot_mapping(
        self,
        *,
        token_to_req_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        compress_ratio: int,
        kv_cache_block_size: int,
        capacity_pages: int | None,
        is_valid_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = token_to_req_indices.shape[0]
        key = (compress_ratio, kv_cache_block_size)
        out = self.decode_compressed_slot_mappings.get(key)
        if out is None or out.shape[0] < num_tokens or out.device != seq_lens.device:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "DeepSeek V4 compressed slot metadata must be allocated before "
                    "CUDA graph capture"
                )
            with torch.inference_mode(False):
                out = torch.empty(num_tokens, dtype=torch.int64, device=seq_lens.device)
            self.decode_compressed_slot_mappings[key] = out

        block_table = self.compressed_block_table(compress_ratio, kv_cache_block_size)
        if block_table is not self.block_table:
            req_idx = token_to_req_indices[:num_tokens].to(torch.int64)
            query_starts = query_start_loc[req_idx].to(torch.int64)
            query_lens = query_start_loc[req_idx + 1].to(torch.int64) - query_starts
            seq_lens_for_token = seq_lens[req_idx].to(torch.int64)
            token_offsets = torch.arange(
                num_tokens,
                dtype=torch.int64,
                device=seq_lens.device,
            )
            positions = seq_lens_for_token - query_lens + token_offsets - query_starts
            compressed_pos = torch.div(
                positions,
                compress_ratio,
                rounding_mode="floor",
            )
            page_indices = torch.div(
                compressed_pos,
                kv_cache_block_size,
                rounding_mode="floor",
            )
            offsets = compressed_pos % kv_cache_block_size
            base_offsets = self.paged_cache_block_table_base_offsets.get(
                v4_compressed_kv_group_id(compress_ratio)
            )
            if base_offsets is not None:
                page_indices = (
                    page_indices
                    - base_offsets.to(
                        device=page_indices.device,
                        dtype=torch.int64,
                    )[req_idx]
                )
            page_ids = _safe_page_ids(block_table, req_idx, page_indices)
            valid_pages = page_ids > 0
            if capacity_pages is not None:
                valid_pages &= page_ids < capacity_pages
            valid_slots = valid_pages & _compressed_boundary_mask(
                positions,
                compress_ratio,
            )
            slot_mapping = torch.where(
                valid_slots,
                page_ids * kv_cache_block_size + offsets,
                torch.full_like(page_ids, -1),
            )
            out.copy_(_mask_invalid_graph_tokens(slot_mapping, is_valid_token))
            return out

        mapping = deepseek_v4_compressed_slot_mapping(
            num_tokens=num_tokens,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_table=self.block_table,
            block_size=kv_cache_block_size,
            compress_ratio=compress_ratio,
            out=out,
        )
        if capacity_pages is not None:
            capacity_slots = capacity_pages * kv_cache_block_size
            valid_slots = (mapping >= kv_cache_block_size) & (mapping < capacity_slots)
            mapping.copy_(
                torch.where(valid_slots, mapping, torch.full_like(mapping, -1))
            )
        if is_valid_token is not None:
            mapping.copy_(_mask_invalid_graph_tokens(mapping, is_valid_token))
        return mapping

    def refresh_decode_compressed_slot_mappings(
        self,
        *,
        token_to_req_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        is_valid_token: torch.Tensor | None = None,
    ) -> None:
        for compress_ratio, kv_cache_block_size in list(
            self.decode_compressed_slot_mappings
        ):
            key = (compress_ratio, kv_cache_block_size)
            self._update_decode_compressed_slot_mapping(
                token_to_req_indices=token_to_req_indices,
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                compress_ratio=compress_ratio,
                kv_cache_block_size=kv_cache_block_size,
                capacity_pages=self.decode_compressed_capacity_pages.get(key),
                is_valid_token=is_valid_token,
            )

    def compressed_slot_mapping(
        self,
        positions: torch.Tensor,
        compress_ratio: int,
        *,
        token_to_req_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_cache_block_size: int | None = None,
        capacity_pages: int | None = None,
        use_decode_cache: bool = False,
        is_valid_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if kv_cache_block_size is None:
            kv_cache_block_size = self.page_size
        if capacity_pages is not None and capacity_pages < 0:
            raise ValueError(f"capacity_pages must be >= 0, got {capacity_pages}")
        if not self.has_legacy_block_table and capacity_pages is None:
            raise RuntimeError(
                "DeepSeek V4 flat compressed slot mapping requires the owner "
                "component page capacity"
            )
        key = (compress_ratio, kv_cache_block_size)
        ratio_capacities = {
            existing_capacity
            for (existing_ratio, _), existing_capacity in (
                self.decode_compressed_capacity_pages.items()
            )
            if existing_ratio == compress_ratio
        }
        if capacity_pages is not None and any(
            existing_capacity != capacity_pages
            for existing_capacity in ratio_capacities
        ):
            raise RuntimeError(
                "DeepSeek V4 co-indexed compressed components disagree on page "
                f"capacity for ratio={compress_ratio}: "
                f"first={min(ratio_capacities)}, current={capacity_pages}"
            )
        if capacity_pages is not None:
            self.decode_compressed_capacity_pages[key] = capacity_pages
        block_table = self.compressed_block_table(compress_ratio, kv_cache_block_size)
        if (
            use_decode_cache
            and positions.is_cuda
            and (block_table.is_cuda or self.block_table.is_cuda)
        ):
            cached = self.decode_compressed_slot_mappings.get(key)
            if (
                cached is not None
                and cached.shape[0] >= positions.numel()
                and cached.device == seq_lens.device
            ):
                return cached[: positions.numel()]
            mapping = self._update_decode_compressed_slot_mapping(
                token_to_req_indices=token_to_req_indices,
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                compress_ratio=compress_ratio,
                kv_cache_block_size=kv_cache_block_size,
                capacity_pages=capacity_pages,
                is_valid_token=is_valid_token,
            )
            return mapping[: positions.numel()]
        compressed_pos = torch.div(
            positions.to(torch.int64), compress_ratio, rounding_mode="floor"
        )
        page_indices = torch.div(
            compressed_pos, kv_cache_block_size, rounding_mode="floor"
        )
        offsets = compressed_pos % kv_cache_block_size
        req_idx = token_to_req_indices[: positions.numel()].long()
        if block_table is self.block_table:
            page_ids = block_table[req_idx, page_indices.long()].to(torch.int64)
        else:
            base_offsets = self.paged_cache_block_table_base_offsets.get(
                v4_compressed_kv_group_id(compress_ratio)
            )
            if base_offsets is not None:
                page_indices = (
                    page_indices
                    - base_offsets.to(
                        device=page_indices.device,
                        dtype=torch.int64,
                    )[req_idx]
                )
            page_ids = _safe_page_ids(block_table, req_idx, page_indices.long())
        slots = page_ids.to(torch.int64) * kv_cache_block_size + offsets
        valid_pages = page_ids > 0
        if capacity_pages is not None:
            valid_pages &= page_ids < capacity_pages
        valid_slots = valid_pages & _compressed_boundary_mask(
            positions,
            compress_ratio,
        )
        slot_mapping = torch.where(
            valid_slots,
            slots,
            torch.full_like(slots, -1),
        )
        return _mask_invalid_graph_tokens(slot_mapping, is_valid_token)


def deepseek_v4_cache_layout_from_config(
    hf_config,
    page_size: int,
    use_fp4_indexer_cache: bool,
    layer_indices: Iterable[int] | None = None,
) -> DeepseekV4CacheLayout:
    compress_ratios = tuple(hf_config.compress_ratios)
    if layer_indices is None:
        layer_ratios = compress_ratios
    else:
        layer_indices = tuple(layer_indices)
        if any(idx < 0 or idx >= len(compress_ratios) for idx in layer_indices):
            raise ValueError(
                "DeepSeek V4 cache layout layer index out of range: "
                f"indices={layer_indices}, ratios={len(compress_ratios)}"
            )
        layer_ratios = [compress_ratios[idx] for idx in layer_indices]
    raw_layer_ratios = tuple(int(x) for x in layer_ratios)
    for ratio in raw_layer_ratios:
        if ratio not in (0, 1, 4, 128):
            raise ValueError(
                "Unsupported DeepSeek V4 cache compress_ratio="
                f"{ratio}; expected one of 0, 1, 4, or 128"
            )

    return DeepseekV4CacheLayout(
        layer_ratio=tuple(max(1, ratio) for ratio in raw_layer_ratios),
        head_dim=int(hf_config.head_dim),
        rope_head_dim=int(hf_config.qk_rope_head_dim),
        page_size=page_size,
        use_fp4_indexer_cache=use_fp4_indexer_cache,
        index_head_dim=int(getattr(hf_config, "index_head_dim", 128)),
    )


class DeepseekV4TokenToKVPool(BaseTokenToKVPool):
    """DeepSeek V4 fp8_ds_mla cache pool.

    TokenSpeed keeps SWA, compressed, compressor-state, and CSA indexer caches
    in dedicated per-group paged pools (see PagedCacheGroup* on the scheduler
    side and ``build_v4_cache_specs`` here), keeping ordinary MLA models on
    their existing single-pool contract. The ``indexer_kv_buffer`` shares its
    page table and page-count budget with the ``v4.c{ratio}a.compressed_kv``
    group rather than owning a separate group of its own.
    """

    supports_hierarchical_kv_cache = False
    supports_pd_transfer = False
    device_cache_buffer_names = (
        "swa_kv_buffer",
        "compressed_kv_buffer",
        "compressor_state_buffer",
        "indexer_kv_buffer",
        "indexer_state_buffer",
    )

    def __init__(
        self,
        size: int,
        model_dtype: torch.dtype,
        layout: DeepseekV4CacheLayout,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        max_batch_size: int,
        max_context_len: int,
        page_size: int,
        rank: int,
        hf_config: Any,
        max_scheduled_tokens: int,
        decode_input_tokens: int = 1,
        overlap_schedule_depth: int = 0,
        cache_owner: FlatCacheOwner = "target",
        flat_arena_set: V4FlatArenaSet | None = None,
    ) -> None:
        if size <= 0:
            raise ValueError(f"DeepSeek V4 KV pool size must be positive, got {size}")
        if layer_num != len(layout.layer_ratio):
            raise ValueError(
                "DeepSeek V4 KV pool layer_num must match cache layout ratios: "
                f"layer_num={layer_num}, ratios={len(layout.layer_ratio)}"
            )
        if cache_owner not in ("target", "draft"):
            raise ValueError(
                f"DeepSeek V4 cache_owner must be target or draft, got {cache_owner!r}"
            )
        super().__init__(
            size=size,
            dtype=torch.uint8,
            device=device,
            max_batch_size=max_batch_size,
            max_context_len=max_context_len,
            page_size=page_size,
            rank=rank,
        )
        self.cache_owner = cache_owner
        self.flat_arena_set = flat_arena_set
        self.flat_memory_plan = (
            flat_arena_set.plan if flat_arena_set is not None else None
        )
        # The arena creates the one allocation region for flat target+draft.
        # Radix keeps the existing per-pool region and allocation behavior.
        self.memory_saver_adapter = (
            flat_arena_set.memory_saver_adapter
            if flat_arena_set is not None
            else TorchMemorySaverAdapter.create(enable=enable_memory_saver)
        )
        self.model_dtype = model_dtype
        self.layout = layout
        self.layer_num = layer_num
        self.max_batch_size = max_batch_size
        self.max_context_len = max_context_len
        owner_mask = (
            CACHE_OWNER_TARGET if cache_owner == "target" else CACHE_OWNER_DRAFT
        )
        expected_owner_specs = tuple(
            build_v4_cache_specs(
                hf_config,
                layer_ratio=layout.layer_ratio,
                owner_mask=owner_mask,
            )
        )
        if flat_arena_set is None:
            owner_specs = expected_owner_specs
            owner_counts = compute_paged_cache_group_page_counts(
                owner_specs,
                max_live_requests=max_batch_size,
                max_scheduled_tokens=max(0, int(max_scheduled_tokens)),
                max_total_tokens=size,
                max_context_len=max_context_len,
                decode_input_tokens=decode_input_tokens,
                overlap_schedule_depth=overlap_schedule_depth,
            )
            self.scheduler_group_specs = owner_specs
            self.scheduler_group_page_counts = dict(owner_counts)
            self.num_pages = (size + page_size - 1) // page_size + 1
        else:
            plan = flat_arena_set.plan
            if size != plan.max_total_tokens:
                raise ValueError(
                    "DeepSeek V4 flat pool size must match shared plan: "
                    f"size={size}, plan={plan.max_total_tokens}"
                )
            if str(flat_arena_set.device) != str(device):
                raise ValueError(
                    "DeepSeek V4 flat pool device must match arena: "
                    f"pool={device}, arena={flat_arena_set.device}"
                )
            owner_specs = (
                plan.target_owner_group_specs
                if cache_owner == "target"
                else plan.draft_owner_group_specs
            )
            expected_by_id = {spec.group_id: spec for spec in expected_owner_specs}
            actual_by_id = {
                spec.group_id: replace(spec, owner_mask=owner_mask)
                for spec in owner_specs
            }
            if actual_by_id != expected_by_id:
                raise ValueError(
                    "DeepSeek V4 flat owner specs disagree with layout/config: "
                    f"owner={cache_owner}, expected={sorted(expected_by_id)}, "
                    f"actual={sorted(actual_by_id)}"
                )
            pool_by_id = {pool.pool_id: pool for pool in plan.pools}
            self.scheduler_group_specs = plan.scheduler_group_specs
            self.scheduler_group_page_counts = {
                spec.group_id: pool_by_id[spec.pool_id].total_blocks
                for spec in plan.scheduler_group_specs
            }
            owner_counts = {
                spec.group_id: pool_by_id[spec.pool_id].total_blocks
                for spec in owner_specs
            }
            self.num_pages = max(pool.total_blocks for pool in plan.pools)

        # ``paged_cache_group_specs`` remains the owner-local consumer view.
        # Scheduler conversion explicitly consumes scheduler_group_specs.
        self.owner_group_specs = tuple(owner_specs)
        self.owner_group_page_counts = dict(owner_counts)
        self.paged_cache_group_specs = self.owner_group_specs
        self.paged_cache_group_page_counts = self.owner_group_page_counts
        self._paged_cache_group_specs_by_id = {
            spec.group_id: spec for spec in self.paged_cache_group_specs
        }
        self._paged_cache_scheduler: object | None = None
        self._paged_cache_state_group_ids = tuple(
            str(spec.group_id)
            for spec in self.paged_cache_group_specs
            if spec.family == "state"
        )

        def _group_rows(group_id: str) -> int:
            spec = self._paged_cache_group_specs_by_id.get(group_id)
            if spec is None:
                raise ValueError(
                    f"DeepSeek V4 cache layout is missing group spec {group_id!r}"
                )
            return int(spec.rows_per_page)

        self.swa_block_size = _group_rows(V4_SWA_KV_GROUP_ID)
        self._legacy_state_block_size = page_size
        self.swa_block_bytes = layout.swa_block_bytes(self.swa_block_size)
        if flat_arena_set is not None:
            self.compressed_block_sizes = tuple(
                (
                    _group_rows(v4_compressed_kv_group_id(ratio))
                    if ratio > 1
                    else page_size
                )
                for ratio in layout.layer_ratio
            )
            self.indexer_block_sizes = tuple(
                (_group_rows(v4_compressed_kv_group_id(ratio)) if ratio == 4 else 0)
                for ratio in layout.layer_ratio
            )
        else:
            # The legacy/radix allocation keeps its existing layout-derived
            # kernel geometry. Flat allocation is spec-derived above.
            self.compressed_block_sizes = tuple(
                layout.storage_block_size(ratio) if ratio > 1 else page_size
                for ratio in layout.layer_ratio
            )
            self.indexer_block_sizes = tuple(
                (
                    max(V4_KERNEL_BLOCK_ROWS, self.compressed_block_sizes[layer_id])
                    if ratio == 4
                    else 0
                )
                for layer_id, ratio in enumerate(layout.layer_ratio)
            )
        self.compressor_state_block_sizes = tuple(
            (
                _group_rows(v4_compressor_state_group_id(ratio))
                if ratio > 1
                else page_size
            )
            for ratio in layout.layer_ratio
        )
        self.indexer_state_block_sizes = tuple(
            (_group_rows(V4_INDEXER_COMPRESSOR_STATE_GROUP_ID) if ratio == 4 else 0)
            for ratio in layout.layer_ratio
        )
        if flat_arena_set is not None:
            self.swa_kv_buffer = tuple(
                flat_arena_set.tensor(
                    cache_owner,
                    V4_SWA_KV_GROUP_ID,
                    layer_id,
                    _V4_SWA_COMPONENT,
                )
                for layer_id in range(layer_num)
            )
            compressed_buffers: list[torch.Tensor | None] = []
            compressor_state_buffers: list[torch.Tensor | None] = []
            indexer_buffers: list[torch.Tensor | None] = []
            indexer_state_buffers: list[torch.Tensor | None] = []
            for layer_id, ratio in enumerate(layout.layer_ratio):
                if ratio > 1:
                    compressed_group_id = v4_compressed_kv_group_id(ratio)
                    compressed_buffers.append(
                        flat_arena_set.tensor(
                            cache_owner,
                            compressed_group_id,
                            layer_id,
                            _V4_COMPRESSED_COMPONENT,
                        )
                    )
                    compressor_state_buffers.append(
                        flat_arena_set.tensor(
                            cache_owner,
                            v4_compressor_state_group_id(ratio),
                            layer_id,
                            _V4_COMPRESSOR_STATE_COMPONENT,
                        )
                    )
                else:
                    compressed_buffers.append(None)
                    compressor_state_buffers.append(None)
                if ratio == 4:
                    indexer_buffers.append(
                        flat_arena_set.tensor(
                            cache_owner,
                            v4_compressed_kv_group_id(ratio),
                            layer_id,
                            _V4_INDEXER_COMPONENT,
                        )
                    )
                    indexer_state_buffers.append(
                        flat_arena_set.tensor(
                            cache_owner,
                            V4_INDEXER_COMPRESSOR_STATE_GROUP_ID,
                            layer_id,
                            _V4_INDEXER_STATE_COMPONENT,
                        )
                    )
                else:
                    indexer_buffers.append(None)
                    indexer_state_buffers.append(None)
            self.compressed_kv_buffer = tuple(compressed_buffers)
            self.compressor_state_buffer = tuple(compressor_state_buffers)
            self.indexer_kv_buffer = tuple(indexer_buffers)
            self.indexer_state_buffer = tuple(indexer_state_buffers)
        else:
            swa_pages = self.paged_cache_group_page_counts.get(
                V4_SWA_KV_GROUP_ID,
                self.num_pages,
            )
            with self.memory_saver_adapter.region(
                tag="kv_cache",
                enable_cpu_backup=False,
            ):
                self.swa_kv_buffer = [
                    torch.zeros(
                        (swa_pages, self.swa_block_bytes),
                        dtype=torch.uint8,
                        device=device,
                    )
                    for _ in range(layer_num)
                ]
                self.compressed_kv_buffer: list[torch.Tensor | None] = []
                self.compressor_state_buffer: list[torch.Tensor | None] = []
                self.indexer_kv_buffer: list[torch.Tensor | None] = []
                self.indexer_state_buffer: list[torch.Tensor | None] = []
                for layer_id, ratio in enumerate(layout.layer_ratio):
                    has_compressed = ratio > 1
                    has_indexer = ratio == 4
                    compressed_block_size = self.compressed_block_sizes[layer_id]
                    compressed_group_id = v4_compressed_kv_group_id(ratio)
                    compressed_pages = self.num_pages
                    if has_compressed:
                        compressed_pages = self.paged_cache_group_page_counts.get(
                            compressed_group_id,
                            self.num_pages,
                        )
                    self.compressed_kv_buffer.append(
                        torch.zeros(
                            (
                                compressed_pages,
                                layout.swa_block_bytes(compressed_block_size),
                            ),
                            dtype=torch.uint8,
                            device=device,
                        )
                        if has_compressed
                        else None
                    )
                    compressor_state_block_size = self.compressor_state_block_sizes[
                        layer_id
                    ]
                    compressor_state_group_id = v4_compressor_state_group_id(ratio)
                    compressor_state_pages = self.num_pages
                    if has_compressed:
                        compressor_state_pages = self.paged_cache_group_page_counts.get(
                            compressor_state_group_id,
                            self.num_pages,
                        )
                    self.compressor_state_buffer.append(
                        torch.empty(
                            (
                                compressor_state_pages,
                                compressor_state_block_size,
                                layout.state_width(layer_id) * 2,
                            ),
                            dtype=torch.float32,
                            device=device,
                        )
                        if has_compressed
                        else None
                    )
                    indexer_block_size = self.indexer_block_sizes[layer_id]
                    self.indexer_kv_buffer.append(
                        torch.zeros(
                            (
                                compressed_pages,
                                indexer_block_size * layout.indexer_row_bytes,
                            ),
                            dtype=torch.uint8,
                            device=device,
                        )
                        if has_indexer
                        else None
                    )
                    indexer_state_block_size = self.indexer_state_block_sizes[layer_id]
                    indexer_state_pages = self.num_pages
                    if has_indexer:
                        indexer_state_pages = self.paged_cache_group_page_counts.get(
                            V4_INDEXER_COMPRESSOR_STATE_GROUP_ID,
                            self.num_pages,
                        )
                    self.indexer_state_buffer.append(
                        torch.empty(
                            (
                                indexer_state_pages,
                                indexer_state_block_size,
                                layout.state_width(layer_id, indexer=True) * 2,
                            ),
                            dtype=torch.float32,
                            device=device,
                        )
                        if has_indexer
                        else None
                    )

        logger.info(
            "Initialized DeepSeek V4 KV pool: %d pages, %d layers, fp4 indexer=%s, compressed block sizes=%s",
            self.num_pages,
            layer_num,
            layout.use_fp4_indexer_cache,
            self.compressed_block_sizes,
        )

    @property
    def prefix_cache_required_group_ids(self) -> tuple[str, ...]:
        return tuple(
            str(spec.group_id)
            for spec in self.scheduler_group_specs
            if spec.prefix_role == "history_anchor"
        )

    @property
    def arena_generation(self) -> int | None:
        """Shared flat-arena generation, or ``None`` on the radix path."""
        if self.flat_arena_set is None:
            return None
        return self.flat_arena_set.arena_generation

    @property
    def device_cache_arena(self) -> V4FlatArenaSet | None:
        """Return the shared flat arena that owns this pool's device tensors."""
        return self.flat_arena_set

    def bind_paged_cache_scheduler(self, scheduler: object) -> None:
        self._paged_cache_scheduler = scheduler

    def maybe_log_paged_cache_group_pages(self) -> None:
        scheduler = self._paged_cache_scheduler
        if self.rank != 0 or scheduler is None or not self._paged_cache_state_group_ids:
            return
        if not logger.isEnabledFor(logging.DEBUG):
            return

        if self.flat_memory_plan is not None:
            snapshot_fn = getattr(scheduler, "flat_pool_snapshots", None)
            if not callable(snapshot_fn):
                return
            snapshots = {str(snapshot.pool_id): snapshot for snapshot in snapshot_fn()}
            specs_by_id = {str(spec.group_id): spec for spec in self.owner_group_specs}
            parts = []
            for group_id in self._paged_cache_state_group_ids:
                spec = specs_by_id[group_id]
                snapshot = snapshots.get(str(spec.pool_id))
                if snapshot is None:
                    raise RuntimeError(
                        "DeepSeek V4 flat scheduler snapshot is missing pool "
                        f"{spec.pool_id!r} for group {group_id!r}"
                    )
                parts.append(
                    f"{group_id}/{spec.pool_id}: "
                    f"active={int(snapshot.active_blocks)}/"
                    f"{int(snapshot.usable_blocks)}, "
                    f"free={int(snapshot.free_blocks)}, "
                    f"cached={int(snapshot.cached_evictable_blocks)}, "
                    f"reserved={int(snapshot.reserved_blocks)}"
                )
            logger.debug(
                "DeepSeek V4 flat cache state group pages. %s", "; ".join(parts)
            )
            return

        parts = []
        for group_id in self._paged_cache_state_group_ids:
            total = scheduler.paged_cache_group_total_pages(group_id)
            available = scheduler.paged_cache_group_available_pages(group_id)
            failed = scheduler.paged_cache_group_failed_alloc_count(group_id)
            parts.append(
                f"{group_id}: used={total - available}/{total}, "
                f"available={available}, failed_alloc={failed}"
            )
        logger.debug("DeepSeek V4 paged-cache state group pages. %s", "; ".join(parts))

    def _require(
        self,
        buffers: Sequence[torch.Tensor | None],
        layer_id: int,
        name: str,
    ) -> torch.Tensor:
        buf = buffers[layer_id]
        if buf is None:
            raise ValueError(f"DeepSeek V4 layer {layer_id} has no {name} cache")
        return buf

    def get_swa_kv_buffer(self, layer_id: int) -> torch.Tensor:
        return self.swa_kv_buffer[layer_id]

    @property
    def state_block_size(self) -> int:
        """Legacy/radix state-block fallback; Flat requires a group table."""
        if self.flat_memory_plan is not None:
            raise RuntimeError(
                "DeepSeek V4 Flat KV state geometry requires its group-specific "
                "block table"
            )
        return self._legacy_state_block_size

    @property
    def swa_capacity_pages(self) -> int:
        """Writable owner-local SWA capacity shared by every layer, in pages."""

        if not self.swa_kv_buffer:
            return 0
        return int(self.swa_kv_buffer[0].shape[0])

    @property
    def swa_capacity_slots(self) -> int:
        """Writable SWA cache capacity shared by every layer, in token slots.

        Every layer's SWA buffer is allocated with the same page count, so a
        single capacity (pages * tokens per block) bounds the write-slot
        mapping shared across layers. Returns 0 when no SWA buffers exist;
        callers must then mask all slots rather than skip the bounds check.
        """
        return self.swa_capacity_pages * int(self.swa_block_size)

    def get_compressed_kv_buffer_2d(self, layer_id: int) -> torch.Tensor:
        return self._require(self.compressed_kv_buffer, layer_id, "compressed KV")

    def get_compressed_block_size(self, layer_id: int) -> int:
        return self.compressed_block_sizes[layer_id]

    def get_indexer_block_size(self, layer_id: int) -> int:
        block_size = self.indexer_block_sizes[layer_id]
        if block_size <= 0:
            raise ValueError(f"DeepSeek V4 layer {layer_id} has no indexer cache")
        return block_size

    def get_compressor_state_block_size(self, layer_id: int) -> int:
        block_size = self.compressor_state_block_sizes[layer_id]
        if block_size <= 0:
            raise ValueError(
                f"DeepSeek V4 layer {layer_id} has no compressor state cache"
            )
        return block_size

    def get_compressor_state_buffer(self, layer_id: int) -> torch.Tensor:
        return self._require(self.compressor_state_buffer, layer_id, "compressor state")

    def get_compressor_state_view(self, layer_id: int) -> torch.Tensor:
        buf = self.get_compressor_state_buffer(layer_id)
        block_size = self.get_compressor_state_block_size(layer_id)
        return buf.view(-1, block_size, buf.shape[-1])

    def get_indexer_kv_buffer_2d(self, layer_id: int) -> torch.Tensor:
        return self._require(self.indexer_kv_buffer, layer_id, "indexer KV")

    def get_indexer_state_block_size(self, layer_id: int) -> int:
        block_size = self.indexer_state_block_sizes[layer_id]
        if block_size <= 0:
            raise ValueError(f"DeepSeek V4 layer {layer_id} has no indexer state cache")
        return block_size

    def get_indexer_state_buffer(self, layer_id: int) -> torch.Tensor:
        return self._require(self.indexer_state_buffer, layer_id, "indexer state")

    def get_indexer_state_view(self, layer_id: int) -> torch.Tensor:
        buf = self.get_indexer_state_buffer(layer_id)
        block_size = self.get_indexer_state_block_size(layer_id)
        return buf.view(-1, block_size, buf.shape[-1])

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.get_swa_kv_buffer(layer_id)

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.get_swa_kv_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        buf = self.get_swa_kv_buffer(layer_id)
        return buf, buf

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "DeepSeek V4 writes KV cache through V4 attention helpers"
        )

    def _move_fp8_ds_mla_rows(
        self,
        buf: torch.Tensor,
        tgt_loc: torch.Tensor,
        src_loc: torch.Tensor,
        block_size: int,
    ) -> None:
        if tgt_loc.numel() == 0:
            return
        flat = buf.reshape(-1)
        tgt = tgt_loc.to(torch.int64)
        src = src_loc.to(torch.int64)
        tgt_page = torch.div(tgt, block_size, rounding_mode="floor")
        src_page = torch.div(src, block_size, rounding_mode="floor")
        tgt_pos = tgt % block_size
        src_pos = src % block_size
        block_stride = buf.stride(0)
        token_stride = self.layout.swa_token_stride
        scale_dim = self.layout.swa_scale_dim

        value_offsets = torch.arange(
            token_stride,
            dtype=torch.int64,
            device=buf.device,
        )
        tgt_value = (
            tgt_page[:, None] * block_stride
            + tgt_pos[:, None] * token_stride
            + value_offsets[None, :]
        )
        src_value = (
            src_page[:, None] * block_stride
            + src_pos[:, None] * token_stride
            + value_offsets[None, :]
        )
        value_rows = flat[src_value].clone()
        flat[tgt_value] = value_rows

        scale_offsets = torch.arange(
            scale_dim,
            dtype=torch.int64,
            device=buf.device,
        )
        scale_base = block_size * token_stride
        tgt_scale = (
            tgt_page[:, None] * block_stride
            + scale_base
            + tgt_pos[:, None] * scale_dim
            + scale_offsets[None, :]
        )
        src_scale = (
            src_page[:, None] * block_stride
            + scale_base
            + src_pos[:, None] * scale_dim
            + scale_offsets[None, :]
        )
        scale_rows = flat[src_scale].clone()
        flat[tgt_scale] = scale_rows

    def _move_rows(
        self,
        buf: torch.Tensor,
        row_bytes: int,
        tgt_loc: torch.Tensor,
        src_loc: torch.Tensor,
        block_size: int,
    ) -> None:
        rows = buf.view(-1, block_size, row_bytes).reshape(-1, row_bytes)
        rows[tgt_loc.long()] = rows[src_loc.long()]

    def _compressed_locs_from_token_locs(
        self,
        loc: torch.Tensor,
        *,
        ratio: int,
        block_size: int,
    ) -> torch.Tensor:
        page = torch.div(loc.to(torch.int64), self.page_size, rounding_mode="floor")
        pos = loc.to(torch.int64) % self.page_size
        return page * block_size + torch.div(pos, ratio, rounding_mode="floor")

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor) -> None:
        if tgt_loc.numel() == 0:
            return
        for layer_id in range(self.layer_num):
            self._move_fp8_ds_mla_rows(
                self.swa_kv_buffer[layer_id],
                tgt_loc,
                src_loc,
                self.swa_block_size,
            )
            buf = self.compressed_kv_buffer[layer_id]
            if buf is not None:
                ratio = self.layout.layer_ratio[layer_id]
                block_size = self.get_compressed_block_size(layer_id)
                self._move_fp8_ds_mla_rows(
                    buf,
                    self._compressed_locs_from_token_locs(
                        tgt_loc, ratio=ratio, block_size=block_size
                    ),
                    self._compressed_locs_from_token_locs(
                        src_loc, ratio=ratio, block_size=block_size
                    ),
                    block_size,
                )
            for buffers, row_bytes in (
                (self.indexer_kv_buffer, self.layout.indexer_row_bytes),
            ):
                buf = buffers[layer_id]
                if buf is not None:
                    ratio = self.layout.layer_ratio[layer_id]
                    block_size = self.get_indexer_block_size(layer_id)
                    self._move_rows(
                        buf,
                        row_bytes,
                        self._compressed_locs_from_token_locs(
                            tgt_loc, ratio=ratio, block_size=block_size
                        ),
                        self._compressed_locs_from_token_locs(
                            src_loc, ratio=ratio, block_size=block_size
                        ),
                        block_size,
                    )
            for buffers in (self.compressor_state_buffer, self.indexer_state_buffer):
                buf = buffers[layer_id]
                if buf is not None:
                    rows = buf.view(-1, buf.shape[-1])
                    rows[tgt_loc.long()] = rows[src_loc.long()]

    def _all_buffers(self) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        for layer_id in range(self.layer_num):
            out.append(self.swa_kv_buffer[layer_id])
            for buffers in (
                self.compressed_kv_buffer,
                self.compressor_state_buffer,
                self.indexer_kv_buffer,
                self.indexer_state_buffer,
            ):
                buf = buffers[layer_id]
                if buf is not None:
                    out.append(buf)
        return out

    def get_kv_size_bytes(self) -> int:
        return int(
            sum(np.prod(buf.shape) * buf.dtype.itemsize for buf in self._all_buffers())
        )

    def get_contiguous_buf_infos(self):
        buffers = self._all_buffers()
        return (
            [buf.data_ptr() for buf in buffers],
            [buf.nbytes for buf in buffers],
            [buf[0].nbytes for buf in buffers],
        )

    def get_layerwise_buf_info_offsets(self, start_idx=0):
        offsets = []
        cursor = start_idx
        for layer_id in range(self.layer_num):
            layer_offsets = [cursor]
            cursor += 1
            for buffers in (
                self.compressed_kv_buffer,
                self.compressor_state_buffer,
                self.indexer_kv_buffer,
                self.indexer_state_buffer,
            ):
                if buffers[layer_id] is not None:
                    layer_offsets.append(cursor)
                    cursor += 1
            offsets.append(layer_offsets)
        return offsets

    def get_cpu_copy(self, token_indices: list[int]) -> list[torch.Tensor]:
        del token_indices
        raise NotImplementedError(
            "DeepSeek V4 KV cache offload is not implemented; the compressed-MQA "
            "and indexer buffers are page-shaped and require page-aware indexing."
        )

    def load_cpu_copy(self, kv_cache_cpu, token_indices: list[int]) -> None:
        del kv_cache_cpu, token_indices
        raise NotImplementedError(
            "DeepSeek V4 KV cache reload is not implemented; the compressed-MQA "
            "and indexer buffers are page-shaped and require page-aware indexing."
        )
