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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    PagedCacheGroupSpec,
    compute_paged_cache_group_page_counts,
)

V4_KERNEL_BLOCK_ROWS: int = 64
V4_SWA_KV_GROUP_ID = "v4.swa_kv"
V4_INDEXER_COMPRESSOR_STATE_GROUP_ID = "v4.c4a.indexer_compressor_state"
DEEPSEEK_V4_FP8_MAX = 448.0
DEEPSEEK_V4_FP8_BLOCK_SIZE = 128
DEEPSEEK_V4_FP8_QUANT_BLOCK = 64
DEEPSEEK_V4_FP8_INDEXER_BLOCK_SIZE = 128
DEEPSEEK_V4_FP8_SCALE_BYTES = 4
DEEPSEEK_V4_MXFP4_BLOCK_SIZE = 32
DEEPSEEK_V4_MXFP4_SCALE_BYTES = 1
DEEPSEEK_V4_SPARSE_PREFILL_TOPK_ALIGNMENT = 128
DEEPSEEK_V4_COMPRESSED_LOGICAL_BLOCK_SIZE = 256
_COMPRESSOR_STATE_WINDOW_TOKENS = {4: 8, 128: 128}
_COMPRESSOR_STATE_ROWS_PER_PAGE = {4: 4, 128: 8}


def deepseek_v4_nope_dim(head_dim: int, rope_dim: int) -> int:
    nope_dim = int(head_dim) - int(rope_dim)
    if nope_dim <= 0:
        raise ValueError(f"head_dim={head_dim} must be larger than rope_dim={rope_dim}")
    return nope_dim


def deepseek_v4_swa_token_stride(head_dim: int, rope_dim: int) -> int:
    return deepseek_v4_nope_dim(head_dim, rope_dim) + int(rope_dim) * 2


def deepseek_v4_swa_scale_dim(head_dim: int, rope_dim: int) -> int:
    nope_dim = deepseek_v4_nope_dim(head_dim, rope_dim)
    if nope_dim % DEEPSEEK_V4_FP8_QUANT_BLOCK != 0:
        raise ValueError(
            "DeepSeek V4 FP8 NoPE dim must be divisible by "
            f"{DEEPSEEK_V4_FP8_QUANT_BLOCK}, got {nope_dim}"
        )
    return nope_dim // DEEPSEEK_V4_FP8_QUANT_BLOCK + 1


def deepseek_v4_swa_row_bytes(head_dim: int, rope_dim: int) -> int:
    return deepseek_v4_swa_token_stride(head_dim, rope_dim) + deepseek_v4_swa_scale_dim(
        head_dim, rope_dim
    )


def deepseek_v4_indexer_mxfp4_value_bytes(index_head_dim: int) -> int:
    index_head_dim = int(index_head_dim)
    if index_head_dim % 2 != 0:
        raise ValueError(f"MXFP4 index head dim must be even, got {index_head_dim}")
    return index_head_dim // 2


def deepseek_v4_indexer_mxfp4_scale_dim(index_head_dim: int) -> int:
    index_head_dim = int(index_head_dim)
    if index_head_dim % DEEPSEEK_V4_MXFP4_BLOCK_SIZE != 0:
        raise ValueError(
            "MXFP4 index head dim must be divisible by "
            f"{DEEPSEEK_V4_MXFP4_BLOCK_SIZE}, got {index_head_dim}"
        )
    return (
        index_head_dim // DEEPSEEK_V4_MXFP4_BLOCK_SIZE * DEEPSEEK_V4_MXFP4_SCALE_BYTES
    )


def deepseek_v4_indexer_mxfp4_row_bytes(index_head_dim: int) -> int:
    return deepseek_v4_indexer_mxfp4_value_bytes(
        index_head_dim
    ) + deepseek_v4_indexer_mxfp4_scale_dim(index_head_dim)


def deepseek_v4_indexer_mxfp4_layout_from_row_bytes(
    row_bytes: int,
) -> tuple[int, int, int]:
    row_bytes = int(row_bytes)
    value_bytes_per_block = DEEPSEEK_V4_MXFP4_BLOCK_SIZE // 2
    bytes_per_block = value_bytes_per_block + DEEPSEEK_V4_MXFP4_SCALE_BYTES
    if row_bytes % bytes_per_block != 0:
        raise ValueError(
            f"MXFP4 indexer row bytes must be value+scale aligned, got {row_bytes}"
        )
    num_blocks = row_bytes // bytes_per_block
    value_bytes = num_blocks * value_bytes_per_block
    scale_bytes = num_blocks * DEEPSEEK_V4_MXFP4_SCALE_BYTES
    index_head_dim = num_blocks * DEEPSEEK_V4_MXFP4_BLOCK_SIZE
    if deepseek_v4_indexer_mxfp4_scale_dim(index_head_dim) != scale_bytes:
        raise ValueError(
            f"invalid MXFP4 indexer row bytes {row_bytes} for "
            f"index_head_dim={index_head_dim}"
        )
    return index_head_dim, value_bytes, scale_bytes


def deepseek_v4_indexer_fp8_scale_bytes(index_head_dim: int) -> int:
    index_head_dim = int(index_head_dim)
    if index_head_dim % DEEPSEEK_V4_FP8_INDEXER_BLOCK_SIZE != 0:
        raise ValueError(
            "FP8 index head dim must be divisible by "
            f"{DEEPSEEK_V4_FP8_INDEXER_BLOCK_SIZE}, got {index_head_dim}"
        )
    return (
        index_head_dim
        // DEEPSEEK_V4_FP8_INDEXER_BLOCK_SIZE
        * DEEPSEEK_V4_FP8_SCALE_BYTES
    )


def deepseek_v4_indexer_fp8_row_bytes(index_head_dim: int) -> int:
    return int(index_head_dim) + deepseek_v4_indexer_fp8_scale_bytes(index_head_dim)


def deepseek_v4_indexer_fp8_layout_from_row_bytes(
    row_bytes: int,
) -> tuple[int, int]:
    row_bytes = int(row_bytes)
    bytes_per_block = DEEPSEEK_V4_FP8_INDEXER_BLOCK_SIZE + DEEPSEEK_V4_FP8_SCALE_BYTES
    if row_bytes % bytes_per_block != 0:
        raise ValueError(
            f"FP8 indexer row bytes must be value+scale aligned, got {row_bytes}"
        )
    index_head_dim = row_bytes // bytes_per_block * DEEPSEEK_V4_FP8_INDEXER_BLOCK_SIZE
    scale_bytes = deepseek_v4_indexer_fp8_scale_bytes(index_head_dim)
    if index_head_dim + scale_bytes != row_bytes:
        raise ValueError(
            f"invalid FP8 indexer row bytes {row_bytes} for "
            f"index_head_dim={index_head_dim}"
        )
    return index_head_dim, scale_bytes


@dataclass(frozen=True)
class DeepseekV4CacheLayout:
    """Per-model cache geometry derived from the HF config: layer compress
    ratios plus the per-row byte formulas the field shapes are built from."""

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

    def swa_block_bytes(self, page_size: int | None = None) -> int:
        if page_size is None:
            page_size = self.page_size
        block_bytes = page_size * self.swa_row_bytes
        alignment = self.swa_token_stride
        return ((block_bytes + alignment - 1) // alignment) * alignment

    def storage_block_size(self, compress_ratio: int) -> int:
        if compress_ratio > 1:
            return max(1, DEEPSEEK_V4_COMPRESSED_LOGICAL_BLOCK_SIZE // compress_ratio)
        return self.page_size

    def compressor_state_block_size(self, compress_ratio: int) -> int:
        if compress_ratio == 4:
            return 4
        if compress_ratio == 128:
            return 8
        return self.page_size

    @property
    def indexer_row_bytes(self) -> int:
        if self.use_fp4_indexer_cache:
            return deepseek_v4_indexer_mxfp4_row_bytes(self.index_head_dim)
        return deepseek_v4_indexer_fp8_row_bytes(self.index_head_dim)

    def state_width(self, layer_id: int, *, indexer: bool = False) -> int:
        if indexer:
            return self.index_head_dim * 2
        return self.head_dim * (2 if self.layer_ratio[layer_id] == 4 else 1)


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


def v4_compressor_state_group_id(ratio: int) -> str:
    return f"v4.c{int(ratio)}a.compressor_state"


def v4_compressed_kv_group_id(ratio: int) -> str:
    return f"v4.c{int(ratio)}a.compressed_kv"


def parse_v4_compressor_state_group_id(group_id: str) -> int | None:
    prefix = "v4.c"
    suffix = "a.compressor_state"
    if not group_id.startswith(prefix) or not group_id.endswith(suffix):
        return None
    ratio_text = group_id[len(prefix) : -len(suffix)]
    try:
        return int(ratio_text)
    except ValueError:
        return None


def parse_v4_compressed_kv_group_id(group_id: str) -> int | None:
    """Return the compress ratio of a ``v4.c{ratio}a.compressed_kv`` group id,
    or ``None`` when the id is not a compressed-KV (full-history) group."""
    prefix = "v4.c"
    suffix = "a.compressed_kv"
    if not group_id.startswith(prefix) or not group_id.endswith(suffix):
        return None
    ratio_text = group_id[len(prefix) : -len(suffix)]
    try:
        return int(ratio_text)
    except ValueError:
        return None


def first_v4_compressed_kv_group_id(group_ids) -> str | None:
    """Pick the smallest-ratio compressed-KV group id present in ``group_ids``.

    Mirrors the executor's ``next(...)`` full-history selection (contract order
    is ratio-ascending), so the base page table for ratio<=1 indexer layers
    resolves to the same group whether it comes from cache metadata or a
    capture-time block-tables dict.
    """
    ratios = {
        parse_v4_compressed_kv_group_id(gid): gid
        for gid in group_ids
        if parse_v4_compressed_kv_group_id(gid) is not None
    }
    if not ratios:
        return None
    return ratios[min(ratios)]


def _compressed_kernel_block_size(ratio: int) -> int:
    if ratio <= 1:
        raise ValueError(f"ratio must be > 1, got {ratio}")
    return max(1, DEEPSEEK_V4_COMPRESSED_LOGICAL_BLOCK_SIZE // ratio)


def _resolve_sliding_window(hf_config: Any) -> int:
    for source in (hf_config, getattr(hf_config, "text_config", None)):
        if source is None:
            continue
        if hasattr(source, "sliding_window"):
            value = source.sliding_window
            if value is None:
                raise ValueError("DeepSeek V4 sliding_window is None")
            window = int(value)
            if window <= 0:
                raise ValueError(f"sliding_window must be positive, got {value!r}")
            return window
    raise ValueError("DeepSeek V4 hf_config is missing sliding_window")


def build_v4_cache_specs(
    hf_config: Any,
    *,
    layer_ratio: Sequence[int],
    cache_blocks_per_lcm_block: Mapping[str, int] | None = None,
    decode_input_tokens: int = 1,
) -> list[PagedCacheGroupSpec]:
    if (
        isinstance(decode_input_tokens, bool)
        or not isinstance(decode_input_tokens, int)
        or decode_input_tokens <= 0
    ):
        raise ValueError("decode_input_tokens must be a positive integer")
    swa_window = _resolve_sliding_window(hf_config)
    unique_compress_ratios = sorted({int(r) for r in layer_ratio if int(r) > 1})
    # c4 compression consumes the prior four-token state plus every token in
    # the target verify block. Preserve the historical eight-token window for
    # verify widths <= 4 and grow it for wider block-speculative decoders.
    c4_state_window = max(
        _COMPRESSOR_STATE_WINDOW_TOKENS[4],
        4 + decode_input_tokens,
    )

    specs: list[PagedCacheGroupSpec] = [
        # SWA kv: trailing window only -> State family.
        PagedCacheGroupSpec(
            group_id=V4_SWA_KV_GROUP_ID,
            retention="sliding_window",
            rows_per_page=V4_KERNEL_BLOCK_ROWS,
            entry_stride_tokens=1,
            sliding_window_tokens=swa_window,
            family="state",
        ),
    ]
    for ratio in unique_compress_ratios:
        if ratio not in _COMPRESSOR_STATE_WINDOW_TOKENS:
            raise ValueError(f"unsupported DeepSeek V4 compress_ratio={ratio}")
        # Compressor state: tail buffer -> State family.
        specs.append(
            PagedCacheGroupSpec(
                group_id=v4_compressor_state_group_id(ratio),
                retention="sliding_window",
                rows_per_page=_COMPRESSOR_STATE_ROWS_PER_PAGE[ratio],
                entry_stride_tokens=1,
                sliding_window_tokens=(
                    c4_state_window
                    if ratio == 4
                    else _COMPRESSOR_STATE_WINDOW_TOKENS[ratio]
                ),
                family="state",
            )
        )
        # Compressed kv: full-history chain (indexer K shares this group).
        specs.append(
            PagedCacheGroupSpec(
                group_id=v4_compressed_kv_group_id(ratio),
                retention="full_history",
                rows_per_page=_compressed_kernel_block_size(ratio),
                entry_stride_tokens=ratio,
                sliding_window_tokens=None,
                family="history",
            )
        )
    if 4 in unique_compress_ratios:
        # Indexer compressor state: tail buffer -> State family.
        specs.append(
            PagedCacheGroupSpec(
                group_id=V4_INDEXER_COMPRESSOR_STATE_GROUP_ID,
                retention="sliding_window",
                rows_per_page=_COMPRESSOR_STATE_ROWS_PER_PAGE[4],
                entry_stride_tokens=1,
                sliding_window_tokens=c4_state_window,
                family="state",
            )
        )
    if cache_blocks_per_lcm_block is None:
        return specs

    packing = dict(cache_blocks_per_lcm_block)
    group_ids = {spec.group_id for spec in specs}
    if set(packing) != group_ids:
        raise ValueError(
            "DeepSeek V4 LCM packing must contain exactly the cache groups"
        )
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count <= 0
        for count in packing.values()
    ):
        raise ValueError("DeepSeek V4 LCM packing values must be positive integers")
    return [
        replace(
            spec,
            cache_blocks_per_lcm_block=packing[spec.group_id],
        )
        for spec in specs
    ]


def deepseek_v4_lcm_blocks_needed(
    specs: Sequence[PagedCacheGroupSpec],
    *,
    logical_block_tokens: int,
    token_capacity: int,
    max_live_requests: int,
    max_scheduled_tokens: int,
    max_context_len: int,
    decode_input_tokens: int = 1,
    overlap_schedule_depth: int = 0,
) -> int:
    """Return physical parents needed by per-group CacheBlock tables."""
    if logical_block_tokens <= 0:
        raise ValueError("logical_block_tokens must be positive")
    if token_capacity <= 0:
        raise ValueError("token_capacity must be positive")
    for spec in specs:
        cache_block_tokens = spec.cache_block_tokens
        if cache_block_tokens <= 0 or logical_block_tokens % cache_block_tokens:
            raise ValueError(
                f"group {spec.group_id!r} cache block tokens must divide "
                f"logical_block_tokens={logical_block_tokens}"
            )
    counts = compute_paged_cache_group_page_counts(
        specs,
        max_live_requests=max_live_requests,
        max_scheduled_tokens=max_scheduled_tokens,
        max_total_tokens=token_capacity,
        max_context_len=max_context_len,
        decode_input_tokens=decode_input_tokens,
        overlap_schedule_depth=overlap_schedule_depth,
    )
    parents = 0
    for spec in specs:
        child_pages = counts[spec.group_id] - 1  # page 0 is the null page
        packing = spec.cache_blocks_per_lcm_block
        parents += (child_pages + packing - 1) // packing
    return parents


def deepseek_v4_token_capacity_for_cache_pool(
    specs: Sequence[PagedCacheGroupSpec],
    *,
    logical_block_tokens: int,
    num_lcm_blocks: int,
    max_live_requests: int,
    max_scheduled_tokens: int,
    max_context_len: int,
    decode_input_tokens: int = 1,
    overlap_schedule_depth: int = 0,
    upper_bound_tokens: int,
) -> int:
    """Invert :func:`deepseek_v4_lcm_blocks_needed` monotonically."""
    if num_lcm_blocks <= 0:
        raise ValueError("num_lcm_blocks must be positive")
    if upper_bound_tokens <= 0:
        raise ValueError("upper_bound_tokens must be positive")
    sizing = {
        "logical_block_tokens": logical_block_tokens,
        "max_live_requests": max_live_requests,
        "max_scheduled_tokens": max_scheduled_tokens,
        "max_context_len": max_context_len,
        "decode_input_tokens": decode_input_tokens,
        "overlap_schedule_depth": overlap_schedule_depth,
    }
    low, high = 0, upper_bound_tokens
    while low < high:
        candidate = (low + high + 1) // 2
        if (
            deepseek_v4_lcm_blocks_needed(
                specs,
                token_capacity=candidate,
                **sizing,
            )
            <= num_lcm_blocks
        ):
            low = candidate
        else:
            high = candidate - 1
    if low == 0:
        raise ValueError(
            f"num_lcm_blocks={num_lcm_blocks} cannot admit one token with "
            "the configured DeepSeek V4 scheduler limits"
        )
    return low


__all__ = [
    "DEEPSEEK_V4_COMPRESSED_LOGICAL_BLOCK_SIZE",
    "DEEPSEEK_V4_FP8_BLOCK_SIZE",
    "DEEPSEEK_V4_FP8_INDEXER_BLOCK_SIZE",
    "DEEPSEEK_V4_FP8_MAX",
    "DEEPSEEK_V4_FP8_QUANT_BLOCK",
    "DEEPSEEK_V4_FP8_SCALE_BYTES",
    "DEEPSEEK_V4_MXFP4_BLOCK_SIZE",
    "DEEPSEEK_V4_MXFP4_SCALE_BYTES",
    "DEEPSEEK_V4_SPARSE_PREFILL_TOPK_ALIGNMENT",
    "V4_INDEXER_COMPRESSOR_STATE_GROUP_ID",
    "V4_KERNEL_BLOCK_ROWS",
    "V4_SWA_KV_GROUP_ID",
    "DeepseekV4CacheLayout",
    "build_v4_cache_specs",
    "deepseek_v4_cache_layout_from_config",
    "deepseek_v4_indexer_fp8_layout_from_row_bytes",
    "deepseek_v4_indexer_fp8_row_bytes",
    "deepseek_v4_indexer_fp8_scale_bytes",
    "deepseek_v4_indexer_mxfp4_layout_from_row_bytes",
    "deepseek_v4_indexer_mxfp4_row_bytes",
    "deepseek_v4_indexer_mxfp4_scale_dim",
    "deepseek_v4_indexer_mxfp4_value_bytes",
    "deepseek_v4_lcm_blocks_needed",
    "deepseek_v4_nope_dim",
    "deepseek_v4_swa_row_bytes",
    "deepseek_v4_swa_scale_dim",
    "deepseek_v4_swa_token_stride",
    "deepseek_v4_token_capacity_for_cache_pool",
    "first_v4_compressed_kv_group_id",
    "parse_v4_compressed_kv_group_id",
    "parse_v4_compressor_state_group_id",
    "v4_compressed_kv_group_id",
    "v4_compressor_state_group_id",
]
