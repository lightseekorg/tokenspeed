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

from collections.abc import Sequence
from typing import Any

from tokenspeed.runtime.configs.paged_cache_spec import (
    CACHE_OWNER_DRAFT,
    CACHE_OWNER_TARGET,
    PagedCacheGroupSpec,
)

V4_KERNEL_BLOCK_ROWS: int = 64
V4_SWA_KV_GROUP_ID = "v4.swa_kv"
V4_INDEXER_COMPRESSOR_STATE_GROUP_ID = "v4.c4a.indexer_compressor_state"
V4_SWA_POOL_ID = "v4.swa"
V4_C4_STATE_POOL_ID = "v4.c4.state"
V4_C4_HISTORY_POOL_ID = "v4.c4.history"
V4_C128_STATE_POOL_ID = "v4.c128.state"
V4_C128_HISTORY_POOL_ID = "v4.c128.history"
V4_INDEX_STATE_POOL_ID = "v4.index.state"

# Producer domains are bounded executor-visible completion domains, not one bit
# per layer/tensor.  They intentionally occupy a different namespace from the
# target/draft owner mask above.
V4_PRODUCER_TARGET_MAIN = 1 << 0
V4_PRODUCER_TARGET_INDEXER = 1 << 1
V4_PRODUCER_DRAFT_MAIN = 1 << 2
V4_PRODUCER_DRAFT_INDEXER = 1 << 3
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


def _compressed_kernel_block_size(ratio: int) -> int:
    if ratio <= 1:
        raise ValueError(f"ratio must be > 1, got {ratio}")
    return max(1, DEEPSEEK_V4_COMPRESSED_LOGICAL_BLOCK_SIZE // ratio)


def _v4_pool_id_for_ratio(ratio: int, *, history: bool) -> str:
    pool_ids = {
        (4, False): V4_C4_STATE_POOL_ID,
        (4, True): V4_C4_HISTORY_POOL_ID,
        (128, False): V4_C128_STATE_POOL_ID,
        (128, True): V4_C128_HISTORY_POOL_ID,
    }
    try:
        return pool_ids[(ratio, history)]
    except KeyError as exc:
        raise ValueError(f"unsupported DeepSeek V4 compress_ratio={ratio}") from exc


def _producer_domain_mask(
    owner_mask: int,
    *,
    main: bool = False,
    indexer: bool = False,
) -> int:
    supported_owners = CACHE_OWNER_TARGET | CACHE_OWNER_DRAFT
    if (
        isinstance(owner_mask, bool)
        or not isinstance(owner_mask, int)
        or owner_mask <= 0
        or owner_mask & ~supported_owners
    ):
        raise ValueError(
            "DeepSeek V4 owner_mask must contain only CACHE_OWNER_TARGET "
            f"and/or CACHE_OWNER_DRAFT, got {owner_mask!r}"
        )
    mask = 0
    if owner_mask & CACHE_OWNER_TARGET:
        if main:
            mask |= V4_PRODUCER_TARGET_MAIN
        if indexer:
            mask |= V4_PRODUCER_TARGET_INDEXER
    if owner_mask & CACHE_OWNER_DRAFT:
        if main:
            mask |= V4_PRODUCER_DRAFT_MAIN
        if indexer:
            mask |= V4_PRODUCER_DRAFT_INDEXER
    return mask


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
    owner_mask: int = CACHE_OWNER_TARGET,
) -> list[PagedCacheGroupSpec]:
    """Return the owner-local V4 logical groups and physical pool bindings.

    Args:
        hf_config: Model config containing a positive ``sliding_window``.
        layer_ratio: Per-layer compression ratios present in this owner.
        owner_mask: Target and/or draft ownership bits used to derive bounded
            producer-domain completion masks.

    Returns:
        Six explicitly bound storage classes for the typical ``1/4/128``
        topology, reduced to only the compression ratios actually present.
    """
    main_producer_mask = _producer_domain_mask(owner_mask, main=True)
    indexer_producer_mask = _producer_domain_mask(owner_mask, indexer=True)
    swa_window = _resolve_sliding_window(hf_config)
    unique_compress_ratios = sorted({int(r) for r in layer_ratio if int(r) > 1})

    specs: list[PagedCacheGroupSpec] = [
        # SWA kv: trailing window only -> State family.
        PagedCacheGroupSpec(
            group_id=V4_SWA_KV_GROUP_ID,
            retention="sliding_window",
            rows_per_page=V4_KERNEL_BLOCK_ROWS,
            entry_stride_tokens=1,
            sliding_window_tokens=swa_window,
            family="state",
            block_size_tokens=V4_KERNEL_BLOCK_ROWS,
            pool_id=V4_SWA_POOL_ID,
            prefix_role="continuation_state",
            table_layout="bounded_window",
            required_producer_domain_mask=main_producer_mask,
            owner_mask=owner_mask,
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
                sliding_window_tokens=_COMPRESSOR_STATE_WINDOW_TOKENS[ratio],
                family="state",
                block_size_tokens=_COMPRESSOR_STATE_ROWS_PER_PAGE[ratio],
                pool_id=_v4_pool_id_for_ratio(ratio, history=False),
                prefix_role="continuation_state",
                table_layout="bounded_window",
                required_producer_domain_mask=main_producer_mask,
                owner_mask=owner_mask,
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
                block_size_tokens=(_compressed_kernel_block_size(ratio) * ratio),
                pool_id=_v4_pool_id_for_ratio(ratio, history=True),
                prefix_role="history_anchor",
                table_layout="absolute",
                required_producer_domain_mask=_producer_domain_mask(
                    owner_mask,
                    main=True,
                    indexer=ratio == 4,
                ),
                owner_mask=owner_mask,
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
                sliding_window_tokens=_COMPRESSOR_STATE_WINDOW_TOKENS[4],
                family="state",
                block_size_tokens=_COMPRESSOR_STATE_ROWS_PER_PAGE[4],
                pool_id=V4_INDEX_STATE_POOL_ID,
                prefix_role="continuation_state",
                table_layout="bounded_window",
                required_producer_domain_mask=indexer_producer_mask,
                owner_mask=owner_mask,
            )
        )
    return specs


__all__ = [
    "CACHE_OWNER_DRAFT",
    "CACHE_OWNER_TARGET",
    "DEEPSEEK_V4_FP8_BLOCK_SIZE",
    "DEEPSEEK_V4_COMPRESSED_LOGICAL_BLOCK_SIZE",
    "DEEPSEEK_V4_FP8_MAX",
    "DEEPSEEK_V4_FP8_INDEXER_BLOCK_SIZE",
    "DEEPSEEK_V4_FP8_QUANT_BLOCK",
    "DEEPSEEK_V4_FP8_SCALE_BYTES",
    "DEEPSEEK_V4_MXFP4_BLOCK_SIZE",
    "DEEPSEEK_V4_MXFP4_SCALE_BYTES",
    "DEEPSEEK_V4_SPARSE_PREFILL_TOPK_ALIGNMENT",
    "V4_INDEXER_COMPRESSOR_STATE_GROUP_ID",
    "V4_C128_HISTORY_POOL_ID",
    "V4_C128_STATE_POOL_ID",
    "V4_C4_HISTORY_POOL_ID",
    "V4_C4_STATE_POOL_ID",
    "V4_INDEX_STATE_POOL_ID",
    "V4_KERNEL_BLOCK_ROWS",
    "V4_PRODUCER_DRAFT_INDEXER",
    "V4_PRODUCER_DRAFT_MAIN",
    "V4_PRODUCER_TARGET_INDEXER",
    "V4_PRODUCER_TARGET_MAIN",
    "V4_SWA_KV_GROUP_ID",
    "V4_SWA_POOL_ID",
    "build_v4_cache_specs",
    "deepseek_v4_indexer_fp8_layout_from_row_bytes",
    "deepseek_v4_indexer_fp8_row_bytes",
    "deepseek_v4_indexer_fp8_scale_bytes",
    "deepseek_v4_indexer_mxfp4_layout_from_row_bytes",
    "deepseek_v4_indexer_mxfp4_row_bytes",
    "deepseek_v4_indexer_mxfp4_scale_dim",
    "deepseek_v4_indexer_mxfp4_value_bytes",
    "deepseek_v4_nope_dim",
    "deepseek_v4_swa_row_bytes",
    "deepseek_v4_swa_scale_dim",
    "deepseek_v4_swa_token_stride",
    "parse_v4_compressor_state_group_id",
    "v4_compressed_kv_group_id",
    "v4_compressor_state_group_id",
]
