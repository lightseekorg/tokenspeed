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

from typing import Any, List, Sequence

from tokenspeed.runtime.configs.paged_cache_spec import PagedCacheGroupSpec

V4_KERNEL_BLOCK_ROWS: int = 64
V4_SWA_KV_GROUP_ID = "v4.swa_kv"
V4_INDEXER_COMPRESSOR_STATE_GROUP_ID = "v4.c4a.indexer_compressor_state"
DEEPSEEK_V4_FP8_MAX = 448.0
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
) -> List[PagedCacheGroupSpec]:
    swa_window = _resolve_sliding_window(hf_config)
    unique_compress_ratios = sorted({int(r) for r in layer_ratio if int(r) > 1})

    specs: List[PagedCacheGroupSpec] = [
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
                sliding_window_tokens=_COMPRESSOR_STATE_WINDOW_TOKENS[ratio],
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
                sliding_window_tokens=_COMPRESSOR_STATE_WINDOW_TOKENS[4],
                family="state",
            )
        )
    return specs


__all__ = [
    "DEEPSEEK_V4_COMPRESSED_LOGICAL_BLOCK_SIZE",
    "DEEPSEEK_V4_FP8_MAX",
    "DEEPSEEK_V4_FP8_INDEXER_BLOCK_SIZE",
    "DEEPSEEK_V4_FP8_QUANT_BLOCK",
    "DEEPSEEK_V4_FP8_SCALE_BYTES",
    "DEEPSEEK_V4_MXFP4_BLOCK_SIZE",
    "DEEPSEEK_V4_MXFP4_SCALE_BYTES",
    "DEEPSEEK_V4_SPARSE_PREFILL_TOPK_ALIGNMENT",
    "V4_INDEXER_COMPRESSOR_STATE_GROUP_ID",
    "V4_KERNEL_BLOCK_ROWS",
    "V4_SWA_KV_GROUP_ID",
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
