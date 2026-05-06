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
_COMPRESSOR_STATE_WINDOW_TOKENS = {4: 8, 128: 128}
_COMPRESSOR_STATE_ROWS_PER_PAGE = {4: 4, 128: 8}
_COMPRESSED_LOGICAL_BLOCK_SIZE = 256


def _compressed_kernel_block_size(ratio: int) -> int:
    if ratio <= 1:
        raise ValueError(f"ratio must be > 1, got {ratio}")
    return max(1, _COMPRESSED_LOGICAL_BLOCK_SIZE // ratio)


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
        PagedCacheGroupSpec(
            group_id="v4.swa",
            retention="sliding_window",
            rows_per_page=V4_KERNEL_BLOCK_ROWS,
            entry_stride_tokens=1,
            sliding_window_tokens=swa_window,
        ),
    ]
    for ratio in unique_compress_ratios:
        if ratio not in _COMPRESSOR_STATE_WINDOW_TOKENS:
            raise ValueError(f"unsupported DeepSeek V4 compress_ratio={ratio}")
        specs.append(
            PagedCacheGroupSpec(
                group_id=f"v4.compressor_state.{ratio}",
                retention="sliding_window",
                rows_per_page=_COMPRESSOR_STATE_ROWS_PER_PAGE[ratio],
                entry_stride_tokens=1,
                sliding_window_tokens=_COMPRESSOR_STATE_WINDOW_TOKENS[ratio],
            )
        )
        specs.append(
            PagedCacheGroupSpec(
                group_id=f"compressed.{ratio}",
                retention="full_history",
                rows_per_page=_compressed_kernel_block_size(ratio),
                entry_stride_tokens=ratio,
                sliding_window_tokens=None,
            )
        )
    if 4 in unique_compress_ratios:
        specs.append(
            PagedCacheGroupSpec(
                group_id="v4.indexer_state.4",
                retention="sliding_window",
                rows_per_page=_COMPRESSOR_STATE_ROWS_PER_PAGE[4],
                entry_stride_tokens=1,
                sliding_window_tokens=_COMPRESSOR_STATE_WINDOW_TOKENS[4],
            )
        )
    return specs


__all__ = [
    "V4_KERNEL_BLOCK_ROWS",
    "build_v4_cache_specs",
]
