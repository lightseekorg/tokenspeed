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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""DeepSeek V4 kernel cache geometry and cache-group vocabulary.

Everything here is what the kernels, the model and the pool all have to agree
on: how many bytes a row of each quantized cache costs, and what its cache
group is called. It knows nothing about recipes, budgets or the scheduler --
:mod:`tokenspeed.runtime.layers.attention.kv_cache.recipes.deepseek_v4` builds
the group specs on top of it, and depends on this module rather than the other
way round.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from tokenspeed.runtime.configs.deepseek_v4_config import get_deepseek_v4_compress_ratio
from tokenspeed.runtime.layers.attention.kernel_page_sizes import (
    DEEPSEEK_V4_PAGE_SIZE,
)

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
# Per compression ratio: how long the compressor tail must be retained, and how
# many rows of it share one cache block. Both the kernel cache layout and the
# recipe's group specs read these, so the tables live here once.
V4_COMPRESSOR_STATE_WINDOW_TOKENS = {4: 8, 128: 128}
V4_COMPRESSOR_STATE_ROWS_PER_PAGE = {4: 4, 128: 8}


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


def v4_compressed_rows_per_page(ratio: int) -> int:
    """Kernel rows one compressed-KV cache block holds.

    Sources from the kernel-page registry constant, never from the scheduler
    prefix granularity.
    """
    if ratio <= 1:
        raise ValueError(f"ratio must be > 1, got {ratio}")
    return max(1, DEEPSEEK_V4_PAGE_SIZE // ratio)


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
            return v4_compressed_rows_per_page(compress_ratio)
        return self.page_size

    def compressor_state_block_size(self, compress_ratio: int) -> int:
        return V4_COMPRESSOR_STATE_ROWS_PER_PAGE.get(compress_ratio, self.page_size)

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
    num_hidden_layers = len(hf_config.layer_types)
    if layer_indices is None:
        layer_indices = range(num_hidden_layers)
    else:
        layer_indices = tuple(layer_indices)
        if any(idx < 0 for idx in layer_indices):
            raise ValueError(
                "DeepSeek V4 cache layout layer index out of range: "
                f"indices={layer_indices}, layers={num_hidden_layers}"
            )
    raw_layer_ratios = tuple(
        get_deepseek_v4_compress_ratio(hf_config, idx) for idx in layer_indices
    )

    return DeepseekV4CacheLayout(
        layer_ratio=raw_layer_ratios,
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
