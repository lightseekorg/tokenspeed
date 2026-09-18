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

"""DeepSeek V4.1 FlatKV geometry.

A slot is page * rows_per_page + row, relative to one owner's field view.
All readers must use tensor strides: fields share one physical LCM plane,
not one contiguous allocation per owner. Packed rows contain values followed
by scales.

Every row width is a property of the cache format the caller selects through
``V41_CACHE_FORMATS``; this module knows nothing about which hardware wants
which.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from tokenspeed.runtime.layers.attention.kernel_page_sizes import (
    DEEPSEEK_V41_GLOBAL_ROWS,
    DEEPSEEK_V41_SWA_PAGE_SIZE,
)

V41_SWA_GROUP_ID = "v41.swa"
V41_GLOBAL_R2_GROUP_ID = "v41.global_r2"
V41_GLOBAL_R1_GROUP_ID = "v41.global_r1"
V41_COMPRESSOR_TAIL_GROUP_ID = "v41.compressor_tail_r2"
V41_HEAD_DIM = 512
V41_INDEX_HEAD_DIM = 128
V41_WINDOW_SIZE = 128
# Shared caller/recipe bound for native prefill query and output scratch.
V41_PREFILL_QUERY_TILE = 2048
V41_TAIL_ROWS = 2

# (physical rows per page, raw tokens per row). Table columns remain absolute
# even when the scheduler releases expired SWA/tail blocks into null holes.
V41_GROUP_GEOMETRY = {
    V41_SWA_GROUP_ID: (DEEPSEEK_V41_SWA_PAGE_SIZE, 1),
    V41_GLOBAL_R2_GROUP_ID: (DEEPSEEK_V41_GLOBAL_ROWS, 2),
    V41_GLOBAL_R1_GROUP_ID: (DEEPSEEK_V41_GLOBAL_ROWS, 1),
    V41_COMPRESSOR_TAIL_GROUP_ID: (V41_TAIL_ROWS, 1),
}


@dataclass(frozen=True, kw_only=True)
class V41CacheFormat:
    """One row encoding for the cache fields, and what it packs to.

    The row width decides how many of each group's blocks fit a parent, so a
    format owns its packing tables and the plane they produce: these mirror
    ``recipes.plan.pack()``, which only re-derives and checks them. With
    same-checkpoint DSpark the SWA page also carries the draft's context rows,
    so the parent grows and the other groups repack.

    Attributes:
        swa_kernel_format: ``dsv41`` kernel format name for the SWA rows.
        global_kernel_format: ``dsv41`` kernel format name for the global rows.
        index_kernel_format: ``dsv41`` kernel format name for the index rows.
        swa_row_bytes: Packed bytes of one SWA row.
        global_row_bytes: Packed bytes of one global KV row.
        index_row_bytes: Packed bytes of one index-K row.
        group_packing: Blocks per LCM block per group, without DSpark.
        lcm_block_bytes: Parent size ``group_packing`` produces.
        dspark_group_packing: Blocks per LCM block per group, with DSpark.
        dspark_lcm_block_bytes: Parent size ``dspark_group_packing`` produces.
    """

    swa_kernel_format: str
    global_kernel_format: str
    index_kernel_format: str
    swa_row_bytes: int
    global_row_bytes: int
    index_row_bytes: int
    group_packing: Mapping[str, int]
    lcm_block_bytes: int
    dspark_group_packing: Mapping[str, int]
    dspark_lcm_block_bytes: int


# "v41" is the checkpoint's own encoding. "v4" re-encodes the fields in the
# layouts Hopper's kernels can read: the KV rows become the wider V4 rows --
# E4M3 values with E8M0 scales per 64, RoPE left in BF16 -- the only quantized
# layout FlashMLA reads below sm100, and the index rows become DeepGEMM's FP8
# index-K rows so its MQA logits kernels score on tensor cores. It costs
# roughly half the token capacity for the same budget.
V41_CACHE_FORMATS: Mapping[str, V41CacheFormat] = {
    "v41": V41CacheFormat(
        swa_kernel_format="swa",
        global_kernel_format="global",
        index_kernel_format="index",
        swa_row_bytes=528,
        global_row_bytes=288,
        index_row_bytes=68,
        group_packing={
            V41_SWA_GROUP_ID: 1,
            V41_GLOBAL_R2_GROUP_ID: 20,
            V41_GLOBAL_R1_GROUP_ID: 60,
            V41_COMPRESSOR_TAIL_GROUP_ID: 54,
        },
        lcm_block_bytes=1_382_400,
        dspark_group_packing={
            V41_SWA_GROUP_ID: 1,
            V41_GLOBAL_R2_GROUP_ID: 22,
            V41_GLOBAL_R1_GROUP_ID: 66,
            V41_COMPRESSOR_TAIL_GROUP_ID: 62,
        },
        dspark_lcm_block_bytes=1_571_328,
    ),
    "v4": V41CacheFormat(
        swa_kernel_format="swa_v4",
        global_kernel_format="global_v4",
        index_kernel_format="index_v4",
        swa_row_bytes=584,
        global_row_bytes=584,
        index_row_bytes=132,
        group_packing={
            V41_SWA_GROUP_ID: 1,
            V41_GLOBAL_R2_GROUP_ID: 11,
            V41_GLOBAL_R1_GROUP_ID: 33,
            V41_COMPRESSOR_TAIL_GROUP_ID: 60,
        },
        lcm_block_bytes=1_520_640,
        dspark_group_packing={
            V41_SWA_GROUP_ID: 1,
            V41_GLOBAL_R2_GROUP_ID: 12,
            V41_GLOBAL_R1_GROUP_ID: 36,
            V41_COMPRESSOR_TAIL_GROUP_ID: 69,
        },
        dspark_lcm_block_bytes=1_695_744,
    ),
}


def v41_dspark_field_name(stage: int) -> str:
    """Return the SWA-group field name holding one DSpark stage's window rows."""
    if stage < 0:
        raise ValueError("DSpark stage must be non-negative")
    return f"dspark_kv{stage}"


def v41_layer_mapping(
    ratios: tuple[int, ...],
    owners: tuple[int, ...],
    sources: tuple[int, ...],
    candidate_source: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Resolve layer -> KV owner/index source; -1 denotes SWA-only layers.

    Sources and owners must precede their consumers and use the same ratio.
    Reindex sources must share the candidate source's global address domain.
    """
    n = len(ratios)
    for name, layers in (("owners", owners), ("sources", sources)):
        if tuple(sorted(set(layers))) != layers or any(i < 0 or i >= n for i in layers):
            raise ValueError(f"V4.1 {name} must be sorted unique backbone layer IDs")
    if not set(owners).issubset(sources) or candidate_source not in owners:
        raise ValueError("V4.1 owners must index; candidate source must own global KV")
    kv, index = [], []
    for layer, ratio in enumerate(ratios):
        if ratio not in (0, 1, 2):
            raise ValueError("V4.1 compression ratios must be 0, 1, or 2")
        owner = max((i for i in owners if i <= layer), default=-1) if ratio else -1
        source = max((i for i in sources if i <= layer), default=-1) if ratio else -1
        if ratio and (
            owner < 0
            or source < owner
            or ratios[owner] != ratio
            or ratios[source] != ratio
        ):
            raise ValueError(f"V4.1 layer {layer} has incompatible owner/index source")
        if layer in owners + sources and not ratio:
            raise ValueError("SWA-only layers cannot own global KV or index selections")
        if (
            source != owner
            and owner >= 0
            and (owner != candidate_source or source < candidate_source)
        ):
            raise ValueError(
                "Reindex must use the candidate source's KV address domain"
            )
        kv.append(owner)
        index.append(source)
    return tuple(kv), tuple(index)


def v41_table_widths(context_len: int, reservation_tokens: int) -> dict[str, int]:
    """Return absolute table column capacities for a context and reserve horizon.

    Args:
        context_len: Maximum raw-token context length.
        reservation_tokens: Additional raw tokens admitted beyond that length.

    Returns:
        Group id to table width, including the possible final partial page.
    """
    if context_len <= 0 or reservation_tokens < 0:
        raise ValueError(
            "context_len must be positive and reservation_tokens nonnegative"
        )
    return {
        gid: (context_len + reservation_tokens + rows * stride - 1) // (rows * stride)
        for gid, (rows, stride) in V41_GROUP_GEOMETRY.items()
    }
