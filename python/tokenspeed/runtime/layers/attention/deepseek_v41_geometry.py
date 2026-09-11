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

"""DeepSeek V4.1 FlatKV geometry; independent of V4's FP8/RoPE layout.

A slot is page * rows_per_page + row, relative to one owner's field view.
All readers must use tensor strides: fields share one physical LCM plane,
not one contiguous allocation per owner. Packed rows contain values followed
by scales (SWA: E4M3/E8M0, global: E2M1/E4M3, index: E2M1/E8M0).
"""

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
V41_SWA_ROW_BYTES = 528
V41_GLOBAL_ROW_BYTES = 288
V41_INDEX_ROW_BYTES = 68
V41_WINDOW_SIZE = 128
V41_TAIL_ROWS = 2

# (physical rows per page, raw tokens per row). Table columns remain absolute
# even when the scheduler releases expired SWA/tail blocks into null holes.
V41_GROUP_GEOMETRY = {
    V41_SWA_GROUP_ID: (DEEPSEEK_V41_SWA_PAGE_SIZE, 1),
    V41_GLOBAL_R2_GROUP_ID: (DEEPSEEK_V41_GLOBAL_ROWS, 2),
    V41_GLOBAL_R1_GROUP_ID: (DEEPSEEK_V41_GLOBAL_ROWS, 1),
    V41_COMPRESSOR_TAIL_GROUP_ID: (V41_TAIL_ROWS, 1),
}
V41_GROUP_PACKING = {
    V41_SWA_GROUP_ID: 1,
    V41_GLOBAL_R2_GROUP_ID: 20,
    V41_GLOBAL_R1_GROUP_ID: 60,
    V41_COMPRESSOR_TAIL_GROUP_ID: 54,
}


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
