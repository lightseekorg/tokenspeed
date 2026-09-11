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

"""Portable DeepSeek V4.1 cache codecs and bounded selected attention.

Rows store value bytes followed by scale bytes (even FP4 element in the low
nibble). Formats: ``global`` = 512D E2M1/E4M3 groups of 16, 288 bytes;
``index`` = 128D E2M1/E8M0 groups of 32, 68 bytes; ``swa`` = 512D
E4M3/E8M0 groups of 32, 528 bytes. The entire post-RoPE vector is quantized.
These layouts are NOT compatible with dsv4_decode.

Paged fields are uint8 [pages, 64, row_bytes] views; their page, row and byte
strides and storage offsets are honored. Slots address that particular field,
not a scheduler block or another owner. Invalid slots read zero / skip writes.
Callers own page mapping, causal lengths, RoPE, normalization and output buffers.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.selection import SelectionObjective, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "cache_pack",
    "cache_unpack",
    "cache_scatter",
    "cache_gather",
    "index_q_quantize",
    "selected_attention",
    "index_score",
    "index_topk",
]


def _kernel(mode: str, x: torch.Tensor):
    return select_kernel(
        "attention",
        "dsv41_" + mode,
        format_signature(x=dense_tensor_format(x.dtype)),
        features=None,
        platform=None,
        objective=SelectionObjective.DEFAULT,
        traits=None,
        solution=None,
        override=None,
    )


def cache_pack(
    rows: torch.Tensor, cache_format: str, out: torch.Tensor | None
) -> torch.Tensor:
    """Quantize BF16/FP32 ``rows[N, D]`` into packed uint8 rows.

    ``cache_format`` is 'swa', 'global', or 'index' (see module layouts).
    ``out`` is a contiguous uint8 [N, row_bytes] destination, or explicitly
    None to allocate. Returns that destination. Finite inputs in the encoding's
    scale range are required; this is activation, not checkpoint quantization.
    """
    return _kernel("cache_pack", rows)(rows, cache_format, out)


def cache_unpack(
    packed: torch.Tensor, cache_format: str, out: torch.Tensor | None
) -> torch.Tensor:
    """Decode uint8 ``packed[N, row_bytes]`` of ``cache_format``.

    ``out`` is contiguous BF16/FP32 [N, D], or None to allocate BF16. Returns
    the destination, with dequantized values rounded to its dtype. Packed input
    may be strided; scale bytes retain their format-specific interpretation.
    """
    return _kernel("cache_unpack", packed)(packed, cache_format, out)


def cache_scatter(
    rows: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
    cache_format: str,
) -> None:
    """Quantize BF16/FP32 ``rows[N, D]`` into a strided paged ``cache`` field.

    ``slots[N]`` are int32/int64 physical row slots. Negative/out-of-capacity
    slots skip writes; valid slots must be unique. ``cache_format`` selects
    the module layout. Mutates cache only, returns None. Input rows and slots
    may be strided. No page-table mapping or model preprocessing is performed.
    """
    return _kernel("cache_scatter", rows)(rows, cache, slots, cache_format)


def cache_gather(
    cache: torch.Tensor,
    slots: torch.Tensor,
    cache_format: str,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """Read/dequantize selected slots from a strided paged uint8 ``cache``.

    ``slots`` is an arbitrary-shaped int32/int64 tensor of physical row slots;
    invalid slots return zero without reading cache. ``cache_format`` selects
    the module layout. ``out`` is contiguous BF16/FP32 [*slots.shape, D], or
    None to allocate BF16. Returns that destination, never the full history.
    """
    return _kernel("cache_gather", cache)(cache, slots, cache_format, out)


def index_q_quantize(q: torch.Tensor, out: torch.Tensor | None) -> torch.Tensor:
    """FP4/E8M0-32 quantize/dequantize BF16 index ``q[T, H, 128]``.

    ``out`` is a contiguous BF16 destination shaped like q, or None to allocate.
    Returns the dequantized destination, matching the reference's inplace
    simulation. No RoPE, RMSNorm or Hadamard is applied. Main attention queries
    must NOT pass through this helper.
    """
    return _kernel("index_q_quantize", q)(q, out)


def selected_attention(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_slots: torch.Tensor,
    swa_lens: torch.Tensor,
    global_cache: torch.Tensor | None,
    global_slots: torch.Tensor | None,
    global_lens: torch.Tensor | None,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None,
    query_chunk_size: int,
) -> torch.Tensor:
    """Joint SWA/global softmax with one zero-valued sink, returning BF16.

    Args:
        q: Post-RoPE BF16 [T, H, 512]; used unchanged, without an added Q norm.
        swa_cache: This layer's strided uint8 [pages, 64, 528] SWA field.
        swa_slots: Int32/int64 [T, W_swa] physical slots, normally W_swa=128.
        swa_lens: Int32/int64 [T] active prefix lengths; negative slots within
            each prefix are also ignored. Caller supplies causal selections.
        global_cache: Owner's uint8 [pages, 64, 288] field, or None for SWA-only.
        global_slots: Int32/int64 [T, W_global], normally W_global=512; None
            iff global_cache is None. This is a separate physical address domain.
        global_lens: Int32/int64 [T] active prefix lengths, or None with no global.
        attn_sink: FP32 [H] finite sink logits, included exactly once per head.
        softmax_scale: QK multiplier; reference uses 512**-0.5.
        out: Contiguous BF16 destination shaped like q, or None to allocate.
        query_chunk_size: Positive query tile bound for gather/dequant scratch.

    Returns:
        Output shaped like q; all-invalid selections return zero. SWA/global
        entries are not deduplicated. Inverse RoPE remains the caller's job.

    Correctness baseline: gathers at most query_chunk_size * (W_swa + W_global)
    BF16 rows, then reuses dsv4_prefill's FP32 online softmax. No full-history
    logits or whole-prefill selected workspace. Not a production latency claim.
    """
    return _kernel("selected_attention", q)(
        q,
        swa_cache,
        swa_slots,
        swa_lens,
        global_cache,
        global_slots,
        global_lens,
        attn_sink,
        softmax_scale,
        out,
        query_chunk_size,
    )


def index_score(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    index_cache: torch.Tensor,
    physical_slots: torch.Tensor,
    process_group: torch.distributed.ProcessGroup | None,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """Score one caller-bounded tile of selected index-cache rows.

    ``index_q[T, H, 128]`` is post-RoPE BF16, quantized internally to FP4/E8M0.
    ``weights[T, H]`` is BF16/FP32, already multiplied by
    128**-0.5 * total_index_heads**-0.5. ``index_cache`` is the strided uint8
    [pages, 64, 68] field; ``physical_slots[T, rows]`` selects only rows to read.
    ``process_group`` reduces head contributions before returning scores, or
    None for unsharded heads. Every rank must use the same slots/tile schedule.
    ``out`` is contiguous [T, rows] with weights' dtype, or None to allocate.

    Returns sum_h(weights_h * relu(BF16 dot(q_h, k))), with reference rounding
    at dot, multiply and head-sum boundaries. Invalid slots score -inf. For a
    history scan use index_topk instead of materializing query*history scores.
    """
    return _kernel("index_score", index_q)(
        index_q,
        weights,
        index_cache,
        physical_slots,
        process_group,
        out,
    )


def index_topk(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    index_cache: torch.Tensor,
    page_table: torch.Tensor,
    visible_lens: torch.Tensor,
    candidate_blocks: torch.Tensor | None,
    topk: int,
    candidate_topk: int,
    candidate_block_size: int,
    query_chunk_size: int,
    score_chunk_size: int,
    process_group: torch.distributed.ProcessGroup | None,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bounded Full/Reindex scoring, Top-K rows, and optional source candidates.

    Args:
        index_q: Post-RoPE BF16 [T, H, 128], quantized internally (index Q only).
        weights: BF16/FP32 [T, H], scaled as documented by index_score.
        index_cache: Owner's strided uint8 [pages, 64, 68] index field.
        page_table: Int32/int64 [T, logical_pages], a request table per query.
            Entries address this index field, not the main-KV field. Invalid
            pages are ignored; tables may be expanded/noncontiguous views.
        visible_lens: Int32/int64 [T] causal row counts: floor((position+1)/ratio).
        candidate_blocks: None for Full; otherwise int32/int64 [T, <=2048]
            unique request-local block IDs, padded with -1. Reindex reads and
            scores ONLY these blocks, reapplying visible_lens to their rows.
        topk: Row selection capacity in [1, 512]; V4.1 uses 512.
        candidate_topk: Source block capacity in [0, 2048], 2048 for candidate-source Full,
            0 for ordinary Full and Reindex. Not derived from selected Top512.
        candidate_block_size: Must be 8; each block score is its visible row max.
        query_chunk_size: Positive bound on simultaneously processed queries.
        score_chunk_size: Positive multiple of 8, bounding scored rows per tile.
        process_group: TP head-sum group or None; see index_score's collective
            contract. Reduction happens before either Top-K selection.
        out: Four contiguous int32 destinations ([T, topk], [T],
            [T, candidate_topk], [T]), or None to allocate them.

    Returns:
        (logical_row_ids, row_lengths, candidate_block_ids, candidate_lengths).
        IDs are position-sorted, packed into valid prefixes, then padded -1.
        Blocks include the latest visible block even if its score is low.
        Empty history has zero lengths. Equal-score boundary ties may select
        any equivalent subset; no cross-chunk/TP bitwise tie guarantee.

    Full scans the table capacity in bounded tiles; Reindex scans at most
    2048*8 candidates. Scratch is bounded by the explicit tile sizes and Top-K
    capacities, not T*history. This torch Top-K merge baseline has a launch/
    bandwidth ceiling; fused streaming selection can replace the same operation.
    """
    return _kernel("index_topk", index_q)(
        index_q,
        weights,
        index_cache,
        page_table,
        visible_lens,
        candidate_blocks,
        topk,
        candidate_topk,
        candidate_block_size,
        query_chunk_size,
        score_chunk_size,
        process_group,
        out,
    )
