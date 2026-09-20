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

"""DeepSeek V4.1 cache codecs, sparse attention and hierarchical selection.

Importing this family registers its Triton, AMD Gluon, and optional native
solutions.

Standalone packed rows place value bytes before scales (even FP4 element in
low nibble). Formats: global = 512D E2M1/E4M3 groups of16, 288 bytes;
index = 128D E2M1/E8M0 groups of32, 68 bytes; SWA = 512D E4M3/E8M0
 groups of32, 528 bytes -- the entire post-RoPE vector is quantized. The
swa_v4/global_v4 rows are the same 512D content in FlashMLA's V4 layout: a
576-byte data row of 448 E4M3 NoPE bytes then the 64 RoPE dims left as BF16,
and an 8-byte scale row of E8M0 per64 with a pad byte. They exist for targets
whose FlashMLA reads V4 only.

Paged fields use page-planar storage: all64 value rows, then all64 scale rows.
Their uint8 [pages,64,row_bytes] shape describes storage, not decoded row axes.
Portable codecs honor every byte stride; native kernels require contiguous
page bytes and aligned page strides. Physical slots address a particular field;
invalid slots read zero or skip writes. Callers own mapping, causal lengths,
RoPE, normalization, output buffers and cache lifetime.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.mla._triton.page_table import bounded_group_slots
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "decode_rows",
    "decode_window",
    "global_slots",
    "selection_table",
    "cache_pack",
    "cache_unpack",
    "cache_scatter",
    "cache_gather",
    "index_q_quantize",
    "selected_attention",
    "new_attention_schedule",
    "index_score",
    "index_topk",
    "compressor_tail_scatter",
    "compressor_pool",
    "compressor_metadata",
    "swa_rope_scatter",
    "rope_inplace",
    "rope_pad_query",
]


def _kernel(mode: str, x: torch.Tensor):
    return select_kernel(
        "attention",
        "dsv41_" + mode,
        format_signature(x=dense_tensor_format(x.dtype)),
        features=None,
        platform=None,
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


def swa_rope_scatter(
    values: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
    cache_format: str,
    out: torch.Tensor | None,
) -> torch.Tensor | None:
    """Rotate normalized BF16 SWA rows and quantize directly into planar pages.

    values is [T,512], positions and physical slots are integer [T], and the
    FP32 RoPE table is [max_position,64]. cache is the LCM uint8 [pages,64,R]
    field with contiguous page bytes, where R is the row width ``cache_format``
    declares. Invalid/null slots skip writes. If out is supplied, write all
    quantized/dequantized BF16 [T,512] rows there, including rows outside
    persistent retention. Read old prefixes before this operation.
    """
    _kernel("swa_rope_scatter", values)(
        values, positions, cos_sin_cache, cache, slots, cache_format, out
    )
    return out


def compressor_pool(
    content: torch.Tensor,
    scores: torch.Tensor,
    previous: torch.Tensor,
    tail: torch.Tensor,
    tail_slots: torch.Tensor,
    active: torch.Tensor,
    out: torch.Tensor | None,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
) -> torch.Tensor:
    """Pool completed compressor pairs into FP32 [T,512] without cache writes.

    content/scores are FP32 current projections; previous[T] names each pair's
    predecessor input row, or -1 to read tail[tail_slots[T]]. tail is the LCM
    FP32 [pages,2,2,512] field. active[T] masks incomplete/padded pairs to zero.
    All arguments share a device; out is contiguous FP32 [T,512], or None to
    allocate. With norm_weight[512], preserve the BF16 pooled boundary and
    return BF16 RMS-normalized rows using norm_eps; None returns raw FP32.
    The caller validates dependencies
    and publishes tail writes only after all pooled reads are enqueued.
    """
    return _kernel("compressor_pool", content)(
        content, scores, previous, tail, tail_slots, active, out, norm_weight, norm_eps
    )


def compressor_tail_scatter(
    content: torch.Tensor,
    scores: torch.Tensor,
    tail: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    """Store FP32 projected compressor inputs without compacting live rows.

    ``content`` and ``scores`` are FP32 [T, 512]; ``tail`` is a strided FP32
    [pages, 2, 2, 512] field (page, token, content/score, channel). ``slots[T]``
    are int32/int64 field-relative token slots. Negative/out-of-capacity slots
    skip writes; valid slots must be unique. Page zero is not special to this
    op: the caller maps null pages to -1. All input strides are honored.
    Mutates tail and returns None. Read prior pairs BEFORE this write, since
    scheduler-reused pages may still hold inputs needed by the current chunk.
    """
    return _kernel("compressor_tail_scatter", content)(content, scores, tail, slots)


def index_q_quantize(
    q: torch.Tensor, cache_format: str, out: torch.Tensor | None
) -> torch.Tensor:
    """Quantize/dequantize BF16 index ``q[T, H, 128]`` as ``cache_format`` does.

    ``cache_format`` is 'index' (FP4 with E8M0-32 scales) or 'index_v4' (E4M3
    with one FP32 scale), matching the keys the queries will be scored against.
    ``out`` is a contiguous BF16 destination shaped like q, or None to allocate.
    Returns the dequantized destination, matching the reference's inplace
    simulation. No RoPE, RMSNorm or Hadamard is applied. Main attention queries
    must NOT pass through this helper.
    """
    return _kernel("index_q_quantize", q)(q, cache_format, out)


def new_attention_schedule() -> object | None:
    """Create an uninitialized FlashMLA schedule when the optional API is available.

    The caller retains it through capture/replay; no request state or global
    tensor cache is allocated here. Portable selected attention accepts None.
    """
    from tokenspeed_kernel.ops.attention.dsv41.flash_mla import (
        is_flash_mla_v41_available,
    )

    if not is_flash_mla_v41_available():
        return None
    from tokenspeed_kernel.ops.attention.dsv41.flash_mla import new_flashmla_schedule

    return new_flashmla_schedule()


def selected_attention(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_slots: torch.Tensor | None,
    swa_lens: torch.Tensor | None,
    global_cache: torch.Tensor | None,
    global_slots: torch.Tensor | None,
    global_lens: torch.Tensor | None,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None,
    query_chunk_size: int,
    schedule: object | None,
    prefill_kv: torch.Tensor | None,
    prefill_indices: torch.Tensor | None,
) -> torch.Tensor:
    """Joint SWA/global softmax with one zero-valued sink, returning BF16.

    Args:
        q: Post-RoPE BF16 [T, H, 512]; used unchanged, without an added Q norm.
        swa_cache: This layer's strided uint8 [pages, 64, R] SWA field.
        swa_slots: Int32/int64 [T, W_swa] physical slots, normally W_swa=128.
        swa_lens: Int32/int64 [T] active prefix lengths; negative slots within
            each prefix are also ignored. Caller supplies causal selections.
        global_cache: Owner's uint8 [pages, 64, R] field, or None for SWA-only.
        global_slots: Int32/int64 [T, W_global], normally W_global=512; None
            iff global_cache is None. This is a separate physical address domain.
        global_lens: Int32/int64 [T] active prefix lengths, or None with no global.
        attn_sink: FP32 [H] sink logits, or a pre-padded 64/128-head vector;
            each real head includes its sink exactly once.
        softmax_scale: QK multiplier; reference uses 512**-0.5.
        out: Contiguous BF16 destination shaped like q, or None to allocate.
        query_chunk_size: Positive query tile bound for gather/dequant scratch.
        schedule: Fresh caller-owned native decode schedule, or None for prefill.
            Eager refresh replaces it; graph capture records its initialization
            and replay rebuilds metadata from the live lengths.
        prefill_kv: Optional BF16 [rows,1,512] forward-owned compact workspace.
        prefill_indices: Optional int32 [T,width] indices into prefill_kv.
            With a workspace, paged slots/lengths are None. The backend gathers
            all required old prefix rows before publishing this layer's writes.

    Returns:
        Output shaped like q; all-invalid selections return zero. SWA/global
        entries are not deduplicated. Inverse RoPE remains the caller's job.

    The optional Blackwell implementation reads packed pages directly for
    decode, and tiles compact BF16 prefill workspaces. The portable implementation
    uses the same selections and bounded gather with FP32 online softmax.
    """
    from tokenspeed_kernel.ops.attention.dsv41.flash_mla import (
        is_flash_mla_v41_available,
    )

    native = (
        schedule is not None or prefill_kv is not None
    ) and is_flash_mla_v41_available()
    kernel = select_kernel(
        "attention",
        "dsv41_selected_attention",
        format_signature(x=dense_tensor_format(q.dtype)),
        features=None,
        platform=None,
        traits={"flashmla_eligible": native},
        solution=None,
        override=None,
    )
    return kernel(
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
        schedule,
        prefill_kv,
        prefill_indices,
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
    ``process_group`` gathers Q/weights across TP heads, or None for unsharded
    heads. Every rank must use the same query/slot schedule. Each shard's head
    sum rounds to weights' dtype; shard sums accumulate in rank-order FP32,
    then round once to weights' dtype. This keeps the reference's dot/product/
    shard-sum boundaries, not NCCL's unspecified final reduction order.
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
    solution: str | None,
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
        process_group: TP group or None; see index_score's numerical contract.
            One packed Q/weights all-gather per query tile, no score-tile
            collectives. All head contributions are combined before selection.
        out: Four contiguous int32 destinations ([T, topk], [T],
            [T, candidate_topk], [T]), or None to allocate them.
        solution: Optional registered implementation restriction. Pass None for
            automatic kernel selection.

    Returns:
        (logical_row_ids, row_lengths, candidate_block_ids, candidate_lengths).
        IDs are position-sorted, packed into valid prefixes, then padded -1.
        Blocks include the latest visible block with valid cache rows even if
        its score is low. Missing or out-of-range pages never become candidates.
        Empty history has zero lengths. Equal-score boundary ties may select
        any equivalent subset; no cross-chunk/TP bitwise tie guarantee.

    The portable Full implementation uses16 fixed partitions bounded by device-visible
    lengths, not table capacity; Reindex scores at most 2048*8 candidate rows.
    Graph replay recomputes those loop bounds without host synchronization.
    Each partition keeps exact row/block TopK, merged once across partitions.
    Scratch is O(query_chunk_size * 16 * (rounded_topk + rounded_candidate_topk))
    plus gathered Q/weights, independent of configured or visible history.
    FP32 scores and int64 IDs use12 bytes per partial entry; capacities round
    to powers of two and at least the score tile width (at most256 lanes).
    With replicated32 heads and compatible Blackwell pages, optional DeepGEMM
    scores packed FP4 inputs with FP32 accumulation and DeepSelect selects rows
    and block maxima. This native accumulation differs from the portable
    reference's intermediate BF16 rounding. Native query tiles cap logits at
    32MiB for paged queries and 128MiB for a zero-row-stride request table.
    That broadcast layout gathers packed history once per call and packs Q
    once before the score tiles; no payload survives the call. For unsharded
    68-byte MXFP4 caches, AMD Gluon Full similarly caps each FP32 logits query
    tile at 32MiB instead of limiting history width; Reindex scores only the
    candidate rows.
    """
    from tokenspeed_kernel.ops.attention.dsv41.deep_gemm import (
        is_hopper_indexer_available,
        is_native_indexer_available,
    )
    from tokenspeed_kernel.thirdparty.deep_select import is_deep_select_available

    # Each target scores the index rows its own cache format stores: FP4 with
    # DeepSelect on Blackwell, FP8 with FlashInfer's selection on Hopper.
    row_bytes = index_cache.shape[2] if index_cache.ndim == 3 else 0
    shaped = (
        index_q.is_cuda
        and process_group is None
        and index_q.shape[1] == 32
        and index_cache.ndim == 3
        and index_cache.shape[1] == 64
        and index_cache.stride(1) == row_bytes
        and index_cache.stride(2) == 1
        and index_cache.stride(0) < 2**31
        and index_cache.stride(0) % 16 == 0
        and index_cache.data_ptr() % 16 == 0
    )
    native = shaped and (
        (
            row_bytes == 68
            and is_native_indexer_available()
            and is_deep_select_available()
        )
        or (row_bytes == 132 and is_hopper_indexer_available())
    )
    index_k_format = {
        68: "mxfp4",
        132: "fp8_scaled",
    }.get(row_bytes, "unknown")
    index_shards = 1
    index_heads = index_q.shape[1]
    if process_group is not None:
        index_shards = torch.distributed.get_world_size(process_group)
        index_heads *= index_shards
    kernel = select_kernel(
        "attention",
        "dsv41_index_topk",
        format_signature(x=dense_tensor_format(index_q.dtype)),
        features=None,
        platform=None,
        traits={
            "native_indexer": native,
            "index_heads": index_heads,
            "index_k_format": index_k_format,
            "index_shards": index_shards,
        },
        solution=solution,
        override=None,
    )
    return kernel(
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


def _validate_rope(values, positions, cos_sin_cache):
    if values.ndim not in (2, 3) or values.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            "V4.1 RoPE values must be BF16/FP16 [tokens,dim] or [tokens,heads,dim]"
        )
    if not values.is_cuda or values.stride(-1) != 1:
        raise ValueError("V4.1 RoPE requires CUDA values with contiguous head channels")
    if positions.shape != values.shape[:1] or positions.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("RoPE positions must be int32/int64 [tokens]")
    if (
        cos_sin_cache.ndim != 2
        or cos_sin_cache.dtype != torch.float32
        or cos_sin_cache.shape[0] <= 0
        or cos_sin_cache.shape[1] <= 0
        or cos_sin_cache.shape[1] % 2
        or cos_sin_cache.shape[1] > values.shape[-1]
        or cos_sin_cache.stride(-1) != 1
    ):
        raise ValueError(
            "RoPE table must be FP32 [positions,even rotary_dim] within the head width"
        )
    if positions.device != values.device or cos_sin_cache.device != values.device:
        raise ValueError("RoPE tensors must share a device")


def rope_inplace(
    values: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    solution: str | None,
) -> torch.Tensor:
    """Rotate the interleaved tail in-place, preserving non-RoPE channels.

    Args:
        values: BF16/FP16 [tokens,dim] or [tokens,heads,dim]. Writable views must
            not overlap; noncontiguous outer strides are supported. Expanded
            zero-stride rows/heads with more than one element are rejected.
        positions: Int32/int64 token positions; negative padding positions use row0.
        cos_sin_cache: FP32 [max_positions,rotary_dim], cosine then sine halves.
            Caller supplies forward or inverse sine signs. Positive out-of-range
            positions assert on device before invalid table reads.
        solution: Optional registered implementation restriction.

    Returns:
        The same values tensor. NoPE channels remain unchanged; FP32 products
        are separately rounded before the final low-precision store.
    """
    _validate_rope(values, positions, cos_sin_cache)
    if any(
        size > 1 and stride == 0
        for size, stride in zip(values.shape[:-1], values.stride()[:-1])
    ):
        raise ValueError(
            "In-place V4.1 RoPE requires non-overlapping writable rows/heads"
        )
    kernel = select_kernel(
        "attention",
        "dsv41_rope_inplace",
        format_signature(values=dense_tensor_format(values.dtype)),
        traits={},
        solution=solution,
    )
    return kernel(values, positions, cos_sin_cache)


def rope_pad_query(
    values: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    solution: str | None,
) -> torch.Tensor:
    """Rotate compact queries while fully writing the FlashMLA 64/128-head input.

    Args:
        values: BF16/FP16 [tokens,real_heads,512], 1..128 heads, strided rows/heads.
        positions: Int32/int64 [tokens], with negative padding positions mapped to0.
        cos_sin_cache: FP32 cosine/sine table for the interleaved head tail.
        solution: Optional registered implementation restriction.

    Returns:
        Fresh contiguous [tokens,64 or128,512] queries. Every padded head is zero;
        input storage is unchanged. This is forward/graph-owned scratch, never
        persistent KV or a request-indexed state buffer.
    """
    _validate_rope(values, positions, cos_sin_cache)
    if values.ndim != 3 or values.shape[-1] != 512 or not 1 <= values.shape[1] <= 128:
        raise ValueError("Padded V4.1 queries require [tokens,1..128 heads,512]")
    kernel = select_kernel(
        "attention",
        "dsv41_rope_pad_query",
        format_signature(values=dense_tensor_format(values.dtype)),
        traits={},
        solution=solution,
    )
    return kernel(values, positions, cos_sin_cache)


def decode_rows(
    seq_lens,
    request_pool_indices,
    positions,
    requests,
    lengths,
    pool_rows,
    actual_bs,
    num_extends,
    width,
):
    """Refresh request-major decode/verify metadata in place; return None.

    Contiguous integer inputs seq_lens/request_pool_indices have >= actual_bs
    entries. Outputs positions/requests are [bs * width], lengths/pool_rows
    are [bs]. Requests are batch-table rows, never request-pool indices.
    width is a positive token count; live lengths clamp to at least width.
    All outputs are overwritten. Padding and leading num_extends requests
    have token coordinates -1; padding also has length 0 and pool row -1.
    """
    bs = lengths.numel()
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise ValueError("decode width must be a positive integer")
    if not 0 <= num_extends <= actual_bs <= bs:
        raise ValueError("invalid live/decode batch sizes")
    if (
        positions.shape != (bs * width,)
        or requests.shape != positions.shape
        or lengths.shape != (bs,)
        or pool_rows.shape != (bs,)
        or seq_lens.ndim != 1
        or request_pool_indices.ndim != 1
        or min(seq_lens.numel(), request_pool_indices.numel()) < actual_bs
    ):
        raise ValueError(
            "decode metadata requires request-sized and packed token vectors"
        )
    tensors = (seq_lens, request_pool_indices, positions, requests, lengths, pool_rows)
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("decode row vectors must be contiguous")
    if any(tensor.device != positions.device for tensor in tensors):
        raise ValueError("decode row vectors must share a device")
    if not positions.is_cuda:
        lengths.zero_()
        lengths[:actual_bs].copy_(seq_lens[:actual_bs].clamp_min(width))
        positions.view(bs, width).copy_(
            lengths.to(torch.int64)[:, None] - width + torch.arange(width)
        )
        requests.view(bs, width).copy_(torch.arange(bs)[:, None])
        for tensor in (positions, requests):
            tensor[: num_extends * width].fill_(-1)
            tensor[actual_bs * width :].fill_(-1)
        pool_rows.fill_(-1)
        pool_rows[:actual_bs].copy_(request_pool_indices[:actual_bs])
        return
    _kernel("decode_rows", positions)(
        seq_lens,
        request_pool_indices,
        positions,
        requests,
        lengths,
        pool_rows,
        actual_bs,
        num_extends,
        width,
    )


def decode_window(
    positions,
    requests,
    write_slots,
    read_slots,
    read_lens,
    status,
    swa_table,
    tail_table,
    swa_pages,
    tail_pages,
):
    """Fill SWA addresses and history errors for N logical query coordinates.

    positions/requests are integer [N] in consecutive request-major spans;
    spans may have different lengths. Tables are LCM group pages. Destinations
    are write_slots[N], read_slots[N,128], read_lens[N], status[N]. Only the
    first live row of each span reports history errors: bit 0 for missing SWA
    history, bit 1 for a required compressor tail. Internal pairs use current
    projections, not retained tails. Padding status is zero. Page zero and
    pages beyond the supplied capacities never address live cache. Returns None.
    """
    if not positions.is_cuda:
        from tokenspeed_kernel.ops.attention.mla._triton.page_table import (
            bounded_group_slots,
        )

        wanted = (positions - 127).clamp_min(0)[:, None] + torch.arange(128)
        wanted.masked_fill_(wanted > positions[:, None], -1)
        slots = bounded_group_slots(
            wanted, requests[:, None].expand_as(wanted), swa_table, 64, 1, 1, swa_pages
        )
        read_slots.copy_(slots)
        read_lens.copy_((positions + 1).clamp(0, 128))
        write_slots.copy_(
            bounded_group_slots(positions, requests, swa_table, 64, 1, 1, swa_pages)
        )
        missing_swa = ((wanted >= 0) & (wanted < positions[:, None]) & (slots < 0)).any(
            -1
        )
        previous = (positions - 1).masked_fill(
            (positions < 0) | (positions % 2 != 1), -1
        )
        tail = bounded_group_slots(previous, requests, tail_table, 2, 1, 1, tail_pages)
        window_start = (positions >= 0) & (requests >= 0)
        window_start[1:] &= (requests[1:] != requests[:-1]) | (
            positions[1:] != positions[:-1] + 1
        )
        status.copy_(
            (
                missing_swa.to(torch.int32)
                | (((previous >= 0) & (tail < 0)).to(torch.int32) * 2)
            ).masked_fill(~window_start, 0)
        )
        return
    _kernel("decode_window", positions)(
        positions,
        requests,
        write_slots,
        read_slots,
        read_lens,
        status,
        swa_table,
        tail_table,
        swa_pages,
        tail_pages,
    )


def global_slots(rows, positions, requests, table, ratio, pages):
    """Map integer selected rows[N,K] into int32 physical slots[N,K].

    Query coordinates [N] determine causality, ratio is raw tokens per global
    row, and pages bounds the LCM field. Invalid/null addresses resolve to -1.
    """
    if not positions.is_cuda:
        from tokenspeed_kernel.ops.attention.mla._triton.page_table import (
            bounded_group_slots,
        )

        visible = (positions[:, None] + 1).clamp_min(0) // ratio
        selected = (rows.to(torch.int64) * ratio).masked_fill(
            (rows < 0) | (rows >= visible), -1
        )
        return bounded_group_slots(
            selected, requests[:, None].expand_as(selected), table, 64, ratio, 1, pages
        ).to(torch.int32)
    return _kernel("global_slots", positions)(
        rows, positions, requests, table, ratio, pages
    )


def selection_table(positions, requests, table, ratio):
    """Gather indexer page rows and visible lengths for logical queries [N].

    Returns int32 table[N,table_width] with null/invalid request entries -1,
    and int32 visible lengths[N]. ratio is raw tokens per global cache row.
    """
    if not positions.is_cuda:
        out = torch.full((positions.numel(), table.shape[1]), -1, dtype=torch.int32)
        if table.numel():
            values = table[requests.clamp(0, table.shape[0] - 1)]
            out.copy_(
                values.masked_fill(
                    (values <= 0)
                    | (requests[:, None] < 0)
                    | (requests[:, None] >= table.shape[0]),
                    -1,
                )
            )
        return out, ((positions + 1).clamp_min(0) // ratio).to(torch.int32)
    return _kernel("selection_table", positions)(positions, requests, table, ratio)


def compressor_metadata(
    positions,
    requests,
    table,
    pages,
    active,
    pair_positions,
    pair_requests,
    previous,
    read_slots,
    write_slots,
):
    """Fill ratio-2 pair coordinates and LCM tail addresses in O(N).

    positions/requests are integer [N] in request-major order, with consecutive
    positions within each request. Whole request spans may be reordered; arbitrary
    token permutations are unsupported. The metadata producer owns this layout.
    table is the LCM tail page table; pages bounds physical residency.

    Destinations are contiguous [N]: bool active and integer pair_positions,
    pair_requests, previous, read_slots, write_slots. previous indexes the input
    rows, or is -1 for a predecessor outside the current span. Inactive pair
    coordinates and absent addresses use -1. Every destination is overwritten
    in place, including padding, so callers can reuse storage under CUDA graphs.
    Returns None. Inputs and destinations must not alias.
    """
    if not positions.is_cuda:
        active.copy_((positions >= 0) & (positions % 2 == 1))
        pair_positions.copy_((positions - 1).masked_fill(~active, -1))
        pair_requests.copy_(requests.masked_fill(~active, -1))
        previous.fill_(-1)
        in_batch = (
            active[1:]
            & (requests[1:] >= 0)
            & (requests[:-1] == requests[1:])
            & (positions[:-1] == pair_positions[1:])
        )
        previous[1:].copy_(
            torch.arange(max(positions.numel() - 1, 0)).masked_fill(~in_batch, -1)
        )
        read_slots.copy_(
            bounded_group_slots(pair_positions, pair_requests, table, 2, 1, 1, pages)
        )
        write_slots.copy_(
            bounded_group_slots(positions, requests, table, 2, 1, 1, pages)
        )
        return
    _kernel("compressor_metadata", positions)(
        positions,
        requests,
        table,
        pages,
        active,
        pair_positions,
        pair_requests,
        previous,
        read_slots,
        write_slots,
    )


# Backend registration (side-effect imports)
# isort: off
import tokenspeed_kernel.ops.attention.dsv41.triton  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.dsv41.gluon  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.dsv41.deep_select  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.dsv41.deep_gemm  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.dsv41.flash_mla  # noqa: E402,F401

# isort: on
