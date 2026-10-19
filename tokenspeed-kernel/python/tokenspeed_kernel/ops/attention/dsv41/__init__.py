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

The ``dspark_*`` helpers serve the checkpoint-local DSpark drafter, whose
context windows are BF16 [pages,64,512] fields of the SWA group: one launch
per stage turns projected rows into window rows (kv_norm, RoPE, the SWA
codec's quantize-dequantize round trip, masked scatter), one launch derives a
verify step's bonus tokens and anchors, and one launch expands anchors into
the block's ids, positions, HC mix and workspace addressing.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.dsv41._inputs import IndexKeyInput, IndexQueryInput
from tokenspeed_kernel.ops.attention.mla._triton.page_table import bounded_group_slots
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "IndexKeyInput",
    "IndexQueryInput",
    "decode_rows",
    "decode_window",
    "global_slots",
    "compact_slots",
    "selection_table",
    "cache_pack",
    "cache_unpack",
    "cache_scatter",
    "cache_gather",
    "index_q_quantize",
    "index_key_norm_rope",
    "index_key_norm_rope_scatter",
    "index_query_rope_weights",
    "index_query_rope_quant",
    "selected_attention",
    "selected_attention_rope",
    "new_attention_schedule",
    "index_score",
    "index_topk",
    "new_index_metadata",
    "index_select_all",
    "local_index_supported",
    "index_candidates",
    "merge_index_candidates",
    "compressor_tail_scatter",
    "compressor_pool",
    "compressor_metadata",
    "swa_rope_scatter",
    "global_rope_scatter",
    "rope_inplace",
    "rope_pad_query",
    "dspark_rows",
    "dspark_anchors",
    "dspark_block",
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


def index_key_norm_rope(values, weight, positions, cos_sin_cache, eps):
    """Normalize BF16 [T,128] index keys and rotate their interleaved tail.

    weight is contiguous BF16 [128], positions is integer [T], cos_sin_cache
    is FP32 [max_position, rotary_dim], and eps is positive. Negative positions
    use row zero. Returns fresh contiguous BF16 [T,128] rows, preserving the
    inputs and the intermediate BF16 normalization rounding before RoPE.
    """
    return _kernel("index_key_norm_rope", values)(
        values, weight, positions, cos_sin_cache, eps
    )


def global_rope_scatter(values, positions, cos_sin_cache, cache, slots):
    """Rotate BF16 [T,512] global keys and write page-planar FP4 cache rows.

    positions and slots are integer [T]; cos_sin_cache is FP32 [max_position,64].
    cache is uint8 [pages,64,288]. Negative positions use table row zero; negative
    and out-of-range slots skip writes. Valid write slots must be unique, and
    runtime null pages must already map to -1. Returns None, preserves values,
    and retains the intermediate BF16 rounding before global E2M1/E4M3 packing.
    """
    _kernel("global_rope_scatter", values)(
        values, positions, cos_sin_cache, cache, slots
    )


def index_key_norm_rope_scatter(
    values, weight, positions, cos_sin_cache, eps, cache, slots
):
    """Normalize and rotate BF16 [T,128] keys, then store packed FP4 pages.

    weight is BF16 [128], positions and slots are integer [T], cos_sin_cache
    is FP32 [max_position, rotary_dim], and eps is positive. cache is the
    owner's uint8 [pages,64,68] field. Negative/out-of-range slots skip writes;
    valid slots must be unique and runtime null pages must map to -1. Returns
    None, preserves inputs and retains BF16 rounding after norm and RoPE.
    """
    _kernel("index_key_norm_rope_scatter", values)(
        values, weight, positions, cos_sin_cache, eps, cache, slots
    )


def index_query_rope_quant(
    query, head_weights, positions, cos_sin_cache, weight_scale, output_dtype
):
    """Rotate and pack BF16 [T,H,128] queries and scale BF16 [T,H] weights.

    positions is integer [T], cos_sin_cache is FP32 [max_position,64], and
    weight_scale is applied in FP32 then rounded to BF16. output_dtype selects
    BF16 or FP32 head weights for the scorer. Returns ((uint8 [T,H,64] FP4
    values, int32 [T,H] E8M0 scale words), weights [T,H]). Inputs are preserved;
    RoPE retains its BF16 rounding before quantization. Negative positions use
    table row zero.
    """
    return _kernel("index_query_rope_quant", query)(
        query, head_weights, positions, cos_sin_cache, weight_scale, output_dtype
    )


def index_query_rope_weights(
    query, head_weights, positions, cos_sin_cache, weight_scale
):
    """Rotate BF16 [T,H,128] queries in place and scale BF16 [T,H] weights.

    positions is integer [T] and cos_sin_cache is FP32 [max_position,64].
    weight_scale is multiplied in FP32 then rounded to BF16. Returns
    (query, scaled_weights), with fresh contiguous BF16 [T,H] weights and
    query rotated in place. Head weights are preserved; negative positions
    use table row zero.
    """
    return _kernel("index_query_rope_weights", query)(
        query, head_weights, positions, cos_sin_cache, weight_scale
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
    norm_weight: torch.Tensor | None,
    norm_eps: float,
) -> torch.Tensor | None:
    """Rotate normalized BF16 SWA rows and quantize directly into planar pages.

    values is [T,512], positions and physical slots are integer [T], and the
    FP32 RoPE table is [max_position,64]. cache is the LCM uint8 [pages,64,R]
    field with contiguous page bytes, where R is the row width ``cache_format``
    declares. Invalid/null slots skip writes. If out is supplied, write all
    quantized/dequantized BF16 [T,512] rows there, including rows outside
    persistent retention. Read old prefixes before this operation.
    If norm_weight is supplied, normalize and round to BF16 before RoPE.
    """
    _kernel("swa_rope_scatter", values)(
        values,
        positions,
        cos_sin_cache,
        cache,
        slots,
        cache_format,
        out,
        norm_weight,
        norm_eps,
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
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    out: torch.Tensor | None,
    query_chunk_size: int,
    schedule: object | None,
    prefill_kv: torch.Tensor | None,
    prefill_indices: torch.Tensor | None,
    *,
    lse: torch.Tensor | None,
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
            each real head includes its sink exactly once. None requests no sink.
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

        lse: Optional contiguous FP32 [T, H] natural-log LSE destination.
            Requires attn_sink=None for distributed partial attention.

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

    if lse is not None:
        if attn_sink is not None:
            raise ValueError("partial attention LSE requires no sink")
        if (
            lse.shape != q.shape[:2]
            or lse.dtype != torch.float32
            or lse.device != q.device
            or not lse.is_contiguous()
        ):
            raise ValueError(
                "lse must be contiguous FP32 [tokens, heads] on the query device"
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
        **({"lse": lse} if lse is not None else {}),
    )


def selected_attention_rope(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_slots: torch.Tensor,
    swa_lens: torch.Tensor,
    global_cache: torch.Tensor | None,
    global_slots: torch.Tensor | None,
    global_lens: torch.Tensor | None,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    schedule: object | None,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    """Paged decode ``selected_attention`` followed by ``rope_inplace`` on its output.

    Arguments follow ``selected_attention`` (paged inputs, query_chunk_size 256)
    and ``rope_inplace``. A kernel that rotates the output in its epilogue is
    used when one is registered; otherwise the two ops run back to back.
    """
    try:
        kernel = select_kernel(
            "attention",
            "dsv41_selected_attention_rope",
            format_signature(x=dense_tensor_format(q.dtype)),
            features=None,
            platform=None,
            traits={},
            solution=None,
            override=None,
        )
    except NoKernelFoundError:
        out = selected_attention(
            q,
            swa_cache,
            swa_slots,
            swa_lens,
            global_cache,
            global_slots,
            global_lens,
            attn_sink,
            softmax_scale,
            None,
            256,
            schedule,
            None,
            None,
            lse=None,
        )
        return rope_inplace(out, positions, cos_sin_cache, None)
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
        schedule,
        positions,
        cos_sin_cache,
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
    index_q: torch.Tensor | IndexQueryInput,
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
    index_metadata: object | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bounded Full/Reindex scoring, Top-K rows, and optional source candidates.

    Args:
        index_q: Post-RoPE BF16 [T,H,128], or IndexQueryInput to fuse RoPE and
            quantization into native scoring's input preparation.
        weights: BF16/FP32 [T,H], already scaled for a tensor index_q, or
            unscaled BF16 weights when index_q is IndexQueryInput.
        index_cache: Owner's strided uint8 [pages, 64, 68] index field.
        page_table: Int32/int64 [T, logical_pages], a request table per query.
            Entries address this index field, not the main-KV field. Invalid
            pages are ignored; tables may be expanded/noncontiguous views.
        visible_lens: Int32/int64 [T] causal row counts: floor((position+1)/ratio).
        candidate_blocks: None for Full; otherwise int32/int64 [T, <=2048]
            request-local block IDs, padded with -1. Reindex selects only from
            these blocks, reapplying visible_lens to their rows. Blackwell
            native sparse scoring reads those blocks directly from the cache.
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
        index_metadata: Object from new_index_metadata for these exact table and
            length tensors, shared only within one invocation, or None (default). Native
            kernels reuse Full schedules and matching immutable sparse candidate
            geometry across layers. Table, lengths and candidate contents must
            not change while sharing this object; graph replay refreshes the
            captured producers. A new invocation needs a fresh object.

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
    and block maxima; on Hopper, DeepGEMM scores FP8 ``index_v4`` rows and the
    in-tree CuTe DSL DeepSelect kernel selects. This native accumulation
    differs from the portable reference's intermediate BF16 rounding.
    Full-scoring query tiles cap logits at 32MiB for paged queries and
    128MiB for a zero-row-stride request table.
    That broadcast layout gathers packed history once per call and packs Q
    once before the score tiles; no payload survives the call. For unsharded
    68-byte MXFP4 caches, AMD Gluon Full similarly caps each FP32 logits query
    tile at 32MiB instead of limiting history width; Reindex scores only the
    candidate rows.
    Blackwell Reindex uses DeepGEMM's native sparse MQA scorer when the index
    page stride is a multiple of 512 bytes. It reads the owner's cache directly,
    with BF16 head weights, weighted head reduction and logits. DeepSelect
    consumes those BF16 scores directly. Tiles follow query_chunk_size, with
    at most 16384 BF16 logits per query; no candidate K payload is copied.
    This applies equally to broadcast and paged query tables, including short
    final tiles. Other page strides keep full scoring
    followed by candidate masking. The candidate selection semantics are the
    same; BF16 scores need not produce the FP32 path's exact Top-K at ties.
    """
    from tokenspeed_kernel.ops.attention.dsv41.deep_gemm import (
        is_hopper_indexer_available,
        is_native_indexer_available,
    )
    from tokenspeed_kernel.thirdparty.deep_select import is_deep_select_available

    if index_metadata is not None:
        from tokenspeed_kernel.ops.attention.dsv41._metadata import IndexMetadata

        if not isinstance(index_metadata, IndexMetadata):
            raise TypeError("index_metadata must come from new_index_metadata")
        index_metadata.validate(page_table, visible_lens)

    # Each target scores the index rows its own cache format stores: FP4 with
    # DeepSelect on Blackwell, FP8 with the CuTe DSL DeepSelect kernel on Hopper.
    query_values = index_q.values if isinstance(index_q, IndexQueryInput) else index_q
    row_bytes = index_cache.shape[2] if index_cache.ndim == 3 else 0
    shaped = (
        query_values.is_cuda
        and process_group is None
        and query_values.shape[1] == 32
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
    index_heads = query_values.shape[1]
    if process_group is not None:
        index_shards = torch.distributed.get_world_size(process_group)
        index_heads *= index_shards
    kernel = select_kernel(
        "attention",
        "dsv41_index_topk",
        format_signature(x=dense_tensor_format(query_values.dtype)),
        features=None,
        platform=None,
        traits={
            "index_heads": index_heads,
            "index_k_format": index_k_format,
            "index_shards": index_shards,
            "native_indexer": native,
        },
        solution=solution,
        override=None,
    )
    spec = KernelRegistry.get().get_by_name(kernel.name)
    if isinstance(index_q, IndexQueryInput) and (
        spec is None or True not in spec.traits.get("supports_index_query_input", ())
    ):
        # Kernels that take only post-RoPE Q get it applied here, so every
        # supplied preprocessing input is still honored. Preserve projected Q
        # for other query windows.
        index_q, weights = index_query_rope_weights(
            query_values.clone(),
            weights,
            index_q.positions,
            index_q.cos_sin_cache,
            index_q.weight_scale,
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
        index_metadata,
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


def dspark_rows(
    values: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float | None,
    positions: torch.Tensor | None,
    cos_sin_cache: torch.Tensor | None,
    window: torch.Tensor | None,
    slots: torch.Tensor | None,
    out: torch.Tensor | None,
) -> torch.Tensor | None:
    """Produce DSpark context-window rows from projected BF16 [T,512] values.

    One launch performs the reference chain for every row: optionally the
    kv_norm RMSNorm in the eager cast order (``norm_weight`` [512] with
    ``norm_eps``; pass both or neither), optionally the interleaved-tail
    RoPE at ``positions`` from the FP32 ``cos_sin_cache`` [max_position,
    rotary_dim] (pass both or neither), then the all-channel 1x32 E4M3/E8M0
    quantize-dequantize round trip to BF16. Rows land in the BF16 paged
    ``window`` [pages,rows,512] at ``slots`` [T] (negative or out-of-field
    slots write nothing; pass both or neither) and/or in ``out`` [T,512].
    ``values`` may be a unit-column-stride slice. Returns ``out``.
    """
    _kernel("dspark_rows", values)(
        values, norm_weight, norm_eps, positions, cos_sin_cache, window, slots, out
    )
    return out


def dspark_anchors(
    output_tokens: torch.Tensor,
    accept_lengths: torch.Tensor,
    positions: torch.Tensor,
    num_extends: int,
    width: int,
    next_tokens: torch.Tensor,
    start_pos: torch.Tensor,
    *,
    num_prefill_outputs: int,
) -> None:
    """Fill each verify row's bonus token and each decode request's anchor.

    ``output_tokens`` [num_prefill_outputs + num_decodes * width] holds
    completing-prefill tokens then each decode request's verify rows.
    ``num_prefill_outputs`` is the active prefix of ``num_extends``; the
    remaining extend rows have no tokens and their destinations are untouched;
    ``accept_lengths`` [bs] counts accepted tokens (clamped to 1..width) and
    ``positions`` [num_decodes * width] are the decode rows' positions.
    Every column of ``next_tokens`` [bs, spec] (int32) receives its row's
    bonus token, and ``start_pos`` [num_decodes] (int64) the position of the
    last accepted verify row. Only output-bearing request rows are written.
    """
    bs = accept_lengths.shape[0]
    num_decodes = bs - num_extends
    if not 0 <= num_prefill_outputs <= num_extends <= bs or width < 1:
        raise ValueError("DSpark anchors need 0 <= num_extends <= bs and width >= 1")
    if (
        output_tokens.shape != (num_prefill_outputs + num_decodes * width,)
        or positions.shape != (num_decodes * width,)
        or next_tokens.shape[:1] != (bs,)
        or next_tokens.ndim != 2
        or next_tokens.dtype != torch.int32
        or next_tokens.stride(1) != 1
        or start_pos.shape != (num_decodes,)
        or start_pos.dtype != torch.int64
    ):
        raise ValueError("DSpark anchor tensors do not match the verify layout")
    for tensor in (output_tokens, accept_lengths, positions, start_pos):
        if tensor.dtype not in (torch.int32, torch.int64) or not tensor.is_contiguous():
            raise ValueError("DSpark anchor inputs must be contiguous integers")
    if not positions.is_cuda:
        rows = torch.arange(num_decodes)
        accepted = accept_lengths[num_extends:].to(torch.int64).clamp(1, width)
        verify = num_prefill_outputs + rows * width + accepted - 1
        next_tokens[:num_prefill_outputs].copy_(
            output_tokens[:num_prefill_outputs, None]
        )
        next_tokens[num_extends:].copy_(output_tokens[verify, None])
        start_pos.copy_(positions[rows * width + accepted - 1])
        return
    _kernel("dspark_anchors", positions)(
        output_tokens,
        accept_lengths,
        positions,
        num_extends,
        num_prefill_outputs,
        width,
        next_tokens,
        start_pos,
    )


def dspark_block(
    bonus: torch.Tensor,
    start_pos: torch.Tensor,
    history_slots: torch.Tensor,
    noise_token: int,
    rows_per_page: int,
    hc_mult: int,
    block_size: int,
) -> tuple[torch.Tensor, ...]:
    """Expand per-request anchors into one draft block's inputs and addressing.

    ``bonus`` [n] and ``start_pos`` [n] are integer anchors, ``history_slots``
    [n, window] the requests' context-window slots (negative = absent).
    Returns ``(ids, positions, pre_mix, request_indices, page, row,
    indices)``: int32 ids [n*block] (bonus then ``noise_token``), int64
    positions [n*block] (``start_pos+1..``), fp32 one-hot pre_mix
    [n*block, hc_mult], int64 request_indices [n*block], int64 page/row
    [n, window] resolving clamped slots with ``rows_per_page`` rows, and
    int32 workspace indices [n*block, window+block] selecting each request's
    history rows (-1 where absent) and its own block rows.
    """
    if (
        history_slots.ndim != 2
        or bonus.shape != history_slots.shape[:1]
        or start_pos.shape != history_slots.shape[:1]
    ):
        raise ValueError("DSpark block inputs must be [n], [n] and [n, window]")
    if block_size < 1 or hc_mult < 1 or rows_per_page < 1:
        raise ValueError("DSpark block geometry must be positive")
    for tensor in (bonus, start_pos, history_slots):
        if tensor.dtype not in (torch.int32, torch.int64):
            raise ValueError("DSpark block inputs must be int32/int64")
    if history_slots.stride(1) != 1:
        raise ValueError("DSpark history slots need a unit column stride")
    if not bonus.is_cuda:
        n, window = history_slots.shape
        device = bonus.device
        ids = torch.full((n, block_size), noise_token, dtype=torch.int32, device=device)
        ids[:, 0] = bonus
        positions = start_pos.to(torch.int64)[:, None] + 1 + torch.arange(block_size)
        pre_mix = torch.zeros((n * block_size, hc_mult), dtype=torch.float32)
        pre_mix[:, 0] = 1
        request_indices = torch.arange(n).repeat_interleave(block_size)
        slots = history_slots.clamp_min(0).long()
        width = window + block_size
        indices = torch.arange(n * width, dtype=torch.int32).view(n, width)
        indices[:, :window].masked_fill_(history_slots < 0, -1)
        indices = indices[:, None, :].expand(-1, block_size, -1).reshape(-1, width)
        return (
            ids.reshape(-1),
            positions.reshape(-1),
            pre_mix,
            request_indices,
            slots // rows_per_page,
            slots % rows_per_page,
            indices,
        )
    return _kernel("dspark_block", history_slots)(
        bonus, start_pos, history_slots, noise_token, rows_per_page, hc_mult, block_size
    )


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
    swa_table,
    swa_pages,
):
    """Fill SWA addresses for N logical query coordinates.

    positions/requests are integer [N] in consecutive request-major spans;
    spans may have different lengths. swa_table holds LCM group pages.
    Destinations are write_slots[N], read_slots[N,128] and read_lens[N].
    Padding rows (negative position or request) resolve to -1. Page zero and
    pages beyond swa_pages never address live cache. Whether the window's
    history is resident is the scheduler's retention contract, not checked
    here. Returns None.
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
        return
    _kernel("decode_window", positions)(
        positions,
        requests,
        write_slots,
        read_slots,
        read_lens,
        swa_table,
        swa_pages,
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


def local_index_supported(index_q, index_cache, candidates):
    """Whether local FP8 candidate scoring has a native implementation."""
    from tokenspeed_kernel.ops.attention.dsv41.deep_gemm import (
        is_hopper_indexer_available,
    )

    return (
        isinstance(index_q, torch.Tensor)
        and index_q.is_cuda
        and is_hopper_indexer_available()
        and index_cache.shape[-1] == 132
        and index_q.shape[1] % 8 == 0
        and (
            candidates is None
            or (candidates.shape[1] > 0 and candidates.shape[1] % 16 == 0)
        )
    )


def index_candidates(index_q, weights, index_cache, table, visible, candidates, topk):
    """Return local TopK as int32 [N,K,2] (logical ID, FP32 score bits).

    Table entries address local Index K pages; nonowned pages are -1.
    Candidate blocks and causal lengths retain global logical coordinates.
    candidates=None scans all locally owned visible pages for a Full pass.
    Scores are transported as raw bits, never rounded or numerically reduced.
    """
    return _kernel("index_candidates", index_q)(
        index_q, weights, index_cache, table, visible, candidates, topk
    )


def merge_index_candidates(packed, rows, lengths):
    """Select from gathered int32 [N,ranks*K,2] ID/score pairs.

    Writes sorted global logical IDs to rows [N,K], padded -1, and valid
    counts to lengths [N]. All tensors have fixed shapes for graph replay.
    """
    return _kernel("merge_index_candidates", packed)(packed, rows, lengths)


# Backend registration (side-effect imports)
# isort: off
import tokenspeed_kernel.ops.attention.dsv41.triton  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.dsv41._triton.rope  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.dsv41.gluon  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.dsv41.deep_select  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.dsv41.deep_gemm  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.dsv41.flash_mla  # noqa: E402,F401

# isort: on


def index_select_all(page_table, visible_lens, pages, candidate_block_size, out):
    """Select every valid causal row without projections, scores or Top-K.

    Args:
        page_table: Integer [T, logical_pages] request tables; null page 0 and
            entries outside [1, pages) are ignored.
        visible_lens: Integer [T] causal row counts, bounded by out[0].shape[1].
        pages: Physical page count, including the reserved null page.
        candidate_block_size: Must be 8, as for index_topk.
        out: Contiguous int32 row IDs, lengths, block IDs, block lengths, with
            shapes [T, K], [T], [T, C], [T]. C=0 disables candidate publication.
            Otherwise C must cover every visible block.

    Returns:
        The four destinations, fully written with sorted IDs and -1 padding.
        The caller must prove that all visible rows/blocks fit these outputs;
        this helper does not score or truncate a candidate selection.
    """
    rows, lengths, blocks, block_lengths = out
    n = page_table.shape[0]
    if (
        page_table.ndim != 2
        or page_table.dtype not in (torch.int32, torch.int64)
        or visible_lens.shape != (n,)
        or visible_lens.dtype not in (torch.int32, torch.int64)
        or visible_lens.device != page_table.device
        or pages < 1
        or candidate_block_size != 8
        or rows.ndim != 2
        or blocks.ndim != 2
        or not 1 <= rows.shape[1] <= 512
        or not 0 <= blocks.shape[1] <= 2048
    ):
        raise ValueError("Invalid all-history selection geometry")
    shapes = ((n, rows.shape[1]), (n,), (n, blocks.shape[1]), (n,))
    for tensor, shape in zip(out, shapes, strict=True):
        if (
            tensor.shape != shape
            or tensor.dtype != torch.int32
            or tensor.device != page_table.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                "All-history outputs must be contiguous int32 destinations"
            )
    if page_table.is_cuda:
        return _kernel("index_select_all", page_table)(
            page_table, visible_lens, pages, candidate_block_size, out
        )
    for destination, counts, stride in ((rows, lengths, 1), (blocks, block_lengths, 8)):
        ids = torch.arange(destination.shape[1], dtype=torch.int32)
        offsets = ids * stride
        valid = offsets[None, :] < visible_lens[:, None]
        if page_table.shape[1]:
            physical = page_table[:, (offsets // 64).clamp_max(page_table.shape[1] - 1)]
            valid &= (
                (offsets[None, :] // 64 < page_table.shape[1])
                & (physical > 0)
                & (physical < pages)
            )
        else:
            valid.fill_(False)
        selected = torch.where(valid, ids[None, :], 2147483647).sort(dim=1).values
        destination.copy_(selected.masked_fill(selected == 2147483647, -1))
        counts.copy_(valid.sum(dim=1))
    return out


def new_index_metadata(page_table: torch.Tensor, visible_lens: torch.Tensor) -> object:
    """Create invocation-owned reusable index geometry for these exact inputs.

    page_table and visible_lens have the index_topk contract and must remain
    unchanged for this object's lifetime. Share it only between layers of one
    forward; create a new object for every eager invocation and graph capture.
    Replay recomputes the captured producers. No KV values or scores are cached.
    Consumers on other streams wait for metadata producers before reading.
    Cached geometry is retained until this object is released.
    """
    from tokenspeed_kernel.ops.attention.dsv41._metadata import IndexMetadata

    return IndexMetadata(page_table, visible_lens)


def compact_slots(slots: torch.Tensor, lengths: torch.Tensor):
    """Stably pack valid physical slots within each row's scan length.

    Integer slots [N,K] use negative values for holes. Integer lengths [N]
    bound each row's input scan. Returns contiguous slots [N,K], padded with
    -1, and int32 valid counts [N]. Output shapes remain fixed for graph replay.
    """
    if (
        slots.ndim != 2
        or lengths.shape != slots.shape[:1]
        or slots.dtype not in (torch.int32, torch.int64)
        or lengths.dtype not in (torch.int32, torch.int64)
        or slots.device != lengths.device
    ):
        raise ValueError("Expected integer slots [N,K] and lengths [N] on one device")
    if not slots.is_cuda:
        columns = torch.arange(slots.shape[1], device=slots.device)
        valid = (slots >= 0) & (columns[None, :] < lengths[:, None])
        order = torch.where(valid, columns, columns + slots.shape[1]).argsort(dim=-1)
        packed = slots.masked_fill(~valid, -1).gather(1, order)
        return packed, valid.sum(dim=-1, dtype=torch.int32)
    return _kernel("compact_slots", slots)(slots, lengths)
