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

"""Length-aware (ragged) DSA decode indexer top-k via the CUDA persistent kernel.

Selects the per-row top-k over a dense indexer-logits matrix with the native CUDA
persistent-radix ``persistent_topk`` kernel: each row's valid context length is
read from ``lengths`` so padded columns beyond a request's context are never
scanned, and tie handling is delegated to the kernel. This is the ragged path
behind the DSA decode top-k selection, so callers pre-mask nothing. On devices
with cluster-launch support, long-row kernels use CUDA programmatic dependent
launch: their shared-memory setup can overlap the preceding grid's tail while
an in-kernel dependency wait keeps score reads ordered. Rows through 128K use a
single-load, register-resident cluster-8 specialization, 128K--256K rows use a
cluster-16 streaming specialization when the device supports it, and longer
rows retain the portable cluster-8 path. Devices without cluster-launch support
fall back to the original single-CTA streaming kernel. CUDA Graphs may capture
a static maximum width because each replay still dispatches from runtime
lengths.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.dsv4.cuda import (
    has_persistent_topk,
    persistent_topk,
)
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "has_ragged_decode_topk",
    "ragged_decode_topk",
]

platform = current_platform()

if platform.is_hopper_plus:
    from tokenspeed_kernel.ops.attention.dsa._triton.split_partials import (
        merge_split_partials,
    )
    from tokenspeed_kernel.ops.attention.mla.cuda import (
        flash_mla_sparse_fwd,
        flash_mla_with_kvcache,
        get_mla_metadata,
    )

_decode_sched_meta_cache: dict[tuple, object] = {}
_sm_counts: dict[int, int] = {}

# FlashMLA's sparse prefill kernel walks a row's columns in pairs of 64-column
# blocks and asserts the row width is a whole number of pairs; a column split
# must keep every slice at that alignment.
_FLASHMLA_SPARSE_TOPK_ALIGN = 128


def _sm_count(device: torch.device) -> int:
    index = device.index if device.index is not None else torch.cuda.current_device()
    count = _sm_counts.get(index)
    if count is None:
        count = torch.cuda.get_device_properties(index).multi_processor_count
        _sm_counts[index] = count
    return count


def _flashmla_sparse_column_splits(
    rows: int, padded_heads: int, width: int, device: torch.device
) -> int:
    """Column splits per query row for a FlashMLA sparse prefill launch.

    The kernel attends every ``_flashmla_sparse_prefill_head_multiple()``
    heads of a query row in one CTA that walks the row's selected columns
    serially, so a launch with fewer CTAs than SMs (decode, verify, short
    prompts) is bound by one CTA's walk while most SMs idle. Slicing the
    columns into independent rows, merged afterwards by their LSE, spreads
    that walk over the idle SMs. Splitting stops while the split launch,
    ``rows * ctas_per_row * splits`` CTAs, still fits one wave, past which it
    adds work without shortening the launch.
    """
    sms = _sm_count(device)
    ctas_per_row = padded_heads // _flashmla_sparse_prefill_head_multiple()
    splits = 1
    while (
        rows * ctas_per_row * splits * 2 <= sms
        and width % (splits * 2 * _FLASHMLA_SPARSE_TOPK_ALIGN) == 0
    ):
        splits *= 2
    return splits


def _flashmla_sparse_prefill_head_multiple() -> int:
    return 128 if platform.is_nvidia and platform.is_blackwell_plus else 64


def _flashmla_sparse_prefill_padded_heads(num_heads: int) -> int:
    head_multiple = _flashmla_sparse_prefill_head_multiple()
    return ((int(num_heads) + head_multiple - 1) // head_multiple) * head_multiple


def _flashmla_sparse_decode_padded_heads(num_heads: int) -> int:
    num_heads = int(num_heads)
    if num_heads <= 64:
        return 64
    if num_heads <= 128:
        return 128
    return num_heads


def _pad_query_heads(q: torch.Tensor, padded_heads: int) -> torch.Tensor:
    # Token counts vary per batch, so a per-shape workspace cache would grow
    # without bound; the caching allocator (or the graph pool under capture)
    # recycles these instead.
    actual_heads = q.shape[-2]
    q_padded = q.new_empty((*q.shape[:-2], padded_heads, q.shape[-1]))
    q_padded[..., :actual_heads, :].copy_(q)
    q_padded[..., actual_heads:, :].zero_()
    return q_padded


def _pad_prefill_query(q: torch.Tensor) -> tuple[torch.Tensor, int]:
    q = q.reshape(-1, q.shape[-2], q.shape[-1]).contiguous()
    actual_heads = q.shape[1]
    padded_heads = _flashmla_sparse_prefill_padded_heads(actual_heads)
    if padded_heads == actual_heads:
        return q, actual_heads
    return _pad_query_heads(q, padded_heads), actual_heads


def _pad_decode_query(q: torch.Tensor, q_len_per_req: int) -> tuple[torch.Tensor, int]:
    if q.dim() == 3:
        q = q.reshape(-1, int(q_len_per_req), q.shape[1], q.shape[2])
    elif q.dim() != 4:
        raise ValueError(
            "FlashMLA sparse decode q must be [tokens, heads, dim] or "
            f"[batch, q_len, heads, dim], got {tuple(q.shape)}"
        )
    q = q.contiguous()
    actual_heads = q.shape[2]
    padded_heads = _flashmla_sparse_decode_padded_heads(actual_heads)
    if padded_heads == actual_heads:
        return q, actual_heads
    return _pad_query_heads(q, padded_heads), actual_heads


def _flatten_regular_kv_cache(kv_cache: torch.Tensor, page_size: int) -> torch.Tensor:
    if kv_cache.dim() == 2:
        return kv_cache.view(-1, 1, kv_cache.shape[-1])
    if kv_cache.dim() == 3:
        return kv_cache.reshape(-1, kv_cache.shape[-2], kv_cache.shape[-1])
    if kv_cache.dim() == 4:
        if kv_cache.shape[1] != int(page_size):
            raise ValueError(
                f"paged kv_cache page size mismatch: got {kv_cache.shape[1]}, "
                f"expected {page_size}"
            )
        return kv_cache.reshape(-1, kv_cache.shape[-2], kv_cache.shape[-1])
    raise ValueError(f"unsupported kv_cache shape {tuple(kv_cache.shape)}")


def _paged_sparse_kv_cache(
    sparse_kv_cache: torch.Tensor, page_size: int
) -> torch.Tensor:
    if sparse_kv_cache.dim() == 2:
        return sparse_kv_cache.view(-1, int(page_size), 1, sparse_kv_cache.shape[-1])
    if sparse_kv_cache.dim() == 4:
        return sparse_kv_cache
    raise ValueError(
        f"unsupported sparse_kv_cache shape {tuple(sparse_kv_cache.shape)}"
    )


def _get_decode_sched_meta(
    *,
    q: torch.Tensor,
    num_reqs: int,
    q_len_per_req: int,
    actual_heads: int,
    topk: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> object:
    key = (
        q.device,
        q.dtype,
        int(num_reqs),
        int(q_len_per_req),
        int(actual_heads),
        int(topk),
        int(kv_lora_rank),
        int(qk_rope_head_dim),
    )
    meta = _decode_sched_meta_cache.get(key)
    if meta is None:
        meta = get_mla_metadata()[0]
        _decode_sched_meta_cache[key] = meta
    return meta


def has_ragged_decode_topk() -> bool:
    """Whether the length-aware CUDA persistent-radix top-k path is importable."""
    return has_persistent_topk()


def ragged_decode_topk(
    logits: torch.Tensor,
    out: torch.Tensor,
    topk: int,
    *,
    lengths: torch.Tensor,
    workspace: torch.Tensor,
    max_seq_len: int | None = None,
    q_len_per_req: int = 1,
) -> None:
    """Select per-row top-``topk`` local offsets over ragged (length-aware) logits.

    Args:
        logits: Indexer logits ``[num_rows, num_cols]``; padded columns beyond
            each row's valid length are skipped in-kernel (no pre-masking).
        out: int32 output buffer ``[num_rows, topk]`` written in place with the
            selected column indices.
        topk: Number of candidates to select per row.
        lengths: Per-row valid context length, cast to int32 and flattened.
        workspace: Scratch uint8 buffer for the persistent kernel.
        max_seq_len: Max scanned columns; defaults to ``logits.shape[1]``.
        q_len_per_req: Query rows per request (speculative decode); 1 for decode.

    Raises:
        RuntimeError: if the CUDA persistent kernel is unavailable.
    """
    if not has_ragged_decode_topk():
        raise RuntimeError("length-aware DSA decode top-k is unavailable.")
    persistent_topk(
        logits.contiguous(),
        lengths.to(torch.int32).contiguous().reshape(-1),
        out,
        workspace,
        int(topk),
        int(max_seq_len or logits.shape[1]),
        int(q_len_per_req),
    )


def _mask_sparse_slots(
    slots: torch.Tensor, lengths: torch.Tensor | None
) -> torch.Tensor:
    if lengths is None:
        return slots
    columns = torch.arange(slots.shape[-1], device=slots.device)
    return slots.masked_fill(columns >= lengths.reshape(-1, 1), -1)


def _pad_sink_for_heads(
    sink: torch.Tensor,
    padded_heads: int,
    actual_heads: int,
) -> torch.Tensor:
    """Widen a per-head sink to the padded query head count.

    The sparse FlashMLA kernels require the query to be padded to a multiple of
    64 (128 on Blackwell). The padded lanes produce discarded output, but they
    still take part in the softmax, so their sink must be ``-inf``: that makes
    ``exp(sink - max)`` zero and leaves the real heads' normalization untouched.

    Args:
        sink: FP32 sink logits covering the layer's ``actual_heads``.
        padded_heads: Head count the kernel will run with.
        actual_heads: Heads the sink actually describes.

    Returns:
        The sink narrowed to the padded width, padded with ``-inf`` when the
        kernel runs wider.
    """
    if sink.dtype != torch.float32:
        raise ValueError(f"attention sink must be float32, got {sink.dtype}")
    if sink.numel() < actual_heads:
        raise ValueError(
            f"attention sink covers {sink.numel()} heads, need {actual_heads}"
        )
    if padded_heads == actual_heads:
        return sink[:actual_heads].contiguous()
    padded = sink.new_full((padded_heads,), float("-inf"))
    padded[:actual_heads].copy_(sink[:actual_heads])
    return padded


def _finish_sparse_attention(
    result: torch.Tensor,
    lse: torch.Tensor,
    slots: torch.Tensor,
    out: torch.Tensor | None,
    return_lse: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    # Empty ranks contribute the identity element to the shared DCP reduction.
    empty = ~(slots >= 0).any(dim=-1)
    result.masked_fill_(empty[:, None, None], 0)
    lse = lse.contiguous().masked_fill(empty[:, None], -float("inf"))
    if out is not None:
        if (
            out.numel() != result.numel()
            or out.dtype != result.dtype
            or not out.is_contiguous()
        ):
            raise ValueError(
                "FlashMLA sparse attention output buffer must be a contiguous "
                f"{result.dtype} tensor of {result.numel()} elements, got "
                f"{out.dtype} {tuple(out.shape)} with strides {out.stride()}"
            )
        out.view(result.shape).copy_(result)
        result = out
    return (result, lse) if return_lse else result


def _flashmla_sparse_prefill_split(
    *,
    q: torch.Tensor,
    kv: torch.Tensor,
    slots: torch.Tensor,
    topk_lens: torch.Tensor | None,
    sm_scale: float,
    d_v: int,
    attn_sink: torch.Tensor | None,
    actual_heads: int,
    splits: int,
    out: torch.Tensor | None,
    return_lse: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """FlashMLA sparse prefill with every row's columns split ``splits`` ways.

    Split ``s`` of row ``r`` becomes kernel row ``r * splits + s`` over the
    columns ``[s * width, (s + 1) * width)``. The kernel runs without the sink
    (it would scale each partial by its own denominator); the merge applies it
    once to the combined softmax.
    """
    rows, padded_heads, head_dim = q.shape
    width = slots.shape[-1] // splits
    q_split = (
        q.unsqueeze(1)
        .expand(rows, splits, padded_heads, head_dim)
        .reshape(rows * splits, padded_heads, head_dim)
    )
    starts = torch.arange(splits, device=q.device, dtype=torch.int32) * width
    if topk_lens is None:
        lens = torch.full(
            (rows, 1), slots.shape[-1], device=q.device, dtype=torch.int32
        )
    else:
        lens = topk_lens.reshape(rows, 1).to(torch.int32)
    split_lens = (lens - starts).clamp_(0, width).reshape(-1)
    partial, _, partial_lse = flash_mla_sparse_fwd(
        q=q_split,
        kv=kv,
        indices=slots.reshape(rows * splits, 1, width),
        sm_scale=sm_scale,
        d_v=d_v,
        attn_sink=None,
        topk_length=split_lens,
    )
    return merge_split_partials(
        partial,
        partial_lse,
        split_lens,
        splits=splits,
        heads=actual_heads,
        sink=None if attn_sink is None else attn_sink[:actual_heads].contiguous(),
        out=out,
        return_lse=return_lse,
    )


if platform.is_nvidia and platform.is_hopper_plus:

    @register_kernel(
        "attention",
        "dsa_decode",
        name="flashmla_dsa_decode",
        solution="flashmla",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=frozenset({format_signature(q=dense_tensor_format(torch.bfloat16))}),
        traits={
            "q_len": frozenset({1, 2, 3, 4, 5, 6}),
            "qk_nope_head_dim": frozenset({128, 192}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({64}),
            "page_size": frozenset({64}),
            "topk": frozenset({512, 1024, 2048}),
            "has_kv_cache": frozenset({False, True}),
            "has_sparse_kv_cache": frozenset({True}),
            "topk_layout": frozenset({"global_slots"}),
            "logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
            "sinks": frozenset({False, True}),
        },
        priority=Priority.PERFORMANT,
    )
    def flashmla_dsa_decode(
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        sparse_kv_cache: torch.Tensor | None,
        topk_slots: torch.Tensor,
        topk_lens: torch.Tensor | None,
        max_seqlen_k: int,
        qk_nope_head_dim: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        softmax_scale: float,
        page_size: int,
        q_len_per_req: int,
        kv_seq_lens: torch.Tensor | None,
        logit_cap: float,
        k_scale: float,
        return_lse: bool,
        out: torch.Tensor | None,
        enable_pdl: bool,
        attn_sink: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del kv_seq_lens, enable_pdl
        if sparse_kv_cache is None:
            raise RuntimeError("FlashMLA sparse decode requires sparse_kv_cache")
        if logit_cap != 0.0:
            raise RuntimeError("FlashMLA sparse decode does not support logit_cap")
        q_padded, actual_heads = _pad_decode_query(q, q_len_per_req)
        num_reqs = q_padded.shape[0]
        kv_paged = _paged_sparse_kv_cache(sparse_kv_cache, page_size)
        slots = _mask_sparse_slots(topk_slots, topk_lens)
        sink = (
            None
            if attn_sink is None
            else _pad_sink_for_heads(attn_sink, q_padded.shape[2], actual_heads)
        )
        result, lse = flash_mla_with_kvcache(
            q=q_padded,
            k_cache=kv_paged,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=int(kv_lora_rank),
            tile_scheduler_metadata=_get_decode_sched_meta(
                q=q_padded,
                num_reqs=num_reqs,
                q_len_per_req=q_padded.shape[1],
                actual_heads=actual_heads,
                topk=topk_slots.shape[-1],
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
            ),
            softmax_scale=float(softmax_scale) * float(k_scale),
            is_fp8_kvcache=True,
            indices=slots.view(num_reqs, q_padded.shape[1], -1),
            attn_sink=sink,
        )
        if result.dim() == 4:
            result = result[:, :, :actual_heads, :].reshape(
                -1, actual_heads, result.shape[-1]
            )
        else:
            result = result[:, :actual_heads, :]
        lse = lse[:, :actual_heads, :].transpose(1, 2).reshape(-1, actual_heads)
        if return_lse and attn_sink is not None:
            # flash_mla_with_kvcache, like flash_mla_sparse_fwd, folds the sink
            # into the output normalization but not into the returned LSE.
            lse = torch.logaddexp(lse, attn_sink[:actual_heads].to(lse.dtype))
        return _finish_sparse_attention(result, lse, slots, out, return_lse)


if platform.is_nvidia and platform.is_hopper_plus:

    @register_kernel(
        "attention",
        "dsa_prefill",
        name="flashmla_dsa_prefill",
        solution="flashmla",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=frozenset({format_signature(q=dense_tensor_format(torch.bfloat16))}),
        traits={
            "q_len": frozenset({1}),
            "qk_nope_head_dim": frozenset({128, 192}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({64}),
            "page_size": frozenset({64}),
            "topk": frozenset({512, 1024, 2048}),
            "has_kv_cache": frozenset({True}),
            "has_sparse_kv_cache": frozenset({False, True}),
            "topk_layout": frozenset({"global_slots"}),
            "logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
            "sinks": frozenset({False, True}),
        },
        priority=Priority.PERFORMANT,
    )
    def flashmla_dsa_prefill(
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        sparse_kv_cache: torch.Tensor | None,
        topk_slots: torch.Tensor,
        topk_lens: torch.Tensor | None,
        max_seqlen_k: int,
        qk_nope_head_dim: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        softmax_scale: float,
        page_size: int,
        q_len_per_req: int,
        kv_seq_lens: torch.Tensor | None,
        logit_cap: float,
        k_scale: float,
        return_lse: bool,
        out: torch.Tensor | None,
        enable_pdl: bool,
        attn_sink: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del enable_pdl
        if kv_cache is None:
            raise RuntimeError("FlashMLA sparse prefill requires kv_cache")
        if logit_cap != 0.0:
            raise RuntimeError("FlashMLA sparse prefill does not support logit_cap")
        q_kernel, actual_heads = _pad_prefill_query(q)
        kv = _flatten_regular_kv_cache(kv_cache, page_size)
        slots = _mask_sparse_slots(topk_slots, topk_lens)
        splits = _flashmla_sparse_column_splits(
            q_kernel.shape[0], q_kernel.shape[1], slots.shape[-1], q_kernel.device
        )
        if splits > 1:
            return _flashmla_sparse_prefill_split(
                q=q_kernel,
                kv=kv,
                slots=slots,
                topk_lens=topk_lens,
                sm_scale=float(softmax_scale) * float(k_scale),
                d_v=int(kv_lora_rank),
                attn_sink=attn_sink,
                actual_heads=actual_heads,
                splits=splits,
                out=out,
                return_lse=return_lse,
            )
        sink = (
            None
            if attn_sink is None
            else _pad_sink_for_heads(attn_sink, q_kernel.shape[1], actual_heads)
        )
        # Every slot at or past topk_lens is masked to -1 above, so topk_length
        # lets the kernel skip those columns instead of iterating them.
        result, _, lse = flash_mla_sparse_fwd(
            q=q_kernel,
            kv=kv,
            indices=slots.unsqueeze(1),
            sm_scale=float(softmax_scale) * float(k_scale),
            d_v=int(kv_lora_rank),
            attn_sink=sink,
            topk_length=(
                None
                if topk_lens is None
                else topk_lens.reshape(-1).to(torch.int32).contiguous()
            ),
        )
        result = result[:, :actual_heads, :]
        lse = lse[:, :actual_heads]
        if return_lse and attn_sink is not None:
            # flash_mla_sparse_fwd folds the sink into the output normalization
            # but not into the returned LSE; add it so DCP merges see the same
            # denominator as the output.
            lse = torch.logaddexp(lse, attn_sink[:actual_heads].to(lse.dtype))
        return _finish_sparse_attention(result, lse, slots, out, return_lse)

    @register_kernel(
        "attention",
        "dsa_decode",
        name="flashmla_dense_dsa_decode",
        solution="flashmla",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=frozenset({format_signature(q=dense_tensor_format(torch.bfloat16))}),
        traits={
            "q_len": frozenset({1, 2, 3, 4, 5, 6}),
            "qk_nope_head_dim": frozenset({128, 192}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({64}),
            "page_size": frozenset({64}),
            "topk": frozenset({512, 1024, 2048}),
            "has_kv_cache": frozenset({True}),
            "has_sparse_kv_cache": frozenset({False}),
            "topk_layout": frozenset({"global_slots"}),
            "logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
            # Sinkless dense-KV decode without LSE stays on Triton (Hopper) and
            # FlashInfer TRT-LLM (Blackwell); this path pads every row's heads
            # to the prefill tile.
            "sinks": frozenset({True}),
        },
        priority=Priority.PERFORMANT,
    )
    def flashmla_dense_dsa_decode(**kwargs):
        # Dense BF16 KV: every query row carries its own global top-k slots, so
        # decode rows go through the same token-parallel sparse kernel as prefill.
        return flashmla_dsa_prefill(**kwargs)

    @register_kernel(
        "attention",
        "dsa_decode",
        name="flashmla_dense_dsa_partial_decode",
        solution="flashmla",
        # Measured against Triton on Hopper only; Blackwell keeps Triton.
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            max_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=frozenset({format_signature(q=dense_tensor_format(torch.bfloat16))}),
        traits={
            "q_len": frozenset({1, 2, 3, 4, 5, 6}),
            "qk_nope_head_dim": frozenset({128, 192}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({64}),
            "page_size": frozenset({64}),
            "topk": frozenset({512, 1024, 2048}),
            "has_kv_cache": frozenset({True}),
            "has_sparse_kv_cache": frozenset({False}),
            "topk_layout": frozenset({"global_slots"}),
            "logit_cap": frozenset({False}),
            # Context-parallel ranks return sinkless partials and apply the
            # sink once, after merging them by their LSE.
            "return_lse": frozenset({True}),
            "sinks": frozenset({False}),
        },
        priority=Priority.PERFORMANT,
    )
    def flashmla_dense_dsa_partial_decode(**kwargs):
        return flashmla_dsa_prefill(**kwargs)
