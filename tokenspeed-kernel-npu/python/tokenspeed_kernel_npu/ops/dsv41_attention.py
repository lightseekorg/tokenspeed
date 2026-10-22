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

"""V4.1 selected attention using FluentLLM's existing Ascend SFA operator.

The complete post-RoPE 512D vector enters the main dot product. A zero 64D
rotary input satisfies the vendor ABI without changing scores. Returned
softmax statistics apply the zero-valued sink once. The native output has
already rounded to BF16 before this rescaling; this is not bitwise equal to
one final BF16 rounding. This eager path allocates temporary zero rotary rows.
"""

import torch
from tokenspeed_kernel_npu.ops import dsv41_cache as cache_ops
from tokenspeed_kernel_npu.ops.dsv41_attention_fusion import apply_sink, compact_indices
from tokenspeed_kernel_npu.ops.dsv41_pointwise import rope_inplace as fused_rope_inplace


def rope_inplace(values, positions, table):
    """Fuse aligned contiguous tails; preserve the strided-layout fallback."""
    if values.stride(-1) == 1 and table.shape[1] % 32 == 0:
        return fused_rope_inplace(values, positions, table)
    width = table.shape[1]
    angles = table[positions.clamp_min(0).long()]
    shape = (positions.numel(),) + (1,) * (values.ndim - 2) + (width // 2,)
    cosine, sine = (x.reshape(shape) for x in angles.chunk(2, -1))
    tail = values[..., -width:].float()
    even, odd = tail[..., ::2], tail[..., 1::2]
    rotated_even = even * cosine - odd * sine
    rotated_odd = odd * cosine + even * sine
    # A stride-2 destination sends each copy through AI CPU ViewCopy on 910B.
    # Interleave first, then write the contiguous channel range. Keep the
    # FP32 products/addition and the final output cast identical to above.
    rotated = torch.stack((rotated_even, rotated_odd), -1).flatten(-2)
    values[..., -width:].copy_(rotated)
    return values


def compact_attention(q, kv, indices, sink, softmax_scale):
    """BF16 q[T,H,512], kv[K,512], integer indices[T,W], FP32 sink[H]."""
    if q.ndim != 3 or q.shape[-1] != 512 or q.dtype != torch.bfloat16:
        raise ValueError("q must be BF16 [tokens,heads,512]")
    if q.shape[1] not in (1, 2, 4, 8, 16, 32, 64, 128):
        raise ValueError("Ascend SFA requires a supported power-of-two head count")
    if kv.ndim != 2 or kv.shape[-1] != 512 or kv.dtype != q.dtype:
        raise ValueError("kv must be BF16 [rows,512]")
    if (
        indices.ndim != 2
        or indices.shape[0] != q.shape[0]
        or indices.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("indices must be integer [tokens,width]")
    if sink.shape != (q.shape[1],) or sink.dtype != torch.float32:
        raise ValueError("sink must be FP32 [heads]")
    if q.device.type != "npu" or any(x.device != q.device for x in (kv, indices, sink)):
        raise ValueError("All attention inputs must share an NPU device")
    if q.shape[0] == 0 or kv.shape[0] == 0 or indices.shape[1] == 0:
        return torch.zeros_like(q)
    import flash_ops  # noqa: F401 - registers the existing FluentLLM operator

    tokens, heads, _ = q.shape
    ids, valid_counts = compact_indices(indices, kv.shape[0])
    key = kv.contiguous().reshape(1, -1, 1, 512)
    output, maximum, total = torch.ops.custom.npu_sparse_flash_attention(
        q.contiguous().unsqueeze(0),
        key,
        key,
        ids.reshape(1, tokens, 1, -1),
        softmax_scale,
        query_rope=torch.zeros((1, tokens, heads, 64), dtype=q.dtype, device=q.device),
        key_rope=torch.zeros((1, kv.shape[0], 1, 64), dtype=q.dtype, device=q.device),
        sparse_block_size=1,
        layout_query="BSND",
        layout_kv="BSND",
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=True,
    )
    return apply_sink(output.reshape_as(q), maximum, total, sink, valid_counts)


def create_paged_context(arena):
    """Create constants owned by the cache pool."""
    from tokenspeed_kernel_npu.ops.dsv41_paged_attention import PagedArenaContext

    return PagedArenaContext(arena)


def selected_attention(
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
    *,
    pa_mode,
    pa_context,
):
    if pa_mode not in ("compact", "direct"):
        raise ValueError("Ascend PA mode must be compact or direct")
    if pa_mode == "direct" and pa_context is None:
        raise ValueError("Direct PA requires a pool-owned context")
    if pa_mode == "compact" and pa_context is not None:
        raise ValueError("Compact PA must not receive a direct context")
    if schedule is not None:
        raise ValueError("Ascend eager attention cannot consume a FlashMLA schedule")
    if query_chunk_size < 1:
        raise ValueError("query_chunk_size must be positive")
    sink = attn_sink[: q.shape[1]].contiguous()
    if out is None:
        out = torch.empty_like(q)
    elif out.shape != q.shape or out.dtype != q.dtype or out.device != q.device:
        raise ValueError("Attention output must match the query")
    if prefill_kv is not None:
        if (
            prefill_indices is None
            or prefill_kv.ndim != 3
            or prefill_kv.shape[1:] != (1, 512)
        ):
            raise ValueError("Compact prefill requires BF16 [rows,1,512] and indices")
        if any(x is not None for x in (swa_slots, swa_lens, global_slots, global_lens)):
            raise ValueError("Compact prefill cannot also supply paged selections")
        out.copy_(
            compact_attention(q, prefill_kv[:, 0], prefill_indices, sink, softmax_scale)
        )
        return out
    if prefill_indices is not None:
        raise ValueError("prefill_indices requires prefill_kv")
    segments = [(swa_cache, swa_slots, swa_lens, "swa_bf16")]
    if global_cache is None:
        if global_slots is not None or global_lens is not None:
            raise ValueError("Global selections require a global cache")
    else:
        segments.append((global_cache, global_slots, global_lens, "global_bf16"))
    for cache, slots, lens, fmt in segments:
        if cache is None or slots is None or lens is None:
            raise ValueError("Paged attention requires cache, slots and lengths")
        if cache.dtype != torch.bfloat16 or cache.shape[1:] != (64, 512):
            raise ValueError("Ascend attention requires BF16 [pages,64,512] KV")
        if (
            slots.ndim != 2
            or slots.shape[0] != q.shape[0]
            or lens.shape != (q.shape[0],)
        ):
            raise ValueError("Paged selections must have one row per query")
        if any(x.device != q.device for x in (cache, slots, lens)):
            raise ValueError("Paged selections must share the query device")
    if pa_mode == "direct":
        if pa_context.storage_ptr != swa_cache.untyped_storage().data_ptr():
            raise ValueError("Direct PA context does not own the supplied cache")
        from tokenspeed_kernel_npu.ops.dsv41_paged_attention import (
            selected_paged_attention,
        )

        for start in range(0, q.shape[0], query_chunk_size):
            end = min(start + query_chunk_size, q.shape[0])
            chunk_segments = [
                (cache, slots[start:end], lens[start:end])
                for cache, slots, lens, _ in segments
            ]
            out[start:end].copy_(
                selected_paged_attention(
                    q[start:end], chunk_segments, sink, softmax_scale, pa_context
                )
            )
        return out
    width = sum(slots.shape[1] for _, slots, _, _ in segments)
    if width == 0:
        return out.zero_()
    for start in range(0, q.shape[0], query_chunk_size):
        end = min(start + query_chunk_size, q.shape[0])
        parts, masks = [], []
        for cache, slots, lens, fmt in segments:
            ids = slots[start:end]
            valid = (
                (torch.arange(ids.shape[1], device=q.device) < lens[start:end, None])
                & (ids >= 0)
                & (ids < cache.shape[0] * 64)
            )
            parts.append(
                cache_ops.cache_gather(cache, ids.masked_fill(~valid, -1), fmt, None)
            )
            masks.append(valid)
        kv = torch.cat(parts, dim=1).reshape(-1, 512)
        valid = torch.cat(masks, dim=1)
        ids = torch.arange(
            (end - start) * width, device=q.device, dtype=torch.int32
        ).reshape(end - start, width)
        ids.masked_fill_(~valid, -1)
        out[start:end].copy_(
            compact_attention(q[start:end], kv, ids, sink, softmax_scale)
        )
    return out
