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

"""Optional native V4.1 selected attention over the shared cache contract."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from tokenspeed_kernel.thirdparty.flash_mla import flash_mla_api


def new_flashmla_schedule() -> object:
    """Create caller-owned metadata; capture must record its first preparation."""
    return flash_mla_api().get_mla_metadata()[0]


def _native_query(q, sink):
    if q.ndim != 3 or q.shape[-1] != 512 or q.dtype != torch.bfloat16:
        raise ValueError("V4.1 query must be BF16 [tokens, heads, 512]")
    heads = q.shape[1]
    if not 1 <= heads <= 128:
        raise ValueError("FlashMLA V4.1 requires 1..128 query heads")
    padded = 64 if heads <= 64 else 128
    if (
        sink.shape not in ((heads,), (padded,))
        or sink.dtype != torch.float32
        or sink.device != q.device
    ):
        raise ValueError(
            "attention sink must be FP32 [query heads] on the query device"
        )
    if padded == heads:
        return q.contiguous(), sink.contiguous()
    return (
        F.pad(q, (0, 0, 0, padded - heads)).contiguous(),
        (
            sink
            if sink.numel() == padded
            else F.pad(sink, (0, padded - heads), value=-float("inf"))
        ).contiguous(),
    )


def _paged(cache, row_bytes, stride_alignment):
    if (
        cache.dtype != torch.uint8
        or cache.ndim != 3
        or cache.shape[1:] != (64, row_bytes)
        or cache.stride(1) != row_bytes
        or cache.stride(2) != 1
    ):
        raise ValueError(
            "FlashMLA requires contiguous bytes within each page-planar field"
        )
    if cache.stride(0) < 64 * row_bytes or cache.stride(0) % stride_alignment:
        raise ValueError("FlashMLA page stride has incompatible alignment")
    if cache.data_ptr() % 16:
        raise ValueError("FlashMLA cache base must be 16-byte aligned")
    return cache.as_strided(
        (cache.shape[0], 64, 1, row_bytes), (cache.stride(0), row_bytes, row_bytes, 1)
    )


def _slots(slots, lengths, tokens):
    if slots is None or lengths is None:
        raise ValueError("paged attention requires slots and lengths")
    if (
        slots.ndim != 2
        or slots.shape[0] != tokens
        or slots.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("paged slots must be integer [tokens, capacity]")
    if lengths.shape != (tokens,) or lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("paged lengths must be integer [tokens]")
    width = max(64, (slots.shape[1] + 63) // 64 * 64)
    slots = slots.to(torch.int32)
    if width != slots.shape[1]:
        slots = F.pad(slots, (0, width - slots.shape[1]), value=-1)
    return slots.contiguous().unsqueeze(1), lengths.to(torch.int32).contiguous()


@register_kernel(
    "attention",
    "dsv41_selected_attention",
    name="flashmla_dsv41_selected_attention",
    solution="flashmla",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=frozenset({format_signature(x=dense_tensor_format(torch.bfloat16))}),
    traits={"flashmla_eligible": frozenset({True})},
    priority=Priority.SPECIALIZED,
)
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
):
    """Use a caller-built prefill workspace or native paged decode, returning BF16.

    Persistent fields and schedules stay caller-owned. Native output views are
    retained when out is None; a supplied destination is filled in place.
    """
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    if (prefill_kv is None) != (prefill_indices is None):
        raise ValueError("prefill_kv and prefill_indices must be supplied together")
    prefill = prefill_kv is not None
    inputs = (
        swa_cache,
        swa_slots,
        swa_lens,
        global_cache,
        global_slots,
        global_lens,
        out,
        prefill_kv,
        prefill_indices,
    )
    if any(t is not None and t.device != q.device for t in inputs):
        raise ValueError("all selected-attention tensors must share the query device")
    if out is not None and (out.shape != q.shape or out.dtype != q.dtype):
        raise ValueError("out must match query shape and dtype")
    if q.shape[0] == 0:
        return q if out is None else out
    api = flash_mla_api()
    if prefill:
        if prefill_kv.dtype != torch.bfloat16 or prefill_kv.shape[-1] != 512:
            raise ValueError("prefill_kv must contain BF16 512-dimensional rows")
        if prefill_kv.ndim == 2:
            prefill_kv = prefill_kv[:, None, :]
        if prefill_kv.ndim != 3 or prefill_kv.shape[1] != 1:
            raise ValueError("prefill_kv must have shape [rows, 1, 512]")
        if prefill_indices.ndim != 2 or prefill_indices.shape[0] != q.shape[0]:
            raise ValueError("prefill_indices must have shape [tokens, capacity]")
        if prefill_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("prefill_indices must be integer")
        indices = prefill_indices.to(torch.int32)
        width = max(64, (indices.shape[1] + 63) // 64 * 64)
        if width != indices.shape[1]:
            indices = F.pad(indices, (0, width - indices.shape[1]), value=-1)
        destination = out
        if destination is None and q.shape[0] > query_chunk_size:
            destination = torch.empty_like(q)
        for start in range(0, q.shape[0], query_chunk_size):
            end = min(start + query_chunk_size, q.shape[0])
            q_native, sink_native = _native_query(q[start:end], attn_sink)
            result, _, _ = api.flash_mla_sparse_fwd(
                q=q_native,
                kv=prefill_kv,
                indices=indices[start:end].contiguous().unsqueeze(1),
                sm_scale=softmax_scale,
                d_v=512,
                attn_sink=sink_native,
                topk_length=None,
            )
            if destination is None:
                return result[:, : q.shape[1], :]
            destination[start:end].copy_(result[:, : q.shape[1], :])
            # Do not keep the previous tile's padded output alive while the next
            # native call allocates its Q/output pair from the same stream pool.
            del result, q_native
        return destination
    else:
        q_native, sink_native = _native_query(q, attn_sink)
        if schedule is None:
            raise ValueError("native decode requires a caller-owned schedule")
        if (global_cache is None) != (global_slots is None) or (
            global_cache is None
        ) != (global_lens is None):
            raise ValueError(
                "global cache, slots and lengths must be supplied together"
            )
        indices, lengths = _slots(swa_slots, swa_lens, q.shape[0])
        extra_indices, extra_lengths = (
            (None, None)
            if global_cache is None
            else _slots(global_slots, global_lens, q.shape[0])
        )
        result, _ = api.flash_mla_with_kvcache(
            q=q_native.unsqueeze(1),
            k_cache=_paged(swa_cache, 528, 512),
            block_table=None,
            cache_seqlens=None,
            head_dim_v=512,
            tile_scheduler_metadata=schedule,
            num_splits=None,
            softmax_scale=softmax_scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=indices,
            attn_sink=sink_native,
            extra_k_cache=(
                None if global_cache is None else _paged(global_cache, 288, 256)
            ),
            extra_indices_in_kvcache=extra_indices,
            topk_length=lengths,
            extra_topk_length=extra_lengths,
        )
        result = result.squeeze(1)
    result = result[:, : q.shape[1], :]
    if out is None:
        return result
    out.copy_(result)
    return out
