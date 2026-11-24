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

"""SM120 V4.1 prefill over protected SWA rows and the original FP4 pages."""

import math

import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.attention.dsv41.triton import cache_scatter
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from tokenspeed_kernel.thirdparty.flashinfer.sparse_mla_sm120 import (
    sm120_bf16_mla_runner,
)


@register_kernel(
    "attention",
    "dsv41_paged_prefill",
    name="flashinfer_sm120_dsv41_paged_prefill",
    solution="flashinfer_sm120",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(12, 0),
        max_arch_version=ArchVersion(12, 1),
    ),
    signatures=frozenset({format_signature(x=dense_tensor_format(torch.bfloat16))}),
    priority=Priority.SPECIALIZED,
)
def paged_prefill(
    q,
    swa_prefix,
    swa_current,
    swa_indices,
    global_cache,
    global_slots,
    global_lens,
    attn_sink,
    softmax_scale,
    out,
    lse,
):
    """Run full-BF16 QK/PV with FP32 softmax; never requantize FP4 history."""
    if q.ndim != 3 or q.shape[-1] != 512 or not 1 <= q.shape[1] <= 128:
        raise ValueError("q must be BF16 [tokens, 1..128 heads, 512]")
    if q.dtype != torch.bfloat16:
        raise ValueError("q must be BF16")
    for rows in (swa_prefix, swa_current):
        if rows.ndim != 2 or rows.shape[1] != 512 or rows.dtype != q.dtype:
            raise ValueError("SWA rows must be codec-rounded BF16 [rows, 512]")
    if swa_indices.ndim != 2 or swa_indices.shape[0] != q.shape[0]:
        raise ValueError("SWA indices must have one row per query")
    if swa_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("SWA indices must be integer")
    if global_cache is None:
        if global_slots is not None or global_lens is not None:
            raise ValueError("absent global cache requires absent global selection")
    else:
        if (
            global_cache.ndim != 3
            or global_cache.dtype != torch.uint8
            or global_cache.shape[1:] != (64, 288)
            or global_cache.stride(1) != 288
            or global_cache.stride(2) != 1
            or global_cache.stride(0) % 16
            or global_cache.data_ptr() % 16
        ):
            raise ValueError("global cache must be aligned page-planar V4.1 FP4")
        if (
            global_slots is None
            or global_lens is None
            or global_slots.ndim != 2
            or global_slots.shape[0] != q.shape[0]
            or global_lens.shape != q.shape[:1]
            or global_slots.dtype not in (torch.int32, torch.int64)
            or global_lens.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("global slots/lengths must be integer [T,K]/[T]")
    if attn_sink is not None:
        if (
            attn_sink.dtype != torch.float32
            or attn_sink.ndim != 1
            or attn_sink.numel() < q.shape[1]
        ):
            raise ValueError("attention sink must provide FP32 logits per head")
        attn_sink = attn_sink[: q.shape[1]].contiguous()
    if out is not None and (out.shape != q.shape or out.dtype != q.dtype):
        raise ValueError("output must match query shape and dtype")
    if lse is not None and (lse.shape != q.shape[:2] or lse.dtype != torch.float32):
        raise ValueError("LSE must be FP32 [T,H]")
    tensors = (
        swa_prefix,
        swa_current,
        swa_indices,
        global_cache,
        global_slots,
        global_lens,
        attn_sink,
        out,
        lse,
    )
    if any(t is not None and t.device != q.device for t in tensors):
        raise ValueError("all prefill tensors must share the query device")
    output = torch.empty_like(q) if out is None else out
    if q.shape[0] == 0:
        return output
    swa_rows = torch.cat((swa_prefix, swa_current), dim=0)
    n = swa_rows.shape[0]
    # Old SWA rows were read before the model overwrote its bounded ring.
    # Repacking preserves decoded BF16 values (scales may change); FP4 is read
    # directly, with no FP4 -> FP8 precision change.
    cache = torch.empty(((n + 63) // 64, 64, 528), dtype=torch.uint8, device=q.device)
    cache_scatter(swa_rows, cache, torch.arange(n, device=q.device), "swa")
    runner = sm120_bf16_mla_runner(
        q.device.index,
        torch.cuda.current_stream(q.device).cuda_stream,
        global_cache is not None,
    )
    tokens = q.shape[0]
    # Keep every extend on the same BF16 prefill route. The library otherwise
    # switches <=64 rows to split decode, whose PV rounding differs across
    # chunk sizes. Padding is masked and applies only to short extends.
    padding = max(0, 65 - tokens)
    native_q = F.pad(q, (0, 0, 0, 0, 0, padding)) if padding else q.contiguous()
    native_si = swa_indices.to(torch.int32).contiguous()
    native_gi = (
        None if global_slots is None else global_slots.to(torch.int32).contiguous()
    )
    native_gl = (
        None if global_lens is None else global_lens.to(torch.int32).contiguous()
    )
    if padding:
        native_si = F.pad(native_si, (0, 0, 0, padding), value=-1)
        if native_gi is not None:
            native_gi = F.pad(native_gi, (0, 0, 0, padding), value=-1)
            native_gl = F.pad(native_gl, (0, padding), value=0)
    native_out = torch.empty_like(native_q) if padding else output
    native_lse = (
        torch.empty(native_q.shape[:2], device=q.device, dtype=torch.float32)
        if padding and lse is not None
        else lse
    )
    runner.run(
        native_q,
        cache,
        native_si,
        native_out,
        softmax_scale,
        topk_length=None,
        attn_sink=attn_sink,
        extra_kv_cache=global_cache,
        extra_indices=native_gi,
        extra_topk_length=native_gl,
        out_lse=native_lse,
        lse_scale=math.log(2),
    )
    if padding:
        output.copy_(native_out[:tokens])
        if lse is not None:
            lse.copy_(native_lse[:tokens])
    return output
