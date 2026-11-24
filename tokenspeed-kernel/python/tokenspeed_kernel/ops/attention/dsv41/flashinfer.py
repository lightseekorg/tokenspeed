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

"""SM120 selected attention over the existing V4.1 page-planar caches."""

import math
import os

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


class FlashInferSchedule:
    """Owner of warmed plans and graph scratch for SWA-only and dual-cache calls."""

    def __init__(self, wrapper):
        self.swa = wrapper(
            kv_scale_format="ue8m0_g32",
            kv_cache_format="fp8",
            extra_kv_fp4=False,
            compute_precision="bf16",
            backend="sparse",
        )
        self.dual = wrapper(
            kv_scale_format="ue8m0_g32",
            kv_cache_format="fp8",
            extra_kv_fp4=True,
            compute_precision="bf16",
            backend="sparse",
        )

        self.cake_decode = None
        self.cake_rows = frozenset({48})
        if os.environ.get("TOKENSPEED_SM120_CAKE_DECODE_MLA") == "1":
            if os.environ.get("TOKENSPEED_SM120_CAKE_DECODE_MLA_WIDE") == "1":
                self.cake_rows = frozenset({24, 48, 96, 192})
            self.cake_decode = wrapper(
                kv_scale_format="ue8m0_g32",
                kv_cache_format="fp8",
                extra_kv_fp4=True,
                compute_precision="bf16",
                backend="cake",
            )

    def run(self, *args, **kwargs):
        runner = self.swa if kwargs["extra_kv_cache"] is None else self.dual
        if (
            self.cake_decode is not None
            and runner is self.dual
            and args[0].shape[0] in self.cake_rows
            and args[0].shape[1:] == (64, 512)
            and kwargs.get("out_lse") is None
        ):
            runner = self.cake_decode
        return runner.run(*args, **kwargs)


def new_flashinfer_schedule():
    """Create a caller-owned runner; captured callers retain its scratch buffers."""
    try:
        from flashinfer.mla import SparseMLASm120Wrapper
    except ImportError:
        return None
    return FlashInferSchedule(SparseMLASm120Wrapper)


def supports_cache(cache, row_bytes):
    """Whether a TS field can be passed without repacking its page bytes."""
    return (
        cache.ndim == 3
        and cache.dtype == torch.uint8
        and cache.shape[1:] == (64, row_bytes)
        and cache.stride(1) == row_bytes
        and cache.stride(2) == 1
        and cache.stride(0) >= 64 * row_bytes
        and cache.stride(0) % 16 == 0
        and cache.data_ptr() % 16 == 0
    )


@triton.jit
def _sanitize_slots(S, O, ROWS, COLS, S0, S1, LIMIT, BLOCK: tl.constexpr):
    x = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = x < ROWS.to(tl.int64) * COLS
    value = tl.load(S + (x // COLS) * S0 + (x % COLS) * S1, valid, other=-1)
    value = tl.where((value >= 0) & (value < LIMIT), value, -1)
    tl.store(O + x, value.to(tl.int32), valid)


def _indices(slots, cache):
    from tokenspeed_kernel.ops.attention.dsv41._cute_dsl.selection import enabled

    if enabled() and slots.ndim == 2:
        out = torch.empty(slots.shape, device=slots.device, dtype=torch.int32)
        if slots.numel():
            _sanitize_slots[(triton.cdiv(slots.numel(), 256),)](
                slots, out, *slots.shape, *slots.stride(), cache.shape[0] * 64, 256
            )
        return out
    # Keep the portable op's out-of-range-slot semantics before native reads.
    return slots.to(torch.int32).masked_fill(
        (slots < 0) | (slots >= cache.shape[0] * 64), -1
    )


@register_kernel(
    "attention",
    "dsv41_selected_attention",
    name="flashinfer_dsv41_selected_attention",
    solution="flashinfer",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(12, 0),
        max_arch_version=ArchVersion(12, 0),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=frozenset({format_signature(x=dense_tensor_format(torch.bfloat16))}),
    traits={"flashinfer_eligible": frozenset({True})},
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
    *,
    lse=None,
):
    """Compute joint SWA/global attention without materializing BF16 KV rows."""
    if schedule is None or prefill_kv is not None or prefill_indices is not None:
        raise ValueError(
            "FlashInfer selected attention requires paged inputs and a runner"
        )
    if q.ndim != 3 or q.shape[-1] != 512 or not 1 <= q.shape[1] <= 128:
        raise ValueError("query must be BF16 [tokens, heads, 512], with 1..128 heads")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    if not supports_cache(swa_cache, 528):
        raise ValueError("SWA cache must have aligned V4.1 FP8 page-planar storage")
    if (global_cache is None) != (global_slots is None) or (global_cache is None) != (
        global_lens is None
    ):
        raise ValueError("global cache, slots and lengths must be supplied together")
    if global_cache is not None and not supports_cache(global_cache, 288):
        raise ValueError("global cache must have aligned V4.1 FP4 page-planar storage")
    if out is None:
        out = torch.empty_like(q, memory_format=torch.contiguous_format)
    elif out.shape != q.shape or out.dtype != q.dtype or not out.is_contiguous():
        raise ValueError("output must be contiguous and match query shape and dtype")
    if q.shape[0] == 0:
        return out
    if lse is not None and attn_sink is not None:
        raise ValueError("partial attention LSE requires no sink")
    implementation = os.environ.get("TOKENSPEED_EXPERIMENT_SM120_SELECTED_ATTENTION")
    if implementation not in (None, "flashinfer", "gather"):
        raise ValueError(
            "TOKENSPEED_EXPERIMENT_SM120_SELECTED_ATTENTION must be flashinfer or gather"
        )
    if implementation == "gather":
        from tokenspeed_kernel.ops.attention.dsv41._triton.gather_attention import (
            selected_attention_gather,
        )

        return selected_attention_gather(
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
            lse,
        )
    schedule.run(
        q.contiguous(),
        swa_cache,
        _indices(swa_slots, swa_cache),
        out,
        sm_scale=softmax_scale,
        topk_length=swa_lens.to(torch.int32),
        attn_sink=None if attn_sink is None else attn_sink[: q.shape[1]].contiguous(),
        extra_kv_cache=global_cache,
        extra_indices=(
            None if global_cache is None else _indices(global_slots, global_cache)
        ),
        extra_topk_length=None if global_lens is None else global_lens.to(torch.int32),
        out_lse=lse,
        lse_scale=math.log(2),
    )
    return out
