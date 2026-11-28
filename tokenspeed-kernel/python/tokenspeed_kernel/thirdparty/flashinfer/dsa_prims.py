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

"""Prims-TS sparse MLA over prepared FP8 queries and native latent KV."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from weakref import WeakValueDictionary

import torch
from tokenspeed_kernel.ops.tuning import is_warming_up
from tokenspeed_kernel.workspace import retain_for_capture

if TYPE_CHECKING:
    from flashinfer.attention.prims_ts.mla_decode import (
        BatchSparseMLADecodePagedTSWrapper,
    )

_WORKSPACE_REGION: Callable[[], AbstractContextManager[object]] = nullcontext


def set_dsa_prims_workspace_region(
    region: Callable[[], AbstractContextManager[object]],
) -> None:
    """Allocate later persistent Prims-TS DSA plan tensors inside ``region()``.

    Both warmed plans and eager LRU plans hold persistent device scratch and
    offset tables across calls. A runtime that releases GPU memory registers a
    CPU-backed named workspace region once at startup so ``weights``
    release/resume unmaps and restores these buffers with zero bytes pinned in
    ordinary Torch CUDA allocator segments.

    Args:
        region: Factory of the allocation scope entered around plan creation,
            persistent offset/length tables, and each plan's first eager run.
            CUDA graph capture never enters this scope.
    """
    global _WORKSPACE_REGION
    _WORKSPACE_REGION = region


def _make_wrapper() -> BatchSparseMLADecodePagedTSWrapper:
    from flashinfer.attention.prims_ts.mla_decode import (
        BatchSparseMLADecodePagedTSWrapper,
    )

    return BatchSparseMLADecodePagedTSWrapper()


@dataclass
class _SparsePlan:
    wrapper: Any
    base_offsets: torch.Tensor
    offsets: torch.Tensor
    full_lengths: torch.Tensor
    run_initialized: bool = False


# Scratch is per stream; captured plans live with their graphs, warmed plans
# live outside the eager LRU so serving traffic cannot evict a warmed bucket.
_plans: OrderedDict[tuple, _SparsePlan] = OrderedDict()
_warmed_plans: dict[tuple, _SparsePlan] = {}
_captured_plans: WeakValueDictionary[tuple, _SparsePlan] = WeakValueDictionary()
# Highest power-of-two capacity warmed per static launch configuration, so
# repeated model layers do not re-run the warmup ladder.
_warmed_buckets: dict[tuple, int] = {}
_MAX_EAGER_PLANS = 64


def _next_power_of_two(n: int) -> int:
    return 1 << (max(1, n) - 1).bit_length()


def _plan_key(
    device_index: int | None,
    cuda_stream: int,
    dtype: torch.dtype,
    heads: int,
    topk: int,
    batch_capacity: int,
    query_width: int,
    return_lse: bool,
) -> tuple:
    return (
        device_index,
        cuda_stream,
        dtype,
        heads,
        topk,
        batch_capacity,
        query_width,
        return_lse,
    )


def _get_plan(
    q: torch.Tensor,
    *,
    topk: int,
    mode: str,
    query_width: int,
    return_lse: bool,
) -> _SparsePlan:
    rows, heads, _ = q.shape
    if mode == "decode":
        batch_capacity = _next_power_of_two(max(1, rows // query_width))
    else:
        batch_capacity = 1
        query_width = _next_power_of_two(rows)
    stream = torch.cuda.current_stream(q.device)
    key = _plan_key(
        q.device.index,
        stream.cuda_stream,
        q.dtype,
        heads,
        topk,
        batch_capacity,
        query_width,
        return_lse,
    )
    capturing = torch.cuda.is_current_stream_capturing()
    warming = is_warming_up() and not capturing
    plan = _captured_plans.get(key)
    if plan is None:
        plan = _warmed_plans.get(key)
    if plan is None:
        plan = _plans.get(key)
    if plan is None:
        if capturing:
            raise RuntimeError("warm up Prims-TS DSA on the capture stream first")
        with _WORKSPACE_REGION():
            wrapper = _make_wrapper()
            wrapper.plan(
                q.device,
                batch_capacity,
                heads,
                max_topk=topk,
                max_extra_topk=0,
                max_q_len=query_width,
                packed_query=True,
                q_dtype=q.dtype,
                qk_rope_head_dim=64,
                kv_cache_format="native",
                extra_kv_cache_format="native",
                kv_layout="NHD",
                use_attention_sinks=False,
                return_lse=return_lse,
                assume_valid_prefix=False,
            )
            base_offsets = (
                torch.arange(batch_capacity + 1, device=q.device, dtype=torch.int32)
                * query_width
            )
            offsets = base_offsets.clone()
            full_lengths = torch.full(
                (batch_capacity * query_width,),
                topk,
                device=q.device,
                dtype=torch.int32,
            )
        plan = _SparsePlan(
            wrapper=wrapper,
            base_offsets=base_offsets,
            offsets=offsets,
            full_lengths=full_lengths,
        )
        if warming:
            _warmed_plans[key] = plan
        else:
            _plans[key] = plan
            if len(_plans) > _MAX_EAGER_PLANS:
                _plans.popitem(last=False)
    elif warming and key not in _warmed_plans and key not in _captured_plans:
        _warmed_plans[key] = plan
        _plans.pop(key, None)
    elif key in _plans:
        _plans.move_to_end(key)
    if capturing:
        retain_for_capture(plan)
        _captured_plans[key] = plan
        _plans.pop(key, None)
        _warmed_plans.pop(key, None)
    return plan


def _run_plan(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor | None,
    *,
    mode: str,
    query_width: int,
    softmax_scale: float,
    return_lse: bool,
    output_3d: torch.Tensor,
    lse: torch.Tensor | None,
) -> None:
    rows = q.shape[0]
    plan = _get_plan(
        q,
        topk=topk_slots.shape[1],
        mode=mode,
        query_width=query_width,
        return_lse=return_lse,
    )
    # Clamp preallocated offsets on-device for partial buckets so unseen row
    # counts never allocate persistent tensors or trigger H2D syncs.
    max_rows = plan.full_lengths.shape[0]
    qo_indptr = (
        plan.base_offsets
        if rows == max_rows
        else torch.clamp(plan.base_offsets, max=rows, out=plan.offsets)
    )
    lengths = plan.full_lengths[:rows] if topk_lens is None else topk_lens
    if not plan.run_initialized and not torch.cuda.is_current_stream_capturing():
        with _WORKSPACE_REGION():
            plan.wrapper.run(
                q,
                kv_cache,
                topk_slots,
                lengths,
                qo_indptr=qo_indptr,
                softmax_scale=softmax_scale,
                output_scale=1.0,
                kv_cache_scale=1.0,
                out=output_3d,
                lse=lse,
                validate=False,
            )
        plan.run_initialized = True
        return
    plan.wrapper.run(
        q,
        kv_cache,
        topk_slots,
        lengths,
        qo_indptr=qo_indptr,
        softmax_scale=softmax_scale,
        output_scale=1.0,
        kv_cache_scale=1.0,
        out=output_3d,
        lse=lse,
        validate=False,
    )


def _warm_smaller_buckets(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor | None,
    *,
    mode: str,
    query_width: int,
    softmax_scale: float,
    return_lse: bool,
    output_3d: torch.Tensor,
    lse: torch.Tensor | None,
) -> tuple[tuple, int]:
    """Plan and run every power-of-two bucket below this launch's bucket.

    FlashInfer's Prims-TS wrapper selects and JIT-compiles different CuTe DSL
    kernels across power-of-two ``max_q_len`` (prefill) and ``batch_capacity``
    (decode) buckets. Startup warmup runs one launch at the peak token count,
    so every strictly smaller bucket ``bucket < max_bucket`` satisfies
    ``sub_rows < rows`` and reuses contiguous prefix slices of the caller's
    buffers with no extra scratch allocations.
    """
    rows, heads, _ = q.shape
    topk = topk_slots.shape[1]
    stream = torch.cuda.current_stream(q.device)
    if mode == "decode":
        max_bucket = _next_power_of_two(max(1, rows // query_width))
        fixed_width = query_width
    else:
        max_bucket = _next_power_of_two(rows)
        fixed_width = 0
    warm_key = (
        q.device.index,
        stream.cuda_stream,
        mode,
        q.dtype,
        heads,
        topk,
        fixed_width,
        return_lse,
    )
    prev = _warmed_buckets.get(warm_key, 0)
    bucket = 1
    while bucket < max_bucket:
        batch_cap = bucket if mode == "decode" else 1
        q_width = query_width if mode == "decode" else bucket
        key = _plan_key(
            q.device.index,
            stream.cuda_stream,
            q.dtype,
            heads,
            topk,
            batch_cap,
            q_width,
            return_lse,
        )
        if bucket > prev or (key not in _warmed_plans and key not in _captured_plans):
            sub_rows = bucket * query_width if mode == "decode" else bucket
            _run_plan(
                q[:sub_rows],
                kv_cache,
                topk_slots[:sub_rows],
                None if topk_lens is None else topk_lens[:sub_rows],
                mode=mode,
                query_width=query_width,
                softmax_scale=softmax_scale,
                return_lse=return_lse,
                output_3d=output_3d[:sub_rows],
                lse=None if lse is None else lse[:sub_rows],
            )
        bucket <<= 1
    return warm_key, max(prev, max_bucket)


def sparse_attention(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor | None,
    *,
    mode: str,
    query_width: int,
    softmax_scale: float,
    return_lse: bool,
    out: torch.Tensor | None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run native E4M3 D576 selected attention with BF16 output.

    Args:
        q: Prepared contiguous E4M3 queries [tokens, heads, 576].
        kv_cache: Contiguous native E4M3 cache [slots, 576].
        topk_slots: INT32 physical slot IDs [tokens, topk]; -1 marks holes.
        topk_lens: Visible prefix lengths, or None to inspect the full width.
        mode: Explicit decode or prefill scheduling policy.
        query_width: Query rows per request for decode.
        softmax_scale: Combined QK scale, including the key multiplier.
        return_lse: Return natural-log FP32 LSE when true.
        out: Optional contiguous BF16 output with tokens * heads * 512 elements.

    Returns:
        Output [tokens, heads, 512], preserving a supplied output's identity,
        and optionally LSE [tokens, heads]. No query or cache quantization occurs.
    """
    rows, heads, _ = q.shape
    output = (
        torch.empty((rows, heads, 512), device=q.device, dtype=torch.bfloat16)
        if out is None
        else out
    )
    lse = (
        torch.empty((rows, heads), device=q.device, dtype=torch.float32)
        if return_lse
        else None
    )
    output_3d = output.view(rows, heads, 512)
    warm_state: tuple[tuple, int] | None = None
    if is_warming_up() and not torch.cuda.is_current_stream_capturing():
        warm_state = _warm_smaller_buckets(
            q,
            kv_cache,
            topk_slots,
            topk_lens,
            mode=mode,
            query_width=query_width,
            softmax_scale=softmax_scale,
            return_lse=return_lse,
            output_3d=output_3d,
            lse=lse,
        )
    _run_plan(
        q,
        kv_cache,
        topk_slots,
        topk_lens,
        mode=mode,
        query_width=query_width,
        softmax_scale=softmax_scale,
        return_lse=return_lse,
        output_3d=output_3d,
        lse=lse,
    )
    if warm_state is not None:
        warm_key, max_bucket = warm_state
        _warmed_buckets[warm_key] = max_bucket
    return (output, lse) if return_lse else output
