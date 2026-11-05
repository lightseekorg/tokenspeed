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
from dataclasses import dataclass, field
from weakref import WeakValueDictionary

import torch
from flashinfer.attention.prims_ts.mla_decode import BatchSparseMLADecodePagedTSWrapper
from tokenspeed_kernel.workspace import retain_for_capture


@dataclass
class _SparsePlan:
    wrapper: BatchSparseMLADecodePagedTSWrapper
    offsets: dict[int, torch.Tensor] = field(default_factory=dict)
    full_lengths: dict[int, torch.Tensor] = field(default_factory=dict)


# Scratch is per stream; captured plans live with their graphs, eager ones in an LRU.
_plans: OrderedDict[tuple, _SparsePlan] = OrderedDict()
_captured_plans: WeakValueDictionary[tuple, _SparsePlan] = WeakValueDictionary()
_MAX_EAGER_PLANS = 64


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
        batch_capacity = 1 << (max(1, rows // query_width) - 1).bit_length()
    else:
        batch_capacity = 1
        query_width = 1 << (max(1, rows) - 1).bit_length()
    stream = torch.cuda.current_stream(q.device)
    key = (
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
    plan = _captured_plans.get(key)
    if plan is None:
        plan = _plans.get(key)
    if plan is None:
        if capturing:
            raise RuntimeError("warm up Prims-TS DSA on the capture stream first")
        wrapper = BatchSparseMLADecodePagedTSWrapper()
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
        plan = _SparsePlan(wrapper)
        _plans[key] = plan
        if len(_plans) > _MAX_EAGER_PLANS:
            _plans.popitem(last=False)
    elif key in _plans:
        _plans.move_to_end(key)
    if rows not in plan.offsets:
        if capturing:
            raise RuntimeError("warm up the Prims-TS DSA row count before capture")
        plan.offsets[rows] = torch.tensor(
            [min(i * query_width, rows) for i in range(batch_capacity + 1)],
            device=q.device,
            dtype=torch.int32,
        )
        plan.full_lengths[rows] = torch.full(
            (rows,), topk, device=q.device, dtype=torch.int32
        )
    if capturing:
        retain_for_capture(plan)
        _captured_plans[key] = plan
        _plans.pop(key, None)
    return plan


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
    plan = _get_plan(
        q,
        topk=topk_slots.shape[1],
        mode=mode,
        query_width=query_width,
        return_lse=return_lse,
    )
    lengths = plan.full_lengths[rows] if topk_lens is None else topk_lens
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
    plan.wrapper.run(
        q,
        kv_cache,
        topk_slots,
        lengths,
        qo_indptr=plan.offsets[rows],
        softmax_scale=softmax_scale,
        output_scale=1.0,
        kv_cache_scale=1.0,
        out=output.view(rows, heads, 512),
        lse=lse,
        validate=False,
    )
    return (output, lse) if return_lse else output
