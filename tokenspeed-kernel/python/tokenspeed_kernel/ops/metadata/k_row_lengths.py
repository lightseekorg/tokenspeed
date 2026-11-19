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

"""Expand request lengths into a k-token verify's per-token rows in one launch.

A sparse-attention verify scores ``k`` query rows per request. The indexer
reads each row's request context length, and the sparse decode core reads each
row's causal KV length: verify row ``t`` sees ``length - (k - 1 - t)`` tokens,
clamped at zero. On CUDA (and ROCm) devices one Triton launch writes both;
other devices run the same arithmetic as tensor ops.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = ["expand_k_row_lengths"]

_BLOCK = 256


@triton.jit(do_not_specialize=["rows"])
def _expand_k_row_lengths_kernel(
    context_lengths_ptr,
    verify_lengths_ptr,
    context_out_ptr,
    causal_out_ptr,
    rows,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = row < rows
    request = row // K
    context = tl.load(context_lengths_ptr + request, mask=live, other=0)
    tl.store(
        context_out_ptr + row,
        context.to(context_out_ptr.dtype.element_ty),
        mask=live,
    )
    if causal_out_ptr is not None:
        verify = tl.load(verify_lengths_ptr + request, mask=live, other=0).to(tl.int32)
        causal = tl.maximum(verify + (row - request * K) - (K - 1), 0)
        tl.store(causal_out_ptr + row, causal, mask=live)


def expand_k_row_lengths(
    context_lengths: torch.Tensor,
    verify_lengths: torch.Tensor | None,
    *,
    k: int,
    context_out: torch.Tensor,
    causal_out: torch.Tensor | None,
) -> None:
    """Write every request's ``k`` per-token context and causal KV lengths.

    Args:
        context_lengths: ``[requests]`` int32 or int64 request context lengths.
        verify_lengths: ``[requests]`` int32 or int64 lengths the causal rows
            derive from, or None when ``causal_out`` is None.
        k: Query rows per request.
        context_out: ``[requests * k]`` int32 or int64 rows:
            ``context_out[r * k + t]`` is ``context_lengths[r]``.
        causal_out: ``[requests * k]`` int32 rows, ``causal_out[r * k + t]`` is
            ``max(verify_lengths[r] + t - (k - 1), 0)``; None writes only the
            context rows.

    Returns:
        None. The outputs are written in place.

    Raises:
        ValueError: A tensor is not the shape, dtype, or layout the expansion
            indexes with.
    """
    requests = context_lengths.shape[0]
    rows = requests * k
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    if (causal_out is None) != (verify_lengths is None):
        raise ValueError("causal rows need verify lengths, and only they do")
    lengths = [context_lengths] + ([] if verify_lengths is None else [verify_lengths])
    for lengths_in in lengths:
        if lengths_in.shape != (requests,) or lengths_in.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError(f"request lengths must be [{requests}] int32/int64")
    outputs = [context_out] + ([] if causal_out is None else [causal_out])
    for out in outputs:
        if out.numel() != rows or not out.is_contiguous():
            raise ValueError(f"per-token rows must be {rows} contiguous values")
    if context_out.dtype not in (torch.int32, torch.int64) or (
        causal_out is not None and causal_out.dtype != torch.int32
    ):
        raise ValueError("context rows are int32/int64 and causal rows int32")
    if any(t.device != context_lengths.device for t in lengths + outputs):
        raise ValueError("expand_k_row_lengths needs colocated tensors")
    if not all(t.stride(0) == 1 for t in lengths):
        raise ValueError("request lengths must be dense")
    if rows == 0:
        return
    if not context_lengths.is_cuda:
        context_out.view(requests, k).copy_(context_lengths.unsqueeze(1))
        if causal_out is not None:
            steps = torch.arange(k, device=verify_lengths.device, dtype=torch.int32)
            causal = verify_lengths.to(torch.int32).unsqueeze(1) + steps - (k - 1)
            causal_out.view(requests, k).copy_(causal.clamp_min(0))
        return

    _expand_k_row_lengths_kernel[(triton.cdiv(rows, _BLOCK),)](
        context_lengths,
        verify_lengths,
        context_out,
        causal_out,
        rows,
        K=k,
        BLOCK=_BLOCK,
        num_warps=4,
    )
