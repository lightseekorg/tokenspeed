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

"""Unified LoRA-B expand for prefill batches (chunked-SGMV style).

Replaces the three separate ``lora_expand`` / ``lora_qkv_expand`` /
``lora_gate_up_expand`` kernels for the prefill path.  A single kernel
handles any number of output slices via the ``NUM_SLICES`` constexpr and a
``slice_offsets`` boundary tensor — the same trick as sglang's
``chunked_sgmv_expand`` (PR sgl-project/sglang#20391).

Key structural difference from the decode-path expand kernels:
* ``OUTPUT_DIM``, ``MAX_RANK``, ``NUM_SLICES`` are **constexpr** — the
  compiler specialises the K-loop trip count and all strides at compile
  time, which gives 2–3× speedup over runtime-stride kernels at prefill
  with rank ≥ 64.
* x strides are derived as compile-time constants:
  ``x_stride_0 = NUM_SLICES * MAX_RANK``, ``x_stride_1 = 1``.

Use :func:`lora_expand_fwd` / :func:`lora_qkv_expand_fwd` /
:func:`lora_gate_up_expand_fwd` for decode (``max_len ≤ 32``); switch to
:func:`lora_expand_prefill_fwd` for prefill.

Adapted from sglang ``python/sglang/srt/lora/triton_ops/chunked_sgmv_expand.py``
(previously ``chunked_sgmv_expand.py`` in this repo)
(Apache-2.0): https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/triton_ops/chunked_sgmv_expand.py.
Local changes: merged SORTED_BY_ADAPTER from our decode kernels (avoids
permutation overhead for unsorted batches), replaced fixed configs with
``@triton.autotune`` + on-disk cache, constexpr ordering.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.lora.triton.kernel_utils import _resolve_token_positions
from tokenspeed_kernel.ops.lora.triton.tuning import load_kernel_cache

_PREFILL_EXPAND_CONFIGS = [
    triton.Config(
        {"BLOCK_S": s, "BLOCK_N": n, "BLOCK_K": k},
        num_warps=w,
        num_stages=stages,
        maxnreg=mr,
    )
    for s in (16, 32)
    for n in (32, 64, 128)
    for k in (16, 32)
    for w in (4, 8)
    for stages in (1, 2, 3)
    for mr in (None, 128, 160)
]


@triton.autotune(
    configs=_PREFILL_EXPAND_CONFIGS,
    key=["OUTPUT_DIM", "MAX_RANK", "NUM_SLICES"],
    restore_value=["output"],
)
@triton.jit(do_not_specialize=["output_stride_0", "output_stride_1"])
def _lora_expand_prefill_kernel(
    x,
    weights,
    output,
    output_stride_0,
    output_stride_1,
    seg_lens,
    seg_indptr,
    weight_indices,
    lora_ranks,
    sorted_token_ids,
    scalings,
    slice_offsets,
    NUM_SLICES: tl.constexpr,
    OUTPUT_DIM: tl.constexpr,
    MAX_RANK: tl.constexpr,
    SORTED_BY_ADAPTER: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Constexpr strides — compiler eliminates all stride multiplications.
    x_stride_0: tl.constexpr = NUM_SLICES * MAX_RANK
    x_stride_1: tl.constexpr = 1
    w_stride_0: tl.constexpr = OUTPUT_DIM * MAX_RANK
    w_stride_1: tl.constexpr = MAX_RANK
    w_stride_2: tl.constexpr = 1

    batch_id = tl.program_id(axis=2)
    w_index = tl.load(weight_indices + batch_id)
    if w_index < 0:
        return
    rank = tl.load(lora_ranks + w_index)
    if rank == 0:
        return

    slice_id = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)
    seg_len = tl.load(seg_lens + batch_id)
    if seg_len == 0:
        return
    seg_start = tl.load(seg_indptr + batch_id)
    slice_start = tl.load(slice_offsets + slice_id)
    slice_end = tl.load(slice_offsets + slice_id + 1)
    n_size = slice_end - slice_start
    scaling = tl.load(scalings + w_index)
    K = tl.minimum(MAX_RANK, rank)

    num_pid_n = tl.cdiv(n_size, BLOCK_N)
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n
    if pid_s * BLOCK_S >= seg_len:
        return

    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)

    s_physical = _resolve_token_positions(
        sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER
    )

    # x: slice i starts at column i * K (actual rank, not MAX_RANK).
    x_ptrs = (
        x
        + slice_id * K * x_stride_1
        + (s_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1)
    )
    w_ptrs = (weights + w_index * w_stride_0 + slice_start * w_stride_1) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )

    n_mask = n_offset[None, :] < n_size
    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset[:, None] < seg_len) & (k_offset[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < K - k * BLOCK_K) & n_mask,
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)
        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    partial_sum *= scaling
    partial_sum = partial_sum.to(x.dtype.element_ty)

    output_ptr = output + (
        s_physical[:, None] * output_stride_0
        + (slice_start + n_offset)[None, :] * output_stride_1
    )
    output_mask = (s_offset[:, None] < seg_len) & n_mask
    partial_sum += tl.load(output_ptr, mask=output_mask, other=0.0)
    tl.store(output_ptr, partial_sum, mask=output_mask)


def lora_expand_prefill_fwd(
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info,
    slice_offsets: torch.Tensor,
    max_slice_size: int,
    base_output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Prefill-optimised LoRA-B expand for one or more output slices.

    Covers all projection types via ``slice_offsets``:
    * plain expand (o/down):  ``slice_offsets = [0, out_dim]``
    * gate/up:                ``slice_offsets = [0, inter, 2*inter]``
    * QKV:                    ``slice_offsets = [0, q, q+kv, q+2*kv]``

    Args:
        x:             ``(s, num_slices * max_rank)`` from lora_shrink.
        weights:       ``(num_lora, out_dim, max_rank)``, contiguous.
        batch_info:    :class:`LoraBatchInfo`.
        slice_offsets: ``(num_slices + 1,)`` int32 boundary tensor.
        max_slice_size: largest ``slice_offsets[i+1] - slice_offsets[i]``.
        base_output:   ``(s, out_dim)`` to fuse-add into; allocated if None.

    Returns:
        ``(s, out_dim)`` (same buffer as ``base_output`` when supplied).
    """
    assert x.is_contiguous()
    assert weights.is_contiguous()
    assert x.dim() == 2
    assert weights.dim() == 3

    S = x.shape[0]
    OUT_DIM = weights.shape[-2]
    MAX_RANK = weights.shape[-1]
    num_slices = len(slice_offsets) - 1
    assert x.shape[1] == num_slices * MAX_RANK

    max_len = batch_info.max_len
    sorted_by_adapter = batch_info.permutation is not None

    def grid(meta):
        return (
            triton.cdiv(max_len, meta["BLOCK_S"])
            * triton.cdiv(max_slice_size, meta["BLOCK_N"]),
            num_slices,
            batch_info.bs,
        )

    output = (
        torch.zeros((S, OUT_DIM), device=x.device, dtype=x.dtype)
        if base_output is None
        else base_output
    )
    _lora_expand_prefill_kernel[grid](
        x,
        weights,
        output,
        output.stride(0),
        output.stride(1),
        batch_info.seg_lens,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.permutation,
        batch_info.scalings,
        slice_offsets,
        NUM_SLICES=num_slices,
        OUTPUT_DIM=OUT_DIM,
        MAX_RANK=MAX_RANK,
        SORTED_BY_ADAPTER=sorted_by_adapter,
    )
    return output


load_kernel_cache(_lora_expand_prefill_kernel)
