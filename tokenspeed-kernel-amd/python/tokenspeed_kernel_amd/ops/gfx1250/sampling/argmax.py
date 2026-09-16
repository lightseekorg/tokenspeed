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

"""Gluon argmax kernels optimized for AMD GFX1250 sampling."""

from __future__ import annotations

from functools import lru_cache

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton

__all__ = [
    "argmax",
    "gluon_argmax_gfx1250",
]

cdna5 = gl.amd.cdna5

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SUPPORTED_OUT_DTYPES = (torch.int32, torch.int64)
_MIN_GLUON_VOCAB_SIZE = 4096
_INT32_MAX = gl.constexpr(2**31 - 1)
_scratch_cache: dict[
    tuple[int, int, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
] = {}


@gluon.jit
def _argmax_combine(value1, index1, value2, index2):
    take1 = (value1 > value2) | ((value1 == value2) & (index1 < index2))
    value = gl.where(take1, value1, value2)
    index = gl.where(take1, index1, index2)
    return value, index


@gluon.jit
def _normalize_argmax_sentinel(value, index):
    return gl.where((index == _INT32_MAX) | (value != value), -1, index)


@gluon.constexpr_function
def _argmax_layout(
    BLOCK: gl.constexpr, NUM_WARPS: gl.constexpr, LOAD_ELEMS: gl.constexpr
):
    return gl.BlockedLayout([LOAD_ELEMS], [32], [NUM_WARPS], [0])


@gluon.jit
def _argmax_accumulate_tile(best_val, best_idx, vals, cols, N: gl.constexpr):
    vals = vals.to(gl.float32)
    valid = (cols < N) & (vals == vals)
    vals = gl.where(valid, vals, -float("inf"))
    indices = gl.where(valid, cols, _INT32_MAX)
    return _argmax_combine(best_val, best_idx, vals, indices)


@gluon.jit
def _argmax_one_stage_kernel(
    logits,
    out,
    stride_m: gl.constexpr,
    out_stride: gl.constexpr,
    N: gl.constexpr,
    BLOCK: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    if N <= BLOCK:
        best_val, best_idx = _argmax_tile(
            logits, row, 0, stride_m, N, BLOCK, BLOCK, LOAD_ELEMS
        )
    else:
        layout: gl.constexpr = _argmax_layout(BLOCK, gl.num_warps(), LOAD_ELEMS)
        shared_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
        buffers = gl.allocate_shared_memory(
            logits.dtype.element_ty, [2, BLOCK], shared_layout
        )
        descriptor = cdna5.tdm.make_tensor_descriptor(
            base=logits + row * stride_m,
            shape=[N],
            strides=[1],
            block_shape=[BLOCK],
            layout=shared_layout,
        )
        offs = gl.arange(0, BLOCK, layout=layout)
        best_val = gl.full([BLOCK], -float("inf"), gl.float32, layout)
        best_idx = gl.full([BLOCK], _INT32_MAX, gl.int32, layout)
        cdna5.tdm.async_load(descriptor, [0], buffers.index(0))
        buffer_index = 0
        for tile in range(gl.cdiv(N, BLOCK) - 1):
            next_buffer = 1 - buffer_index
            # Load the next tile while processing the current one. Keep
            # per-lane candidates until the final workgroup reduction.
            cdna5.tdm.async_load(
                descriptor, [(tile + 1) * BLOCK], buffers.index(next_buffer)
            )
            cdna5.tdm.async_wait(1)
            vals = buffers.index(buffer_index).load(layout)
            best_val, best_idx = _argmax_accumulate_tile(
                best_val, best_idx, vals, tile * BLOCK + offs, N
            )
            buffer_index = next_buffer

        cdna5.tdm.async_wait(0)
        vals = buffers.index(buffer_index).load(layout)
        # TDM pads out-of-bounds elements with zero. Mask them before the
        # reduction so negative logits and all-NaN rows retain their meaning.
        best_val, best_idx = _argmax_accumulate_tile(
            best_val, best_idx, vals, (gl.cdiv(N, BLOCK) - 1) * BLOCK + offs, N
        )
        best_val, best_idx = gl.reduce(
            (best_val, best_idx), axis=0, combine_fn=_argmax_combine
        )

    best_idx = _normalize_argmax_sentinel(best_val, best_idx)
    gl.store(out + row * out_stride, best_idx.to(out.dtype.element_ty))


@gluon.jit
def _argmax_tile(
    logits,
    row,
    start,
    stride_m: gl.constexpr,
    N: gl.constexpr,
    CHUNK_SIZE: gl.constexpr,
    BLOCK: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
):
    offs = gl.arange(0, BLOCK, layout=_argmax_layout(BLOCK, gl.num_warps(), LOAD_ELEMS))
    cols = start + offs
    mask = (offs < CHUNK_SIZE) & (cols < N)
    vals = cdna5.buffer_load(
        logits,
        row * stride_m + cols,
        mask=mask,
        other=-float("inf"),
    ).to(gl.float32)
    mask = mask & (vals == vals)
    vals = gl.where(mask, vals, -float("inf"))
    indices = gl.where(mask, cols.to(gl.int32), _INT32_MAX)
    return gl.reduce((vals, indices), axis=0, combine_fn=_argmax_combine)


@gluon.jit
def _argmax_split_atomic_kernel(
    logits,
    partial_values,
    partial_indices,
    counters,
    out,
    stride_m: gl.constexpr,
    out_stride: gl.constexpr,
    N: gl.constexpr,
    CHUNK_SIZE: gl.constexpr,
    BLOCK: gl.constexpr,
    NUM_SPLITS: gl.constexpr,
    REDUCE_BLOCK: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    split = gl.program_id(1)
    tile_val, tile_idx = _argmax_tile(
        logits, row, split * CHUNK_SIZE, stride_m, N, CHUNK_SIZE, BLOCK, LOAD_ELEMS
    )
    partial_offset = row * NUM_SPLITS + split
    gl.store(partial_values + partial_offset, tile_val)
    gl.store(partial_indices + partial_offset, tile_idx)

    # Publish both partials before announcing completion. The last workgroup
    # acquires all partials and resets the counter for the next invocation.
    old = gl.atomic_add(counters + row, 1, sem="acq_rel", scope="gpu")
    if old == NUM_SPLITS - 1:
        offs = gl.arange(
            0, REDUCE_BLOCK, layout=_argmax_layout(REDUCE_BLOCK, gl.num_warps(), 1)
        )
        base = row * NUM_SPLITS + offs
        mask = offs < NUM_SPLITS
        vals = gl.load(
            partial_values + base, mask=mask, other=-float("inf"), volatile=True
        )
        indices = gl.load(
            partial_indices + base, mask=mask, other=_INT32_MAX, volatile=True
        )
        best_val, best_idx = gl.reduce(
            (vals, indices), axis=0, combine_fn=_argmax_combine
        )
        best_idx = _normalize_argmax_sentinel(best_val, best_idx)
        gl.store(out + row * out_stride, best_idx.to(out.dtype.element_ty))
        gl.store(counters + row, 0)


def _validate_argmax_out(logits: torch.Tensor, out: torch.Tensor) -> None:
    if out.shape != (logits.shape[0],):
        raise ValueError(
            f"out must have shape (M,)={(logits.shape[0],)}, got {tuple(out.shape)}"
        )
    if out.dtype not in _SUPPORTED_OUT_DTYPES:
        raise ValueError(f"out must be int32 or int64; got {out.dtype}")
    if out.device != logits.device:
        raise ValueError("out must be on the same device as logits")


def _argmax_torch_fallback(
    logits: torch.Tensor,
    *,
    out: torch.Tensor | None,
) -> torch.Tensor:
    if out is not None:
        _validate_argmax_out(logits, out)
    result = torch.argmax(logits, dim=-1)
    if out is not None:
        out.copy_(result)
        return out
    return result


def _supports_gluon(logits: torch.Tensor) -> bool:
    if logits.dim() != 2 or not logits.is_cuda:
        return False
    if logits.dtype not in _SUPPORTED_DTYPES:
        return False
    if logits.shape[1] < _MIN_GLUON_VOCAB_SIZE:
        return False
    if logits.stride(1) != 1:
        return False
    return True


def _get_atomic_scratch(
    M: int, num_splits: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    stream = triton.runtime.driver.active.get_current_stream(device.index)
    key = (device.index, stream, M, num_splits)
    scratch = _scratch_cache.get(key)
    if scratch is None:
        partial_values = torch.empty(
            (M, num_splits), dtype=torch.float32, device=device
        )
        partial_indices = torch.empty((M, num_splits), dtype=torch.int32, device=device)
        counters = torch.zeros((M,), dtype=torch.int32, device=device)
        scratch = (partial_values, partial_indices, counters)
        # Never retain allocations from a graph-private pool for eager reuse.
        if not torch.cuda.is_current_stream_capturing():
            _scratch_cache[key] = scratch
    return scratch


def _load_elements_per_thread(dtype: torch.dtype) -> int:
    return 4 if dtype == torch.float32 else 8


@lru_cache(maxsize=512)
def _select_config(M: int, N: int, dtype: torch.dtype) -> tuple[int, int, int]:
    """Return (tile width, warps, splits) for a wave32 reduction.

    Small batches split each row to expose parallelism. Larger batches use
    double-buffered TDM tiles, capped by bytes to limit shared-memory usage.
    """
    if M > 128 and N > 16384:
        block = 8192 if dtype == torch.float32 else 16384
        if M > 512:
            # More rows supply parallelism; smaller workgroups and buffers
            # let multiple rows reside on each compute unit.
            return block // 2, 8, 1
        return block, 16, 1
    if M > 256:
        return 16384, 8, 1
    if M <= 2:
        splits = 32
    elif M <= 4:
        splits = 16
    elif M <= 8:
        splits = max(16, triton.next_power_of_2(triton.cdiv(N, 8192)))
    elif M <= 32:
        # Keep medium batches in 16K tiles without rounding the workgroup
        # count up to a power of two for vocabularies wider than 128K.
        splits = max(8, triton.cdiv(N, 16384))
    elif M <= 64:
        splits = 8
    else:
        splits = 4
    # Avoid tiny splits for smaller vocabularies.
    splits = min(splits, max(1, triton.next_power_of_2(N) // 4096))
    if dtype == torch.float32:
        # 64K FP32 tiles increase register pressure; expose more parallelism
        # with at most 32K elements per tile instead.
        splits = max(splits, triton.next_power_of_2(triton.cdiv(N, 32768)))
    block = triton.next_power_of_2(triton.cdiv(N, splits))
    return block, 8, splits


def gluon_argmax_gfx1250(
    logits: torch.Tensor,
    *,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """Return row-wise argmax indices, ignoring NaNs and preferring first ties.

    Args:
        logits: Logits of shape ``(M, N)``. Contiguous vocabulary columns with
            N >= 4096 and fp16/bf16/fp32 use the GFX1250 kernel.
        out: Caller-owned int32/int64 ``(M,)`` tensor, or None to allocate int64.

    Returns:
        Output indices (the same tensor as out when supplied). All-NaN rows
        produce -1. Unsupported inputs retain torch.argmax semantics.
    """
    if out is not None:
        _validate_argmax_out(logits, out)

    if not _supports_gluon(logits):
        return _argmax_torch_fallback(logits, out=out)

    M, N = logits.shape
    if out is None:
        out = torch.empty((M,), dtype=torch.int64, device=logits.device)
    if M == 0:
        return out

    block, num_warps, num_splits = _select_config(M, N, logits.dtype)
    load_elems = _load_elements_per_thread(logits.dtype)
    if num_splits > 1:
        chunk_size = triton.cdiv(N, num_splits)
        partial_values, partial_indices, counters = _get_atomic_scratch(
            M, num_splits, logits.device
        )
        _argmax_split_atomic_kernel[(M, num_splits)](
            logits,
            partial_values,
            partial_indices,
            counters,
            out,
            stride_m=logits.stride(0),
            out_stride=out.stride(0),
            N=N,
            CHUNK_SIZE=chunk_size,
            BLOCK=block,
            NUM_SPLITS=num_splits,
            REDUCE_BLOCK=triton.next_power_of_2(num_splits),
            LOAD_ELEMS=load_elems,
            num_warps=num_warps,
        )
    else:
        _argmax_one_stage_kernel[(M,)](
            logits,
            out,
            stride_m=logits.stride(0),
            out_stride=out.stride(0),
            N=N,
            BLOCK=block,
            LOAD_ELEMS=load_elems,
            num_warps=num_warps,
        )
    return out


argmax = gluon_argmax_gfx1250
