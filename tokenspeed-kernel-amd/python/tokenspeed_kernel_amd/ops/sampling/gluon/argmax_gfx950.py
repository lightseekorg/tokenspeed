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

"""Gluon argmax kernels optimized for AMD GFX950 sampling."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton

__all__ = [
    "argmax",
    "argmax_pair",
    "gluon_argmax_gfx950",
]

cdna4 = gl.amd.cdna4

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SUPPORTED_OUT_DTYPES = (torch.int32, torch.int64)
_MIN_GLUON_VOCAB_SIZE = 4096
_INT32_MAX = gl.constexpr(2**31 - 1)
_scratch_cache: dict[
    tuple[int, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
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
    return gl.BlockedLayout([LOAD_ELEMS], [64], [NUM_WARPS], [0])


@gluon.jit
def _argmax_fixed_block_size_tile(
    logits,
    row,
    start,
    stride_m: gl.constexpr,
    N: gl.constexpr,
    BLOCK: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
    SANITIZE_NAN: gl.constexpr,
):
    offs = gl.arange(0, BLOCK, layout=_argmax_layout(BLOCK, gl.num_warps(), LOAD_ELEMS))
    cols = start + offs
    mask = cols < N
    vals = cdna4.buffer_load(
        logits,
        row * stride_m + cols,
        mask=mask,
        other=-float("inf"),
    ).to(gl.float32)
    if SANITIZE_NAN:
        mask = mask & (vals == vals)
        vals = gl.where(mask, vals, -float("inf"))
    indices = gl.where(mask, cols.to(gl.int32), _INT32_MAX)
    return gl.reduce((vals, indices), axis=0, combine_fn=_argmax_combine)


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
    best_val = gl.full((), -float("inf"), gl.float32)
    best_idx = gl.full((), _INT32_MAX, gl.int32)

    for start in range(0, N, BLOCK):
        tile_val, tile_idx = _argmax_fixed_block_size_tile(
            logits, row, start, stride_m, N, BLOCK, LOAD_ELEMS, True
        )
        best_val, best_idx = _argmax_combine(best_val, best_idx, tile_val, tile_idx)

    best_idx = _normalize_argmax_sentinel(best_val, best_idx)
    gl.store(out + row * out_stride, best_idx.to(out.dtype.element_ty))


@gluon.jit
def _argmax_split_atomic_fixed_block_size_kernel(
    logits,
    partial_values,
    partial_indices,
    counters,
    out,
    stride_m: gl.constexpr,
    out_stride: gl.constexpr,
    N: gl.constexpr,
    NUM_SPLITS: gl.constexpr,
    BLOCK: gl.constexpr,
    REDUCE_BLOCK: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    split = gl.program_id(1)
    tile_val, tile_idx = _argmax_fixed_block_size_tile(
        logits, row, split * BLOCK, stride_m, N, BLOCK, LOAD_ELEMS, True
    )
    partial_offset = row * NUM_SPLITS + split
    gl.store(partial_values + partial_offset, tile_val)
    gl.store(partial_indices + partial_offset, tile_idx)

    old = gl.atomic_add(counters + row, 1, sem="acq_rel", scope="gpu")
    if old == NUM_SPLITS - 1:
        offs = gl.arange(
            0, REDUCE_BLOCK, layout=_argmax_layout(REDUCE_BLOCK, gl.num_warps(), 1)
        )
        mask = offs < NUM_SPLITS
        base = row * NUM_SPLITS + offs
        vals = gl.load(
            partial_values + base, mask=mask, other=-float("inf"), volatile=True
        )
        indices = gl.load(partial_indices + base, mask=mask, other=_INT32_MAX)
        best_val, best_idx = gl.reduce(
            (vals, indices), axis=0, combine_fn=_argmax_combine
        )
        best_idx = _normalize_argmax_sentinel(best_val, best_idx)
        gl.store(out + row * out_stride, best_idx.to(out.dtype.element_ty))
        gl.store(counters + row, 0)


@gluon.jit
def _argmax_fixed_split_count_tile(
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
    vals = cdna4.buffer_load(
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
def _argmax_split_atomic_fixed_split_count_kernel(
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
    tile_val, tile_idx = _argmax_fixed_split_count_tile(
        logits, row, split * CHUNK_SIZE, stride_m, N, CHUNK_SIZE, BLOCK, LOAD_ELEMS
    )
    partial_offset = row * NUM_SPLITS + split
    gl.store(partial_values + partial_offset, tile_val)
    gl.store(partial_indices + partial_offset, tile_idx)

    old = gl.atomic_add(counters + row, 1, sem="acq_rel", scope="gpu")
    if old == NUM_SPLITS - 1:
        offs = gl.arange(
            0, REDUCE_BLOCK, layout=_argmax_layout(REDUCE_BLOCK, gl.num_warps(), 1)
        )
        base = row * NUM_SPLITS + offs
        vals = gl.load(partial_values + base, volatile=True)
        indices = gl.load(partial_indices + base)
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


def _validate_argmax_pair_out(logits: torch.Tensor, out: torch.Tensor) -> None:
    shape = (logits.shape[0], 2)
    if out.shape != shape:
        raise ValueError(f"out must have shape (M, 2)={shape}, got {tuple(out.shape)}")
    if out.dtype != torch.float32 or out.device != logits.device:
        raise ValueError("out must be float32 on the same device as logits")


def _argmax_torch_fallback(
    logits: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is not None:
        _validate_argmax_out(logits, out)
    result = torch.argmax(logits, dim=-1)
    if out is not None:
        out.copy_(result)
        return out
    return result


def _argmax_pair_torch_fallback(
    logits: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.dim() != 2:
        raise ValueError(f"argmax_pair expects 2D input, got {logits.dim()}D")
    if out is None:
        out = torch.empty(
            (logits.shape[0], 2), dtype=torch.float32, device=logits.device
        )
    else:
        _validate_argmax_pair_out(logits, out)
    max_vals, max_indices = torch.max(logits, dim=-1, keepdim=True)
    out[:, 0:1].copy_(max_vals.to(torch.float32))
    out[:, 1:2].copy_(max_indices.to(torch.float32))
    return out


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
    key = (device.index, M, num_splits)
    scratch = _scratch_cache.get(key)
    if scratch is None:
        partial_values = torch.empty(
            (M, num_splits), dtype=torch.float32, device=device
        )
        partial_indices = torch.empty((M, num_splits), dtype=torch.int32, device=device)
        counters = torch.zeros((M,), dtype=torch.int32, device=device)
        scratch = (partial_values, partial_indices, counters)
        _scratch_cache[key] = scratch
    return scratch


def _load_elements_per_thread(dtype: torch.dtype) -> int:
    return 128 // torch.finfo(dtype).bits


def _select_config(M: int, N: int) -> tuple[int, int, bool, int | None]:
    """Return Gluon launch config for ``M x N`` logits.

    Returns ``(block_size, num_warps, use_split_atomic, fixed_split_count)``:
    the physical tile width, wave grouping, whether to use a split-atomic
    path, and an optional fixed split count. When ``fixed_split_count`` is
    set, ``block`` is the power-of-two tile width for each derived chunk.
    """
    if M <= 4 and N >= 65536:
        num_splits = 32 if M == 1 else 16
        block = triton.next_power_of_2(triton.cdiv(N, num_splits))
        return block, 4, True, num_splits
    if M <= 32:
        return 16384, 4, True, None
    if M <= 64:
        block = 32768 if N >= 180000 else 16384
        return block, 4, True, None
    return 8192, 4, False, None


def gluon_argmax_gfx950(
    logits: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is not None:
        _validate_argmax_out(logits, out)

    if not _supports_gluon(logits):
        return _argmax_torch_fallback(logits, out=out)

    M, N = logits.shape
    if out is None:
        out = torch.empty((M,), dtype=torch.int64, device=logits.device)
    if M == 0:
        return out

    block, num_warps, use_split, fixed_split_count = _select_config(M, N)
    load_elems = _load_elements_per_thread(logits.dtype)
    if fixed_split_count is not None:
        num_splits = fixed_split_count
        chunk_size = triton.cdiv(N, num_splits)
        partial_values, partial_indices, counters = _get_atomic_scratch(
            M, num_splits, logits.device
        )
        _argmax_split_atomic_fixed_split_count_kernel[(M, num_splits)](
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
            REDUCE_BLOCK=num_splits,
            LOAD_ELEMS=load_elems,
            num_warps=num_warps,
        )
    elif use_split:
        num_splits = triton.cdiv(N, block)
        split_block = triton.next_power_of_2(num_splits)
        partial_values, partial_indices, counters = _get_atomic_scratch(
            M, num_splits, logits.device
        )
        _argmax_split_atomic_fixed_block_size_kernel[(M, num_splits)](
            logits,
            partial_values,
            partial_indices,
            counters,
            out,
            stride_m=logits.stride(0),
            out_stride=out.stride(0),
            N=N,
            NUM_SPLITS=num_splits,
            BLOCK=block,
            REDUCE_BLOCK=split_block,
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


argmax = gluon_argmax_gfx950
argmax_pair = _argmax_pair_torch_fallback
