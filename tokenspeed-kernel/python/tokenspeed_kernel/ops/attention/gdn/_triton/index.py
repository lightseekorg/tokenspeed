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

# -*- coding: utf-8 -*-

import weakref
from collections.abc import Callable

import numpy as np
import torch
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.ops.attention.gdn._triton.utils import tensor_cache

# Host-side chunk counting for the cu_seqlens tensor of the next chunked scan,
# so prepare_chunk_indices sizes its grid without reading the device prefix
# sum back (a stream sync). The scan picks its chunk sizes itself (chunk_fwd_o
# narrows 64 to 16 or 32 for batches of at most 32 tokens), so the hint keeps
# the sequence lengths and counts chunks for whichever size is asked. Every
# set_*() call replaces it; the weak reference binds it to that exact tensor,
# since a bare id() can be reused by a later allocation.
_chunks_hint: (
    tuple[weakref.ReferenceType[torch.Tensor], Callable[[int], int]] | None
) = None


def _bind_chunks_hint(
    cu_seqlens: torch.Tensor, total_chunks: Callable[[int], int]
) -> None:
    global _chunks_hint
    _chunks_hint = (weakref.ref(cu_seqlens), total_chunks)


def _hinted_total_chunks(cu_seqlens: torch.Tensor, chunk_size: int) -> int | None:
    if _chunks_hint is None:
        return None
    hinted_cu_seqlens, total_chunks = _chunks_hint
    if hinted_cu_seqlens() is not cu_seqlens:
        return None
    return total_chunks(chunk_size)


def set_total_chunks_hint(seq_lens_cpu, cu_seqlens: torch.Tensor) -> None:
    """Bind the host lengths of the sequences ``cu_seqlens`` delimits to it,
    so chunk indices for any chunk size are planned without a device read."""
    lens = np.array(seq_lens_cpu, dtype=np.int64)
    _bind_chunks_hint(
        cu_seqlens, lambda chunk_size: int(np.sum(-(-lens // chunk_size)))
    )


def set_total_chunks_hint_uniform(
    bs: int,
    tokens_per_seq: int,
    cu_seqlens: torch.Tensor,
) -> None:
    """Fast path for spec verify / draft-extend where every sequence has the
    same length (tokens_per_seq). Avoids allocating a per-seq numpy array."""
    _bind_chunks_hint(
        cu_seqlens, lambda chunk_size: bs * (-(-tokens_per_seq // chunk_size))
    )


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    nums = triton.cdiv(prepare_lens(cu_seqlens), chunk_size)
    offsets = torch.zeros(nums.shape[0] + 1, dtype=nums.dtype, device=nums.device)
    torch.cumsum(nums, dim=0, out=offsets[1:])
    total_int = _hinted_total_chunks(cu_seqlens, chunk_size)
    if total_int is None:
        total_int = offsets[-1].item()
    chunk_global = torch.arange(total_int, device=nums.device)
    seq_ids = torch.searchsorted(offsets[1:], chunk_global, right=True)
    local_indices = chunk_global - offsets[seq_ids]
    return torch.stack([seq_ids, local_indices], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    nums = triton.cdiv(prepare_lens(cu_seqlens), chunk_size)
    offsets = torch.zeros(nums.shape[0] + 1, dtype=nums.dtype, device=nums.device)
    torch.cumsum(nums, dim=0, out=offsets[1:])
    return offsets
