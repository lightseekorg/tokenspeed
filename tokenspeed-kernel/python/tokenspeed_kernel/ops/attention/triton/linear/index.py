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

import numpy as np
import torch
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.ops.attention.triton.linear.utils import tensor_cache

# Pre-computed total chunk counts. Keyed by (chunk_size, id(cu_seqlens)) and
# cleared at the start of every set_*() call.
_total_chunks_hint: dict[tuple[int, int], int] = {}


def set_total_chunks_hint(
    seq_lens_cpu,
    cu_seqlens: torch.Tensor,
    chunk_sizes: tuple[int, ...] = (16, 64),
) -> None:
    """Pre-compute total chunk counts on CPU and bind them to a specific
    cu_seqlens tensor (by id), avoiding cross-batch hint pollution."""
    lens = np.asarray(seq_lens_cpu, dtype=np.int64)
    key_id = id(cu_seqlens)
    _total_chunks_hint.clear()
    for cs in chunk_sizes:
        _total_chunks_hint[(cs, key_id)] = int(np.sum(-(-lens // cs)))


def set_total_chunks_hint_uniform(
    bs: int,
    tokens_per_seq: int,
    cu_seqlens: torch.Tensor,
    chunk_sizes: tuple[int, ...] = (16, 64),
) -> None:
    """Fast path for spec verify / draft-extend where every sequence has the
    same length (tokens_per_seq). Avoids allocating a per-seq numpy array."""
    key_id = id(cu_seqlens)
    _total_chunks_hint.clear()
    for cs in chunk_sizes:
        _total_chunks_hint[(cs, key_id)] = bs * (-(-tokens_per_seq // cs))


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
    total_int = _total_chunks_hint.pop((chunk_size, id(cu_seqlens)), None)
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
