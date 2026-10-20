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

"""Argmax over a vocabulary sharded across ranks.

Each rank packs its shard's row maximum and global index into one int64 key
that orders like ``(value, -index)``, so an all-reduce MAX of the keys picks
the global maximum with ties toward the lowest index, as ``argmax`` over the
gathered row would. NaN follows the local ``argmax``: a NaN it picks outranks
every number, and a row it reports as ``-1`` on every shard decodes to ``-1``.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.sampling import argmax

_LOW_WORD = 0xFFFFFFFF


@triton.jit
def _pack_argmax_key_kernel(
    logits_ptr,
    stride_m,
    index_ptr,
    key_ptr,
    vocab_offset,
):
    row = tl.program_id(0)
    index = tl.load(index_ptr + row).to(tl.int64)
    safe_index = tl.maximum(index, 0)
    value = tl.load(logits_ptr + row.to(tl.int64) * stride_m + safe_index).to(
        tl.float32
    )
    bits = value.to(tl.int32, bitcast=True)
    # Canonicalize signed zero explicitly: compilers may fold away x + 0.0.
    bits = tl.where(value == 0.0, 0, bits)
    # Flip the magnitude bits of negatives so int32 order matches float order.
    ordered = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
    # A local argmax that picks NaN ranks it above every number, as
    # torch.argmax does; any NaN sign bit is ignored.
    ordered = tl.where(value != value, 0x7FFFFFFF, ordered).to(tl.int64)
    key = (ordered << 32) | (0xFFFFFFFF - (safe_index + vocab_offset))
    # The local argmax returns -1 for an all-NaN row; it never wins.
    key = tl.where(index < 0, -0x7FFFFFFFFFFFFFFF - 1, key)
    tl.store(key_ptr + row, key)


@triton.jit
def _argmax_key_index_kernel(key_ptr, out_ptr):
    row = tl.program_id(0)
    key = tl.load(key_ptr + row)
    # An all-NaN row keeps the sentinel on every rank; report -1 like argmax.
    index = tl.where(
        key == -0x7FFFFFFFFFFFFFFF - 1, -1, 0xFFFFFFFF - (key & 0xFFFFFFFF)
    )
    tl.store(out_ptr + row, index.to(out_ptr.dtype.element_ty))


def vocab_parallel_argmax_keys(
    logits: torch.Tensor, *, vocab_offset: int, out: torch.Tensor
) -> torch.Tensor:
    """Pack each row's shard argmax into an all-reduce-MAX-able key.

    Args:
        logits: This rank's ``[M, V_shard]`` logits shard; rows may be strided.
        vocab_offset: Global vocabulary index of the shard's first column.
        out: ``[M]`` int64 destination for the keys.

    Returns:
        ``out``. Reduce it with MAX across the vocabulary-parallel group, then
        decode with :func:`argmax_key_index`.
    """
    if logits.dim() != 2 or logits.stride(1) != 1:
        raise ValueError("logits must be a 2D tensor with unit column stride")
    if out.shape != (logits.shape[0],) or out.dtype != torch.int64:
        raise ValueError("out must be an int64 tensor with one key per row")
    if vocab_offset < 0 or vocab_offset + logits.shape[1] > _LOW_WORD:
        raise ValueError("global vocabulary indices must fit in 32 bits")
    rows = logits.shape[0]
    if rows == 0:
        return out
    local_index = argmax(logits)
    _pack_argmax_key_kernel[(rows,)](
        logits, logits.stride(0), local_index, out, vocab_offset
    )
    return out


def argmax_key_index(keys: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    """Decode the global vocabulary index from reduced argmax keys.

    Rows that are NaN on every shard decode to ``-1``, the ``argmax``
    sentinel.

    Args:
        keys: ``[M]`` int64 keys after the cross-rank MAX reduction.
        out: ``[M]`` int32 or int64 destination for the indices.

    Returns:
        ``out``.
    """
    if keys.dtype != torch.int64 or keys.dim() != 1:
        raise ValueError("keys must be a 1D int64 tensor")
    if out.shape != keys.shape or out.dtype not in (torch.int32, torch.int64):
        raise ValueError("out must be an int32 or int64 tensor shaped like keys")
    if keys.numel():
        _argmax_key_index_kernel[(keys.numel(),)](keys, out)
    return out
