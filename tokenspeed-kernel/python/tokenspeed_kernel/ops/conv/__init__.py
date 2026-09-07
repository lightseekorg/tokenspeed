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

"""Convolution kernel entry points.

:func:`dflash2_grouped_conv` is independent of everything below it: a
block-local grouped depthwise convolution for DFlash2's draft layers, with no
cache, no ring and no cross-request state.

Depthwise causal FIR convolution with a short window ``W`` (typically 4) and
an optional residual connection, ``y = x + conv(window)``, as used by TML
hybrid layers. One entry point, :func:`inkling_ring_sconv`, dispatching on
``num_extends`` to two kernels with hard-coded state sources: extend
batches (``num_extends > 0``) tap the paged checkpoint and never read the
ring; decode/verify batches tap the ring and never read the paged
cache. Both publish covered boundary checkpoints unconditionally. The per-request convolution state lives in a slot-indexed ring
of the last ``R`` input rows, shape ``[num_slots, R, D]`` with D-contiguous
rows: the ring row of absolute position ``p`` is ``p % R``, positions derive
from the through-chunk ``seq_lens``, and there is no stored cursor —
speculative rows are overwritten when their positions recur.
``PAD_SLOT_ID`` (-1) marks padded batch rows that must never touch the ring.

The ring may be a channel-sliced view of a wider buffer
(``ring[:, :, off:off + D]``): all kernels receive explicit strides, so only
``conv_cache.stride(-1) == 1`` is required.
"""

from __future__ import annotations

import torch

# Aliased because the conv.triton submodule import below rebinds the name ``triton``.
from tokenspeed_kernel._triton import triton as _triton
from tokenspeed_kernel.ops.conv.triton import (
    _inkling_ring_sconv_decode_kernel,
    _inkling_ring_sconv_prefill_kernel,
    dflash2_grouped_conv,
    select_prefill_config,
)
from tokenspeed_kernel.platform import pdl_enabled

PAD_SLOT_ID = -1

__all__ = [
    "PAD_SLOT_ID",
    "dflash2_grouped_conv",
    "inkling_ring_sconv",
    "seq_idx_from_cu_seqlens",
]


def seq_idx_from_cu_seqlens(
    cu_seqlens: torch.Tensor, total_tokens: int
) -> torch.Tensor:
    """Map each packed token position to the index of its sequence.

    Args:
        cu_seqlens: Cumulative sequence lengths ``[B + 1]`` (integer tensor,
            starting at 0).
        total_tokens: Total number of packed tokens ``T``.

    Returns:
        Int32 tensor ``[T]`` where entry ``t`` is the sequence index that
        token ``t`` belongs to. Indices are clamped to ``B - 1`` so that
        tokens beyond ``cu_seqlens[-1]`` (e.g. CUDA-graph warmup padding with
        dummy zero-length sequences) stay in range.
    """
    t = torch.arange(total_tokens, dtype=torch.int64, device=cu_seqlens.device)
    num_seqs = cu_seqlens.shape[0] - 1
    return (
        (torch.searchsorted(cu_seqlens, t, side="right") - 1)
        .clamp(max=num_seqs - 1)
        .to(torch.int32)
    )


def inkling_ring_sconv(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_cache: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_idx: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    ckpt_a: torch.Tensor,
    ckpt_b: torch.Tensor | None,
    *,
    num_extends: int,
    page_size: int,
    activation: str | None = None,
    use_residual: bool = True,
) -> torch.Tensor:
    """Causal conv over ``[state ++ chunk]``, dispatched on ``num_extends``.

    ``num_extends > 0`` (extend batches; mixed batches are unsupported):
    each request's pre-chunk taps read the ``W - 1`` checkpoint rows at its
    aligned chunk-start boundary (zeros when ``has_initial_state`` is False
    or the table entry is a hole); the ring is never read. The last
    ``min(chunk_len + W - 1, R)`` positions persist to the ring in place —
    chunk rows plus the checkpoint window itself.

    ``num_extends == 0`` (decode/verify batches): chunks are
    uniform per request and must fit ``R - (W - 1)`` (asserted) so every
    chunk row persists without aliasing any request's tap reads; pre-chunk
    taps read the request's ring rows; the paged cache is never read.

    Both modes publish the conv window at every covered ``page_size``
    boundary, accept-independent — rejected content is overwritten by a
    later round covering the same boundary.

    Args:
        x: Varlen-packed input ``[T, D]`` (e.g. bf16). May be a strided
            slice of a wider buffer.
        weight: Per-channel FIR taps ``[D, W]``; tap ``W - 1`` multiplies
            the current token.
        conv_cache: Conv state ring ``[num_slots, R, D]`` with
            ``stride(-1) == 1`` and ``R >= W``. May be a channel-sliced
            view of a wider buffer. Updated in place.
        cu_seqlens: Cumulative sequence lengths ``[B + 1]``, int32.
        seq_idx: Sequence index per token ``[T]``, int32 (see
            :func:`seq_idx_from_cu_seqlens`).
        cache_indices: Ring slot per request ``[B]``, int32;
            ``PAD_SLOT_ID`` (-1) for padded rows.
        has_initial_state: Bool ``[B]``; False for prefix-0 requests, whose
            pre-chunk taps are zeros.
        seq_lens: Per-request lengths THROUGH the chunk ``[B]``, int32.
        block_table: Int32 ``[B, num_blocks]`` for the stream's cache group
            (entries <= 0 are holes).
        ckpt_a: Checkpoint field ``[pages, W - 1, width_a]`` covering
            channels ``[0, width_a)``.
        ckpt_b: Checkpoint field covering the remaining channels, or None
            when ``ckpt_a`` covers all ``D``.
        num_extends: Extend requests in the batch; 0 selects the decode
            kernel, anything else the prefill kernel.
        page_size: Tokens per cache page of the stream's group.
        activation: Optional activation before the residual: ``None``,
            ``"silu"`` or ``"swish"``.
        use_residual: Add the residual connection ``y = x + conv(...)``.
    Returns:
        Output tensor ``[T, D]`` with the same dtype as ``x``.
    """
    T, D = x.shape
    W = weight.shape[1]
    R = conv_cache.shape[1]
    assert R >= W, f"conv ring holds {R} rows per slot, needs at least W={W}"
    assert conv_cache.stride(-1) == 1, "conv_cache must be D-contiguous"

    y = torch.empty_like(x)
    if T == 0:
        return y

    if num_extends > 0:
        kernel = _inkling_ring_sconv_prefill_kernel
    else:
        B = cache_indices.shape[0]
        assert B > 0 and T % B == 0 and T // B <= R - (W - 1), (
            f"decode sconv needs uniform chunks of at most R-(W-1)="
            f"{R - (W - 1)} tokens; got T={T}, B={B}"
        )
        kernel = _inkling_ring_sconv_decode_kernel

    a_width = ckpt_a.shape[-1]
    a_strides = (ckpt_a.stride(0), ckpt_a.stride(1), ckpt_a.stride(2))
    if ckpt_b is None:
        assert a_width == D, "single checkpoint field must cover all channels"
        ckpt_b = ckpt_a
        b_strides = (0, 0, 0)
    else:
        assert (
            a_width + ckpt_b.shape[-1] == D
        ), "checkpoint fields must cover the stream's channels"
        b_strides = (ckpt_b.stride(0), ckpt_b.stride(1), ckpt_b.stride(2))

    enable_pdl = pdl_enabled()
    use_silu = activation in ("silu", "swish")
    block_t, block_d, num_warps, num_stages = select_prefill_config(T, D)
    grid = (_triton.cdiv(T, block_t), _triton.cdiv(D, block_d))
    kernel[grid](
        x,
        weight,
        conv_cache,
        cu_seqlens,
        seq_idx,
        cache_indices,
        has_initial_state,
        y,
        seq_lens,
        block_table,
        ckpt_a,
        ckpt_b,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        weight.stride(0),
        weight.stride(1),
        conv_cache.stride(0),
        conv_cache.stride(1),
        conv_cache.stride(2),
        block_table.stride(0),
        block_table.stride(1),
        a_strides[0],
        a_strides[1],
        a_strides[2],
        b_strides[0],
        b_strides[1],
        b_strides[2],
        a_width,
        block_table.shape[1],
        T,
        D,
        USE_SILU=use_silu,
        USE_RESIDUAL=use_residual,
        PAGE_SIZE=page_size,
        ENABLE_PDL=enable_pdl,
        R=R,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        W=W,
        W_POW2=_triton.next_power_of_2(W),
        num_warps=num_warps,
        num_stages=num_stages,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return y
