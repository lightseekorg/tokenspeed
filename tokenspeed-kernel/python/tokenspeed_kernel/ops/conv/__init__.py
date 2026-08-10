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

"""Short causal convolution (sconv) kernel entry points.

Depthwise causal FIR convolution with a short window ``W`` (typically 4) and
an optional residual connection, ``y = x + conv(window)``, as used by TML
hybrid layers. The per-request convolution state lives in a slot-indexed ring
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
    _inkling_ring_restore_kernel,
    _inkling_ring_sconv_kernel,
    _inkling_ring_sconv_update_kernel,
    select_prefill_config,
)

PAD_SLOT_ID = -1

__all__ = [
    "PAD_SLOT_ID",
    "inkling_ring_restore",
    "inkling_ring_sconv",
    "inkling_ring_sconv_update",
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
    *,
    activation: str | None = None,
    use_residual: bool = True,
    publish: tuple | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Causal conv over ``[ring history ++ chunk]`` for a varlen batch.

    For each request, the convolution window at token ``t`` spans the last
    ``W`` positions; pre-chunk taps read the request's ring rows (zeros when
    the request has no initial state or its slot is ``PAD_SLOT_ID``). Chunks
    of at most ``R - (W - 1)`` tokens (decode/verify/catch-up shapes) are
    persisted in-kernel at their positions' ring rows; call
    :func:`inkling_ring_sconv_update` afterwards for longer chunks (prefill/extend).

    Args:
        x: Varlen-packed input ``[T, D]`` (e.g. bf16), D-contiguous.
        weight: Per-channel FIR taps ``[D, W]``; tap ``W - 1`` multiplies the
            current token.
        conv_cache: Conv state ring ``[num_slots, R, D]`` with
            ``stride(-1) == 1`` and ``R >= W``. May be a channel-sliced view
            of a wider buffer.
        cu_seqlens: Cumulative sequence lengths ``[B + 1]``, int32.
        seq_idx: Sequence index per token ``[T]``, int32 (see
            :func:`seq_idx_from_cu_seqlens`).
        cache_indices: Ring slot per request ``[B]``, int32;
            ``PAD_SLOT_ID`` (-1) for padded rows.
        has_initial_state: Bool ``[B]``; when False pre-chunk taps are zeros.
        seq_lens: Per-request lengths THROUGH the chunk ``[B]``, int32; the
            absolute position of chunk token ``t`` is
            ``seq_lens[si] - (eos - t)`` and every ring row derives from it.
        activation: Optional activation applied to the conv output before the
            residual: ``None`` (TML default), ``"silu"`` or ``"swish"``.
        use_residual: Add the residual connection ``y = x + conv(...)``.
        publish: Optional speculative boundary-checkpoint publication as
            ``(block_table, checkpoint, checkpoint_b, page_size)``:
            ``block_table`` int32 ``[B, num_blocks]`` for the stream's cache
            group (entries <= 0 are holes and skip the write), ``checkpoint``
            ``[pages, W - 1, width_a]`` receiving channels ``[0, width_a)``
            and ``checkpoint_b`` the rest (or None when one field covers all
            channels). Every token landing exactly on a ``page_size``
            boundary writes the conv window at its position —
            accept-independent; later rounds covering the same boundary
            overwrite rejected content.
        enable_pdl: launch with Programmatic Dependent Launch (Hopper+).

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

    use_silu = activation in ("silu", "swish")
    block_t, block_d, num_warps, num_stages = select_prefill_config(T, D)

    if publish is not None:
        block_table, ckpt_a, ckpt_b, page_size = publish
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
        bt_strides = (block_table.stride(0), block_table.stride(1))
        num_blocks = block_table.shape[1]
    else:
        block_table = cache_indices
        ckpt_a = ckpt_b = x
        bt_strides = (0, 0)
        a_strides = b_strides = (0, 0, 0)
        a_width = D
        num_blocks = 0
        page_size = 1

    grid = (_triton.cdiv(T, block_t), _triton.cdiv(D, block_d))
    _inkling_ring_sconv_kernel[grid](
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
        bt_strides[0],
        bt_strides[1],
        a_strides[0],
        a_strides[1],
        a_strides[2],
        b_strides[0],
        b_strides[1],
        b_strides[2],
        a_width,
        num_blocks,
        T,
        D,
        USE_SILU=use_silu,
        USE_RESIDUAL=use_residual,
        PUBLISH=publish is not None,
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


def inkling_ring_sconv_update(
    x: torch.Tensor,
    conv_cache: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    cache_indices: torch.Tensor,
    *,
    kernel_width: int,
) -> None:
    """Persist each request's last ``min(chunk_len, R)`` chunk rows into its
    ring, in place — the full ring depth, so any consumer rewind (e.g. the
    draft decode lookback window) finds its taps.

    Required after :func:`inkling_ring_sconv` for batches that may contain chunks
    longer than ``R - (W - 1)`` (prefill/extend), which the compute kernel
    does not persist. Requests at or under that bound exit early (already
    written in-kernel); rows whose source position precedes the chunk are
    skipped — the ring already holds them (no shift borrow).

    Args:
        x: Varlen-packed input ``[T, D]`` that was fed to
            :func:`inkling_ring_sconv`.
        conv_cache: Conv state ring ``[num_slots, R, D]`` with
            ``stride(-1) == 1``; updated in place. May be a channel-sliced
            view of a wider buffer.
        cu_seqlens: Cumulative sequence lengths ``[B + 1]``, int32.
        seq_lens: Per-request lengths THROUGH the chunk ``[B]``, int32.
        cache_indices: Ring slot per request ``[B]``, int32;
            ``PAD_SLOT_ID`` (-1) rows are skipped entirely.
        kernel_width: The conv window ``W`` (taps per channel).

    Returns:
        None. ``conv_cache`` is modified in place.
    """
    B = cache_indices.shape[0]
    D = x.shape[-1]
    R = conv_cache.shape[1]
    assert conv_cache.stride(-1) == 1, "conv_cache must be D-contiguous"
    if B == 0:
        return

    block_d = min(_triton.next_power_of_2(D), 1024)
    grid = (B, _triton.cdiv(D, block_d))
    _inkling_ring_sconv_update_kernel[grid](
        x,
        conv_cache,
        cu_seqlens,
        seq_lens,
        cache_indices,
        x.stride(0),
        x.stride(1),
        conv_cache.stride(0),
        conv_cache.stride(1),
        conv_cache.stride(2),
        D,
        BLOCK_D=block_d,
        R=R,
        W_MINUS_1=kernel_width - 1,
        num_warps=4,
    )


def inkling_ring_restore(
    conv_cache: torch.Tensor,
    ckpt_a: torch.Tensor,
    ckpt_b: torch.Tensor | None,
    block_table: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    cache_indices: torch.Tensor,
    *,
    page_size: int,
) -> None:
    """Restore each request's aligned-prefix boundary checkpoint into its
    ring rows, in place, in one launch.

    A restore source exists only where the prefix boundary
    (``seq_lens - chunk_len``) is a positive ``page_size`` multiple whose
    ``block_table`` entry is a real page; PAD slots, unaligned or zero
    prefixes and hole pages (entries <= 0) are skipped per request.

    Args:
        conv_cache: Conv state ring ``[num_slots, R, D]`` with
            ``stride(-1) == 1``; updated in place. May be a channel-sliced
            view of a wider buffer.
        ckpt_a: Checkpoint field ``[pages, W - 1, width_a]`` providing
            channels ``[0, width_a)``.
        ckpt_b: Checkpoint field providing the remaining channels, or None
            when ``ckpt_a`` covers all ``D`` channels.
        block_table: Int32 ``[B, num_blocks]`` for the stream's cache group.
        cu_seqlens: Cumulative sequence lengths ``[B + 1]``, int32.
        seq_lens: Per-request lengths THROUGH the chunk ``[B]``, int32.
        cache_indices: Ring slot per request ``[B]``, int32;
            ``PAD_SLOT_ID`` (-1) rows are skipped entirely.
        page_size: Tokens per cache page of the stream's group.

    Returns:
        None. ``conv_cache`` is modified in place.
    """
    B = cache_indices.shape[0]
    D = conv_cache.shape[-1]
    R = conv_cache.shape[1]
    assert conv_cache.stride(-1) == 1, "conv_cache must be D-contiguous"
    if B == 0:
        return

    a_width = ckpt_a.shape[-1]
    if ckpt_b is None:
        assert a_width == D, "single checkpoint field must cover all channels"
        ckpt_b = ckpt_a
        b_strides = (0, 0, 0)
    else:
        assert (
            a_width + ckpt_b.shape[-1] == D
        ), "checkpoint fields must cover the stream's channels"
        b_strides = (ckpt_b.stride(0), ckpt_b.stride(1), ckpt_b.stride(2))

    block_d = min(_triton.next_power_of_2(D), 1024)
    grid = (B, _triton.cdiv(D, block_d))
    _inkling_ring_restore_kernel[grid](
        ckpt_a,
        ckpt_b,
        conv_cache,
        block_table,
        cu_seqlens,
        seq_lens,
        cache_indices,
        ckpt_a.stride(0),
        ckpt_a.stride(1),
        ckpt_a.stride(2),
        b_strides[0],
        b_strides[1],
        b_strides[2],
        conv_cache.stride(0),
        conv_cache.stride(1),
        conv_cache.stride(2),
        block_table.stride(0),
        block_table.stride(1),
        D,
        a_width,
        BLOCK_D=block_d,
        R=R,
        W_MINUS_1=ckpt_a.shape[1],
        PAGE_SIZE=page_size,
        num_warps=4,
    )
