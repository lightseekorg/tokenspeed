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

:func:`causal_conv1d_fn` (prefill) and :func:`causal_conv1d_update` (decode,
chain verify and draft trees) select the depthwise causal conv in front of
the Mamba / GDN recurrences, over a slot-indexed conv-window pool; the
selected leaves are registered in ``conv/triton.py`` and ``conv/cute_dsl.py``.

:func:`dflash2_grouped_conv` is independent of everything below it: a
block-local grouped depthwise convolution for DFlash2's draft layers, with no
cache, no ring and no cross-request state.

Depthwise causal FIR convolution with a short window ``W`` (typically 4) and
an optional residual connection, ``y = x + conv(window)``, as used by TML
hybrid layers. One entry point, :func:`inkling_ring_sconv`, dispatching on
``num_extends`` to two kernels with hard-coded state sources: extend
batches (``num_extends > 0``) tap the checkpoint in the request's input slot
and never read the ring; decode/verify batches tap the ring and never read
the checkpoint cache. Both may publish the chunk-end window to the request's
live slot. The per-request convolution state lives in a slot-indexed ring
of the last ``R`` input rows, shape ``[num_slots, R, D]`` with D-contiguous
rows: the ring row of absolute position ``p`` is ``p % R``, positions derive
from the through-chunk ``seq_lens``, and there is no stored cursor --
speculative rows are overwritten when their positions recur.
``PAD_SLOT_ID`` (-1) marks padded batch rows that must never touch the ring.

The ring may be a channel-sliced view of a wider buffer
(``ring[:, :, off:off + D]``): all kernels receive explicit strides, so only
``conv_cache.stride(-1) == 1`` is required.

A target verify's decode kernel publishes nothing: the window to keep ends
at the accepted frontier, known only after acceptance.
:func:`inkling_ring_commit` then publishes it from the ring for every
layer's streams at once, through the tables :func:`inkling_commit_tables`
builds once.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

# Aliased because importing the conv.triton and conv._triton submodules below
# rebinds the names ``triton`` and ``_triton`` in this package.
from tokenspeed_kernel._triton import tl as _tl
from tokenspeed_kernel._triton import triton as _triton_lang
from tokenspeed_kernel.ops.attention.gdn.triton import CausalConv1dPrefillMetadata
from tokenspeed_kernel.ops.conv import cute_dsl as _cute_dsl_solutions  # noqa: F401
from tokenspeed_kernel.ops.conv.triton import (
    PAD_SLOT_ID,
    _inkling_ring_commit_kernel,
    _inkling_ring_sconv_decode_kernel,
    _inkling_ring_sconv_prefill_kernel,
    dflash2_grouped_conv,
    select_prefill_config,
)
from tokenspeed_kernel.platform import pdl_enabled
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "PAD_SLOT_ID",
    "InklingCommitTable",
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "dflash2_grouped_conv",
    "inkling_commit_tables",
    "inkling_ring_commit",
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
    publish_live: bool,
    activation: str | None = None,
    use_residual: bool = True,
) -> torch.Tensor:
    """Causal conv over ``[state ++ chunk]``, dispatched on ``num_extends``.

    ``num_extends > 0`` (extend batches; mixed batches are unsupported):
    each request's pre-chunk taps read the ``W - 1`` checkpoint rows of its
    input slot (zeros when ``has_initial_state`` is False or the slot is a
    hole); the ring is never read. The last ``min(chunk_len + W - 1, R)``
    positions persist to the ring in place -- chunk rows plus the checkpoint
    window itself.

    ``num_extends == 0`` (decode/verify batches): chunks are
    uniform per request and must fit ``R - (W - 1)`` (asserted) so every
    chunk row persists without aliasing any request's tap reads; pre-chunk
    taps read the request's ring rows; the checkpoint cache is never read.

    With ``publish_live`` each request's last token writes the conv window
    ending at ``seq_lens`` to its live slot. That window is the committed one
    for an extend chunk, a one-token decode, or a draft chunk re-anchored at
    the committed frontier; a target verify leaves ``publish_live`` off and
    publishes the accepted window from the ring after acceptance.

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
        block_table: Int32 ``[B, >= 2]`` for the stream's cache group:
            column 0 the input checkpoint block, column 1 the live block
            (entries <= 0 are holes; wider rows are padding).
        ckpt_a: Checkpoint field ``[pages, W - 1, width_a]`` covering
            channels ``[0, width_a)``.
        ckpt_b: Checkpoint field covering the remaining channels, or None
            when ``ckpt_a`` covers all ``D``.
        num_extends: Extend requests in the batch; 0 selects the decode
            kernel, anything else the prefill kernel.
        publish_live: Write each request's chunk-end window to its live slot.
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
    assert (
        block_table.dim() == 2 and block_table.shape[1] >= 2
    ), "the conv group's block table holds an input and a live slot per request"

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
    grid = (_triton_lang.cdiv(T, block_t), _triton_lang.cdiv(D, block_d))
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
        T,
        D,
        USE_SILU=use_silu,
        USE_RESIDUAL=use_residual,
        PUBLISH=publish_live,
        ENABLE_PDL=enable_pdl,
        R=R,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        W=W,
        W_POW2=_triton_lang.next_power_of_2(W),
        num_warps=num_warps,
        num_stages=num_stages,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return y


_COMMIT_DTYPES = {
    torch.bfloat16: _tl.bfloat16,
    torch.float16: _tl.float16,
    torch.float32: _tl.float32,
    torch.float8_e5m2: _tl.float8e5,
    torch.float8_e4m3fn: _tl.float8e4nv,
}


@dataclass(frozen=True)
class InklingCommitTable:
    """The checkpoint fields one :func:`inkling_ring_commit` launch writes.

    Each entry is one field of one layer's stream; every field is of
    ``dtype``. Device tensors, one element per entry.

    Attributes:
        src_offsets: Int64 ring offset of the entry's first channel
            (``layer * ring.stride(0) + channel``).
        widths: Int32 channels the field holds.
        addresses: Uint64 address of the field's page 0.
        page_strides: Int64 field page stride, in elements.
        row_strides: Int64 field row stride, in elements.
        groups: Int32 row of the entry's group in ``live_pages``.
        dtype: The fields' dtype.
        state_rows: Rows per checkpoint (``W - 1``).
        max_width: The widest entry.
    """

    src_offsets: torch.Tensor
    widths: torch.Tensor
    addresses: torch.Tensor
    page_strides: torch.Tensor
    row_strides: torch.Tensor
    groups: torch.Tensor
    dtype: torch.dtype
    state_rows: int
    max_width: int


def inkling_commit_tables(
    ring: torch.Tensor,
    streams: Sequence[tuple[int, int, int, Sequence[torch.Tensor]]],
) -> list[InklingCommitTable]:
    """Describe the checkpoint fields :func:`inkling_ring_commit` writes.

    Args:
        ring: Conv rings of every layer ``[L, num_slots, R, C]``,
            channel-contiguous.
        streams: ``(layer, channel_offset, group, fields)`` per stream. The
            stream's channels start at ``channel_offset`` of the layer's ring
            and its fields (``[pages, W - 1, width]``, width-contiguous) split
            them in order; ``group`` is the stream's row of ``live_pages``.

    Returns:
        One table per field dtype.
    """
    if ring.dim() != 4 or ring.stride(-1) != 1:
        raise ValueError("the rings must be [L, num_slots, R, C], C-contiguous")
    entries: dict[torch.dtype, list[tuple[int, ...]]] = {}
    state_rows: int | None = None
    for layer, channel_offset, group, fields in streams:
        channel = channel_offset
        for field in fields:
            if field.dim() != 3 or field.stride(-1) != 1:
                raise ValueError("a checkpoint field must be [pages, W - 1, width]")
            if state_rows not in (None, field.shape[1]):
                raise ValueError("every checkpoint field must hold W - 1 rows")
            state_rows = field.shape[1]
            entries.setdefault(field.dtype, []).append(
                (
                    layer * ring.stride(0) + channel,
                    field.shape[-1],
                    field.data_ptr(),
                    field.stride(0),
                    field.stride(1),
                    group,
                )
            )
            channel += field.shape[-1]
        if channel > ring.shape[-1]:
            raise ValueError(f"layer {layer}'s stream runs past the ring channels")

    def device(values: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(values, dtype=dtype, device=ring.device)

    tables = []
    for dtype, rows in entries.items():
        offsets, widths, addresses, page_strides, row_strides, groups = zip(*rows)
        tables.append(
            InklingCommitTable(
                src_offsets=device(offsets, torch.int64),
                widths=device(widths, torch.int32),
                addresses=device(addresses, torch.uint64),
                page_strides=device(page_strides, torch.int64),
                row_strides=device(row_strides, torch.int64),
                groups=device(groups, torch.int32),
                dtype=dtype,
                state_rows=state_rows,
                max_width=max(widths),
            )
        )
    return tables


def inkling_ring_commit(
    ring: torch.Tensor,
    tables: Sequence[InklingCommitTable],
    live_pages: torch.Tensor,
    cache_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    accepted_lengths: torch.Tensor,
    *,
    verify_width: int,
) -> None:
    """Publish each request's accepted conv window to its live block.

    After a target verify the ring holds all ``verify_width`` speculative
    rows of every request; the live block must hold the ``W - 1`` rows ending
    at the committed frontier ``seq_lens - verify_width + clamp(accepted, 1,
    verify_width)``, zeros before position 0. One launch per table covers
    every layer's streams.

    Args:
        ring: The rings the tables were built over.
        tables: :func:`inkling_commit_tables` of ``ring``.
        live_pages: Int32 ``[G, >= B]`` live block per group and request,
            request-contiguous; entries ``<= 0`` are holes and store nothing.
        cache_indices: Contiguous int32 ``[B]`` ring slot per request;
            ``PAD_SLOT_ID`` rows store nothing.
        seq_lens: Contiguous integer ``[B]`` lengths through the verify chunk.
        accepted_lengths: Contiguous integer ``[B]`` accepted tokens per
            request.
        verify_width: Tokens per request in the verify chunk.
    """
    num_reqs = accepted_lengths.shape[0]
    if num_reqs == 0:
        return
    if live_pages.stride(-1) != 1:
        raise ValueError("live_pages must be request-contiguous")
    if not all(v.is_contiguous() for v in (cache_indices, seq_lens, accepted_lengths)):
        raise ValueError(
            "cache_indices, seq_lens and accepted_lengths must be contiguous"
        )
    enable_pdl = pdl_enabled()
    for table in tables:
        block_c = min(1024, _triton_lang.next_power_of_2(table.max_width))
        grid = (
            table.widths.shape[0],
            num_reqs,
            _triton_lang.cdiv(table.max_width, block_c),
        )
        _inkling_ring_commit_kernel[grid](
            ring,
            table.src_offsets,
            table.widths,
            table.addresses,
            table.page_strides,
            table.row_strides,
            table.groups,
            live_pages,
            cache_indices,
            seq_lens,
            accepted_lengths,
            ring.stride(1),
            ring.stride(2),
            live_pages.stride(0),
            CKPT_DTYPE=_COMMIT_DTYPES[table.dtype],
            STATE_ROWS=table.state_rows,
            R=ring.shape[2],
            VERIFY_WIDTH=verify_width,
            BLOCK_C=block_c,
            ENABLE_PDL=enable_pdl,
            **({"launch_pdl": True} if enable_pdl else {}),
        )


def _state_layout(state: torch.Tensor, weight: torch.Tensor) -> str:
    if (
        state.ndim == 3
        and state.shape[1] == weight.shape[1] - 1
        and state.shape[2] == weight.shape[0]
    ):
        return "width_major"
    return "channel_major"


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    validate_data=False,
    *,
    prefill_metadata: CausalConv1dPrefillMetadata,
):
    """Select a prefill convolution leaf while preserving the original ABI."""
    kernel = select_kernel(
        "conv",
        "causal_conv1d_prefill",
        format_signature(x=dense_tensor_format(x.dtype)),
        traits={
            "width": weight.shape[1],
            "state_layout": _state_layout(conv_states, weight),
            "channels": x.shape[0],
            "channel_contiguous": x.stride(0) == 1,
            "token_stride": x.stride(1),
            "weight_dtype": weight.dtype,
            "bias_dtype": bias.dtype if bias is not None else None,
            "state_dtype": conv_states.dtype,
            "indexed_state": cache_indices is not None
            and has_initial_state is not None,
            "pad_slot_id": pad_slot_id,
            "block_m": prefill_metadata.block_m,
        },
    )
    return kernel(
        x=x,
        weight=weight,
        bias=bias,
        conv_states=conv_states,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation=activation,
        pad_slot_id=pad_slot_id,
        validate_data=validate_data,
        prefill_metadata=prefill_metadata,
    )


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    cache_seqlens: torch.Tensor | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    intermediate_conv_window: torch.Tensor | None = None,
    output_state_indices: torch.Tensor | None = None,
    pad_slot_id: int = PAD_SLOT_ID,
    validate_data=False,
    *,
    parent_indices: torch.Tensor | None,
    out: torch.Tensor | None = None,
    replay_payload: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
):
    """Select a decode/tree convolution leaf for the supplied state indices."""
    kernel = select_kernel(
        "conv",
        "causal_conv1d_update",
        format_signature(x=dense_tensor_format(x.dtype)),
        traits={
            "width": weight.shape[1],
            "tree": parent_indices is not None,
            "state_layout": _state_layout(conv_state, weight),
        },
    )
    return kernel(
        x=x,
        conv_state=conv_state,
        weight=weight,
        bias=bias,
        activation=activation,
        cache_seqlens=cache_seqlens,
        conv_state_indices=conv_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        intermediate_conv_window=intermediate_conv_window,
        output_state_indices=output_state_indices,
        pad_slot_id=pad_slot_id,
        validate_data=validate_data,
        parent_indices=parent_indices,
        out=out,
        replay_payload=replay_payload,
    )
