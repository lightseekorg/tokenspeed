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
hybrid layers. The per-request convolution state (the last ``W - 1`` inputs)
lives in a slot-indexed cache of shape ``[num_slots, W - 1, D]`` with
D-contiguous rows; ``PAD_SLOT_ID`` (-1) marks padded batch rows that must
never touch the cache.

The cache may be a channel-sliced view of a wider buffer
(``cache[:, :, off:off + D]``): all kernels receive explicit strides, so only
``conv_cache.stride(-1) == 1`` is required.
"""

from __future__ import annotations

import torch

# Aliased because the conv.triton submodule import below rebinds the name ``triton``.
from tokenspeed_kernel._triton import triton as _triton
from tokenspeed_kernel.ops.conv.triton import (
    _sconv_cache_update_kernel,
    _sconv_decode_kernel,
    _sconv_decode_paged_kernel,
    _sconv_prefill_kernel,
    _sconv_prefill_paged_kernel,
    select_decode_config,
    select_prefill_config,
)
from tokenspeed_kernel.platform import current_platform

PAD_SLOT_ID = -1

__all__ = [
    "PAD_SLOT_ID",
    "sconv_cache_update",
    "sconv_decode",
    "sconv_decode_paged",
    "sconv_prefill",
    "sconv_prefill_paged",
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


def sconv_prefill(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_cache: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_idx: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    *,
    activation: str | None = None,
    use_residual: bool = True,
) -> torch.Tensor:
    """Causal conv over ``[cached-prefix ++ chunk]`` for a varlen batch.

    For each request, the convolution window at token ``t`` spans the last
    ``W`` positions of ``[prefix ++ chunk]`` where ``prefix`` is the
    request's ``[W - 1, D]`` conv cache row (zeros when the request has no
    initial state or its slot is ``PAD_SLOT_ID``). The cache is read-only
    here; call :func:`sconv_cache_update` afterwards to persist final states.

    Args:
        x: Varlen-packed input ``[T, D]`` (e.g. bf16), D-contiguous.
        weight: Per-channel FIR taps ``[D, W]``; tap ``W - 1`` multiplies the
            current token.
        conv_cache: Conv state cache ``[num_slots, W - 1, D]`` with
            ``stride(-1) == 1``. May be a channel-sliced view of a wider
            buffer. Not modified.
        cu_seqlens: Cumulative sequence lengths ``[B + 1]``, int32.
        seq_idx: Sequence index per token ``[T]``, int32 (see
            :func:`seq_idx_from_cu_seqlens`).
        cache_indices: Cache slot per request ``[B]``, int32;
            ``PAD_SLOT_ID`` (-1) for padded rows.
        has_initial_state: Bool ``[B]``; when False the prefix is zeros.
        activation: Optional activation applied to the conv output before the
            residual: ``None`` (TML default), ``"silu"`` or ``"swish"``.
        use_residual: Add the residual connection ``y = x + conv(...)``.

    Returns:
        Output tensor ``[T, D]`` with the same dtype as ``x``.
    """
    T, D = x.shape
    W = weight.shape[1]
    assert (
        conv_cache.shape[1] == W - 1
    ), f"conv_cache holds {conv_cache.shape[1]} states per slot, expected {W - 1}"
    assert conv_cache.stride(-1) == 1, "conv_cache must be D-contiguous"

    y = torch.empty_like(x)
    if T == 0:
        return y

    use_silu = activation in ("silu", "swish")
    block_t, block_d, num_warps, num_stages = select_prefill_config(T, D)

    grid = (_triton.cdiv(T, block_t), _triton.cdiv(D, block_d))
    _sconv_prefill_kernel[grid](
        x,
        weight,
        conv_cache,
        cu_seqlens,
        seq_idx,
        cache_indices,
        has_initial_state,
        y,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        weight.stride(0),
        weight.stride(1),
        conv_cache.stride(0),
        conv_cache.stride(1),
        conv_cache.stride(2),
        T,
        D,
        USE_SILU=use_silu,
        USE_RESIDUAL=use_residual,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        W=W,
        W_POW2=_triton.next_power_of_2(W),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y


def sconv_cache_update(
    x: torch.Tensor,
    conv_cache: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> None:
    """Write back each request's final ``W - 1`` conv states, in place.

    For requests with ``query_len >= W - 1`` the slot receives the last
    ``W - 1`` input tokens. For shorter requests the old cache content is
    shifted left by ``query_len`` (when ``has_initial_state`` is True) or
    zero-filled (when False) before appending the new tokens. Rows with
    ``cache_indices == PAD_SLOT_ID`` are skipped entirely — they never write
    to any slot (the TML reference clamped them to slot 0 and clobbered it).

    Args:
        x: Varlen-packed input ``[T, D]`` that was fed to
            :func:`sconv_prefill`.
        conv_cache: Conv state cache ``[num_slots, W - 1, D]`` with
            ``stride(-1) == 1``; updated in place. May be a channel-sliced
            view of a wider buffer.
        cu_seqlens: Cumulative sequence lengths ``[B + 1]``, int32.
        cache_indices: Cache slot per request ``[B]``, int32;
            ``PAD_SLOT_ID`` (-1) for padded rows.
        has_initial_state: Bool ``[B]``; selects shift vs zero fill for the
            ``query_len < W - 1`` path.

    Returns:
        None. ``conv_cache`` is modified in place.
    """
    B = cache_indices.shape[0]
    D = x.shape[-1]
    w_minus_1 = conv_cache.shape[1]
    assert conv_cache.stride(-1) == 1, "conv_cache must be D-contiguous"
    if B == 0:
        return

    block_d = min(_triton.next_power_of_2(D), 1024)
    grid = (B, _triton.cdiv(D, block_d))
    _sconv_cache_update_kernel[grid](
        x,
        conv_cache,
        cu_seqlens,
        cache_indices,
        has_initial_state,
        x.stride(0),
        x.stride(1),
        conv_cache.stride(0),
        conv_cache.stride(1),
        conv_cache.stride(2),
        D,
        BLOCK_D=block_d,
        W_MINUS_1=w_minus_1,
        num_warps=4,
    )


def sconv_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_cache: torch.Tensor,
    cache_indices: torch.Tensor,
    *,
    activation: str | None = None,
    use_residual: bool = True,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Fused single-token decode: conv over ``[cache ++ x_t]`` + cache update.

    Computes the causal conv for one new token per request, reading the
    ``W - 1`` state tokens directly from the cache, then shifts each cache
    row left by one and stores the new token — all in a single launch. Rows
    with ``cache_indices == PAD_SLOT_ID`` still produce an output (conv over
    a zeroed prefix) but never write to the cache.

    Args:
        x: Current tokens ``[B, D]`` (one per request).
        weight: Per-channel FIR taps ``[D, W]``.
        conv_cache: Conv state cache ``[num_slots, W - 1, D]`` with
            ``stride(-1) == 1``; updated in place. May be a channel-sliced
            view of a wider buffer.
        cache_indices: Cache slot per request ``[B]``, int32;
            ``PAD_SLOT_ID`` (-1) for padded rows.
        activation: Optional activation applied to the conv output before the
            residual: ``None`` (TML default), ``"silu"`` or ``"swish"``.
        use_residual: Add the residual connection ``y = x + conv(...)``.
        enable_pdl: launch with Programmatic Dependent Launch (Hopper+):
            the kernel waits for its producer before the first load and
            signals dependents after its last store.

    Returns:
        Output tensor ``[B, D]`` with the same dtype as ``x``.
    """
    T, D = x.shape
    W = weight.shape[1]
    assert (
        conv_cache.shape[1] == W - 1
    ), f"conv_cache holds {conv_cache.shape[1]} states per slot, expected {W - 1}"
    assert conv_cache.stride(-1) == 1, "conv_cache must be D-contiguous"

    y = torch.empty_like(x)
    if T == 0:
        return y

    use_silu = activation in ("silu", "swish")
    block_t, block_d, num_warps, num_stages = select_decode_config(T, D)

    grid = (_triton.cdiv(T, block_t), _triton.cdiv(D, block_d))
    _sconv_decode_kernel[grid](
        x,
        weight,
        conv_cache,
        cache_indices,
        y,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        weight.stride(0),
        weight.stride(1),
        conv_cache.stride(0),
        conv_cache.stride(1),
        conv_cache.stride(2),
        T,
        D,
        USE_SILU=use_silu,
        USE_RESIDUAL=use_residual,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        W=W,
        W_POW2=_triton.next_power_of_2(W),
        ENABLE_PDL=enable_pdl,
        num_warps=num_warps,
        num_stages=num_stages,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return y


def sconv_decode_paged(
    x: torch.Tensor,
    weight: torch.Tensor,
    col_pool: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    block_tokens: int,
    col_offset: int = 0,
    col_pool2: torch.Tensor | None = None,
    half_d: int = 0,
    activation: str | None = None,
    use_residual: bool = True,
    enable_pdl: bool = False,
    early_release: bool = False,
) -> torch.Tensor:
    """Single-token decode conv over paged per-token input columns.

    The conv group stores each token's raw conv INPUT column in paged
    storage (SWA semantics, window ``W``), so state is position-addressed:
    prefix hits and MTP rollbacks need no state reconstruction. Taps
    ``0..W-2`` read positions ``pos-W+1..pos-1`` through ``page_table``
    (zeros past holes / sequence start); the current token is read from
    ``x`` and persisted to its own column slice.

    Args:
        x: Current tokens ``[B, D]`` (one per request).
        weight: Per-channel FIR taps ``[D, W]``.
        col_pool: Column pool ``[num_blocks, block_tokens, C]``; the last
            dim must be contiguous, slot/row strides may carry slack (a
            view over larger byte slots stays zero-copy).
        page_table: ``[B, max_blocks]`` int32 block ids for the conv group;
            ``-1`` marks holes (punched or padded rows).
        seq_lens: ``[B]`` int32 lengths INCLUDING the current token.
        block_tokens: Tokens per conv block (the group's block_size).
        col_offset: First channel of this conv site's slice in the column.
        activation: ``None`` (TML default), ``"silu"`` or ``"swish"``.
        use_residual: Add the residual ``y = x + conv(...)``.
        enable_pdl: Launch with Programmatic Dependent Launch (Hopper+).
        col_pool2: Optional second column pool for the fused K+V call: taps
            for channels ``[half_d, D)`` read/persist there instead of
            ``col_pool`` (the two conv streams keep separate pages).
        half_d: Channel split point for ``col_pool2``; 0 disables the split
            (all channels ride ``col_pool``). When set, ``D == 2 * half_d``.
        early_release: Signal PDL dependents right after the column persist
            instead of at kernel end (lets the next kernel's prologue
            overlap the epilogue).

    Returns:
        Conv output ``[B, D]``, same dtype as ``x``.
    """
    assert x.ndim == 2 and weight.ndim == 2 and col_pool.ndim == 3
    assert col_pool.stride(-1) == 1, "col_pool last dim must be contiguous"
    assert col_pool.shape[1] == block_tokens, (
        f"col_pool rows-per-slot {col_pool.shape[1]} != block_tokens " f"{block_tokens}"
    )
    B, D = x.shape
    W = weight.shape[1]
    y = torch.empty_like(x)
    block_t, block_d, num_warps, num_stages = select_decode_config(B, D)
    if half_d:
        # Fused K+V halves: a program must lie wholly in one half.
        assert col_pool2 is not None and D == 2 * half_d
        block_d = min(block_d, half_d)
    grid = (_triton.cdiv(B, block_t), _triton.cdiv(D, block_d))
    kwargs = {}
    if current_platform().is_nvidia:
        kwargs["launch_pdl"] = enable_pdl
    _sconv_decode_paged_kernel[grid](
        x,
        weight,
        col_pool,
        col_pool2 if col_pool2 is not None else col_pool,
        page_table,
        seq_lens,
        y,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        weight.stride(0),
        weight.stride(1),
        col_pool.stride(0),
        col_pool.stride(1),
        col_pool.stride(2),
        page_table.stride(0),
        col_offset,
        B,
        D,
        BT=block_tokens,
        HALF_D=half_d,
        USE_SILU=activation in ("silu", "swish"),
        USE_RESIDUAL=use_residual,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        W=W,
        W_POW2=_triton.next_power_of_2(W),
        ENABLE_PDL=enable_pdl,
        EARLY_RELEASE=early_release,
        num_warps=num_warps,
        num_stages=num_stages,
        **kwargs,
    )
    return y


def sconv_prefill_paged(
    x: torch.Tensor,
    weight: torch.Tensor,
    col_pool: torch.Tensor,
    page_table: torch.Tensor,
    seq_idx: torch.Tensor,
    cu_seqlens: torch.Tensor,
    prefix_lens: torch.Tensor,
    *,
    block_tokens: int,
    col_offset: int = 0,
    col_pool2: torch.Tensor | None = None,
    half_d: int = 0,
    lcm_align: int = 0,
    activation: str | None = None,
    use_residual: bool = True,
) -> torch.Tensor:
    """Varlen prefill conv over paged input columns (see sconv_decode_paged).

    Taps resolve per position: inside the chunk -> ``x`` rows; inside the
    cached prefix -> pool via ``page_table`` (holes read zero); before the
    sequence -> zero. Every chunk token's column is persisted, so no
    separate cache_update pass is needed.

    Args:
        x: Varlen-packed chunk ``[T_total, D]``.
        weight: Per-channel FIR taps ``[D, W]``.
        col_pool / page_table / block_tokens / col_offset: See
            :func:`sconv_decode_paged` (page_table is ``[num_reqs, max_blocks]``).
        seq_idx: ``[T_total]`` int32 request index per token.
        cu_seqlens: ``[num_reqs + 1]`` int32 chunk boundaries.
        prefix_lens: ``[num_reqs]`` int32 cached-prefix length per request.
        lcm_align: Restore-boundary alignment in tokens (the groups' LCM
            block size). When > 0, only columns that future restores or the
            next chunk/decode can tap (last ``W-1`` of each aligned window
            and of the chunk) are persisted; 0 persists every column.
        activation / use_residual: As in :func:`sconv_decode_paged`.

    Returns:
        Conv output ``[T_total, D]``, same dtype as ``x``.
    """
    assert x.ndim == 2 and col_pool.ndim == 3 and col_pool.stride(-1) == 1
    assert col_pool.shape[1] == block_tokens, (
        f"col_pool rows-per-slot {col_pool.shape[1]} != block_tokens " f"{block_tokens}"
    )
    T, D = x.shape
    W = weight.shape[1]
    y = torch.empty_like(x)
    block_t, block_d, num_warps, num_stages = select_prefill_config(T, D)
    if half_d:
        assert col_pool2 is not None and D == 2 * half_d
        block_d = min(block_d, half_d)
    grid = (_triton.cdiv(T, block_t), _triton.cdiv(D, block_d))
    _sconv_prefill_paged_kernel[grid](
        x,
        weight,
        col_pool,
        col_pool2 if col_pool2 is not None else col_pool,
        page_table,
        seq_idx,
        cu_seqlens,
        prefix_lens,
        y,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        weight.stride(0),
        weight.stride(1),
        col_pool.stride(0),
        col_pool.stride(1),
        col_pool.stride(2),
        page_table.stride(0),
        col_offset,
        T,
        D,
        BT=block_tokens,
        HALF_D=half_d,
        LCM_ALIGN=lcm_align,
        USE_SILU=activation in ("silu", "swish"),
        USE_RESIDUAL=use_residual,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        W=W,
        W_POW2=_triton.next_power_of_2(W),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y
