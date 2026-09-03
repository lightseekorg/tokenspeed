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

"""Fused kernels for the PLE (predictive latent embedding) block.

Every launcher here works on the packed-free layout: a flat
``[total_tokens, ...]`` tensor plus per-token ``(req, col)`` coordinates and
per-request ``starts``, with the pre-batch state carried in a separate
``[bs, ...]`` tensor. Kernels that span the request boundary address the
virtual sequence ``[carried state | tokens]`` and pick each tap's source from
the two, so the padded ``[bs, max_len, ...]`` intermediates of the eager path
are never built.

Block configurations are static heuristics, so every launcher is CUDA-graph
capturable. PDL is enabled per launch from :func:`pdl_enabled`, matching the
rest of the package.
"""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel._triton import triton as _triton
from tokenspeed_kernel.ops.ple.triton import (
    _ngram_ids_kernel,
    _ple_conv_final_kernel,
    _ple_dilated_conv_kernel,
    _ple_gate_norm_kernel,
    _ple_page_gather_kernel,
    _ple_page_scatter_kernel,
)
from tokenspeed_kernel.platform import pdl_enabled

__all__ = [
    "ple_conv_sequences",
    "ple_gate_norm",
    "ple_ngram_ids",
    "ple_page_gather",
    "ple_page_scatter",
]


def ple_ngram_ids(
    input_ids: torch.Tensor,
    initial: torch.Tensor,
    req: torch.Tensor,
    col: torch.Tensor,
    starts: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
    *,
    ngram_size: int,
    heads_per_ngram: int,
    eos_token_id: int,
    need_tail: bool = False,
    tail_out: torch.Tensor | None = None,
    tail_block_rows: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Hash one n-gram id per head straight from the flat token stream.

    Args:
        input_ids: Flat token ids shaped ``[total_tokens]``.
        initial: Carried context ids shaped ``[bs, ngram_size - 1]``.
        req: Owning request index per token row.
        col: Position within the request per token row.
        starts: First flat row of each request, shaped ``[bs]``.
        multipliers: Int64 SplitMix multipliers shaped ``[ngram_size]``.
        vocab_sizes: Per-head hash moduli shaped ``[ngram_heads]``.
        offsets: Per-head id base shaped ``[ngram_heads]``.
        ngram_size: Window width, including the anchor.
        heads_per_ngram: Independent hash heads per window position.
        eos_token_id: Token that blocks the window from reaching further left.
        need_tail: Also return the packed raw trailing windows.
        tail_out: Optional verify scratch receiving carried and trailing contexts.
        tail_block_rows: Rows reserved per request in ``tail_out``.

    Returns:
        A pair of the int64 ids shaped ``[total_tokens, ngram_heads]`` and, when
        ``need_tail`` is set, the raw window shaped
        ``[total_tokens, ngram_size - 1]`` (``None`` otherwise).
    """

    if need_tail and tail_out is not None:
        raise ValueError("need_tail and tail_out are mutually exclusive")
    total = input_ids.shape[0]
    batch_size = initial.shape[0]
    device = input_ids.device
    context_len = ngram_size - 1
    ngram_heads = context_len * heads_per_ngram
    ids = torch.empty((total, ngram_heads), dtype=torch.long, device=device)
    tail = (
        torch.empty((total, context_len), dtype=torch.long, device=device)
        if need_tail
        else None
    )
    scatter_tail = tail_out is not None
    if scatter_tail:
        if (
            tail_out.device != device
            or tail_out.dtype != initial.dtype
            or not tail_out.is_contiguous()
            or tail_out.ndim != 2
            or tail_out.shape[1] != context_len
        ):
            raise ValueError("tail_out must be a contiguous context scratch tensor")
        if tail_block_rows <= 0 or tail_out.shape[0] < batch_size * tail_block_rows:
            raise ValueError("tail_out is smaller than the requested block layout")
    work_items = max(total, batch_size) if scatter_tail else total
    if work_items == 0:
        return ids, tail
    block = 256
    use_pdl = pdl_enabled()
    _ngram_ids_kernel[(_triton.cdiv(work_items, block),)](
        input_ids.contiguous(),
        initial.contiguous(),
        req,
        col,
        starts,
        multipliers,
        vocab_sizes,
        offsets,
        ids,
        tail_out if scatter_tail else tail if need_tail else ids,
        total,
        batch_size,
        tail_block_rows,
        eos_token_id,
        N=ngram_size,
        HPN=heads_per_ngram,
        H=ngram_heads,
        WRITE_TAIL=need_tail or scatter_tail,
        SCATTER_TAIL=scatter_tail,
        ENABLE_PDL=use_pdl,
        BLOCK=block,
        **({"launch_pdl": True} if use_pdl else {}),
    )
    return ids, tail


def ple_page_gather(
    field: torch.Tensor,
    page_ids: torch.Tensor,
    row_stride: int,
    default: int | float = 0,
) -> torch.Tensor:
    """Read one cache row per page id, with null pages read as ``default``.

    Args:
        field: Cache field whose first dimension is indexed by page id.
        page_ids: One page id per output row; id 0 is the null page.
        row_stride: Element stride between pages, which the arena may pad past
            the row's own extent.
        default: Value substituted for every null-page row.

    Returns:
        A tensor shaped ``[len(page_ids), *field.shape[1:]]``.
    """

    rows = page_ids.shape[0]
    out = field.new_empty((rows, *field.shape[1:]))
    if rows == 0:
        return out
    numel = out[0].numel()
    block = min(1024, _triton.next_power_of_2(numel))
    use_pdl = pdl_enabled()
    _ple_page_gather_kernel[(rows, _triton.cdiv(numel, block))](
        field,
        page_ids,
        out,
        default,
        row_stride,
        numel,
        ENABLE_PDL=use_pdl,
        BLOCK=block,
        **({"launch_pdl": True} if use_pdl else {}),
    )
    return out


def ple_page_scatter(
    field: torch.Tensor,
    page_ids: torch.Tensor,
    values: torch.Tensor,
    row_stride: int,
) -> None:
    """Store one row per page id, skipping null pages.

    Args:
        field: Cache field whose first dimension is indexed by page id.
        page_ids: One page id per input row; id 0 is the null page.
        values: Rows to store, shaped ``[len(page_ids), *field.shape[1:]]``.
        row_stride: Element stride between pages.
    """

    rows = page_ids.shape[0]
    if rows == 0:
        return
    numel = values[0].numel()
    block = min(1024, _triton.next_power_of_2(numel))
    use_pdl = pdl_enabled()
    _ple_page_scatter_kernel[(rows, _triton.cdiv(numel, block))](
        field,
        page_ids,
        values.contiguous(),
        row_stride,
        numel,
        ENABLE_PDL=use_pdl,
        BLOCK=block,
        **({"launch_pdl": True} if use_pdl else {}),
    )


def ple_conv_sequences(
    values: torch.Tensor,
    initial: torch.Tensor,
    weight: torch.Tensor,
    req: torch.Tensor,
    col: torch.Tensor,
    lengths: torch.Tensor,
    starts: torch.Tensor,
    *,
    total_tokens: int,
    batch_size: int,
    dilation: int,
    kernel_size: int,
    state_len: int,
    add_terms: tuple[torch.Tensor, ...] = (),
    windows: torch.Tensor | None = None,
    windows_block_rows: int = 0,
    scatter_windows: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dilated depthwise conv + SiLU, plus the state carried to the next step.

    Args:
        values: Flat conv input shaped ``[total_tokens, channels]``.
        initial: Carried conv state shaped ``[bs, channels, state_len]``.
        weight: Depthwise taps shaped ``[channels, kernel_size]``.
        req: Owning request index per token row.
        col: Position within the request per token row.
        lengths: Token count per request, shaped ``[bs]``.
        starts: First flat row of each request, shaped ``[bs]``.
        total_tokens: Real rows of ``values``. A padded-bucket replay passes
            more rows than that, whose tail is filler: those output rows stay
            untouched rather than being computed.
        batch_size: Number of requests, i.e. the row count of ``initial``.
        dilation: Stride between consecutive taps.
        kernel_size: Number of taps.
        state_len: Width of the carried state, ``(kernel_size - 1) * dilation``.
        add_terms: Up to two full-width addends folded into the epilogue, each
            applied in order with a round to the output dtype in between. Only
            the last dimension has to be dense.
        windows: Destination for sliding state windows. ``None`` skips them.
        windows_block_rows: Rows per request in ``windows`` when scattering.
        scatter_windows: Place carried state in each block's first row and token
            windows at ``req * windows_block_rows + 1 + col`` instead of packed
            rows.

    Returns:
        A triple of the conv output shaped like ``values``, the trailing state
        shaped ``[bs, channels, state_len]``, and the window tensor (an empty
        placeholder when ``windows`` is ``None``).
    """

    if len(add_terms) > 2:
        raise ValueError("the fused conv epilogue folds at most two addends")
    channels = values.shape[-1]
    initial_c = initial.contiguous()
    values_c = values.contiguous()
    conv_output = torch.empty_like(values_c)
    # Row strides let strided addends feed the kernel without a contiguity
    # copy; only the last dim must be dense. Unused slots alias values_c,
    # which the ADD_* constexprs keep unread.
    addends = [
        term if term.stride(-1) == 1 else term.contiguous() for term in add_terms
    ]
    gated = addends[0] if len(addends) > 0 else values_c
    residual = addends[1] if len(addends) > 1 else values_c
    if windows is None:
        if scatter_windows:
            raise ValueError("scatter_windows requires a windows destination")
        # Dummy target: WRITE_WINDOWS=False never stores through it. Decode
        # and non-verify prefill skip the [T, C, state_len] materialization.
        windows = values.new_empty((0, channels, state_len))
        write_windows = False
    else:
        # The kernel addresses rows as a dense [rows, C, state_len] block, and
        # a copy would silently drop the writes, so refuse anything else
        # rather than repack.
        if (
            windows.device != values.device
            or not windows.is_contiguous()
            or windows.ndim != 3
            or tuple(windows.shape[1:]) != (channels, state_len)
        ):
            raise ValueError("windows must be a contiguous state-window tensor")
        if scatter_windows and (
            windows_block_rows <= 0
            or windows.shape[0] < batch_size * windows_block_rows
        ):
            raise ValueError("windows is smaller than the requested block layout")
        write_windows = True
    block_c = 256
    use_pdl = pdl_enabled()
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}
    if total_tokens:
        grid = (total_tokens, _triton.cdiv(channels, block_c))
        _ple_dilated_conv_kernel[grid](
            values_c,
            initial_c,
            weight.contiguous(),
            req,
            col,
            starts,
            conv_output,
            windows,
            gated,
            residual,
            windows_block_rows,
            gated.stride(0),
            residual.stride(0),
            channels,
            D=dilation,
            K=kernel_size,
            STATE=state_len,
            WRITE_WINDOWS=write_windows,
            SCATTER_WINDOWS=scatter_windows,
            ADD_GATED=len(addends) > 0,
            ADD_RESIDUAL=len(addends) > 1,
            ENABLE_PDL=use_pdl,
            BLOCK_C=block_c,
            **pdl_kwargs,
        )
    final_conv = values.new_empty((batch_size, channels, state_len))
    if batch_size:
        _ple_conv_final_kernel[(batch_size, _triton.cdiv(channels, block_c))](
            values_c,
            initial_c,
            lengths,
            starts,
            final_conv,
            windows,
            windows_block_rows,
            channels,
            STATE=state_len,
            WRITE_CARRIED=scatter_windows,
            ENABLE_PDL=use_pdl,
            BLOCK_C=block_c,
            **pdl_kwargs,
        )
    return conv_output, final_conv, windows


def ple_gate_norm(
    key: torch.Tensor,
    query: torch.Tensor,
    value: torch.Tensor,
    key_weight: torch.Tensor,
    query_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    *,
    hc_count: int,
    hidden_size: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse the three grouped Gemma RMSNorms and the query-key gate.

    Args:
        key: Key stream shaped ``[total_tokens, hc_count * hidden_size]``. Only
            the last dimension has to be dense.
        query: Query stream with the same shape as ``key``.
        value: Value stream shaped ``[total_tokens, hidden_size]``, shared by
            every branch.
        key_weight: Grouped Gemma gamma for the key norm, shaped
            ``[hc_count * hidden_size]``.
        query_weight: Grouped Gemma gamma for the query norm.
        conv_weight: Grouped Gemma gamma for the conv-input norm.
        hc_count: Number of hyper-connection branches.
        hidden_size: Width of one branch.
        eps: Epsilon shared by the three norms.

    Returns:
        A pair of the gated value stream and its normalized form, both shaped
        ``[total_tokens, hc_count * hidden_size]``.
    """

    total = key.shape[0]
    gated = key.new_empty((total, hc_count * hidden_size))
    normalized = torch.empty_like(gated)
    if total == 0:
        return gated, normalized
    # Row strides let kv_proj's split views feed the kernel without a
    # contiguity copy; only the last dim must be dense.
    if key.stride(-1) != 1:
        key = key.contiguous()
    if query.stride(-1) != 1:
        query = query.contiguous()
    if value.stride(-1) != 1:
        value = value.contiguous()
    use_pdl = pdl_enabled()
    _ple_gate_norm_kernel[(total, hc_count)](
        key,
        query,
        value,
        key_weight,
        query_weight,
        conv_weight,
        gated,
        normalized,
        eps,
        1.0 / math.sqrt(hidden_size),
        key.stride(0),
        query.stride(0),
        value.stride(0),
        HC=hc_count,
        D=hidden_size,
        ENABLE_PDL=use_pdl,
        BLOCK_D=_triton.next_power_of_2(hidden_size),
        **({"launch_pdl": True} if use_pdl else {}),
    )
    return gated, normalized
