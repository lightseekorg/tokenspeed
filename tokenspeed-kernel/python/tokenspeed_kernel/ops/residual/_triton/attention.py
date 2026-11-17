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

"""Portable Triton residual accumulation for depth attention residuals."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.activation.triton import (
    _attnres_combine_kernel,
    _attnres_partial_kernel,
)
from tokenspeed_kernel.platform import pdl_enabled
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@triton.jit(do_not_specialize=["size"])
def _residual_sum_kernel(Inputs, Output, size, BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    value = tl.full((BLOCK,), 0, tl.float32)
    for source in tl.static_range(len(Inputs)):
        value = value + tl.load(Inputs[source] + index, index < size, 0).to(tl.float32)
    tl.store(Output + index, value, index < size)


@register_kernel(
    "residual",
    "sum_fp32",
    name="triton_residual_sum",
    solution="triton",
    signatures=format_signatures(("source",), "dense", {torch.bfloat16, torch.float32}),
    priority=Priority.PORTABLE,
)
def triton_residual_sum(inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Accumulate contiguous tensors in FP32 and round once to the input dtype."""
    output = torch.empty_like(inputs[0])
    size = output.numel()
    if size:
        _residual_sum_kernel[(triton.cdiv(size, 1024),)](
            inputs, output, size, BLOCK=1024, enable_fp_fusion=False
        )
    return output


@triton.jit(do_not_specialize=["tokens"])
def _weighted_sum_kernel(
    Values,
    Weights,
    Output,
    tokens,
    SOURCES: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < WIDTH
    value = tl.full((BLOCK,), 0, tl.float32)
    for source in tl.static_range(SOURCES):
        weight = tl.load(Weights + source).to(tl.float32)
        row = tl.load(Values + (token * SOURCES + source) * WIDTH + col, mask, 0)
        value = value + row.to(tl.float32) * weight
    tl.store(Output + token * WIDTH + col, value, mask)


@register_kernel(
    "residual",
    "weighted_sum_fp32",
    name="triton_residual_weighted_sum",
    solution="triton",
    signatures=format_signatures(("source",), "dense", {torch.bfloat16, torch.float32}),
    priority=Priority.PORTABLE,
)
def triton_residual_weighted_sum(
    values: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Accumulate ``values[:, s] * weights[s]`` in FP32 in source order."""
    tokens, sources, width = values.shape
    output = torch.empty((tokens, width), dtype=torch.float32, device=values.device)
    if tokens:
        _weighted_sum_kernel[(tokens, triton.cdiv(width, 1024))](
            values,
            weights,
            output,
            tokens,
            SOURCES=sources,
            WIDTH=width,
            BLOCK=1024,
            enable_fp_fusion=False,
        )
    return output


def _attnres_block(width: int) -> int:
    # The split kernels sweep each row in two halves.
    return triton.next_power_of_2(width) // 2


def attnres_query_partials(
    blocks: torch.Tensor,
    weights: torch.Tensor,
    eps: float,
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    normalize_values: bool,
) -> None:
    """Blocks-side online-softmax partials of several queries in one launch.

    Each (token, query) program runs the :func:`attnres_partial` sweep, so a
    query's partial does not depend on which other queries share the launch.

    Args:
        blocks: Contiguous ``[KB, T, H]`` snapshots, ``2 <= H <= 8192``;
            ``KB`` may be 0.
        weights: Contiguous ``[Q, H]`` score weights for the RMS-normalized
            blocks.
        eps: RMS epsilon.
        scratch: Contiguous FP32 ``(m [Q, T], s [Q, T], acc [Q, T, H])``,
            overwritten with the partials.
        normalize_values: Accumulate RMS-normalized blocks instead of raw ones.
    """
    KB, T, H = blocks.shape
    Q = weights.shape[0]
    m, s_, acc = scratch
    if not 2 <= H <= 8192:
        raise ValueError("attnres partials need 2 <= H <= 8192")
    if T == 0:
        return
    enable_pdl = pdl_enabled()
    pdl_kwargs = {"launch_pdl": True} if enable_pdl else {}
    _attnres_partial_kernel[(T, Q)](
        blocks,
        weights,
        m,
        s_,
        acc,
        KB,
        T,
        n_cols=H,
        stride_bk=T * H,
        stride_bt=H,
        eps=eps,
        NORMALIZE_VALUES=normalize_values,
        BLOCK=_attnres_block(H),
        ENABLE_PDL=enable_pdl,
        num_warps=8 if H > 2048 else 4,
        **pdl_kwargs,
    )


def attnres_combine_queries(
    terms: tuple[torch.Tensor, ...],
    weights: torch.Tensor,
    out_norm_weights: torch.Tensor | None,
    eps: float,
    out_eps: float,
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    *,
    candidate_out: torch.Tensor | None,
    normalize_values: bool,
    round_mix: bool,
) -> torch.Tensor:
    """Fold one candidate into the partials of several queries.

    Under PDL the launch releases its dependents before waiting for its
    predecessor, so a dependent may start before that predecessor finishes:
    dependents must read only constant operands (such as weights) before
    their own wait.

    Args:
        terms: One to three contiguous ``[T, H]`` rows whose FP32 sum is the
            candidate.
        weights: Contiguous ``[Q, H]`` score weights of the partials.
        out_norm_weights: Optional contiguous ``[Q, H]`` RMSNorm gains applied
            to each query's mix.
        eps: RMS epsilon of the scores (and of normalized values).
        out_eps: RMSNorm epsilon of the outputs.
        scratch: ``(m, s, acc)`` from :func:`attnres_query_partials` for the
            same queries.
        out: Contiguous ``[Q, T, H]`` destination.
        candidate_out: When given, the term sum starts from zero, rounds once
            to this ``[T, H]`` tensor's dtype, is stored there, and the rounded
            value is mixed. When ``None`` the FP32 sum is mixed unrounded.
        normalize_values: Mix the RMS-normalized candidate (the partials must
            have normalized values too).
        round_mix: Round each mix to BF16 before the out-norm and store.

    Returns:
        ``out``.
    """
    Q, H = weights.shape
    T = terms[0].shape[0]
    if not 2 <= H <= 8192:
        raise ValueError("attnres combine needs 2 <= H <= 8192")
    if T == 0:
        return out
    m, s_, acc = scratch
    enable_pdl = pdl_enabled()
    pdl_kwargs = {"launch_pdl": True} if enable_pdl else {}
    first = terms[0]
    _attnres_combine_kernel[(T, Q)](
        first,
        terms[1] if len(terms) > 1 else first,
        terms[2] if len(terms) > 2 else first,
        first if candidate_out is None else candidate_out,
        weights,
        weights if out_norm_weights is None else out_norm_weights,
        m,
        s_,
        acc,
        out,
        T,
        n_cols=H,
        stride_p=H,
        stride_o=H,
        eps=eps,
        out_eps=out_eps,
        NUM_TERMS=len(terms),
        STORE_CANDIDATE=candidate_out is not None,
        NORMALIZE_VALUES=normalize_values,
        ROUND_MIX=round_mix,
        HAS_OUTNORM=out_norm_weights is not None,
        BLOCK=_attnres_block(H),
        # A program holds one row: at 4608 wide, 4 warps finish the 8-token
        # mixes behind a PDL producer 0.2-0.5 us sooner than 8 on B300.
        num_warps=4,
        ENABLE_PDL=enable_pdl,
        # The successor projections read only their weights before waiting.
        TRIGGER_FIRST=True,
        **pdl_kwargs,
    )
    return out


@triton.jit
def _load_row_tiles(ptr, col0, col1, mask0, mask1):
    """Both column tiles of one row, in the stored dtype."""
    a = tl.load(ptr + col0, mask=mask0, other=0.0)
    b = tl.load(ptr + col1, mask=mask1, other=0.0)
    return a, b


@triton.jit(do_not_specialize=["n_blocks", "n_tokens", "first_token", "n_queries"])
def _attnres_mix_kernel(
    blocks_ptr,  # [KB, T, H]
    prefix_ptr,  # [T, H] first candidate term
    term1_ptr,  # [T, H] second candidate term or dummy
    term2_ptr,  # [T, H] third candidate term or dummy
    cand_ptr,  # [T, H] rounded candidate destination or dummy
    wp_ptr,  # [Q, H] score weights
    outw_ptr,  # [Q, H] out-norm weight or dummy
    out_ptr,  # [Q, T, H]
    n_blocks,
    n_tokens,
    first_token,
    n_queries,
    n_cols: tl.constexpr,
    eps,
    out_eps,
    NUM_TERMS: tl.constexpr,
    STORE_CANDIDATE: tl.constexpr,
    NORMALIZE_VALUES: tl.constexpr,
    HAS_OUTNORM: tl.constexpr,
    BLOCK0: tl.constexpr,
    BLOCK1: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    """The block sweep of ``_attnres_partial_kernel`` and the candidate fold of
    ``_attnres_combine_kernel`` in one program per (token, query): the online
    softmax statistics stay in registers instead of FP32 scratch.

    The row is split into exact power-of-two tiles (``BLOCK0 + BLOCK1``
    covers ``n_cols`` with less than ``BLOCK1`` lanes masked), so a 4608-wide
    row runs 4096 + 512 lanes instead of two 4096 halves. Under PDL the
    weights load before the wait and dependents are released first: the
    successor projections read only their weights before their own wait.
    """
    # A token's queries run in adjacent programs, which share its block rows in L2.
    program = tl.program_id(0)
    t = first_token + (program // n_queries).to(tl.int64)
    q = program % n_queries
    col0 = tl.arange(0, BLOCK0)
    col1 = BLOCK0 + tl.arange(0, BLOCK1)
    mask0 = col0 < n_cols
    mask1 = col1 < n_cols
    wp0, wp1 = _load_row_tiles(wp_ptr + q * n_cols, col0, col1, mask0, mask1)
    wp0 = wp0.to(tl.float32)
    wp1 = wp1.to(tl.float32)
    if HAS_OUTNORM:
        ow0, ow1 = _load_row_tiles(outw_ptr + q * n_cols, col0, col1, mask0, mask1)
        ow0 = ow0.to(tl.float32)
        ow1 = ow1.to(tl.float32)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()

    acc0 = tl.zeros([BLOCK0], tl.float32)
    acc1 = tl.zeros([BLOCK1], tl.float32)
    m_run = -float("inf")
    s_run = 0.0
    # Load the next block row before reducing this one, so loads overlap reductions.
    row = t * n_cols
    stride_b = n_tokens.to(tl.int64) * n_cols
    next0, next1 = _load_row_tiles(
        blocks_ptr + row, col0, col1, mask0 & (n_blocks > 0), mask1 & (n_blocks > 0)
    )
    b = 0
    while b < n_blocks:
        v0 = next0.to(tl.float32)
        v1 = next1.to(tl.float32)
        more = b + 1 < n_blocks
        next0, next1 = _load_row_tiles(
            blocks_ptr + (b + 1) * stride_b + row,
            col0,
            col1,
            mask0 & more,
            mask1 & more,
        )
        sq = tl.sum(v0 * v0) + tl.sum(v1 * v1)
        dot = tl.sum(v0 * wp0) + tl.sum(v1 * wp1)
        rsig = tl.math.rsqrt(sq / n_cols + eps)
        logit = dot * rsig
        m_new = tl.maximum(m_run, logit)
        corr = tl.exp(m_run - m_new)
        wgt = tl.exp(logit - m_new)
        if NORMALIZE_VALUES:
            v0 = v0 * rsig
            v1 = v1 * rsig
        acc0 = acc0 * corr + wgt * v0
        acc1 = acc1 * corr + wgt * v1
        s_run = s_run * corr + wgt
        m_run = m_new
        b += 1

    v0, v1 = _load_row_tiles(prefix_ptr + row, col0, col1, mask0, mask1)
    v0 = v0.to(tl.float32)
    v1 = v1.to(tl.float32)
    if STORE_CANDIDATE:
        v0 = 0.0 + v0
        v1 = 0.0 + v1
    if NUM_TERMS >= 2:
        a0, a1 = _load_row_tiles(term1_ptr + row, col0, col1, mask0, mask1)
        v0 += a0.to(tl.float32)
        v1 += a1.to(tl.float32)
    if NUM_TERMS == 3:
        a0, a1 = _load_row_tiles(term2_ptr + row, col0, col1, mask0, mask1)
        v0 += a0.to(tl.float32)
        v1 += a1.to(tl.float32)
    if STORE_CANDIDATE:
        c0 = v0.to(cand_ptr.dtype.element_ty)
        c1 = v1.to(cand_ptr.dtype.element_ty)
        tl.store(cand_ptr + row + col0, c0, mask=mask0 & (q == 0))
        tl.store(cand_ptr + row + col1, c1, mask=mask1 & (q == 0))
        v0 = c0.to(tl.float32)
        v1 = c1.to(tl.float32)
    sq = tl.sum(v0 * v0) + tl.sum(v1 * v1)
    dot = tl.sum(v0 * wp0) + tl.sum(v1 * wp1)
    rsig = tl.math.rsqrt(sq / n_cols + eps)
    logit_p = dot * rsig
    m = tl.maximum(m_run, logit_p)
    corr = tl.exp(m_run - m)
    w_p = tl.exp(logit_p - m)
    inv_s = 1.0 / (s_run * corr + w_p)
    if NORMALIZE_VALUES:
        v0 = v0 * rsig
        v1 = v1 * rsig
    mix0 = (acc0 * corr + w_p * v0) * inv_s
    mix1 = (acc1 * corr + w_p * v1) * inv_s
    if HAS_OUTNORM:
        mix_sq = tl.sum(mix0 * mix0) + tl.sum(mix1 * mix1)
        rsig_mix = tl.math.rsqrt(tl.div_rn(mix_sq, n_cols) + out_eps)
        mix0 = mix0 * rsig_mix * ow0
        mix1 = mix1 * rsig_mix * ow1
    out_ptr += q.to(tl.int64) * n_tokens * n_cols + row
    tl.store(out_ptr + col0, mix0.to(out_ptr.dtype.element_ty), mask=mask0)
    tl.store(out_ptr + col1, mix1.to(out_ptr.dtype.element_ty), mask=mask1)


def _attnres_mix_tiles(width: int) -> tuple[int, int]:
    """Power-of-two tiles that cover ``width`` with the fewest masked lanes."""
    low = 1 << (width.bit_length() - 1)
    if low == width:
        return width // 2, width // 2
    return low, triton.next_power_of_2(width - low)


def attnres_mix_queries(
    blocks: torch.Tensor,
    terms: tuple[torch.Tensor, ...],
    weights: torch.Tensor,
    out_norm_weights: torch.Tensor | None,
    eps: float,
    out_eps: float,
    out: torch.Tensor,
    *,
    candidate_out: torch.Tensor | None,
    normalize_values: bool,
    tokens: tuple[int, int],
) -> torch.Tensor:
    """:func:`attnres_query_partials` and :func:`attnres_combine_queries` in
    one launch, without their FP32 scratch.

    Under PDL the launch releases its dependents before waiting for its
    predecessor: dependents must read only constant operands before their
    own wait.

    Args:
        blocks: Contiguous ``[KB, T, H]`` snapshots, ``2 <= H <= 8192``;
            ``KB`` may be 0.
        terms: One to three contiguous ``[T, H]`` rows whose FP32 sum is the
            candidate.
        weights: Contiguous ``[Q, H]`` score weights.
        out_norm_weights: Optional contiguous ``[Q, H]`` RMSNorm gains applied
            to each query's mix.
        eps: RMS epsilon of the scores (and of normalized values).
        out_eps: RMSNorm epsilon of the outputs.
        out: Contiguous ``[Q, T, H]`` destination.
        candidate_out: When given, the term sum starts from zero, rounds once
            to this ``[T, H]`` tensor's dtype, is stored there, and the rounded
            value is mixed.
        normalize_values: Mix RMS-normalized blocks and candidate.
        tokens: ``(first, count)``: mix only these rows of the ``T`` (and
            store only their candidate rows); the other rows of ``out`` are
            left alone.

    Returns:
        ``out``.
    """
    Q, H = weights.shape
    T = terms[0].shape[0]
    if not 2 <= H <= 8192:
        raise ValueError("attnres mix needs 2 <= H <= 8192")
    first_token, count = tokens
    if not 0 <= first_token <= first_token + count <= T:
        raise ValueError("attnres mix tokens must lie within the rows")
    if count == 0 or Q == 0:
        return out
    block0, block1 = _attnres_mix_tiles(H)
    enable_pdl = pdl_enabled()
    first = terms[0]
    _attnres_mix_kernel[(count * Q,)](
        blocks,
        first,
        terms[1] if len(terms) > 1 else first,
        terms[2] if len(terms) > 2 else first,
        first if candidate_out is None else candidate_out,
        weights,
        weights if out_norm_weights is None else out_norm_weights,
        out,
        blocks.shape[0],
        T,
        first_token,
        Q,
        n_cols=H,
        eps=eps,
        out_eps=out_eps,
        NUM_TERMS=len(terms),
        STORE_CANDIDATE=candidate_out is not None,
        NORMALIZE_VALUES=normalize_values,
        HAS_OUTNORM=out_norm_weights is not None,
        BLOCK0=block0,
        BLOCK1=block1,
        ENABLE_PDL=enable_pdl,
        # 4 warps hold a 4608-wide row fastest on B300 (2089 tokens: 39 us
        # for four blocks against 51 us with 8 warps).
        num_warps=4,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return out
