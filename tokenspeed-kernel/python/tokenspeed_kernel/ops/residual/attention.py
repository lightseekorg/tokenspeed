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

"""Depth-wise attention residuals with FP32 mixtures and explicit rounding."""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel.ops.residual._triton.attention import (
    attnres_combine_queries,
    attnres_mix_queries,
    attnres_query_partials,
)
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


def _check_eps(eps: float) -> None:
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("attention residual needs a positive finite epsilon")


def _check_rows(tensors: tuple[torch.Tensor, ...], like: torch.Tensor) -> None:
    for x in tensors:
        if (
            x.shape != like.shape
            or x.dtype != like.dtype
            or x.device != like.device
            or not x.is_contiguous()
        ):
            raise ValueError(
                "attention residual sources must share shape, dtype, device and contiguous layout"
            )


def _check_sources(blocks: torch.Tensor, weights: torch.Tensor) -> None:
    _, _, hidden = blocks.shape
    queries = weights.shape[0]
    if (
        blocks.dtype not in (torch.bfloat16, torch.float32)
        or not blocks.is_contiguous()
        or weights.shape != (queries, hidden)
        or weights.dtype != torch.float32
        or not weights.is_contiguous()
        or weights.device != blocks.device
    ):
        raise ValueError(
            "partials need contiguous [S, T, H] sources and FP32 [Q, H] weights"
        )


def _check_candidate(
    terms: tuple[torch.Tensor, ...],
    weights: torch.Tensor,
    candidate_out: torch.Tensor | None,
    norm_weights: torch.Tensor | None,
    output_dtype: torch.dtype,
) -> None:
    if not 1 <= len(terms) <= 3:
        raise ValueError("attention residual combine needs one to three terms")
    first = terms[0]
    _check_rows(terms, first)
    if candidate_out is not None:
        _check_rows((candidate_out,), first)
    queries, hidden = weights.shape
    if first.shape[1] != hidden:
        raise ValueError("terms and weights disagree on shape")
    if not weights.is_contiguous():
        raise ValueError("weights must be contiguous")
    if norm_weights is None and output_dtype != torch.float32:
        raise ValueError("an unnormalized mixture is returned in FP32")
    if norm_weights is not None and (
        norm_weights.shape != (queries, hidden) or not norm_weights.is_contiguous()
    ):
        raise ValueError("norm_weights must be contiguous [queries, hidden]")


def attention_residual_partials(
    blocks: torch.Tensor,
    weights: torch.Tensor,
    *,
    normalize_values: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Online-softmax statistics of each query over completed sources.

    Each source's score is its RMS-normalized row dotted with a query weight.
    The statistics fold a later candidate in with
    :func:`attention_residual_combine`.

    Args:
        blocks: Contiguous BF16 or FP32 ``[sources, tokens, hidden]``;
            ``sources`` may be 0.
        weights: Contiguous FP32 ``[queries, hidden]`` score weights (the
            pseudo-queries already scaled by ``hidden**-0.5``).
        normalize_values: Accumulate RMS-normalized sources instead of raw ones.
        eps: Finite positive epsilon added to source mean squares.

    Returns:
        FP32 ``(max [queries, tokens], sum [queries, tokens],
        accumulator [queries, tokens, hidden])``: the running maximum score,
        the sum of ``exp(score - max)`` and the matching weighted source sum.
    """
    _check_eps(eps)
    _check_sources(blocks, weights)
    sources, tokens, hidden = blocks.shape
    queries = weights.shape[0]
    device = blocks.device
    maximum = torch.empty((queries, tokens), dtype=torch.float32, device=device)
    total = torch.empty((queries, tokens), dtype=torch.float32, device=device)
    accumulator = torch.empty(
        (queries, tokens, hidden), dtype=torch.float32, device=device
    )
    if blocks.is_cuda:
        attnres_query_partials(
            blocks,
            weights,
            eps,
            (maximum, total, accumulator),
            normalize_values=normalize_values,
        )
        return maximum, total, accumulator
    # The host reference runs the same recurrence in FP64.
    running_max = torch.full((queries, tokens), -math.inf, dtype=torch.float64)
    running_sum = torch.zeros((queries, tokens), dtype=torch.float64)
    running_acc = torch.zeros((queries, tokens, hidden), dtype=torch.float64)
    for source in blocks.double():
        inv_rms = torch.rsqrt(source.square().mean(dim=-1) + eps)
        logits = (weights.double() @ source.T) * inv_rms
        new_maximum = torch.maximum(running_max, logits)
        correction = torch.exp(running_max - new_maximum)
        weight = torch.exp(logits - new_maximum)
        value = source * inv_rms.unsqueeze(-1) if normalize_values else source
        running_acc = (
            running_acc * correction.unsqueeze(-1) + weight.unsqueeze(-1) * value
        )
        running_sum = running_sum * correction + weight
        running_max = new_maximum
    maximum.copy_(running_max)
    total.copy_(running_sum)
    accumulator.copy_(running_acc)
    return maximum, total, accumulator


def attention_residual_combine(
    terms: tuple[torch.Tensor, ...],
    weights: torch.Tensor,
    partials: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    candidate_out: torch.Tensor | None,
    normalize_values: bool,
    eps: float,
    norm_weights: torch.Tensor | None,
    norm_eps: float,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Fold the current candidate into the partials and finish each mixture.

    Under PDL the next kernel may start before this one's predecessor
    finishes; it must read only constant operands before its own wait.

    The candidate is the FP32 sum of ``terms`` in order. With
    ``candidate_out`` the sum starts from zero, rounds once to
    ``candidate_out``'s dtype, is stored there, and the rounded value is
    mixed, exactly like :func:`residual_sum`. Otherwise the FP32 sum is
    mixed unrounded. The mixture stays in FP32 until the optional RMSNorm
    and the single output store.

    Args:
        terms: One to three contiguous ``[tokens, hidden]`` rows of the
            partials' dtype.
        weights: The contiguous FP32 ``[queries, hidden]`` score weights of
            the partials.
        partials: Contiguous statistics from
            :func:`attention_residual_partials` with the same weights and
            ``normalize_values``.
        candidate_out: Optional ``[tokens, hidden]`` destination of the rounded
            candidate.
        normalize_values: Mix the RMS-normalized candidate.
        eps: Epsilon of the score and value normalization.
        norm_weights: Optional ``[queries, hidden]`` RMSNorm gains; the mixture
            is then normalized in FP32 and multiplied by the gain.
        norm_eps: Epsilon of that RMSNorm.
        output_dtype: Dtype of the returned rows. FP32 without ``norm_weights``.

    Returns:
        ``[queries, tokens, hidden]`` mixtures (normalized with
        ``norm_weights``), rounded once to ``output_dtype``.
    """
    _check_eps(eps)
    _check_eps(norm_eps)
    _check_candidate(terms, weights, candidate_out, norm_weights, output_dtype)
    first = terms[0]
    queries = weights.shape[0]
    if (
        partials[0].shape != (queries, first.shape[0])
        or partials[1].shape != partials[0].shape
        or partials[2].shape != (queries, *first.shape)
    ):
        raise ValueError("partials, weights and terms disagree on shape")
    if not all(x.is_contiguous() for x in partials):
        raise ValueError("partials must be contiguous")
    out = torch.empty((queries, *first.shape), dtype=output_dtype, device=first.device)
    if first.is_cuda:
        return attnres_combine_queries(
            terms,
            weights,
            norm_weights,
            eps,
            norm_eps,
            partials,
            out,
            candidate_out=candidate_out,
            normalize_values=normalize_values,
            round_mix=False,
        )
    if candidate_out is not None:
        candidate_out.copy_(residual_sum(terms))
        value = candidate_out.float()
    else:
        value = first.float()
        for term in terms[1:]:
            value = value + term.float()
    # The FP32 candidate is exact in FP64; the host reference mixes in FP64.
    value = value.double()
    maximum, total, accumulator = (x.double() for x in partials)
    inv_rms = torch.rsqrt(value.square().mean(dim=-1) + eps)
    logits = (weights.double() @ value.T) * inv_rms
    new_maximum = torch.maximum(maximum, logits)
    correction = torch.exp(maximum - new_maximum)
    weight = torch.exp(logits - new_maximum)
    if normalize_values:
        value = value * inv_rms.unsqueeze(-1)
    mixture = (
        accumulator * correction.unsqueeze(-1) + weight.unsqueeze(-1) * value
    ) / (total * correction + weight).unsqueeze(-1)
    if norm_weights is not None:
        mixture = mixture.float().double()
        mixture = mixture * torch.rsqrt(
            mixture.square().mean(dim=-1, keepdim=True) + norm_eps
        )
        mixture = mixture * norm_weights.double().unsqueeze(1)
    out.copy_(mixture)
    return out


def attention_residual_mix(
    blocks: torch.Tensor,
    terms: tuple[torch.Tensor, ...],
    weights: torch.Tensor,
    *,
    candidate_out: torch.Tensor | None,
    normalize_values: bool,
    eps: float,
    norm_weights: torch.Tensor | None,
    norm_eps: float,
    output_dtype: torch.dtype,
    tokens: tuple[int, int] | None,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """:func:`attention_residual_partials` over ``blocks`` folded into
    :func:`attention_residual_combine` with ``terms``, in one pass.

    On CUDA one launch keeps each (token, query) program's online-softmax
    statistics in registers instead of FP32 scratch; it computes the same
    equations with another FP32 summation order. Under PDL the next kernel
    may start before this one's predecessor finishes; it must read only
    constant operands before its own wait.

    Args:
        blocks: Contiguous BF16 or FP32 ``[sources, tokens, hidden]``;
            ``sources`` may be 0.
        terms: One to three contiguous ``[tokens, hidden]`` rows; their FP32
            sum is the candidate.
        weights: Contiguous FP32 ``[queries, hidden]`` score weights.
        candidate_out: Optional ``[tokens, hidden]`` destination of the rounded
            candidate, as in :func:`attention_residual_combine`.
        normalize_values: Mix RMS-normalized sources and candidate.
        eps: Epsilon of the score and value normalization.
        norm_weights: Optional ``[queries, hidden]`` RMSNorm gains.
        norm_eps: Epsilon of that RMSNorm.
        output_dtype: Dtype of the returned rows. FP32 without ``norm_weights``.
        tokens: ``(first, count)`` to mix only those rows (and store only
            their candidate rows), as a sequence-parallel rank does with its
            share; None mixes every row.
        out: Contiguous ``[queries, tokens, hidden]`` destination of
            ``output_dtype`` (it may alias a multicast buffer); rows outside
            ``tokens`` are left alone. None allocates one.

    Returns:
        ``[queries, tokens, hidden]`` mixtures, as
        :func:`attention_residual_combine` returns them, in ``out`` when
        given.
    """
    num_tokens = terms[0].shape[0]
    first_token, count = (0, num_tokens) if tokens is None else tokens
    if not 0 <= first_token <= first_token + count <= num_tokens:
        raise ValueError("attention residual mix tokens must lie within the rows")
    shape = (weights.shape[0], *terms[0].shape)
    if out is not None and (
        out.shape != shape
        or out.dtype != output_dtype
        or out.device != terms[0].device
        or not out.is_contiguous()
    ):
        raise ValueError("out must be a contiguous [queries, tokens, hidden] tensor")
    if not blocks.is_cuda:
        rows = slice(first_token, first_token + count)
        partials = attention_residual_partials(
            blocks[:, rows].contiguous(),
            weights,
            normalize_values=normalize_values,
            eps=eps,
        )
        mixed = attention_residual_combine(
            tuple(term[rows] for term in terms),
            weights,
            partials,
            candidate_out=None if candidate_out is None else candidate_out[rows],
            normalize_values=normalize_values,
            eps=eps,
            norm_weights=norm_weights,
            norm_eps=norm_eps,
            output_dtype=output_dtype,
        )
        if out is None:
            if tokens is None:
                return mixed
            out = torch.empty(shape, dtype=output_dtype, device=terms[0].device)
        out[:, rows] = mixed
        return out
    _check_eps(eps)
    _check_eps(norm_eps)
    _check_sources(blocks, weights)
    _check_candidate(terms, weights, candidate_out, norm_weights, output_dtype)
    first = terms[0]
    if blocks.shape[1:] != first.shape or blocks.device != first.device:
        raise ValueError("sources and terms disagree on rows or device")
    if out is None:
        out = torch.empty(shape, dtype=output_dtype, device=first.device)
    return attnres_mix_queries(
        blocks,
        terms,
        weights,
        norm_weights,
        eps,
        norm_eps,
        out,
        candidate_out=candidate_out,
        normalize_values=normalize_values,
        tokens=(first_token, count),
    )


def residual_sum(inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Sum same-shaped sources in FP32, then store the first source's dtype.

    Args:
        inputs: Nonempty tuple of contiguous BF16 or FP32 tensors on one device.

    Returns:
        A fresh tensor with the first input's shape and dtype, rounded once.
    """
    if not inputs:
        raise ValueError("residual sum needs at least one input")
    first = inputs[0]
    if first.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("residual sum needs BF16 or FP32 inputs")
    if any(
        x.shape != first.shape
        or x.dtype != first.dtype
        or x.device != first.device
        or not x.is_contiguous()
        for x in inputs
    ):
        raise ValueError(
            "residual sum inputs must share shape, dtype, device and contiguous layout"
        )
    if first.is_cuda:
        kernel = select_kernel(
            "residual",
            "sum_fp32",
            format_signature(source=dense_tensor_format(first.dtype)),
        )
        return kernel(inputs)
    total = torch.zeros_like(first, dtype=torch.float32)
    for value in inputs:
        total = total + value.float()
    return total.to(first.dtype)


def residual_weighted_sum(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Mix ``S`` sources with fixed weights: start from FP32 zero and add
    ``float(values[:, s]) * float(weights[s])`` for ``s = 0 .. S-1``, each
    product and sum rounded separately.

    Args:
        values: Contiguous BF16 or FP32 ``[tokens, S, width]`` sources.
        weights: ``[S]`` weights, promoted to FP32.

    Returns:
        A fresh FP32 ``[tokens, width]`` mixture.
    """
    if (
        values.ndim != 3
        or not values.is_contiguous()
        or weights.shape != values.shape[1:2]
    ):
        raise ValueError(
            "weighted sum needs contiguous [T, S, H] values and [S] weights"
        )
    if values.is_cuda:
        kernel = select_kernel(
            "residual",
            "weighted_sum_fp32",
            format_signature(source=dense_tensor_format(values.dtype)),
        )
        return kernel(values, weights.contiguous())
    total = torch.zeros(
        (values.shape[0], values.shape[2]), dtype=torch.float32, device=values.device
    )
    for source in range(values.shape[1]):
        total = total + values[:, source].float() * weights[source].float()
    return total
