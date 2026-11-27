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

"""Portable torch leaves for the sampling family."""

from __future__ import annotations

import torch
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@register_kernel(
    "sampling",
    "block_sumexp",
    name="torch_block_sumexp",
    solution="torch",
    features={"batch_invariant"},
    signatures=format_signatures("shifted", "dense", {torch.float32}),
    priority=Priority.PORTABLE,
)
def torch_block_sumexp(shifted: torch.Tensor, *, block_size: int) -> torch.Tensor:
    """Per-block ``sum(exp(x))`` with one fixed binary tree per block.

    Every block is reduced by halving: element ``i`` is added to element
    ``i + half`` at each level, so the association order depends only on
    ``block_size`` and every row of every batch shape follows the same tree
    (the ``batch_invariant`` feature). Vendor leaves (e.g. the fixed-order
    operator kit's ``sumexp``) register the same op with their own in-block
    order; this leaf is the portable one.

    Args:
        shifted: ``[rows, vocab]`` fp32 logits already shifted by the row max;
            ``vocab`` is a multiple of ``block_size``.
        block_size: Width of each block; a power of two.

    Returns:
        ``[rows, vocab // block_size]`` fp32 block sums.
    """
    rows, vocab = shifted.shape
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError(f"block_size must be a power of two, got {block_size}")
    if vocab % block_size:
        raise ValueError(f"vocab {vocab} is not a multiple of block_size {block_size}")
    # ``exp`` allocates the one buffer the fold runs in: ``reshape`` takes a
    # non-contiguous ``shifted`` (``view`` would not), and each level adds the
    # upper half onto the lower half in place, so no level allocates.
    values = torch.exp(shifted).reshape(rows, vocab // block_size, block_size)
    width = block_size
    while width > 1:
        half = width // 2
        values[..., :half] += values[..., half:width]
        width = half
    return values[..., 0]


@register_kernel(
    "sampling",
    "full_vocab_sum",
    name="torch_full_vocab_sum",
    solution="torch",
    signatures=format_signatures("values", "dense", {torch.float32}),
    priority=Priority.PORTABLE,
)
def torch_full_vocab_sum(values: torch.Tensor) -> torch.Tensor:
    """Fold block partials left to right, starting at block zero.

    Args:
        values: FP32 [rows, blocks], at least one block.

    Returns:
        FP32 [rows] in exactly ((p0 + p1) + p2) order.
    """
    total = values[:, 0]
    for block in range(1, values.shape[1]):
        total = total + values[:, block]
    return total


@register_kernel(
    "sampling",
    "full_vocab_head",
    name="torch_full_vocab_head",
    solution="torch",
    signatures=format_signatures("values", "dense", {torch.float32}),
    priority=Priority.PORTABLE,
)
def torch_full_vocab_head(shifted, ids, log_z, *, form):
    """Gather [rows, K] shifted logits and apply the explicit subtraction form."""
    selected = shifted.gather(1, ids)
    if form == "direct":
        return selected - log_z[:, None]
    if form == "negative-cross-entropy":
        return -(log_z[:, None] - selected)
    raise ValueError(f"unknown full-vocabulary logprob form: {form!r}")


@register_kernel(
    "sampling",
    "full_vocab_pack",
    name="torch_full_vocab_pack",
    solution="torch",
    signatures=format_signatures("values", "dense", {torch.float32}),
    priority=Priority.PORTABLE,
)
def torch_full_vocab_pack(
    shifted,
    log_z,
    ids,
    head,
    tail,
    tokens,
    *,
    prediction_rows,
    form,
):
    """Pack binary records and sampled-score bits in output-token order.

    Args:
        shifted: FP32 [rows, vocab] original logits minus each row's maximum.
        log_z: FP32 [rows] log denominators in shifted-logit space.
        ids: Int64 [rows, K] head vocabulary IDs, unit column stride.
        head: FP32 [rows, K] head scores, or None to gather and score while packing.
        tail: FP32 [rows] tail log masses.
        tokens: Int32 or int64 [rows] output IDs; padded IDs are retained.
        prediction_rows: Int64 [rows] prediction-row mapping, or None.
        form: "direct" or "negative-cross-entropy", preserving zero sign.

    Returns:
        Int32 [rows, 3 + 2*K]: token, tail bits, IDs, head bits, sampled bits.
    """
    if head is None:
        head = torch_full_vocab_head(shifted, ids, log_z, form=form)
    target = tokens.long().clamp(0, shifted.shape[1] - 1)
    if prediction_rows is None:
        selected = shifted.gather(1, target[:, None]).squeeze(1)
    else:
        selected = shifted[prediction_rows, target]
        log_z = log_z.index_select(0, prediction_rows)
        ids, head, tail = (
            value.index_select(0, prediction_rows) for value in (ids, head, tail)
        )
    sampled = selected - log_z if form == "direct" else -(log_z - selected)
    return torch.cat(
        (
            tokens.to(torch.int32)[:, None],
            tail.contiguous().view(torch.int32)[:, None],
            ids.to(torch.int32),
            head.contiguous().view(torch.int32),
            sampled.contiguous().view(torch.int32)[:, None],
        ),
        dim=1,
    )


@register_kernel(
    "sampling",
    "full_vocab_topk",
    name="torch_full_vocab_topk",
    solution="torch",
    signatures=format_signatures("values", "dense", {torch.float32}),
    priority=Priority.PORTABLE,
)
def torch_full_vocab_topk(logits, *, k, active_rows):
    """Reference sorted values/IDs; inactive rows are not consumed."""
    return torch.topk(logits, k, dim=-1)


@register_kernel(
    "sampling",
    "full_vocab_maximum",
    name="torch_full_vocab_maximum",
    solution="torch",
    signatures=format_signatures("values", "dense", {torch.float32}),
    priority=Priority.PORTABLE,
)
def torch_full_vocab_maximum(logits, topk_values):
    """Keep torch.max's original first-index behavior, including signed zero."""
    return logits.max(dim=-1).values


@register_kernel(
    "sampling",
    "full_vocab_tail",
    name="torch_full_vocab_tail",
    solution="torch",
    signatures=format_signatures("values", "dense", {torch.float32}),
    priority=Priority.PORTABLE,
)
def torch_full_vocab_tail(logits, ids, log_z, *, row_maximum, active_rows):
    """Masked raw-logit logsumexp minus (row_maximum + log_z), restoring input."""
    selected = logits.gather(1, ids)
    logits.scatter_(1, ids, -torch.inf)
    tail = torch.logsumexp(logits, dim=-1) - (row_maximum + log_z)
    logits.scatter_(1, ids, selected)
    return tail


@register_kernel(
    "sampling",
    "full_vocab_active_rows",
    name="torch_full_vocab_active_rows",
    solution="torch",
    signatures=format_signatures("values", "dense", {torch.float32}),
    priority=Priority.PORTABLE,
)
def torch_full_vocab_active_rows(
    accept_lengths,
    req_pool_indices,
    *,
    num_prefill_outputs,
    num_extends,
    decode_width,
    padding_index,
    prediction_rows,
):
    """Fixed-size source-row mask for mixed prefills, chain and tree decode."""
    rows = num_prefill_outputs + (accept_lengths.numel() - num_extends) * decode_width
    offsets = torch.arange(rows, device=accept_lengths.device)
    decode = offsets - num_prefill_outputs
    requests = torch.where(
        offsets < num_prefill_outputs, offsets, num_extends + decode // decode_width
    )
    positions = torch.where(offsets < num_prefill_outputs, 0, decode % decode_width)
    valid = (positions < accept_lengths[requests]) & (
        req_pool_indices[requests] != padding_index
    )
    sources = offsets if prediction_rows is None else prediction_rows
    # Invalid tree slots can alias accepted source rows. Only true flags may
    # contribute; scatter-add avoids a false duplicate clearing a valid row.
    counts = torch.zeros(rows, device=accept_lengths.device, dtype=torch.int32)
    counts.scatter_add_(0, sources, valid.to(torch.int32))
    return counts != 0


@register_kernel(
    "sampling",
    "full_vocab_source_shift",
    name="torch_full_vocab_source_shift",
    solution="torch",
    signatures=format_signatures("values", "dense", {torch.float32}),
    priority=Priority.PORTABLE,
)
def torch_full_vocab_source_shift(logits, maximum):
    """Own original FP32 logits and subtract the same per-row maximum."""
    source = logits.clone()
    return source, source - maximum[:, None]


@register_kernel(
    "sampling",
    "full_vocab_row_selection_supports",
    name="torch_full_vocab_row_selection_supports",
    solution="torch",
    signatures=format_signatures("values", "dense", {torch.float32}),
    priority=Priority.PORTABLE,
)
def torch_full_vocab_row_selection_supports(logits):
    """The reference computes complete rows before sampling."""
    return False
