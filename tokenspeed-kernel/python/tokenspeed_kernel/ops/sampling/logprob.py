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

"""Selected-token log-probabilities in the trainer's vocab-parallel order.

Megatron's vocab-parallel cross-entropy computes, per row: the max over the
local vocab shard and a MAX all-reduce; ``shard -= max``; the target logit
gathered from the owning shard (zero elsewhere) and a SUM all-reduce; the
denominator as ``sum(exp)`` over fixed 32768-wide vocab blocks whose partials
are folded in fp32 in block (rank) order; ``nll = log(sum_exp) - target``.
``vocab_parallel_logprobs`` reproduces that order on the full vocabulary one
rank already holds, which is bit-identical to the collective form: the max
is exact, adding the other shards' exact zeros cannot move the target, and
the all-gathered block partials fold in the same order. The one solution-
specific piece is the in-block ``sum(exp)``, the registered
``sampling.block_sumexp`` leaf.

The leaf contract (``register_kernel("sampling", "block_sumexp", ...)``):

* signature ``format_signatures("shifted", "dense", {torch.float32})`` and
  the ``batch_invariant`` feature, which the op REQUIRES — a leaf's tree must
  depend on the block width only, never on the row count;
* call ``leaf(shifted, *, block_size)`` with ``shifted`` fp32 ``[rows,
  vocab]`` already shifted by the row max (``vocab`` a multiple of
  ``block_size``), returning fp32 ``[rows, vocab // block_size]`` where
  column ``j`` is ``sum(exp(shifted[:, j * block_size:(j + 1) *
  block_size]))`` reduced in the leaf's own fixed order;
* the op folds the columns left to right in fp32 (``((p0 + p1) + p2) ...``),
  Megatron's rank-ordered reduction of the per-shard partials: a shard
  narrower than a block pairs back into the block's own tree at its top
  level, so one full-vocabulary call and the trainer's sharded sum agree
  bitwise whenever the leaf's in-block tree is pairwise over its tiles.

The portable ``torch`` leaf registers at ``Priority.PORTABLE``; a vendor leaf
with the trainer's in-block order (the fixed-order operator kit's
``sumexp`` over 32768-wide blocks) registers the same op above it, so
normal selection prefers it wherever it is installed, and ``solution=`` /
``override=`` pin one explicitly.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = ["vocab_parallel_logprobs"]

_BATCH_INVARIANT = frozenset({"batch_invariant"})


def vocab_parallel_logprobs(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    vocab_block: int,
    solution: str | None = None,
    override: str | None = None,
) -> torch.Tensor:
    """Return each row's log-probability of ``target_ids`` in Megatron's order.

    Args:
        logits: ``[rows, vocab]`` logits of the whole vocabulary (any float
            dtype; computed in fp32). ``vocab`` must be a multiple of
            ``vocab_block``.
        target_ids: ``[rows]`` integer token ids.
        vocab_block: Width of the fixed ``sum(exp)`` blocks (the trainer uses
            32768); the fold across blocks is a fp32 left fold in block order.
        solution: Optional kernel solution for the block ``sum(exp)`` leaf;
            only leaves declaring the ``batch_invariant`` feature are eligible.
            None takes the highest-priority eligible leaf (a vendor leaf
            registered above the portable one wins).
        override: Optional exact kernel-name or solution override.

    Returns:
        ``[rows]`` fp32 log-probabilities, ``-(log(sum_exp) - target)`` with
        ``target = logits[row, id] - max`` and ``sum_exp`` the fp32 left fold
        of the leaf's ``[rows, vocab // vocab_block]`` block partials.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be [rows, vocab], got {tuple(logits.shape)}")
    rows, vocab = logits.shape
    if target_ids.shape != (rows,):
        raise ValueError(
            f"target_ids must have shape {(rows,)}, got {tuple(target_ids.shape)}"
        )
    if target_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"target_ids must be int32 or int64, got {target_ids.dtype}")
    if vocab_block <= 0 or vocab % vocab_block:
        raise ValueError(
            f"vocab {vocab} must be a positive multiple of vocab_block {vocab_block}"
        )
    kernel = select_kernel(
        "sampling",
        "block_sumexp",
        format_signature(shifted=dense_tensor_format(torch.float32)),
        features=_BATCH_INVARIANT,
        solution=solution,
        override=override,
    )
    logits = logits.float()
    row_max = logits.max(dim=-1).values
    shifted = logits - row_max.unsqueeze(-1)
    target = shifted.gather(1, target_ids.to(torch.int64).unsqueeze(-1)).squeeze(-1)
    shape_params = {"rows": rows, "vocab": vocab, "vocab_block": vocab_block}
    ShapeCapture.get().record(
        "sampling", "block_sumexp", kernel.name, torch.float32, shape_params
    )
    with kernel_scope(
        "sampling",
        "block_sumexp",
        torch.float32,
        kernel_name=kernel.name,
        **shape_params,
    ):
        partials = kernel(shifted, block_size=vocab_block)
    # Megatron's cross-shard reduction: a fp32 left fold in block order.
    sum_exp = partials[:, 0]
    for block in range(1, partials.shape[1]):
        sum_exp = sum_exp + partials[:, block]
    # Megatron returns the NLL; the log-probability is its exact negation.
    return -(torch.log(sum_exp) - target)
