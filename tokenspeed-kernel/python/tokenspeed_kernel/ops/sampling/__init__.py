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

"""Sampling kernel entry points."""

from __future__ import annotations

# ``tokenspeed_kernel.ops.sampling.torch`` (the portable leaves) is imported
# below for registration and would rebind the bare name ``torch`` in this
# package namespace, hence the alias.
import torch as _torch
from tokenspeed_kernel.ops.sampling.logprob import blockwise_logprobs
from tokenspeed_kernel.ops.sampling.positions import prepare_sampling_offsets
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "argmax",
    "eagle4_candidate_supports",
    "eagle4_candidate_tail",
    "eagle4_shard_topk",
    "blockwise_logprobs",
    "prepare_sampling_offsets",
    "VOCAB_SHARD_DRAFT_BLOCK",
    "VOCAB_SHARD_DRAFT_FIELDS",
    "vocab_shard_draft_record",
    "vocab_shard_draft_stats",
    "vocab_shard_draft_stats_width",
    "vocab_shard_topk",
    "vocab_shard_topk_supports",
    "vocab_topk_merge",
    "vocab_topk_merge_supports",
]

_SUPPORTED_DTYPES = (_torch.float16, _torch.bfloat16, _torch.float32)
_SUPPORTED_OUT_DTYPES = (_torch.int32, _torch.int64)


def _validate_argmax_out(logits: _torch.Tensor, out: _torch.Tensor) -> None:
    if out.shape != (logits.shape[0],):
        raise ValueError(
            f"out must have shape (M,)={(logits.shape[0],)}, got {tuple(out.shape)}"
        )
    if out.dtype not in _SUPPORTED_OUT_DTYPES:
        raise ValueError(f"out must be int32 or int64; got {out.dtype}")
    if out.device != logits.device:
        raise ValueError("out must be on the same device as logits")


def _argmax_torch_fallback(
    logits: _torch.Tensor,
    *,
    out: _torch.Tensor | None = None,
) -> _torch.Tensor:
    if out is not None:
        _validate_argmax_out(logits, out)
    result = _torch.argmax(logits, dim=-1)
    if out is not None:
        out.copy_(result)
        return out
    return result


def argmax(
    logits: _torch.Tensor,
    *,
    out: _torch.Tensor | None = None,
    solution: str | None = None,
    override: str | None = None,
) -> _torch.Tensor:
    """Return row-wise argmax indices over the last logits dimension.

    Args:
        logits: Input logits with shape ``(M, N)``. The argmax is taken
            over the last dimension.
        out: Optional output buffer with shape ``(M,)`` and dtype int32 or
            int64 on the same device as ``logits``.
        solution: Optional kernel solution to force through normal selection.
        override: Optional exact kernel-name or solution override.

    NaN handling:
        Kernel-backed sampling paths treat NaNs as invalid candidates.
        Rows with at least one non-NaN value return the index of the
        maximum non-NaN value, with ties broken toward the lowest index.
        Rows with no valid non-NaN values return the ``-1`` sentinel.
        Unsupported inputs fall back to ``torch.argmax`` semantics.

    Returns:
        A tensor containing argmax indices for each row of ``logits``.
    """
    if (
        logits.dim() != 2
        or logits.shape[0] == 0
        or not logits.is_cuda
        or logits.dtype not in _SUPPORTED_DTYPES
    ):
        return _argmax_torch_fallback(logits, out=out)

    signature = format_signature(logits=dense_tensor_format(logits.dtype))
    try:
        kernel = select_kernel(
            "sampling",
            "argmax",
            signature,
            solution=solution,
            override=override,
        )
    except NoKernelFoundError:
        return _argmax_torch_fallback(logits, out=out)

    shape_params = {
        "M": logits.shape[0],
        "N": logits.shape[1],
        "has_out": out is not None,
    }
    ShapeCapture.get().record(
        "sampling", "argmax", kernel.name, logits.dtype, shape_params
    )
    with kernel_scope(
        "sampling", "argmax", logits.dtype, kernel_name=kernel.name, **shape_params
    ):
        return kernel(logits, out=out)


# The CUDA shard top-k's widest launch: 8-CTA clusters of 1024 threads x 32
# elements.
_EAGLE4_MAX_SHARD_VOCAB = 8 * 1024 * 32


def eagle4_candidate_supports(
    dtype: _torch.dtype, shard_vocab: int, num_candidates: int, ranks: int
) -> bool:
    """Whether the EAGLE-4 candidate head kernels cover a head.

    Args:
        dtype: The candidate scores' and logits' dtype; bf16 only.
        shard_vocab: Tokens per vocab shard, ``num_candidates`` to 262144.
        num_candidates: Candidates per shard and kept after the all-gather
            (NC): 32, 64, 128 or 256.
        ranks: Vocab shards (R), with ``ranks * num_candidates <= 1024``.

    Returns:
        True when both ``eagle4_shard_topk`` and ``eagle4_candidate_tail``
        take the head. The tail also takes a draft top-k of at most 32.
    """
    return (
        dtype == _torch.bfloat16
        and num_candidates <= shard_vocab <= _EAGLE4_MAX_SHARD_VOCAB
        and num_candidates in (32, 64, 128, 256)
        and ranks * num_candidates <= 1024
    )


def eagle4_shard_topk(
    scores: _torch.Tensor,
    k: int,
    vocab_start: int,
    global_ids: _torch.Tensor,
) -> None:
    """The EAGLE-4 candidate head's per-shard selection, before the all-gather.

    Writes ``torch.topk(scores, k, dim=-1, sorted=False).indices +
    vocab_start`` in CUDA's order on every platform: the ids of keys above the
    k-th largest by position, then the first k-th largest ties by position.

    Args:
        scores: ``[M, N]`` bf16 scores of a vocab shard, unit column stride.
        k: Ids kept per row.
        vocab_start: The vocab id of the shard's column 0.
        global_ids: ``[M, k]`` int64 destination, unit column stride; any row
            stride.
    """
    kernel = select_kernel(
        "sampling",
        "eagle4_shard_topk",
        format_signature(scores=dense_tensor_format(scores.dtype)),
    )
    kernel(scores, k, vocab_start, global_ids)


def eagle4_candidate_tail(
    cand_logits: _torch.Tensor,
    cand_tokens: _torch.Tensor,
    top_k: int,
    logits_out: _torch.Tensor,
    tokens_out: _torch.Tensor,
    topk_probs: _torch.Tensor | None,
    topk_tokens: _torch.Tensor | None,
) -> None:
    """Everything after the EAGLE-4 candidate all-gather, in one launch.

    The same, bit for bit, as::

        logits, pos = torch.topk(gathered_logits, NC, sorted=False)  # R > 1 only
        tokens = torch.gather(gathered_tokens, -1, pos)
        logits = logits.float()
        top = torch.topk(torch.softmax(logits, -1), top_k, -1)  # top_k > 0 only
        top_tokens = torch.gather(tokens, -1, top.indices)

    where ``gathered_*`` concatenates the ``R`` rank blocks. The NC-way top-k
    keeps CUDA's order, as ``eagle4_shard_topk``.

    Args:
        cand_logits: ``[R, M, NC]`` bf16 view of the gathered rank blocks, any
            rank and row strides, unit column stride.
        cand_tokens: ``[R, M, NC]`` int64 ids alike.
        top_k: Draft top-k, 0 to skip it.
        logits_out: Contiguous ``[M, NC]`` fp32 candidate logits.
        tokens_out: Contiguous ``[M, NC]`` int64 candidate ids; not written
            when R == 1.
        topk_probs: Contiguous ``[M, top_k]`` fp32, or None when top_k is 0.
        topk_tokens: Contiguous ``[M, top_k]`` int64, or None when top_k is 0.
    """
    kernel = select_kernel(
        "sampling",
        "eagle4_candidate_tail",
        format_signature(cand_logits=dense_tensor_format(cand_logits.dtype)),
    )
    kernel(
        cand_logits, cand_tokens, top_k, logits_out, tokens_out, topk_probs, topk_tokens
    )


# A shard column rides in the low 15 bits of a 32-bit selection key next to
# its 16-bit value key, so a shard holds at most this many columns.
_VOCAB_TOPK_MAX_SHARD_COLUMNS = 1 << 15
# The merge holds every candidate of a row in one program, its rank-major
# slot in 16 bits next to a 32-bit value key.
_VOCAB_TOPK_MAX_MERGE_CANDIDATES = 1 << 12


def vocab_shard_topk_supports(dtype: _torch.dtype, shard_columns: int, k: int) -> bool:
    """Whether ``vocab_shard_topk`` takes a shard of this geometry.

    Args:
        dtype: The shard logits' dtype.
        shard_columns: Columns in the shard.
        k: Candidates kept per row.
    """
    return (
        dtype in (_torch.bfloat16, _torch.float16)
        and 1 <= k <= shard_columns <= _VOCAB_TOPK_MAX_SHARD_COLUMNS
    )


def vocab_topk_merge_supports(ranks: int, k: int) -> bool:
    """Whether ``vocab_topk_merge`` takes ``ranks`` blocks of ``k`` pairs."""
    return k >= 1 and ranks * k <= _VOCAB_TOPK_MAX_MERGE_CANDIDATES


def vocab_shard_topk(
    logits: _torch.Tensor,
    k: int,
    logit_scale: float | None,
    vocab_start: int,
    out: _torch.Tensor,
    *,
    softcap: float | None,
    vocab_mask: _torch.Tensor | None,
) -> None:
    """One vocab shard's ``k`` best logits and their global ids, packed for
    the all-gather ``vocab_topk_merge`` reads.

    Args:
        logits: ``[M, N]`` 16-bit logits of the shard, unit column stride;
            ``vocab_shard_topk_supports`` must hold.
        k: Candidates kept per row.
        logit_scale: Multiplier applied to every logit (rounded to the logits'
            dtype) before the selection, or None.
        vocab_start: Global vocab id of shard column 0.
        out: ``[M, 2 * k]`` fp32: the values, then the global ids as exact
            fp32 integers, in descending value order (lowest column first
            among equal values).
        softcap: Final logit softcapping ``c`` in ``c * tanh(x / c)``, applied
            after the scale and rounded to the logits' dtype before the
            selection (as the logits processor writes softcapped logits), or
            None. ``vocab_topk_merge`` can instead cap the kept values.
        vocab_mask: Contiguous int32 ``[M, ceil(V / 32)]`` whole-vocabulary
            token bitmask (bit ``t % 32`` of word ``t // 32`` set when token
            ``t`` is allowed, the xgrammar layout); disallowed logits become
            ``-inf`` before the selection. None for no mask.
    """
    kernel = select_kernel(
        "sampling",
        "vocab_shard_topk",
        format_signature(logits=dense_tensor_format(logits.dtype)),
    )
    kernel(
        logits, k, logit_scale, vocab_start, out, softcap=softcap, vocab_mask=vocab_mask
    )


def vocab_topk_merge(
    pairs: _torch.Tensor,
    k: int,
    softcap: float | None,
    ids_out: _torch.Tensor,
    values_out: _torch.Tensor,
) -> None:
    """Every rank's ``vocab_shard_topk`` pairs to each row's sorted ``k`` best.

    Args:
        pairs: ``[R, M, 2 * k]`` fp32 view of the gathered rank blocks, any
            rank and row strides, unit column stride.
        k: Candidates kept per row.
        softcap: Final logit softcapping ``c`` in ``c * tanh(x / c)`` applied
            to the kept values, or None.
        ids_out: Contiguous ``[M, k]`` int64 or int32 global ids, descending
            value order, lowest rank-major slot first among equal values.
        values_out: Contiguous ``[M, k]`` fp32 values, in the same order.
    """
    kernel = select_kernel(
        "sampling",
        "vocab_topk_merge",
        format_signature(pairs=dense_tensor_format(pairs.dtype)),
    )
    kernel(pairs, k, softcap, ids_out, values_out)


# Columns of a shard each draft-stats block reduces, and the fp32 partials it writes.
VOCAB_SHARD_DRAFT_BLOCK = 2048
VOCAB_SHARD_DRAFT_FIELDS = 8


def vocab_shard_draft_stats_width(shard_columns: int) -> int:
    """The fp32 columns per row of ``vocab_shard_draft_stats``' output for a shard of ``shard_columns``.

    Args:
        shard_columns: Columns in the shard.

    Returns:
        One block of ``VOCAB_SHARD_DRAFT_FIELDS`` partials per
        ``VOCAB_SHARD_DRAFT_BLOCK`` columns.
    """
    blocks = -(-shard_columns // VOCAB_SHARD_DRAFT_BLOCK)
    return blocks * VOCAB_SHARD_DRAFT_FIELDS


def vocab_shard_draft_stats(
    logits: _torch.Tensor,
    vocab_start: int,
    slots: _torch.Tensor,
    temperature_pool: _torch.Tensor,
    seed_pool: _torch.Tensor,
    offsets_pool: _torch.Tensor,
    out: _torch.Tensor,
    *,
    logit_scale: float | None,
    softcap: float | None,
) -> None:
    """Reduce one rank's shard of draft logits to the partials every rank gathers.

    The first half of sampling a draft token from ``q = softmax(logits / T)``
    over a vocabulary-parallel head without gathering the logits: each block
    of ``VOCAB_SHARD_DRAFT_BLOCK`` columns yields the maximum and exp-sum of
    ``logits / T``, the best Gumbel-max score (noise keyed by slot seed,
    slot offset and global token id, as ``gumbel_sample_from_pools``) and the
    largest logit with their lowest global ids, and a count of non-finite
    values. ``vocab_shard_draft_record`` combines the ranks' partials.

    Args:
        logits: ``[M, N]`` raw head logits of one vocabulary shard (before
            the processor's scale and softcap), unit column stride.
        vocab_start: Global token id of shard column 0.
        slots: ``[>= M]`` int32 or int64 pool slot of each row.
        temperature_pool: ``[P]`` temperature by slot (floored at 1e-20).
        seed_pool: ``[P]`` int64 Philox seed by slot.
        offsets_pool: ``[P]`` int64 Philox offset by slot.
        out: ``[>= M, vocab_shard_draft_stats_width(N)]`` fp32 partials.
        logit_scale: The logits processor's multiplier, rounded to the
            logits' dtype, or None.
        softcap: The processor's final logit softcap ``c * tanh(x / c)``,
            after the scale and rounded to the logits' dtype, or None.
    """
    kernel = select_kernel(
        "sampling",
        "vocab_shard_draft_stats",
        format_signature(logits=dense_tensor_format(logits.dtype)),
    )
    kernel(
        logits,
        vocab_start,
        slots,
        temperature_pool,
        seed_pool,
        offsets_pool,
        out,
        logit_scale=logit_scale,
        softcap=softcap,
    )


def vocab_shard_draft_record(
    logits: _torch.Tensor,
    stats: _torch.Tensor,
    vocab_start: int,
    slots: _torch.Tensor,
    write_slots: _torch.Tensor,
    temperature_pool: _torch.Tensor,
    top_k_pool: _torch.Tensor,
    q_out: _torch.Tensor,
    tokens_out: _torch.Tensor,
    *,
    logit_scale: float | None,
    softcap: float | None,
    sentinel: float,
) -> None:
    """Each row's draft token from every rank's partials, and its ``q`` over this rank's shard.

    The second half after ``vocab_shard_draft_stats`` and the gather. Every
    rank draws the same token: the one ``gumbel_sample_from_pools`` draws
    from the whole row.

    Args:
        logits: ``[M, N]`` raw head logits of this rank's shard, as passed to
            ``vocab_shard_draft_stats``.
        stats: ``[R, M, >= vocab_shard_draft_stats_width(N)]`` fp32 view of
            every rank's partials, rank-major, unit column stride; ``R * N``
            is the vocabulary.
        vocab_start: Global token id of shard column 0.
        slots: ``[>= M]`` int32 or int64 pool slot of each row.
        write_slots: ``[>= M]`` int32 or int64 row of ``q_out`` each row writes.
        temperature_pool: ``[P]`` temperature by slot.
        top_k_pool: ``[P]`` int32 top-k by slot; 1 marks a greedy request.
        q_out: ``[P', N]`` fp32 with unit column stride. Each row writes
            ``softmax(logits / T)`` over this shard's columns; a greedy row
            one-hot at its lowest-id maximum logit; a row whose softmax is
            not finite (a NaN or an overflow) ``sentinel`` everywhere.
        tokens_out: ``[>= M]`` int32 global token ids: the Gumbel-max draw
            of ``logits / T``, lowest id among equal scores, or the lowest-id
            maximum for greedy rows; clamped into the vocabulary for rows
            without a finite softmax.
        logit_scale: As for ``vocab_shard_draft_stats``.
        softcap: As for ``vocab_shard_draft_stats``.
        sentinel: The recorded probability that marks no proposal.
    """
    kernel = select_kernel(
        "sampling",
        "vocab_shard_draft_record",
        format_signature(logits=dense_tensor_format(logits.dtype)),
    )
    kernel(
        logits,
        stats,
        vocab_start,
        slots,
        write_slots,
        temperature_pool,
        top_k_pool,
        q_out,
        tokens_out,
        logit_scale=logit_scale,
        softcap=softcap,
        sentinel=sentinel,
    )


# Backend registration (side-effect imports).
import tokenspeed_kernel.ops.sampling.cuda  # noqa: E402,F401
import tokenspeed_kernel.ops.sampling.cute_dsl  # noqa: E402,F401
import tokenspeed_kernel.ops.sampling.gluon  # noqa: E402,F401
import tokenspeed_kernel.ops.sampling.torch  # noqa: E402,F401
import tokenspeed_kernel.ops.sampling.triton.vocab_shard_draft  # noqa: E402,F401
import tokenspeed_kernel.ops.sampling.triton.vocab_topk  # noqa: E402,F401
