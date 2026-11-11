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

"""sonic-sampler's kernels, launched for the runtime's ``sonic`` sampling backend.

The kernels are vendored in ``tokenspeed_kernel.thirdparty.sonic_sampler.ops`` (see the
package README for the upstream commit and the local changes); this module
launches them over the backend's slot-indexed state and picks their config
from a table keyed by vocabulary size class and logits rows, so no server
tunes at startup. On GB200 (13 vocabularies from 32k to 262k, 1 to 512 rows,
plain decode and three-draft verify, 60 configs swept) its picks run within 8%
of the best config at every point and 0.3% on average, and 3.6x to 11x faster
than the FlashInfer backend. The other Blackwell GPUs share it; any other GPU
uses it unmeasured.

Every config draws the same tokens (the selection is exact; only its tiling
and warps differ), so the table changes speed only. A row with no top-k, top-p
or min-p draws from its whole vocabulary: each block keeps its best logit plus
the slot's Gumbel noise at each token, and the best over the blocks (or ranks)
is the draw. The other sampled rows select among the top-128 candidates their
bf16 keys pick. bf16 logits rank at bf16 resolution; fp32 logits (soft-capped
heads) are sampled exactly in fp32.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from dataclasses import dataclass

import torch
from tokenspeed_kernel._triton import libdevice, tl, triton
from tokenspeed_kernel.thirdparty.sonic_sampler.core.flags import Indicator
from tokenspeed_kernel.thirdparty.sonic_sampler.ops.base import (
    MAX_K,
    gdc_launch_dependents,
    gdc_wait,
    next_power_of_2,
    philox_gumbel,
    philox_key,
)
from tokenspeed_kernel.thirdparty.sonic_sampler.ops.multistep import (
    cumulative_unpack_kernel,
)
from tokenspeed_kernel.thirdparty.sonic_sampler.ops.prologue import (
    bitpacked_reduction_kernel,
    whole_reduction_kernel,
    whole_row,
)
from tokenspeed_kernel.thirdparty.sonic_sampler.ops.shard import (
    shard_candidates_kernel,
)
from tokenspeed_kernel.thirdparty.sonic_sampler.ops.singular import (
    cumulative_selection_kernel,
)
from tokenspeed_kernel.thirdparty.sonic_sampler.ops.verify import (
    chain_speculative_verification_kernel,
)
from torch import Tensor

# The kernels' block for top-k logprob targets, which the runtime never requests.
_LOGPROB_BLOCK = 8

# Adaptive top-k's scale factor, unread with adaptive top-k off.
_TOP_K_SCALE = 4

# The kernels keep row * stride offsets into the logits and the scratchpad in int32.
MAX_OFFSET = 2**31


@dataclass(frozen=True)
class SonicConfig:
    """One way to run sonic's pipeline: the vocab tile, the top-k strategy and the warps of its first two stages."""

    block_n: int
    radix: bool
    first_warps: int
    second_warps: int


_BITONIC_2K = SonicConfig(2048, False, 8, 8)
_BITONIC_4K = SonicConfig(4096, False, 8, 8)
_RADIX_2K = SonicConfig(2048, True, 4, 8)
_RADIX_4K = SonicConfig(4096, True, 8, 8)
_RADIX_8K = SonicConfig(8192, True, 16, 8)
_RADIX_16K = SonicConfig(16384, True, 16, 8)

# Per vocabulary class (largest size, None for the rest): (most logits rows, config), the last bound None.
# Few rows want fewer, wider tiles until the first stage's tiles about fill the SMs; many rows want 2k-4k radix tiles.
TABLE: tuple[tuple[int | None, tuple[tuple[int | None, SonicConfig], ...]], ...] = (
    (80_000, ((4, _BITONIC_2K), (64, _RADIX_4K), (None, _RADIX_2K))),
    (
        140_000,
        (
            (1, _BITONIC_2K),
            (4, _BITONIC_4K),
            (8, _RADIX_8K),
            (16, _RADIX_16K),
            (64, _RADIX_4K),
            (None, _RADIX_2K),
        ),
    ),
    (160_000, ((4, _BITONIC_4K), (16, _RADIX_8K), (None, _RADIX_4K))),
    (None, ((2, _BITONIC_4K), (4, _RADIX_8K), (8, _RADIX_16K), (None, _RADIX_4K))),
)


def _fit(config: SonicConfig, vocab_size: int) -> SonicConfig:
    """``config`` with its tile halved until the vocabulary spans two tiles, as sonic's reduction needs."""
    block_n = config.block_n
    while 2 * block_n > vocab_size and block_n > 1024:
        block_n //= 2
    if 2 * block_n <= vocab_size:
        return SonicConfig(
            block_n, config.radix, config.first_warps, config.second_warps
        )
    # Below 2048 columns: two or three bitonic tiles.
    return SonicConfig(1 << (vocab_size.bit_length() - 2), False, 4, 4)


@functools.cache
def pick(vocab_size: int, rows: int) -> SonicConfig:
    """The config for one launch over ``[rows, vocab_size]`` logits.

    Args:
        vocab_size: the logits width (at least 512, sonic's reduction minimum).
        rows: the launch's logits rows (requests, times drafts plus one for verify).

    Returns:
        The table's config for the vocabulary's class and the rows, with its
        tile narrowed when the vocabulary does not span two tiles.
    """
    rows_table = next(t for size, t in TABLE if size is None or vocab_size <= size)
    config = next(c for bound, c in rows_table if bound is None or rows <= bound)
    return _fit(config, vocab_size)


def configs(vocab_size: int) -> tuple[SonicConfig, ...]:
    """Every config ``pick`` returns for ``vocab_size``, for warm-up and the scratchpad."""
    rows_table = next(t for size, t in TABLE if size is None or vocab_size <= size)
    return tuple(dict.fromkeys(_fit(c, vocab_size) for _, c in rows_table))


def scratchpad_columns(vocab_size: int) -> int:
    """Scratchpad columns one logits row needs: the narrowest picked tile's blocks, ``MAX_K`` keys each."""
    return triton.cdiv(vocab_size, min(c.block_n for c in configs(vocab_size))) * MAX_K


@dataclass(frozen=True)
class SlotState:
    """The per-slot tensors sonic's kernels gather at each row's slot.

    Scalars are bf16, or fp32 for fp32 logits (``with_scalars``); every
    tensor's first dimension is the slot.
    """

    indicators: Tensor  # [P] int16 ``Indicator`` bits (at most 0x800).
    temperature: Tensor  # [P].
    top_k: Tensor  # [P] int32, at most ``MAX_K``.
    top_p: Tensor  # [P].
    min_p: Tensor  # [P].
    repetition: Tensor  # [P] multiplicative repetition penalty.
    frequency: Tensor  # [P].
    presence: Tensor  # [P].
    grammar: Tensor  # [P, timesteps, ceil(V / 32)] int32 bitmask (xgrammar's), set bits allowed.
    context_counts: Tensor  # [P, V] int32 prompt counts.
    decode_counts: Tensor  # [P, V] int32 counts of drawn tokens.
    logit_bias: Tensor  # [P, V] bf16.
    seeds: Tensor  # [P] int64 Philox seeds.

    @classmethod
    def allocate(
        cls,
        slots: int,
        timesteps: int,
        vocab_size: int,
        seed: int,
        device: torch.device | str,
    ) -> SlotState:
        """Every slot in its never-admitted state: no manipulators, bounded top-k, all tokens allowed.

        Args:
            slots: Slots, the padding sentinel included.
            timesteps: Logits rows per request (drafts plus one).
            vocab_size: The logits width.
            seed: Philox seed of never-admitted slots.
            device: Where the tensors live.

        Returns:
            The state, bf16 scalars.
        """
        bf16 = torch.bfloat16
        with torch.device(device):
            context_counts, decode_counts = torch.zeros(
                (2, slots, vocab_size), dtype=torch.int32
            )
            return cls(
                indicators=torch.zeros(slots, dtype=torch.int16),
                temperature=torch.ones(slots, dtype=bf16),
                top_k=torch.full((slots,), MAX_K, dtype=torch.int32),
                top_p=torch.ones(slots, dtype=bf16),
                min_p=torch.zeros(slots, dtype=bf16),
                repetition=torch.ones(slots, dtype=bf16),
                frequency=torch.zeros(slots, dtype=bf16),
                presence=torch.zeros(slots, dtype=bf16),
                grammar=torch.full(
                    (slots, timesteps, triton.cdiv(vocab_size, 32)),
                    -1,
                    dtype=torch.int32,
                ),
                context_counts=context_counts,
                decode_counts=decode_counts,
                logit_bias=torch.zeros((slots, vocab_size), dtype=bf16),
                seeds=torch.full((slots,), seed, dtype=torch.int64),
            )

    def with_scalars(self, scalars: Tensor) -> SlotState:
        """This state reading ``scalars``' rows (temperature, top-p, min-p, repetition, frequency, presence) instead."""
        temperature, top_p, min_p, repetition, frequency, presence = scalars
        return dataclasses.replace(
            self,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            repetition=repetition,
            frequency=frequency,
            presence=presence,
        )


def _check_offsets(logits: Tensor, scratchpad: Tensor) -> None:
    """Refuse a launch whose logits or scratchpad rows reach past the kernels' int32 row offsets."""
    for name, tensor in (("logits", logits), ("scratchpad", scratchpad)):
        if tensor.shape[0] * tensor.stride(0) > MAX_OFFSET:
            raise ValueError(
                f"sonic's {name} span {tensor.shape[0]} rows of stride "
                f"{tensor.stride(0)}, past its int32 offsets ({MAX_OFFSET} elements)"
            )


@dataclass(frozen=True)
class VocabShard:
    """This rank's slice of vocabulary-parallel logits, and how its candidates meet the other ranks'.

    A launch given one reads ``logits`` as this rank's raw head logits ``[rows,
    V_s]`` (before the logit scale and softcap) for global tokens
    ``vocab_start`` onward; each rank selects its best ``MAX_K`` candidates
    (``shard_candidates``), ``gather`` collects every rank's, and the draw
    merges them and samples them as whole-vocabulary logits would be.
    """

    vocab_start: int  # Global token id of shard column 0.
    logit_scale: float | None  # The logits processor's multiplier, or None.
    softcap: float | None  # The processor's final logit softcap, or None.
    # This rank's contiguous [rows, w] pairs to every rank's [W, rows, w], W a power of two.
    gather: Callable[[Tensor], Tensor]
    # Contiguous fp32 scratch of >= rows * 3 * MAX_K: this rank's candidates, w = 2 * MAX_K wide (3 * MAX_K with shard-local draft probabilities).
    pairs: Tensor


def _exact(logits: Tensor) -> bool:
    """Whether the logits are fp32, which the kernels sample exactly; bf16 is the other supported dtype."""
    if logits.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"sonic samples bf16 or fp32 logits, got {logits.dtype}")
    return logits.dtype == torch.float32


def _reduce(
    logits: Tensor,
    state: SlotState,
    slots: Tensor,
    offsets: Tensor,
    drafted: Tensor | None,
    scratchpad: Tensor,
    config: SonicConfig,
    enable_pdl: bool,
    shard: VocabShard | None,
) -> None:
    """The first stage: each block's keys (greedy maximum, or best perturbed value of a whole-vocabulary row) in ``scratchpad``; ``drafted`` is None for one row per request."""
    exact = _exact(logits)
    rows, vocab_size = logits.shape
    blocks = triton.cdiv(vocab_size, config.block_n)
    singular = drafted is None
    if singular:
        batch_size, gamma, block_g = rows, 0, None
        stride_g = state.grammar.stride(0)
    else:
        batch_size, timesteps = drafted.shape
        gamma, block_g = timesteps - 1, next_power_of_2(timesteps)
        stride_g = state.grammar.flatten(0, -2).stride(0)
    bitpacked_reduction_kernel[(rows, blocks)](
        logits,
        state.indicators,
        slots,
        None,
        None,
        state.grammar,
        state.context_counts,
        state.decode_counts,
        drafted,
        state.repetition,
        state.frequency,
        state.presence,
        state.logit_bias,
        state.temperature,
        scratchpad,
        vocab_size,
        0 if shard is None else shard.vocab_start,
        gamma,
        MAX_K,
        _TOP_K_SCALE,
        batch_size,
        False,
        not singular,
        True,
        singular,
        False,
        False,
        config.radix,
        enable_pdl,
        logits.stride(0),
        stride_g,
        state.decode_counts.stride(0),
        None,
        scratchpad.stride(0),
        config.block_n,
        None,
        block_g,
        exact,
        tl.float32 if exact else tl.bfloat16,
        logit_scale=None if shard is None else shard.logit_scale,
        softcap=None if shard is None else shard.softcap,
        whole=True,
        num_warps=config.first_warps,
    )
    # The whole-vocabulary rows, a few programs per row walking consecutive blocks.
    group_blocks = triton.cdiv(
        blocks, min(blocks, triton.cdiv(_whole_programs(logits.device), rows))
    )
    whole_reduction_kernel[(rows, triton.cdiv(blocks, group_blocks))](
        logits,
        state.indicators,
        slots,
        None,
        state.grammar,
        state.context_counts,
        state.decode_counts,
        drafted,
        state.repetition,
        state.frequency,
        state.presence,
        state.logit_bias,
        state.temperature,
        state.seeds,
        offsets,
        scratchpad,
        vocab_size,
        0 if shard is None else shard.vocab_start,
        gamma,
        batch_size,
        blocks,
        group_blocks,
        False,
        not singular,
        True,
        singular,
        enable_pdl,
        logits.stride(0),
        stride_g,
        state.decode_counts.stride(0),
        None,
        scratchpad.stride(0),
        config.block_n,
        block_g,
        tl.float32 if exact else tl.bfloat16,
        logit_scale=None if shard is None else shard.logit_scale,
        softcap=None if shard is None else shard.softcap,
        num_warps=4,
    )


@functools.cache
def _whole_programs(device: torch.device) -> int:
    """Programs the whole-vocabulary reduction spreads its rows over: eight per SM."""
    return 8 * torch.cuda.get_device_properties(device).multi_processor_count


def shard_candidates(
    logits: Tensor,
    state: SlotState,
    slots: Tensor,
    offsets: Tensor,
    drafted: Tensor | None,
    scratchpad: Tensor,
    shard: VocabShard,
    config: SonicConfig,
    enable_pdl: bool,
    draft_probs: Tensor | None = None,
) -> Tensor:
    """This rank's best ``MAX_K`` candidates per logits row, for the ranks' gather.

    Args:
        logits: ``[rows, V_s]`` bf16 or fp32 raw head logits of this rank's shard.
        state: The slot state; fp32 scalars with fp32 logits.
        slots: ``[B]`` int64 slot of each request.
        offsets: ``[P]`` int64 Philox offset of each slot (its first row's position).
        drafted: ``[B, g + 1]`` int32 drafts of a verify launch (rows request-major), or None for one row per request.
        scratchpad: ``[rows, >= scratchpad_columns(V_s)]`` uint32 keys.
        shard: This rank's slice; its ``pairs`` receive the candidates.
        config: The launch config (``pick`` over the shard's columns).
        enable_pdl: Launch with programmatic dependent launch.
        draft_probs: ``[P, >= g, V_s]`` fp32 draft probabilities this rank
            recorded over its shard's columns, for a verify launch, or None.

    Returns:
        Contiguous ``[rows, 2 * MAX_K]`` fp32 view of ``shard.pairs``: each
        row's candidate values in descending order (bf16 values for bf16
        logits, fp32 for fp32), the lower token id first among equal values,
        then their global token ids as exact fp32 integers; a greedy row keeps
        only its first maximum, the rest -inf, and a row drawing from its
        whole vocabulary its best perturbed value (``merged_whole`` reads
        it). With ``draft_probs``,
        ``[rows, 3 * MAX_K]``: then the probability the row's draft recorded
        at each candidate (zero on greedy rows and on each request's last row).

    Raises:
        ValueError: The logits are neither bf16 nor fp32, or a row offset
            would pass ``MAX_OFFSET``.
    """
    _check_offsets(logits, scratchpad)
    _reduce(
        logits, state, slots, offsets, drafted, scratchpad, config, enable_pdl, shard
    )
    exact = _exact(logits)
    rows, vocab_size = logits.shape
    blocks = triton.cdiv(vocab_size, config.block_n)
    block_v = next_power_of_2(blocks)
    width = (2 if draft_probs is None else 3) * MAX_K
    pairs = shard.pairs.view(-1)[: rows * width].view(rows, width)
    if drafted is None:
        gamma, block_g, stride_g = 0, None, state.grammar.stride(0)
    else:
        gamma, block_g = drafted.shape[1] - 1, next_power_of_2(drafted.shape[1])
        stride_g = state.grammar.flatten(0, -2).stride(0)
    shard_candidates_kernel[(rows,)](
        scratchpad,
        state.indicators,
        slots,
        pairs,
        blocks,
        gamma,
        MAX_K,
        block_v * MAX_K,
        not config.radix,
        enable_pdl,
        scratchpad.stride(0),
        pairs.stride(0),
        config.block_n,
        block_v,
        logits if exact else None,
        state.grammar if exact else None,
        state.context_counts if exact else None,
        state.decode_counts if exact else None,
        drafted if exact else None,
        state.repetition if exact else None,
        state.frequency if exact else None,
        state.presence if exact else None,
        state.logit_bias if exact else None,
        state.temperature if exact else None,
        vocab_size,
        shard.vocab_start,
        logits.stride(0),
        stride_g,
        state.decode_counts.stride(0),
        block_g,
        exact,
        shard.logit_scale,
        shard.softcap,
        d_ptr=draft_probs,
        stride_d=None if draft_probs is None else draft_probs.stride(0),
        stride_dg=None if draft_probs is None else draft_probs.stride(1),
        whole=True,
        num_warps=config.second_warps,
    )
    return pairs


def _merged(
    logits: Tensor,
    state: SlotState,
    slots: Tensor,
    offsets: Tensor,
    drafted: Tensor | None,
    scratchpad: Tensor,
    shard: VocabShard | None,
    config: SonicConfig,
    enable_pdl: bool,
    draft_probs: Tensor | None = None,
) -> dict:
    """The selection kernels' candidate arguments: none for whole rows, else every rank's gathered candidates (with ``draft_probs``, carrying them), which they merge."""
    if shard is None:
        _reduce(
            logits, state, slots, offsets, drafted, scratchpad, config, enable_pdl, None
        )
        return {}
    pairs = shard.gather(
        shard_candidates(
            logits,
            state,
            slots,
            offsets,
            drafted,
            scratchpad,
            shard,
            config,
            enable_pdl,
            draft_probs,
        )
    )
    ranks = pairs.shape[0]
    if ranks & (ranks - 1):
        raise ValueError(f"sonic merges a power-of-two count of ranks, got {ranks}")
    return dict(
        cand_ptr=pairs,
        ranks=ranks,
        stride_mr=pairs.stride(0),
        stride_mb=pairs.stride(1),
    )


def sample(
    logits: Tensor,
    state: SlotState,
    slots: Tensor,
    offsets: Tensor,
    tokens: Tensor,
    scratchpad: Tensor,
    config: SonicConfig,
    enable_pdl: bool,
    shard: VocabShard | None = None,
) -> Tensor:
    """Draw one token per logits row from its slot's distribution, counting it in the slot's decode counts.

    Args:
        logits: ``[B, V]`` bf16 or fp32 logits, or this rank's ``[B, V_s]``
            raw head logits with ``shard``.
        state: The slot state; fp32 scalars with fp32 logits.
        slots: ``[B]`` int64 slot of each row.
        offsets: ``[P]`` int64 Philox offset of each slot (the drawn token's position).
        tokens: ``[B, 1]`` int32 output.
        scratchpad: ``[B, >= scratchpad_columns(V)]`` uint32 keys.
        config: The launch config (``pick``, over the shard's columns with ``shard``).
        enable_pdl: Launch with programmatic dependent launch.
        shard: This rank's vocabulary slice when the logits are sharded;
            every rank must launch with the same rows.

    Returns:
        ``tokens``.

    Raises:
        ValueError: The logits are neither bf16 nor fp32, or a row offset
            would pass ``MAX_OFFSET``.
    """
    _check_offsets(logits, scratchpad)
    exact = _exact(logits)
    rows, vocab_size = logits.shape
    blocks = triton.cdiv(vocab_size, config.block_n)
    block_v = next_power_of_2(blocks)
    stride_x, stride_y = logits.stride(0), scratchpad.stride(0)
    stride_g, stride_c = state.grammar.stride(0), state.decode_counts.stride(0)

    merged = _merged(
        logits, state, slots, offsets, None, scratchpad, shard, config, enable_pdl
    )
    cumulative_selection_kernel[(rows,)](
        scratchpad,
        state.indicators,
        slots,
        None,
        state.decode_counts,
        None,
        state.seeds,
        offsets,
        state.top_k,
        state.top_p,
        state.min_p,
        None,
        tokens,
        None,
        None,
        None,
        None,
        blocks,
        1,
        vocab_size,
        MAX_K,
        None,
        block_v,
        block_v * MAX_K,
        False,
        True,
        False,
        False,
        True,
        False,
        not config.radix,
        enable_pdl,
        False,
        False,
        stride_y,
        None,
        None,
        tokens.stride(0),
        None,
        None,
        stride_c,
        None,
        None,
        config.block_n,
        block_v,
        _LOGPROB_BLOCK,
        logits if exact else None,
        state.grammar if exact else None,
        state.context_counts if exact else None,
        None,
        state.repetition if exact else None,
        state.frequency if exact else None,
        state.presence if exact else None,
        state.logit_bias if exact else None,
        state.temperature if exact else None,
        vocab_size,
        0,
        0,
        False,
        stride_x,
        stride_g,
        None,
        None,
        exact,
        **merged,
        whole=True,
        num_warps=config.second_warps,
    )
    return tokens


def verify(
    logits: Tensor,
    state: SlotState,
    slots: Tensor,
    offsets: Tensor,
    drafted: Tensor,
    accepted: Tensor,
    scratchpad: Tensor,
    values: Tensor,
    indices: Tensor,
    config: SonicConfig,
    enable_pdl: bool,
    shard: VocabShard | None = None,
    draft_probs: Tensor | None = None,
    reject_threshold: float | None = None,
    candidate_drafts: Tensor | None = None,
) -> None:
    """Verify each request's drafts against its slot's distribution by rejection sampling.

    Point-draw drafts (``draft_probs`` None, or a slot's ``Indicator``
    greedy-draft bit set): accepts draft ``j`` with probability ``p(draft)``
    and draws the rejected position from ``p`` without the draft. Drafts
    sampled from a recorded ``q``: accepts with probability
    ``min(1, p(draft) / q(draft))`` and draws the rejected position from
    ``max(p - q, 0)``, which leaves the target ``p``. Emitted tokens count in
    the slot's decode counts.

    Args:
        logits: ``[B * (g + 1), V]`` bf16 or fp32 logits, request-major, or
            this rank's ``[B * (g + 1), V_s]`` raw head logits with ``shard``.
        state: The slot state; fp32 scalars with fp32 logits.
        slots: ``[B]`` int64 slot of each request.
        offsets: ``[P]`` int64 Philox offset of each slot (its first row's position).
        drafted: ``[B, g + 1]`` int32 drafts in the first ``g`` columns; on
            return the accepted prefix and the next token.
        accepted: ``[B, 1]`` int32 output: accepted drafts per request.
        scratchpad: ``[B * (g + 1), >= scratchpad_columns(V)]`` uint32 keys.
        values: ``[B * (g + 1), MAX_K]`` fp32 candidate probabilities (scratch).
        indices: ``[B * (g + 1), MAX_K]`` int32 candidate tokens (scratch).
        config: The launch config (``pick``, over the shard's columns with ``shard``).
        enable_pdl: Launch with programmatic dependent launch.
        shard: This rank's vocabulary slice when the logits are sharded;
            every rank must launch with the same rows.
        draft_probs: ``[P, >= g, V]`` fp32 draft distributions by slot, row
            ``j`` the ``q`` draft ``j`` was sampled from (the runtime's
            ``draft_probs``), or None for point-draw drafts. With ``shard``
            it may instead be ``[P, >= g, V_s]``, the distributions over this
            rank's columns only: each rank's candidates then carry their
            probabilities through the gather.
        reject_threshold: Recorded probabilities above it mark a slot with
            no proposal: its draft is rejected and the token drawn from ``p``.
            Required with ``draft_probs``.
        candidate_drafts: ``[B * (g + 1), MAX_K]`` fp32 scratch for the
            candidates' carried draft probabilities; required with
            shard-local ``draft_probs``.

    Raises:
        ValueError: The logits are neither bf16 nor fp32, a row offset
            would pass ``MAX_OFFSET``, or ``draft_probs`` rows are not dense
            rows over the whole vocabulary or this rank's shard, come without
            ``reject_threshold``, or are shard-local without
            ``candidate_drafts``.
    """
    _check_offsets(logits, scratchpad)
    exact = _exact(logits)
    rows, vocab_size = logits.shape
    batch_size, timesteps = drafted.shape
    stride_c = state.decode_counts.stride(0)
    if draft_probs is not None and reject_threshold is None:
        raise ValueError(
            "sonic reads draft_probs with the no-proposal sentinel: pass reject_threshold"
        )
    # A shard's columns are fewer than the vocabulary's, so the width tells a shard-local table.
    local = (
        draft_probs is not None
        and shard is not None
        and draft_probs.shape[2] == vocab_size != stride_c
    )
    columns = vocab_size if local else stride_c
    if draft_probs is not None and (
        draft_probs.dtype != torch.float32
        or draft_probs.shape[1] < timesteps - 1
        or draft_probs.shape[2] != columns
        or draft_probs.stride(1) != columns
        or draft_probs.stride(2) != 1
    ):
        raise ValueError(
            f"sonic reads draft_probs as fp32 [P, >= {timesteps - 1}, {columns}] "
            f"dense rows, got {draft_probs.dtype} {tuple(draft_probs.shape)} "
            f"strides {draft_probs.stride()}"
        )
    if local and (
        candidate_drafts is None
        or candidate_drafts.dtype != torch.float32
        or candidate_drafts.shape[0] < rows
        or candidate_drafts.shape[1] != MAX_K
        or not candidate_drafts.is_contiguous()
    ):
        raise ValueError(
            f"shard-local draft_probs need a contiguous fp32 [>= {rows}, {MAX_K}] "
            "candidate_drafts scratch"
        )
    gamma = timesteps - 1
    blocks = triton.cdiv(vocab_size, config.block_n)
    block_v, block_g = next_power_of_2(blocks), next_power_of_2(timesteps)
    stride_x, stride_y = logits.stride(0), scratchpad.stride(0)
    stride_g = state.grammar.flatten(0, -2).stride(0)

    merged = _merged(
        logits,
        state,
        slots,
        offsets,
        drafted,
        scratchpad,
        shard,
        config,
        enable_pdl,
        draft_probs if local else None,
    )
    if local:
        merged["m_ptr"] = candidate_drafts
    cumulative_unpack_kernel[(rows,)](
        scratchpad,
        state.indicators,
        slots,
        None,
        state.top_k,
        state.top_p,
        state.min_p,
        values,
        indices,
        blocks,
        1,
        vocab_size,
        gamma,
        MAX_K,
        None,
        batch_size,
        block_v,
        block_v * MAX_K,
        False,
        True,
        False,
        False,
        not config.radix,
        enable_pdl,
        stride_y,
        None,
        None,
        config.block_n,
        block_v,
        None,
        logits if exact else None,
        state.grammar if exact else None,
        state.context_counts if exact else None,
        state.decode_counts if exact else None,
        drafted,
        state.repetition if exact else None,
        state.frequency if exact else None,
        state.presence if exact else None,
        state.logit_bias if exact else None,
        state.temperature if exact else None,
        vocab_size,
        0,
        stride_x,
        stride_g,
        stride_c,
        block_g,
        exact,
        **merged,
        whole=True,
        num_warps=config.second_warps,
    )
    chain_speculative_verification_kernel[(batch_size,)](
        values,
        indices,
        state.indicators,
        slots,
        None,
        drafted,
        None if local else draft_probs,
        None,
        None,
        state.seeds,
        offsets,
        state.decode_counts,
        None,
        drafted,
        accepted,
        None,
        None,
        None,
        gamma,
        MAX_K,
        True,
        False,
        enable_pdl,
        True,
        False,
        None if draft_probs is None or local else draft_probs.stride(0),
        None,
        None,
        stride_c,
        None,
        block_g,
        _LOGPROB_BLOCK,
        reject_threshold=None if draft_probs is None else reject_threshold,
        m_ptr=candidate_drafts if local else None,
        num_warps=next_power_of_2(timesteps),
    )


__all__ = [
    "MAX_K",
    "MAX_OFFSET",
    "TABLE",
    "Indicator",
    "SlotState",
    "SonicConfig",
    "VocabShard",
    "configs",
    "pick",
    "sample",
    "scratchpad_columns",
    "shard_candidates",
    "synthetic_targets",
    "verify",
]


@triton.jit
def _synthetic_target_kernel(
    values_ptr,
    indices_ptr,
    indicators_ptr,
    slots_ptr,
    seeds_ptr,
    offsets_ptr,
    lengths_ptr,
    tokens_ptr,
    timesteps: tl.constexpr,
    max_k: tl.constexpr,
    enable_pdl: tl.constexpr,
):
    """One program per request: its target token at the forced position ``lengths - 1``.

    A greedy request (indicator bit 0x40) reads the argmax ``verify`` stored at
    its first row, and a request drawing from its whole vocabulary the draw
    ``verify`` stored in its row's third slot. Any other request takes the
    Gumbel-max over that position's ``MAX_K`` candidates, keyed by the slot's
    seed and offset and the position, which is the draw ``verify`` makes past
    an all-accepted window.
    """
    request = tl.program_id(axis=0)
    if enable_pdl:
        gdc_wait()
    slot = tl.load(slots_ptr + request)
    indicator = tl.load(indicators_ptr + slot)
    step = tl.load(lengths_ptr + request) - 1
    margin = request * timesteps
    if (indicator & 0x40) > 0:
        # Greedy requests: ``verify`` stores each position's argmax at the first row.
        token = tl.load(indices_ptr + margin * max_k + step)
    elif whole_row(indicator, True, False):
        # The position's whole-vocabulary draw, its draft included.
        token = tl.load(indices_ptr + (margin + step) * max_k + 2)
    else:
        # The draw ``verify`` makes past an all-accepted window: Gumbel-max over the
        # position's candidates, keyed by the slot's seed, offset and the position.
        columns = tl.arange(0, max_k)
        row = (margin + step) * max_k
        probabilities = tl.load(values_ptr + row + columns)
        indices = tl.load(indices_ptr + row + columns)
        gumbel = philox_gumbel(philox_key(seeds_ptr, offsets_ptr, slot, step), indices)
        selection = tl.argmax(libdevice.fast_logf(probabilities) - gumbel, axis=0)
        token = tl.sum(tl.where(columns == selection, indices, 0))
    if enable_pdl:
        gdc_launch_dependents()
    tl.store(tokens_ptr + request, token)


def synthetic_targets(
    values: Tensor,
    indices: Tensor,
    state: SlotState,
    slots: Tensor,
    offsets: Tensor,
    lengths: Tensor,
    tokens: Tensor,
    enable_pdl: bool,
) -> Tensor:
    """Draw each request's target token at a forced verify position.

    Synthetic acceptance replaces ``verify``'s outcome with a forced length
    ``L``: the first ``L - 1`` drafts, then a token drawn from the target at
    position ``L - 1``. This draws that token from the candidates ``verify``
    left in its scratch, as ``verify`` itself draws past an all-accepted
    window: a greedy request's argmax, a whole-vocabulary request's draw
    there, otherwise Gumbel-max over the position's candidates with the
    slot's Philox noise for that position.

    Args:
        values: ``[B * T, MAX_K]`` fp32 candidate probabilities ``verify`` wrote.
        indices: ``[B * T, MAX_K]`` int32 candidate tokens ``verify`` wrote.
        state: The slot state ``verify`` ran with.
        slots: ``[B]`` int64 slot of each request.
        offsets: ``[P]`` int64 Philox offset of each slot, as passed to ``verify``.
        lengths: ``[B]`` int32 forced lengths in ``[1, T]``.
        tokens: ``[B]`` int32 output.
        enable_pdl: Launch with programmatic dependent launch.

    Returns:
        ``tokens``.

    Raises:
        ValueError: The candidates are not dense ``[B * T, MAX_K]`` rows, or
            their offsets would pass ``MAX_OFFSET``.
    """
    batch = slots.shape[0]
    rows = values.shape[0]
    if (
        rows % batch
        or indices.shape != values.shape
        or values.shape[1] != MAX_K
        or values.stride() != (MAX_K, 1)
        or indices.stride() != (MAX_K, 1)
    ):
        raise ValueError(
            f"sonic synthetic targets need dense [B * T, {MAX_K}] candidates for "
            f"{batch} requests, got {tuple(values.shape)} and {tuple(indices.shape)}"
        )
    # The kernel keeps (request * T + position) * MAX_K in int32.
    if rows * MAX_K > MAX_OFFSET:
        raise ValueError(
            f"sonic synthetic targets offsets pass {MAX_OFFSET}: {rows} rows"
        )
    _synthetic_target_kernel[(batch,)](
        values,
        indices,
        state.indicators,
        slots,
        state.seeds,
        offsets,
        lengths,
        tokens,
        rows // batch,
        MAX_K,
        enable_pdl,
        num_warps=1,
    )
    return tokens
