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

The kernels are vendored in ``tokenspeed_kernel.thirdparty.sonic.ops`` (see the
package README for the upstream commit and the local changes); this module
launches them over the backend's slot-indexed state and picks their config
from a table keyed by vocabulary size class and logits rows, so no server
tunes at startup. On GB200 (13 vocabularies from 32k to 262k, 1 to 512 rows,
plain decode and three-draft verify, 60 configs swept) its picks run within 8%
of the best config at every point and 0.3% on average, and 3.6x to 11x faster
than the FlashInfer backend. The other Blackwell GPUs share it; any other GPU
uses it unmeasured.

Every config draws the same tokens (the selection is exact; only its tiling
and warps differ), so the table changes speed only. bf16 logits rank at bf16
resolution; fp32 logits (soft-capped heads) are sampled exactly in fp32 among
the top-128 candidates their bf16 keys pick.
"""

from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.thirdparty.sonic.core.flags import Indicator
from tokenspeed_kernel.thirdparty.sonic.ops.base import MAX_K, next_power_of_2
from tokenspeed_kernel.thirdparty.sonic.ops.multistep import cumulative_unpack_kernel
from tokenspeed_kernel.thirdparty.sonic.ops.prologue import bitpacked_reduction_kernel
from tokenspeed_kernel.thirdparty.sonic.ops.singular import cumulative_selection_kernel
from tokenspeed_kernel.thirdparty.sonic.ops.verify import (
    chain_speculative_verification_kernel,
)
from torch import Tensor

# The kernels' block for top-k logprob targets, which the runtime never requests.
_LOGPROB_BLOCK = 8

# Adaptive top-k's scale factor, unread with adaptive top-k off.
_TOP_K_SCALE = 4


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

    indicators: Tensor  # [P] uint16 ``Indicator`` bits.
    temperature: Tensor  # [P].
    top_k: Tensor  # [P] int32, at most ``MAX_K``.
    top_p: Tensor  # [P].
    min_p: Tensor  # [P].
    repetition: Tensor  # [P] multiplicative repetition penalty.
    frequency: Tensor  # [P].
    presence: Tensor  # [P].
    grammar: Tensor  # [P, timesteps, ceil(V / 32)] uint32 bitmask, set bits allowed.
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
                indicators=torch.zeros(slots, dtype=torch.uint16),
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
                    dtype=torch.uint32,
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


def _exact(logits: Tensor) -> bool:
    """Whether the logits are fp32, which the kernels sample exactly; bf16 is the other supported dtype."""
    if logits.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"sonic samples bf16 or fp32 logits, got {logits.dtype}")
    return logits.dtype == torch.float32


def sample(
    logits: Tensor,
    state: SlotState,
    slots: Tensor,
    offsets: Tensor,
    tokens: Tensor,
    scratchpad: Tensor,
    config: SonicConfig,
    enable_pdl: bool,
) -> Tensor:
    """Draw one token per logits row from its slot's distribution, counting it in the slot's decode counts.

    Args:
        logits: ``[B, V]`` bf16 or fp32 logits.
        state: The slot state; fp32 scalars with fp32 logits.
        slots: ``[B]`` int64 slot of each row.
        offsets: ``[P]`` int64 Philox offset of each slot (the drawn token's position).
        tokens: ``[B, 1]`` int32 output.
        scratchpad: ``[B, >= scratchpad_columns(V)]`` uint32 keys.
        config: The launch config (``pick``).
        enable_pdl: Launch with programmatic dependent launch.

    Returns:
        ``tokens``.
    """
    exact = _exact(logits)
    rows, vocab_size = logits.shape
    blocks = triton.cdiv(vocab_size, config.block_n)
    block_v = next_power_of_2(blocks)
    stride_x, stride_y = logits.stride(0), scratchpad.stride(0)
    stride_g, stride_c = state.grammar.stride(0), state.decode_counts.stride(0)

    bitpacked_reduction_kernel[(rows, blocks)](
        logits,
        state.indicators,
        slots,
        None,
        None,
        state.grammar,
        state.context_counts,
        state.decode_counts,
        None,
        state.repetition,
        state.frequency,
        state.presence,
        state.logit_bias,
        state.temperature,
        scratchpad,
        vocab_size,
        0,
        0,
        MAX_K,
        _TOP_K_SCALE,
        rows,
        False,
        False,
        True,
        True,
        False,
        False,
        config.radix,
        enable_pdl,
        stride_x,
        stride_g,
        stride_c,
        None,
        stride_y,
        config.block_n,
        None,
        None,
        exact,
        tl.float32 if exact else tl.bfloat16,
        num_warps=config.first_warps,
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
) -> None:
    """Verify each request's point-draw drafts against its slot's distribution by masked rejection.

    Accepts draft ``j`` with probability ``p(draft)``, draws the rejected
    position from ``p`` without the draft, counting emitted tokens in the
    slot's decode counts.

    Args:
        logits: ``[B * (g + 1), V]`` bf16 or fp32 logits, request-major.
        state: The slot state; fp32 scalars with fp32 logits.
        slots: ``[B]`` int64 slot of each request.
        offsets: ``[P]`` int64 Philox offset of each slot (its first row's position).
        drafted: ``[B, g + 1]`` int32 drafts in the first ``g`` columns; on
            return the accepted prefix and the next token.
        accepted: ``[B, 1]`` int32 output: accepted drafts per request.
        scratchpad: ``[B * (g + 1), >= scratchpad_columns(V)]`` uint32 keys.
        values: ``[B * (g + 1), MAX_K]`` fp32 candidate probabilities (scratch).
        indices: ``[B * (g + 1), MAX_K]`` int32 candidate tokens (scratch).
        config: The launch config (``pick``).
        enable_pdl: Launch with programmatic dependent launch.
    """
    exact = _exact(logits)
    rows, vocab_size = logits.shape
    batch_size, timesteps = drafted.shape
    gamma = timesteps - 1
    blocks = triton.cdiv(vocab_size, config.block_n)
    block_v, block_g = next_power_of_2(blocks), next_power_of_2(timesteps)
    stride_x, stride_y = logits.stride(0), scratchpad.stride(0)
    stride_g = state.grammar.flatten(0, -2).stride(0)
    stride_c = state.decode_counts.stride(0)

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
        0,
        gamma,
        MAX_K,
        _TOP_K_SCALE,
        batch_size,
        False,
        True,
        True,
        False,
        False,
        False,
        config.radix,
        enable_pdl,
        stride_x,
        stride_g,
        stride_c,
        None,
        stride_y,
        config.block_n,
        None,
        block_g,
        exact,
        tl.float32 if exact else tl.bfloat16,
        num_warps=config.first_warps,
    )
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
        drafted if exact else None,
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
        num_warps=config.second_warps,
    )
    chain_speculative_verification_kernel[(batch_size,)](
        values,
        indices,
        state.indicators,
        slots,
        None,
        drafted,
        None,
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
        None,
        None,
        None,
        stride_c,
        None,
        block_g,
        _LOGPROB_BLOCK,
        num_warps=next_power_of_2(timesteps),
    )


__all__ = [
    "MAX_K",
    "TABLE",
    "Indicator",
    "SlotState",
    "SonicConfig",
    "configs",
    "pick",
    "sample",
    "scratchpad_columns",
    "verify",
]
