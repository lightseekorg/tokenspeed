# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026 SonicSampler Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from tokenspeed_kernel._triton import tl, triton  # noqa.

cdiv, jit = triton.cdiv, triton.jit

from tokenspeed_kernel.thirdparty.sonic_sampler.ops.base import (
    gdc_launch_dependents,
    gdc_wait,
    ones_like,
)
from tokenspeed_kernel.thirdparty.sonic_sampler.ops.encoding import bitpack_scalar
from tokenspeed_kernel.thirdparty.sonic_sampler.ops.topk.bitpack import top_k_reduction

__all__ = ["bitpacked_reduction_kernel"]


@jit
def grammar(
    g_ptr,
    logits,
    indicator,
    batch_id,
    step_id,
    timesteps,
    cols,
    mask,
    minima,
    stride_g: tl.constexpr,
):

    if g_ptr is not None and (indicator & 0x1):

        target_id = (batch_id * timesteps) + step_id

        bitmask = tl.load(
            g_ptr + (target_id * stride_g) + (cols // 32), mask=mask, other=-1
        ) >> (cols % 32)

        logits = tl.where((bitmask & 0x1) > 0, logits, minima)

    return logits


@jit
def repetition(
    c_ptr,
    r_ptr,
    f_ptr,
    p_ptr,
    batch_id,
    indicator,
    logits,
    decode,
    minima,
    cols,
    mask,
    stride_c: tl.constexpr,
    block_n: tl.constexpr,
    value_dtype: tl.constexpr,
):

    values = logits

    if c_ptr is not None and (indicator & 0x2) > 0:

        # Apply multiplicative penalties.

        total = decode + tl.load(
            c_ptr + (batch_id * stride_c) + cols, mask=mask, other=0
        )

        scales = tl.load(r_ptr + batch_id).broadcast_to(block_n)
        scales = tl.where(total > 0, scales, ones_like(scales))

        values = tl.where(values < 0, values * scales, values / scales).to(value_dtype)

    if (indicator & 0x4) > 0:

        # Apply frequency penalties.

        values -= decode.to(value_dtype) * tl.load(f_ptr + batch_id)

    if (indicator & 0x8) > 0:

        # Apply presence penalties.

        values -= (decode > 0).to(value_dtype) * tl.load(p_ptr + batch_id)

    return tl.where(mask, values, minima)


@jit
def logit_bias(b_ptr, indicator, logits, batch_id, cols, mask, stride_b: tl.constexpr):

    values = logits

    if b_ptr is not None and (indicator & 0x10) > 0:

        values += tl.load(b_ptr + (batch_id * stride_b) + cols, mask=mask, other=0.0)

    return values


@jit
def greedy_reduction(
    y_ptr,
    batch_id,
    block_id,
    logits,
    enable_pdl: tl.constexpr,
    stride_y: tl.constexpr,
    block_n: tl.constexpr,
):

    value, index = tl.max(logits, axis=0, return_indices=True)

    maximum = bitpack_scalar(value.to(tl.bfloat16), index, block_n)

    if enable_pdl:

        gdc_launch_dependents()

    target = (batch_id * stride_y) + block_id

    tl.store(y_ptr + target, maximum)


@jit
def decode_counts(
    d_ptr,
    t_ptr,
    batch_id,
    step_id,
    margin,
    cols,
    mask,
    lookahead: tl.constexpr,
    tokens: tl.constexpr,
    singular: tl.constexpr,
    stride_d: tl.constexpr,
    stride_t: tl.constexpr,
    block_t: tl.constexpr,
):
    """
    Load decode counts block and conditionally update based on drafted tokens availability.

    Note(s):

        • The drafted tokens i.e. `t_ptr` here should index column-wise from the first draft
          timestep i.e. the first drafted token:

                [ t_1, t_2, ..., t_γ ]

        • In the non-singular case, it is assumed that the drafted tokens has an additional
          slot for the additional rejection-sampled token by the target model:

                [ t_1, t_2, ..., t_γ, t_{γ+1} ]

    """
    decode = tl.load(d_ptr + (batch_id * stride_d) + cols, mask=mask, other=0)

    if tokens:

        if singular:

            timesteps = tl.arange(0, block_t)
            step_mask = timesteps < lookahead

            drafted = tl.load(
                t_ptr + (batch_id * stride_t) + timesteps, mask=step_mask, other=-1
            )

            decode += tl.sum((drafted[:, None] == cols[None, :]).to(tl.int32), axis=0)

        elif step_id > 0:

            time_steps = tl.arange(0, block_t)

            block_mask = time_steps < step_id
            draft_tokens = tl.load(
                t_ptr + margin + time_steps, mask=block_mask, other=-1
            )

            decode += tl.sum(
                (draft_tokens[:, None] == cols[None, :]).to(tl.int32), axis=0
            )

    return decode


@jit
def transform(
    logits,
    indicator,
    cols,
    mask,
    m_ptr,
    g_ptr,
    c_ptr,
    d_ptr,
    t_ptr,
    r_ptr,
    f_ptr,
    p_ptr,
    b_ptr,
    e_ptr,
    batch_id,
    step_id,
    margin,
    timesteps,
    shard_offset: tl.constexpr,
    lookahead: tl.constexpr,
    indirection: tl.constexpr,
    tokens: tl.constexpr,
    singular: tl.constexpr,
    stride_g: tl.constexpr,
    stride_c: tl.constexpr,
    stride_t: tl.constexpr,
    block_t: tl.constexpr,
    width: tl.constexpr,
    value_dtype: tl.constexpr,
):
    """A request's grammar mask, repetition penalties, logit bias and temperature applied to ``logits`` at ``cols``, kept in ``value_dtype``."""

    minima: tl.constexpr = -float("inf")

    if (indicator & 0x1F) > 0:

        # Has at least one of grammar, repetition, and / or logit bias.

        cols += shard_offset

        if indirection:

            cols = tl.load(m_ptr + cols, mask=mask, other=0)

        # Apply grammar bit-mask.

        logits = grammar(
            g_ptr,
            logits,
            indicator,
            batch_id,
            step_id,
            timesteps,
            cols,
            mask,
            minima,
            stride_g,
        )

        # Apply repetition penalties (multiplicative, frequency, and / or presence).

        if d_ptr is not None and (indicator & 0xE) > 0:

            decode = decode_counts(
                d_ptr,
                t_ptr,
                batch_id,
                step_id,
                margin,
                cols,
                mask,
                lookahead,
                tokens,
                singular,
                stride_c,
                stride_t,
                block_t,
            )

            logits = repetition(
                c_ptr,
                r_ptr,
                f_ptr,
                p_ptr,
                batch_id,
                indicator,
                logits,
                decode,
                minima,
                cols,
                mask,
                stride_c,
                width,
                value_dtype,
            )

        # Apply logit bias.

        logits = logit_bias(b_ptr, indicator, logits, batch_id, cols, mask, stride_c)

    if (indicator & 0x20) > 0:

        # Apply temperature.

        logits = (logits / tl.load(e_ptr + batch_id)).to(value_dtype)

    return logits


@jit
def exact_maximum(
    y_ptr,
    row_id,
    block_id,
    blocks,
    shift,
    logits,
    enable_pdl: tl.constexpr,
    stride_y: tl.constexpr,
):
    """Store a block's exact fp32 maximum and its first column: the value bits in the row's first ``blocks`` slots, the column after them."""

    value, index = tl.max(logits, axis=0, return_indices=True)

    if enable_pdl:

        gdc_launch_dependents()

    target = (row_id * stride_y) + block_id

    tl.store(y_ptr + target, value.to(tl.uint32, bitcast=True))
    tl.store(y_ptr + target + blocks, (shift + index).to(tl.uint32))


@jit
def exact_greedy(
    s_ptr,
    row_id,
    vocab_blocks: tl.constexpr,
    enable_pdl: tl.constexpr,
    stride_s: tl.constexpr,
    block_v: tl.constexpr,
):
    """The first column holding the row's exact fp32 maximum, from the blocks' ``exact_maximum`` slots."""

    span = tl.arange(0, block_v)
    span_mask = span < vocab_blocks
    offsets = (row_id * stride_s) + span

    if enable_pdl:

        gdc_wait()

    maxima = tl.load(s_ptr + offsets, mask=span_mask, other=0xFF800000)
    columns = tl.load(s_ptr + offsets + vocab_blocks, mask=span_mask, other=0)

    # Blocks run in vocabulary order, so the first maximal block holds the first maximal column.

    _, block = tl.max(maxima.to(tl.float32, bitcast=True), axis=0, return_indices=True)

    return tl.sum(tl.where(span == block, columns, 0), axis=0).to(tl.int32)


@jit
def exact_candidates(
    x_ptr,
    row_id,
    indices,
    indicator,
    m_ptr,
    g_ptr,
    c_ptr,
    d_ptr,
    t_ptr,
    r_ptr,
    f_ptr,
    p_ptr,
    b_ptr,
    e_ptr,
    batch_id,
    step_id,
    margin,
    timesteps,
    vocab_size: tl.constexpr,
    shard_offset: tl.constexpr,
    lookahead: tl.constexpr,
    indirection: tl.constexpr,
    tokens: tl.constexpr,
    singular: tl.constexpr,
    stride_x: tl.constexpr,
    stride_g: tl.constexpr,
    stride_c: tl.constexpr,
    stride_t: tl.constexpr,
    block_t: tl.constexpr,
    max_k: tl.constexpr,
):
    """The candidates' exact fp32 values, recomputed from the logits, sorted descending with the lower column first on ties."""

    index_dtype: tl.constexpr = indices.dtype

    mask = indices < vocab_size
    columns = tl.where(mask, indices, 0).to(tl.int32)

    logits = tl.load(
        x_ptr + (row_id * stride_x) + columns, mask=mask, other=-float("inf")
    )

    logits = transform(
        logits,
        indicator,
        columns,
        mask,
        m_ptr,
        g_ptr,
        c_ptr,
        d_ptr,
        t_ptr,
        r_ptr,
        f_ptr,
        p_ptr,
        b_ptr,
        e_ptr,
        batch_id,
        step_id,
        margin,
        timesteps,
        shard_offset,
        lookahead,
        indirection,
        tokens,
        singular,
        stride_g,
        stride_c,
        stride_t,
        block_t,
        max_k,
        tl.float32,
    )

    # Order-preserving fp32 bits above the inverted column: one descending sort orders both.

    bits = logits.to(tl.uint32, bitcast=True)
    ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits | 0x80000000)

    keys = (ordered.to(tl.uint64) << 32) | (~columns.to(tl.uint32)).to(tl.uint64)
    keys = tl.sort(keys, descending=True)

    ordered = (keys >> 32).to(tl.uint32)
    bits = tl.where((ordered & 0x80000000) != 0, ordered & 0x7FFFFFFF, ~ordered)

    values = bits.to(tl.float32, bitcast=True)
    indices = (~(keys & 0xFFFFFFFF).to(tl.uint32)).to(index_dtype)

    return values, indices


@jit
def resolve_indices(
    q_ptr,
    s_ptr,
    row_id,
    lookahead: tl.constexpr,
    batch_size: tl.int32,
    slotted: tl.constexpr,
    singular: tl.constexpr,
    varlen: tl.constexpr,
    block_q: tl.constexpr,
):
    """
    Resolves the batch id, step id, and margin given the row id of the current tile
    and supporting metadata under `singular`, `varlen`, and `multistep` settings.

    Note(s):

        • The `batch_id` resolves to the sequence index of the current tile, possibly
          re-mapped via slot indirection.

        • The `step_id`, applicable only in `multistep` setting, resolves to the time
          index of the current tile.

        • The `margin` resolves to the offset of the current tile within the in-flight batch.

        • The `timesteps` refers to the maximum number of tokens per query in the batch,
          which defaults to 1 for singular and `γ + 1` for multistep.

    """
    if singular:

        batch_id = row_id
        margin = row_id

        step_id = 0
        timesteps = 1

    elif varlen:

        row_ids = tl.arange(0, block_q)
        row_mask = row_ids < (batch_size + 1)

        # Note: Ensuring cached here allows for leveraging L2 in loading the `margin`.

        lengths = tl.load(q_ptr + row_ids, mask=row_mask, other=0, cache_modifier=".cv")
        marked = ((lengths <= row_id) & row_mask).to(tl.uint32)

        batch_id = tl.sum(marked, axis=0) - 1
        margin = tl.load(q_ptr + batch_id)

        # Padded batch handling requires skipping tail end of row(s).

        # skipped = (batch_id >= batch_size)

        step_id = row_id - margin
        timesteps = lookahead + 1

    else:

        timesteps = lookahead + 1

        batch_id = row_id // timesteps
        step_id = row_id % timesteps

        margin = batch_id * timesteps

    if slotted:

        batch_id = tl.load(s_ptr + batch_id)

    return batch_id, step_id, margin, timesteps


@jit
def bitpacked_reduction_kernel(
    # Input Data Pointer(s).
    x_ptr,  # Logits -> [ L, V_d' / W ].
    z_ptr,  # Indicators -> [ B ].
    s_ptr,  # Slot Mapping -> [ B ].
    q_ptr,  # Cumulative Lengths -> [ B + 1 ].
    m_ptr,  # D2T Mapping -> [ V_d ].
    g_ptr,  # Grammar Bit-Mask -> [ B • (γ' + 1), V_t / 32 ].
    c_ptr,  # Context Counts -> [ B, V_t ].
    d_ptr,  # Decode Counts -> [ B, V_t ].
    t_ptr,  # Tokens -> [ T ].
    r_ptr,  # Repetition Penalties -> [ B ].
    f_ptr,  # Frequency Penalties -> [ B ].
    p_ptr,  # Presence Penalties -> [ B ].
    b_ptr,  # Logit Bias -> [ B, V_t ].
    e_ptr,  # Temperature -> [ B ].
    # Output Data Pointer(s).
    y_ptr,  # Scratch-Pad -> [ L, Z_v • M ].
    # Model Specific(s).
    vocab_size: tl.constexpr,  # Local Vocab Size.
    shard_offset: tl.constexpr,  # Local Shard Offset.
    lookahead: tl.constexpr,  # Maximum Lookahead -> γ.
    max_k: tl.constexpr,  # Maximum Top-K Value -> M.
    scale: tl.constexpr,  # Power-of-2 Integer Scale Factor.
    batch_size: tl.int32,  # Total Sequences -> B.
    # Conditional Flag(s).
    indirection: tl.constexpr,  # Vocab Indirection Flag.
    tokens: tl.constexpr,  # Drafted / Decoded Tokens Flag.
    slotted: tl.constexpr,  # Slot Indirection Flag.
    singular: tl.constexpr,  # Singular Timestep Flag.
    varlen: tl.constexpr,  # Variable-Length Lookahead Flag.
    enable_adaptive: tl.constexpr,  # Apply Adaptive Filtering & Reduction.
    enable_radix: tl.constexpr,  # Apply Radix Filtering & Reduction.
    enable_pdl: tl.constexpr,  # PDL Enablement Flag.
    # Stride(s).
    stride_x: tl.constexpr,  # Batch Stride of Logits.
    stride_g: tl.constexpr,  # Batch Stride of Grammar Bit-Mask.
    stride_c: tl.constexpr,  # Batch Stride of Counts (Target Vocab Size) -> V_t.
    stride_t: tl.constexpr,  # Batch Stride of Tokens.
    stride_y: tl.constexpr,  # Batch Stride of Scratch-Pad.
    # Kernel Specific(s).
    block_n: tl.constexpr,  # Block Size of Vocab Block(s).
    block_q: tl.constexpr,  # Block Size of Cumulative Length(s).
    block_t: tl.constexpr,  # Block Size of Token(s).
    exact: tl.constexpr,  # Exact fp32 Logits Flag (bf16 keys only pick candidates).
    value_dtype: tl.constexpr,  # Dtype the logit manipulators keep.
) -> None:
    """
    Note that the value of L and T in the above shapes differ as follows:

              •----------•-------------------•---------------------------•
              | Singular | Fixed-γ Multistep | Variable-Length Multistep |
        •-----•----------•-------------------•---------------------------•
        |  L  |     B    |    B • (γ + 1)    |       Σ_i^B (γ_i + 1)     |
        •-----•----------•-------------------•---------------------------•
        |  T  |   B • γ  |    B • (γ + 1)    |       Σ_i^B (γ_i + 1)     |
        •-----•----------•-------------------•---------------------------•

    In the case of the grammar bit-mask, the value of γ' is given as follows:

        • Singular / Fixed-γ Multistep -> γ

        • Variable-Length Multistep -> max_i { γ_i | 0 ≤ i < B }

    """
    # Define compile-time program-level constant(s).

    minima: tl.constexpr = -float("inf")

    # Load and extract batch and block id(s) from program id(s).

    row_id = tl.program_id(axis=0)
    col_id = tl.program_id(axis=1)

    # Compute the column indices and corresponding mask.

    shift = col_id * block_n
    steps = tl.arange(0, block_n)

    cols = shift + steps
    mask = cols < vocab_size

    # Compute the block offsets and slot id (if has slot indirection).

    offsets = (row_id * stride_x) + cols

    if enable_pdl:

        gdc_wait()

    batch_id, step_id, margin, timesteps = resolve_indices(
        q_ptr,
        s_ptr,
        row_id,
        lookahead,
        batch_size,
        slotted,
        singular,
        varlen,
        block_q,
    )

    # Load the logits and the bit-indicator(s).

    logits = tl.load(x_ptr + offsets, mask=mask, other=-float("inf"))
    indicator = tl.load(z_ptr + batch_id)

    logits = transform(
        logits,
        indicator,
        cols,
        mask,
        m_ptr,
        g_ptr,
        c_ptr,
        d_ptr,
        t_ptr,
        r_ptr,
        f_ptr,
        p_ptr,
        b_ptr,
        e_ptr,
        batch_id,
        step_id,
        margin,
        timesteps,
        shard_offset,
        lookahead,
        indirection,
        tokens,
        singular,
        stride_g,
        stride_c,
        stride_t,
        block_t,
        block_n,
        value_dtype,
    )

    if (indicator & 0x40) > 0:

        # Greedy (Top-1) selection.

        if exact:

            exact_maximum(
                y_ptr,
                row_id,
                col_id,
                tl.num_programs(axis=1),
                shift,
                logits,
                enable_pdl,
                stride_y,
            )

        else:

            greedy_reduction(
                y_ptr, row_id, col_id, logits, enable_pdl, stride_y, block_n
            )

    else:

        # Top-K selection by bf16 keys; exact logits are re-read at the candidates.

        top_k_reduction(
            y_ptr,
            row_id,
            col_id,
            logits.to(tl.bfloat16),
            steps,
            max_k,
            scale,
            enable_adaptive,
            enable_radix,
            enable_pdl,
            stride_y,
            block_n,
        )
