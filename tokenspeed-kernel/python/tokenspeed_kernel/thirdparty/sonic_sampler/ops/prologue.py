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
    exponential_log,
    fast_expf,
    gdc_launch_dependents,
    gdc_wait,
    ones_like,
    philox_gumbel,
    philox_key,
)
from tokenspeed_kernel.thirdparty.sonic_sampler.ops.encoding import bitpack_scalar
from tokenspeed_kernel.thirdparty.sonic_sampler.ops.topk.bitpack import top_k_reduction

__all__ = ["bitpacked_reduction_kernel", "whole_reduction_kernel"]


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
def head_logits(logits, mask, logit_scale, softcap):
    """A vocabulary shard's raw head logits scaled and softcapped as the logits processor stores them, in their dtype."""

    if logit_scale is not None or softcap is not None:

        dtype: tl.constexpr = logits.dtype
        values = logits.to(tl.float32)

        if logit_scale is not None:

            # Rounded to the logits' dtype, as the processor's in-place multiply.
            values = (values * logit_scale).to(dtype).to(tl.float32)

        if softcap is not None:

            # The processor's stable tanh form.
            values = values / softcap
            values = 2 * tl.sigmoid(2 * values) - 1
            values = values * softcap

        logits = tl.where(mask, values.to(dtype), -float("inf"))

    return logits


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
def exact_first(
    s_ptr,
    row_id,
    vocab_blocks: tl.constexpr,
    enable_pdl: tl.constexpr,
    stride_s: tl.constexpr,
    block_v: tl.constexpr,
):
    """The row's exact fp32 maximum and the first column holding it, from the blocks' ``exact_maximum`` slots."""

    span = tl.arange(0, block_v)
    span_mask = span < vocab_blocks
    offsets = (row_id * stride_s) + span

    if enable_pdl:

        gdc_wait()

    maxima = tl.load(s_ptr + offsets, mask=span_mask, other=0xFF800000)
    columns = tl.load(s_ptr + offsets + vocab_blocks, mask=span_mask, other=0)

    # Blocks run in vocabulary order, so the first maximal block holds the first maximal column.

    value, block = tl.max(
        maxima.to(tl.float32, bitcast=True), axis=0, return_indices=True
    )

    return value, tl.sum(tl.where(span == block, columns, 0), axis=0).to(tl.int32)


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

    _, column = exact_first(s_ptr, row_id, vocab_blocks, enable_pdl, stride_s, block_v)

    return column


@jit
def whole_row(indicator, whole: tl.constexpr, singular: tl.constexpr):
    """Whether a row draws from its whole vocabulary: no greedy, top-k, top-p or min-p bit and, in a verify, point-draw drafts (bit 0x800), whose residual drops only the draft."""

    if whole:

        plain = (indicator & 0x3C0) == 0

        if not singular:

            plain = plain & ((indicator & 0x800) != 0)

    else:

        # Never true (the bits are non-negative): no row takes the whole path.
        plain = (indicator & 0x3C0) < 0

    return plain


@jit
def block_gumbel(key, start, size: tl.constexpr):
    """``philox_gumbel`` at the ``size`` token ids from ``start`` (a multiple of four), one Philox round per four of them."""

    b0, b1, b2, b3 = tl.randint4x(key, start // 4 + tl.arange(0, size // 4))

    lanes = tl.arange(0, 4)[None, :]
    bits = tl.where(
        lanes == 0,
        b0[:, None],
        tl.where(
            lanes == 1, b1[:, None], tl.where(lanes == 2, b2[:, None], b3[:, None])
        ),
    )

    return tl.reshape(exponential_log(bits), (size,))


@jit
def whole_block(
    x_ptr,
    y_ptr,
    h_ptr,
    l_ptr,
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
    indicator,
    row_id,
    block_id,
    blocks,
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
    stride_y: tl.constexpr,
    block_n: tl.constexpr,
    block_t: tl.constexpr,
    value_dtype: tl.constexpr,
    logit_scale,
    softcap,
):
    """
    Stores a whole-vocabulary row's block as ``exact_maximum`` stores a
    maximum: its best value plus the Gumbel noise the candidates' draw adds
    (``philox_gumbel`` keyed by the slot's seed, offset and the row's
    position, at global token ids). The best over every block is then the
    Gumbel-max over the whole vocabulary, the token the candidates' draw
    takes whenever it lies among them.

    Note(s):

        • The block is read in tiles of at most 4096 columns, so its registers
          stay those of one tile whatever ``block_n``.

        • A verify row leaves its draft out of that best, which is then the
          residual draw past a rejected draft, and from slot ``2 • blocks``
          on stores the block's maximum value, its exp-sum below that
          maximum, and the draft's value and perturbed value (-inf in the
          blocks without it).

    """
    minima: tl.constexpr = -float("inf")
    tile: tl.constexpr = 4096 if block_n > 4096 else block_n

    key = philox_key(h_ptr, l_ptr, batch_id, step_id)

    if singular:

        draft = -1

    else:

        # The bonus row has no draft.
        draft = tl.load(t_ptr + margin + tl.minimum(step_id, lookahead - 1))
        draft = tl.where(step_id < lookahead, draft, -1)

    best = tl.full((), minima, tl.float32)
    column = tl.full((), 0, tl.int32)
    maximum = tl.full((), minima, tl.float32)
    total = tl.full((), 0.0, tl.float32)
    draft_value = tl.full((), minima, tl.float32)
    draft_score = tl.full((), minima, tl.float32)

    for start in tl.range(0, block_n, tile):

        cols = block_id * block_n + start + tl.arange(0, tile)
        mask = cols < vocab_size

        logits = tl.load(x_ptr + (row_id * stride_x) + cols, mask=mask, other=minima)
        logits = head_logits(logits, mask, logit_scale, softcap)
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
            tile,
            value_dtype,
        )

        ids = cols + shard_offset

        if indirection:

            ids = tl.load(m_ptr + ids, mask=mask, other=0)

        if indirection or (shard_offset % 4 != 0):

            noise = philox_gumbel(key, ids)

        else:

            noise = block_gumbel(key, block_id * block_n + start + shard_offset, tile)

        values = logits.to(tl.float32)
        scores = tl.where(mask, values - noise, minima)

        if not singular:

            drafting = ids == draft

            # Online exp-sum below the running maximum.
            peak = tl.maximum(maximum, tl.max(values, axis=0))
            part = tl.sum(
                tl.where(values > minima, fast_expf(values - peak), 0.0), axis=0
            )
            total = tl.where(
                peak > minima, total * fast_expf(maximum - peak) + part, 0.0
            )
            maximum = peak

            draft_value = tl.maximum(
                draft_value, tl.max(tl.where(drafting, values, minima), axis=0)
            )
            draft_score = tl.maximum(
                draft_score, tl.max(tl.where(drafting, scores, minima), axis=0)
            )
            scores = tl.where(drafting, minima, scores)

        score, index = tl.max(scores, axis=0, return_indices=True)

        # Strictly greater, so the first tile holding the best keeps it.
        better = score > best
        column = tl.where(better, start + index, column)
        best = tl.where(better, score, best)

    target = (row_id * stride_y) + block_id

    tl.store(y_ptr + target, best.to(tl.uint32, bitcast=True))
    tl.store(y_ptr + target + blocks, (block_id * block_n + column).to(tl.uint32))

    if not singular:

        tl.store(y_ptr + target + 2 * blocks, maximum.to(tl.uint32, bitcast=True))
        tl.store(y_ptr + target + 3 * blocks, total.to(tl.uint32, bitcast=True))
        tl.store(y_ptr + target + 4 * blocks, draft_value.to(tl.uint32, bitcast=True))
        tl.store(y_ptr + target + 5 * blocks, draft_score.to(tl.uint32, bitcast=True))


@jit
def whole_normalizer(
    s_ptr,
    row_id,
    vocab_blocks: tl.constexpr,
    stride_s: tl.constexpr,
    block_v: tl.constexpr,
):
    """A verify row's maximum value, its exp-sum below that maximum, and its draft's value and perturbed value, from the blocks' ``whole_block`` slots."""

    minima: tl.constexpr = -float("inf")

    span = tl.arange(0, block_v)
    span_mask = span < vocab_blocks
    offsets = (row_id * stride_s) + span + 2 * vocab_blocks

    maxima = tl.load(s_ptr + offsets, mask=span_mask, other=0xFF800000)
    totals = tl.load(s_ptr + offsets + vocab_blocks, mask=span_mask, other=0)
    drafts = tl.load(
        s_ptr + offsets + 2 * vocab_blocks, mask=span_mask, other=0xFF800000
    )
    scores = tl.load(
        s_ptr + offsets + 3 * vocab_blocks, mask=span_mask, other=0xFF800000
    )

    maxima = maxima.to(tl.float32, bitcast=True)
    totals = totals.to(tl.float32, bitcast=True)

    maximum = tl.max(maxima, axis=0)
    total = tl.sum(
        tl.where(maxima > minima, fast_expf(maxima - maximum) * totals, 0.0), axis=0
    )

    return (
        maximum,
        total,
        tl.max(drafts.to(tl.float32, bitcast=True), axis=0),
        tl.max(scores.to(tl.float32, bitcast=True), axis=0),
    )


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
    logit_scale,
    softcap,
):
    """The candidates' exact fp32 values, recomputed from the logits, sorted descending with the lower column first on ties."""

    index_dtype: tl.constexpr = indices.dtype

    mask = indices < vocab_size
    columns = tl.where(mask, indices, 0).to(tl.int32)

    logits = tl.load(
        x_ptr + (row_id * stride_x) + columns, mask=mask, other=-float("inf")
    )
    logits = head_logits(logits, mask, logit_scale, softcap)

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
    logit_scale=None,  # Logit Scale of a Vocabulary Shard's Raw Head Logits.
    softcap=None,  # Final Logit Softcap of a Vocabulary Shard's Raw Head Logits.
    whole: tl.constexpr = False,  # Leave Whole-Vocabulary Rows to ``whole_reduction_kernel``.
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

    # Load the bit-indicator(s), then the logits of the rows reduced here.

    indicator = tl.load(z_ptr + batch_id)

    if whole_row(indicator, whole, singular):

        # Reduced by ``whole_reduction_kernel``.

        pass

    else:

        logits = tl.load(x_ptr + offsets, mask=mask, other=-float("inf"))
        logits = head_logits(logits, mask, logit_scale, softcap)

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


@triton.jit(do_not_specialize=["batch_size", "group_blocks"])
def whole_reduction_kernel(
    # Input Data Pointer(s).
    x_ptr,  # Logits -> [ L, V_d' / W ].
    z_ptr,  # Indicators -> [ B ].
    s_ptr,  # Slot Mapping -> [ B ].
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
    h_ptr,  # Noise Seeds -> [ B ].
    l_ptr,  # Noise Offsets -> [ B ].
    # Output Data Pointer(s).
    y_ptr,  # Scratch-Pad -> [ L, Z_v • M ].
    # Model Specific(s).
    vocab_size: tl.constexpr,  # Local Vocab Size.
    shard_offset: tl.constexpr,  # Local Shard Offset.
    lookahead: tl.constexpr,  # Maximum Lookahead -> γ.
    batch_size: tl.int32,  # Total Sequences -> B.
    vocab_blocks: tl.constexpr,  # Vocab Blocks -> Z_v.
    group_blocks,  # Vocab Blocks per Program.
    # Conditional Flag(s).
    indirection: tl.constexpr,  # Vocab Indirection Flag.
    tokens: tl.constexpr,  # Drafted / Decoded Tokens Flag.
    slotted: tl.constexpr,  # Slot Indirection Flag.
    singular: tl.constexpr,  # Singular Timestep Flag.
    enable_pdl: tl.constexpr,  # PDL Enablement Flag.
    # Stride(s).
    stride_x: tl.constexpr,  # Batch Stride of Logits.
    stride_g: tl.constexpr,  # Batch Stride of Grammar Bit-Mask.
    stride_c: tl.constexpr,  # Batch Stride of Counts (Target Vocab Size) -> V_t.
    stride_t: tl.constexpr,  # Batch Stride of Tokens.
    stride_y: tl.constexpr,  # Batch Stride of Scratch-Pad.
    # Kernel Specific(s).
    block_n: tl.constexpr,  # Block Size of Vocab Block(s).
    block_t: tl.constexpr,  # Block Size of Token(s).
    value_dtype: tl.constexpr,  # Dtype the logit manipulators keep.
    logit_scale=None,  # Logit Scale of a Vocabulary Shard's Raw Head Logits.
    softcap=None,  # Final Logit Softcap of a Vocabulary Shard's Raw Head Logits.
) -> None:
    """
    The first stage of the rows ``bitpacked_reduction_kernel`` leaves out under
    its ``whole`` flag, which draw from their whole vocabulary: the
    ``whole_block`` slots of ``group_blocks`` consecutive vocab blocks per
    program, the same blocks the selection reads.

    Note(s):

        • A kernel of its own keeps the other rows' reduction compiled as it
          is without these rows, and its few programs per row keep the grid
          small for the batches that have none.

    """
    row_id = tl.program_id(axis=0)
    group_id = tl.program_id(axis=1)

    if enable_pdl:

        gdc_wait()

    batch_id, step_id, margin, timesteps = resolve_indices(
        None, s_ptr, row_id, lookahead, batch_size, slotted, singular, False, None
    )

    indicator = tl.load(z_ptr + batch_id)

    if whole_row(indicator, True, singular):

        first = group_id * group_blocks

        for block_id in tl.range(first, tl.minimum(first + group_blocks, vocab_blocks)):

            whole_block(
                x_ptr,
                y_ptr,
                h_ptr,
                l_ptr,
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
                indicator,
                row_id,
                block_id,
                vocab_blocks,
                batch_id,
                step_id,
                margin,
                timesteps,
                vocab_size,
                shard_offset,
                lookahead,
                indirection,
                tokens,
                singular,
                stride_x,
                stride_g,
                stride_c,
                stride_t,
                stride_y,
                block_n,
                block_t,
                value_dtype,
                logit_scale,
                softcap,
            )

    if enable_pdl:

        gdc_launch_dependents()
