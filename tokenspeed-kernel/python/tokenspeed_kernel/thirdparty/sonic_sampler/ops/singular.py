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
from tokenspeed_kernel._triton import libdevice  # noqa.

fast_expf = libdevice.fast_expf

from tokenspeed_kernel.thirdparty.sonic_sampler.ops.base import (
    cumulative_masking,
    gdc_launch_dependents,
    gdc_wait,
    log_softmax,
    philox_gumbel,
    philox_key,
    to_scalar,
)
from tokenspeed_kernel.thirdparty.sonic_sampler.ops.prologue import (
    exact_candidates,
    exact_greedy,
    nucleus_normalizer,
    nucleus_row,
    whole_row,
)
from tokenspeed_kernel.thirdparty.sonic_sampler.ops.shard import (
    merged_candidates,
    merged_first,
    merged_normalizer,
    merged_whole,
)
from tokenspeed_kernel.thirdparty.sonic_sampler.ops.topk.unpack import unpack_top_k


@jit
def sharded_maximum(
    s_ptr,
    row_id,
    max_k: tl.constexpr,
    vocab_blocks: tl.constexpr,
    world_size: tl.constexpr,
    shard_size: tl.constexpr,
    enable_pdl: tl.constexpr,
    stride_s: tl.constexpr,
    block_n: tl.constexpr,
    block_v: tl.constexpr,
):

    rows = tl.arange(0, block_v)[None, :]
    cols = tl.arange(0, world_size)[:, None] * (vocab_blocks * max_k)

    block_offsets = (row_id * stride_s) + (rows + cols)
    row_mask = rows < vocab_blocks

    if enable_pdl:

        gdc_wait()

    maximums = tl.load(s_ptr + block_offsets, mask=row_mask, other=0).reshape(
        (block_v * world_size,)
    )

    target, index = tl.max(maximums, axis=0, return_indices=True)

    shift = ((index // world_size) * shard_size) + ((index % world_size) * block_n)

    return target, shift


@jit
def unsharded_maximum(
    s_ptr,
    row_id,
    vocab_blocks: tl.constexpr,
    enable_pdl: tl.constexpr,
    stride_s: tl.constexpr,
    block_n: tl.constexpr,
    block_v: tl.constexpr,
):

    block_span = tl.arange(0, block_v)

    row_mask = block_span < vocab_blocks
    row_offsets = (row_id * stride_s) + block_span

    if enable_pdl:

        gdc_wait()

    targets = tl.load(s_ptr + row_offsets, mask=row_mask, other=0)
    target, index = tl.max(targets, axis=0, return_indices=True)

    return target, (index * block_n)


@jit
def greedy_maximum(
    s_ptr,
    row_id,
    max_k: tl.constexpr,
    vocab_blocks: tl.constexpr,
    world_size: tl.constexpr,
    shard_size: tl.constexpr,
    sharded: tl.constexpr,
    enable_pdl: tl.constexpr,
    stride_s: tl.constexpr,
    block_n: tl.constexpr,
    block_v: tl.constexpr,
):

    if sharded:

        return sharded_maximum(
            s_ptr,
            row_id,
            max_k,
            vocab_blocks,
            world_size,
            shard_size,
            enable_pdl,
            stride_s,
            block_n,
            block_v,
        )

    else:

        return unsharded_maximum(
            s_ptr, row_id, vocab_blocks, enable_pdl, stride_s, block_n, block_v
        )


@jit
def decode_update(
    d_ptr,
    batch_id,
    token_id,
    prefill: tl.constexpr,
    stride_d: tl.constexpr,
):

    if prefill:

        tl.store(d_ptr + (batch_id * stride_d) + token_id, 1)

    else:

        value = tl.load(d_ptr + (batch_id * stride_d) + token_id) + 1

        tl.store(d_ptr + (batch_id * stride_d) + token_id, value)


@jit
def cumulative_selection_kernel(
    # Input Data Pointer(s).
    s_ptr,  # Scratch-Pad -> [ L, Z_v • M ].
    z_ptr,  # Indicators -> [ B ].
    f_ptr,  # Slot Mapping -> [ B ].
    m_ptr,  # D2T Mapping -> [ V_d ].
    d_ptr,  # Decode Counts -> [ B, V_t ].
    g_ptr,  # Gumbel Noise -> [ B, V_t ].
    h_ptr,  # Noise Seeds -> [ B ].
    l_ptr,  # Noise Offsets -> [ B ].
    k_ptr,  # Top-K -> [ B ].
    p_ptr,  # Top-P -> [ B ].
    n_ptr,  # Min-P -> [ B ].
    r_ptr,  # Top-K Log-Probabilities -> [ B ].
    # Output Data Pointer(s).
    t_ptr,  # Selected Tokens -> [ B ].
    w_ptr,  # Log-Probabilities -> [ B ].
    y_ptr,  # Draft Probabilities -> [ B, V_t ].
    q_ptr,  # Top-K Log-Probabilities (Values) -> [ B, T ].
    j_ptr,  # Top-K Log-Probabilities (Indices) -> [ B, T ].
    # Model Specific(s).
    vocab_blocks: tl.constexpr,  # Local Vocab Blocks -> Z_v // W.
    world_size: tl.constexpr,  # World Size -> W.
    shard_size: tl.constexpr,  # Shard Size -> V_d // W.
    max_k: tl.constexpr,  # Maximum Top-K Value -> M.
    scale: tl.constexpr,  # Reciprocal of Power-of-2 Integer Scale Factor.
    total_rows: tl.constexpr,  # Total (Sharded) Logical Vocab Blocks -> W • Z_v'.
    total_cols: tl.constexpr,  # Total Logical Bit-Packed Columns -> W • Z_v' • M.
    # Conditional Flag(s).
    indirection: tl.constexpr,  # Vocab Indirection Flag.
    slotted: tl.constexpr,  # Slot Indirection Flag.
    sharded: tl.constexpr,  # Sharded Scratch-Pad Flag.
    prefill: tl.constexpr,  # Prefill (First Update) Flag.
    update_counts: tl.constexpr,  # Update Decode Counts Flag.
    is_adaptive: tl.constexpr,  # Adaptive Scaling Applied Flag.
    is_alternating: tl.constexpr,  # Alternating Ascending-Descending Tile(s) Flag.
    enable_pdl: tl.constexpr,  # PDL Enablement Flag.
    logprobs: tl.constexpr,  # Log-Probabilities Flag.
    probabilities: tl.constexpr,  # Draft Probabilities Flag.
    # Stride(s).
    stride_s: tl.constexpr,  # Batch Stride of Scratch-Pad.
    stride_r: tl.constexpr,  # Batch Stride of Sharded Scratch-Pad -> Z_v • M.
    stride_b: tl.constexpr,  # Logical Batch Stride of Sharded Scratch-Pad -> Z_v' • M.
    stride_t: tl.constexpr,  # Batch Stride of Selected Tokens.
    stride_w: tl.constexpr,  # Batch Stride of Log-Probabilities.
    stride_y: tl.constexpr,  # Batch Stride of Draft Probabilities.
    stride_d: tl.constexpr,  # Batch Stride of Decode Counts (Target Vocab Size) -> V_t.
    stride_g: tl.constexpr,  # Batch Stride of Gumbel Noise.
    stride_q: tl.constexpr,  # Batch Stride of Top-K Log-Probabilities (Values).
    # Kernel Specific(s).
    block_n: tl.constexpr,  # Block Size of Vocab Block(s).
    block_v: tl.constexpr,  # Next Power-of-2 Vocab Block(s) -> Z_v'.
    block_q: tl.constexpr,  # Next Power-of-2 Top-K Log-Probabilities Targets -> T'.
    # Exact fp32 Re-Read of the Candidates (exact only).
    x_ptr,  # Logits -> [ B, V_d' / W ].
    gr_ptr,  # Grammar Bit-Mask -> [ B, V_t / 32 ].
    cr_ptr,  # Context Counts -> [ B, V_t ].
    tr_ptr,  # Drafted Tokens -> [ B, γ ].
    rr_ptr,  # Repetition Penalties -> [ B ].
    fr_ptr,  # Frequency Penalties -> [ B ].
    pr_ptr,  # Presence Penalties -> [ B ].
    br_ptr,  # Logit Bias -> [ B, V_t ].
    er_ptr,  # Temperature -> [ B ].
    vocab_size: tl.constexpr,  # Local Vocab Size.
    shard_offset: tl.constexpr,  # Local Shard Offset.
    lookahead: tl.constexpr,  # Drafted Tokens per Row.
    tokens: tl.constexpr,  # Drafted Tokens Flag.
    stride_x: tl.constexpr,  # Batch Stride of Logits.
    stride_gr: tl.constexpr,  # Batch Stride of Grammar Bit-Mask.
    stride_tr: tl.constexpr,  # Batch Stride of Drafted Tokens.
    block_t: tl.constexpr,  # Block Size of Drafted Tokens.
    exact: tl.constexpr,  # Exact fp32 Logits Flag.
    # Every Rank's Candidates of Vocabulary-Sharded Logits (in place of the scratch-pad).
    cand_ptr=None,  # Gathered Candidate Pairs -> [ W, L, 2 • M ] fp32: values, then token ids.
    ranks: tl.constexpr = None,  # Ranks -> W.
    stride_mr=None,  # Rank Stride of Gathered Candidate Pairs.
    stride_mb: tl.constexpr = None,  # Batch Stride of Gathered Candidate Pairs.
    whole: tl.constexpr = False,  # Whole-Vocabulary Rows Flag (``whole_block``).
) -> None:

    # Extract the batch id from program id and load the indicator.

    row_id = tl.program_id(axis=0)
    batch_id = row_id

    if slotted:

        batch_id = tl.load(f_ptr + batch_id)

    indicator = tl.load(z_ptr + batch_id)

    if whole_row(indicator, whole, True):

        # Whole-vocabulary Gumbel-max: the best of the blocks' perturbed values.

        if cand_ptr is not None:

            _, token_id, _, _, _, _ = merged_whole(
                cand_ptr, row_id, ranks, stride_mr, stride_mb, enable_pdl, max_k
            )

        else:

            token_id = exact_greedy(
                s_ptr, row_id, vocab_blocks, enable_pdl, stride_s, block_v
            )

        if indirection:

            token_id = tl.load(m_ptr + token_id)

        if update_counts:

            decode_update(d_ptr, batch_id, token_id, prefill, stride_d)

        if enable_pdl:

            gdc_launch_dependents()

        tl.store(t_ptr + (row_id * stride_t), token_id)

    elif (indicator & 0x40) > 0:

        # Load, reduce, and select greedy i.e. top-1 maximum.

        if cand_ptr is not None:

            selection = merged_first(
                cand_ptr, row_id, ranks, stride_mr, stride_mb, enable_pdl, max_k
            )

        elif exact:

            selection = exact_greedy(
                s_ptr, row_id, vocab_blocks, enable_pdl, stride_s, block_v
            )

        else:

            target, shift = greedy_maximum(
                s_ptr,
                row_id,
                max_k,
                vocab_blocks,
                world_size,
                shard_size,
                sharded,
                enable_pdl,
                stride_s,
                block_n,
                block_v,
            )

            # Unpack the selected token id from the target index.

            selection = shift + ((block_n - 1) ^ (target & 0xFFFF))

        if indirection:

            # Re-map the selected token id to the target vocabulary.

            selection = tl.load(m_ptr + selection)

        if update_counts:

            # Increment the decode count for the selected token.

            decode_update(d_ptr, batch_id, selection, prefill, stride_d)

        if enable_pdl:

            gdc_launch_dependents()

        tl.store(t_ptr + (row_id * stride_t), selection)

    else:

        # Load, reduce, and unpack top-k stochastic subset.

        if cand_ptr is not None:

            values, indices = merged_candidates(
                cand_ptr, row_id, ranks, stride_mr, stride_mb, exact, enable_pdl, max_k
            )

        else:

            values, indices = unpack_top_k(
                s_ptr,
                row_id,
                vocab_blocks,
                world_size,
                shard_size,
                max_k,
                scale,
                total_rows,
                total_cols,
                sharded,
                is_adaptive,
                is_alternating,
                enable_pdl,
                stride_r,
                stride_b,
                stride_s,
                block_n,
                block_v,
            )

        if exact and cand_ptr is None:

            # Re-read the candidates' exact values: bf16 keys only chose them.

            values, indices = exact_candidates(
                x_ptr,
                row_id,
                indices,
                indicator,
                m_ptr,
                gr_ptr,
                cr_ptr,
                d_ptr,
                tr_ptr,
                rr_ptr,
                fr_ptr,
                pr_ptr,
                br_ptr,
                er_ptr,
                batch_id,
                0,
                row_id,
                1,
                vocab_size,
                shard_offset,
                lookahead,
                indirection,
                tokens,
                True,
                stride_x,
                stride_gr,
                stride_d,
                stride_tr,
                block_t,
                max_k,
                None,
                None,
            )

        if (indicator & 0x380) > 0:

            # Prepare for and apply top-k, top-p, and / or min-p subset masking.

            maximum = tl.gather(values, tl.full((1,), 0, dtype=tl.int32), axis=0)
            offsets = tl.arange(0, max_k)

            # A nucleus over the whole softmax takes the row's exp-sum; zero keeps it over the candidates.
            total = tl.zeros((1,), dtype=tl.float32)

            if nucleus_row(indicator):

                if cand_ptr is not None:

                    total += merged_normalizer(
                        cand_ptr, row_id, ranks, stride_mr, stride_mb, 2, max_k, maximum
                    )

                else:

                    peak, mass = nucleus_normalizer(
                        s_ptr, row_id, vocab_blocks, max_k, stride_s, block_v
                    )
                    total += mass * fast_expf(peak - maximum)

            values = cumulative_masking(
                k_ptr,
                p_ptr,
                n_ptr,
                batch_id,
                indicator,
                values,
                maximum,
                offsets,
                max_k,
                total,
            )

        if indirection:

            # Re-map the selected token indices to the target vocabulary.

            indices = tl.load(m_ptr + indices)

        if g_ptr is not None:

            noise = tl.load(g_ptr + (batch_id * stride_g) + indices)

        else:

            # Draw the gumbel noise in-kernel at the unpacked subset.

            key = philox_key(h_ptr, l_ptr, batch_id, 0)
            noise = philox_gumbel(key, indices)

        selected = tl.argmax(values - noise, axis=0, keep_dims=True)
        token_id = to_scalar(tl.gather(indices, selected, axis=0))

        if update_counts:

            # Increment the decode count for the selected token.

            decode_update(d_ptr, batch_id, token_id, prefill, stride_d)

        if logprobs or probabilities:

            maximum = tl.gather(values, tl.full((1,), 0, dtype=tl.int32), axis=0)
            log_scores = log_softmax(values, maximum)

            if enable_pdl:

                gdc_launch_dependents()

            if logprobs:

                log_score = to_scalar(tl.gather(log_scores, selected, axis=0))

                tl.store(w_ptr + (row_id * stride_w), log_score)

                if (indicator & 0x400) > 0:

                    logprobs_ids = tl.arange(0, block_q)
                    logprobs_mask = logprobs_ids < tl.load(r_ptr + batch_id)

                    top_logprobs = tl.gather(log_scores, logprobs_ids, axis=0)
                    top_tokens = tl.gather(indices, logprobs_ids, axis=0)

                    logprobs_offsets = (row_id * stride_q) + logprobs_ids

                    tl.store(q_ptr + logprobs_offsets, top_logprobs, mask=logprobs_mask)
                    tl.store(j_ptr + logprobs_offsets, top_tokens, mask=logprobs_mask)

            if probabilities:

                probs_offsets = (batch_id * stride_y) + indices

                tl.store(y_ptr + probs_offsets, fast_expf(log_scores))

        elif enable_pdl:

            gdc_launch_dependents()

        tl.store(t_ptr + (row_id * stride_t), token_id)
