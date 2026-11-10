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

jit = triton.jit
from tokenspeed_kernel._triton import libdevice  # noqa.

fast_expf = libdevice.fast_expf

from tokenspeed_kernel.thirdparty.sonic_sampler.ops.base import (
    cumulative_masking,
    gdc_launch_dependents,
    gdc_wait,
    log_softmax,
)
from tokenspeed_kernel.thirdparty.sonic_sampler.ops.prologue import (
    exact_candidates,
    exact_greedy,
)
from tokenspeed_kernel.thirdparty.sonic_sampler.ops.shard import (
    merged_candidates,
    merged_candidates_with_drafts,
    merged_first,
)
from tokenspeed_kernel.thirdparty.sonic_sampler.ops.singular import greedy_maximum
from tokenspeed_kernel.thirdparty.sonic_sampler.ops.topk.unpack import unpack_top_k


@jit
def resolve_indices(
    q_ptr,
    f_ptr,
    row_id,
    lookahead: tl.constexpr,
    batch_size: tl.int32,
    slotted: tl.constexpr,
    varlen: tl.constexpr,
    block_q: tl.constexpr,
):
    """
    Compute the batch id, step id, and margin given the row id of the current tile.

    """
    if varlen:

        row_ids = tl.arange(0, block_q)
        row_mask = row_ids < (batch_size + 1)

        lengths = tl.load(q_ptr + row_ids, mask=row_mask, other=0, cache_modifier=".cv")
        marked = ((lengths <= row_id) & row_mask).to(tl.uint32)

        batch_id = tl.sum(marked, axis=0) - 1
        margin = tl.load(q_ptr + batch_id)

        step_id = row_id - margin

    else:

        timesteps = lookahead + 1

        batch_id = row_id // timesteps
        step_id = row_id % timesteps

        margin = batch_id * timesteps

    if slotted:

        batch_id = tl.load(f_ptr + batch_id)

    return batch_id, margin, step_id


@jit
def cumulative_unpack_kernel(
    # Input Data Pointer(s).
    s_ptr,  # Scratch-Pad -> [ L, Z_v • M ].
    z_ptr,  # Indicators -> [ B ].
    f_ptr,  # Slot Mapping -> [ B ].
    q_ptr,  # Cumulative Lengths -> [ B + 1 ].
    k_ptr,  # Top-K -> [ B ].
    p_ptr,  # Top-P -> [ B ].
    n_ptr,  # Min-P -> [ B ].
    # Output Data Pointer(s).
    v_ptr,  # Selected Values -> [ B • (γ + 1), M ].
    j_ptr,  # Selected Indices -> [ B • (γ + 1), M ].
    # Model Specific(s).
    vocab_blocks: tl.constexpr,  # Local Vocab Blocks -> Z_v // W.
    world_size: tl.constexpr,  # World Size -> W.
    shard_size: tl.constexpr,  # Shard Size -> V_d // W.
    lookahead: tl.constexpr,  # Lookahead -> γ.
    max_k: tl.constexpr,  # Maximum Top-K Value -> M.
    scale: tl.constexpr,  # Reciprocal of  Power-of-2 Integer Scale Factor.
    batch_size: tl.int32,  # Total Sequences -> B.
    total_rows: tl.constexpr,  # Total (Sharded) Logical Vocab Blocks -> W • Z_v'.
    total_cols: tl.constexpr,  # Total Logical Bit-Packed Columns -> W • Z_v' • M.
    # Conditional Flag(s).
    sharded: tl.constexpr,  # Sharded Scratch-Pad Flag.
    slotted: tl.constexpr,  # Slot Indirection Flag.
    varlen: tl.constexpr,  # Variable-Length Lookahead Flag.
    is_adaptive: tl.constexpr,  # Adaptive Scaling Applied Flag.
    is_alternating: tl.constexpr,  # Alternating Ascending-Descending Tile(s) Flag.
    enable_pdl: tl.constexpr,  # PDL Enablement Flag.
    # Stride(s).
    stride_s: tl.constexpr,  # Batch Stride of Scratch-Pad.
    stride_r: tl.constexpr,  # Batch Stride of Sharded Scratch-Pad -> Z_v • M.
    stride_b: tl.constexpr,  # Logical Batch Stride of Sharded Scratch-Pad -> Z_v' • M.
    # Kernel Specific(s).
    block_n: tl.constexpr,  # Block Size of Vocab Block(s).
    block_v: tl.constexpr,  # Next Power-of-2 Vocab Block(s) -> Z_v'.
    block_q: tl.constexpr,  # Block Size of Cumulative Length(s).
    # Exact fp32 Re-Read of the Candidates (exact only).
    x_ptr,  # Logits -> [ L, V_d' / W ].
    gr_ptr,  # Grammar Bit-Mask -> [ B • (γ' + 1), V_t / 32 ].
    cr_ptr,  # Context Counts -> [ B, V_t ].
    dr_ptr,  # Decode Counts -> [ B, V_t ].
    tr_ptr,  # Drafted Tokens -> [ L ].
    rr_ptr,  # Repetition Penalties -> [ B ].
    fr_ptr,  # Frequency Penalties -> [ B ].
    pr_ptr,  # Presence Penalties -> [ B ].
    br_ptr,  # Logit Bias -> [ B, V_t ].
    er_ptr,  # Temperature -> [ B ].
    vocab_size: tl.constexpr,  # Local Vocab Size.
    shard_offset: tl.constexpr,  # Local Shard Offset.
    stride_x: tl.constexpr,  # Batch Stride of Logits.
    stride_gr: tl.constexpr,  # Batch Stride of Grammar Bit-Mask.
    stride_cr: tl.constexpr,  # Batch Stride of Counts and Bias.
    block_t: tl.constexpr,  # Block Size of Drafted Tokens.
    exact: tl.constexpr,  # Exact fp32 Logits Flag.
    # Every Rank's Candidates of Vocabulary-Sharded Logits (in place of the scratch-pad).
    cand_ptr=None,  # Gathered Candidate Pairs -> [ W, L, 2 • M ] fp32: values, then token ids.
    ranks: tl.constexpr = None,  # Ranks -> W.
    stride_mr=None,  # Rank Stride of Gathered Candidate Pairs.
    stride_mb: tl.constexpr = None,  # Batch Stride of Gathered Candidate Pairs.
    # Draft Probabilities the Pairs Carry as a Third Block (with ``cand_ptr``).
    m_ptr=None,  # Candidate Draft Probabilities -> [ B • (γ + 1), M ].
) -> None:

    # Extract the batch id from program id and load the indicator.

    row_id = tl.program_id(axis=0)

    if enable_pdl:

        gdc_wait()

    batch_id, margin, step_id = resolve_indices(
        q_ptr,
        f_ptr,
        row_id,
        lookahead,
        batch_size,
        slotted,
        varlen,
        block_q,
    )

    indicator = tl.load(z_ptr + batch_id)

    if (indicator & 0x40) > 0:

        # Load, reduce, and unpack greedy i.e. top-1 maximum.

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

        if enable_pdl:

            gdc_launch_dependents()

        # Store at margin boundary for later contiguous read-out.

        tl.store(j_ptr + (margin * max_k) + step_id, selection)

    else:

        # Load, reduce, and unpack top-k stochastic subset.

        if m_ptr is not None:

            values, indices, proposals = merged_candidates_with_drafts(
                cand_ptr, row_id, ranks, stride_mr, stride_mb, exact, enable_pdl, max_k
            )

        elif cand_ptr is not None:

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
                None,
                gr_ptr,
                cr_ptr,
                dr_ptr,
                tr_ptr,
                rr_ptr,
                fr_ptr,
                pr_ptr,
                br_ptr,
                er_ptr,
                batch_id,
                step_id,
                margin,
                lookahead + 1,
                vocab_size,
                shard_offset,
                lookahead,
                False,
                True,
                False,
                stride_x,
                stride_gr,
                stride_cr,
                None,
                block_t,
                max_k,
                None,
                None,
            )

        # Prepare for later softmax computation and write-out offset(s).

        maximum = tl.gather(values, tl.full((1,), 0, dtype=tl.int32), axis=0)
        offsets = tl.arange(0, max_k)

        if (indicator & 0x380) > 0:

            # Apply top-k, top-p, and / or min-p subset masking.

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
            )

        # Compute softmax from values with potentially masked subset.

        values = fast_expf(log_softmax(values, maximum))

        if enable_pdl:

            gdc_launch_dependents()

        # Write-out the target values and top-k indices.

        positions = (row_id * max_k) + offsets

        tl.store(v_ptr + positions, values)
        tl.store(j_ptr + positions, indices)

        if m_ptr is not None:

            tl.store(m_ptr + positions, proposals)
