# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2024, Tri Dao
#
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


"""Triton kernels for causal depthwise conv1d over a slot-indexed conv-state pool.

``_causal_conv1d_fwd_kernel`` runs varlen prefill chunks,
``_causal_conv1d_update_kernel`` decode and chain verify steps, and
``_causal_conv1d_tree_update_kernel`` draft-tree steps. The registered leaves
that launch them live in ``tokenspeed_kernel.ops.conv.triton``.
"""

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import pdl_enabled

PAD_SLOT_ID = -1


@triton.jit(
    do_not_specialize=["seqlen"],
    do_not_specialize_on_alignment=["token_chunk_offset_ptr"],
)
def _causal_conv1d_fwd_kernel(  # continuous batching
    # Pointers to matrices
    x_ptr,  # (dim, cu_seqlen) holding `batch` of actual sequences + padded sequences
    w_ptr,  # (dim, width)
    bias_ptr,
    initial_states_ptr,  # conv_states_ptr
    cache_indices_ptr,  # conv_state_indices_ptr
    has_initial_states_ptr,
    query_start_loc_ptr,
    batch_ptr,
    token_chunk_offset_ptr,
    o_ptr,  # (dim, seqlen) - actually pointing to x_ptr
    # Matrix dimensions
    dim: tl.constexpr,
    seqlen: tl.int32,  # cu_seqlen
    num_cache_lines: tl.constexpr,
    # Strides
    stride_x_seq: tl.constexpr,  # stride to get to next sequence,
    stride_x_dim: tl.constexpr,  # stride to get to next feature-value,
    stride_x_token: tl.constexpr,  # stride to get to next token (same feature-index, same sequence-index)
    stride_w_dim: tl.constexpr,  # stride to get to next dim-axis value
    stride_w_width: tl.constexpr,  # stride to get to next width-axis value
    stride_istate_seq: tl.constexpr,
    stride_istate_dim: tl.constexpr,
    stride_istate_token: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    HAS_INITIAL_STATES: tl.constexpr,
    HAS_CACHE: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        # Release successor setup; its wait still guards all dependent reads.
        tl.extra.cuda.gdc_launch_dependents()
    conv_states_ptr = initial_states_ptr
    conv_state_indices_ptr = cache_indices_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = (
        KERNEL_WIDTH - 1
    )  # can be passed via argument if it's not the same as this value

    # one program handles one chunk in a single sequence
    # rather than mixing sequences - to make updating initial_states across sequences efficiently

    # single-sequence id
    idx_seq = tl.load(batch_ptr + tl.program_id(0))
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))

    # BLOCK_N elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    # Token offsets times a projection-wide row stride outgrow int32.
    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq).to(tl.int64)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    # find the actual sequence length
    seqlen = sequence_end_index - sequence_start_index

    token_offset = BLOCK_M * chunk_offset.to(tl.int64)
    segment_len = min(BLOCK_M, seqlen - token_offset)

    # base of the sequence
    x_base = (
        x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim
    )  # [BLOCK_N,]

    if IS_CONTINUOUS_BATCHING:
        # cache_idx
        conv_state_batch_coord = tl.load(conv_state_indices_ptr + idx_seq).to(tl.int64)
    else:
        # cache_idx
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:
        if conv_state_batch_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return
    conv_states_base = (
        conv_states_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )  # [BLOCK_N,]

    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]

    # Does 2 things:
    # 1. READ prior-block init-state data - [done by every Triton programs]
    # 2. update conv_state with new data [only by the Triton program handles chunk_offset=0]
    if chunk_offset == 0:
        # read from conv_states
        load_init_state = False
        if HAS_INITIAL_STATES:  # the new HAS_INITIAL_STATES
            load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init_state:
            # load from conv_states
            prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
            mask_w = idx_feats < dim
            if KERNEL_WIDTH == 2:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 3:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 4:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 5:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col3 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 3 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
        else:
            # prior-tokens are zeros
            if KERNEL_WIDTH >= 2:  # STRATEGY1
                # first chunk and does not have prior-token, so just set to 0
                col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 3:  # STRATEGY1
                col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 4:  # STRATEGY1
                col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 5:  # STRATEGY1
                col3 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

        # STEP 2:
        # here prepare data for updating conv_state
        if (
            state_len <= seqlen
        ):  # SMALL_CACHE=True (only move part of 'x' into conv_state cache)
            # just read from 'x'
            # copy 'x' data to conv_state
            # load only 'x' data (and set 0 before 'x' if seqlen < state_len)
            idx_tokens_last = (seqlen - state_len) + tl.arange(
                0, NP2_STATELEN
            )  # [BLOCK_M]
            x_ptrs = (
                x_ptr
                + ((sequence_start_index + idx_tokens_last) * stride_x_token)[:, None]
                + (idx_feats * stride_x_dim)[None, :]
            )  # [BLOCK_M,BLOCK_N,]
            mask_x = (
                (idx_tokens_last >= 0)[:, None]
                & (idx_tokens_last < seqlen)[:, None]
                & (idx_feats < dim)[None, :]
            )  # token-index  # token-index  # feature-index
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]
            conv_states_ptrs_target = (
                conv_states_base[None, :]
                + (idx_tokens_conv * stride_conv_state_tok)[:, None]
            )

            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()  #   use this due to bug in Triton compiler
            tl.store(conv_states_ptrs_target, new_conv_state, mask)

        else:
            if load_init_state:
                # update conv_state by shifting left, i.e. take last few cols from conv_state + cols from 'x'
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                conv_states_ptrs_source = (
                    conv_states_ptr
                    + (conv_state_batch_coord * stride_conv_state_seq)
                    + (idx_feats * stride_conv_state_dim)[None, :]
                    + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (
                    (conv_state_batch_coord < num_cache_lines)
                    & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)

                VAL = state_len - seqlen

                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )  # [BLOCK_M, BLOCK_N]

                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )  # token-index  # token-index  # feature-index
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)

                tl.debug_barrier()  # need this due to the bug in tl.where not enforcing this when data is the result of another tl.load
                new_conv_state = tl.where(
                    mask, conv_state, loaded_x
                )  # BUG in 'tl.where'  which requires a barrier before this
                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)
            else:  # load_init_state == False
                # update conv_state by shifting left, BUT
                # set cols prior to 'x' as zeros + cols from 'x'
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                VAL = state_len - seqlen

                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )  # [BLOCK_M, BLOCK_N]

                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )  # token-index  # token-index  # feature-index
                new_conv_state = tl.load(x_ptrs, mask_x, 0.0)

                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)

    else:  # chunk_offset > 0
        # read prior-token data from `x`
        load_init_state = True
        prior_tokens = x_base + (token_offset - 1) * stride_x_token
        mask_w = idx_feats < dim
        if KERNEL_WIDTH == 2:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 3:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 4:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 5:
            # ruff: noqa: F841
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col3 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 3 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")

    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
            tl.float32
        )  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base_1d = x_base + token_offset * stride_x_token  # starting of chunk

    # PRE-LOAD WEIGHTS
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    mask_x_1d = idx_feats < dim
    for idx_token in range(segment_len):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):

            if KERNEL_WIDTH == 2:
                if j == 1:  # KERNEL_WIDTH-1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w  # [BLOCK_N]

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < segment_len) & (
            idx_feats < dim
        )  # token-index  # feature-index
        o_ptrs = (
            o_ptr
            + (sequence_start_index + token_offset + idx_token) * stride_o_token
            + (idx_feats * stride_o_dim)
        )

        tl.store(o_ptrs, acc, mask=mask_1d)


@triton.jit(do_not_specialize=["batch"])
def _causal_conv1d_update_kernel(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    cache_seqlens_ptr,  # circular buffer
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    intermediate_conv_window_ptr,
    output_state_indices_ptr,
    o_ptr,  # (batch, dim, seqlen)
    replay_ptr,  # (batch * seqlen, replay width)
    replay_a_ptr,  # (batch * seqlen, gate heads)
    replay_b_ptr,  # (batch * seqlen, gate heads)
    # Matrix dimensions
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,
    # Strides
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_inter_seq: tl.constexpr,
    stride_inter_step: tl.constexpr,
    stride_inter_dim: tl.constexpr,
    stride_inter_win: tl.constexpr,
    stride_output_state_indices_seq: tl.constexpr,
    stride_output_state_indices_step: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    stride_replay_row: tl.constexpr,
    stride_replay_a_row: tl.constexpr,
    stride_replay_b_row: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SAVE_INTERMEDIATE: tl.constexpr,
    HAS_OUTPUT_STATE_INDICES: tl.constexpr,
    HAS_REPLAY_PAYLOAD: tl.constexpr,
    REPLAY_FIRST_CHANNEL: tl.constexpr,
    GATE_HEADS: tl.constexpr,
    GATE_BLOCK: tl.constexpr,
    NP2_SEQLEN: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    # ruff: noqa: E501
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        # Release successor setup; its wait still guards all dependent reads.
        tl.extra.cuda.gdc_launch_dependents()
    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    if HAS_REPLAY_PAYLOAD:
        # Channel block 0 copies the gates: payload row = [REPLAY_FIRST_CHANNEL on | a | b].
        if tl.program_id(1) == 0:
            gate_tokens = tl.arange(0, NP2_SEQLEN)
            gate_heads = tl.arange(0, GATE_BLOCK)
            gate_rows = (idx_seq * seqlen + gate_tokens).to(tl.int64)
            gate_mask = (gate_tokens < seqlen)[:, None] & (gate_heads < GATE_HEADS)[
                None, :
            ]
            gate_dst = (
                replay_ptr
                + gate_rows[:, None] * stride_replay_row
                + (dim - REPLAY_FIRST_CHANNEL)
                + gate_heads[None, :]
            )
            gate_a = tl.load(
                replay_a_ptr
                + gate_rows[:, None] * stride_replay_a_row
                + gate_heads[None, :],
                mask=gate_mask,
            )
            gate_b = tl.load(
                replay_b_ptr
                + gate_rows[:, None] * stride_replay_b_row
                + gate_heads[None, :],
                mask=gate_mask,
            )
            tl.store(gate_dst, gate_a, mask=gate_mask)
            tl.store(gate_dst + GATE_HEADS, gate_b, mask=gate_mask)

    # [BLOCK_N,] elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if IS_CONTINUOUS_BATCHING:
        conv_state_batch_coord = tl.load(
            conv_state_indices_ptr + idx_seq * stride_state_indices
        ).to(tl.int64)
    else:
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:
        if conv_state_batch_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return

    if IS_SPEC_DECODING:
        # The rolling of conv state:
        #
        # Before forward, the conv_state is:
        # [history1, history2, ..., historyM].
        #
        # After forward, the conv_state becomes:
        # [history2, ..., historyM, draft1, draft2, ..., draftN].
        #
        # After acceptance, it becomes:
        #
        # - accept 1 tokens: [history2, ..., historyM, draft1]
        # - accept 2 tokens: [history3, ..., historyM, draft1, draft2]
        # - and so on.
        conv_state_token_offset = tl.load(num_accepted_tokens_ptr + idx_seq) - 1
    else:
        conv_state_token_offset = 0

    # STEP 1: READ init_state data
    conv_states_base = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    mask_w = idx_feats < dim

    prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok
    if KERNEL_WIDTH >= 2:
        conv_states_ptrs = prior_tokens  # [BLOCK_N]
        col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 3:
        conv_states_ptrs = prior_tokens + 1 * stride_conv_state_tok  # [BLOCK_N]
        col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 4:
        conv_states_ptrs = prior_tokens + 2 * stride_conv_state_tok  # [BLOCK_N]
        col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH == 5:
        conv_states_ptrs = prior_tokens + 3 * stride_conv_state_tok  # [BLOCK_N]
        col3 = tl.load(conv_states_ptrs, mask_w, 0.0)

    x_base = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim)  # [BLOCK_N]

    if not SAVE_INTERMEDIATE and not HAS_OUTPUT_STATE_INDICES:
        # STEP 2: update conv_state in place.  Speculative verify uses
        # SAVE_INTERMEDIATE and scatters the accepted intermediate window after
        # verification, so writing the real conv_state here is both wrong and
        # can address beyond its width-1 storage for multi-token verify.
        idx_tokens = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

        # The conv_state updates works in a sliding window manner,
        # at each forward pass, the tokens are shift by 1, so we
        # load since idx_tokens + 1.
        conv_state_ptrs_source = (
            conv_state_ptr
            + (conv_state_batch_coord * stride_conv_state_seq)
            + conv_state_token_offset * stride_conv_state_tok
            + (idx_feats * stride_conv_state_dim)[None, :]
            + ((idx_tokens + 1) * stride_conv_state_tok)[:, None]
        )  # [BLOCK_M, BLOCK_N]
        mask = (
            (conv_state_batch_coord < num_cache_lines)
            & ((idx_tokens + seqlen) < state_len)[:, None]
            & (idx_feats < dim)[None, :]
        )
        conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

        VAL = state_len - seqlen
        x_ptrs = (
            x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
        )  # [BLOCK_M, BLOCK_N]

        mask_x = (
            (idx_tokens - VAL >= 0)[:, None]
            & (idx_tokens - VAL < seqlen)[:, None]
            & (idx_feats < dim)[None, :]
        )  # token-index  # token-index  # feature-index
        loaded_x = tl.load(x_ptrs, mask_x, 0.0)
        tl.debug_barrier()

        new_conv_state = tl.where(mask, conv_state, loaded_x)

        conv_state_base = (
            conv_state_ptr
            + (conv_state_batch_coord * stride_conv_state_seq)
            + (idx_feats * stride_conv_state_dim)
        )  # [BLOCK_N,]
        conv_state_ptrs_target = (
            conv_state_base + (idx_tokens * stride_conv_state_tok)[:, None]
        )  # [BLOCK_M, BLOCK_N]
        mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
        tl.store(conv_state_ptrs_target, new_conv_state, mask)

    # STEP 3: init accumulator
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
            tl.float32
        )  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # STEP 4:
    # PRE-LOAD WEIGHTS
    # first kernel column, configured for weights to handle BLOCK_N features in range
    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)

    x_base_1d = x_base  # starting of chunk [BLOCK_N]
    mask_x_1d = idx_feats < dim
    # Load every token's input before any output store, which may alias x and so pins later loads.
    x_tokens = ()
    for idx_token in tl.static_range(seqlen):
        x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
        x_tokens = x_tokens + (tl.load(x_ptrs_1d, mask=mask_x_1d),)

    # STEP 5: compute each token
    for idx_token in tl.static_range(seqlen):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:  # KERNEL_WIDTH-1:
                    matrix_w = w_col1
                    matrix_x = x_tokens[idx_token]
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = x_tokens[idx_token]
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = x_tokens[idx_token]

            acc += matrix_x * matrix_w  # [BLOCK_N]

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < seqlen) & (
            idx_feats < dim
        )  # token-index  # feature-index
        o_ptrs = (
            o_ptr
            + (idx_seq) * stride_o_seq
            + idx_token * stride_o_token
            + (idx_feats * stride_o_dim)
        )

        tl.store(o_ptrs, acc, mask=mask_1d)
        if HAS_REPLAY_PAYLOAD:
            # The same rounded output, at the token's payload row.
            replay_ptrs = (
                replay_ptr
                + (idx_seq * seqlen + idx_token).to(tl.int64) * stride_replay_row
                + (idx_feats - REPLAY_FIRST_CHANNEL)
            )
            tl.store(
                replay_ptrs, acc, mask=mask_1d & (idx_feats >= REPLAY_FIRST_CHANNEL)
            )

        if SAVE_INTERMEDIATE:
            # Save the window state after consuming this token
            # Layout: [batch_position, step, dim, win(K-1)]
            # Use idx_seq (batch position) instead of conv_state_batch_coord
            # (pool index) so the intermediate cache can be sized to
            # max_batch_size rather than the full mamba pool.
            base_ptr = (
                intermediate_conv_window_ptr
                + idx_seq * stride_inter_seq
                + idx_token * stride_inter_step
                + idx_feats * stride_inter_dim
            )
            if KERNEL_WIDTH >= 2:
                tl.store(base_ptr + 0 * stride_inter_win, col0, mask=mask_w)
            if KERNEL_WIDTH >= 3:
                tl.store(base_ptr + 1 * stride_inter_win, col1, mask=mask_w)
            if KERNEL_WIDTH >= 4:
                tl.store(base_ptr + 2 * stride_inter_win, col2, mask=mask_w)
        if HAS_OUTPUT_STATE_INDICES:
            output_state_idx = tl.load(
                output_state_indices_ptr
                + idx_seq * stride_output_state_indices_seq
                + idx_token * stride_output_state_indices_step
            ).to(tl.int64)
            if output_state_idx >= 0:
                output_base = (
                    conv_state_ptr
                    + output_state_idx * stride_conv_state_seq
                    + idx_feats * stride_conv_state_dim
                )
                if KERNEL_WIDTH >= 2:
                    tl.store(output_base + 0 * stride_conv_state_tok, col0, mask=mask_w)
                if KERNEL_WIDTH >= 3:
                    tl.store(output_base + 1 * stride_conv_state_tok, col1, mask=mask_w)
                if KERNEL_WIDTH >= 4:
                    tl.store(output_base + 2 * stride_conv_state_tok, col2, mask=mask_w)


@triton.jit()
def _causal_conv1d_tree_update_kernel(
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,  # (batch,) initial window rows
    output_state_indices_ptr,  # (batch, seqlen) per-token window rows
    parent_indices_ptr,  # (batch, seqlen)
    o_ptr,  # (batch, dim, seqlen), not aliasing x
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_out_rows_seq: tl.constexpr,
    stride_out_rows_step: tl.constexpr,
    stride_parent_seq: tl.constexpr,
    stride_parent_step: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    PAD_SLOT: tl.constexpr,
):
    """One (request x token, feature block) per program: a width-W causal conv
    needs only the token's last W - 1 ancestors (or the initial window when
    the path is shallower), so tree tokens run in parallel."""
    idx_seq = tl.program_id(0) // seqlen
    token = tl.program_id(0) % seqlen
    feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = feats < dim

    # Ahead of the PDL wait only weights and the step's parent and row tables, all written before the previous kernel.
    init_row = tl.load(conv_state_indices_ptr + idx_seq).to(tl.int64)
    live = init_row != PAD_SLOT
    parents = parent_indices_ptr + idx_seq * stride_parent_seq

    if HAS_BIAS:
        acc = tl.load(bias_ptr + feats, mask=mask, other=0.0).to(tl.float32)
    else:
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    # Column c is the (W - 1 - c)-th ancestor, else the initial window's column.
    out_rows = output_state_indices_ptr + idx_seq * stride_out_rows_seq
    out_row = tl.load(out_rows + token * stride_out_rows_step).to(tl.int64)
    out_base = (
        conv_state_ptr + out_row * stride_conv_state_seq + feats * stride_conv_state_dim
    )
    nodes, past_roots, weights = (), (), ()
    for c in tl.static_range(KERNEL_WIDTH):
        # Ancestor distance for this column, and how far past the root it lands.
        node = token
        past_root = 0
        for j in tl.static_range(KERNEL_WIDTH - 1):
            if j + c < KERNEL_WIDTH - 1:
                parent = tl.load(
                    parents + tl.maximum(node, 0) * stride_parent_step,
                    mask=live,
                    other=-1,
                )
                past_root += (node < 0).to(tl.int32) + ((node >= 0) & (parent < 0)).to(
                    tl.int32
                )
                node = tl.where(node >= 0, parent, node)
        weight = tl.load(
            w_ptr + feats * stride_w_dim + c * stride_w_width, mask=mask, other=0.0
        )
        nodes, past_roots, weights = (
            nodes + (node,),
            past_roots + (past_root,),
            weights + (weight,),
        )
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    # Padded programs still wait, so this grid never completes before its predecessor.
    if init_row == PAD_SLOT:
        # not processing as this is not the actual sequence
        return
    init = (
        conv_state_ptr
        + init_row * stride_conv_state_seq
        + feats * stride_conv_state_dim
    )
    x_row = x_ptr + idx_seq * stride_x_seq + feats * stride_x_dim
    # Every load before any store, which the compiler cannot move loads past.
    values = ()
    for c in tl.static_range(KERNEL_WIDTH):
        node = nodes[c]
        from_x = tl.load(
            x_row + tl.maximum(node, 0) * stride_x_token,
            mask=mask & (node >= 0),
            other=0.0,
        )
        init_col = KERNEL_WIDTH - 1 - past_roots[c]
        from_init = tl.load(
            init + tl.maximum(init_col, 0) * stride_conv_state_tok,
            mask=mask & (node < 0),
            other=0.0,
        )
        values = values + (tl.where(node >= 0, from_x, from_init),)
    for c in tl.static_range(KERNEL_WIDTH):
        acc += values[c] * weights[c]
        if c > 0:
            if out_row >= 0:
                tl.store(
                    out_base + (c - 1) * stride_conv_state_tok, values[c], mask=mask
                )
    if SILU_ACTIVATION:
        acc = acc / (1 + tl.exp(-acc))
    tl.store(
        o_ptr + idx_seq * stride_o_seq + token * stride_o_token + feats * stride_o_dim,
        acc,
        mask=mask,
    )


def _causal_conv1d_tree_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: bool | str | None,
    conv_state_indices: torch.Tensor,
    output_state_indices: torch.Tensor,
    parent_indices: torch.Tensor,
) -> torch.Tensor:
    """Draft-tree causal_conv1d_update: every token in parallel, output out of place."""
    enable_pdl = pdl_enabled()
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    out = torch.empty_like(x)
    block_n = 256
    _causal_conv1d_tree_update_kernel[(batch * seqlen, triton.cdiv(dim, block_n))](
        x,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        output_state_indices,
        parent_indices,
        out,
        dim,
        seqlen,
        *x.stride(),
        *weight.stride(),
        *conv_state.stride(),
        *output_state_indices.stride(),
        *parent_indices.stride(),
        *out.stride(),
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ("silu", "swish"),
        BLOCK_N=block_n,
        ENABLE_PDL=enable_pdl,
        PAD_SLOT=PAD_SLOT_ID,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return out
