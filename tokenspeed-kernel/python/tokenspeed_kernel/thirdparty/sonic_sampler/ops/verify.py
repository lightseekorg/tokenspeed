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

fast_logf = libdevice.fast_logf

from tokenspeed_kernel.thirdparty.sonic_sampler.ops.base import (
    gdc_launch_dependents,
    gdc_wait,
    ones_like,
    philox_gumbel,
    philox_key,
    philox_uniform,
)


@jit
def resolve_shift_margin(
    q_ptr,
    f_ptr,
    block_id: tl.constexpr,
    lookahead: tl.constexpr,
    slotted: tl.constexpr,
    varlen: tl.constexpr,
):

    batch_id = block_id

    if varlen:

        margin, upper = tl.split(tl.load(q_ptr + block_id + tl.arange(0, 2)))
        timesteps = upper - margin

    else:

        timesteps = lookahead + 1
        margin = block_id * timesteps

    if slotted:

        batch_id = tl.load(f_ptr + batch_id)

    return batch_id, margin, timesteps


@jit
def verify_greedy(
    j_ptr,
    margin,
    span,
    drafted,
    steps_mask,
    draft_mask,
    max_k: tl.constexpr,
):

    selections = tl.load(j_ptr + (margin * max_k) + span, mask=steps_mask, other=-1)

    matched = (selections == drafted) & draft_mask
    accepted = tl.argmin(matched, axis=0)

    token = tl.gather(selections, accepted[None], axis=0)

    return token, accepted, selections


@jit
def decode_update(
    c_ptr,
    batch_id,
    span,
    indices,
    accepted,
    stride_c: tl.constexpr,
):

    tl.atomic_add(
        c_ptr + (batch_id * stride_c) + indices,
        ones_like(indices),
        mask=(span <= accepted),
        sem="acq_rel",
        scope="cta",
    )


@jit
def load_targets(
    v_ptr,
    j_ptr,
    margin,
    span,
    cols,
    mask,
    max_k: tl.constexpr,
):

    offsets = ((margin + span) * max_k)[:, None] + cols[None, :]

    indices = tl.load(j_ptr + offsets, mask=mask[:, None], other=-1)
    values = tl.load(v_ptr + offsets, mask=mask[:, None], other=0)

    return values, indices


@jit
def verify_drafted(
    d_ptr,
    u_ptr,
    h_ptr,
    l_ptr,
    batch_id,
    indicator,
    span,
    drafted,
    probabilities,
    indices,
    mask,
    block_g: tl.constexpr,
    stride_c: tl.constexpr,
    stride_d: tl.constexpr,
    stride_u: tl.constexpr,
    reject_threshold,
):

    # Extract the selected target probabilities via masked-sum HFMA reduction tree.

    matches = indices == drafted[:, None]
    targets = tl.sum(probabilities * matches, axis=1)

    # Load or draw the uniform random values.

    if u_ptr is not None:

        uniform = tl.load(u_ptr + (batch_id * stride_u) + span, mask=mask, other=0)

    else:

        uniform = philox_uniform(
            h_ptr,
            l_ptr,
            batch_id,
            span,
            mask,
            stride_c,
            block_g,
        )

    if d_ptr is not None and (indicator & 0x800) == 0:

        # Load the drafted token source probabilities.

        positions = (batch_id * stride_d) + (span * stride_c) + drafted
        sources = tl.load(d_ptr + positions, mask=mask, other=0)

        # Update the LHS of the verification inequality.

        uniform *= sources

        if reject_threshold is not None:

            # A probability above the threshold marks a slot with no proposal; a draft the target cannot emit is never accepted either.
            rejected = (sources > reject_threshold) | (targets <= 0)
            uniform = tl.where(rejected, float("inf"), uniform)

    # Compute the rejection offset(s) i.e. total accepted.

    marked = (uniform <= targets) & mask

    accepted = tl.argmin(marked, axis=0)

    return accepted, targets, matches


@jit
def rejection_sample(
    d_ptr,
    e_ptr,
    h_ptr,
    l_ptr,
    batch_id,
    timesteps,
    indicator,
    span,
    drafted,
    accepted,
    targets,
    matches,
    probabilities,
    indices,
    block_g,
    max_k: tl.constexpr,
    update_token: tl.constexpr,
    update_scores: tl.constexpr,
    stride_c: tl.constexpr,
    stride_d: tl.constexpr,
    stride_e: tl.constexpr,
    reject_threshold,
):

    # Gather the rejected row from the target probabilities and indices.

    rejected = accepted.broadcast_to(1, max_k)

    scores = tl.gather(probabilities, rejected, axis=0).reshape((max_k,))

    positions = tl.gather(indices, rejected, axis=0).reshape((max_k,))

    # Load or draw the gumbel noise for the corresponding position(s).

    shifts = (accepted * stride_c) + positions

    if e_ptr is not None:

        gumbel = tl.load(e_ptr + (batch_id * stride_e) + shifts)

    else:

        key = philox_key(h_ptr, l_ptr, batch_id, accepted)
        gumbel = philox_gumbel(key, positions)

    if accepted == (timesteps - 1):

        # Skip residual correction when all drafted tokens are accepted.

        residuals = scores

    elif d_ptr is None or (indicator & 0x800) > 0:

        # Masked rejection sampling under the stochastic-verify-greedy-draft regime.

        predicate = tl.gather(matches, rejected, axis=0).reshape((max_k,))

        residuals = tl.where(predicate, 0.0, scores)

    else:

        # Load the draft probabilities for the corresponding position(s).

        marginals = tl.load(d_ptr + (batch_id * stride_d) + shifts)

        if reject_threshold is not None:

            # A sentinel row (no recorded proposal) contributes nothing: the residual is the target.
            marginals = tl.where(marginals > reject_threshold, 0.0, marginals)
        residuals = tl.clamp(scores - marginals, 0.0, 1.0)

    # Apply rejection sampling to surface the next token.

    selection = tl.argmax(fast_logf(residuals) - gumbel, axis=0, keep_dims=True)
    token = tl.gather(positions, selection, axis=0)

    if update_token:

        drafted = tl.where(span == accepted, token.broadcast_to((block_g,)), drafted)

    if update_scores:

        score = tl.gather(scores, selection, axis=0).broadcast_to((block_g,))

        targets = tl.where(span == accepted, score, targets)

    return token, targets, drafted


@jit
def chain_speculative_verification_kernel(
    # Input Data Pointer(s).
    v_ptr,  # Selected Values -> [ B • (γ + 1), M ].
    j_ptr,  # Selected Indices -> [ B • (γ + 1), M ].
    z_ptr,  # Indicators -> [ B ].
    f_ptr,  # Slot Mapping -> [ B ].
    q_ptr,  # Cumulative Lengths -> [ B + 1 ].
    t_ptr,  # Drafted Tokens -> [ B • (γ + 1) ].
    d_ptr,  # Draft Probabilities -> [ B, γ, V ].
    u_ptr,  # Uniform -> [ B, γ ].
    e_ptr,  # Gumbel Noise -> [ B • (γ + 1), V ].
    h_ptr,  # Noise Seeds -> [ B ].
    l_ptr,  # Noise Offsets -> [ B ].
    c_ptr,  # Decode Counts -> [ B, V ].
    r_ptr,  # Top-K Log-Probabilities -> [ B ].
    # Output Data Pointer(s).
    o_ptr,  # Output Tokens -> [ B • (γ + 1) ].
    a_ptr,  # Total Accepted Token(s) -> [ B ].
    w_ptr,  # Log-Probabilities -> [ B • (γ + 1) ].
    y_ptr,  # Top-K Log-Probabilities (Values) -> [ B • (γ + 1), Z ].
    k_ptr,  # Top-K Log-Probabilities (Indices) -> [ B • (γ + 1), Z ].
    # Model Specific(s).
    lookahead: tl.constexpr,  # Lookahead -> γ.
    max_k: tl.constexpr,  # Maximum Top-K Value -> M.
    # Conditional Flag(s).
    slotted: tl.constexpr,  # Slot Indirection Flag.
    varlen: tl.constexpr,  # Variable-Length Lookahead Flag.
    enable_pdl: tl.constexpr,  # PDL Enablement Flag.
    update_counts: tl.constexpr,  # Update Decode Counts Flag.
    logprobs: tl.constexpr,  # Log-Probabilities Flag.
    # Stride(s).
    stride_d: tl.constexpr,  # Batch Stride of Draft Probabilities.
    stride_e: tl.constexpr,  # Batch Stride of Gumbel Noise.
    stride_u: tl.constexpr,  # Batch Stride of Uniform.
    stride_c: tl.constexpr,  # Batch Stride of Decode Counts (Target Vocab Size) -> V.
    stride_z: tl.constexpr,  # Batch Stride of Top-K Log-Probabilities (Values).
    # Kernel Specific(s).
    block_g: tl.constexpr,  # Next Power-of-2 Timesteps (γ + 1) -> T.
    block_z: tl.constexpr,  # Next Power-of-2 Top-K Log-Probabilities Targets -> Z'.
    reject_threshold=None,  # Draft Probabilities above this mark a Slot with no Proposal.
) -> None:

    block_id = tl.program_id(axis=0)

    if enable_pdl:

        gdc_wait()

    # Extract the batch id from program id and load the indicator.

    batch_id, margin, timesteps = resolve_shift_margin(
        q_ptr,
        f_ptr,
        block_id,
        lookahead,
        slotted,
        varlen,
    )

    indicator = tl.load(z_ptr + batch_id)

    # Set-up the token and column-wise span(s).

    span = tl.arange(0, block_g)
    cols = tl.arange(0, max_k)

    # Define the (γ + 1) steps mask and the γ draft mask.

    steps_mask = span < timesteps
    draft_mask = span < (timesteps - 1)

    # Load the drafted tokens block.

    drafted = tl.load(t_ptr + margin + span, mask=draft_mask, other=0)

    if (indicator & 0x40) > 0:

        # Apply greedy verification.

        token, accepted, selections = verify_greedy(
            j_ptr,
            margin,
            span,
            drafted,
            steps_mask,
            draft_mask,
            max_k,
        )

        if enable_pdl:

            gdc_launch_dependents()

    else:

        # Load target token indices and probabilities -> [ T, M ].

        probabilities, indices = load_targets(
            v_ptr, j_ptr, margin, span, cols, steps_mask, max_k
        )

        # Verify the drafted tokens against the target probabilities.

        accepted, targets, matches = verify_drafted(
            d_ptr,
            u_ptr,
            h_ptr,
            l_ptr,
            batch_id,
            indicator,
            span,
            drafted,
            probabilities,
            indices,
            draft_mask,
            block_g,
            stride_c,
            stride_d,
            stride_u,
            reject_threshold,
        )

        # Rejection-sample the next token and conditionally update selected token(s) and score(s).

        token, targets, selections = rejection_sample(
            d_ptr,
            e_ptr,
            h_ptr,
            l_ptr,
            batch_id,
            timesteps,
            indicator,
            span,
            drafted,
            accepted,
            targets,
            matches,
            probabilities,
            indices,
            block_g,
            max_k,
            update_counts,
            logprobs,
            stride_c,
            stride_d,
            stride_e,
            reject_threshold,
        )

        if logprobs:

            # Compute selected token log-probabilities.

            log_scores = fast_logf(targets)
            span_mask = span <= accepted

            if (indicator & 0x400) > 0:

                # Extract top-k token indices and compute their corresponding log-probabilities.

                logprobs_ids = tl.arange(0, block_z)[None, :].broadcast_to(
                    (block_g, block_z)
                )

                logprobs_mask = (logprobs_ids < tl.load(r_ptr + batch_id)) & span_mask[
                    :, None
                ]
                offsets = ((margin + span)[:, None] * stride_z) + logprobs_ids

                top_logprobs = tl.gather(probabilities, logprobs_ids, axis=1).to(
                    tl.float32
                )

                top_tokens = tl.gather(indices, logprobs_ids, axis=1)

                if enable_pdl:

                    gdc_launch_dependents()

                tl.store(y_ptr + offsets, fast_logf(top_logprobs), mask=logprobs_mask)
                tl.store(k_ptr + offsets, top_tokens, mask=logprobs_mask)

            elif enable_pdl:

                gdc_launch_dependents()

            tl.store(w_ptr + margin + span, log_scores, mask=span_mask)

        elif enable_pdl:

            gdc_launch_dependents()

    if update_counts:

        # Increment the decode count(s) for the accepted and selected token(s).

        decode_update(c_ptr, batch_id, span, selections, accepted, stride_c)

    # Store the rejection-sampled / next seed token and the total accepted.

    tl.store(o_ptr + margin + accepted[None], token)
    tl.store(a_ptr + block_id, accepted)
