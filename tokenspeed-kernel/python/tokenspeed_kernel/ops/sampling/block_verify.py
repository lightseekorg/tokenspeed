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

"""Block verification using clipped prefix ratios and positive residuals.

Sun et al. (2024), https://arxiv.org/abs/2403.10444. Inputs are already
normalized conditional distributions, including target sampling filters.

Notation, per request: ``x_1..x_K`` are the proposed tokens, ``q_i`` the
law that drew ``x_i`` and ``p_i`` the target law at the same position
(row ``i - 1`` of ``draft_probs`` / ``target_probs``); row ``K`` of
``target_probs`` is the bonus position.

1. Prefix ratio: ``r_0 = 1``, ``r_i = min(1, r_{i-1} * p_i(x_i) / q_i(x_i))``.
2. Residual mass after prefix ``i``:
   ``z_i = sum_v max(r_i * p_{i+1}(v) - q_{i+1}(v), 0)``.
3. Acceptance threshold of prefix ``i``: ``h_i = z_i / (z_i + 1 - r_i)``
   for ``i < K`` and ``h_K = r_K``.
4. With one independent coin per prefix, the accepted length ``tau`` is the
   largest ``i`` whose coin is below ``h_i`` (``0`` if none). A later prefix
   can be accepted after an earlier one failed; token-wise rejection stops at
   the first failure.
5. One more token: from the residual ``max(r_tau * p_{tau+1} - q_{tau+1}, 0)``
   if ``tau < K``, else from ``p_{K+1}`` (bonus).

The output then follows the target's joint law, and its expected length is at
least that of token-wise rejection. A proposal whose recorded ``q`` is not in
``(0, threshold]`` is a sentinel (no proposal in that slot) and truncates
``K`` there.
"""

import torch
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    format_signatures,
)


def block_speculative_sampling(
    *,
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    reject_draft_prob_threshold,
):
    """Write an accepted prefix and one residual/bonus token in chain layout.

    Args:
        predicts: Flattened int32 output [B*N].
        accept_index: Int32 [B,N] flat output indices, -1 beyond accepted output.
        accept_token_num: Int32 [B], accepted draft count (excludes bonus).
        candidates: Int32 [B,N], anchor followed by K=N-1 proposed tokens.
        uniform_samples: Float32 [B,N] independent prefix coins in [0,1).
        uniform_samples_for_final_sampling: Float32 [B], residual/bonus coins.
        target_probs: Contiguous float32 [B,N,V] normalized target laws.
        draft_probs: Contiguous float32 [B,N,V]; first K rows are proposal laws.
        reject_draft_prob_threshold: Values above this mark unavailable proposals.

    Returns:
        None. Writes the three output buffers in place.
    """
    if target_probs.ndim != 3 or draft_probs.shape != target_probs.shape:
        raise ValueError(
            "block verification needs matching [B,N,V] probability tensors"
        )
    if target_probs.dtype != torch.float32 or draft_probs.dtype != torch.float32:
        raise ValueError("block verification probabilities must be float32")
    if not target_probs.is_contiguous() or not draft_probs.is_contiguous():
        raise ValueError("block verification probability tensors must be contiguous")
    if not target_probs.is_cuda:
        kernel = torch_block_verify
    else:
        import tokenspeed_kernel.ops.sampling.triton.block_verify  # noqa: F401

        # The torch oracle loops over rows on the host and shares the triton
        # leaf's priority, so device tensors name the triton solution.
        kernel = select_kernel(
            "sampling",
            "block_verify",
            format_signature(target_probs=dense_tensor_format(torch.float32)),
            solution="triton",
        )
    kernel(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        reject_draft_prob_threshold=reject_draft_prob_threshold,
    )


@register_kernel(
    "sampling",
    "block_verify",
    name="torch_block_verify",
    solution="torch",
    priority=Priority.PORTABLE,
    signatures=format_signatures("target_probs", "dense", {torch.float32}),
)
def torch_block_verify(
    *,
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    reject_draft_prob_threshold,
):
    """CPU oracle: the module docstring's steps 1-5, one request at a time."""
    bs, n, vocab = target_probs.shape
    accept_index.fill_(-1)
    for b in range(bs):
        # Step 1: prefix ratios; k shrinks to the first sentinel proposal.
        k = n - 1
        ratios = [1.0]
        for i in range(k):
            token = int(candidates[b, i + 1])
            q = float(draft_probs[b, i, token]) if 0 <= token < vocab else -1.0
            if q <= 0 or q > reject_draft_prob_threshold:
                k = i
                break
            ratios.append(min(1.0, ratios[-1] * float(target_probs[b, i, token]) / q))
        # Steps 2-4: threshold h of each prefix length i + 1; keep the longest
        # prefix whose coin passes.
        accepted = 0
        for i in range(k):
            r = ratios[i + 1]
            if i + 1 == k:
                h = r
            else:
                z = float(
                    (r * target_probs[b, i + 1] - draft_probs[b, i + 1])
                    .clamp_min(0)
                    .sum()
                )
                denom = z + 1 - r
                h = z / denom if denom > 0 else 1.0
            # torch.rand includes zero: an impossible prefix must not pass
            # when both its threshold and its coin are exactly zero.
            if float(uniform_samples[b, i]) < h:
                accepted = i + 1
        # Step 5: residual token after a rejection, bonus token otherwise.
        if accepted == k:
            law = target_probs[b, accepted]
        else:
            law = (
                ratios[accepted] * target_probs[b, accepted] - draft_probs[b, accepted]
            ).clamp_min(0)
            if float(law.sum()) <= 0:
                # Zero-probability rejection reached through fp32 rounding:
                # the residual is empty, so draw from the target law.
                law = target_probs[b, accepted]
        cdf = law.double().cumsum(0)
        coin = float(uniform_samples_for_final_sampling[b]) * float(cdf[-1])
        token = min(int(torch.searchsorted(cdf, coin, right=True)), vocab - 1)
        predicts[b * n : b * n + accepted].copy_(candidates[b, 1 : accepted + 1])
        predicts[b * n + accepted] = token
        accept_index[b, : accepted + 1] = torch.arange(
            b * n,
            b * n + accepted + 1,
            dtype=accept_index.dtype,
            device=accept_index.device,
        )
        accept_token_num[b] = accepted
