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

"""Pure-numpy core for the n-gram (prompt-lookup) speculative drafter.

Split out from ``ngram.py`` so the algorithm can be exercised without
pulling in the runtime/torch/tokenspeed_kernel import chain.
"""

from __future__ import annotations

import numpy as np


def find_longest_matched_ngram_and_propose_tokens(
    origin_tokens: np.ndarray,
    min_ngram: int,
    max_ngram: int,
    k: int,
) -> np.ndarray:
    """Find the longest ngram suffix-match in ``origin_tokens`` of length
    within ``[min_ngram, max_ngram]`` and return up to ``k`` tokens that
    follow the rightmost match.

    Returns an empty array when ``len(origin_tokens) < min_ngram``, ``k <=
    0``, or no ngram of length >= ``min_ngram`` is found. Matches are
    ranked by the longest suffix length found within ``max_ngram``; ties
    follow the scan order of the reversed KMP pass.
    """
    total = origin_tokens.shape[0]
    if total < min_ngram or k <= 0:
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Work on the reversed token sequence so that "rightmost match in
    # original" becomes "match closest to the front of the reversed
    # sequence". Track the longest prefix-as-suffix in the reversed view,
    # capped at max_ngram.
    tokens = origin_tokens[::-1]
    lps = np.zeros(max_ngram, dtype=np.int32)

    longest_ngram = 0
    position = 0
    prev_lps = 0
    i = 1
    while i < total:
        if tokens[prev_lps] == tokens[i]:
            prev_lps += 1
            if prev_lps >= longest_ngram:
                longest_ngram = prev_lps
                position = i
            if i < max_ngram:
                lps[i] = prev_lps
            if prev_lps == max_ngram:
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            prev_lps = lps[prev_lps - 1]
        else:
            i += 1

    if longest_ngram < min_ngram:
        return np.empty((0,), dtype=origin_tokens.dtype)

    # In origin_tokens, the matched ngram lives at indices
    # [total-1-position : total-1-position+longest_ngram]; drafts start
    # right after it.
    start = total - 1 - position + longest_ngram
    take = min(k, total - start)
    if take <= 0:
        return np.empty((0,), dtype=origin_tokens.dtype)
    return origin_tokens[start : start + take]


def propose_batch_into(
    history: np.ndarray,
    history_len: np.ndarray,
    pool_indices: np.ndarray,
    out: np.ndarray,
    min_ngram: int,
    max_ngram: int,
    spec_num_steps: int,
) -> None:
    """Fill ``out[i]`` with ``[last_verified, d1, ..., d_K]`` for each
    request in the batch by running the KMP suffix-ngram lookup against
    that request's history slot.

    ``out`` must have shape ``(bs, spec_num_steps + 1)`` and is written in
    place. Slots whose ``history_len`` is zero are zeroed defensively; a
    well-formed run never proposes before prefill). Slots with no match
    are padded with ``last_verified`` to preserve the fixed verify width
    without adding a dedicated no-match mask.
    """
    bs = pool_indices.shape[0]
    spec_num_tokens = spec_num_steps + 1
    if out.shape != (bs, spec_num_tokens):
        raise ValueError(
            f"out must be (bs, {spec_num_tokens})-shaped, got {out.shape!r}"
        )

    for i in range(bs):
        pool_idx = int(pool_indices[i])
        length = int(history_len[pool_idx])
        if length == 0:
            out[i, :] = 0
            continue

        ctx = history[pool_idx, :length]
        last_verified = int(ctx[-1])
        out[i, 0] = last_verified

        drafts = find_longest_matched_ngram_and_propose_tokens(
            ctx,
            min_ngram=min_ngram,
            max_ngram=max_ngram,
            k=spec_num_steps,
        )
        n = drafts.size
        if n > 0:
            out[i, 1 : 1 + n] = drafts
        if 1 + n < spec_num_tokens:
            out[i, 1 + n :] = last_verified
