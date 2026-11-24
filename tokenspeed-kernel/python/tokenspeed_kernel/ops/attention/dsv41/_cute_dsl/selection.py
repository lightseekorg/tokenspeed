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

"""Explicit SM120 DeepSelect experiment using the Hopper selection contract."""

import os

import torch
from tokenspeed_kernel.platform import ArchVersion, current_platform


def enabled():
    mode = os.environ.get("TOKENSPEED_EXPERIMENT_SM120_DEEPSELECT")
    if mode not in (None, "0", "1"):
        raise ValueError("TOKENSPEED_EXPERIMENT_SM120_DEEPSELECT must be 0 or 1")
    platform = current_platform()
    return (
        mode == "1"
        and platform.is_nvidia
        and platform.arch_version == ArchVersion(12, 0)
    )


def select(scores, ends, topk):
    from tokenspeed_kernel.ops.attention.dsa._cute_dsl import deep_select

    k = min(topk, scores.shape[1])
    if k == 0:
        return scores[:, :0], torch.empty(
            (scores.shape[0], 0), device=scores.device, dtype=torch.int32
        )
    ids, values = deep_select.deepselect_topk(
        scores,
        ends.contiguous(),
        k,
        capacity=deep_select.capacity_for(k),
        cluster_size=1,
    )
    return values, ids


def select_candidates(scores, topk):
    # The scan writes -inf for invalid candidates. Zero-visible queries skip
    # the scan, so its caller must initialize the complete scratch matrix.
    ends = torch.full(
        (scores.shape[0],), scores.shape[1], device=scores.device, dtype=torch.int32
    )
    return select(scores, ends, topk)


def select_full(scores, table, lengths, pages, topk, candidate_topk, out):
    from tokenspeed_kernel.ops.attention.dsv41.triton import (
        _finish_topk,
        clean_block_maxima,
    )

    clean, maxima, block_ends = clean_block_maxima(scores, lengths, table, pages)
    values, ids = select(clean, lengths, topk)
    _finish_topk(values, ids, out[0], out[1])
    if candidate_topk:
        values, ids = select(maxima, block_ends, candidate_topk)
        _finish_topk(values, ids, out[2], out[3])
    return out
