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

"""Opt-in SM120 full-index scoring experiment; candidate reindex stays portable."""

from __future__ import annotations

import logging

import torch
from tokenspeed_kernel.thirdparty import deepgemm_sm120 as deep_gemm

_LOGGED = False


def full_index_topk(q, weights, cache, table, visible, topk, candidate_topk, out):
    from tokenspeed_kernel.ops.attention.dsv41.triton import (
        _index_finish_parts,
        pack_index_inputs,
        safe_metadata,
    )

    global _LOGGED
    if not _LOGGED:
        logging.getLogger(__name__).warning(
            "SM120 experimental DeepGEMM full indexer active: heads=%s; candidate reindex remains Triton",
            q.shape[1],
        )
        _LOGGED = True
    rows, lengths, blocks, block_lengths = out
    capacity = table.shape[1] * 64
    rows.fill_(-1)
    lengths.zero_()
    blocks.fill_(-1)
    block_lengths.zero_()
    if not q.shape[0] or not capacity:
        return out
    packed, w = pack_index_inputs(q, weights, torch.float32)
    safe_table, safe_lengths = safe_metadata(
        table, visible, cache.shape[0], capacity, 64
    )
    lens = safe_lengths[:, None]
    metadata = deep_gemm.get_paged_mqa_logits_metadata(
        lens, 64, deep_gemm.get_num_sms()
    )
    kv = cache.as_strided((cache.shape[0], 64, 1, 68), (cache.stride(0), 68, 68, 1))
    scores = deep_gemm.fp8_fp4_paged_mqa_logits(
        q=(
            packed[0].contiguous().view(torch.int8).unsqueeze(1),
            packed[1].contiguous().unsqueeze(1),
        ),
        kv_cache=kv,
        weights=w,
        context_lens=lens,
        block_table=safe_table,
        schedule_meta=metadata,
        max_context_len=capacity,
        logits_dtype=torch.float32,
    )
    from tokenspeed_kernel.thirdparty.deepgemm_sm120_index_select_experiment import (
        select_scores,
    )

    return select_scores(
        scores,
        table,
        safe_lengths,
        cache.shape[0],
        topk,
        candidate_topk,
        out,
        32 if candidate_topk else 16,
    )
