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

from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def api():
    """Load the independently built SM120 vendor namespace.

    The ordinary deep_gemm distribution remains available to other targets.
    This module never substitutes its binaries or changes its metadata.
    """
    import deep_gemm_sm120

    return deep_gemm_sm120


def set_pdl(enabled: bool) -> None:
    api().set_pdl(enabled)


def get_pdl() -> bool:
    return api().get_pdl()


def get_num_sms() -> int:
    return api().get_num_sms()


def fp8_fp4_mqa_logits(
    q,
    kv,
    weights,
    cu_seq_len_k_start,
    cu_seq_len_k_end,
    max_seqlen_k,
    logits_dtype,
    schedule_meta=None,
):
    """Adapt the modern dense MQA contract to the pinned SM120 implementation.

    The caller masks invalid physical pages and ignores unwritten score tails.
    A supplied modern schedule is unsupported and must not be discarded.
    """
    if schedule_meta is not None:
        raise ValueError("SM120 dense MQA does not accept a prepared schedule")
    return api().fp8_fp4_mqa_logits(
        q=q,
        kv=kv,
        weights=weights,
        cu_seq_len_k_start=cu_seq_len_k_start,
        cu_seq_len_k_end=cu_seq_len_k_end,
        clean_logits=False,
        max_seqlen_k=max_seqlen_k,
        logits_dtype=logits_dtype,
    )


def get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms, indices=None):
    return api().get_paged_mqa_logits_metadata(
        context_lens,
        block_kv,
        num_sms,
        indices=indices,
    )


def fp8_fp4_paged_mqa_logits(
    q,
    kv_cache,
    weights,
    context_lens,
    block_table,
    schedule_meta,
    max_context_len,
    logits_dtype,
    indices=None,
):
    """Keep all modern paged inputs and the caller-owned invalid-page mask."""
    return api().fp8_fp4_paged_mqa_logits(
        q=q,
        kv_cache=kv_cache,
        weights=weights,
        context_lens=context_lens,
        block_table=block_table,
        schedule_meta=schedule_meta,
        max_context_len=max_context_len,
        clean_logits=False,
        logits_dtype=logits_dtype,
        indices=indices,
    )


def warmup_dense_indexer(heads: int, device: torch.device) -> None:
    """Compile the SM120 dense FP4 scorer through its own vendor namespace."""
    capacity = 4096
    keys = torch.zeros((capacity, 64), dtype=torch.int8, device=device)
    key_scales = torch.zeros(capacity, dtype=torch.int32, device=device)
    for rows in (1, 256):
        query = torch.zeros((rows, heads, 64), dtype=torch.int8, device=device)
        query_scales = torch.zeros((rows, heads), dtype=torch.int32, device=device)
        weights = torch.ones((rows, heads), dtype=torch.float32, device=device)
        starts = torch.zeros(rows, dtype=torch.int32, device=device)
        ends = torch.full((rows,), capacity, dtype=torch.int32, device=device)
        fp8_fp4_mqa_logits(
            q=(query, query_scales),
            kv=(keys, key_scales),
            weights=weights,
            cu_seq_len_k_start=starts,
            cu_seq_len_k_end=ends,
            max_seqlen_k=capacity,
            logits_dtype=torch.float32,
        )
