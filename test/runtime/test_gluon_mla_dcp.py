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

"""Gluon MLA decode under the decode-context-parallel (DCP) layout.

KV parallelism shards every request's MLA pages page-cyclically over the KVP
ranks. DCP decode runs each rank over its owned pages only (a compact local
page table and one local causal length per query row) and merges the per-rank
partials by their LSE. This mirrors ``test_flashmla_shards_merge_to_full_attention``
for the gfx950 Gluon kernels: every rank is emulated on one GPU, and the merge
must reproduce the unsharded decode over the full page table.

Geometry is Kimi-K3.1's EAGLE-3.1 MLA draft layer: 12 heads per TP rank, so a
KVP group of ``degree`` ranks attends ``12 * degree`` gathered heads.
"""

import os
import sys

import pytest
import torch
from tokenspeed_kernel.ops.kvcache.triton_cache_placement import (
    virtual_slots_to_local,
)
from tokenspeed_kernel.platform import current_platform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, suite="runtime-1gpu")

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and current_platform().is_cdna4),
    reason="the Gluon MLA decode kernels require AMD gfx950 (CDNA4)",
)

KV_LORA, ROPE, NOPE = 512, 64, 128
WIDTH = KV_LORA + ROPE
SCALE = (NOPE + ROPE) ** -0.5
PAGE, GRANULARITY = 64, 128
PAGES_PER_BLOCK = GRANULARITY // PAGE
VIRTUAL_BLOCKS = 24
# The backend's max_context_len bound, which selects split-K with a reduce.
MAX_CONTEXT = 4096
# The gfx950 single-query kernels that return an LSE take at most 16 heads.
MAX_KERNEL_HEADS = 16
# Scheduler blocks per request (block 0 is the null block) and their lengths:
# one long request that ends inside its last block, one short one whose
# table is padded with the null page.
BLOCKS = ([5, 2, 8, 13, 21, 3, 17], [9, 4])
LENGTHS = (6 * GRANULARITY + 37, GRANULARITY + 12)

DTYPES = {
    "bf16": (torch.bfloat16, torch.bfloat16),
    "bf16q_fp8kv": (torch.bfloat16, torch.float8_e4m3fn),
    "fp8": (torch.float8_e4m3fn, torch.float8_e4m3fn),
}
# Merged-vs-unsharded and kernel-vs-torch tolerances per dtype.
TOLERANCES = {
    "bf16": (dict(atol=2e-3, rtol=3e-2), dict(atol=1e-2, rtol=5e-2)),
    "bf16q_fp8kv": (dict(atol=2e-3, rtol=3e-2), dict(atol=2e-2, rtol=1e-1)),
    "fp8": (dict(atol=2e-2, rtol=1e-1), dict(atol=4e-2, rtol=1e-1)),
}


def _kernel_page_table(device):
    pages = max(len(b) for b in BLOCKS) * PAGES_PER_BLOCK
    table = torch.zeros(len(BLOCKS), pages, dtype=torch.int32, device=device)
    for row, blocks in enumerate(BLOCKS):
        for i, block in enumerate(blocks):
            for sub in range(PAGES_PER_BLOCK):
                table[row, i * PAGES_PER_BLOCK + sub] = block * PAGES_PER_BLOCK + sub
    return table


def _decode(q, cache, page_table, lengths, max_seqlen_k):
    """One q_len=1 Gluon decode row per query row, returning (out, lse).

    Head groups beyond the kernels' 16-head limit fold onto the batch axis:
    every row and head attends independently, so a [rows, heads] problem is
    [rows * groups, heads / groups] against repeated tables and lengths.
    """
    from tokenspeed_kernel.ops.attention.mla import mla_decode_with_kvcache

    rows, heads = q.shape[0], q.shape[2]
    group = max(g for g in range(1, MAX_KERNEL_HEADS + 1) if heads % g == 0)
    groups = heads // group
    out, lse = mla_decode_with_kvcache(
        q=q.reshape(rows * groups, 1, group, WIDTH),
        kv_cache=cache.view(-1, PAGE, 1, WIDTH),
        page_table=page_table.repeat_interleave(groups, dim=0),
        cache_seqlens=lengths.repeat_interleave(groups),
        max_seqlen_k=max_seqlen_k,
        qk_nope_head_dim=NOPE,
        kv_lora_rank=KV_LORA,
        qk_rope_head_dim=ROPE,
        softmax_scale=SCALE,
        return_lse=True,
        solution="gluon",
    )
    assert out.dtype == torch.bfloat16 and lse.dtype == torch.float32
    return out.reshape(rows, heads, KV_LORA), lse.reshape(rows, heads)


def _torch_reference(q, source, page_table, endpoints, width):
    source = source.float()
    outputs, lses = [], []
    for row, end in enumerate(endpoints.tolist()):
        positions = torch.arange(end, device=q.device)
        pages = page_table[row // width, positions // PAGE].long()
        keys = source[pages * PAGE + positions % PAGE]
        scores = q[row, 0].float() @ keys.T * SCALE
        outputs.append(torch.softmax(scores, dim=-1) @ keys[:, :KV_LORA])
        lses.append(torch.logsumexp(scores, dim=-1))
    return torch.stack(outputs), torch.stack(lses)


@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("degree", [2, 4, 8])
@pytest.mark.parametrize("heads", ["tp_group", 16])
@pytest.mark.parametrize("query_width,causal", [(1, True), (4, True), (4, False)])
# The exact bound leaves a short FP8 scan in one split, whose LSE stage 1
# writes directly.
@pytest.mark.parametrize("max_seqlen_k", [MAX_CONTEXT, max(LENGTHS)])
def test_gluon_mla_shards_merge_to_full_attention(
    dtype, degree, heads, query_width, causal, max_seqlen_k
):
    from tokenspeed_kernel.ops.attention.dsv4._triton.dcp import (
        normalize_dcp_partials,
    )

    from tokenspeed.runtime.layers.attention.dcp.metadata import (
        CompactDCPLayout,
        refresh_dcp_page_table_metadata,
    )

    q_dtype, kv_dtype = DTYPES[dtype]
    merge_tolerance, reference_tolerance = TOLERANCES[dtype]
    heads = 12 * degree if heads == "tp_group" else heads
    torch.manual_seed(17)
    device = "cuda"
    tokens = VIRTUAL_BLOCKS * GRANULARITY
    source = torch.randn(tokens, WIDTH, device=device) * 0.1
    # Skew whole blocks so the shards carry very different attention mass.
    for block, shift in ((5, 0.4), (2, -0.3), (8, 0.1), (13, 0.25), (9, -0.2)):
        source[block * GRANULARITY : (block + 1) * GRANULARITY] += shift
    source[:GRANULARITY] = 0
    source = source.to(kv_dtype)
    table = _kernel_page_table(device)
    lengths = torch.tensor(LENGTHS, dtype=torch.int32, device=device)
    requests = len(LENGTHS)
    rows = requests * query_width
    q = torch.randn(rows, 1, heads, WIDTH, device=device) + 0.25
    q = q.to(q_dtype)

    # The backend's layout: one global endpoint per causal verify query, or
    # the block end for every query of a non-causal block draft.
    offsets = (
        torch.arange(1 - query_width, 1, dtype=torch.int32, device=device)
        if causal
        else torch.zeros(query_width, dtype=torch.int32, device=device)
    )
    endpoints = (lengths[:, None] + offsets).reshape(-1)
    ref, ref_lse = _decode(
        q, source, table.repeat_interleave(query_width, dim=0), endpoints, max_seqlen_k
    )
    torch_out, torch_lse = _torch_reference(q, source, table, endpoints, query_width)
    torch.testing.assert_close(ref.float(), torch_out, **reference_tolerance)
    torch.testing.assert_close(ref_lse, torch_lse, atol=2e-2, rtol=1e-2)

    loc = torch.arange(tokens, device=device)
    outputs, lses, local_total = [], [], 0
    for rank in range(degree):
        local_cache = torch.zeros(
            ((VIRTUAL_BLOCKS - 1 + degree - 1) // degree + 1) * GRANULARITY,
            WIDTH,
            device=device,
            dtype=kv_dtype,
        )
        slots, owned = virtual_slots_to_local(
            loc,
            rows_per_page=GRANULARITY,
            virtual_block_count=VIRTUAL_BLOCKS,
            degree=degree,
            rank=rank,
        )
        local_cache.view(torch.uint8)[slots[owned]] = source.view(torch.uint8)[
            loc[owned]
        ]
        assert not local_cache[:GRANULARITY].view(torch.uint8).any()
        metadata = refresh_dcp_page_table_metadata(
            page_table=table,
            virtual_block_count=VIRTUAL_BLOCKS,
            degree=degree,
            rank=rank,
            layout=CompactDCPLayout(
                lengths, PAGE, GRANULARITY, lengths[:, None] + offsets
            ),
            previous=None,
        )
        local = metadata.local_visible_lens.reshape(-1)
        local_total = local_total + local
        # An empty row attends the zero page at length 1: a zero length makes
        # the kernel skip the row and leave its output and LSE unwritten.
        partial, lse = _decode(
            q,
            local_cache,
            metadata.local_page_table.repeat_interleave(query_width, dim=0),
            local.clamp_min(1),
            max_seqlen_k,
        )
        assert torch.isfinite(lse).all()
        partial, lse = normalize_dcp_partials(partial, lse, local, None)
        empty = local == 0
        assert torch.isneginf(lse[empty]).all() and not partial[empty].any()
        assert torch.isfinite(lse[~empty]).all()
        outputs.append(partial.float())
        lses.append(lse)
    # Every token of every query row is owned by exactly one rank.
    torch.testing.assert_close(local_total, endpoints, rtol=0, atol=0)
    weights = torch.softmax(torch.stack(lses), dim=0).unsqueeze(-1)
    merged = (torch.stack(outputs) * weights).sum(0)
    merged_lse = torch.logsumexp(torch.stack(lses), dim=0)
    torch.testing.assert_close(merged, ref.float(), **merge_tolerance)
    torch.testing.assert_close(merged_lse, ref_lse, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(merged, torch_out, **reference_tolerance)
