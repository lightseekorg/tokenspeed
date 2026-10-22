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


"""Experimental FP32 native indexer; intentionally changes rounded score arithmetic.

Candidate lists are sorted logical block IDs, as produced by V4.1 _finish.
A partial last logical block must be last among valid candidates. Full page
metadata is compacted so invalid pages never participate in native scoring.
"""

import flash_ops  # noqa: F401 - registers custom NPU operators
import torch
from tokenspeed_kernel_npu._triton import tl, triton
from tokenspeed_kernel_npu.ops.dsv41_candidate_gather import candidate_keys
from tokenspeed_kernel_npu.ops.dsv41_index_finish import finish


@triton.jit
def _compact_metadata(
    Table,
    Visible,
    Candidates,
    Map,
    NativeTable,
    Counts,
    TS0,
    TS1,
    VS,
    CS0,
    CS1,
    MS,
    WIDTH,
    PAGES,
    CAND: tl.constexpr,
    T: tl.constexpr,
):
    q = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, T)
    visible = tl.minimum(tl.maximum(tl.load(Visible + q * VS), 0), WIDTH * 64)
    count = 0
    rows = 0
    limit = tl.cdiv(visible, 64)
    if CAND >= 0:
        limit = CAND
    for start in range(0, limit, T):
        col = start + lane
        if CAND >= 0:
            logical = tl.load(Candidates + q * CS0 + col * CS1, col < CAND, other=-1)
            valid = (col < CAND) & (logical >= 0) & (logical * 8 < visible)
            page = tl.load(Table + q * TS0 + (logical // 8) * TS1, valid, other=0)
            valid = valid & (page > 0) & (page < PAGES)
            size = tl.minimum(8, tl.maximum(visible - logical * 8, 0))
        else:
            logical = col
            page = tl.load(Table + q * TS0 + col * TS1, col < limit, other=0)
            valid = (col < limit) & (page > 0) & (page < PAGES)
            size = tl.minimum(64, tl.maximum(visible - col * 64, 0))
        pos = count + tl.cumsum(valid.to(tl.int32), 0) - 1
        tl.store(Map + q * MS + pos, logical, valid)
        if CAND < 0:
            tl.store(NativeTable + q * MS + pos, page, valid)
        count += tl.sum(valid.to(tl.int32), 0)
        rows += tl.sum(tl.where(valid, size, 0), 0)
    m = tl.arange(0, 16)
    tl.store(Counts + q * 16 + m, rows)


@triton.jit
def _map_selected(
    Ids,
    Vals,
    Map,
    Counts,
    OutIds,
    OutVals,
    K: tl.constexpr,
    MS,
    UNIT: tl.constexpr,
    B: tl.constexpr,
):
    q = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, B)
    idx = tl.load(Ids + q * K + lane, lane < K, other=-1)
    num = tl.load(Counts + q * 16)
    valid = (lane < K) & (idx >= 0) & (idx < num)
    logical = tl.load(Map + q * MS + idx // UNIT, valid, other=-1) * UNIT + idx % UNIT
    value = tl.load(Vals + q * K + lane, lane < K, other=-float("inf")).to(tl.float32)
    tl.store(OutIds + q * K + lane, tl.where(valid, logical, -1), lane < K)
    tl.store(OutVals + q * K + lane, tl.where(valid, value, -float("inf")), lane < K)


def native_index(
    q,
    w,
    cache,
    table,
    visible,
    candidates=None,
    k=512,
    out=None,
    *,
    prepared_reindex=None
):
    if (
        cache.dtype != torch.bfloat16
        or not cache.is_contiguous()
        or tuple(cache.shape[1:]) != (64, 128)
    ):
        raise ValueError("Contiguous BF16 index cache required")
    n = q.shape[0]
    if out is None:
        out = (
            torch.empty((n, k), device=q.device, dtype=torch.int32),
            torch.empty(n, device=q.device, dtype=torch.int32),
        )
    if not n:
        return out
    if (
        not table.shape[1]
        or not cache.shape[0]
        or (candidates is not None and not candidates.shape[1])
    ):
        out[0].fill_(-1)
        out[1].zero_()
        return out
    if prepared_reindex is not None:
        if candidates is None:
            raise ValueError("Prepared reindex requires candidate blocks")
        signature = (
            n,
            cache.data_ptr(),
            tuple(cache.shape),
            table.data_ptr(),
            tuple(table.shape),
            visible.data_ptr(),
            candidates.data_ptr(),
            tuple(candidates.shape),
            tuple(table.stride()),
            tuple(visible.stride()),
            tuple(candidates.stride()),
            table.dtype,
            visible.dtype,
            candidates.dtype,
        )
        if prepared_reindex and prepared_reindex["signature"] != signature:
            raise ValueError(
                "Prepared reindex belongs to another forward query window/cache"
            )
        prepared = prepared_reindex.get("payload")
    else:
        prepared = None
    if prepared is None:
        width = table.shape[1]
        count = width if candidates is None else candidates.shape[1]
        stride = (max(1, count) + 15) // 16 * 16
        mapping = torch.full((n, stride), -1, device=q.device, dtype=torch.int32)
        nt = torch.zeros_like(mapping)
        counts = torch.empty((n, 16), device=q.device, dtype=torch.int32)
        from tokenspeed_kernel_npu.ops.dsv41_index_metadata_native import (
            compact_index_metadata,
        )

        if not compact_index_metadata(
            table, visible, candidates, cache.shape[0], mapping, nt, counts
        ):
            _compact_metadata[(n,)](
                table,
                visible,
                candidates,
                mapping,
                nt,
                counts,
                *table.stride(),
                visible.stride(0),
                *(candidates.stride() if candidates is not None else (0, 0)),
                stride,
                width,
                cache.shape[0],
                -1 if candidates is None else count,
                256,
                multibuffer=False
            )
        lengths = counts[:, 0].contiguous()
        qlens = torch.arange(1, n + 1, device=q.device, dtype=torch.int32)
        zero = torch.zeros_like(qlens)
        if candidates is None:
            keys = cache.view(cache.shape[0], 64, 1, 128)
            layout = "PA_BSND"
            blocktable = nt
        else:
            # Eight 8-token candidate blocks form one native 64-token page. Pad the
            # physical storage only; compacted lengths still exclude invalid blocks.
            pages_per_query = (count + 7) // 8
            keys = candidate_keys(
                mapping[:, : pages_per_query * 8], table, visible, cache
            ).view(n * pages_per_query, 64, 1, 128)
            layout = "PA_BSND"
            blocktable = torch.arange(
                n * pages_per_query, device=q.device, dtype=torch.int32
            ).reshape(n, pages_per_query)
        if prepared_reindex is not None:
            prepared_reindex.update(
                signature=signature,
                payload=(
                    mapping,
                    counts,
                    lengths,
                    qlens,
                    zero,
                    keys,
                    blocktable,
                    stride,
                ),
                sources=(cache, table, visible, candidates),
            )
    else:
        mapping, counts, lengths, qlens, zero, keys, blocktable, stride = prepared
    layout = "PA_BSND"
    ids, vals = torch.ops.custom.npu_lightning_indexer(
        q.contiguous(),
        keys,
        w.contiguous(),
        actual_seq_lengths_query=qlens,
        actual_seq_lengths_key=lengths,
        block_table=blocktable,
        init_tensor=zero,
        local_tensor=zero,
        layout_query="TND",
        layout_key=layout,
        sparse_count=k,
        sparse_mode=3,
        return_value=True,
    )
    mapped = torch.empty((n, k), device=q.device, dtype=torch.int32)
    values = torch.empty((n, k), device=q.device)
    _map_selected[(n,)](
        ids,
        vals,
        mapping,
        counts,
        mapped,
        values,
        k,
        stride,
        64 if candidates is None else 8,
        max(8, triton.next_power_of_2(k)),
        multibuffer=False,
    )
    finish(values, mapped, *out)
    return out
