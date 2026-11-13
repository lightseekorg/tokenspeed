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

"""Fused token-major input preparation for capacity-planned native KDA."""

import torch
from tokenspeed_kernel._triton import tl, triton

# Sequences per prefix-sum tile; a fixed width keeps the batch out of the key.
_SEQUENCE_BLOCK = 32


@triton.jit(do_not_specialize=["T", "N", "MAX_CHUNKS", "SEARCH_STEPS"])
def _prepare_capacity_scan_kernel(
    Q,
    K,
    V,
    G,
    B,
    OQ,
    OK,
    OV,
    OG,
    OB,
    CU,
    CHUNKS,
    CHUNK_ROWS,
    # Batch-shaped counts stay unspecialized, so every batch shape shares one binary.
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    N,
    QS: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    GS: tl.constexpr,
    BS: tl.constexpr,
    MAX_CHUNKS,
    SEARCH_STEPS,
    SEQ_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = x // (H * D), x % (H * D)
    live = tl.load(CU + N)
    valid = (row < T) & (row < live)
    # Capacity padding may hold anything (the gate projection turns padded
    # inputs into NaNs), so every input is copied with its padding scrubbed.
    tl.store(OG + x, tl.load(G + row * GS + col, valid, 0).to(tl.float32), row < T)
    tl.store(OQ + x, tl.load(Q + row * QS + col, valid, 0), row < T)
    tl.store(OK + x, tl.load(K + row * KS + col, valid, 0), row < T)
    tl.store(OV + x, tl.load(V + row * VS + col, valid, 0), row < T)
    beta_row, beta_col = x // H, x % H
    tl.store(
        OB + x,
        tl.load(B + beta_row * BS + beta_col, (beta_row < T) & (beta_row < live), 0),
        beta_row < T,
    )
    if tl.program_id(0) == 0:
        tl.store(CHUNKS, 0)
        total = tl.full((), 0, CU.dtype.element_ty)
        for first in range(0, N, SEQ_BLOCK):
            slot = first + tl.arange(0, SEQ_BLOCK)
            begin = tl.load(CU + slot, slot < N, 0)
            end = tl.load(CU + slot + 1, slot < N, 0)
            counts = tl.where(slot < N, tl.cdiv(end - begin, 16), 0)
            tl.store(CHUNKS + slot + 1, total + tl.cumsum(counts), slot < N)
            total += tl.sum(counts)
        # The searches read chunk ends that other threads of this program stored.
        tl.debug_barrier()
        for base in range(tl.cdiv(MAX_CHUNKS, 128)):
            index = base * 128 + tl.arange(0, 128)
            # A chunk's sequence is the number of sequence ends at or before it.
            lo = tl.zeros((128,), tl.int32)
            hi = tl.zeros((128,), tl.int32) + N
            for _ in range(SEARCH_STEPS):
                active = lo < hi
                mid = (lo + hi) // 2
                chunk_end = tl.load(CHUNKS + 1 + mid, active, 0)
                lo = tl.where(active & (chunk_end <= index), mid + 1, lo)
                hi = tl.where(active & (chunk_end > index), mid, hi)
            tl.store(CHUNK_ROWS + index, tl.minimum(lo, N - 1), index < MAX_CHUNKS)


def prepare_capacity_scan(q, k, v, gate, beta, boundaries):
    """Return native contiguous inputs and a compact device chunk plan.

    Q/K/V/gate have dense features [1, capacity, H, 128]; beta is
    [1, capacity, H]. Live positive sequence lengths come from boundaries.
    Every input is copied with its capacity padding scrubbed; none is mutated.

    Returns (q, k, v, gate, beta, cu_chunks, chunk_to_seq). Gate is FP32 and
    both plan tensors are int32. The native
    cutedsl_kda_forward_with_prepared_plan adapter defines the chunk bounds.
    """
    _, tokens, heads, dim = q.shape
    sequences = boundaries.numel() - 1
    if not all(t.stride(-1) == 1 and t.stride(-2) == dim for t in (q, k, v, gate)):
        raise ValueError("KDA preparation requires dense feature dimensions")
    if beta.stride(-1) != 1:
        raise ValueError("KDA beta requires dense heads")
    oq = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    ok = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    ov = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    ob = torch.empty(beta.shape, dtype=beta.dtype, device=beta.device)
    og = torch.empty(gate.shape, dtype=torch.float32, device=gate.device)
    count = triton.cdiv(tokens, 16) + sequences - 1
    chunks = torch.empty(sequences + 1, dtype=torch.int32, device=q.device)
    chunk_rows = torch.empty(count, dtype=torch.int32, device=q.device)
    _prepare_capacity_scan_kernel[(triton.cdiv(tokens * heads * dim, 1024),)](
        q,
        k,
        v,
        gate,
        beta,
        oq,
        ok,
        ov,
        og,
        ob,
        boundaries,
        chunks,
        chunk_rows,
        T=tokens,
        H=heads,
        D=dim,
        N=sequences,
        QS=q.stride(1),
        KS=k.stride(1),
        VS=v.stride(1),
        GS=gate.stride(1),
        BS=beta.stride(1),
        MAX_CHUNKS=count,
        SEARCH_STEPS=sequences.bit_length(),
        SEQ_BLOCK=_SEQUENCE_BLOCK,
        BLOCK=1024,
    )
    return oq, ok, ov, og, ob, chunks, chunk_rows
