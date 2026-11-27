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

"""GFX950 Gluon GDN chunked prefill as a two-level parallel scan.

The chunk recurrence of a gated delta rule (FLA's
``chunk_gated_delta_rule_fwd_h``) is affine in the state: chunk ``c`` maps
the ``[K, V]`` state ``h`` to ``A_c h + b_c`` with

    A_c = exp(g_last) I - k^T diag(exp(g_last - g)) w     ([K, K])
    b_c = k^T diag(exp(g_last - g)) u                      ([K, V])

so a run of chunks (a *segment*) maps its start state ``S`` to ``M S + B``.
Each sequence is cut into segments of ``seg_chunks`` chunks (segments never
cross sequences) and scanned in three launches:

1. ``transfer``: every segment but a sequence's last, in parallel, runs the
   chunk recurrence on ``[M | B]`` from ``[I | 0]`` (the first segment
   starts ``B`` from the initial state, so its ``B`` is its end state).
2. ``carry``: per sequence, a short serial pass ``S_{j+1} = M_j S_j + B_j``
   gives every segment's start state.
3. ``gluon_gdn_chunk_prefill_gfx950``: every segment, in parallel, runs the
   chunk recurrence from its start state and stores its outputs (FLA's
   ``chunk_fwd_o`` fused, so per-chunk states never leave the chip), the
   state checkpoints inside it and, for a sequence's last segment, the final
   state.

A call no longer than one segment launches only step 3, which is then the
serial chunk scan; in longer calls, sequences of one segment skip steps 1 and
2. The chained ``M`` products round differently from a chunk-by-chunk scan,
so states and outputs of later segments differ from the serial scan by
rounding (see the kernel README for the tolerance).

Segments are numbered without a prefix sum over sequences: segment ``j`` of
sequence ``i`` takes slot ``bos_i // P + i + j`` (``P`` the segment length in
tokens), which is increasing and below ``T // P + N``. Launches cover all
slots; a gap slot exits at once.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

cdna4 = gl.amd.cdna4
async_copy = gl.amd.cdna4.async_copy

# Scan chunk (tokens); state checkpoints fall on chunk boundaries.
GDN_CHUNK_PREFILL_CHUNK = 64
# Head dimensions the kernels take (key and value).
GDN_CHUNK_PREFILL_HEAD_DIM = 128
_BT = GDN_CHUNK_PREFILL_CHUNK
# Warps of the transfer and output programs (one head's 128 state columns
# each), and of the carry programs (32 state rows each).
_SCAN_WARPS = 8
_CARRY_ROWS = 32
_CARRY_WARPS = 4
# Segment lengths in chunks: sequences of up to _SINGLE_SEGMENT_CHUNKS chunks
# run unsplit; split ones take _MIN to _MAX chunks per segment (a shorter one
# does not pay for its transfer work, a longer one makes serial chains long).
_SINGLE_SEGMENT_CHUNKS = 12
_MIN_SEGMENT_CHUNKS = 4
_MAX_SEGMENT_CHUNKS = 64
# MFMA layouts of the 8-warp scan programs. _TOK holds token-major products
# [64, 128] (waves own 16 rows x 64 columns); _ST holds 128 state columns
# transposed, [128, K] (waves own 64 columns x 32 keys).
_TOK = gl.constexpr(
    gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[4, 2]
    )
)
_ST = gl.constexpr(
    gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 4]
    )
)


def gdn_chunk_prefill_segment_chunks(
    total_tokens: int,
    num_seqs: int,
    num_value_heads: int,
    num_cus: int,
    checkpoint_every_n_tokens: int | None,
) -> int:
    """Segment length in chunks for a call (a runtime kernel argument).

    The output and transfer programs each hold a compute unit (one value head
    of one segment), so a call is cut into about ``num_cus //
    num_value_heads`` segments. When the average sequence is not much longer
    than that (or at most ``_SINGLE_SEGMENT_CHUNKS`` chunks), a segment is
    the average sequence instead: sequences then scan in parallel without
    the transfer and carry passes, and only longer ones split. With state
    checkpoints, a length within 25% of that choice that divides or is a
    multiple of the checkpoint interval is preferred, so checkpoints land on
    segment boundaries or evenly inside segments.

    Args:
        total_tokens: Tokens of the call (all sequences, with any padding).
        num_seqs: Sequences of the call.
        num_value_heads: Value heads (programs per segment).
        num_cus: Compute units of the device.
        checkpoint_every_n_tokens: State checkpoint interval, or ``None``.

    Returns:
        Chunks per segment.
    """
    chunks = max(1, -(-total_tokens // _BT))
    target = -(-chunks // max(1, num_cus // num_value_heads))
    average = -(-chunks // max(1, num_seqs))
    if average <= max(_SINGLE_SEGMENT_CHUNKS, target * 5 // 4):
        return average
    seg = max(_MIN_SEGMENT_CHUNKS, min(_MAX_SEGMENT_CHUNKS, target))
    if checkpoint_every_n_tokens is not None:
        every = checkpoint_every_n_tokens // _BT
        aligned = [
            n
            for n in range(-(-seg * 3 // 4), seg * 5 // 4 + 1)
            if _MIN_SEGMENT_CHUNKS <= n <= _MAX_SEGMENT_CHUNKS
            and (every % n == 0 or n % every == 0)
        ]
        if aligned:
            seg = min(aligned, key=lambda n: (abs(n - seg), n))
    return seg


def gdn_chunk_prefill_num_slots(
    total_tokens: int, num_seqs: int, seg_chunks: int
) -> int:
    """Segment slots of a call (the launches' slot dimension)."""
    return total_tokens // (seg_chunks * _BT) + num_seqs


def _io_bytes(args, *names):
    return sum(args[n].numel() * args[n].element_size() for n in names)


def _transfer_launch_metadata(grid, kernel, args):
    """Report the transfer pass's traffic and MFMA FLOPs to Proton (an upper
    bound: every slot counted as a whole non-final segment)."""
    tokens, heads = args["w"].shape[1], args["HV"]
    key_dim, value_dim = args["K"], args["V"]
    cols = key_dim + value_dim
    io = _io_bytes(args, "k", "w", "u", "g") + args["transfer"].numel() * 4
    return {
        "name": kernel.name,
        "bytes": io,
        "flops": tokens // _BT * heads * 2 * 2 * _BT * key_dim * cols,
    }


def _carry_launch_metadata(grid, kernel, args):
    """Report the carry pass's traffic and MFMA FLOPs to Proton."""
    key_dim, value_dim, heads = args["K"], args["V"], args["HV"]
    slots = args["transfer"].shape[0]
    return {
        "name": kernel.name,
        "bytes": slots * heads * (key_dim + 2 * value_dim) * key_dim * 4,
        "flops": slots * heads * 3 * 2 * value_dim * key_dim * key_dim,
    }


def _chunk_prefill_launch_metadata(grid, kernel, args):
    """Report the fused recurrence and output's traffic and FLOPs to Proton."""
    tokens, heads = args["w"].shape[1], args["HV"]
    key_dim, value_dim = args["K"], args["V"]
    io = _io_bytes(args, "q", "k", "w", "u", "g", "o", "ht")
    io += grid[1] * heads * value_dim * key_dim * 4  # start states
    # w h, q h, k^T v (state), q k^T, A v (per chunk and head).
    per_chunk = 3 * 2 * _BT * key_dim * value_dim
    per_chunk += 2 * _BT * _BT * key_dim + 2 * _BT * _BT * value_dim
    return {
        "name": kernel.name,
        "bytes": io,
        "flops": tokens // _BT * heads * per_chunk,
    }


@gluon.jit
def _gdn_segment(slot, cu_seqlens, num_seqs, seg_tokens):
    """The (sequence, segment) of ``slot``: ``(i_n, j, nseg, bos, T)``.

    Sequence ``i`` owns slots ``[f(i), f(i) + nseg_i)`` with ``f(i) = bos_i //
    seg_tokens + i``, increasing in ``i``; the sequence of ``slot`` is the
    last ``i`` with ``f(i) <= slot``, found by a binary search of scalar
    loads (the results stay wave-uniform, in SGPRs). ``j`` outside ``[0,
    nseg)`` marks a gap slot (``j < 0`` before the first sequence when
    ``cu_seqlens[0] > 0``). ``bos`` is int64.
    """
    lo = 0
    hi = num_seqs
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if gl.load(cu_seqlens + mid) // seg_tokens + mid <= slot:
            lo = mid
        else:
            hi = mid
    bos = gl.load(cu_seqlens + lo)
    T = gl.load(cu_seqlens + lo + 1) - bos
    j = slot - (bos // seg_tokens + lo)
    nseg = gl.cdiv(T, seg_tokens)
    return lo, j, nseg, bos.to(gl.int64), T


@gluon.jit
def _gdn_issue_tile(buf, base, t0, n_tok, rows, cols, ROW_STRIDE: gl.constexpr):
    """Copy rows ``[t0, t0 + 64)`` of a row-strided ``[*, cols]`` tile into
    ``buf`` with 16-byte asynchronous copies; rows from ``n_tok`` on read
    zeros (out-of-range buffer loads)."""
    offsets = gl.max_contiguous(
        gl.multiple_of((t0 + rows)[:, None] * ROW_STRIDE + cols[None, :], [1, 8]),
        [1, 8],
    )
    # A row mask, broadcast along the 16-byte vectors.
    async_copy.buffer_load_to_shared(
        buf, base, offsets, mask=((t0 + rows) < n_tok)[:, None]
    )


@gluon.jit
def _gdn_issue_gates(buf, base, t0, n_tok, slots, HV: gl.constexpr):
    """Copy the chunk's cumulative gates into ``buf[:64]`` and its last
    token's gate into every later slot (read back in several layouts)."""
    BT: gl.constexpr = 64
    last = gl.minimum(t0 + BT, n_tok) - 1
    rows = gl.where(slots < BT, t0 + slots, last)
    async_copy.buffer_load_to_shared(buf, base, rows * HV, mask=rows < n_tok)


@gluon.jit(
    launch_metadata=_transfer_launch_metadata,
    do_not_specialize=("num_seqs", "seg_chunks"),
)
def gluon_gdn_chunk_prefill_transfer_gfx950(
    k,
    w,
    u,
    g,
    h0,
    transfer,
    cu_seqlens,
    num_seqs,
    seg_chunks,
    H: gl.constexpr,
    HV: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
):
    """One (value head, slot): a segment's ``[M | B]``.

    Runs the chunk recurrence on ``M`` (from ``I``, with ``u = 0``) and on
    ``B`` (from 0, or from ``h0`` for a sequence's first segment, whose ``B``
    is then its end state and whose ``M`` goes unused) side by side, so the
    chunk's k and w are read once. Stores both transposed (FP32, K-last like
    the state pool) at ``transfer[slot, hv]`` (``[slots, HV, K + V, K]``:
    ``M^T`` then ``B^T``). A sequence's last segment exits.
    """
    BT: gl.constexpr = 64
    NUM_WARPS: gl.constexpr = 8
    LOG2E: gl.constexpr = 1.4426950408889634
    gl.static_assert(K == 128 and V == 128)
    tok: gl.constexpr = _TOK
    st: gl.constexpr = _ST
    tok_a: gl.constexpr = gl.DotOperandLayout(0, tok, 8)
    tok_b: gl.constexpr = gl.DotOperandLayout(1, tok, 8)
    st_a: gl.constexpr = gl.DotOperandLayout(0, st, 8)
    st_b: gl.constexpr = gl.DotOperandLayout(1, st, 8)
    copy: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [NUM_WARPS, 1], [1, 0])
    copy_g: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    G_SLOTS: gl.constexpr = 64 * NUM_WARPS
    # XOR-swizzled 16-byte vectors (a copy writes each wave's 1 KiB linearly,
    # so the swizzle permutes the copy sources instead).
    tile: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, order=[1, 0])
    flat: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0])

    hv = gl.program_id(0)
    slot = gl.program_id(1)
    seg_tokens = seg_chunks * BT
    i_n, j, nseg, bos, T = _gdn_segment(slot, cu_seqlens, num_seqs, seg_tokens)
    if (j >= 0) & (j + 1 < nseg):
        hk = hv // (HV // H)
        # A whole segment: no row is masked.
        tok0 = bos + j * seg_tokens
        k_seg = k + (tok0 * H + hk) * K
        w_seg = w + (tok0 * HV + hv) * K
        u_seg = u + (tok0 * HV + hv) * V
        g_seg = g + tok0 * HV + hv

        rows = gl.arange(0, BT, layout=gl.SliceLayout(1, copy))
        cols = gl.arange(0, K, layout=gl.SliceLayout(0, copy))
        g_slots = gl.arange(0, G_SLOTS, layout=copy_g)
        x_rows = gl.arange(0, K, layout=gl.SliceLayout(1, st))
        x_keys = gl.arange(0, K, layout=gl.SliceLayout(0, st))
        x_offsets = x_rows[:, None] * K + x_keys[None, :]

        k_bufs = gl.allocate_shared_memory(gl.bfloat16, [2, BT, K], tile)
        w_bufs = gl.allocate_shared_memory(gl.bfloat16, [2, BT, K], tile)
        u_bufs = gl.allocate_shared_memory(gl.bfloat16, [2, BT, V], tile)
        g_bufs = gl.allocate_shared_memory(gl.float32, [2, G_SLOTS], flat)
        # One half of the BF16 operand [M | B] at a time.
        smem_x = gl.allocate_shared_memory(gl.bfloat16, [K, K], tile)
        for s in gl.static_range(2):
            _gdn_issue_tile(
                k_bufs.index(s), k_seg, s * BT, seg_tokens, rows, cols, H * K
            )
            _gdn_issue_tile(
                w_bufs.index(s), w_seg, s * BT, seg_tokens, rows, cols, HV * K
            )
            _gdn_issue_tile(
                u_bufs.index(s), u_seg, s * BT, seg_tokens, rows, cols, HV * V
            )
            _gdn_issue_gates(g_bufs.index(s), g_seg, s * BT, seg_tokens, g_slots, HV)
            async_copy.commit_group()

        # M^T starts at I; B^T at 0, or h0^T for a first segment.
        xm = gl.where(x_rows[:, None] == x_keys[None, :], 1.0, 0.0)
        if j == 0:
            xb = gl.load(h0 + (i_n.to(gl.int64) * HV + hv) * (V * K) + x_offsets).to(
                gl.float32
            )
        else:
            xb = gl.zeros([V, K], gl.float32, layout=st)
        zeros = gl.zeros([BT, K], gl.float32, layout=tok)
        for c in range(seg_chunks):
            stage = c % 2
            k_buf = k_bufs.index(stage)
            w_buf = w_bufs.index(stage)
            u_buf = u_bufs.index(stage)
            g_buf = g_bufs.index(stage)
            async_copy.wait_group(1)
            w_a = w_buf.load(tok_a)
            smem_x.store(xm.to(gl.bfloat16))
            vm = -cdna4.mfma(w_a, smem_x.permute([1, 0]).load(tok_b), zeros)
            gl.barrier()
            smem_x.store(xb.to(gl.bfloat16))
            vb = u_buf.load(tok).to(gl.float32) - cdna4.mfma(
                w_a, smem_x.permute([1, 0]).load(tok_b), zeros
            )
            g_rows = g_buf.slice(0, BT).load(gl.SliceLayout(1, tok)) * LOG2E
            g_last_rows = g_buf.slice(BT, BT).load(gl.SliceLayout(1, tok)) * LOG2E
            g_last_keys = g_buf.slice(2 * BT, K).load(gl.SliceLayout(0, st)) * LOG2E
            # Cumulative gates do not increase: g_last - g <= 0.
            decay_rows = gl.exp2(g_last_rows - g_rows)[:, None]
            decay_keys = gl.exp2(g_last_keys)[None, :]
            # Every wave is past w(c) and u(c): their stages take the scaled
            # values of M and B.
            gl.barrier()
            w_buf.store((vm * decay_rows).to(gl.bfloat16))
            u_buf.store((vb * decay_rows).to(gl.bfloat16))
            gl.barrier()
            k_b = k_buf.load(st_b)
            xm = cdna4.mfma(w_buf.permute([1, 0]).load(st_a), k_b, xm * decay_keys)
            xb = cdna4.mfma(u_buf.permute([1, 0]).load(st_a), k_b, xb * decay_keys)
            # Refill this stage with chunk c + 2.
            gl.barrier()
            t2 = (c + 2) * BT
            _gdn_issue_tile(k_buf, k_seg, t2, seg_tokens, rows, cols, H * K)
            _gdn_issue_tile(w_buf, w_seg, t2, seg_tokens, rows, cols, HV * K)
            _gdn_issue_tile(u_buf, u_seg, t2, seg_tokens, rows, cols, HV * V)
            _gdn_issue_gates(g_buf, g_seg, t2, seg_tokens, g_slots, HV)
            async_copy.commit_group()
        async_copy.wait_group(0)
        x_base = transfer + (slot.to(gl.int64) * HV + hv) * ((K + V) * K)
        gl.store(x_base + x_offsets, xm)
        gl.store(x_base + K * K + x_offsets, xb)


@gluon.jit
def _gdn_bf16_split(x):
    """FP32 ``x`` as BF16 ``(hi, lo)`` with ``hi + lo`` ~ ``x`` to 16 bits."""
    hi = x.to(gl.bfloat16)
    return hi, (x - hi.to(gl.float32)).to(gl.bfloat16)


@gluon.jit(
    launch_metadata=_carry_launch_metadata,
    do_not_specialize=("seg_chunks",),
)
def gluon_gdn_chunk_prefill_carry_gfx950(
    transfer,
    starts,
    cu_seqlens,
    seg_chunks,
    HV: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BV: gl.constexpr,
):
    """One (V block, value head, sequence): each segment's start state.

    ``S_1 = B_0`` and ``S_{j+1} = M_j S_j + B_j``, in K-last form ``S^T M^T +
    B^T``; both FP32 operands enter the MFMAs as BF16 hi/lo pairs (three
    products), so the carry adds about FP32-level rounding only. ``S_j`` goes
    to ``starts[slot_j]`` (FP32 K-last). The next segment's ``M`` and ``B``
    load while the current product runs.
    """
    st: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4]
    )
    st_a: gl.constexpr = gl.DotOperandLayout(0, st, 8)
    st_b: gl.constexpr = gl.DotOperandLayout(1, st, 8)
    m_blocked: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    BT: gl.constexpr = 64

    i_v = gl.program_id(0)
    hv = gl.program_id(1)
    i_n = gl.program_id(2)
    seg_tokens = seg_chunks * BT
    bos = gl.load(cu_seqlens + i_n)
    T = gl.load(cu_seqlens + i_n + 1) - bos
    nseg = gl.cdiv(T, seg_tokens)
    first = bos // seg_tokens + i_n

    s_rows = gl.arange(0, BV, layout=gl.SliceLayout(1, st))
    s_keys = gl.arange(0, K, layout=gl.SliceLayout(0, st))
    m_rows = gl.arange(0, K, layout=gl.SliceLayout(1, m_blocked))
    m_keys = gl.arange(0, K, layout=gl.SliceLayout(0, m_blocked))
    b_offsets = (K + i_v * BV + s_rows)[:, None] * K + s_keys[None, :]
    s_offsets = (i_v * BV + s_rows)[:, None] * K + s_keys[None, :]
    m_offsets = m_rows[:, None] * K + m_keys[None, :]
    head = hv.to(gl.int64)
    stride = HV * (K + V) * K

    if nseg > 1:
        x_base = transfer + (first.to(gl.int64) * HV + head) * ((K + V) * K)
        s = gl.load(x_base + b_offsets)
        # Segment j's M and B (prefetched one step ahead).
        x_base += stride
        m = gl.load(x_base + m_offsets)
        b = gl.load(x_base + b_offsets)
        for jj in range(1, nseg):
            slot = (first + jj).to(gl.int64)
            gl.store(starts + (slot * HV + head) * (V * K) + s_offsets, s)
            if jj + 1 < nseg:
                m_hi, m_lo = _gdn_bf16_split(m)
                acc = b
                if jj + 2 < nseg:
                    x_base += stride
                    m = gl.load(x_base + m_offsets)
                    b = gl.load(x_base + b_offsets)
                s_hi, s_lo = _gdn_bf16_split(s)
                s_hi = gl.convert_layout(s_hi, st_a)
                m_hi = gl.convert_layout(m_hi, st_b)
                acc = cdna4.mfma(gl.convert_layout(s_lo, st_a), m_hi, acc)
                acc = cdna4.mfma(s_hi, gl.convert_layout(m_lo, st_b), acc)
                s = cdna4.mfma(s_hi, m_hi, acc)


@gluon.jit(
    launch_metadata=_chunk_prefill_launch_metadata,
    do_not_specialize=("num_seqs", "seg_chunks", "checkpoint_every_chunks"),
)
def gluon_gdn_chunk_prefill_gfx950(
    q,
    k,
    w,
    u,
    g,
    o,
    h0,
    starts,
    ht,
    cu_seqlens,
    num_seqs,
    seg_chunks,
    checkpoint_states,
    checkpoint_cu_starts,
    checkpoint_every_chunks,
    scale,
    H: gl.constexpr,
    HV: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    STORE_CHECKPOINTS: gl.constexpr,
):
    """One (value head, slot): a segment's chunk scan and outputs.

    q/k: ``[1, T, H, K]`` (normalized), w: ``[1, T, HV, K]``, u: ``[1, T, HV,
    V]`` (the WY form, ``u`` the corrected values before the state term), g:
    ``[1, T, HV]`` FP32 within-chunk cumulative log decay; all contiguous.
    o: ``[1, T, HV, V]``. h0: K-last ``[N, HV, V, K]`` (any float dtype), the
    start of each sequence's first segment; starts: FP32 K-last ``[slots, HV,
    V, K]`` from the carry pass for the others. ht and checkpoint_states:
    FP32 K-last. For chunk c, with the state ``h`` before it (``[K, V]``,
    used in BF16 by the products):

        v_new = u - w h
        o = (q h) * exp(g) * scale + ((q k^T) * D).bf16 v_new.bf16 * scale
        h = h * exp(g_last) + k^T (v_new * exp(g_last - g)).bf16

    with ``D[i, j] = exp(g_i - g_j)`` for ``j <= i`` (else 0), as FLA's
    ``chunk_gated_delta_rule_fwd_h`` + ``chunk_fwd_o``. The program keeps the
    head's state transposed (K-last, like the pool) in FP32 MFMA
    accumulators; LDS turns it into the BF16 operand of each chunk. One
    program covers all of V, so a chunk's q/k/w are fetched once per head.

    Chunk inputs stream into LDS with asynchronous copies: k/w/u/g in two
    stages (chunk c + 2 is issued once chunk c is consumed), q in one buffer
    refilled with chunk c + 1 once chunk c's products have read it. A
    chunk's output is stored after the next chunk's wait, so no store sits
    between a copy and its wait (gfx950 retires loads and stores in order on
    one counter).
    """
    BT: gl.constexpr = 64
    NUM_WARPS: gl.constexpr = 8
    LOG2E: gl.constexpr = 1.4426950408889634
    gl.static_assert(K == 128 and V == 128)
    tok: gl.constexpr = _TOK
    st: gl.constexpr = _ST
    tok_a: gl.constexpr = gl.DotOperandLayout(0, tok, 8)
    tok_b: gl.constexpr = gl.DotOperandLayout(1, tok, 8)
    st_a: gl.constexpr = gl.DotOperandLayout(0, st, 8)
    st_b: gl.constexpr = gl.DotOperandLayout(1, st, 8)
    copy: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [NUM_WARPS, 1], [1, 0])
    copy_g: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    G_SLOTS: gl.constexpr = 64 * NUM_WARPS
    tile: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, order=[1, 0])
    scores_tile: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 8, order=[1, 0])
    flat: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0])

    hv = gl.program_id(0)
    slot = gl.program_id(1)
    seg_tokens = seg_chunks * BT
    i_n, j, nseg, bos, T = _gdn_segment(slot, cu_seqlens, num_seqs, seg_tokens)
    if (j >= 0) & (j < nseg):
        hk = hv // (HV // H)
        t_begin = j * seg_tokens
        n_tok = gl.minimum(seg_tokens, T - t_begin)
        n_chunks = gl.cdiv(n_tok, BT)
        # Segment bases in int64; offsets within a segment stay int32.
        tok0 = bos + t_begin
        q_seg = q + (tok0 * H + hk) * K
        k_seg = k + (tok0 * H + hk) * K
        w_seg = w + (tok0 * HV + hv) * K
        u_seg = u + (tok0 * HV + hv) * V
        o_seg = o + (tok0 * HV + hv) * V
        g_seg = g + tok0 * HV + hv

        rows = gl.arange(0, BT, layout=gl.SliceLayout(1, copy))
        cols = gl.arange(0, K, layout=gl.SliceLayout(0, copy))
        g_slots = gl.arange(0, G_SLOTS, layout=copy_g)
        c_rows = gl.arange(0, BT, layout=gl.SliceLayout(1, tok))
        c_vals = gl.arange(0, V, layout=gl.SliceLayout(0, tok))
        c_cols = gl.arange(0, BT, layout=gl.SliceLayout(0, tok))
        s_rows = gl.arange(0, V, layout=gl.SliceLayout(1, st))
        s_keys = gl.arange(0, K, layout=gl.SliceLayout(0, st))
        state_offsets = s_rows[:, None] * K + s_keys[None, :]
        out_offsets = c_rows[:, None] * (HV * V) + c_vals[None, :]

        k_bufs = gl.allocate_shared_memory(gl.bfloat16, [2, BT, K], tile)
        w_bufs = gl.allocate_shared_memory(gl.bfloat16, [2, BT, K], tile)
        u_bufs = gl.allocate_shared_memory(gl.bfloat16, [2, BT, V], tile)
        g_bufs = gl.allocate_shared_memory(gl.float32, [2, G_SLOTS], flat)
        q_buf = gl.allocate_shared_memory(gl.bfloat16, [BT, K], tile)
        smem_h = gl.allocate_shared_memory(gl.bfloat16, [V, K], tile)
        smem_a = gl.allocate_shared_memory(gl.bfloat16, [BT, BT], scores_tile)

        # Groups: q(0); kwug(0); kwug(1); then per chunk c: q(c+1), kwug(c+2).
        _gdn_issue_tile(q_buf, q_seg, 0, n_tok, rows, cols, H * K)
        async_copy.commit_group()
        for s in gl.static_range(2):
            _gdn_issue_tile(k_bufs.index(s), k_seg, s * BT, n_tok, rows, cols, H * K)
            _gdn_issue_tile(w_bufs.index(s), w_seg, s * BT, n_tok, rows, cols, HV * K)
            _gdn_issue_tile(u_bufs.index(s), u_seg, s * BT, n_tok, rows, cols, HV * V)
            _gdn_issue_gates(g_bufs.index(s), g_seg, s * BT, n_tok, g_slots, HV)
            async_copy.commit_group()

        if j == 0:
            h = gl.load(h0 + (i_n.to(gl.int64) * HV + hv) * (V * K) + state_offsets).to(
                gl.float32
            )
        else:
            h = gl.load(
                starts + (slot.to(gl.int64) * HV + hv) * (V * K) + state_offsets
            )
        zeros = gl.zeros([BT, V], gl.float32, layout=tok)
        out = zeros
        ckpt_row = -1
        c_begin = j * seg_chunks
        for c in range(n_chunks):
            stage = c % 2
            k_buf = k_bufs.index(stage)
            w_buf = w_bufs.index(stage)
            u_buf = u_bufs.index(stage)
            g_buf = g_bufs.index(stage)
            # q(c) and kwug(c) have landed; kwug(c + 1) may be in flight.
            async_copy.wait_group(1)
            # The previous chunk's output and checkpoint, behind the wait.
            if c > 0:
                cdna4.buffer_store(
                    (out * scale).to(o_seg.dtype.element_ty),
                    o_seg,
                    out_offsets + (c - 1) * BT * (HV * V),
                    mask=((c - 1) * BT + c_rows < n_tok)[:, None],
                )
            if STORE_CHECKPOINTS:
                if ckpt_row >= 0:
                    gl.store(
                        checkpoint_states
                        + (ckpt_row.to(gl.int64) * HV + hv) * (V * K)
                        + state_offsets,
                        h,
                    )
            smem_h.store(h.to(gl.bfloat16))
            h_b = smem_h.permute([1, 0]).load(tok_b)
            # Gates pre-scaled by log2(e): exp(x) = exp2(x * log2(e)).
            g_rows = g_buf.slice(0, BT).load(gl.SliceLayout(1, tok)) * LOG2E
            g_last_rows = g_buf.slice(BT, BT).load(gl.SliceLayout(1, tok)) * LOG2E
            g_last_keys = g_buf.slice(2 * BT, K).load(gl.SliceLayout(0, st)) * LOG2E
            g_cols = g_buf.slice(0, BT).load(gl.SliceLayout(0, tok)) * LOG2E
            v_new = u_buf.load(tok).to(gl.float32) - cdna4.mfma(
                w_buf.load(tok_a), h_b, zeros
            )
            q_a = q_buf.load(tok_a)
            scores = cdna4.mfma(
                q_a,
                k_buf.permute([1, 0]).load(tok_b),
                gl.zeros([BT, BT], gl.float32, layout=tok),
            )
            out = cdna4.mfma(q_a, h_b, zeros) * gl.exp2(g_rows)[:, None]
            # Cumulative gates do not increase: g_i - g_j <= 0 for j <= i, and
            # g_last - g_i <= 0 (rows past the sequence have zero v_new).
            scores = gl.where(
                c_rows[:, None] >= c_cols[None, :],
                scores * gl.exp2(g_rows[:, None] - g_cols[None, :]),
                0.0,
            )
            vs = v_new * gl.exp2(g_last_rows - g_rows)[:, None]
            # Every wave is past q(c) and w(c): refill q, and w's and u's
            # stages take the scaled values and v_new.
            gl.barrier()
            _gdn_issue_tile(q_buf, q_seg, (c + 1) * BT, n_tok, rows, cols, H * K)
            async_copy.commit_group()
            w_buf.store(vs.to(gl.bfloat16))
            u_buf.store(v_new.to(gl.bfloat16))
            smem_a.store(scores.to(gl.bfloat16))
            gl.barrier()
            h = cdna4.mfma(
                w_buf.permute([1, 0]).load(st_a),
                k_buf.load(st_b),
                h * gl.exp2(g_last_keys)[None, :],
            )
            out = cdna4.mfma(smem_a.load(tok_a), u_buf.load(tok_b), out)
            if STORE_CHECKPOINTS:
                chunk = c_begin + c + 1
                ckpt_row = -1
                if (chunk % checkpoint_every_chunks == 0) & ((c + 1) * BT <= n_tok):
                    ckpt_row = gl.load(checkpoint_cu_starts + i_n) + (
                        chunk // checkpoint_every_chunks - 1
                    )
            # Refill this stage with chunk c + 2.
            gl.barrier()
            t2 = (c + 2) * BT
            _gdn_issue_tile(k_buf, k_seg, t2, n_tok, rows, cols, H * K)
            _gdn_issue_tile(w_buf, w_seg, t2, n_tok, rows, cols, HV * K)
            _gdn_issue_tile(u_buf, u_seg, t2, n_tok, rows, cols, HV * V)
            _gdn_issue_gates(g_buf, g_seg, t2, n_tok, g_slots, HV)
            async_copy.commit_group()

        async_copy.wait_group(0)
        cdna4.buffer_store(
            (out * scale).to(o_seg.dtype.element_ty),
            o_seg,
            out_offsets + (n_chunks - 1) * BT * (HV * V),
            mask=((n_chunks - 1) * BT + c_rows < n_tok)[:, None],
        )
        if STORE_CHECKPOINTS:
            if ckpt_row >= 0:
                gl.store(
                    checkpoint_states
                    + (ckpt_row.to(gl.int64) * HV + hv) * (V * K)
                    + state_offsets,
                    h,
                )
        if j + 1 == nseg:
            gl.store(ht + (i_n.to(gl.int64) * HV + hv) * (V * K) + state_offsets, h)


def launch_gluon_gdn_chunk_prefill_gfx950(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    *,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    checkpoint_states: torch.Tensor | None,
    checkpoint_cu_starts: torch.Tensor | None,
    checkpoint_every_n_tokens: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the chunk recurrence and output of a GDN prefill as a segment scan.

    Args:
        q, k: ``[1, T, H, 128]`` contiguous, L2-normalized.
        w, u: ``[1, T, HV, 128]`` contiguous WY terms (``recompute_w_u``).
        g: ``[1, T, HV]`` contiguous FP32 within-chunk cumulative log decay.
        initial_state: K-last ``[N, HV, V, K]`` contiguous, any float dtype.
        cu_seqlens: int32 ``[N + 1]`` sequence bounds on the device; the
            last may stop short of ``T`` (trailing padding is ignored).
        scale: Query scale.
        checkpoint_states: Optional FP32 K-last ``[rows, HV, V, K]`` buffer
            receiving the state after every ``checkpoint_every_n_tokens``
            whole tokens of sequence ``i`` at row ``checkpoint_cu_starts[i] +
            j``; the three checkpoint arguments go together.
        checkpoint_cu_starts: int32 ``[N + 1]`` per-sequence first rows.
        checkpoint_every_n_tokens: Positive multiple of 64.

    Returns:
        ``(o [1, T, HV, V] in q's dtype, final FP32 K-last [N, HV, V, K])``.
    """
    tokens, heads, key_dim = q.shape[1], q.shape[2], q.shape[3]
    value_heads, value_dim = u.shape[2], u.shape[3]
    sequences = cu_seqlens.numel() - 1
    if (
        key_dim != GDN_CHUNK_PREFILL_HEAD_DIM
        or value_dim != GDN_CHUNK_PREFILL_HEAD_DIM
        or value_heads % heads
    ):
        raise ValueError(f"unsupported GDN geometry K={key_dim} V={value_dim}")
    for name, tensor in (("q", q), ("k", k), ("w", w), ("u", u), ("g", g)):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if cu_seqlens.dtype != torch.int32:
        raise ValueError("cu_seqlens must be int32")
    if (
        initial_state.shape != (sequences, value_heads, value_dim, key_dim)
        or not initial_state.is_contiguous()
    ):
        raise ValueError("initial_state must be a contiguous K-last [N, HV, V, K]")
    store_checkpoints = checkpoint_states is not None
    if store_checkpoints != (checkpoint_cu_starts is not None) or store_checkpoints != (
        checkpoint_every_n_tokens is not None
    ):
        raise ValueError("the checkpoint arguments go together")
    if store_checkpoints and (
        checkpoint_every_n_tokens <= 0
        or checkpoint_every_n_tokens % _BT
        or checkpoint_states.dtype != torch.float32
        or not checkpoint_states.is_contiguous()
        or checkpoint_states.shape[1:] != (value_heads, value_dim, key_dim)
    ):
        raise ValueError(
            f"state checkpoints need a positive multiple of {_BT} tokens and a "
            "contiguous FP32 K-last [rows, HV, V, K] buffer"
        )
    out = torch.empty_like(u, dtype=q.dtype)
    final_state = torch.empty(
        sequences, value_heads, value_dim, key_dim, device=q.device, dtype=torch.float32
    )
    if sequences == 0:
        return out, final_state
    num_cus = torch.cuda.get_device_properties(q.device).multi_processor_count
    seg_chunks = gdn_chunk_prefill_segment_chunks(
        tokens, sequences, value_heads, num_cus, checkpoint_every_n_tokens
    )
    slots = gdn_chunk_prefill_num_slots(tokens, sequences, seg_chunks)
    # One segment per sequence needs no transfer or carry (no sequence is
    # longer than the call).
    if tokens > seg_chunks * _BT:
        # Segment transfers [M^T; B^T] and start states, FP32 K-last per slot.
        # Rows of slots no pass writes stay unread.
        transfer = torch.empty(
            slots,
            value_heads,
            key_dim + value_dim,
            key_dim,
            device=q.device,
            dtype=torch.float32,
        )
        starts = torch.empty(
            slots, value_heads, value_dim, key_dim, device=q.device, dtype=torch.float32
        )
        gluon_gdn_chunk_prefill_transfer_gfx950[(value_heads, slots)](
            k,
            w,
            u,
            g,
            initial_state,
            transfer,
            cu_seqlens,
            sequences,
            seg_chunks,
            H=heads,
            HV=value_heads,
            K=key_dim,
            V=value_dim,
            num_warps=_SCAN_WARPS,
        )
        gluon_gdn_chunk_prefill_carry_gfx950[
            (value_dim // _CARRY_ROWS, value_heads, sequences)
        ](
            transfer,
            starts,
            cu_seqlens,
            seg_chunks,
            HV=value_heads,
            K=key_dim,
            V=value_dim,
            BV=_CARRY_ROWS,
            num_warps=_CARRY_WARPS,
        )
    else:
        # Every sequence starts from initial_state; starts is never read.
        starts = final_state
    gluon_gdn_chunk_prefill_gfx950[(value_heads, slots)](
        q,
        k,
        w,
        u,
        g,
        out,
        initial_state,
        starts,
        final_state,
        cu_seqlens,
        sequences,
        seg_chunks,
        final_state if checkpoint_states is None else checkpoint_states,
        cu_seqlens if checkpoint_cu_starts is None else checkpoint_cu_starts,
        1 if checkpoint_every_n_tokens is None else checkpoint_every_n_tokens // _BT,
        scale,
        H=heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        STORE_CHECKPOINTS=store_checkpoints,
        num_warps=_SCAN_WARPS,
    )
    return out, final_state
