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

"""GFX950 Gluon GDN decode / MTP-verify recurrence over a K-last state pool."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton

cdna4 = gl.amd.cdna4

# Where each token's post-update state goes (STEP_MODE).
GDN_DECODE_STEPS_NONE = 0
# Row ``step_indices[n, t]`` of ``step_states``.
GDN_DECODE_STEPS_INDEXED = 1
# Row ``n * T + t`` of a dense ``step_states`` buffer.
GDN_DECODE_STEPS_DENSE = 2

# Tokens of one launch: the WY form solves all of them at once.
GDN_DECODE_MAX_TOKENS = 8
# Head dims the launcher accepts; the registrations declare the same sets so
# selection routes other shapes to another solution.
GDN_DECODE_KEY_DIM = 128
GDN_DECODE_VALUE_DIMS = frozenset(range(16, 257, 16))
GDN_DECODE_STATE_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32})
# Waves per (request, value head), by state element bytes. A BF16 head
# (128 VGPRs per lane in one wave) stays in one wave, so the T x T algebra
# runs once per head; an FP32 head would take all 256 VGPRs of one wave, so
# four waves split its rows (each repeats the small algebra).
_NUM_WARPS_BY_STATE_BYTES = {2: 1, 4: 4}


def _gdn_decode_launch_metadata(grid, kernel, args):
    """Report the recurrence's state traffic and FLOPs to Proton."""
    q, v = args["q"], args["v"]
    batch, tokens = q.shape[0], args["T"]
    heads, value_dim, key_dim = args["HV"], args["V"], args["K"]
    state_bytes = heads * value_dim * key_dim * args["state_in"].element_size()
    writes = (tokens if args["STEP_MODE"] else 0) + int(args["WRITE_FINAL"])
    activations = (
        2 * batch * tokens * args["H"] * key_dim * q.element_size()
        + 2 * batch * tokens * heads * value_dim * v.element_size()
    )
    return {
        "name": kernel.name,
        # S0 times T effective queries (multiply-add), as the MFMAs compute it.
        "flops": 2 * batch * tokens * heads * value_dim * key_dim,
        "bytes": batch * (1 + writes) * state_bytes + activations,
    }


@gluon.jit
def _gdn_bf16_parts(x):
    """``x`` as a tuple of BF16 terms that sum to it.

    BF16 stays as is, FP16 (11 significant bits) takes two terms and FP32
    (24 bits) three, so the MFMA products of the terms, summed in FP32, keep
    the operand's full precision.
    """
    if x.dtype == gl.bfloat16:
        parts = (x,)
    else:
        PARTS: gl.constexpr = 2 if x.dtype == gl.float16 else 3
        parts = ()
        rest = x.to(gl.float32)
        for _ in gl.static_range(PARTS):
            part = rest.to(gl.bfloat16)
            parts += (part,)
            rest = rest - part.to(gl.float32)
    return parts


@gluon.jit
def _gdn_exact_mfma(a_parts, b_parts, acc):
    """``acc + sum_ij a_i b_j`` over the BF16 terms of both operands."""
    for i in gl.static_range(len(a_parts)):
        for j in gl.static_range(len(b_parts)):
            acc = cdna4.mfma(a_parts[i], b_parts[j], acc)
    return acc


@gluon.jit
def _gdn_small_mm(x, y, L3: gl.constexpr, out_layout: gl.constexpr):
    """``x @ y`` of tiny ``[M, I] x [I, N]`` matrices as a rank-3 broadcast sum."""
    x3 = gl.expand_dims(gl.convert_layout(x, gl.SliceLayout(2, L3)), 2)
    y3 = gl.expand_dims(gl.convert_layout(y, gl.SliceLayout(0, L3)), 0)
    return gl.convert_layout(gl.sum(x3 * y3, axis=1), out_layout)


@gluon.jit
def _gdn_token_rows(q_head, k_head, T, q_token_stride, k_token_stride,rows, keys, TB: gl.constexpr, ROWS_FIRST: gl.constexpr):  # fmt: skip
    """Loads of raw rows ``X = [q_0 .. q_{TB-1}; k_0 .. k_{TB-1}; 0]``.

    ``rows``/``keys`` are the 1-D index vectors of a 2-D layout, rows on dim 0
    when ROWS_FIRST else on dim 1.
    """
    is_q = rows < TB
    token = gl.where(is_q, rows, rows - TB)
    live_q = is_q & (token < T)
    live_k = (~is_q) & (token < T) & (rows < 2 * TB)
    if ROWS_FIRST:
        q_offsets = token[:, None] * q_token_stride + keys[None, :]
        k_offsets = token[:, None] * k_token_stride + keys[None, :]
        q_mask = live_q[:, None] & (keys >= 0)[None, :]
        k_mask = live_k[:, None] & (keys >= 0)[None, :]
    else:
        q_offsets = token[None, :] * q_token_stride + keys[:, None]
        k_offsets = token[None, :] * k_token_stride + keys[:, None]
        q_mask = live_q[None, :] & (keys >= 0)[:, None]
        k_mask = live_k[None, :] & (keys >= 0)[:, None]
    return (
        cdna4.buffer_load(q_head, q_offsets, mask=q_mask),
        cdna4.buffer_load(k_head, k_offsets, mask=k_mask),
    )


@gluon.jit
def _gdn_merge_rows(rows):
    """Merge the q and k parts of :func:`_gdn_token_rows`.

    Masked buffer lanes read zero bits, so OR-ing the bits selects q or k.
    """
    q_rows, k_rows = rows
    merged = q_rows.to(gl.int16, bitcast=True) | k_rows.to(gl.int16, bitcast=True)
    return merged.to(q_rows.dtype, bitcast=True)


@gluon.jit
def _gdn_gate_logits(a, b, T, a_token_stride, b_token_stride,TB: gl.constexpr, layout: gl.constexpr):  # fmt: skip
    """Load the per-token decay and beta logits ``[TB]`` in ``layout``."""
    tokens = gl.arange(0, TB, layout=layout)
    live = tokens < T
    a_raw = cdna4.buffer_load(a, tokens * a_token_stride, mask=live)
    b_raw = cdna4.buffer_load(b, tokens * b_token_stride, mask=live)
    return a_raw, b_raw, live


@gluon.jit
def _gdn_gate_values(a_raw, b_raw, live, neg_a, bias):
    """Per-token log2 decay and beta from the logits (0 past T).

    The decay is ``exp(-exp(A_log) softplus(a + dt_bias))``, returned as its
    base-2 logarithm so products of decays are single ``exp2`` calls.
    """
    LOG2E: gl.constexpr = 1.4426950408889634
    x = a_raw.to(gl.float32) + bias
    # softplus(x) * log2(e) = log2(1 + e^x), linear past 20 like torch.
    softplus2 = gl.where(x <= 20.0, gl.log2(1.0 + gl.exp2(x * LOG2E)), x * LOG2E)
    log2_decay = gl.where(live, neg_a * softplus2, 0.0)
    beta = gl.where(live, 1.0 / (1.0 + gl.exp2(b_raw.to(gl.float32) * -LOG2E)), 0.0)
    return log2_decay, beta


@gluon.jit
def _gdn_decode_store_state(state, pool, row, row_stride, head_offset, tile_offsets):
    """Store a ``[V, K]`` state tile to one pool row; a negative row skips it."""
    cdna4.buffer_store(
        state.to(pool.dtype.element_ty),
        pool + gl.maximum(row, 0) * row_stride + head_offset,
        tile_offsets,
        mask=(row >= 0) & (tile_offsets >= 0),
    )


@gluon.jit
def _gdn_load_rows(
    base, token_stride, T, TB: gl.constexpr, N: gl.constexpr, layout: gl.constexpr
):
    """``[TB, N]`` FP32 rows of tokens ``0 .. T - 1`` (zeros past T)."""
    tokens = gl.arange(0, TB, layout=gl.SliceLayout(1, layout))
    cols = gl.arange(0, N, layout=gl.SliceLayout(0, layout))
    return cdna4.buffer_load(
        base,
        tokens[:, None] * token_stride + cols[None, :],
        mask=(tokens < T)[:, None] & (cols >= 0)[None, :],
    ).to(gl.float32)


@gluon.jit
def _gdn_decode_write_states(state_raw, k_head, v_head, a_head, b_head, neg_a, bias, seq, T, k_token_stride, v_token_stride, a_token_stride, b_token_stride, state_in, final_indices, step_states, step_indices, state_in_row_stride, step_states_row_stride, head_offset, K: gl.constexpr, V: gl.constexpr, TB: gl.constexpr, NUM_WARPS: gl.constexpr, USE_QK_L2NORM: gl.constexpr, STEP_MODE: gl.constexpr, WRITE_FINAL: gl.constexpr,):  # fmt: skip
    """Run the token recurrence on the FP32 state to store the requested states.

    Only state destinations need this: the outputs come from the projected
    form. Each lane holds rows by 16-byte K runs, so stores fill whole lines.
    """
    KEYS_PER_ACCESS: gl.constexpr = 128 // state_in.dtype.element_ty.primitive_bitwidth
    state_layout: gl.constexpr = gl.BlockedLayout(
        [V // (8 * NUM_WARPS), KEYS_PER_ACCESS], [8, 8], [NUM_WARPS, 1], [1, 0]
    )
    row_layout: gl.constexpr = gl.SliceLayout(1, state_layout)
    key_layout: gl.constexpr = gl.SliceLayout(0, state_layout)
    token_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8], [4, 16], [NUM_WARPS, 1], [1, 0]
    )
    token_value_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [4, 16], [NUM_WARPS, 1], [1, 0]
    )
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    row_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0])
    smem_k = gl.allocate_shared_memory(gl.float32, [TB, K], smem_layout)
    smem_v = gl.allocate_shared_memory(gl.float32, [TB, V], smem_layout)
    smem_beta = gl.allocate_shared_memory(gl.float32, [TB, V], smem_layout)
    smem_decay = gl.allocate_shared_memory(gl.float32, [TB, V], smem_layout)
    rows_k = smem_k.reinterpret(layout=row_smem_layout)
    rows_v = smem_v.reinterpret(layout=row_smem_layout)
    rows_beta = smem_beta.reinterpret(layout=row_smem_layout)
    rows_decay = smem_decay.reinterpret(layout=row_smem_layout)

    k_rows = _gdn_load_rows(k_head, k_token_stride, T, TB, K, token_layout)
    if USE_QK_L2NORM:
        k_rows = k_rows * gl.rsqrt(gl.sum(k_rows * k_rows, axis=1) + 1e-6)[:, None]
    a_raw, b_raw, live = _gdn_gate_logits(
        a_head, b_head, T, a_token_stride, b_token_stride, TB,
        gl.SliceLayout(1, token_value_layout),
    )  # fmt: skip
    log_decay, beta = _gdn_gate_values(a_raw, b_raw, live, neg_a, bias)
    v_rows = _gdn_load_rows(v_head, v_token_stride, T, TB, V, token_value_layout)
    zeros_v = gl.zeros([TB, V], gl.float32, layout=token_value_layout)
    smem_k.store(k_rows)
    smem_v.store(v_rows * beta[:, None])
    smem_beta.store(zeros_v + beta[:, None])
    smem_decay.store(zeros_v + gl.exp2(log_decay)[:, None])

    value_offsets = gl.arange(0, V, layout=row_layout)
    key_offsets = gl.arange(0, K, layout=key_layout)
    tile_offsets = value_offsets[:, None] * K + key_offsets[None, :]
    state = gl.convert_layout(state_raw, state_layout).to(gl.float32)
    for t in range(0, T):
        k_t = rows_k.index(t).load(key_layout)
        beta_v = rows_v.index(t).load(row_layout)
        beta_t = rows_beta.index(t).load(row_layout)
        decay = rows_decay.index(t).load(row_layout)
        state = state * decay[:, None]
        prediction = gl.sum(state * k_t[None, :], axis=1)
        state = state + (beta_v - beta_t * prediction)[:, None] * k_t[None, :]
        if STEP_MODE == 1:
            step_row = gl.load(step_indices + seq * T + t).to(gl.int64)
            _gdn_decode_store_state(
                state, step_states, step_row, step_states_row_stride,
                head_offset, tile_offsets,
            )  # fmt: skip
        elif STEP_MODE == 2:
            _gdn_decode_store_state(
                state, step_states, seq.to(gl.int64) * T + t,
                step_states_row_stride, head_offset, tile_offsets,
            )  # fmt: skip
    if WRITE_FINAL:
        final_row = gl.load(final_indices + seq).to(gl.int64)
        _gdn_decode_store_state(
            state, state_in, final_row, state_in_row_stride, head_offset, tile_offsets
        )


@gluon.jit(launch_metadata=_gdn_decode_launch_metadata, do_not_specialize=("T",))
def gluon_gdn_decode_gfx950(
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    o,
    state_in,
    read_indices,
    final_indices,
    step_states,
    step_indices,
    scale,
    T,
    q_batch_stride,
    q_token_stride,
    k_batch_stride,
    k_token_stride,
    v_batch_stride,
    v_token_stride,
    a_batch_stride,
    a_token_stride,
    a_head_stride,
    b_batch_stride,
    b_token_stride,
    b_head_stride,
    state_in_row_stride,
    step_states_row_stride,
    norm_gate,
    norm_weight,
    norm_eps,
    norm_gate_row_stride,
    H: gl.constexpr,
    HV: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    STEP_MODE: gl.constexpr,
    WRITE_FINAL: gl.constexpr,
    USE_QK_L2NORM: gl.constexpr,
    TB: gl.constexpr,
    LOG2_TB: gl.constexpr,
    OUTPUT_NORM: gl.constexpr,
):
    """Sigmoid-gated delta-rule recurrence of T <= TB tokens for one head.

    q/k: ``[B, T, H, K]``; v: ``[B, T, HV, V]``; a/b: ``[B, T, HV]``; o:
    contiguous ``[B, T, HV, V]``. ``state_in`` and ``step_states`` are K-last
    ``[rows, HV, V, K]`` pools (they may alias). Request ``n`` reads row
    ``read_indices[n]`` (negative: zero state), writes each token's state per
    STEP_MODE and, with WRITE_FINAL, the last state to ``state_in`` row
    ``final_indices[n]``; negative destination rows are skipped. With
    OUTPUT_NORM, each head's output is ``rmsnorm(o) * norm_weight *
    sigmoid(gate)`` over V, the gate row of token ``n * T + t`` starting
    ``norm_gate_row_stride`` apart with heads contiguous.

    The outputs use the chunk (WY) form of the recurrence. With normalized
    keys/queries, cumulative decay ``G_t``, ``D[t, i] = G_t / G_i`` (i <= t),
    ``A[t, i] = beta_t D[t, i] k_t.k_i`` (i < t) and
    ``N = (D * q k^T) (I + A)^-1``, token t's output for state row v is
    ``G_t (S0 q_t)_v + sum_i N[t, i] beta_i (v_i - G_i (S0 k_i)_v)``.
    So the state is read once, straight into an MFMA operand, and multiplied
    by the raw BF16 q/k rows (exact products, FP32 sums) in one
    ``[V, K] x [K, 2T]`` product; the same rows give the Gram matrix of the
    T x T algebra. The rest is O(V T^2) per head on the vector units.
    """
    NCOL: gl.constexpr = 16
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=False,
        warps_per_cta=[NUM_WARPS, 1],
    )  # fmt: skip
    a_layout: gl.constexpr = gl.DotOperandLayout(0, mfma_layout, 8)
    b_layout: gl.constexpr = gl.DotOperandLayout(1, mfma_layout, 8)
    # Per-row algebra: rows over lanes, the (t, i) token pairs in registers;
    # its dim-0 slice is the T x T algebra, in registers of every lane.
    L3: gl.constexpr = gl.BlockedLayout(
        [1, TB, TB], [64, 1, 1], [NUM_WARPS, 1, 1], [2, 1, 0]
    )
    rows_t: gl.constexpr = gl.SliceLayout(2, L3)  # [V, t]
    rows_i: gl.constexpr = gl.SliceLayout(1, L3)  # [V, i]
    pairs: gl.constexpr = gl.SliceLayout(0, L3)  # [t, i]
    by_t: gl.constexpr = gl.SliceLayout(1, pairs)
    by_i: gl.constexpr = gl.SliceLayout(0, pairs)
    # Products of T x T matrices, all in registers.
    L3_pairs: gl.constexpr = gl.BlockedLayout(
        [TB, TB, TB], [64, 1, 1], [NUM_WARPS, 1, 1], [2, 1, 0]
    )

    sequence_head = gl.program_id(0)
    seq = sequence_head // HV
    head_v = sequence_head % HV
    head_k = head_v // (HV // H)

    read_idx = gl.load(read_indices + seq).to(gl.int64)
    seq64 = seq.to(gl.int64)
    q_head = q + seq64 * q_batch_stride + head_k * K
    k_head = k + seq64 * k_batch_stride + head_k * K
    v_head = v + seq64 * v_batch_stride + head_v * V
    a_head = a + seq64 * a_batch_stride + head_v * a_head_stride
    b_head = b + seq64 * b_batch_stride + head_v * b_head_stride

    # Activations first: their latency overlaps the state's.
    x_a = _gdn_token_rows(
        q_head, k_head, T, q_token_stride, k_token_stride,
        gl.arange(0, NCOL, layout=gl.SliceLayout(1, a_layout)),
        gl.arange(0, K, layout=gl.SliceLayout(0, a_layout)), TB, True,
    )  # fmt: skip
    x_b = _gdn_token_rows(
        q_head, k_head, T, q_token_stride, k_token_stride,
        gl.arange(0, NCOL, layout=gl.SliceLayout(0, b_layout)),
        gl.arange(0, K, layout=gl.SliceLayout(1, b_layout)), TB, False,
    )  # fmt: skip
    a_raw, b_raw, live = _gdn_gate_logits(
        a_head, b_head, T, a_token_stride, b_token_stride, TB, by_t
    )
    value_rows = gl.arange(0, V, layout=gl.SliceLayout(1, rows_i))
    value_tokens = gl.arange(0, TB, layout=gl.SliceLayout(0, rows_i))
    v_tokens = cdna4.buffer_load(
        v_head,
        value_rows[:, None] + value_tokens[None, :] * v_token_stride,
        mask=(value_rows >= 0)[:, None] & (value_tokens < T)[None, :],
    ).to(gl.float32)
    out_rows = gl.arange(0, V, layout=gl.SliceLayout(1, rows_t))
    out_tokens = gl.arange(0, TB, layout=gl.SliceLayout(0, rows_t))
    out_mask = (out_tokens < T)[None, :] & (out_rows >= 0)[:, None]

    # Keep every activation load ahead of the state's: loads complete in
    # order, so the algebra waits on them alone.
    gl.amd.hint.sched_barrier()
    state_rows = gl.arange(0, V, layout=gl.SliceLayout(1, a_layout))
    keys = gl.arange(0, K, layout=gl.SliceLayout(0, a_layout))
    tile_offsets = state_rows[:, None] * K + keys[None, :]
    head_offset = head_v * (V * K)
    # No `other`: masked buffer lanes read zero, without a select that would
    # wait on the load right here.
    state_raw = cdna4.buffer_load(
        state_in + gl.maximum(read_idx, 0) * state_in_row_stride + head_offset,
        tile_offsets,
        mask=(read_idx >= 0) & (tile_offsets >= 0),
    )
    gl.amd.hint.sched_barrier()

    neg_a = -gl.exp(gl.load(A_log + head_v).to(gl.float32))
    bias = gl.load(dt_bias + head_v).to(gl.float32)
    log_decay_t, beta_t = _gdn_gate_values(a_raw, b_raw, live, neg_a, bias)
    # Every lane holds all T gates, so re-indexing them moves no data.
    log_decay_i = gl.convert_layout(log_decay_t, by_i)

    x_a = _gdn_bf16_parts(_gdn_merge_rows(x_a))
    x_b = _gdn_bf16_parts(_gdn_merge_rows(x_b))
    # Gram matrix of the raw rows; LDS regroups its q/k blocks.
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    smem_gram = gl.allocate_shared_memory(gl.float32, [NCOL, NCOL], smem_layout)
    smem_gram.store(
        _gdn_exact_mfma(
            x_a, x_b, gl.zeros([NCOL, NCOL], gl.float32, layout=mfma_layout)
        )
    )
    qq = smem_gram.slice(0, TB).slice(0, TB, dim=1).load(pairs)
    qk = smem_gram.slice(0, TB).slice(TB, TB, dim=1).load(pairs)
    kk = smem_gram.slice(TB, TB).slice(TB, TB, dim=1).load(pairs)

    tt = gl.arange(0, TB, layout=by_t)[:, None]
    ii = gl.arange(0, TB, layout=by_i)[None, :]
    query_scale_t = gl.full([TB], scale, gl.float32, layout=by_t)
    key_scale_t = gl.full([TB], 1.0, gl.float32, layout=by_t)
    key_scale_i = gl.full([TB], 1.0, gl.float32, layout=by_i)
    if USE_QK_L2NORM:
        query_scale_t = query_scale_t * gl.rsqrt(
            gl.sum(gl.where(ii == tt, qq, 0.0), axis=1) + 1e-6
        )
        key_scale_t = gl.rsqrt(gl.sum(gl.where(ii == tt, kk, 0.0), axis=1) + 1e-6)
        key_scale_i = gl.rsqrt(gl.sum(gl.where(ii == tt, kk, 0.0), axis=0) + 1e-6)
    cum_t = gl.sum(gl.where(ii <= tt, log_decay_i[None, :], 0.0), axis=1)
    cum_i = gl.sum(gl.where(tt <= ii, log_decay_t[:, None], 0.0), axis=0)
    ratio = gl.where(ii <= tt, gl.exp2(cum_t[:, None] - cum_i[None, :]), 0.0)
    lower = gl.where(
        ii < tt,
        (beta_t * key_scale_t)[:, None] * ratio * kk * key_scale_i[None, :],
        0.0,
    )
    eye = gl.where(ii == tt, 1.0, 0.0)
    # A is strictly lower triangular, so (I + A)^-1 = (I - A)(I + A^2)(I + A^4)...
    inverse = eye - lower
    power = lower
    for _ in gl.static_range(1, LOG2_TB):
        power = _gdn_small_mm(power, power, L3_pairs, pairs)
        inverse = _gdn_small_mm(inverse, eye + power, L3_pairs, pairs)
    mix = _gdn_small_mm(
        query_scale_t[:, None] * ratio * qk * key_scale_i[None, :],
        inverse,
        L3_pairs,
        pairs,
    )

    # The algebra needs no state. Parking its results in LDS keeps the
    # compiler from sinking it past the state wait: it runs while the state
    # streams in.
    smem_mix = gl.allocate_shared_memory(gl.float32, [TB, TB], smem_layout)
    smem_vectors = gl.allocate_shared_memory(gl.float32, [4, TB], smem_layout)
    vectors = smem_vectors.reinterpret(
        layout=gl.SwizzledSharedLayout(1, 1, 1, order=[0])
    )
    smem_mix.store(mix)
    decay = gl.exp2(cum_t)
    vectors.index(0).store(decay * query_scale_t)
    vectors.index(1).store(decay * key_scale_t)
    vectors.index(2).store(beta_t)
    gl.amd.hint.sched_barrier()
    if OUTPUT_NORM:
        # Issued behind the state, not with the activations: the gate would
        # hold registers through the algebra. It lands while the state does.
        gate_raw = cdna4.buffer_load(
            norm_gate + seq64 * T * norm_gate_row_stride + head_v * V,
            out_tokens[None, :] * norm_gate_row_stride + out_rows[:, None],
            mask=out_mask,
        )
        weight_raw = gl.load(norm_weight + out_rows)

    # Y = S0 [q | k]^T: the raw projections of every token, [V, 16].
    smem_y = gl.allocate_shared_memory(gl.float32, [V, NCOL], smem_layout)
    smem_y.store(
        _gdn_exact_mfma(
            _gdn_bf16_parts(state_raw), x_b,
            gl.zeros([V, NCOL], gl.float32, layout=mfma_layout),
        )  # fmt: skip
    )
    proj_q = smem_y.slice(0, TB, dim=1).load(rows_t)
    proj_k = smem_y.slice(TB, TB, dim=1).load(rows_i)
    decay_t = vectors.index(0).load(gl.SliceLayout(0, rows_t))
    decay_i = vectors.index(1).load(gl.SliceLayout(0, rows_i))
    beta_rows = vectors.index(2).load(gl.SliceLayout(0, rows_i))
    mix = smem_mix.load(pairs)
    # u_i = beta_i (v_i - G_i S0 k_i) per state row.
    update = beta_rows[None, :] * (v_tokens - decay_i[None, :] * proj_k)
    out = decay_t[None, :] * proj_q + gl.sum(
        gl.expand_dims(update, 1) * gl.expand_dims(mix, 0), axis=2
    )

    if OUTPUT_NORM:
        # The head is within the wave, so the RMS over V is a lane reduction.
        LOG2E: gl.constexpr = 1.4426950408889634
        inv_rms = gl.rsqrt(gl.sum(out * out, axis=0) / V + norm_eps)
        gate = 1.0 / (1.0 + gl.exp2(gate_raw.to(gl.float32) * -LOG2E))
        out = out * inv_rms[None, :] * weight_raw.to(gl.float32)[:, None] * gate
    gl.store(
        o + (seq64 * T * HV + head_v) * V
        + (out_tokens[None, :] * (HV * V) + out_rows[:, None]),
        out.to(o.dtype.element_ty),
        mask=out_mask,
    )  # fmt: skip

    if STEP_MODE != 0 or WRITE_FINAL:
        _gdn_decode_write_states(
            state_raw, k_head, v_head, a_head, b_head, neg_a, bias, seq, T,
            k_token_stride, v_token_stride, a_token_stride, b_token_stride,
            state_in, final_indices, step_states, step_indices,
            state_in_row_stride, step_states_row_stride, head_offset,
            K, V, TB, NUM_WARPS, USE_QK_L2NORM, STEP_MODE, WRITE_FINAL,
        )  # fmt: skip


def launch_gluon_gdn_decode_gfx950(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state: torch.Tensor,
    read_indices: torch.Tensor,
    final_indices: torch.Tensor | None,
    step_states: torch.Tensor | None,
    step_indices: torch.Tensor | None,
    scale: float,
    use_qk_l2norm: bool,
    norm_gate: torch.Tensor | None,
    norm_weight: torch.Tensor | None,
    norm_eps: float | None,
) -> torch.Tensor:
    """Run the gfx950 GDN recurrence of ``T = q.shape[1]`` tokens per request.

    Args:
        q, k: ``[B, T, H, 128]`` with contiguous heads (any batch/token
            strides), ``T <= GDN_DECODE_MAX_TOKENS``.
        v: ``[B, T, HV, V]`` with contiguous heads, V a multiple of 16 up
            to 256.
        a, b: ``[B, T, HV]`` decay and beta logits (any strides).
        A_log, dt_bias: ``[HV]`` per-head decay parameters.
        state: K-last ``[rows, HV, V, K]`` pool holding the initial states.
        read_indices: int32 ``[B]`` rows of ``state``; a negative row starts
            from a zero state.
        final_indices: int32 ``[B]`` rows of ``state`` receiving the last
            state, or ``None`` for no final write.
        step_states: K-last pool receiving every token's state, or ``None``.
        step_indices: int32 ``[B, T]`` rows of ``step_states``; ``None`` with
            ``step_states`` writes the dense row ``n * T + t``.
        scale: Query scale applied after the optional L2 norm.
        use_qk_l2norm: Whether to L2-normalize q and k over K.
        norm_gate: ``[B * T, HV * V]`` output-gate logits (unit inner stride,
            any row stride) of the fused gated RMSNorm, or ``None`` for the
            raw output.
        norm_weight: ``[V]`` RMSNorm weight, given exactly with norm_gate.
        norm_eps: RMSNorm epsilon, given exactly with norm_gate.

    Negative destination rows are skipped. Returns the ``[B, T, HV, V]``
    output in q's dtype, ``rmsnorm(o) * norm_weight * sigmoid(norm_gate)``
    per head with the norm.
    """
    batch, tokens, heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2], v.shape[3]
    if key_dim != GDN_DECODE_KEY_DIM or value_dim not in GDN_DECODE_VALUE_DIMS:
        raise ValueError(f"unsupported GDN head dims K={key_dim} V={value_dim}")
    if not 1 <= tokens <= GDN_DECODE_MAX_TOKENS:
        raise ValueError(
            f"the gfx950 GDN decode runs 1..{GDN_DECODE_MAX_TOKENS} tokens, got {tokens}"
        )
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.stride(3) != 1 or tensor.stride(2) != tensor.shape[3]:
            raise ValueError(f"{name} heads must be contiguous")
    pools = [("state", state)]
    if step_states is not None:
        pools.append(("step_states", step_states))
    elif step_indices is not None:
        raise ValueError("step_indices requires step_states")
    for name, pool in pools:
        if (
            pool.shape[1:] != (value_heads, value_dim, key_dim)
            or not pool[0].is_contiguous()
        ):
            raise ValueError(f"{name} must be a K-last [rows, HV, V, K] pool")
        if pool.dtype not in GDN_DECODE_STATE_DTYPES:
            raise ValueError(f"unsupported GDN {name} dtype {pool.dtype}")
    output_norm = norm_gate is not None
    if output_norm != (norm_weight is not None) or output_norm != (
        norm_eps is not None
    ):
        raise ValueError("norm_gate, norm_weight and norm_eps go together")
    if output_norm and (
        norm_gate.shape != (batch * tokens, value_heads * value_dim)
        or norm_gate.stride(1) != 1
        or norm_weight.shape != (value_dim,)
        or not norm_weight.is_contiguous()
    ):
        raise ValueError(
            "norm_gate must be [B * T, HV * V] with unit inner stride and "
            "norm_weight a contiguous [V]"
        )
    out = q.new_empty(batch, tokens, value_heads, value_dim)
    if batch == 0:
        return out

    step_mode = GDN_DECODE_STEPS_NONE
    if step_states is not None:
        step_mode = (
            GDN_DECODE_STEPS_DENSE if step_indices is None else GDN_DECODE_STEPS_INDEXED
        )
    # Tokens solved together, bucketed so a server's fixed T compiles once.
    token_block = triton.next_power_of_2(tokens)
    num_warps = _NUM_WARPS_BY_STATE_BYTES[state.element_size()]
    gluon_gdn_decode_gfx950[(batch * value_heads,)](
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        out,
        state,
        read_indices,
        read_indices if final_indices is None else final_indices,
        state if step_states is None else step_states,
        read_indices if step_indices is None else step_indices,
        scale,
        tokens,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        state.stride(0),
        state.stride(0) if step_states is None else step_states.stride(0),
        out if norm_gate is None else norm_gate,
        A_log if norm_weight is None else norm_weight,
        0.0 if norm_eps is None else norm_eps,
        0 if norm_gate is None else norm_gate.stride(0),
        H=heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        NUM_WARPS=num_warps,
        STEP_MODE=step_mode,
        WRITE_FINAL=final_indices is not None,
        USE_QK_L2NORM=use_qk_l2norm,
        TB=token_block,
        LOG2_TB=token_block.bit_length() - 1,
        OUTPUT_NORM=output_norm,
        num_warps=num_warps,
    )
    return out
