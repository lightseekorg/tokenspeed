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

"""Gated DeltaNet decode stages (TokenSpeed ``Qwen3_5GatedDeltaNet``, one token).

Each stage serves every row of the launch. B1 runs all rows as MFMA rows and
reads each weight once. B3 runs one task per (row, value head).

* projection (B1), 128 tasks of 128 rows of ``in_proj_qkvz`` (MXFP4) on the
  mixed row (MXFP8). Tasks 0-79 own the ``q | k | v`` channels and apply the
  depthwise causal conv: ``silu(w . [s0, s1, s2, x])``. Its per-slot state
  ``[QKV, 3]`` (channel-major BF16) becomes ``[s1, s2, x]``.
  Tasks 80-127 own ``z`` and each also computes two of the 96 BF16
  ``in_proj_b | in_proj_a`` logits on the BF16 mixed row.
* recurrence (B3), one task per value head. The task runs the gated delta
  rule on the head's FP32 state ``[V, K]``. It reads the state from the
  slot's input page and writes it to the slot's output page. Then it runs the
  gated RMS norm ``rms(o) * sigmoid(z) * w`` and publishes the result as
  MXFP8 for the output projection.

A launch can verify ``Plan.chain`` tokens per request (TokenSpeed's MTP
verify). In a verify, B1 builds the conv window of token ``t`` from the
request's earlier rows. B1 writes each window to its own row of ``conv_out``
(the verify scratch). B3 runs the tokens of a request in order. It writes the
``[k | v | a | b]`` of each token to TokenSpeed's replay payload instead of a
state page.

State pages follow TokenSpeed's two-slot paging. ``state_in`` / ``state_out``
point at the row's input and output page. A padding row has -1: the stages
read and write no state for it, and nothing uses its outputs. Pages are
``page_stride`` elements apart (the state arena interleaves other planes).
Within a page, the planes are dense.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import aggregate, gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.megakernel import mx, schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.megakernel.plan import Stage, placement
from tokenspeed_kernel_amd.ops.gfx950.megakernel.qwen4_exp.dims import EPS, HIDDEN

KEY_HEADS = gl.constexpr(16)
VALUE_HEADS = gl.constexpr(48)
HEAD = gl.constexpr(128)
KEY = gl.constexpr(2048)  # KEY_HEADS * HEAD
VALUE = gl.constexpr(6144)  # VALUE_HEADS * HEAD
QKV = gl.constexpr(10240)  # conv channels: q | k | v
CONV = gl.constexpr(4)
ROWS = gl.constexpr(128)  # projection rows of a B1 task
CONV_TASKS = gl.constexpr(80)  # QKV / ROWS
PROJ_TASKS = gl.constexpr(128)  # (QKV + VALUE) / ROWS
SCALE = gl.constexpr(0.08838834764831845)  # HEAD ** -0.5

REGIONS = (
    ("qkvz", 8192),  # BF16 pairs: conv(q | k | v), then z
    ("ba", 96),  # in_proj_b then in_proj_a logits (f32 of BF16 values)
    ("o8", 1536),  # MXFP8 words of the normed, gated output
    ("oe", 192),  # its exponents
)


def stages(bases: dict[str, int], rows: int, chain: int, X: str) -> tuple[Stage, ...]:
    """B1 and B3, placed by ``bases``. ``X`` is the region prefix of the
    mixed row (the attention mixer's)."""
    b1, b3 = placement(bases, ("b1", "b3"))
    return (
        Stage(
            "b1",
            int(PROJ_TASKS),
            b1,
            reads=(X + "x8", X + "xe", X + "xbf"),
            writes=("qkvz", "ba"),
        ),
        Stage(
            "b3",
            int(VALUE_HEADS) * (rows // chain),
            b3,
            reads=("qkvz", "ba"),
            writes=("o8", "oe"),
        ),
    )


@aggregate
class Weights:
    """A Gated DeltaNet layer's weights (``weights.DeltaLayerWeights``)."""

    qkvz_w: gl.tensor  # in_proj_qkvz [QKV + VALUE, HIDDEN], N16 tiles
    qkvz_s: gl.tensor  # and scale dwords
    conv_w: gl.tensor  # conv weight, BF16 [QKV, 4]
    ba_w: gl.tensor  # [in_proj_b | in_proj_a], BF16 [2 * VALUE_HEADS, HIDDEN]
    a_log: gl.tensor  # FP32 [VALUE_HEADS]
    dt_bias: gl.tensor  # FP32 [VALUE_HEADS]
    norm_w: gl.tensor  # gated RMS norm weight, BF16 [HEAD]


@aggregate
class State:
    """A launch's recurrent state operands.

    The pages of the pools are ``*_page_stride`` elements apart. ``state_in``
    and ``state_out`` are int32 ``[rows]``. They hold the input page of each
    row (the tokens of a request share it) and the output page of each row.
    An output page of -1 means none (a verify writes no state).
    Each row's conv window after its token goes to ``[QKV, 3]`` at
    ``conv_out + conv_out_rows[row] * conv_out_stride``. That address is the
    output page in decode and a verify scratch row in a verify. A
    ``conv_out_rows`` value of -1 means none.
    """

    conv_state: gl.tensor  # BF16 [pages, QKV, 3]
    conv_page_stride: gl.tensor | gl.constexpr
    conv_out: gl.tensor
    conv_out_rows: gl.tensor
    conv_out_stride: gl.tensor | gl.constexpr
    ssm_state: gl.tensor  # FP32 [pages, VALUE_HEADS, V, K], K last
    ssm_page_stride: gl.tensor | gl.constexpr
    # None, or TokenSpeed's replay payload of the layer (a verify). Row
    # ``row`` at ``replay + row * replay_stride`` gets the token's
    # ``[k | v | a | b]``.
    replay: gl.tensor | gl.constexpr
    replay_stride: gl.tensor | gl.constexpr
    state_in: gl.tensor
    state_out: gl.tensor


@gluon.jit
def projection_stage(mb, task, weights, state, X: gl.constexpr, STAGE: gl.constexpr):
    """B1 task ``task``. ``weights`` is a :class:`Weights` and ``state`` is a
    :class:`State`. ``X`` is the region prefix of the mixed row, the attention
    mixer's ``a_``."""
    mb.stamp(STAGE, trace.START)
    tiles, scales = _projection_issue(task, weights)
    _projection_finish(mb, task, tiles, scales, weights, state, X, STAGE)


@gluon.jit
def projection_recurrence_stage(
    mb,
    b1_task,
    b3_request,
    b3_task,
    weights,
    state,
    X: gl.constexpr,
):
    """B1 task ``b1_task`` then B3 task ``b3_task`` of request
    ``b3_request`` on one CTA (arguments as :func:`projection_stage` and
    :func:`recurrence_stage`).

    The recurrent state page needs no mailbox input, so the CTA issues its
    loads right after B1's weight loads, at launch, not after B1 ends.
    """
    mb.stamp("b1", trace.START)
    tiles, scales = _projection_issue(b1_task, weights)
    gl.amd.hint.sched_barrier()
    first = b3_request * schedule.chain(mb.plan)
    head_state, a_bias, neg_a, norm = _recurrence_issue(b3_task, weights, state, first)
    gl.amd.hint.sched_barrier()
    _projection_finish(mb, b1_task, tiles, scales, weights, state, X, "b1")
    mb.stamp("b3", trace.START)
    _recurrence_finish(
        mb, b3_request, b3_task, head_state, a_bias, neg_a, norm, state, "b3"
    )


@gluon.jit
def _projection_issue(task, weights):
    """Issue B1 task ``task``'s weight loads: one 16-row block per wave, no
    split-K (160 KiB per task)."""
    return mx.load_weight_tiles(
        weights.qkvz_w,
        weights.qkvz_s,
        nb0=task * (ROWS // 16),
        K=HIDDEN,
        kit0=0,
        KITS=20,
        SPLIT=1,
        NW=8,
        GROUP=8,
        GROUP_STRIDE=0,
        SCALES="k4",
    )


@gluon.jit
def _projection_finish(
    mb,
    task,
    tiles,
    scales,
    weights,
    state,
    X: gl.constexpr,
    STAGE: gl.constexpr,
):
    """B1 task ``task`` after its weight loads, for every row."""
    conv_w = weights.conv_w
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    if task < CONV_TASKS:
        # [rows, channels] with a channel's rows in one thread, so a token's
        # window can take the request's earlier tokens (rows) in registers.
        lr: gl.constexpr = gl.BlockedLayout(
            [RP, 1], [1, 64], [1, gl.num_warps()], [1, 0]
        )
        r = gl.arange(0, RP, gl.SliceLayout(1, lr))[:, None]
        n = gl.arange(0, ROWS, gl.SliceLayout(0, lr))[None, :]
        rows = gl.minimum(r, R - 1)
        channel = task * ROWS + n
        # Every load that needs no mailbox input issues before the poll.
        slot_in = gl.load(state.state_in + rows)
        slot_out = gl.load(state.conv_out_rows + rows)
        taps = (
            state.conv_state
            + slot_in.to(gl.int64) * state.conv_page_stride
            + channel * 3
        )
        has_in = (r < R) & (slot_in >= 0)
        s0 = gl.load(taps, mask=has_in, other=0.0)
        s1 = gl.load(taps + 1, mask=has_in, other=0.0)
        s2 = gl.load(taps + 2, mask=has_in, other=0.0)
        w0 = gl.load(conv_w + channel * CONV).to(gl.float32)
        w1 = gl.load(conv_w + channel * CONV + 1).to(gl.float32)
        w2 = gl.load(conv_w + channel * CONV + 2).to(gl.float32)
        w3 = gl.load(conv_w + channel * CONV + 3).to(gl.float32)
        gl.amd.hint.sched_barrier()
        x_smem, xs_smem, _ = mx.load_mxfp8_rows(mb, X + "x8", X + "xe", 20, R)
        mb.stamp(STAGE, trace.READY)
        x = mx.gemv(
            x_smem,
            xs_smem,
            tiles,
            scales,
            SPLIT=1,
            NW=8,
            SEG=0,
            SEG_STRIDE=0,
            FIRST=0,
            ROWS=R,
        ).to(gl.bfloat16)
        x = gl.convert_layout(x, lr)
        mb.stamp(STAGE, trace.COMPUTED)
        # The request's sequence is S = (s0, s1, s2, x of token 0, x of token 1,
        # ...). Token t's window holds S[t], S[t + 1], S[t + 2].
        t = r % schedule.chain(mb.plan)
        k0 = _window_tap(s0, s1, s2, x, t, 0, schedule.chain(mb.plan))
        k1 = _window_tap(s0, s1, s2, x, t, 1, schedule.chain(mb.plan))
        k2 = _window_tap(s0, s1, s2, x, t, 2, schedule.chain(mb.plan))
        out = mx.silu(
            w0 * k0.to(gl.float32)
            + w1 * k1.to(gl.float32)
            + w2 * k2.to(gl.float32)
            + w3 * x.to(gl.float32)
        )
        new = (
            state.conv_out + slot_out.to(gl.int64) * state.conv_out_stride + channel * 3
        )
        has_out = (r < R) & (slot_out >= 0)
        gl.store(new, k1, mask=has_out)
        gl.store(new + 1, k2, mask=has_out)
        gl.store(new + 2, x, mask=has_out)
        mx.put_bf16_rows(mb, "qkvz", task * (ROWS // 2), out)
    else:
        # Two b / a logits on the BF16 mixed rows, with the z rows' GEMV.
        row = (task - CONV_TASKS) * 2
        bl: gl.constexpr = gl.BlockedLayout(
            [1, 8], [1, 64], [1, gl.num_warps()], [1, 0]
        )
        k = gl.arange(0, 4096, gl.SliceLayout(0, bl))
        wr = gl.load(weights.ba_w + row * HIDDEN + k, mask=k < HIDDEN, other=0.0)
        wr2 = gl.load(weights.ba_w + (row + 1) * HIDDEN + k, mask=k < HIDDEN, other=0.0)
        gl.amd.hint.sched_barrier()
        # One poll: every row's MXFP8 and BF16 mixed row.
        rk = gl.arange(0, RP, gl.SliceLayout(1, bl))
        x_smem, xs_smem, extras = mx.load_mxfp8_rows(
            mb,
            X + "x8",
            X + "xe",
            20,
            R,
            (X + "xbf",),
            (mb.row_offset(rk)[:, None] + (gl.minimum(k, HIDDEN - 1) // 2)[None, :],),
        )
        mb.stamp(STAGE, trace.READY)
        # TokenSpeed's projections output BF16.
        xk = mx.bf16_half(extras[0], k[None, :])
        logit = gl.sum(xk * wr.to(gl.float32)[None, :], axis=1).to(gl.bfloat16)
        logit2 = gl.sum(xk * wr2.to(gl.float32)[None, :], axis=1).to(gl.bfloat16)
        two = gl.arange(0, 2, gl.SliceLayout(0, bl))[None, :]
        pair = gl.where(
            two == 0,
            logit.to(gl.float32)[:, None],
            logit2.to(gl.float32)[:, None],
        )
        mb.put(
            "ba",
            mb.row_offset(rk)[:, None] + row + two,
            pair,
            mask=(rk < R)[:, None],
        )
        projs = mx.gemv(
            x_smem,
            xs_smem,
            tiles,
            scales,
            SPLIT=1,
            NW=8,
            SEG=0,
            SEG_STRIDE=0,
            FIRST=0,
            ROWS=R,
        )
        mb.stamp(STAGE, trace.COMPUTED)
        mx.put_bf16_rows(mb, "qkvz", task * (ROWS // 2), projs)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def _head_pairs(mb, first, STRIDE: gl.constexpr, head, layout: gl.constexpr):
    """``(pairs, element)``. ``pairs`` holds the ``qkvz`` pair indices
    ``[token, 2, 128]`` of every token of the request whose first row is
    ``first``. The indices cover head ``head`` of the channels and the same
    head ``STRIDE`` channels later: q and k, or v and z. ``element`` holds the
    element index of each value. Both are in ``layout``."""
    N: gl.constexpr = schedule.chain(mb.plan)
    NP: gl.constexpr = triton.next_power_of_2(N)
    tok = gl.arange(0, NP, gl.SliceLayout(1, gl.SliceLayout(2, layout)))[:, None, None]
    two = gl.arange(0, 2, gl.SliceLayout(0, gl.SliceLayout(2, layout)))[None, :, None]
    e = gl.arange(0, HEAD, gl.SliceLayout(0, gl.SliceLayout(1, layout)))[None, None, :]
    channel = two * STRIDE + head * HEAD + e
    return mb.row_offset(first + gl.minimum(tok, N - 1)) + channel // 2, e


@gluon.jit
def _token_vectors(x, layout: gl.constexpr):
    """``(firsts, seconds)``: per token, the two ``[128]`` vectors of ``x
    [token, 2, 128]`` (tokens and halves in each thread's registers), in
    ``layout``. Splits pick registers. Nothing moves between lanes."""
    NP: gl.constexpr = x.shape[0]
    a, b = gl.split(gl.permute(x, [2, 0, 1]))
    return _tokens(a, NP, layout), _tokens(b, NP, layout)


@gluon.jit
def _tokens(x, NP: gl.constexpr, layout: gl.constexpr):
    """The ``NP`` (1, 2 or 4) columns of ``x [128, NP]`` as ``[128]`` vectors."""
    if NP == 1:
        out = (gl.convert_layout(gl.reshape(x, [HEAD]), layout),)
    elif NP == 2:
        t0, t1 = gl.split(x)
        out = (gl.convert_layout(t0, layout), gl.convert_layout(t1, layout))
    else:
        # Token 2 a + b of [128, 2 (a), 2 (b)].
        lo, hi = gl.split(gl.reshape(x, [HEAD, 2, 2]))
        t0, t2 = gl.split(lo)
        t1, t3 = gl.split(hi)
        out = (
            gl.convert_layout(t0, layout),
            gl.convert_layout(t1, layout),
            gl.convert_layout(t2, layout),
            gl.convert_layout(t3, layout),
        )
    return out


@gluon.jit
def _window_tap(s0, s1, s2, x, t, K: gl.constexpr, N: gl.constexpr):
    """Tap ``K`` of each row's conv window: ``S[t + K]`` of its request's
    sequence ``S = (s0, s1, s2, x of token 0, ...)``. ``t`` is the row's
    token."""
    tap = gl.where(t + K == 0, s0, gl.where(t + K == 1, s1, s2))
    if N > 3 - K:
        # Tokens past the first 3 - K take an earlier token of the request.
        tap = gl.where(t + K >= 3, _shift_rows(x, 3 - K), tap)
    return tap


@gluon.jit
def _shift_rows(x, D: gl.constexpr):
    """``y[j] = x[j - D]`` (0 for ``j < D``) of ``x [RP, N]`` whose rows sit
    in each thread's registers."""
    RP: gl.constexpr = x.shape[0]
    r = gl.arange(0, RP, gl.SliceLayout(1, x.type.layout))[:, None]
    y = gl.zeros_like(x)
    for j in gl.static_range(D, RP):
        src = gl.sum(gl.where(r == j - D, x.to(gl.float32), 0.0), axis=0)
        y = gl.where(r == j, src.to(x.dtype)[None, :], y)
    return y


@gluon.constexpr_function
def _state_layout(num_warps):
    """``[V, K]`` head state: 8 consecutive K per lane, 16 lanes span K."""
    return gl.BlockedLayout([4, 8], [4, 16], [num_warps, 1], [1, 0])


@gluon.jit
def _softplus(x):
    return gl.where(x > 20.0, x, gl.log(1.0 + gl.exp(x)))


@gluon.jit
def recurrence_stage(mb, request, task, weights, state, STAGE: gl.constexpr):
    """B3 task ``task`` = value head ``vh`` of request ``request``. The task
    runs the request's tokens (``schedule.chain`` rows) one after another.
    ``weights`` is a :class:`Weights` and ``state`` is a :class:`State`."""
    mb.stamp(STAGE, trace.START)
    first = request * schedule.chain(mb.plan)
    head_state, a_bias, neg_a, w = _recurrence_issue(task, weights, state, first)
    gl.amd.hint.sched_barrier()
    _recurrence_finish(mb, request, task, head_state, a_bias, neg_a, w, state, STAGE)


@gluon.jit
def _recurrence_issue(task, weights, state, first):
    """Issue B3 task ``task``'s loads that need no mailbox input: the head's
    state from the input page of row ``first``, its gate parameters and norm
    weight."""
    vh = task
    sl: gl.constexpr = _state_layout(gl.num_warps())
    r = gl.arange(0, HEAD, gl.SliceLayout(1, sl))
    c = gl.arange(0, HEAD, gl.SliceLayout(0, sl))
    slot_in = gl.load(state.state_in + first)
    head = vh * (HEAD * HEAD) + r[:, None] * HEAD + c[None, :]
    head_state = gl.load(
        state.ssm_state + slot_in.to(gl.int64) * state.ssm_page_stride + head,
        mask=(r[:, None] >= 0) & (slot_in >= 0),
        other=0.0,
    )
    a_bias = gl.load(weights.dt_bias + vh)
    neg_a = -gl.exp(gl.load(weights.a_log + vh))
    w = gl.load(weights.norm_w + r).to(gl.float32)
    return head_state, a_bias, neg_a, w


@gluon.jit
def _recurrence_finish(
    mb,
    request,
    task,
    head_state,
    a_bias,
    neg_a,
    w,
    state,
    STAGE: gl.constexpr,
):
    """B3 task ``task`` of ``request`` after :func:`_recurrence_issue`: poll
    every token's inputs, then update and publish token by token."""
    replay = state.replay
    N: gl.constexpr = schedule.chain(mb.plan)
    NP: gl.constexpr = triton.next_power_of_2(N)
    first = request * N
    vh = task
    kh = vh // (VALUE_HEADS // KEY_HEADS)
    sl: gl.constexpr = _state_layout(gl.num_warps())
    r = gl.arange(0, HEAD, gl.SliceLayout(1, sl))
    c = gl.arange(0, HEAD, gl.SliceLayout(0, sl))
    # One poll for every token reads q and k of the key head and v and z of
    # the value head (each [token, 2, 128]). The poll then reads b and a. A
    # pair holds two BF16 values. kl / vl hold every token's vectors in each
    # thread, in the lanes of the state layout's K / V slices. A poll in these
    # layouts would make a thread hold 16 pairs a token. Thus, past one token,
    # the poll is dense (two pairs a thread) and one conversion moves the
    # values.
    # The code loads slot_out before the poll. After the poll, the load would
    # delay the state store.
    slot_out = gl.load(state.state_out + first + N - 1)
    NW: gl.constexpr = gl.num_warps()
    kl: gl.constexpr = gl.BlockedLayout([NP, 2, 8], [1, 4, 16], [1, 1, NW], [2, 1, 0])
    vl: gl.constexpr = gl.BlockedLayout([NP, 2, 4], [1, 16, 4], [1, 1, NW], [1, 2, 0])
    lp: gl.constexpr = gl.BlockedLayout([1, 1, 2], [1, 1, 64], [4, 2, 1], [2, 1, 0])
    ql: gl.constexpr = kl if N == 1 else lp
    vpl: gl.constexpr = vl if N == 1 else lp
    qk_i, qk_e = _head_pairs(mb, first, KEY, kh, ql)
    vz_i, vz_e = _head_pairs(mb, first, QKV - 2 * KEY, 2 * KEY // HEAD + vh, vpl)
    bl: gl.constexpr = gl.BlockedLayout([1, 1], [32, 2], [NW, 1], [1, 0])
    tok2 = gl.arange(0, NP, gl.SliceLayout(1, bl))[:, None]
    i = gl.arange(0, 2, gl.SliceLayout(0, bl))[None, :]
    ba_i = mb.row_offset(first + gl.minimum(tok2, N - 1)) + i * VALUE_HEADS + vh
    got_qk, got_vz, got_ba = mb.poll_many(("qkvz", "qkvz", "ba"), (qk_i, vz_i, ba_i))
    mb.stamp(STAGE, trace.READY)
    qk_all = gl.convert_layout(mx.bf16_half(got_qk, qk_e), kl)
    vz_all = gl.convert_layout(mx.bf16_half(got_vz, vz_e), vl)
    ba_all = got_ba.to(gl.float32, bitcast=True)
    qs, ks = _token_vectors(qk_all, gl.SliceLayout(0, sl))
    vs, zs = _token_vectors(vz_all, gl.SliceLayout(1, sl))
    for t in gl.static_range(N):
        q, k, v, z = qs[t], ks[t], vs[t], zs[t]
        ba = gl.sum(gl.where(tok2 == t, ba_all, 0.0), axis=0)
        ib = gl.arange(0, 2, ba.type.layout)
        b = gl.sum(gl.where(ib == 0, ba, 0.0), axis=0)
        a = gl.sum(gl.where(ib == 1, ba, 0.0), axis=0)
        if replay is not None:
            # TokenSpeed's replay payload: k, v after the conv (before the L2
            # norm), then the raw a and b.
            row_ptr = replay + (first + t).to(gl.int64) * state.replay_stride
            if vh % (VALUE_HEADS // KEY_HEADS) == 0:
                gl.store(row_ptr + kh * HEAD + c, k.to(gl.bfloat16))
            gl.store(row_ptr + KEY + vh * HEAD + r, v.to(gl.bfloat16))
            ga = gl.arange(0, 1, gl.SliceLayout(1, sl))
            gl.store(row_ptr + KEY + VALUE + vh + ga, (a + ga * 0.0).to(gl.bfloat16))
            gl.store(
                row_ptr + KEY + VALUE + VALUE_HEADS + vh + ga,
                (b + ga * 0.0).to(gl.bfloat16),
            )
        q = q * gl.rsqrt(gl.sum(q * q, axis=0) + EPS)
        k = k * gl.rsqrt(gl.sum(k * k, axis=0) + EPS)
        decay = gl.exp(neg_a * _softplus(a + a_bias))
        beta = mx.sigmoid(b)
        head_state = head_state * decay
        delta = beta * (v - gl.sum(head_state * k[None, :], axis=1))
        head_state = head_state + delta[:, None] * k[None, :]
        o = gl.sum(head_state * q[None, :], axis=1) * SCALE
        if t == N - 1:
            # The state after the request's last token (no page in a verify).
            head = vh * (HEAD * HEAD) + r[:, None] * HEAD + c[None, :]
            new = state.ssm_state + slot_out.to(gl.int64) * state.ssm_page_stride + head
            gl.store(new, head_state, mask=(r[:, None] >= 0) & (slot_out >= 0))
        # Gated RMS norm of the head (TokenSpeed: BF16 core output, FP32 norm).
        o = o.to(gl.bfloat16).to(gl.float32)
        rms = gl.rsqrt(gl.sum(o * o, axis=0) * (1.0 / HEAD) + EPS)
        y = (o * rms * mx.sigmoid(z) * w).to(gl.bfloat16).to(gl.float32)
        mx.put_mxfp8(mb.row(first + t), "o8", "oe", vh * 4, gl.reshape(y, [4, 32]))
    mb.stamp(STAGE, trace.COMPUTED)
    mb.stamp(STAGE, trace.END)
