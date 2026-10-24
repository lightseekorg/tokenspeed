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

"""PLE decode stages (TokenSpeed ``Qwen4ExpPLELayer``).

P1 serves every row of the launch in each task and reads each weight once.
P2 runs one row per task (``mb`` is that row's view).

* projection (P1), 200 tasks of 64 rows of ``kv_proj`` (BF16, 320 KiB a
  task) on the token's n-gram embedding row. Each task forms the row itself, as
  TokenSpeed's lookup does:

  1. The 16 n-gram ids hash the token with the two previous ones (from the
     row's context page). An EOS blocks the older one.
  2. The task gathers 16 table rows of 160 values.
  3. It projects key ``[4 * HIDDEN]`` then value ``[HIDDEN]``, BF16 out.
* gate and conv (P2), 80 tasks of 128 channels of one stream ``b``. The
  stage computes as TokenSpeed's fused kernels do:

  - ``kn``, ``qn`` are the Gemma-normed key and stream (BF16 ``1 + w``).
  - The stream's gate is ``sigmoid(sign(g) * sqrt(max(|g|, 1e-6)))`` with
    ``g = (kn . qn) / sqrt(HIDDEN)`` in FP32.
  - ``gated = bf16(gate * v)``, and ``nrm`` is its Gemma norm.
  - The dilated depthwise conv (taps 0, 3, 6 of the state, then ``nrm``)
    feeds a SiLU.
  - The new stream is ``bf16(bf16(silu + gated) + h)``.

  Each task also writes its channels' next conv state (taps 1-8, then
  ``nrm``).
* :func:`write_context` stores the next context ``[t-1, t]``. The layer
  calls it after every P1 task reads the input page (which can be the
  output page).

State pages are TokenSpeed's two-slot PLE pages. A page that is not positive
reads the default (EOS context, zero conv state), and the stages do not write
it.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import aggregate, gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.megakernel import bf16, mx, schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.megakernel.plan import Stage, placement
from tokenspeed_kernel_amd.ops.gfx950.megakernel.qwen4_exp.dims import (
    EPS,
    HIDDEN,
    WIDTH,
)

KV_N = gl.constexpr(12800)  # key WIDTH, then value HIDDEN
PROJ_TASKS = gl.constexpr(200)  # KV_N / 64
CHUNK = gl.constexpr(128)  # channels of a P2 task
CONV_TASKS = gl.constexpr(80)  # WIDTH / CHUNK
STATE = gl.constexpr(9)  # (kernel 4 - 1) * dilation 3
INV_SQRT_D = gl.constexpr(0.019764235376052372)  # 1 / sqrt(HIDDEN)
NGRAM_HEADS = gl.constexpr(16)  # 8 bigram heads, then 8 trigram heads
NGRAM_DIM = gl.constexpr(160)  # HIDDEN / NGRAM_HEADS

REGIONS = (
    ("pkv", 6400),  # BF16 pairs of the key and value projections
    ("ph", 5120),  # BF16 pairs of the new streams
)


def stages(bases: dict[str, int], rows: int) -> tuple[Stage, ...]:
    """P1 and P2, placed by ``bases``."""
    p1, p2 = placement(bases, ("p1", "p2"))
    return (
        Stage("p1", int(PROJ_TASKS), p1, writes=("pkv",)),
        Stage("p2", int(CONV_TASKS) * rows, p2, reads=("pkv",), writes=("ph",)),
    )


@aggregate
class Weights:
    """The PLE block's weights (``weights.PleWeights``)."""

    kv_w: gl.tensor  # kv_proj, BF16 [KV_N, HIDDEN] (weights.bf16_tiles)
    gk: gl.tensor  # norm_key as the BF16 1 + w [WIDTH]
    gq: gl.tensor  # norm_query, the same
    gc: gl.tensor  # norm_conv, the same
    conv_w: gl.tensor  # depthwise conv weight, BF16 [WIDTH, 4]
    table: gl.tensor  # n-gram embedding table, BF16 [rows, NGRAM_DIM]
    multipliers: gl.tensor  # int64 [3], the n-gram hash multipliers
    sizes: gl.tensor  # int64 [NGRAM_HEADS], per-head vocabulary sizes
    offsets: gl.tensor  # int64 [NGRAM_HEADS], per-head table offsets
    eos: gl.tensor  # the EOS token id (blocks older context)


@aggregate
class Step:
    """The PLE block's per-step operands (``weights.PleStep``), seen from
    one row (:meth:`row`): ``token`` and ``page_out`` point at that row's
    entry, ``page_in`` at its request's."""

    token: gl.tensor  # int: the row's token
    page_in: gl.tensor  # int32: the row's input page (its request's)
    page_out: gl.tensor  # int32: the row's output row of the *_out pools
    context: gl.tensor  # int64 context pool [pages, 2] (oldest first)
    context_stride: gl.tensor | gl.constexpr
    conv_state: gl.tensor  # conv state pool [pages, WIDTH, STATE]
    page_stride: gl.tensor | gl.constexpr
    context_out: gl.tensor  # int64 [*, 2]: the context pool or verify scratch
    context_out_stride: gl.tensor | gl.constexpr
    conv_out: gl.tensor  # [*, WIDTH, STATE]: the conv pool or verify scratch
    conv_out_stride: gl.tensor | gl.constexpr

    @gluon.jit
    def row(self, r, CHAIN: gl.constexpr):
        """This step as seen by row ``r`` (token ``r % CHAIN`` of request
        ``r // CHAIN``)."""
        return Step(
            self.token + r,
            self.page_in + r // CHAIN,
            self.page_out + r,
            self.context,
            self.context_stride,
            self.conv_state,
            self.page_stride,
            self.context_out,
            self.context_out_stride,
            self.conv_out,
            self.conv_out_stride,
        )


@gluon.jit
def _context(weights, step, tok: gl.constexpr):
    """The context ``(t-2, t-1)`` of ``step``'s row, token ``tok`` of its
    request: the request's earlier tokens in this launch (the rows before),
    else the input page's (EOS without a page)."""
    page = gl.load(step.page_in)
    ok = page > 0
    row = step.context + gl.maximum(page, 0).to(gl.int64) * step.context_stride
    older = gl.load(row, mask=ok, other=0)
    newer = gl.load(row + 1, mask=ok, other=0)
    eos = weights.eos.to(gl.int64)
    older = gl.where(ok, older, eos)
    newer = gl.where(ok, newer, eos)
    if tok == 1:
        older = newer
        newer = gl.load(step.token - 1).to(gl.int64)
    elif tok >= 2:
        older = gl.load(step.token - 2).to(gl.int64)
        newer = gl.load(step.token - 1).to(gl.int64)
    return older, newer


@gluon.jit
def _ngram_rows(weights, step, tok: gl.constexpr, e):
    """Table element of embedding element ``e`` (head ``e // 160``).

    Head ``h`` hashes ``t * m0 ^ (t-1) * m1`` (bigram heads). Trigram heads
    also apply ``^ (t-2) * m2``, with ``t-2`` replaced by EOS after an EOS
    ``t-1``. The table row is then the hash modulo the head's vocabulary size,
    plus the head's offset.
    """
    t = gl.load(step.token).to(gl.int64)
    older, newer = _context(weights, step, tok)
    eos = weights.eos.to(gl.int64)
    h = e // NGRAM_DIM
    m0 = gl.load(weights.multipliers)
    m1 = gl.load(weights.multipliers + 1)
    m2 = gl.load(weights.multipliers + 2)
    older = gl.where(newer == eos, eos, older)
    mixed = (t * m0) ^ (newer * m1)
    mixed = gl.where(h >= NGRAM_HEADS // 2, mixed ^ (older * m2), mixed)
    row = mixed % gl.load(weights.sizes + h) + gl.load(weights.offsets + h)
    return row * NGRAM_DIM + e % NGRAM_DIM


@gluon.jit
def write_context(weights, step, tok: gl.constexpr):
    """Store the next context ``[t-1, t]`` of ``step``'s row, token ``tok``
    of its request, at row ``page_out`` of ``context_out``. ``page_out`` is an
    output page in decode or a verify scratch row in a verify. A value that is
    not positive means none."""
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    i = gl.arange(0, 2, lanes)
    older, newer = _context(weights, step, tok)
    t = gl.load(step.token).to(gl.int64)
    page = gl.load(step.page_out)
    dst = step.context_out + gl.maximum(page, 0).to(gl.int64) * step.context_out_stride
    gl.store(dst + i, gl.where(i == 0, newer, t), mask=(i < 2) & (page > 0))


@gluon.jit
def projection_stage(mb, task, weights, step, STAGE: gl.constexpr):
    """P1 task ``task``: 64 rows of ``kv_proj`` on each row's n-gram
    embedding row (``weights``: :class:`Weights`, ``step``: :class:`Step`)."""
    mb.stamp(STAGE, trace.START)
    tiles = bf16.load_weight_tiles(
        weights.kv_w,
        nb0=task * 4,
        K=HIDDEN,
        kit0=0,
        KITS=HIDDEN // 32,
        SPLIT=2,
        NW=4,
        GROUP=4,
        GROUP_STRIDE=0,
    )
    # Each row as 128 k-iterations of 32 values (80 used).
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    bl: gl.constexpr = gl.BlockedLayout(
        [1, 1, 8], [1, 16, 4], [1, gl.num_warps(), 1], [2, 1, 0]
    )
    g = gl.arange(0, 128, gl.SliceLayout(0, gl.SliceLayout(2, bl)))
    j = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(1, bl)))
    used = (g < HIDDEN // 32)[None, :, None] & (j >= 0)[None, None, :]
    e = gl.minimum(g, HIDDEN // 32 - 1)[None, :, None] * 32 + j[None, None, :]
    xs = ()
    for r in gl.static_range(R):
        ids = _ngram_rows(
            weights,
            step.row(r, schedule.chain(mb.plan)),
            r % schedule.chain(mb.plan),
            e,
        )
        xs += (gl.load(weights.table + ids, mask=used, other=0.0),)
    gl.amd.hint.sched_barrier()
    # Rows end in 32 bytes of padding (see bf16.load_rows).
    x_smem = gl.allocate_shared_memory(
        gl.bfloat16, [RP, 128, 32], mx._padded_rows([RP, 128, 32], 128 * 32, 16)
    )
    for r in gl.static_range(R):
        x_smem.slice(r, 1, 0).store(xs[r])
    gl.barrier()
    mb.stamp(STAGE, trace.READY)
    projs = bf16.gemv(x_smem, tiles, FIRST=0, ROWS=R)
    mb.stamp(STAGE, trace.COMPUTED)
    mx.put_bf16_rows(mb, "pkv", task * 32, projs)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def _gemma(x, gamma):
    """Gemma RMS norm over the whole tensor, ``gamma`` the BF16 ``1 + w``."""
    n = gl.sum(gl.sum(x * x, axis=1), axis=0) * (1.0 / HIDDEN)
    return x * gl.rsqrt(n + EPS) * gamma


@gluon.jit
def conv_stage(mb, task, tok, streams, weights, step, STAGE: gl.constexpr):
    """P2 task ``task`` of a row that is token ``tok`` of its request: stream
    ``b = task // 20``, channels ``128 (task % 20) ..``.

    ``mb``, ``streams`` (the layer's global input :class:`hc.Streams`) and
    ``step`` are the row's views. The state after the row's token goes to
    row ``page_out[0]`` of ``conv_out`` (not positive: none). The state after
    the request's token ``tok + d`` goes to ``page_out[d]``. These rows are
    the output page in decode and the verify scratch rows in a verify.
    """
    gk = weights.gk
    gq = weights.gq
    gc = weights.gc
    conv_w = weights.conv_w
    conv_out = step.conv_out
    out_rows = step.page_out
    out_stride = step.conv_out_stride
    mb.stamp(STAGE, trace.START)
    N: gl.constexpr = schedule.chain(mb.plan)
    b = task // (HIDDEN // CHUNK)
    first = b * HIDDEN
    cb = (task % (HIDDEN // CHUNK)) * CHUNK
    # The whole stream as [32, 128] (rows 20-31 empty) for the norms and gate.
    sl: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [gl.num_warps(), 1], [1, 0])
    r = gl.arange(0, 32, gl.SliceLayout(1, sl))
    c = gl.arange(0, CHUNK, gl.SliceLayout(0, sl))
    valid = (r < HIDDEN // CHUNK)[:, None] & (c >= 0)[None, :]
    e = gl.minimum(r, HIDDEN // CHUNK - 1)[:, None] * CHUNK + c[None, :]
    q = gl.load(streams.element(0, first + e), mask=valid, other=0.0).to(gl.float32)
    wk = gl.load(gk + first + e, mask=valid, other=0.0).to(gl.float32)
    wq = gl.load(gq + first + e, mask=valid, other=0.0).to(gl.float32)
    wc = gl.load(gc + first + e, mask=valid, other=0.0).to(gl.float32)
    # The task's channels. Token tok's taps are S[tok], S[tok + 3], S[tok +
    # 6] of its request's sequence S: the input page's 9 taps, then each
    # token's nrm.
    cl: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    ch = first + cb + gl.arange(0, CHUNK, cl)
    p_in = gl.load(step.page_in)
    state_in = step.conv_state + gl.maximum(p_in, 0).to(gl.int64) * step.page_stride
    has_in = (ch >= 0) & (p_in > 0)
    s0 = gl.load(state_in + ch * STATE + tok, mask=has_in, other=0.0).to(gl.float32)
    s3 = gl.load(state_in + ch * STATE + tok + 3, mask=has_in, other=0.0).to(gl.float32)
    s6 = gl.load(
        state_in + ch * STATE + tok + 6, mask=has_in & (tok + 6 < STATE), other=0.0
    ).to(gl.float32)
    w0 = gl.load(conv_w + ch * 4).to(gl.float32)
    w1 = gl.load(conv_w + ch * 4 + 1).to(gl.float32)
    w2 = gl.load(conv_w + ch * 4 + 2).to(gl.float32)
    w3 = gl.load(conv_w + ch * 4 + 3).to(gl.float32)
    h_ch = gl.load(streams.element(0, ch)).to(gl.float32)
    gl.amd.hint.sched_barrier()
    gated, nrm = _gated_norm(mb, q, first, e, valid, wk, wq, wc)
    mb.stamp(STAGE, trace.READY)
    # The task's channels of gated and nrm.
    sel = valid & (e >= cb) & (e < cb + CHUNK)
    gated_ch = gl.convert_layout(gl.sum(gl.where(sel, gated, 0.0), axis=0), cl)
    nrm_ch = gl.convert_layout(gl.sum(gl.where(sel, nrm, 0.0), axis=0), cl)
    if N > 3:
        # Token 3 of a request takes S[9], token 0's nrm. This task recomputes
        # it from that token's inputs (3 rows back) and does not read it from
        # the task of that token.
        if tok >= 3:
            q0 = gl.load(streams.element(-3, first + e), mask=valid, other=0.0).to(
                gl.float32
            )
            _, nrm0 = _gated_norm(mb.row(-3), q0, first, e, valid, wk, wq, wc)
            s6 = gl.convert_layout(gl.sum(gl.where(sel, nrm0, 0.0), axis=0), cl)
    acc = (((w0 * s0) + w1 * s3) + w2 * s6) + w3 * nrm_ch
    act = (acc * (1.0 / (1.0 + gl.exp(-acc)))).to(gl.bfloat16).to(gl.float32)
    act = (act + gated_ch).to(gl.bfloat16).to(gl.float32)
    out = (act + h_ch).to(gl.bfloat16).to(gl.float32)
    mb.stamp(STAGE, trace.COMPUTED)
    mx.put_bf16(mb, "ph", (first + cb) // 2, out)
    # The state after this token is S[tok + 1 .. tok + 9]: the input page's
    # taps tok + 1 .. 8, then nrm of the request's tokens 0 .. tok. This task
    # writes the page's taps of its own row and its nrm into the rows of its
    # token and the request's later tokens.
    page = gl.load(out_rows)
    dst = conv_out + page.to(gl.int64) * out_stride + ch * STATE
    write = (ch >= 0) & (page > 0)
    for k in gl.static_range(STATE - 1):
        from_page = tok + 1 + k < STATE
        tap = gl.load(
            state_in + ch * STATE + tok + 1 + k, mask=has_in & from_page, other=0.0
        )
        gl.store(dst + k, tap, mask=write & from_page)
    for d in gl.static_range(N):
        later = gl.load(out_rows + d, mask=tok + d < N, other=0)
        to = conv_out + later.to(gl.int64) * out_stride + ch * STATE + STATE - 1 - d
        gl.store(
            to, nrm_ch.to(gl.bfloat16), mask=(ch >= 0) & (later > 0) & (tok + d < N)
        )
    mb.stamp(STAGE, trace.END)


@gluon.jit
def _gated_norm(mb, q, first, e, valid, wk, wq, wc):
    """``(gated, nrm)`` of stream ``q`` (``[32, 128]``, from element
    ``first``) with the key and value of ``mb``'s row (polled)."""
    got_k, got_v = mb.poll_many(("pkv", "pkv"), ((first + e) // 2, (WIDTH + e) // 2))
    k = gl.where(valid, mx.bf16_half(got_k, e), 0.0)
    v = gl.where(valid, mx.bf16_half(got_v, e), 0.0)
    kn = _gemma(k, wk).to(gl.bfloat16).to(gl.float32)
    qn = _gemma(q, wq).to(gl.bfloat16).to(gl.float32)
    g = gl.sum(gl.sum(kn * qn, axis=1), axis=0) * INV_SQRT_D
    m = gl.sqrt(gl.maximum(gl.abs(g), 1e-6))
    g = gl.where(g > 0, m, gl.where(g < 0, -m, 0.0))
    gate = 1.0 / (1.0 + gl.exp(-g))
    gated = (gate * v).to(gl.bfloat16).to(gl.float32)
    nrm = _gemma(gated, wc).to(gl.bfloat16).to(gl.float32)
    return gated, nrm


__all__ = [
    "CONV_TASKS",
    "PROJ_TASKS",
    "REGIONS",
    "Step",
    "Weights",
    "conv_stage",
    "projection_stage",
    "write_context",
]
