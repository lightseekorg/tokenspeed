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

"""The DFlash2 draft head of Muse Spark as one launch, one TP8 rank.

After the draft's final norm, the regular path takes each draft step's top 16
tokens over the vocabulary-parallel LM head (``vocab_parallel_topk.py``) and
scores the 16 x 16 transitions between consecutive steps with the candidate
selector (``models/dflash2.py``). This launch does both for the draft steps
of one or two requests (each block's rows 1 .. 7, request by request), on
every rank of the TP8 group:

* lmh, 226 tasks of 7 16-row blocks (112 vocabulary rows) of the rank's
  25256-row shard: the logits ``bf16(bf16(x W^T) * scale)`` over the whole K
  (8192, in 16 parts of 512 in one task), then each step's best 16 of the
  task's rows, into ``tk``. The eighth wave of a task reads its seventh
  block again (an L2 hit), so the 226 tasks cover the shard on 226 CTAs.
* hproj, 16 tasks: the selector's hidden projection (8192 -> 256), FP32
  K-part partials into ``hpart``, on CTAs idle during lmh.
* merge1, a task per step: the rank's best 16 of ``tk``, put to every rank's
  ``rk``.
* merge2, on merge1's CTAs: the best 16 of the 8 ranks' 128, sorted by
  value (ties: the lower token id first), softcapped: the step's candidate
  ids and unary scores. The ids also go to ``cid`` for the next step.
* score, on merge2's CTAs: the selector's edge scores
  ``unary[c] + bf16(sum_r bf16(pred[p, r] h[r]) succ[c, r])``, where ``pred``
  holds the codebook rows of the previous step's candidates (the anchor
  token for a request's step 0) and ``succ`` those of this step's. A
  request's step 0 task also stores its anchor, the block's first token
  clamped to the vocabulary, for the walk that follows.

Every rank computes the same outputs, as the regular path does.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.monokernel import (
    bf16,
    code_warm,
    mailbox,
    mx,
    schedule,
    trace,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import draft_mlp
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.dims import HIDDEN, WORLD
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import Plan, Stage, placement
from tokenspeed_kernel_amd.ops.gfx950.monokernel.weights import bf16_tiles

cdna4 = gl.amd.cdna4

VOCAB = 202048
SHARD = gl.constexpr(25256)  # vocabulary rows a rank: VOCAB / WORLD
BLOCKS = gl.constexpr(1579)  # 16-row blocks of the shard, the last half padding
TASK_BLOCKS = gl.constexpr(7)
LMH_TASKS = gl.constexpr(226)  # ceil(BLOCKS / TASK_BLOCKS)
LMH_KS = gl.constexpr(512)  # K values of one weight load round
TOPK = gl.constexpr(16)
RANK_DIM = gl.constexpr(256)  # the selector's rank
HPROJ_KS = gl.constexpr(1024)
HPROJ_KPARTS = gl.constexpr(8)
HPROJ_TASKS = gl.constexpr(16)  # 2 groups of 128 rows x 8 K parts
STEPS = 7  # draft steps a block: rows 1 .. 7
REQUEST_STEPS = gl.constexpr(STEPS)
MAX_REQUESTS = 2  # requests a launch: their steps one after another
VOCAB_IDS = gl.constexpr(VOCAB)
ID_BITS = gl.constexpr(15)  # a shard row's index fits 15 bits
GID_BITS = gl.constexpr(18)  # a token id fits 18 bits

PEERS = ("rk",)


def regions() -> tuple[tuple[str, int], ...]:
    """The draft head's regions as (name, int64 pairs): tk, rk, cid, hpart."""
    return (
        ("tk", int(LMH_TASKS) * int(TOPK)),  # packed (value, shard row) [task][16]
        ("rk", int(WORLD) * int(TOPK)),  # packed (value, shard row) [rank][16]
        ("cid", int(TOPK)),  # candidate token ids
        ("hpart", int(HPROJ_KPARTS) * int(RANK_DIM)),  # FP32 [K part][rank dim]
    )


BASES = {"lmh": 0, "hproj": 226, "merge": 242}


def stages(bases: dict[str, int], rows: int) -> tuple[Stage, ...]:
    """The stages of ``rows`` draft steps, placed by ``bases`` (stage name to
    first CTA)."""
    lmh, hproj, merge = placement(bases, ("lmh", "hproj", "merge"))
    return (
        Stage("lmh", int(LMH_TASKS), lmh, writes=("tk",)),
        Stage("hproj", int(HPROJ_TASKS), hproj, writes=("hpart",)),
        Stage("merge1", rows, merge, reads=("tk",), writes=("rk",)),
        Stage("merge2", rows, merge, reads=("rk",), writes=("cid",)),
        Stage("score", rows, merge, reads=("cid", "hpart")),
    )


# CTAs 0 .. CODE_TASKS - 1 first read the launch's own code into the L2s
# (code_warm.py): 8 tasks on each XCD.
CODE_TASKS = 64


@functools.lru_cache(maxsize=None)
def head_plan(rows: int) -> Plan:
    """The plan of the draft head for ``rows`` draft steps."""
    return Plan(
        grid=256,
        stages=(Stage("code", CODE_TASKS, 0),) + stages(BASES, rows),
        regions=regions(),
        rows=rows,
        chain=rows,
        world=int(WORLD),
        peers=PEERS,
    )


@dataclass(frozen=True)
class DraftHeadWeights:
    """One TP8 rank's draft head, packed for the launch.

    Attributes:
        rank: The rank in the TP8 group.
        lm_head: BF16 ``[1579 * 16, 8192]``: the rank's vocabulary shard in
            ``weights.bf16_tiles`` order, 8 zero rows of padding last.
        hproj: BF16 ``[256, 8192]`` selector hidden projection, tiled.
        pred: BF16 ``[202048, 256]`` predecessor codebook (row-major).
        succ: BF16 ``[202048, 256]`` successor codebook (row-major).
        scale: The logits' multiplier (``output_multiplier``).
        softcap: The final logit softcap.
    """

    rank: int
    lm_head: torch.Tensor
    hproj: torch.Tensor
    pred: torch.Tensor
    succ: torch.Tensor
    scale: float
    softcap: float


def pack_draft_head(
    tensors: dict[str, torch.Tensor], rank: int, scale: float, softcap: float
) -> DraftHeadWeights:
    """Pack one rank's draft head.

    Args:
        tensors: BF16 by role: ``lm_head`` ``[25256, 8192]`` (the rank's
            vocabulary shard, token ids ``25256 rank ..``); ``hidden_proj``
            ``[256, 8192]``; ``pred_codebook`` and ``succ_codebook``
            ``[202048, 256]``.
        rank: The rank in the TP8 group.
        scale: The logits' multiplier.
        softcap: The final logit softcap (> 0).
    """
    shapes = {
        "lm_head": (int(SHARD), int(HIDDEN)),
        "hidden_proj": (int(RANK_DIM), int(HIDDEN)),
        "pred_codebook": (VOCAB, int(RANK_DIM)),
        "succ_codebook": (VOCAB, int(RANK_DIM)),
    }
    for name, shape in shapes.items():
        t = tensors[name]
        if t.dtype != torch.bfloat16 or tuple(t.shape) != shape:
            raise ValueError(f"{name}: BF16 {shape}, got {t.dtype} {tuple(t.shape)}")
    if not softcap > 0:
        raise ValueError(f"softcap {softcap}: the head softcaps its scores")
    head = tensors["lm_head"]
    padded = head.new_zeros(int(BLOCKS) * 16, int(HIDDEN))
    padded[: int(SHARD)] = head
    return DraftHeadWeights(
        rank=rank,
        lm_head=bf16_tiles(padded),
        hproj=bf16_tiles(tensors["hidden_proj"].contiguous()),
        pred=tensors["pred_codebook"].contiguous(),
        succ=tensors["succ_codebook"].contiguous(),
        scale=float(scale),
        softcap=float(softcap),
    )


@gluon.jit
def _lmh_tiles(w, nb0, last, kit0, NW: gl.constexpr):
    """:func:`bf16.load_weight_tiles` of one lmh K round (16 k-iterations,
    no K split), where N wave ``v`` reads block ``nb0 + min(v, last)``."""
    bl: gl.constexpr = bf16.b_layout(1, NW)
    s = gl.arange(0, 1, gl.SliceLayout(1, gl.SliceLayout(2, bl)))
    k = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(2, bl)))
    n = gl.arange(0, 16 * NW, gl.SliceLayout(0, gl.SliceLayout(1, bl)))
    block = nb0 + gl.minimum(n // 16, last)
    lane = ((k // 8) * 128 + k % 8)[None, :, None] + ((n % 16) * 8)[None, None, :]
    KI: gl.constexpr = HIDDEN // 32
    # The shard's tiles span 414 MB: offsets in 32 bits are 16-bit elements
    # from the task's first block.
    rel = ((block - nb0)[None, None, :] * KI + (kit0 + s)[:, None, None]) * 512 + lane
    base = w + nb0.to(gl.int64) * (KI * 512)
    tiles = ()
    for i in gl.static_range(LMH_KS // 32):
        tiles += (cdna4.buffer_load(base, rel + i * 512, cache=".cg"),)
    return tiles


@gluon.jit
def _x_round(x, words, kr, R: gl.constexpr):
    """Store K round ``kr`` (512 values) of the BF16 rows ``x [R, 8192]``
    into ``words`` and return them as :func:`bf16.gemv`'s LDS rows."""
    RP: gl.constexpr = triton.next_power_of_2(R)
    KP: gl.constexpr = LMH_KS // 32
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    p = gl.arange(0, RP * KP * 16, lanes)
    row = gl.minimum(p // (KP * 16), R - 1)
    at = p % (KP * 16)
    words_ptr = x.to(gl.pointer_type(gl.int32), bitcast=True)
    got = gl.load(words_ptr + row * (HIDDEN // 2) + kr * (LMH_KS // 2) + at)
    # The previous round's MFMAs have read the buffer.
    gl.barrier()
    words.store(gl.reshape(got, [RP, KP, 16]))
    gl.barrier()
    return words.reinterpret(
        gl.bfloat16, [RP, KP, 32], mx._padded_rows([RP, KP, 32], KP * 32, 16)
    )


@gluon.jit
def _ordered(value_bf16_bits):
    """A BF16 value's bits as an unsigned 16-bit key in the values' order."""
    b = value_bf16_bits & 0xFFFF
    return gl.where((b & 0x8000) != 0, (~b) & 0xFFFF, b | 0x8000)


@gluon.jit
def _unordered(key):
    """The FP32 value of an :func:`_ordered` key."""
    b = gl.where((key & 0x8000) != 0, key & 0x7FFF, (~key) & 0xFFFF)
    return (b << 16).to(gl.float32, bitcast=True)


@gluon.jit
def lmh_stage(mb, task, x, w, scale):
    """Lmh task ``task``: the logits of shard rows ``112 task ..`` for every
    step, then each step's best 16 of them into ``tk``."""
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    NW: gl.constexpr = gl.num_warps()
    KP: gl.constexpr = LMH_KS // 32
    mb.stamp("lmh", trace.START)
    nb0 = task * TASK_BLOCKS
    last = gl.minimum(TASK_BLOCKS, BLOCKS - nb0) - 1
    words = gl.allocate_shared_memory(
        gl.int32, [RP, KP, 16], mx._padded_rows([RP, KP, 16], KP * 16, 8)
    )
    tiles = _lmh_tiles(w, nb0, last, 0, NW)
    acc = bf16.gemv_template(1, NW, RP)
    for kr in gl.static_range(HIDDEN // LMH_KS):
        if kr + 1 < HIDDEN // LMH_KS:
            nxt = _lmh_tiles(w, nb0, last, (kr + 1) * (LMH_KS // 32), NW)
        xs = _x_round(x, words, kr, R)
        acc += bf16.gemv(xs, tiles, FIRST=0, ROWS=R)
        if kr + 1 < HIDDEN // LMH_KS:
            tiles = nxt
    mb.stamp("lmh", trace.COMPUTED)
    bl: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [NW, 1], [1, 0])
    v = gl.convert_layout(acc, bl)
    r = gl.arange(0, RP, gl.SliceLayout(1, bl))[:, None]
    n = gl.arange(0, 16 * NW, gl.SliceLayout(0, bl))[None, :]
    lid = nb0 * 16 + n
    valid = (n // 16 <= last) & (lid < SHARD) & (r < R)
    # The GEMM's BF16 logits, then the in-place BF16 multiply by the scale.
    y = (v.to(gl.bfloat16).to(gl.float32) * scale).to(gl.bfloat16)
    key = _ordered(y.to(gl.int16, bitcast=True).to(gl.int32))
    # Larger is better; on equal values the lower row wins.
    packed = gl.where(valid, (key << ID_BITS) | ((1 << ID_BITS) - 1 - lid), -1)
    j = gl.arange(0, TOPK, gl.SliceLayout(0, bl))[None, :]
    best = gl.full([RP, TOPK], -1, gl.int32, bl)
    for i in range(TOPK):  # a runtime loop, as in _best16
        m = gl.max(packed, axis=1)[:, None]
        best = gl.where(j == i, m, best)
        packed = gl.where(packed == m, -1, packed)
    mb.put("tk", mb.row_offset(r) + task * TOPK + j, best, mask=(r < R) & (j >= 0))
    mb.stamp("lmh", trace.END)


@gluon.jit
def hproj_stage(mb, task, x, w):
    """Hproj task ``task``: the selector's hidden projection, rows
    ``128 (task // 8) ..``, K part ``task % 8``, into ``hpart``."""
    R: gl.constexpr = schedule.rows(mb.plan)
    tiles = draft_mlp.proj_issue(task, w, RANK_DIM, HIDDEN, 128, HPROJ_KS)
    mb.stamp("hproj", trace.START)
    xs = bf16.load_global_rows(x, HIDDEN, (task % HPROJ_KPARTS) * HPROJ_KS, HPROJ_KS, R)
    draft_mlp.proj_compute(
        mb, task, tiles, xs, "hpart", RANK_DIM, HIDDEN, 128, HPROJ_KS, "hproj"
    )


@gluon.jit
def _best16(values, axis: gl.constexpr, layout: gl.constexpr):
    """The 16 largest of ``values`` along ``axis``, largest first, as a new
    last dimension of 16 (the values are distinct; -1 is below them all)."""
    j = gl.arange(0, TOPK, layout)
    if axis == 1:
        j = j[None, :]
    best = gl.full(
        j.shape if axis == 0 else [values.shape[0], TOPK],
        -1,
        values.dtype,
        j.type.layout,
    )
    # A runtime loop: unrolled, the passes are straight-line code that runs
    # once a step on a few CUs, cold in the instruction cache in serving.
    for i in range(TOPK):
        m = gl.max(values, axis=axis)
        if axis == 1:
            m = m[:, None]
        best = gl.where(j == i, m, best)
        values = gl.where(values == m, -1, values)
    return best


@gluon.jit
def merge_stages(mb, row, RANK: gl.constexpr, softcap, cand_ids, unary):
    """Merge1 and merge2 of step ``row``. Returns the step's sorted
    candidate ids (int32) and unary scores (FP32), ``[16]`` in a layout every
    warp holds whole."""
    nw: gl.constexpr = gl.num_warps()
    # A warp per row of 512 candidates: its best 16 with in-warp reductions.
    wl: gl.constexpr = gl.BlockedLayout([1, 8], [1, 64], [nw, 1], [1, 0])
    # 128 values that every warp holds whole: in-warp reductions again.
    ww: gl.constexpr = gl.BlockedLayout([2], [64], [nw], [0])
    mr = mb.row(row)
    # merge1: the rank's best 16 of the lmh tasks' 16s.
    mb.stamp("merge1", trace.START)
    N1: gl.constexpr = LMH_TASKS * TOPK
    W1: gl.constexpr = triton.next_power_of_2(N1) // nw
    w = gl.arange(0, nw, gl.SliceLayout(1, wl))[:, None]
    c = gl.arange(0, W1, gl.SliceLayout(0, wl))[None, :]
    i1 = w * W1 + c
    (got,) = mr.poll_many(("tk",), (gl.minimum(i1, N1 - 1),))
    mb.stamp("merge1", trace.READY)
    per_warp = _best16(gl.where(i1 < N1, got, -1), 1, gl.SliceLayout(0, wl))
    flat = gl.convert_layout(gl.reshape(per_warp, [nw * TOPK]), ww)
    best = _best16(flat, 0, ww)
    j = gl.arange(0, TOPK, ww)
    for p in gl.static_range(WORLD):
        mr.put_peer("rk", p, RANK * TOPK + j, best)
    mb.stamp("merge1", trace.END)
    # merge2: the best 16 of every rank's 16, by value, then token id.
    mb.stamp("merge2", trace.START)
    i2 = gl.arange(0, WORLD * TOPK, ww)
    (got,) = mr.poll_many(("rk",), (i2,))
    mb.stamp("merge2", trace.READY)
    lid = (1 << ID_BITS) - 1 - (got & ((1 << ID_BITS) - 1))
    gid = (i2 // TOPK) * SHARD + lid
    key = ((got >> ID_BITS).to(gl.int64) << GID_BITS) | ((1 << GID_BITS) - 1 - gid).to(
        gl.int64
    )
    best = _best16(key, 0, ww)
    ids = ((1 << GID_BITS) - 1 - (best & ((1 << GID_BITS) - 1))).to(gl.int32)
    value = _unordered((best >> GID_BITS).to(gl.int32))
    # The regular path's softcap: c (2 sigmoid(2 x / c) - 1).
    u = softcap * (2.0 / (1.0 + gl.exp(-2.0 * (value / softcap))) - 1.0)
    gl.store(cand_ids + row * TOPK + j, ids.to(gl.int64))
    gl.store(unary + row * TOPK + j, u)
    mr.put("cid", j, ids)
    mb.stamp("merge2", trace.END)
    return ids, u


@gluon.jit
def score_stage(mb, row, ids, u, first_token, anchor, pred_cb, succ_cb, scores):
    """Score task of step row ``row`` (request ``row // 7``'s step ``row %
    7``): the selector's 16 x 16 edge scores ``scores[row, p, c]`` from the
    previous step's candidate ``p`` (the request's anchor token for its step
    0) to this step's candidate ``c``."""
    mb.stamp("score", trace.START)
    nw: gl.constexpr = gl.num_warps()
    l3: gl.constexpr = gl.BlockedLayout([1, 1, 4], [1, 1, 64], [nw, 1, 1], [2, 1, 0])
    lp: gl.constexpr = gl.SliceLayout(1, l3)  # [p, r]
    lc: gl.constexpr = gl.SliceLayout(0, l3)  # [c, r]
    ls: gl.constexpr = gl.SliceLayout(2, l3)  # [p, c]
    mr = mb.row(row)
    # The hidden projection: its K parts in order, rounded to BF16.
    hl: gl.constexpr = gl.BlockedLayout([1, 4], [1, 64], [nw, 1], [1, 0])
    k = gl.arange(0, HPROJ_KPARTS, gl.SliceLayout(1, hl))[:, None]
    rr = gl.arange(0, RANK_DIM, gl.SliceLayout(0, hl))[None, :]
    (hp,) = mr.poll_many(("hpart",), (k * RANK_DIM + rr,))
    h = gl.sum(hp.to(gl.float32, bitcast=True), axis=0).to(gl.bfloat16).to(gl.float32)
    h = gl.convert_layout(h, gl.SliceLayout(0, lp))
    # The predecessors: the previous step's candidates (or the anchor).
    pj = gl.arange(0, TOPK, gl.SliceLayout(1, lp))
    request = row // REQUEST_STEPS
    if row % REQUEST_STEPS == 0:
        token = gl.load(first_token + request).to(gl.int64)
        token = gl.minimum(gl.maximum(token, 0), VOCAB_IDS - 1)
        gl.store(anchor + request, token)
        pid = gl.zeros([TOPK], gl.int64, gl.SliceLayout(1, lp)) + token
    else:
        (prev,) = mb.row(row - 1).poll_many(("cid",), (pj,))
        pid = prev.to(gl.int64)
    mb.stamp("score", trace.READY)
    r_p = gl.arange(0, RANK_DIM, gl.SliceLayout(0, lp))
    pred = gl.load(pred_cb + pid[:, None] * RANK_DIM + r_p[None, :]).to(gl.float32)
    q = (pred * h[None, :]).to(gl.bfloat16).to(gl.float32)
    cid = gl.convert_layout(ids, gl.SliceLayout(1, lc)).to(gl.int64)
    r_c = gl.arange(0, RANK_DIM, gl.SliceLayout(0, lc))
    succ = gl.load(succ_cb + cid[:, None] * RANK_DIM + r_c[None, :]).to(gl.float32)
    edge = gl.sum(q[:, None, :] * succ[None, :, :], axis=2)
    edge = edge.to(gl.bfloat16).to(gl.float32)
    uc = gl.convert_layout(u, gl.SliceLayout(0, ls))
    p = gl.arange(0, TOPK, gl.SliceLayout(1, ls))[:, None]
    c = gl.arange(0, TOPK, gl.SliceLayout(0, ls))[None, :]
    gl.store(scores + (row * TOPK + p) * TOPK + c, uc[None, :] + edge)
    mb.stamp("score", trace.END)


def _launch_metadata(grid, kernel, args):
    """Report the launch's rows (draft steps) and the weight bytes it reads
    to Proton: the LM head and hidden projection shards (the codebook rows
    it gathers are a few KiB)."""
    rows = args["PLAN"].plan.rows
    weight = args["lmh_w"].nbytes + args["hproj_w"].nbytes
    return {"name": f"{kernel.name} (rows={rows})", "bytes": weight}


# The code range is the compiled binary's own, known only after the compile.
@gluon.jit(
    launch_metadata=_launch_metadata, do_not_specialize=("code_offset", "code_quads")
)
def gluon_muse_spark_draft_head_gfx950(
    scratch,
    step,
    bases,
    x,
    lmh_w,
    hproj_w,
    pred_cb,
    succ_cb,
    first_token,
    anchor,
    cand_ids,
    unary,
    scores,
    scale,
    softcap,
    code_offset,
    code_quads,
    trace_buf,
    PLAN: gl.constexpr,
    RANK: gl.constexpr,
):
    """The draft head of one or two requests' draft steps on rank ``RANK`` (see the
    module docstring and :func:`launch_gluon_muse_spark_draft_head_gfx950`)."""
    table_ = trace.allocate(trace_buf)
    peers = mailbox.load_peers(bases, RANK, gl.load(step) & 1, WORLD)
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, table_, peers)
    task, active = schedule.stage_task(PLAN, "code")
    if active:
        code_warm.code_warm_stage(
            mb,
            task,
            schedule.stage_tasks(PLAN, "code"),
            scratch,
            code_offset,
            code_quads,
            "code",
        )
    task, active = schedule.stage_task(PLAN, "lmh")
    if active:
        lmh_stage(mb, task, x, lmh_w, scale)
    task, active = schedule.stage_task(PLAN, "hproj")
    if active:
        hproj_stage(mb, task, x, hproj_w)
    row, active = schedule.stage_task(PLAN, "merge1")
    if active:
        ids, u = merge_stages(mb, row, RANK, softcap, cand_ids, unary)
        score_stage(mb, row, ids, u, first_token, anchor, pred_cb, succ_cb, scores)
    trace.flush(table_, trace_buf)


def launch_gluon_muse_spark_draft_head_gfx950(
    weights: DraftHeadWeights,
    *,
    x: torch.Tensor,
    first_token: torch.Tensor,
    anchor: torch.Tensor,
    cand_ids: torch.Tensor,
    unary: torch.Tensor,
    scores: torch.Tensor,
    scratch: torch.Tensor,
    step: torch.Tensor,
    peer_bases: torch.Tensor,
    trace_buf,
    compile_only: bool,
) -> None:
    """The draft head of one or two requests' draft steps, on every rank of
    the TP8 group at once (the launches exchange candidates).

    Args:
        weights: From :func:`pack_draft_head`.
        x: BF16 ``[steps, 8192]`` contiguous: the draft's final-norm output
            of each request's block rows 1 .. 7, request by request (7 rows
            a request; the same on every rank).
        first_token: int32 ``[requests]``: each block's first token (its
            step 0's predecessor).
        anchor: int64 ``[requests]`` out: those tokens clamped to the
            vocabulary.
        cand_ids: int64 ``[steps, 16]`` out: each step's candidate token ids,
            by descending unary score.
        unary: FP32 ``[steps, 16]`` out: their softcapped logits.
        scores: FP32 ``[steps, 16, 16]`` out: the selector's edge scores from
            the previous step's candidate ``p`` to this step's ``c``.
        scratch: int64 mailbox pairs of ``head_plan(steps).scratch_pairs`` on
            the Iris symmetric heap.
        step: int32 ``[1]`` counter that changes from launch to launch.
        peer_bases: int64 ``[8]`` heap bases of the group's ranks.
        trace_buf: None, or a trace buffer (``trace.py``).
        compile_only: Only compile the launch for these operands.
    """
    rows = x.shape[0]
    requests = rows // STEPS
    if rows % STEPS or not 1 <= requests <= MAX_REQUESTS:
        raise ValueError(
            f"the head serves {STEPS} draft steps a request, at most "
            f"{MAX_REQUESTS} requests, got {rows} rows"
        )
    if (
        x.dtype != torch.bfloat16
        or tuple(x.shape) != (rows, 8192)
        or not x.is_contiguous()
    ):
        raise ValueError(f"x: contiguous BF16 [{rows}, 8192]")
    for name, t, dtype, shape in (
        ("first_token", first_token, torch.int32, (requests,)),
        ("anchor", anchor, torch.int64, (requests,)),
        ("cand_ids", cand_ids, torch.int64, (rows, 16)),
        ("unary", unary, torch.float32, (rows, 16)),
        ("scores", scores, torch.float32, (rows, 16, 16)),
    ):
        if t.dtype != dtype or tuple(t.shape) != shape or not t.is_contiguous():
            raise ValueError(f"{name}: contiguous {dtype} {shape}")
    plan = head_plan(rows)
    w = weights

    def kernel_args(code: tuple[int, int]) -> tuple:
        return (
            scratch,
            step,
            peer_bases,
            x,
            w.lm_head,
            w.hproj,
            w.pred,
            w.succ,
            first_token,
            anchor,
            cand_ids,
            unary,
            scores,
            w.scale,
            w.softcap,
            *code,
            trace_buf,
            plan.key,
            w.rank,
        )

    kernel = gluon_muse_spark_draft_head_gfx950.warmup(
        *kernel_args((0, 0)), num_warps=8, grid=(plan.grid,)
    )
    if compile_only:
        return
    # The code range is a runtime argument: the same binary.
    gluon_muse_spark_draft_head_gfx950[(plan.grid,)](
        *kernel_args(code_warm.text_range(kernel)), num_warps=8
    )


__all__ = [
    "MAX_REQUESTS",
    "BASES",
    "DraftHeadWeights",
    "STEPS",
    "gluon_muse_spark_draft_head_gfx950",
    "head_plan",
    "launch_gluon_muse_spark_draft_head_gfx950",
    "pack_draft_head",
    "regions",
    "stages",
]
