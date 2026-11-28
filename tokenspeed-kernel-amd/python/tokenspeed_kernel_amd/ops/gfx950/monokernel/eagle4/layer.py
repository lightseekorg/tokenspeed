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

"""One chain decode step of an EAGLE-4 draft layer, with the draft's final
norm, as one launch (TP1, BF16 weights).

The drafter (``runtime/models/llama_eagle4.py``, an EAGLE-3 layer without
``attn_norm``, MoE or NoPE) runs one decoder layer per draft step: each row
is one request's next draft token. The launch replaces the regular path's
kernels from the input norms to the final norm, with the same roundings:

* inorm, 20 tasks per row: the input RMSNorms, ``norm(embeds) ||
  norm(hidden)``, 512 columns a task (each task sums the squares of its
  whole half from global memory), put as BF16 into ``xin``.
* qkv (rounds): ``xin W_qkv^T`` (15360 x 10240), 128 rows x 512-wide K parts,
  FP32 partials into ``qpart`` (``draft_mlp.proj_task``).
* head, 120 tasks per row (40 q, 40 k, 40 v heads): the K parts summed and
  rounded to BF16, NeoX RoPE on q and k in FP32, rounded once. q goes to
  ``qh``; k and v go to the KV cache at the row's slot and to ``kb`` /
  ``vb``: a CTA never reads another CTA's store of the same launch, so the
  attention takes the row's own key from the mailbox.
* attn, ``attn_splits(rows)`` tasks per row and head: one split of the
  cached window ``[max(seq_len - 4096, 0), seq_len - 1)`` (the layer's
  ``window_left`` of 4095 plus the row's own key), 256 keys a step, 32 a
  wave. Each wave keeps its own online base-2 softmax, so a step needs no
  barrier; the waves merge once at the end, and the last split adds the
  row's own key. A task stores ``acc / l`` and the LSE as plain words, then
  raises one flag.
* merge, 40 tasks per row: the LSE merge of the splits, BF16, into ``ao``.
* oproj (rounds): ``ao W_o^T`` (5120 x 5120), FP32 partials into ``opart``.
* osum, pnorm, 40 tasks of 128 columns per row: the post-attention
  RMSNorm with the residual (the layer's input ``hidden``): ``s = bf16(o) +
  hidden`` in FP32, the sum of squares into ``ss``, then ``bf16(s rsqrt(
  mean(s^2) + eps) w)`` into ``hin``.
* gu (rounds), act (256 tasks of 64 columns), down (rounds): the MLP, each
  projection rounded to BF16 and ``silu(g) u`` rounded to BF16.
* dsum, fnorm, 40 tasks of 128 columns per row: the final RMSNorm of the MLP
  output plus the BF16 residual ``bf16(s)``, stored in ``normed``.

The sum and norm stages recompute ``s`` (and the MLP output) from the
partials instead of carrying registers from stage to stage, so their tasks
can run in rounds at any row count.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch
from tokenspeed_kernel_amd._extern import extern_compile_options
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.monokernel import (
    code_warm,
    mailbox,
    mx,
    schedule,
    trace,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import draft_mlp
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import Plan, Stage
from tokenspeed_kernel_amd.ops.gfx950.monokernel.weights import bf16_tiles

# The drafter's dimensions (Qwen3.8-27B-EAGLE4): the launcher checks the
# weights against them.
HIDDEN = gl.constexpr(5120)
HEADS = gl.constexpr(40)  # q heads, and as many KV heads
HEAD = gl.constexpr(128)
HALF = gl.constexpr(64)  # rotary pairs of a head (NeoX: i and i + 64)
QKV = gl.constexpr(3 * 5120)  # q, k, v rows
QKV_IN = gl.constexpr(2 * 5120)  # norm(embeds) || norm(hidden)
INTER = gl.constexpr(16384)
GU = gl.constexpr(2 * 16384)  # gate then up rows
EPS = gl.constexpr(1e-6)
PAGE = gl.constexpr(64)  # tokens a KV page
WINDOW = gl.constexpr(4096)  # keys a row attends to, its own included
WAVE_KEYS = gl.constexpr(32)  # keys a wave takes each attention step
# exp2 softmax of (q . k) / sqrt(128).
QK_SCALE = gl.constexpr(0.08838834764831845 * 1.4426950408889634)

# Projection tasks: 128 output rows x a 512-wide K part (128 KiB of weight).
PROJ_ROWS = gl.constexpr(128)
PROJ_KS = gl.constexpr(512)
QKV_KP = gl.constexpr(QKV_IN // PROJ_KS)  # 20
O_KP = gl.constexpr(HIDDEN // PROJ_KS)  # 10
GU_KP = gl.constexpr(HIDDEN // PROJ_KS)  # 10
DOWN_KP = gl.constexpr(INTER // PROJ_KS)  # 32

INORM_COLS = gl.constexpr(512)
INORM_TASKS = gl.constexpr(QKV_IN // INORM_COLS)  # per row
HEAD_TASKS = gl.constexpr(3 * HEADS)  # per row: q heads, k heads, v heads
SUM_COLS = gl.constexpr(128)
SUM_TASKS = gl.constexpr(HIDDEN // SUM_COLS)  # 40 per row
ACT_COLS = gl.constexpr(64)
ACT_TASKS = gl.constexpr(INTER // ACT_COLS)  # 256 for every row
# CTAs 0 .. CODE_TASKS - 1 first read the launch's own code into the L2s
# (code_warm.py): 8 tasks on each XCD.
CODE_TASKS = 64
GRID = 256
MAX_ROWS = 16  # the projections' GEMV covers 16 rows at most
MAX_SPLITS = 8


@gluon.constexpr_function
def attn_splits(rows):
    """Attention splits per row and head (also callable on the host).

    The tasks run in rounds over the 256 CTAs. A CTA's cost is its rounds
    times a task's: ``1 / splits`` of a head's keys plus a fixed part (the
    poll, the cache loads' first latency, the flag and the merge), taken as
    1/8 of a head (one step of a 2048-key window). The count of 1 to 8 with
    the least cost wins, the fewest on a tie."""
    best, best_cost = 1, None
    for splits in range(1, 9):
        rounds = (40 * rows * splits + 255) // 256
        cost = rounds * (8 + 64 // splits)  # in 1/64 of a head
        if best_cost is None or cost < best_cost:
            best, best_cost = splits, cost
    return best


def regions() -> tuple[tuple[str, int], ...]:
    """The layer's regions as (name, int64 pairs), one copy per row."""
    hidden, heads, head = int(HIDDEN), int(HEADS), int(HEAD)
    return (
        ("xin", int(QKV_IN) // 2),  # BF16 pairs of the attention input
        ("qpart", int(QKV_KP) * int(QKV)),  # FP32 [K part][q, k, v row]
        ("qh", heads * head // 2),  # BF16 pairs of the rotated q heads
        ("kb", heads * head // 2),  # BF16 pairs of the row's rotated k heads
        ("vb", heads * head // 2),  # BF16 pairs of the row's v heads
        ("apart", heads * MAX_SPLITS * head // 2),  # FP32 words [head][split][dim]
        ("alse", heads * MAX_SPLITS // 2),  # FP32 words [head][split]
        ("aflag", heads * MAX_SPLITS),  # a flag per attn task
        ("ao", heads * head // 2),  # BF16 pairs of the attention output
        ("opart", int(O_KP) * hidden),  # FP32 [K part][column]
        ("ss", int(SUM_TASKS)),  # FP32 sums of squares of s
        ("hin", hidden // 2),  # BF16 pairs of the MLP input
        ("gpart", int(GU_KP) * int(GU)),  # FP32 [K part][gate, up row]
        ("act", int(INTER) // 2),  # BF16 pairs of silu(g) u
        ("dpart", int(DOWN_KP) * hidden),  # FP32 [K part][column]
        ("ss2", int(SUM_TASKS)),  # FP32 sums of squares before the final norm
    )


# Every stage starts at CTA 0 but the input norm, which runs on the CTAs
# past the code tasks.
BASES = {
    "inorm": CODE_TASKS,
    "qkv": 0,
    "head": 0,
    "attn": 0,
    "merge": 0,
    "oproj": 0,
    "osum": 0,
    "pnorm": 0,
    "gu": 0,
    "act": 0,
    "down": 0,
    "dsum": 0,
    "fnorm": 0,
}


def stages(rows: int) -> tuple[Stage, ...]:
    """The stages of ``rows`` rows, placed by :data:`BASES`."""
    b = BASES
    sums = int(SUM_TASKS) * rows
    proj = int(PROJ_ROWS)
    return (
        Stage("code", CODE_TASKS, 0),
        Stage("inorm", int(INORM_TASKS) * rows, b["inorm"], writes=("xin",)),
        Stage(
            "qkv",
            int(QKV) // proj * int(QKV_KP),
            b["qkv"],
            reads=("xin",),
            writes=("qpart",),
        ),
        Stage(
            "head",
            int(HEAD_TASKS) * rows,
            b["head"],
            reads=("qpart",),
            writes=("qh", "kb", "vb"),
        ),
        Stage(
            "attn",
            int(HEADS) * attn_splits(rows) * rows,
            b["attn"],
            reads=("qh", "kb", "vb"),
            writes=("apart", "alse", "aflag"),
        ),
        Stage(
            "merge",
            int(HEADS) * rows,
            b["merge"],
            reads=("apart", "alse", "aflag"),
            writes=("ao",),
        ),
        Stage(
            "oproj",
            int(HIDDEN) // proj * int(O_KP),
            b["oproj"],
            reads=("ao",),
            writes=("opart",),
        ),
        Stage("osum", sums, b["osum"], reads=("opart",), writes=("ss",)),
        Stage("pnorm", sums, b["pnorm"], reads=("opart", "ss"), writes=("hin",)),
        Stage(
            "gu",
            int(GU) // proj * int(GU_KP),
            b["gu"],
            reads=("hin",),
            writes=("gpart",),
        ),
        Stage("act", int(ACT_TASKS), b["act"], reads=("gpart",), writes=("act",)),
        Stage(
            "down",
            int(HIDDEN) // proj * int(DOWN_KP),
            b["down"],
            reads=("act",),
            writes=("dpart",),
        ),
        Stage("dsum", sums, b["dsum"], reads=("opart", "dpart"), writes=("ss2",)),
        Stage("fnorm", sums, b["fnorm"], reads=("opart", "dpart", "ss2")),
    )


@functools.lru_cache(maxsize=None)
def layer_plan(rows: int) -> Plan:
    """The plan of ``rows`` rows (one draft token of as many requests).

    The row count is compile-time, as in every monokernel plan: the stages'
    task counts and the GEMV's MFMA rows follow it. A server compiles the
    row counts it serves ahead of time (one binary each)."""
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"1 to {MAX_ROWS} rows, got {rows}")
    return Plan(grid=GRID, stages=stages(rows), regions=regions(), rows=rows, chain=1)


@dataclass(frozen=True)
class Eagle4DraftLayerWeights:
    """The EAGLE-4 draft layer packed for the launch: the projections in
    ``weights.bf16_tiles`` order, the norm gains as the checkpoint's BF16."""

    input_norm: torch.Tensor
    hidden_norm: torch.Tensor
    qkv: torch.Tensor
    o: torch.Tensor
    post_norm: torch.Tensor
    gu: torch.Tensor
    down: torch.Tensor
    final_norm: torch.Tensor


def pack_eagle4_draft_layer(
    tensors: dict[str, torch.Tensor],
) -> Eagle4DraftLayerWeights:
    """Pack the draft layer's weights (copies of the projections).

    Args:
        tensors: BF16 by role: ``input_norm``, ``hidden_norm``,
            ``post_norm``, ``final_norm`` ``[5120]``; ``qkv`` ``[15360,
            10240]`` (q, k, v rows); ``o_proj`` ``[5120, 5120]``; ``gate_up``
            ``[32768, 5120]`` (gate rows, then up rows); ``down`` ``[5120,
            16384]``.
    """
    hidden = int(HIDDEN)
    shapes = {
        "input_norm": (hidden,),
        "hidden_norm": (hidden,),
        "post_norm": (hidden,),
        "final_norm": (hidden,),
        "qkv": (int(QKV), int(QKV_IN)),
        "o_proj": (hidden, hidden),
        "gate_up": (int(GU), hidden),
        "down": (hidden, int(INTER)),
    }
    for name, shape in shapes.items():
        t = tensors[name]
        if t.dtype != torch.bfloat16 or tuple(t.shape) != shape:
            raise ValueError(f"{name}: BF16 {shape}, got {t.dtype} {tuple(t.shape)}")
    return Eagle4DraftLayerWeights(
        input_norm=tensors["input_norm"].contiguous(),
        hidden_norm=tensors["hidden_norm"].contiguous(),
        qkv=bf16_tiles(tensors["qkv"].contiguous()),
        o=bf16_tiles(tensors["o_proj"].contiguous()),
        post_norm=tensors["post_norm"].contiguous(),
        gu=bf16_tiles(tensors["gate_up"].contiguous()),
        down=bf16_tiles(tensors["down"].contiguous()),
        final_norm=tensors["final_norm"].contiguous(),
    )


@gluon.jit
def inorm_task(mb, task, embeds, hidden, in_gain, hidden_gain):
    """Inorm task ``task``: columns ``512 (task % 10) ..`` of half ``task //
    10 % 2`` (embeds, then hidden) of row ``task // 20``."""
    mb.stamp("inorm", trace.START)
    row, t, _ = schedule.row_task(mb.plan, task, INORM_TASKS)
    parts: gl.constexpr = HIDDEN // INORM_COLS
    half = t // parts
    part = t % parts
    nw: gl.constexpr = gl.num_warps()
    fl: gl.constexpr = gl.BlockedLayout([8], [64], [nw], [0])
    cl: gl.constexpr = gl.BlockedLayout([1], [64], [nw], [0])
    f = gl.arange(0, triton.next_power_of_2(HIDDEN), fl)
    c = part * INORM_COLS + gl.arange(0, INORM_COLS, cl)
    at = row * HIDDEN
    # Both inputs come from earlier launches: plain loads.
    if half == 0:
        x = gl.load(embeds + at + f, mask=f < HIDDEN, other=0.0).to(gl.float32)
        y = gl.load(embeds + at + c).to(gl.float32)
        w = gl.load(in_gain + c).to(gl.float32)
    else:
        x = gl.load(hidden + at + f, mask=f < HIDDEN, other=0.0).to(gl.float32)
        y = gl.load(hidden + at + c).to(gl.float32)
        w = gl.load(hidden_gain + c).to(gl.float32)
    inv = gl.rsqrt(gl.sum(x * x, axis=0) / HIDDEN + EPS)
    mx.put_bf16(
        mb.row(row), "xin", half * (HIDDEN // 2) + part * (INORM_COLS // 2), y * inv * w
    )
    mb.stamp("inorm", trace.END)


@gluon.jit
def head_task(mb, task, cos_sin, positions, slots, k_cache, v_cache):
    """Head task ``task``: head ``t % 40`` of q (``t < 40``), k or v (``t =
    task % 120``) of row ``task // 120``."""
    mb.stamp("head", trace.START)
    row, t, _ = schedule.row_task(mb.plan, task, HEAD_TASKS)
    kind = t // HEADS  # 0: q, 1: k, 2: v
    head = t % HEADS
    mb = mb.row(row)
    nw: gl.constexpr = gl.num_warps()
    KQ: gl.constexpr = triton.next_power_of_2(QKV_KP)
    kl: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [nw, 1], [1, 0])
    k = gl.arange(0, KQ, gl.SliceLayout(1, kl))[:, None]
    # Dimension d of the head and its NeoX partner d +- 64.
    d = gl.arange(0, HEAD, gl.SliceLayout(0, kl))
    partner = (d + HALF) % HEAD
    # The parameters load before the poll: a load after a poll would start
    # only when the poll returns.
    position = gl.load(positions + row).to(gl.int64)
    cos = gl.load(cos_sin + position * HEAD + d % HALF)
    sin = gl.load(cos_sin + position * HEAD + HALF + d % HALF)
    slot = gl.load(slots + row).to(gl.int64)
    at = gl.minimum(k, QKV_KP - 1) * QKV + kind * HIDDEN + head * HEAD
    own_w, partner_w = mb.poll_many(
        ("qpart", "qpart"), (at + d[None, :], at + partner[None, :])
    )
    mb.stamp("head", trace.READY)
    live = k < QKV_KP
    # The projection's output is BF16.
    x = gl.sum(gl.where(live, own_w.to(gl.float32, bitcast=True), 0.0), axis=0)
    y = gl.sum(gl.where(live, partner_w.to(gl.float32, bitcast=True), 0.0), axis=0)
    x = x.to(gl.bfloat16).to(gl.float32)
    y = y.to(gl.bfloat16).to(gl.float32)
    # NeoX RoPE as the attention prologue computes it: fma(x1, cos, -x2 sin)
    # for the first half, fma(x2, cos, x1 sin) for the second.
    turn = gl.where(d < HALF, -(y * sin), y * sin)
    vals = gl.where(kind < 2, gl.fma(x, cos, turn), x).to(gl.bfloat16)
    cols = slot * (HEADS * HEAD) + head * HEAD + d
    if kind == 0:
        mx.put_bf16(mb, "qh", head * HALF, vals)
    elif kind == 1:
        mx.put_bf16(mb, "kb", head * HALF, vals)
        gl.store(k_cache + cols, vals)
    else:
        mx.put_bf16(mb, "vb", head * HALF, vals)
        gl.store(v_cache + cols, vals)
    mb.stamp("head", trace.END)


@gluon.constexpr_function
def _kv_layout(num_warps):
    """``[waves, 32 keys, 128 dims]``: a wave per key group, 4 lanes along
    the keys and 16 along the dimensions (8 contiguous each)."""
    return gl.BlockedLayout([1, 1, 8], [1, 4, 16], [num_warps, 1, 1], [2, 1, 0])


@gluon.jit
def _load_kv(tab, k_cache, v_cache, c0, hi, key, d, hoff):
    """Keys and values ``[waves, 32, 128]`` (BF16) of positions ``c0 + key``
    of one head (zero from ``hi``)."""
    at = c0 + key
    live = at < hi
    page = gl.load(tab + at // PAGE, mask=live, other=0)
    slot = page.to(gl.int64) * PAGE + at % PAGE
    off = slot * (HEADS * HEAD) + hoff + d
    keys = gl.load(k_cache + off, mask=live, other=0.0)
    values = gl.load(v_cache + off, mask=live, other=0.0)
    return keys, values


@gluon.jit
def attn_task(
    mb, task, table, table_stride, seq_len, k_cache, v_cache, SPLITS: gl.constexpr
):
    """Attn task ``task``: split ``s`` of head ``h`` of row ``r`` (``task =
    (r * 40 + h) * SPLITS + s``).

    Args:
        table: int32 page tables, row ``r``'s at ``table + r table_stride``.
        seq_len: int32 ``[rows]``: each row's length, its own token included.
        k_cache, v_cache: BF16 ``[slots, 40, 128]`` this layer's cache.
    """
    mb.stamp("attn", trace.START)
    row, t, _ = schedule.row_task(mb.plan, task, HEADS * SPLITS)
    head = t // SPLITS
    s = t % SPLITS
    mb = mb.row(row)
    nw: gl.constexpr = gl.num_warps()
    BN: gl.constexpr = nw * WAVE_KEYS
    kv: gl.constexpr = _kv_layout(nw)
    s2: gl.constexpr = gl.SliceLayout(2, kv)  # [waves, keys] and [waves, 1]
    dl: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, kv))
    wave = gl.arange(0, nw, gl.SliceLayout(1, s2))[:, None]
    key = wave * WAVE_KEYS + gl.arange(0, WAVE_KEYS, gl.SliceLayout(0, s2))[None, :]
    d = gl.arange(0, HEAD, dl)
    key3 = key[:, :, None]
    d3 = d[None, None, :]
    end = gl.load(seq_len + row)
    own = end - 1  # the row's own position: its key comes from kb
    first = gl.maximum(end - WINDOW, 0)
    per = (own - first + SPLITS - 1) // SPLITS
    lo = first + s * per
    hi = gl.minimum(lo + per, own)
    tab = table + row.to(gl.int64) * table_stride
    hoff = head * HEAD
    # The first step's cache loads go before the poll: they do not depend on
    # this launch.
    keys, values = _load_kv(tab, k_cache, v_cache, lo, hi, key3, d3, hoff)
    (q_w,) = mb.poll_many(("qh",), (head * HALF + d // 2,))
    mb.stamp("attn", trace.READY)
    q1 = mx.bf16_half(q_w, d)
    q = q1[None, None, :]
    m = gl.full([nw, 1], float("-inf"), gl.float32, s2)
    l = gl.zeros([nw, 1], gl.float32, s2)
    acc = gl.zeros([nw, 1, HEAD], gl.float32, kv)
    for c0 in range(lo, hi, BN):
        # The next step's loads go out before this step's math.
        next_k, next_v = _load_kv(tab, k_cache, v_cache, c0 + BN, hi, key3, d3, hoff)
        scores = gl.sum(keys.to(gl.float32) * q, axis=2) * QK_SCALE
        scores = gl.where(c0 + key < hi, scores, float("-inf"))
        m_new = gl.maximum(m, gl.max(scores, axis=1, keep_dims=True))
        # A wave whose keys are all masked keeps m at -inf: subtract 0.
        m_ref = gl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = gl.exp2(m - m_ref)
        p = gl.exp2(scores - m_ref)
        l = l * alpha + gl.sum(p, axis=1, keep_dims=True)
        pv = gl.sum(p[:, :, None] * values.to(gl.float32), axis=1, keep_dims=True)
        acc = acc * alpha[:, :, None] + pv
        m = m_new
        keys = next_k
        values = next_v
    # The waves' merge: the only cross-wave step.
    top = gl.max(m, axis=0, keep_dims=True)
    top_ref = gl.where(top == float("-inf"), 0.0, top)
    scale = gl.exp2(m - top_ref)  # 0 for a wave without keys
    l_all = gl.sum(gl.sum(l * scale, axis=0), axis=0)
    out = gl.sum(gl.sum(acc * scale[:, :, None], axis=0), axis=0)  # [128]
    m_all = gl.max(gl.max(top, axis=0), axis=0)
    # The row's own key, in the last split.
    ol: gl.constexpr = out.type.layout
    do = gl.arange(0, HEAD, ol)
    k_w, v_w = mb.poll_many(
        ("kb", "vb"), (head * HALF + do // 2, head * HALF + do // 2)
    )
    score = gl.sum(mx.bf16_half(k_w, do) * gl.convert_layout(q1, ol), axis=0)
    score = gl.where(s == SPLITS - 1, score * QK_SCALE, float("-inf"))
    m_new = gl.maximum(m_all, score)
    m_ref = gl.where(m_new == float("-inf"), 0.0, m_new)
    alpha = gl.exp2(m_all - m_ref)
    p = gl.exp2(score - m_ref)
    l_all = l_all * alpha + p
    out = out * alpha + p * mx.bf16_half(v_w, do)
    mb.stamp("attn", trace.COMPUTED)
    out = out / gl.where(l_all > 0, l_all, 1.0)
    lse = gl.where(
        l_all > 0, m_new + gl.log2(gl.where(l_all > 0, l_all, 1.0)), float("-inf")
    )
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [nw], [0])
    one = gl.arange(0, 1, lanes)
    mb.store_words("apart", (head * SPLITS + s) * HEAD + do, out)
    mb.store_words(
        "alse", head * SPLITS + s + one, gl.zeros([1], gl.float32, lanes) + lse
    )
    mb.raise_flag("aflag", head * SPLITS + s)
    mb.stamp("attn", trace.END)


@gluon.jit
def merge_task(mb, task, SPLITS: gl.constexpr):
    """Merge task ``task`` = head ``task % 40`` of row ``task // 40``: the
    LSE merge of its splits, rounded to BF16, into ``ao``."""
    mb.stamp("merge", trace.START)
    row, h, _ = schedule.row_task(mb.plan, task, HEADS)
    mb = mb.row(row)
    nw: gl.constexpr = gl.num_warps()
    SP: gl.constexpr = triton.next_power_of_2(SPLITS)
    ml: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [nw, 1], [1, 0])
    s = gl.arange(0, SP, gl.SliceLayout(1, ml))
    c = gl.arange(0, HEAD, gl.SliceLayout(0, ml))
    at = h * SPLITS + gl.minimum(s, SPLITS - 1)
    mb.poll_flags("aflag", at)
    # Each wave polls the flags: the words load after every wave's poll
    # returned.
    gl.barrier()
    got_p = mb.load_words("apart", at[:, None] * HEAD + c[None, :])
    got_l = mb.load_words("alse", at)
    mb.stamp("merge", trace.READY)
    lse = gl.where(s < SPLITS, got_l.to(gl.float32, bitcast=True), float("-inf"))
    top = gl.max(lse, axis=0)
    w = gl.where(lse == float("-inf"), 0.0, gl.exp2(lse - top))
    p = got_p.to(gl.float32, bitcast=True)
    out = gl.sum(p * w[:, None], axis=0) / gl.sum(w, axis=0)
    mb.stamp("merge", trace.COMPUTED)
    mx.put_bf16(mb, "ao", h * HALF, out)
    mb.stamp("merge", trace.END)


@gluon.jit
def _sum_layout():
    nw: gl.constexpr = gl.num_warps()
    return gl.BlockedLayout([1, 1], [1, 64], [nw, 1], [1, 0])


@gluon.jit
def _bf16_sum(mb, SRC: gl.constexpr, KP: gl.constexpr, col):
    """The BF16-rounded sum of the ``KP`` FP32 K parts of a projection's
    columns ``col`` (``[128]``) in ``SRC [K part][5120]`` of the row of
    ``mb``."""
    KQ: gl.constexpr = triton.next_power_of_2(KP)
    sl: gl.constexpr = _sum_layout()
    k = gl.arange(0, KQ, gl.SliceLayout(1, sl))[:, None]
    (got,) = mb.poll_many((SRC,), (gl.minimum(k, KP - 1) * HIDDEN + col[None, :],))
    total = gl.sum(gl.where(k < KP, got.to(gl.float32, bitcast=True), 0.0), axis=0)
    return total.to(gl.bfloat16).to(gl.float32)


@gluon.jit
def _attn_residual(mb, row, t, hidden):
    """``(col, s)``: columns ``128 t ..`` of row ``row`` and their FP32
    ``bf16(o) + hidden`` (the post-attention norm's sum, unrounded)."""
    sl: gl.constexpr = _sum_layout()
    col = t * SUM_COLS + gl.arange(0, SUM_COLS, gl.SliceLayout(0, sl))
    h = gl.load(hidden + row * HIDDEN + col).to(gl.float32)
    return col, _bf16_sum(mb, "opart", O_KP, col) + h


@gluon.jit
def _put_sumsq(mb, REGION: gl.constexpr, t, x):
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    total = gl.sum(x * x, axis=0)
    word = (gl.zeros([1], gl.float32, lanes) + total).to(gl.int32, bitcast=True)
    mb.put(REGION, t + gl.arange(0, 1, lanes), word)


@gluon.jit
def _rsqrt_mean(mb, REGION: gl.constexpr):
    """``rsqrt(mean(x^2) + eps)`` of a row from its tasks' sums of squares."""
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    i = gl.arange(0, triton.next_power_of_2(SUM_TASKS), lanes)
    (got,) = mb.poll_many((REGION,), (gl.minimum(i, SUM_TASKS - 1),))
    total = gl.sum(
        gl.where(i < SUM_TASKS, got.to(gl.float32, bitcast=True), 0.0), axis=0
    )
    return gl.rsqrt(total / HIDDEN + EPS)


@gluon.jit
def osum_task(mb, task, hidden):
    mb.stamp("osum", trace.START)
    row, t, _ = schedule.row_task(mb.plan, task, SUM_TASKS)
    mb = mb.row(row)
    _, s = _attn_residual(mb, row, t, hidden)
    mb.stamp("osum", trace.READY)
    _put_sumsq(mb, "ss", t, s)
    mb.stamp("osum", trace.END)


@gluon.jit
def pnorm_task(mb, task, hidden, gain):
    mb.stamp("pnorm", trace.START)
    row, t, _ = schedule.row_task(mb.plan, task, SUM_TASKS)
    mb = mb.row(row)
    col, s = _attn_residual(mb, row, t, hidden)
    w = gl.load(gain + col).to(gl.float32)
    inv = _rsqrt_mean(mb, "ss")
    mb.stamp("pnorm", trace.READY)
    mx.put_bf16(mb, "hin", t * (SUM_COLS // 2), s * inv * w)
    mb.stamp("pnorm", trace.END)


@gluon.jit
def _final_sum(mb, row, t, hidden):
    """``(col, s2)``: the final norm's FP32 sum of columns ``128 t ..``: the
    MLP output plus the BF16 residual after the attention sublayer."""
    col, s = _attn_residual(mb, row, t, hidden)
    return col, _bf16_sum(mb, "dpart", DOWN_KP, col) + s.to(gl.bfloat16).to(gl.float32)


@gluon.jit
def dsum_task(mb, task, hidden):
    mb.stamp("dsum", trace.START)
    row, t, _ = schedule.row_task(mb.plan, task, SUM_TASKS)
    mb = mb.row(row)
    _, s2 = _final_sum(mb, row, t, hidden)
    mb.stamp("dsum", trace.READY)
    _put_sumsq(mb, "ss2", t, s2)
    mb.stamp("dsum", trace.END)


@gluon.jit
def fnorm_task(mb, task, hidden, gain, normed):
    mb.stamp("fnorm", trace.START)
    row, t, _ = schedule.row_task(mb.plan, task, SUM_TASKS)
    mb = mb.row(row)
    col, s2 = _final_sum(mb, row, t, hidden)
    w = gl.load(gain + col).to(gl.float32)
    inv = _rsqrt_mean(mb, "ss2")
    mb.stamp("fnorm", trace.READY)
    gl.store(normed + row * HIDDEN + col, (s2 * inv * w).to(gl.bfloat16))
    mb.stamp("fnorm", trace.END)


@gluon.jit
def act_task(mb, task):
    """Act task ``task``: dimensions ``64 task ..`` of every row."""
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    KQ: gl.constexpr = triton.next_power_of_2(GU_KP)
    mb.stamp("act", trace.START)
    nw: gl.constexpr = gl.num_warps()
    ll: gl.constexpr = draft_mlp.kq_layout(KQ, ACT_COLS // 2, nw)
    r = gl.arange(0, RP, gl.SliceLayout(1, gl.SliceLayout(2, ll)))[:, None, None]
    k = gl.arange(0, KQ, gl.SliceLayout(0, gl.SliceLayout(2, ll)))[None, :, None]
    q = gl.arange(0, ACT_COLS // 2, gl.SliceLayout(0, gl.SliceLayout(1, ll)))[
        None, None, :
    ]
    at = (
        mb.row_offset(gl.minimum(r, R - 1))
        + gl.minimum(k, GU_KP - 1) * GU
        + task * ACT_COLS
        + 2 * q
    )
    # Two polls: one poll of all four (64 pairs a thread at 16 rows) made
    # an illegal memory access on gfx950; two of 32 pairs a thread do not.
    g_lo, g_hi = mb.poll_many(("gpart", "gpart"), (at, at + 1))
    u_lo, u_hi = mb.poll_many(("gpart", "gpart"), (at + INTER, at + INTER + 1))
    mb.stamp("act", trace.READY)
    live = k < GU_KP
    # The regular path's GEMV rounds gate and up to BF16 before silu * mul.
    lo_g = _parts_bf16(g_lo, live)
    hi_g = _parts_bf16(g_hi, live)
    lo_u = _parts_bf16(u_lo, live)
    hi_u = _parts_bf16(u_hi, live)
    lo = (lo_g / (1.0 + gl.exp(-lo_g)) * lo_u).to(gl.bfloat16)
    hi = (hi_g / (1.0 + gl.exp(-hi_g)) * hi_u).to(gl.bfloat16)
    # Packed by hand: mx.pack_bf16_pairs(gl.join(...)) returns the words in
    # another layout than the put's offsets.
    word = (lo.to(gl.int16, bitcast=True).to(gl.int32) & 0xFFFF) | (
        hi.to(gl.int16, bitcast=True).to(gl.int32) << 16
    )
    sl: gl.constexpr = gl.SliceLayout(1, ll)
    r2 = gl.arange(0, RP, gl.SliceLayout(1, sl))[:, None]
    q2 = gl.arange(0, ACT_COLS // 2, gl.SliceLayout(0, sl))[None, :]
    mb.put(
        "act",
        mb.row_offset(r2) + task * (ACT_COLS // 2) + q2,
        word,
        mask=(r2 < R) & (q2 >= 0),
    )
    mb.stamp("act", trace.END)


@gluon.jit
def _parts_bf16(words, live):
    total = gl.sum(gl.where(live, words.to(gl.float32, bitcast=True), 0.0), axis=1)
    return total.to(gl.bfloat16).to(gl.float32)


@gluon.jit
def _proj_rounds(
    mb,
    PLAN: gl.constexpr,
    w,
    SRC: gl.constexpr,
    DST: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    STAGE: gl.constexpr,
):
    """This CTA's tasks of projection stage ``STAGE`` (in rounds)."""
    G: gl.constexpr = schedule.grid(PLAN)
    task, rounds = schedule.stage_rounds(PLAN, STAGE)
    for i in range(rounds):
        draft_mlp.proj_task(
            mb, task + i * G, w, SRC, DST, N, K, PROJ_ROWS, PROJ_KS, STAGE
        )


def _launch_metadata(grid, kernel, args):
    """Report the launch's rows and the weight bytes it reads to Proton (the
    projections; the norm gains are a few KiB)."""
    rows = args["PLAN"].plan.rows
    weight = sum(args[name].nbytes for name in ("qkv_w", "o_w", "gu_w", "down_w"))
    return {"name": f"{kernel.name} (rows={rows})", "bytes": weight}


# The layer slot changes every draft step and the page tables' row stride
# with the table's width: a specialization on them would compile new
# binaries while serving. The code range is the compiled binary's own,
# known only after the compile.
@gluon.jit(
    launch_metadata=_launch_metadata,
    do_not_specialize=("layer", "table_stride", "code_offset", "code_quads"),
)
def gluon_eagle4_draft_layer_gfx950(
    scratch,
    step,
    layer,
    embeds,
    hidden,
    in_gain,
    hidden_gain,
    qkv_w,
    cos_sin,
    positions,
    slots,
    table,
    table_stride,
    seq_len,
    k_cache,
    v_cache,
    o_w,
    post_gain,
    gu_w,
    down_w,
    final_gain,
    normed,
    code_offset,
    code_quads,
    trace_buf,
    PLAN: gl.constexpr,
):
    """One draft step (see the module docstring and
    :func:`launch_gluon_eagle4_draft_layer_gfx950`). ``code_offset``,
    ``code_quads``: this binary's ``code_warm.text_range``."""
    table_ = trace.allocate(trace_buf)
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, layer), PLAN, table_)
    G: gl.constexpr = schedule.grid(PLAN)
    R: gl.constexpr = schedule.rows(PLAN)
    SPLITS: gl.constexpr = attn_splits(R)
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
    task, rounds = schedule.stage_rounds(PLAN, "inorm")
    for i in range(rounds):
        inorm_task(mb, task + i * G, embeds, hidden, in_gain, hidden_gain)
    _proj_rounds(mb, PLAN, qkv_w, "xin", "qpart", QKV, QKV_IN, "qkv")
    task, rounds = schedule.stage_rounds(PLAN, "head")
    for i in range(rounds):
        head_task(mb, task + i * G, cos_sin, positions, slots, k_cache, v_cache)
    task, rounds = schedule.stage_rounds(PLAN, "attn")
    for i in range(rounds):
        attn_task(
            mb, task + i * G, table, table_stride, seq_len, k_cache, v_cache, SPLITS
        )
    task, rounds = schedule.stage_rounds(PLAN, "merge")
    for i in range(rounds):
        merge_task(mb, task + i * G, SPLITS)
    _proj_rounds(mb, PLAN, o_w, "ao", "opart", HIDDEN, HIDDEN, "oproj")
    task, rounds = schedule.stage_rounds(PLAN, "osum")
    for i in range(rounds):
        osum_task(mb, task + i * G, hidden)
    task, rounds = schedule.stage_rounds(PLAN, "pnorm")
    for i in range(rounds):
        pnorm_task(mb, task + i * G, hidden, post_gain)
    _proj_rounds(mb, PLAN, gu_w, "hin", "gpart", GU, HIDDEN, "gu")
    task, active = schedule.stage_task(PLAN, "act")
    if active:
        act_task(mb, task)
    _proj_rounds(mb, PLAN, down_w, "act", "dpart", HIDDEN, INTER, "down")
    task, rounds = schedule.stage_rounds(PLAN, "dsum")
    for i in range(rounds):
        dsum_task(mb, task + i * G, hidden)
    task, rounds = schedule.stage_rounds(PLAN, "fnorm")
    for i in range(rounds):
        fnorm_task(mb, task + i * G, hidden, final_gain, normed)
    trace.flush(table_, trace_buf)


def _check(name: str, t: torch.Tensor, dtype, shape: tuple[int, ...]) -> None:
    if t.dtype != dtype or tuple(t.shape) != shape or not t.is_contiguous():
        raise ValueError(
            f"{name}: contiguous {dtype} {shape}, got {t.dtype} {tuple(t.shape)}"
        )


def launch_gluon_eagle4_draft_layer_gfx950(
    weights: Eagle4DraftLayerWeights,
    *,
    embeds: torch.Tensor,
    hidden: torch.Tensor,
    normed: torch.Tensor,
    positions: torch.Tensor,
    seq_len: torch.Tensor,
    slots: torch.Tensor,
    page_table: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cos_sin: torch.Tensor,
    scratch: torch.Tensor,
    step: torch.Tensor,
    layer: int,
    trace_buf,
    compile_only: bool,
) -> None:
    """One chain draft step of the packed layer and the final norm, for
    ``rows`` requests (one row each).

    Args:
        weights: From :func:`pack_eagle4_draft_layer`.
        embeds: BF16 ``[rows, 5120]``: the draft tokens' embeddings.
        hidden: BF16 ``[rows, 5120]``: the hidden state the step drafts from
            (the previous step's output), also the layer's residual.
        normed: BF16 ``[rows, 5120]``: the final norm's output.
        positions: int32 ``[rows]``: the rows' positions (RoPE).
        seq_len: int32 ``[rows]``: each request's length, the row's own
            token included (its key sits at ``seq_len - 1``).
        slots: int32 or int64 ``[rows]``: the rows' cache slots.
        page_table: int32 ``[>= rows, pages]``: row ``r`` is request ``r``'s
            page table (64 tokens a page).
        k_cache, v_cache: BF16 ``[slots, 40, 128]``: the draft's cache.
        cos_sin: FP32 ``[positions, 128]`` rotary table, cos then sin.
        scratch: int64 mailbox pairs of ``layer_plan(rows).scratch_pairs``,
            zero before the first launch.
        step: int32 ``[1]`` counter that changes once per draft forward.
        layer: The launch's layer slot (the draft step), below
            ``plan.LAYER_SLOTS``: launches of one ``step`` need distinct
            slots.
        trace_buf: None, or a trace buffer (``trace.py``).
        compile_only: Only compile the launch for these operands.
    """
    rows = embeds.shape[0]
    plan = layer_plan(rows)
    hidden_size = int(HIDDEN)
    for name, t in (("embeds", embeds), ("hidden", hidden), ("normed", normed)):
        _check(name, t, torch.bfloat16, (rows, hidden_size))
    # The launch reads embeds and hidden while other CTAs already write normed.
    if normed.data_ptr() in (embeds.data_ptr(), hidden.data_ptr()):
        raise ValueError("normed must not alias the inputs")
    _check("positions", positions, torch.int32, (rows,))
    _check("seq_len", seq_len, torch.int32, (rows,))
    if slots.dtype not in (torch.int32, torch.int64):
        raise ValueError("slots: int32 or int64")
    _check("slots", slots, slots.dtype, (rows,))
    if (
        page_table.dtype != torch.int32
        or page_table.dim() != 2
        or page_table.shape[0] < rows
        or page_table.stride(1) != 1
    ):
        raise ValueError(f"page_table: int32 [>= {rows}, pages], rows contiguous")
    heads, head = int(HEADS), int(HEAD)
    for name, c in (("k_cache", k_cache), ("v_cache", v_cache)):
        if (
            c.dtype != torch.bfloat16
            or tuple(c.shape[1:]) != (heads, head)
            or not c.is_contiguous()
        ):
            raise ValueError(f"{name}: contiguous BF16 [slots, {heads}, {head}]")
    if cos_sin.dtype != torch.float32 or cos_sin.shape[-1] != head:
        raise ValueError(f"cos_sin: FP32 [positions, {head}]")
    if scratch.dtype != torch.int64 or scratch.numel() < plan.scratch_pairs:
        raise ValueError(f"scratch: int64 of {plan.scratch_pairs} pairs")
    if not 0 <= layer < mailbox.LAYER_SLOTS.value:
        raise ValueError(f"layer slot {layer} out of range")
    w = weights

    def kernel_args(code: tuple[int, int]) -> tuple:
        return (
            scratch,
            step,
            layer,
            embeds,
            hidden,
            w.input_norm,
            w.hidden_norm,
            w.qkv,
            cos_sin,
            positions,
            slots,
            page_table,
            page_table.stride(0),
            seq_len,
            k_cache,
            v_cache,
            w.o,
            w.post_norm,
            w.gu,
            w.down,
            w.final_norm,
            normed,
            *code,
            trace_buf,
            plan.key,
        )

    kernel = gluon_eagle4_draft_layer_gfx950.warmup(
        *kernel_args((0, 0)),
        num_warps=8,
        grid=(plan.grid,),
        **extern_compile_options(),
    )
    if compile_only:
        return
    # The code range is a runtime argument: the same binary.
    gluon_eagle4_draft_layer_gfx950[(plan.grid,)](
        *kernel_args(code_warm.text_range(kernel)),
        num_warps=8,
        **extern_compile_options(),
    )


__all__ = [
    "BASES",
    "Eagle4DraftLayerWeights",
    "MAX_ROWS",
    "gluon_eagle4_draft_layer_gfx950",
    "launch_gluon_eagle4_draft_layer_gfx950",
    "layer_plan",
    "pack_eagle4_draft_layer",
    "regions",
    "stages",
]
