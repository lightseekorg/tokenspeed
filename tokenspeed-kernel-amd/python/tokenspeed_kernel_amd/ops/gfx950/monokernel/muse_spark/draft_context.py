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

"""The DFlash2 draft's context write of Muse Spark as one launch, one TP8
rank.

After a verify, the regular drafter projects the target's tapped hidden
states (5 taps of 8192 a row, ``models/dflash.py`` ``project_target_hidden``)
through ``fc`` and ``hidden_norm``, then writes every draft layer's K and V
of those rows into the draft cache (``drafter/_dflash_fused_kv.py``: one
stacked K|V GEMM, then RMSNorm and RoPE on K). This launch does it for the
verify rows of one or two requests (8 a request; the rows are independent)
on every rank of the TP8 group:

* fc, 256 tasks: 128 of the rank's 1024 fc output columns (``fc`` rows
  ``1024 rank ..``) x a 1280-wide K part of the taps, FP32 partials into
  ``fpart``. ``fc`` is the draft's largest weight (671 MB); each rank reads
  its eighth.
* fsum, 8 tasks a row (128 owned columns each): the K parts' sum rounded to
  BF16 (the projection's output) and its sum of squares, put to every rank.
* fnorm, on fsum's CTAs: ``hidden_norm`` of the owned columns, put to every
  rank's ``chin``: the whole normed rows on every rank.
* kv, 160 tasks: the 5 layers' K and V rows of the rank's KV head (1280
  rows) x 512-wide K parts of the normed rows, into ``kvpart``. The tasks
  issue their weight loads before they poll ``chin``.
* kvout, 10 tasks a row (layer, then K or V): the K parts' sum rounded
  to BF16, then for K the per-head RMSNorm and NeoX RoPE in FP32, rounded
  once, stored at the row's slot of the layer's cache.
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
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (
    draft_attn,
    draft_layer,
    draft_mlp,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.dims import (
    DRAFT_BLOCK,
    DRAFT_HEAD_DIM,
    HIDDEN,
    OWNED,
    WORLD,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import Plan, Stage, placement
from tokenspeed_kernel_amd.ops.gfx950.monokernel.weights import bf16_tiles

TAPS = 5
TAPS_K = gl.constexpr(TAPS * HIDDEN)  # fc's input width
FC_ROWS = gl.constexpr(128)  # owned columns of an fc task: 16 a wave
FC_KS = gl.constexpr(1280)  # K values of an fc task
FC_KPARTS = gl.constexpr(TAPS_K // FC_KS)  # 32
FC_ROUND = gl.constexpr(256)  # K values of one weight load round
FC_TASKS = gl.constexpr(OWNED // FC_ROWS * FC_KPARTS)  # 8 groups x 32 K parts
SUM_COLS = gl.constexpr(128)
SUM_TASKS = gl.constexpr(OWNED // SUM_COLS)  # per row: 128 owned columns each
LAYERS = draft_layer.DRAFT_LAYERS
HEAD = DRAFT_HEAD_DIM
HALF = draft_attn.HALF
KV = gl.constexpr(LAYERS * 2 * HEAD)  # K then V rows of the 5 layers
KV_ROWS = gl.constexpr(128)  # K, V rows of a kv task
KV_KS = gl.constexpr(512)
KV_KPARTS = gl.constexpr(HIDDEN // KV_KS)  # 16
KV_TASKS = gl.constexpr(KV // KV_ROWS * KV_KPARTS)  # 10 groups x 16 K parts
OUT_TASKS = gl.constexpr(2 * LAYERS)  # per row: layer, then K or V
EPS = draft_mlp.EPS
ROWS = int(DRAFT_BLOCK)  # one request's verify rows
MAX_REQUESTS = 2  # requests a launch: their verify rows one after another

PEERS = ("css", "chin")


def regions() -> tuple[tuple[str, int], ...]:
    """The context write's regions as (name, int64 pairs): fpart, css, chin,
    kvpart."""
    return (
        ("fpart", int(FC_KPARTS) * int(OWNED)),  # FP32 [K part][owned column]
        ("css", int(WORLD) * int(SUM_TASKS)),  # FP32 sums of squares [rank][task]
        ("chin", int(HIDDEN) // 2),  # BF16 pairs of the normed rows
        ("kvpart", int(KV_KPARTS) * int(KV)),  # FP32 [K part][K, V row]
    )


GRID = 256
# kv's tasks end on the last CTA.
BASES = {"fc": 0, "fsum": 0, "kv": GRID - int(KV_TASKS), "kvout": 0}


def stages(bases: dict[str, int], rows: int) -> tuple[Stage, ...]:
    """The stages of ``rows`` verify rows, placed by ``bases`` (stage name to
    first CTA)."""
    fc, fsum, kv, kvout = placement(bases, ("fc", "fsum", "kv", "kvout"))
    tasks = int(SUM_TASKS) * rows
    return (
        Stage("fc", int(FC_TASKS), fc, writes=("fpart",)),
        Stage("fsum", tasks, fsum, reads=("fpart",), writes=("css",)),
        Stage("fnorm", tasks, fsum, reads=("css",), writes=("chin",)),
        Stage("kv", int(KV_TASKS), kv, reads=("chin",), writes=("kvpart",)),
        Stage("kvout", int(OUT_TASKS) * rows, kvout, reads=("kvpart",)),
    )


# CTAs 0 .. CODE_TASKS - 1 first read the launch's own code into the L2s
# (code_warm.py): 8 tasks on each XCD.
CODE_TASKS = 64


@functools.lru_cache(maxsize=None)
def context_plan(rows: int) -> Plan:
    """The plan of the context write for ``rows`` verify rows."""
    return Plan(
        grid=GRID,
        stages=(Stage("code", CODE_TASKS, 0),) + stages(BASES, rows),
        regions=regions(),
        rows=rows,
        chain=rows,
        world=int(WORLD),
        peers=PEERS,
    )


@dataclass(frozen=True)
class DraftContextWeights:
    """One TP8 rank's context write, packed for the launch.

    Attributes:
        rank: The rank in the TP8 group.
        fc: BF16 ``[1024, 40960]``: ``fc`` rows ``1024 rank ..``, tiled.
        gain: FP32 ``[1024]``: ``hidden_norm``'s gain of the owned columns.
        kv: BF16 ``[1280, 8192]``: each layer's K then V rows of the rank's KV
            head (``qkv_proj`` rows 1024 .. 1279), layer by layer, tiled.
        k_norm: BF16 ``[5, 128]``: each layer's per-head K RMSNorm weight.
    """

    rank: int
    fc: torch.Tensor
    gain: torch.Tensor
    kv: torch.Tensor
    k_norm: torch.Tensor


def pack_draft_context(
    tensors: dict[str, torch.Tensor], rank: int
) -> DraftContextWeights:
    """Pack one rank's context write.

    Args:
        tensors: BF16 by role: ``fc`` ``[1024, 40960]``, the rank's rows of
            the column-parallel ``fc`` (output columns ``1024 rank ..``);
            ``hidden_norm`` ``[8192]``; ``qkv_0`` .. ``qkv_4`` ``[1280, 8192]``
            (each layer's q, k, v rows of the rank) and ``k_norm_0`` ..
            ``k_norm_4`` ``[128]``.
        rank: The rank in the TP8 group.
    """
    qkv_rows = int(draft_attn.QKV)
    shapes = {
        "fc": (int(OWNED), int(TAPS_K)),
        "hidden_norm": (int(HIDDEN),),
        **{f"qkv_{i}": (qkv_rows, int(HIDDEN)) for i in range(int(LAYERS))},
        **{f"k_norm_{i}": (int(HEAD),) for i in range(int(LAYERS))},
    }
    for name, shape in shapes.items():
        t = tensors[name]
        if t.dtype != torch.bfloat16 or tuple(t.shape) != shape:
            raise ValueError(f"{name}: BF16 {shape}, got {t.dtype} {tuple(t.shape)}")
    own = slice(rank * int(OWNED), (rank + 1) * int(OWNED))
    # Each layer's k and v rows follow its q rows.
    q_rows = int(draft_attn.Q_ROWS)
    kv = torch.cat([tensors[f"qkv_{i}"][q_rows:] for i in range(int(LAYERS))])
    return DraftContextWeights(
        rank=rank,
        fc=bf16_tiles(tensors["fc"].contiguous()),
        gain=tensors["hidden_norm"][own].float().contiguous(),
        kv=bf16_tiles(kv.contiguous()),
        k_norm=torch.stack([tensors[f"k_norm_{i}"] for i in range(int(LAYERS))]),
    )


@gluon.jit
def _x_round(x, words, row_stride, k0, KS: gl.constexpr, R: gl.constexpr):
    """Store values ``k0 .. k0 + KS - 1`` of the BF16 rows ``x`` (row ``r``
    at ``x + r * row_stride``) into ``words`` and return them as
    :func:`bf16.gemv`'s LDS rows."""
    RP: gl.constexpr = triton.next_power_of_2(R)
    KP: gl.constexpr = KS // 32
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    p = gl.arange(0, RP * KP * 16, lanes)
    row = gl.minimum(p // (KP * 16), R - 1)
    at = p % (KP * 16)
    words_ptr = x.to(gl.pointer_type(gl.int32), bitcast=True)
    # The taps come from every rank's layer launches (write-through stores
    # over the GPU links), and a peer's store does not update this GPU's L2:
    # load past the caches (sc0 sc1), or an earlier step's line could serve.
    got = gl.load(
        words_ptr + row * (row_stride // 2) + k0 // 2 + at, cache_modifier=".cv"
    )
    # The previous round's MFMAs have read the buffer.
    gl.barrier()
    words.store(gl.reshape(got, [RP, KP, 16]))
    gl.barrier()
    return words.reinterpret(
        gl.bfloat16, [RP, KP, 32], mx._padded_rows([RP, KP, 32], KP * 32, 16)
    )


@gluon.jit
def fc_stage(mb, task, x, w):
    """Fc task ``task``: owned columns ``128 (task // 32) ..``, K part
    ``task % 32``, from the taps ``x [rows, 40960]``, into ``fpart``."""
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    NW: gl.constexpr = gl.num_warps()
    gl.static_assert(16 * NW == FC_ROWS)
    KI: gl.constexpr = FC_ROUND // 32
    ROUNDS: gl.constexpr = FC_KS // FC_ROUND
    group = task // FC_KPARTS
    kpart = task % FC_KPARTS
    mb.stamp("fc", trace.START)
    words = gl.allocate_shared_memory(
        gl.int32, [RP, KI, 16], mx._padded_rows([RP, KI, 16], KI * 16, 8)
    )
    kit0 = kpart * (FC_KS // 32)
    tiles = bf16.load_weight_tiles(w, group * NW, TAPS_K, kit0, KI, 1, NW, NW, 0)
    acc = bf16.gemv_template(1, NW, RP)
    for i in gl.static_range(ROUNDS):
        if i + 1 < ROUNDS:
            nxt = bf16.load_weight_tiles(
                w, group * NW, TAPS_K, kit0 + (i + 1) * KI, KI, 1, NW, NW, 0
            )
        xs = _x_round(x, words, TAPS_K, kpart * FC_KS + i * FC_ROUND, FC_ROUND, R)
        acc += bf16.gemv(xs, tiles, FIRST=0, ROWS=R)
        if i + 1 < ROUNDS:
            tiles = nxt
    mb.stamp("fc", trace.COMPUTED)
    bl: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [NW, 1], [1, 0])
    out = gl.convert_layout(acc, bl)
    r = gl.arange(0, RP, gl.SliceLayout(1, bl))[:, None]
    n = gl.arange(0, 16 * NW, gl.SliceLayout(0, bl))[None, :]
    at = mb.row_offset(r) + kpart * OWNED + group * (16 * NW) + n
    mb.put("fpart", at, out.to(gl.int32, bitcast=True), mask=(r < R) & (n >= 0))
    mb.stamp("fc", trace.END)


@gluon.jit
def norm_stages(mb, task, gain, RANK: gl.constexpr):
    """Fsum and fnorm task ``task``: owned columns ``128 (task % 8) ..`` of
    row ``task // 8``: fc's output, then ``hidden_norm`` into every rank's
    ``chin``."""
    mb.stamp("fsum", trace.START)
    row, t, _ = schedule.row_task(mb.plan, task, SUM_TASKS)
    mb = mb.row(row)
    nw: gl.constexpr = gl.num_warps()
    kl: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [nw, 1], [1, 0])
    k = gl.arange(0, FC_KPARTS, gl.SliceLayout(1, kl))[:, None]
    c = gl.arange(0, SUM_COLS, gl.SliceLayout(0, kl))
    g = gl.load(gain + t * SUM_COLS + c)
    (got,) = mb.poll_many(("fpart",), (k * OWNED + t * SUM_COLS + c[None, :],))
    mb.stamp("fsum", trace.READY)
    # The projection's output: the K parts summed, rounded once to BF16.
    v = gl.sum(got.to(gl.float32, bitcast=True), axis=0).to(gl.bfloat16)
    v = v.to(gl.float32)
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [nw], [0])
    one = gl.arange(0, 1, lanes)
    word = (gl.zeros([1], gl.float32, lanes) + gl.sum(v * v, axis=0)).to(
        gl.int32, bitcast=True
    )
    for p in gl.static_range(WORLD):
        mb.put_peer("css", p, RANK * SUM_TASKS + t + one, word)
    mb.stamp("fsum", trace.END)
    mb.stamp("fnorm", trace.START)
    (ss,) = mb.poll_many(("css",), (gl.arange(0, WORLD * SUM_TASKS, lanes),))
    mb.stamp("fnorm", trace.READY)
    inv = gl.rsqrt(gl.sum(ss.to(gl.float32, bitcast=True), axis=0) / HIDDEN + EPS)
    h = (v * inv * g).to(gl.bfloat16)
    pl: gl.constexpr = gl.BlockedLayout([2], [64], [nw], [0])
    hw = mx.pack_bf16_pairs(gl.reshape(gl.convert_layout(h, pl), [SUM_COLS // 2, 2]))
    j = gl.arange(0, SUM_COLS // 2, hw.type.layout)
    at = RANK * (OWNED // 2) + t * (SUM_COLS // 2) + j
    for p in gl.static_range(WORLD):
        mb.put_peer("chin", p, at, hw)
    mb.stamp("fnorm", trace.END)


@gluon.jit
def kvout_stage(
    mb,
    task,
    k_norm,
    cos_sin,
    positions,
    slots,
    k0,
    k1,
    k2,
    k3,
    k4,
    v0,
    v1,
    v2,
    v3,
    v4,
):
    """Kvout task ``task``: layer ``(task % 10) // 2``'s K (even) or V (odd)
    of row ``task // 10``, into the layer's cache at the row's slot."""
    mb.stamp("kvout", trace.START)
    row, h, _ = schedule.row_task(mb.plan, task, OUT_TASKS)
    mb = mb.row(row)
    layer = h // 2
    is_k = h % 2 == 0
    nw: gl.constexpr = gl.num_warps()
    kl: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [nw, 1], [1, 0])
    k = gl.arange(0, KV_KPARTS, gl.SliceLayout(1, kl))[:, None]
    i = gl.arange(0, HALF, gl.SliceLayout(0, kl))
    w1 = gl.load(k_norm + layer * HEAD + i).to(gl.float32)
    w2 = gl.load(k_norm + layer * HEAD + HALF + i).to(gl.float32)
    position = gl.load(positions + row).to(gl.int64)
    cos = gl.load(cos_sin + position * HEAD + i)
    sin = gl.load(cos_sin + position * HEAD + HALF + i)
    slot = gl.load(slots + row).to(gl.int64)
    at = k * KV + h * HEAD + i[None, :]
    lo_w, hi_w = mb.poll_many(("kvpart", "kvpart"), (at, at + HALF))
    mb.stamp("kvout", trace.READY)
    vals = draft_attn.head_values(lo_w, hi_w, w1, w2, cos, sin, is_k)
    cols = gl.arange(0, HEAD, vals.type.layout)
    at = slot * HEAD + cols
    # The layer's cache: one store, picked by the task.
    if h == 0:
        gl.store(k0 + at, vals)
    elif h == 1:
        gl.store(v0 + at, vals)
    elif h == 2:
        gl.store(k1 + at, vals)
    elif h == 3:
        gl.store(v1 + at, vals)
    elif h == 4:
        gl.store(k2 + at, vals)
    elif h == 5:
        gl.store(v2 + at, vals)
    elif h == 6:
        gl.store(k3 + at, vals)
    elif h == 7:
        gl.store(v3 + at, vals)
    elif h == 8:
        gl.store(k4 + at, vals)
    else:
        gl.store(v4 + at, vals)
    mb.stamp("kvout", trace.END)


def _launch_metadata(grid, kernel, args):
    """Report the launch's rows and the weight bytes it reads to Proton: the
    fc and K / V projection shards (the norm gains are a few KiB)."""
    rows = args["PLAN"].plan.rows
    weight = args["fc_w"].nbytes + args["kv_w"].nbytes
    return {"name": f"{kernel.name} (rows={rows})", "bytes": weight}


# The code range is the compiled binary's own, known only after the compile.
@gluon.jit(
    launch_metadata=_launch_metadata, do_not_specialize=("code_offset", "code_quads")
)
def gluon_muse_spark_draft_context_gfx950(
    scratch,
    step,
    bases,
    x,
    fc_w,
    gain,
    kv_w,
    k_norm,
    cos_sin,
    positions,
    slots,
    k0,
    k1,
    k2,
    k3,
    k4,
    v0,
    v1,
    v2,
    v3,
    v4,
    code_offset,
    code_quads,
    trace_buf,
    PLAN: gl.constexpr,
    RANK: gl.constexpr,
):
    """The context write of the verify rows on rank ``RANK`` (see the module
    docstring and :func:`launch_gluon_muse_spark_draft_context_gfx950`)."""
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
    task, active = schedule.stage_task(PLAN, "fc")
    if active:
        fc_stage(mb, task, x, fc_w)
    task, active = schedule.stage_task(PLAN, "fsum")
    if active:
        norm_stages(mb, task, gain, RANK)
    task, active = schedule.stage_task(PLAN, "kv")
    if active:
        draft_mlp.proj_task(
            mb, task, kv_w, "chin", "kvpart", KV, HIDDEN, KV_ROWS, KV_KS, "kv"
        )
    task, active = schedule.stage_task(PLAN, "kvout")
    if active:
        kvout_stage(
            mb,
            task,
            k_norm,
            cos_sin,
            positions,
            slots,
            k0,
            k1,
            k2,
            k3,
            k4,
            v0,
            v1,
            v2,
            v3,
            v4,
        )
    trace.flush(table_, trace_buf)


def launch_gluon_muse_spark_draft_context_gfx950(
    weights: DraftContextWeights,
    *,
    taps: torch.Tensor,
    positions: torch.Tensor,
    slots: torch.Tensor,
    k_caches: tuple[torch.Tensor, ...],
    v_caches: tuple[torch.Tensor, ...],
    cos_sin: torch.Tensor,
    scratch: torch.Tensor,
    step: torch.Tensor,
    peer_bases: torch.Tensor,
    trace_buf,
    compile_only: bool,
) -> None:
    """The context write of one or two requests' verify rows, on every rank of
    the TP8 group at once (the launches exchange the norm and its output).

    Args:
        weights: From :func:`pack_draft_context`.
        taps: BF16 ``[rows, 40960]`` (8 or 16 rows) contiguous: the
            target's tapped hidden states of the verify rows (the same on
            every rank).
        positions: int32 ``[rows]``: the rows' positions.
        slots: int32 or int64 ``[rows]``: the rows' draft cache slots.
        k_caches, v_caches: 5 BF16 ``[slots, 1, 128]`` each: the draft
            layers' caches of the rank's KV head.
        cos_sin: FP32 ``[positions, 128]`` RoPE table, cos then sin.
        scratch: int64 mailbox pairs of ``context_plan(rows).scratch_pairs`` on
            the Iris symmetric heap.
        step: int32 ``[1]`` counter that changes from launch to launch.
        peer_bases: int64 ``[8]`` heap bases of the group's ranks.
        trace_buf: None, or a trace buffer (``trace.py``).
        compile_only: Only compile the launch for these operands.
    """
    rows = taps.shape[0]
    if rows % ROWS or not 1 <= rows // ROWS <= MAX_REQUESTS:
        raise ValueError(
            f"the context write serves {ROWS} verify rows a request, at most "
            f"{MAX_REQUESTS} requests, got {rows} rows"
        )
    if (
        taps.dtype != torch.bfloat16
        or tuple(taps.shape) != (rows, int(TAPS_K))
        or not taps.is_contiguous()
    ):
        raise ValueError(f"taps: contiguous BF16 [{rows}, {int(TAPS_K)}]")
    if len(k_caches) != int(LAYERS) or len(v_caches) != int(LAYERS):
        raise ValueError(f"{int(LAYERS)} K and V caches")
    for c in (*k_caches, *v_caches):
        if (
            c.dtype != torch.bfloat16
            or c.shape[-1] != int(HEAD)
            or not c.is_contiguous()
        ):
            raise ValueError("caches: contiguous BF16 [slots, 1, 128]")
    plan = context_plan(rows)
    w = weights

    def kernel_args(code: tuple[int, int]) -> tuple:
        return (
            scratch,
            step,
            peer_bases,
            taps,
            w.fc,
            w.gain,
            w.kv,
            w.k_norm,
            cos_sin,
            positions,
            slots,
            *k_caches,
            *v_caches,
            *code,
            trace_buf,
            plan.key,
            w.rank,
        )

    kernel = gluon_muse_spark_draft_context_gfx950.warmup(
        *kernel_args((0, 0)), num_warps=8, grid=(plan.grid,)
    )
    if compile_only:
        return
    # The code range is a runtime argument: the same binary.
    gluon_muse_spark_draft_context_gfx950[(plan.grid,)](
        *kernel_args(code_warm.text_range(kernel)), num_warps=8
    )


__all__ = [
    "MAX_REQUESTS",
    "BASES",
    "DraftContextWeights",
    "ROWS",
    "context_plan",
    "gluon_muse_spark_draft_context_gfx950",
    "launch_gluon_muse_spark_draft_context_gfx950",
    "pack_draft_context",
    "regions",
    "stages",
]
