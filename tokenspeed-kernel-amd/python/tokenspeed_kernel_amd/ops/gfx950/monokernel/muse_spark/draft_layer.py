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

"""A whole DFlash2 draft layer of Muse Spark as one launch, one TP8 rank.

The attention half (``draft_attn``) and then the MLP half (``draft_mlp``) in
one plan. Two hand-offs change, since a CTA never reads another CTA's store of
the same launch:

* the attention half's last stage (afin) puts its partial into the owner
  ranks' ``red`` slots itself, in place of the MLP half's scat;
* the residual after the input norm stays in the registers of the CTA that
  formed it: isum1 and sum1 have the same tasks on the same CTAs, and sum1
  takes isum1's residual (``draft_mlp.sum_norm_regs``). Only the residual
  after the attention sublayer is stored, as the layer's output with the MLP
  half's partial.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch
from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.monokernel import (
    code_warm,
    mailbox,
    schedule,
    trace,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (
    draft_attn,
    draft_mlp,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.dims import (
    DRAFT_BLOCK,
    WORLD,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import Plan, Stage
from tokenspeed_kernel_amd.ops.gfx950.monokernel.weights import bf16_tiles

DRAFT_LAYERS = gl.constexpr(5)  # launches a draft forward runs
SPLITS = 16  # attention splits: 16 x 128 keys hold the 2041 cached
BLOCK_ROWS = int(DRAFT_BLOCK)  # one request's block
MAX_ROWS = 16  # two requests' blocks, one after the other

# The last layer's launch also runs the draft's final norm: fin puts the MLP
# partial into the owners' ``fred``, fsum and fnorm (on sum1's CTAs, with its
# residual in registers) form the normed rows in every rank's ``fhin``, and
# fout stores them. The regions come last, so the other regions sit where the
# other layers' plan has them: both plans share one scratch.
FINAL_PEERS = ("fred", "fss", "fhin")
PEERS = draft_attn.PEERS + draft_mlp.PEERS + FINAL_PEERS
FOUT_TASKS = gl.constexpr(8)  # per row: 1024 columns each
# CTAs 0 .. CODE_TASKS - 1 first read the launch's own code into the L2s
# (code_warm.py): 8 tasks on each XCD.
CODE_TASKS = 64


def regions(splits: int) -> tuple[tuple[str, int], ...]:
    world, owned, tasks = int(WORLD), int(draft_mlp.OWNED), int(draft_mlp.SUM_TASKS)
    final = (
        ("fred", world * owned // 2),
        ("fss", world * tasks),
        ("fhin", int(draft_mlp.HIDDEN) // 2),
    )
    return draft_attn.regions(splits) + draft_mlp.regions() + final


def stages(
    bases: dict[str, int], splits: int, rows: int, last: bool
) -> tuple[Stage, ...]:
    """The layer's stages of ``rows`` token rows (one block), placed by
    ``bases`` (both halves' names; ``isum`` and ``sum`` must match). The
    ``last`` layer's launch ends in the draft's final norm."""
    if bases["isum"] != bases["sum"]:
        raise ValueError(
            "isum and sum must share CTAs: sum takes isum's residual from registers"
        )
    attn = [
        (
            Stage(st.name, st.tasks, st.base, reads=st.reads, writes=("red",))
            if st.name == "afin"
            else st
        )
        for st in draft_attn.stages(
            {k: v for k, v in bases.items() if k in _ATTN_NAMES}, splits, rows
        )
    ]
    mlp_bases = {"scat": 0, **{k: v for k, v in bases.items() if k in _MLP_NAMES}}
    mlp = [st for st in draft_mlp.stages(mlp_bases, rows) if st.name != "scat"]
    if not last:
        return tuple(attn + mlp)
    fin = mlp[-1]
    tasks = int(draft_mlp.SUM_TASKS) * rows
    final = [
        Stage(fin.name, fin.tasks, fin.base, reads=fin.reads, writes=("fred",)),
        Stage("fsum", tasks, bases["sum"], reads=("fred",), writes=("fss",)),
        Stage("fnorm", tasks, bases["sum"], reads=("fss",), writes=("fhin",)),
        Stage("fout", int(FOUT_TASKS) * rows, bases["fout"], reads=("fhin",)),
    ]
    return tuple(attn + mlp[:-1] + final)


_ATTN_NAMES = (
    "iscat",
    "isum",
    "acoef",
    "aconv0",
    "qkv",
    "head",
    "attn",
    "merge",
    "oproj",
    "afin",
)
_MLP_NAMES = ("sum", "coef", "conv0", "gu", "act", "down", "fin")


# Placement: the attention half (draft_attn) then the MLP half (draft_mlp).
# isum and sum share CTAs, so sum takes isum's residual from registers.
BASES = {
    "iscat": 0,
    "isum": 128,
    "acoef": 0,
    "aconv0": 128,
    "qkv": 0,
    "head": 160,
    "attn": 240,
    "merge": 0,
    "oproj": 64,
    "afin": 0,
    "sum": 128,
    "coef": 0,
    "conv0": 128,
    "gu": 0,
    "act": 144,
    "down": 0,
    "fin": 0,
    "fout": 192,
}


@functools.lru_cache(maxsize=None)
def layer_plan(rows: int, last: bool) -> Plan:
    """The plan of a draft layer for ``rows`` token rows (one block); the
    ``last`` layer's ends in the draft's final norm."""
    return Plan(
        grid=256,
        stages=(Stage("code", CODE_TASKS, 0),) + stages(BASES, SPLITS, rows, last),
        regions=regions(SPLITS),
        rows=rows,
        chain=min(rows, BLOCK_ROWS),
        world=int(WORLD),
        peers=PEERS,
    )


@dataclass(frozen=True)
class DraftLayerWeights:
    """One TP8 rank's DFlash2 draft layer, packed for the launch.

    The BF16 projections are in ``weights.bf16_tiles`` order; the norm gains
    are FP32 of the rank's owned columns (``1024 rank ..``).
    """

    rank: int
    in_gain: torch.Tensor
    a_coef: torch.Tensor
    a_base: torch.Tensor
    qkv: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    o: torch.Tensor
    post_gain: torch.Tensor
    m_coef: torch.Tensor
    m_base: torch.Tensor
    gu: torch.Tensor
    down: torch.Tensor


def pack_draft_layer(tensors: dict[str, torch.Tensor], rank: int) -> DraftLayerWeights:
    """Pack one rank's draft layer weights.

    Args:
        tensors: BF16 by role: ``input_norm`` and ``post_norm`` ``[8192]``;
            ``attn_coef`` and ``mlp_coef`` ``[2048, 8192]`` (the replicated
            convolution coefficient projections); ``attn_base`` and
            ``mlp_base`` ``[2, 2, 8192]`` (``base_kernel``); ``qkv``
            ``[1280, 8192]`` (the rank's q, k, v rows); ``q_norm`` and
            ``k_norm`` ``[128]``; ``o_proj`` ``[8192, 1024]`` (the rank's
            columns); ``gate_up`` ``[7168, 8192]`` (the rank's gate rows,
            then its up rows); ``down`` ``[8192, 3584]`` (the rank's
            columns).
        rank: The rank in the TP8 group.
    """
    shapes = {
        "input_norm": (8192,),
        "post_norm": (8192,),
        "attn_coef": (2048, 8192),
        "mlp_coef": (2048, 8192),
        "attn_base": (2, 2, 8192),
        "mlp_base": (2, 2, 8192),
        "qkv": (1280, 8192),
        "q_norm": (128,),
        "k_norm": (128,),
        "o_proj": (8192, 1024),
        "gate_up": (7168, 8192),
        "down": (8192, 3584),
    }
    for name, shape in shapes.items():
        t = tensors[name]
        if t.dtype != torch.bfloat16 or tuple(t.shape) != shape:
            raise ValueError(f"{name}: BF16 {shape}, got {t.dtype} {tuple(t.shape)}")
    own = slice(rank * 1024, (rank + 1) * 1024)
    return DraftLayerWeights(
        rank=rank,
        in_gain=tensors["input_norm"][own].float().contiguous(),
        a_coef=bf16_tiles(tensors["attn_coef"]),
        a_base=tensors["attn_base"].contiguous(),
        qkv=bf16_tiles(tensors["qkv"]),
        q_norm=tensors["q_norm"].contiguous(),
        k_norm=tensors["k_norm"].contiguous(),
        o=bf16_tiles(tensors["o_proj"].contiguous()),
        post_gain=tensors["post_norm"][own].float().contiguous(),
        m_coef=bf16_tiles(tensors["mlp_coef"]),
        m_base=tensors["mlp_base"].contiguous(),
        gu=bf16_tiles(tensors["gate_up"]),
        down=bf16_tiles(tensors["down"].contiguous()),
    )


# The layer index changes every launch and the page tables' row stride with
# the table's width: a specialization on them would compile new binaries
# while serving. The code range is the compiled binary's own,
# known only after the compile.
@gluon.jit(do_not_specialize=("layer", "table_stride", "code_offset", "code_quads"))
def gluon_muse_spark_draft_layer_gfx950(
    scratch,
    step,
    layer,
    bases,
    x,
    residual,
    residual_out,
    in_gain,
    a_coef,
    a_base,
    qkv_w,
    q_norm,
    k_norm,
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
    m_coef,
    m_base,
    gu_w,
    down_w,
    out,
    final_gain,
    normed,
    code_offset,
    code_quads,
    trace_buf,
    PLAN: gl.constexpr,
    RANK: gl.constexpr,
    SPLITS_: gl.constexpr,
    LAST: gl.constexpr,
):
    """One step of draft layer ``layer`` on rank ``RANK`` (see the module
    docstring and :func:`launch_gluon_muse_spark_draft_layer_gfx950`).
    ``code_offset``, ``code_quads``: this binary's ``code_warm.text_range``."""
    table_ = trace.allocate(trace_buf)
    # The peer copies alternate from launch to launch, across forwards too:
    # a forward runs DRAFT_LAYERS launches.
    parity = (gl.load(step) * DRAFT_LAYERS + layer) & 1
    peers = mailbox.load_peers(bases, RANK, parity, WORLD)
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, layer), PLAN, table_, peers)
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
    H: gl.constexpr = draft_mlp.HIDDEN
    G: gl.constexpr = schedule.grid(PLAN)
    # The attention half.
    task, active = schedule.stage_task(PLAN, "iscat")
    if active:
        draft_mlp.scat_stage(mb, task, x, RANK, "ared", "iscat")
    r_lo = draft_mlp.residual_zeros()
    r_hi = draft_mlp.residual_zeros()
    sum_task, has_sum = schedule.stage_task(PLAN, "isum1")
    if has_sum:
        r_lo, r_hi = draft_mlp.sum_norm_stages(
            mb,
            sum_task,
            residual,
            residual_out,
            in_gain,
            RANK,
            "ared",
            "ass",
            "ahin",
            "isum1",
            "inorm",
            False,
        )
    task, active = schedule.stage_task(PLAN, "acoef")
    if active:
        draft_mlp.proj_task(
            mb,
            task,
            a_coef,
            "ahin",
            "acpart",
            draft_mlp.COEFS,
            H,
            draft_mlp.COEF_ROWS,
            draft_mlp.COEF_KS,
            "acoef",
        )
    task, active = schedule.stage_task(PLAN, "aconv0")
    if active:
        draft_mlp.conv0_stage(mb, task, a_base, "ahin", "acpart", "ahc", "aconv0")
    task, active = schedule.stage_task(PLAN, "qkv")
    if active:
        draft_mlp.proj_task(
            mb, task, qkv_w, "ahc", "qpart", draft_attn.QKV, H, 128, 512, "qkv"
        )
    task, active = schedule.stage_task(PLAN, "head")
    if active:
        draft_attn.head_stage(
            mb,
            task,
            q_norm,
            k_norm,
            cos_sin,
            positions,
            slots,
            k_cache,
            v_cache,
            "head",
        )
    task, active = schedule.stage_task(PLAN, "attn")
    if active:
        draft_attn.attn_stage(
            mb, task, table, table_stride, seq_len, k_cache, v_cache, SPLITS_, "attn"
        )
    task, active = schedule.stage_task(PLAN, "merge")
    if active:
        draft_attn.merge_stage(mb, task, SPLITS_, "merge")
    task, active = schedule.stage_task(PLAN, "oproj")
    if active:
        draft_mlp.proj_task(mb, task, o_w, "ao", "opart", H, 1024, 128, 512, "oproj")
    task, active = schedule.stage_task(PLAN, "afin")
    if active:
        draft_mlp.fin_to_owners(
            mb,
            task,
            a_base,
            "opart",
            draft_attn.OPROJ_KPARTS,
            "acpart",
            draft_attn.AFIN_COLS,
            RANK,
            "red",
            "afin",
        )
    # The MLP half: sum1 on isum1's CTAs, with its residual.
    if has_sum:
        r_lo, r_hi = draft_mlp.sum_norm_regs(
            mb,
            sum_task,
            r_lo,
            r_hi,
            residual_out,
            post_gain,
            RANK,
            "red",
            "ss",
            "hin",
            "sum1",
            "norm",
            True,
        )
    task, active = schedule.stage_task(PLAN, "coef")
    if active:
        draft_mlp.proj_task(
            mb,
            task,
            m_coef,
            "hin",
            "cpart",
            draft_mlp.COEFS,
            H,
            draft_mlp.COEF_ROWS,
            draft_mlp.COEF_KS,
            "coef",
        )
    # The CTAs that host no conv0 task start their first gate/up task right
    # after coef (see draft_mlp).
    gu_task, rounds = schedule.stage_rounds(PLAN, "gu")
    _, has_conv0 = schedule.stage_task(PLAN, "conv0")
    free = not has_conv0
    if free:
        draft_mlp.proj_task(
            mb,
            gu_task,
            gu_w,
            "hc",
            "gpart",
            draft_mlp.GU,
            H,
            draft_mlp.GU_ROWS,
            draft_mlp.GU_KS,
            "gu",
        )
    task, active = schedule.stage_task(PLAN, "conv0")
    if active:
        draft_mlp.conv0_stage(mb, task, m_base, "hin", "cpart", "hc", "conv0")
    for i in range(gl.where(free, 1, 0), rounds):
        draft_mlp.proj_task(
            mb,
            gu_task + i * G,
            gu_w,
            "hc",
            "gpart",
            draft_mlp.GU,
            H,
            draft_mlp.GU_ROWS,
            draft_mlp.GU_KS,
            "gu",
        )
    d_task, d_rounds = schedule.stage_rounds(PLAN, "down")
    a_task, has_act = schedule.stage_task(PLAN, "act")
    I: gl.constexpr = draft_mlp.INTER
    if has_act:
        tiles = draft_mlp.proj_issue(
            d_task, down_w, H, I, draft_mlp.DOWN_ROWS, draft_mlp.DOWN_KS
        )
        draft_mlp.act_stage(mb, a_task, "act")
        xs = draft_mlp.proj_poll(mb, d_task, "act", I, draft_mlp.DOWN_KS, "down")
        draft_mlp.proj_compute(
            mb,
            d_task,
            tiles,
            xs,
            "dpart",
            H,
            I,
            draft_mlp.DOWN_ROWS,
            draft_mlp.DOWN_KS,
            "down",
        )
    for i in range(gl.where(has_act, 1, 0), d_rounds):
        draft_mlp.proj_task(
            mb,
            d_task + i * G,
            down_w,
            "act",
            "dpart",
            H,
            I,
            draft_mlp.DOWN_ROWS,
            draft_mlp.DOWN_KS,
            "down",
        )
    task, active = schedule.stage_task(PLAN, "fin")
    if LAST:
        # The draft's final norm: the MLP partial goes to its owners, and
        # sum1's CTAs add their residual and norm it (as the next layer's
        # input norm would).
        if active:
            draft_mlp.fin_to_owners(
                mb,
                task,
                m_base,
                "dpart",
                draft_mlp.DOWN_KPARTS,
                "cpart",
                draft_mlp.FIN_COLS,
                RANK,
                "fred",
                "fin",
            )
        if has_sum:
            draft_mlp.sum_norm_regs(
                mb,
                sum_task,
                r_lo,
                r_hi,
                residual_out,
                final_gain,
                RANK,
                "fred",
                "fss",
                "fhin",
                "fsum",
                "fnorm",
                False,
            )
        task, active = schedule.stage_task(PLAN, "fout")
        if active:
            fout_stage(mb, task, normed)
    elif active:
        draft_mlp.fin_stage(
            mb,
            task,
            m_base,
            out,
            "dpart",
            draft_mlp.DOWN_KPARTS,
            "cpart",
            draft_mlp.FIN_COLS,
            "fin",
        )
    trace.flush(table_, trace_buf)


@gluon.jit
def fout_stage(mb, task, normed):
    """Fout task ``task``: columns ``1024 (task % 8) ..`` of row ``task // 8``
    of the final norm's output, from ``fhin`` into ``normed`` (BF16 ``[rows,
    8192]``, the whole rows on every rank)."""
    mb.stamp("fout", trace.START)
    row, t, _ = schedule.row_task(mb.plan, task, FOUT_TASKS)
    H: gl.constexpr = draft_mlp.HIDDEN
    W: gl.constexpr = H // FOUT_TASKS // 2  # pairs of a task
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    j = t * W + gl.arange(0, W, lanes)
    (got,) = mb.row(row).poll_many(("fhin",), (j,))
    mb.stamp("fout", trace.READY)
    words = normed.to(gl.pointer_type(gl.int32), bitcast=True)
    gl.store(words + row * (H // 2) + j, got)
    mb.stamp("fout", trace.END)


def launch_gluon_muse_spark_draft_layer_gfx950(
    weights: DraftLayerWeights,
    *,
    x: torch.Tensor,
    residual: torch.Tensor,
    residual_out: torch.Tensor,
    out: torch.Tensor | None,
    final_gain: torch.Tensor | None,
    normed: torch.Tensor | None,
    seq_len: torch.Tensor,
    positions: torch.Tensor,
    slots: torch.Tensor,
    page_table: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cos_sin: torch.Tensor,
    scratch: torch.Tensor,
    step: torch.Tensor,
    peer_bases: torch.Tensor,
    layer: int,
    trace_buf,
    compile_only: bool,
) -> None:
    """One step of a packed draft layer for the blocks of one or two
    requests (8 rows each, request by request), on every rank of the TP8
    group at once (the launches exchange values).

    Args:
        weights: From :func:`pack_draft_layer`.
        x: BF16 ``[rows, 8192]``: this rank's partial of the layer input
            (the previous layer's MLP partial, or the embedding's).
        residual: BF16 ``[rows, 8192]``: the residual before the layer (the
            same on every rank).
        residual_out: BF16 ``[rows, 8192]``: this rank stores its owned
            columns (``1024 rank ..``) of the residual after the attention
            sublayer.
        out: BF16 ``[rows, 8192]``: this rank's partial of the layer
            output; None for the draft's last layer, whose launch runs the
            final norm instead.
        final_gain: None, or (the last layer) FP32 ``[1024]``: the final
            norm's gain of the rank's owned columns.
        normed: None, or (the last layer) BF16 ``[rows, 8192]``: the final
            norm's output, the whole rows on every rank.
        seq_len: int32 ``[requests]``: each request's length after its
            block.
        positions: int32 ``[rows]``: the block rows' positions.
        slots: int32 or int64 ``[rows]``: the block rows' cache slots.
        page_table: int32 ``[>= requests, pages]``: row ``q`` is request
            ``q``'s page table (64 tokens a page).
        k_cache, v_cache: BF16 ``[slots, 1, 128]``: this layer's cache.
        cos_sin: FP32 ``[positions, 128]`` rotary table, cos then sin.
        scratch: int64 mailbox pairs of ``layer_plan(rows, last)
            .scratch_pairs`` (the last layer's plan is the larger) on the Iris
            symmetric heap.
        step: int32 ``[1]`` counter that changes once per draft forward.
        peer_bases: int64 ``[8]`` heap bases of the group's ranks.
        layer: The draft layer index, below ``DRAFT_LAYERS``.
        trace_buf: None, or a trace buffer (``trace.py``).
        compile_only: Only compile the launch for these operands: a rank's
            launch polls its peers for about a minute before it traps, so the
            ranks compile before any of them launches.
    """
    rows = x.shape[0]
    if not 1 <= rows <= MAX_ROWS or rows % BLOCK_ROWS:
        raise ValueError(f"draft blocks of {BLOCK_ROWS} rows, got {rows}")
    requests = rows // BLOCK_ROWS
    if seq_len.dtype != torch.int32 or tuple(seq_len.shape) != (requests,):
        raise ValueError(f"seq_len: int32 [{requests}]")
    if (
        page_table.dtype != torch.int32
        or page_table.dim() != 2
        or page_table.shape[0] < requests
        or page_table.stride(1) != 1
    ):
        raise ValueError(f"page_table: int32 [>= {requests}, pages], rows contiguous")
    last = normed is not None
    if (out is None) != last or (final_gain is not None) != last:
        raise ValueError("either out, or (the last layer) final_gain and normed")
    if last and (
        final_gain.dtype != torch.float32 or tuple(final_gain.shape) != (1024,)
    ):
        raise ValueError("final_gain: FP32 [1024]")
    for name, t in (
        ("x", x),
        ("residual", residual),
        ("residual_out", residual_out),
        ("out" if not last else "normed", normed if last else out),
    ):
        if (
            t.dtype != torch.bfloat16
            or tuple(t.shape) != (rows, 8192)
            or not t.is_contiguous()
        ):
            raise ValueError(f"{name}: contiguous BF16 [{rows}, 8192]")
    # The launch reads x and residual while other CTAs already write the outputs.
    written = normed if last else out
    if {x.data_ptr(), residual.data_ptr()} & {
        residual_out.data_ptr(),
        written.data_ptr(),
    }:
        raise ValueError("the outputs must not alias the inputs")
    if k_cache.dtype != torch.bfloat16 or v_cache.dtype != torch.bfloat16:
        raise ValueError("k_cache, v_cache: BF16 (the kernel reads BF16 keys)")
    if not 0 <= layer < int(DRAFT_LAYERS):
        raise ValueError(f"layer {layer} out of range({int(DRAFT_LAYERS)})")
    plan = layer_plan(rows, last)
    w = weights

    def kernel_args(code: tuple[int, int]) -> tuple:
        return (
            scratch,
            step,
            layer,
            peer_bases,
            x,
            residual,
            residual_out,
            w.in_gain,
            w.a_coef,
            w.a_base,
            w.qkv,
            w.q_norm,
            w.k_norm,
            cos_sin,
            positions,
            slots,
            page_table,
            page_table.stride(0),
            seq_len,
            k_cache,
            v_cache,
            w.o,
            w.post_gain,
            w.m_coef,
            w.m_base,
            w.gu,
            w.down,
            # The pointers a launch does not use point at a written tensor.
            written,
            final_gain if last else w.post_gain,
            written,
            *code,
            trace_buf,
            plan.key,
            w.rank,
            SPLITS,
            last,
        )

    kernel = gluon_muse_spark_draft_layer_gfx950.warmup(
        *kernel_args((0, 0)), num_warps=8, grid=(plan.grid,)
    )
    if compile_only:
        return
    # The code range is a runtime argument: the same binary.
    gluon_muse_spark_draft_layer_gfx950[(plan.grid,)](
        *kernel_args(code_warm.text_range(kernel)), num_warps=8
    )


__all__ = [
    "BLOCK_ROWS",
    "BASES",
    "DraftLayerWeights",
    "MAX_ROWS",
    "SPLITS",
    "gluon_muse_spark_draft_layer_gfx950",
    "launch_gluon_muse_spark_draft_layer_gfx950",
    "layer_plan",
    "pack_draft_layer",
    "regions",
    "stages",
]
