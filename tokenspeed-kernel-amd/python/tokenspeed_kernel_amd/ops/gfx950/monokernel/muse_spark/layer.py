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

"""One Muse Spark decode layer (layers 1 .. 60: MXFP4 experts) of one TP8
rank as one persistent launch.

Rank ``r`` owns hidden columns ``1024 r ..``: it keeps that slice of the FP32
residual, and every rank keeps the whole BF16 attention input. The stages,
in plan order (see each module):

* attention: QKVG, gate, head, attn, merge (``qkvg``, ``attention``);
* output projection and attention tail (``attn_tail``);
* routing, pooler, experts, combine, post-expert projection and boundary
  (``moe_route``, ``moe_pool``, ``moe_experts``, ``moe_tail``).

The peer regions' copy of a launch is ``layer & 1``. The monokernel launches
layers 1 to 60 (layers 0 and 61 keep the regular path), an even count, so the
copy alternates from one launch to the next, across steps too: layer 60
uses copy 0 and the next step's layer 1 copy 1.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._extern import extern_compile_options
from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.monokernel import (
    code_warm,
    mailbox,
    schedule,
    stuck,
    trace,
    warm,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (
    attention,
    attn_tail,
    moe_experts,
    moe_pool,
    moe_route,
    moe_tail,
    qkvg,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.dims import (
    EXPERTS,
    GATE_ROW,
    HIDDEN,
    OWNED,
    QKVG,
    WORLD,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import (
    DEFAULT_POLL_CHECK,
    DEFAULT_POLL_TIMEOUT_SHIFT,
    Plan,
    Stage,
)

SLIDING_SPLITS = 16
# A full layer's attention splits: one request's tasks (2 KV heads x splits)
# fill the grid, so at long context each walks half the keys that 64 splits
# leave it. Several requests' tasks
# run in rounds: the splits, and so each request's sums, do not depend on how
# many requests share the launch.
FULL_SPLITS = 128
WINDOW = attention.WINDOW

# The first placement: weight-heavy stages on the CTAs that finish their
# earlier tasks first. The trace decides the next one.
BASES = {
    "qkvg": 0,
    "gate": 0,
    "head": 158,
    "attn": 162,
    "merge": 240,
    "oproj": 0,
    "tail": 128,
    "router": 136,
    "topk": 152,
    "pool": 192,
    # The slot table, once, on a topk CTA (after its topk task).
    "slt": 152,
    # The expert input, once (up to 16 tasks): on ex1's last lane, which runs
    # the fewest slots, past pool's CTAs (past one row, pool's own).
    "xq": 240,
    "ex1": 0,
    # The K sums on the CTAs of ex1's last lanes, which run the fewest slots
    # at one row (8 slots: one each). Past one row they spread over every CTA
    # (CHAINED_BASES, moe_experts.EXR_PARTS).
    "exr": 192,
    "ex2": 0,
    "comb": 64,
    "unpool": 128,
    "out": 216,
}


# Past one row the attention tail has 8 tasks per row: it moves to CTAs
# whose only earlier task is oproj. The router (16 tasks per pair of rows)
# goes to CTAs free from the end of a sliding layer's attention, so its
# weight stream, alpha-h logits and their puts finish before the tail's b
# lands; after oproj they would delay the router's ready past b. The pool
# moves past the router's CTAs: its input lands late.
# The merge (16 tasks per row) leaves oproj's CTAs 0 .. 127: oproj streams
# its 128 KiB of weights before it polls z, and behind a merge task that
# stream would start only at the merge's end. exr spreads over every CTA
# (moe_experts.EXR_PARTS).
CHAINED_BASES = {
    **BASES,
    "tail": 64,
    "router": 160,
    "pool": 224,
    "merge": 128,
    "exr": 0,
    # Up to 8 rows the comb tasks (8 a row, at most 64) take CTAs 0 .. 63;
    # past 8 rows WIDE moves them to 192. At 8 or 16 rows the 128 unpool
    # tasks (moe_tail.unpool_halves) take CTAs 64 .. 191, after ex2 tasks
    # only: behind a comb task an unpool poll also waited for comb's cmb
    # puts to the 8 ranks to be acknowledged (vmcnt counts stores in order).
    "comb": 0,
    "unpool": 64,
}

# A launch that ends with the next layer's qkvg tasks (qnext, 160 tasks of
# 128 KiB of weights): they take CTAs 192 .. 255 and 0 .. 95, which end their
# expert tasks first (ex2's last lanes) or run only comb before (CTAs 0 .. 63
# at 8 rows); the out tasks (8 a row) move to 96 .. 159, after comb or unpool,
# so no qnext weight stream waits behind an out task.
# exr runs in 2 parts on CTAs 128 .. 255, not spread over every CTA: spread,
# its tasks on qnext's CTAs hold their qnext weight streams back. In one part
# on 64 CTAs, the ex2 tasks behind it end after the others: the tail follows
# the exr work, so halving each CTA's share removes it.
NEXT_BASES = {**CHAINED_BASES, "qnext": 192, "out": 96, "exr": 128}

# Past 8 rows (two requests) the attention tail takes CTAs 64 .. 191 (8 tasks
# a row), and the router's 128 tasks from CTA 160 would wait behind tail
# tasks there or wrap to CTAs behind oproj. The router and topk
# take CTAs 192 .. 255 and 0 .. 63, whose last earlier task is oproj or an
# attention task. The pool's two row groups (64 tasks) take CTAs 208 .. 255
# and 0 .. 15, after their router task and past topk's CTAs 192 .. 207. xq's
# 32 tasks take CTAs 160 .. 191, idle from the attention tail's end: behind a
# pool task, an xq poll would also wait for that task's cg puts to land.
# The comb tasks (128) take CTAs 192 .. 255 and 0 .. 63, so that unpool's
# CTAs 64 .. 191 follow ex2 tasks only (see CHAINED_BASES).
WIDE = {"router": 192, "topk": 192, "pool": 208, "xq": 160, "comb": 192}
# exr's base and parts in a launch of more than 8 rows that ends with qnext:
# spread over every CTA. From CTA 128 in 2 parts (the 8-row qnext placement),
# exr would share CTAs 128 .. 255 with ex2's lanes 4 .. 7, whose ex2 then
# starts only as their own exr task ends, well after the lane's last
# intermediate.
WIDE_NEXT_EXR = (0, 4)


def _bases(rows: int, next_qkvg: bool, *names: str) -> dict[str, int]:
    table = NEXT_BASES if next_qkvg else BASES if rows == 1 else CHAINED_BASES
    if rows > 8:
        table = {**table, **WIDE}
        if next_qkvg:
            table = {**table, "exr": WIDE_NEXT_EXR[0]}
    return {name: table[name] for name in names}


@gluon.jit
def _head(
    mb,
    task,
    seq_len,
    slot,
    k_cache,
    v_cache,
    cos_sin,
    proj_in,
    ROPE: gl.constexpr,
    OWN: gl.constexpr,
):
    """The head task ``task`` (:func:`attention.head_stage`). The kernel
    calls it before or after gate (HEAD_FIRST): one call written once."""
    attention.head_stage(
        mb, task, seq_len, slot, k_cache, v_cache, cos_sin, proj_in, ROPE, OWN, "head"
    )


def _gate_halves(full: bool, rows: int, chain: int) -> int:
    """Gate tasks a 16-row block (qkvg.stages). With one request a full
    layer's attention has one task on every CTA (2 KV heads x FULL_SPLITS),
    so gate's CTAs run their attention task only after their gate task:
    two 8-row tasks a block on every CTA halve that wait. Otherwise gate
    stays off the attention's CTAs, a task a block."""
    one_task_a_cta = (
        full and rows // chain * int(attention.KV_HEADS) * FULL_SPLITS == GRID
    )
    return 2 if one_task_a_cta else 1


def _exr_parts(rows: int, next_qkvg: bool) -> int:
    if rows == 1:
        return 1
    if next_qkvg:
        return WIDE_NEXT_EXR[1] if rows > 8 else 2
    return int(moe_experts.EXR_PARTS)


# The most rows of one launch: two requests' verifies of 7 draft tokens
# (the drafters' block of 8). The GEMVs' MFMAs hold 16 rows.
MAX_ROWS = 16
# KV cache dtypes the attention reads and writes (FP8 at unit scale).
KV_DTYPES = (torch.bfloat16, torch.float8_e4m3fn)


# With warm_next, CTAs 0 .. WARM_TASKS - 1 read the next launch's QKVG and
# gate weights into the memory-side cache (warm.py) after their expert tasks:
# the next launch then starts its projections from the MALL. Earlier, the
# expert weights streaming after them would evict them. The comb tasks start
# at CTA 64.
WARM_TASKS = 64
# CTAs 0 .. CODE_TASKS - 1 first read the launch's own code into the L2s
# (code_warm.py): 8 tasks on each XCD.
CODE_TASKS = 64


# The launch's grid (CTAs) and waves a CTA (num_warps).
GRID = 256
WAVES = 8
# Rows of a launch's stuck-poll record buffer: a record per wave of the
# layer's CTAs, then the header.
STUCK_ROWS = stuck.buffer_rows(GRID, WAVES)


def layer_plan(
    *,
    full: bool,
    rows: int,
    chain: int,
    warm_next: bool,
    own_qkvg: bool,
    next_qkvg: bool,
    check_parity: bool,
    poll_check: int = DEFAULT_POLL_CHECK,
    poll_timeout_shift: int = DEFAULT_POLL_TIMEOUT_SHIFT,
) -> Plan:
    """The plan of a full-attention (``full``) or sliding-window layer for
    ``rows`` token rows: requests of ``chain`` tokens each, 1 for decode,
    ``k + 1`` for a speculative verify of ``k`` draft tokens. With
    ``warm_next`` the launch also warms the next launch's QKVG weights.

    ``own_qkvg``: the launch projects its own q | k | v (qkvg tasks), else
    the previous launch did (its qnext tasks; the launch's attention then
    starts at once). ``next_qkvg``: the launch ends with the next layer's
    projection (qnext), which replaces the warm stage; past one row only.
    ``check_parity``: the scratch keeps the last
    launch's parity (``Mailbox.check_parity``); without it, serving's layout
    is the same as with no check at all.
    ``poll_check`` and ``poll_timeout_shift`` select the plan's poll bound;
    omission uses the production bound."""
    if not 1 <= rows <= MAX_ROWS or rows % chain:
        raise ValueError(
            f"a launch serves 1 to {MAX_ROWS} rows of requests of {chain}, got {rows}"
        )
    if next_qkvg and (rows == 1 or warm_next):
        raise ValueError("qnext: past one row, and in place of the warm stage")
    splits = FULL_SPLITS if full else SLIDING_SPLITS
    # Both kinds of layer lay their regions out alike: the attention
    # regions are sized for the full layer's splits. Consecutive launches
    # alternate the peer regions' two copies, which keeps a rank's puts for
    # the next launch off the copy a slower rank still reads; with two
    # layouts in one scratch, a sliding and a full launch overlapped and the
    # group deadlocked.
    regions = (
        qkvg.REGIONS
        + attention.regions(FULL_SPLITS)
        + attn_tail.regions()
        + moe_route.regions()
        + moe_pool.regions()
        + moe_experts.regions()
        + moe_tail.regions(rows)
        + ((("parity", 1),) if check_parity else ())
    )
    nq = next_qkvg
    stages = (
        (Stage("code", CODE_TASKS, 0),)
        + qkvg.stages(
            _bases(rows, nq, "qkvg", "gate"), own_qkvg, _gate_halves(full, rows, chain)
        )
        + attention.stages(
            _bases(rows, nq, "head", "attn", "merge"), splits, rows, chain, own_qkvg
        )
        + attn_tail.stages(_bases(rows, nq, "oproj", "tail"), rows)
        + moe_route.stages(_bases(rows, nq, "router", "topk"), rows)
        + moe_pool.stages(_bases(rows, nq, "pool"), rows)
        + moe_experts.stages(
            _bases(rows, nq, "slt", "xq", "ex1", "exr", "ex2"),
            rows,
            _exr_parts(rows, next_qkvg),
        )
        + ((Stage("warm", WARM_TASKS, 0),) if warm_next else ())
        + moe_tail.stages(
            _bases(rows, nq, "comb", "unpool", "out"),
            rows,
            8 if next_qkvg else 16,
            next_qkvg,
        )
        + (qkvg.next_stages(_bases(rows, nq, "qnext")) if next_qkvg else ())
    )
    peers = (
        attn_tail.PEERS
        + moe_route.PEERS
        + moe_pool.PEERS
        + moe_experts.PEERS
        + moe_tail.peers(rows)
    )
    return Plan(
        grid=GRID,
        stages=stages,
        regions=regions,
        rows=rows,
        chain=chain,
        world=int(WORLD),
        peers=peers,
        flags=attn_tail.FLAGS + moe_tail.flags(rows),
        poll_check=poll_check,
        poll_timeout_shift=poll_timeout_shift,
    )


def _launch_metadata(grid, kernel, args):
    """Report the launch's rows and the weight bytes it reads to Proton (the
    norm gains and gate coefficients, a few KiB, are left out).

    The experts count each row's 8 picks as distinct experts (at most all
    256): the launch's distinct experts are known only on the device, so
    this is an upper bound."""
    plan = args["PLAN"].plan
    attn_ops, moe_ops = args["attn_ops"], args["moe_ops"]
    # Unpacked as gluon_muse_spark_layer_gfx950 unpacks them.
    w_qkvg, w_o = attn_ops[0], attn_ops[8]
    router_w, w_pool, w13, w13s = moe_ops[0], moe_ops[2], moe_ops[3], moe_ops[4]
    w2, w2s, w_post = moe_ops[9], moe_ops[10], moe_ops[17]
    qkvg_rows, qkv_rows, all_experts = int(QKVG), int(GATE_ROW), int(EXPERTS)
    # The gate rows, and the q | k | v rows when the launch projects them.
    rows_read = qkvg_rows if plan.has_stage("qkvg") else qkvg_rows - qkv_rows
    weight = w_qkvg.nbytes * rows_read // qkvg_rows
    weight += w_o.nbytes + router_w.nbytes + w_pool.nbytes + w_post.nbytes
    experts = min(all_experts, plan.rows * int(moe_experts.TOP_K))
    expert_bytes = w13.nbytes + w13s.nbytes + w2.nbytes + w2s.nbytes
    weight += expert_bytes * experts // all_experts
    if plan.has_stage("warm"):
        weight += 16 * args["warm_quads"]  # 16 bytes a quad
    if plan.has_stage("qnext"):
        weight += args["w_next"].nbytes * qkv_rows // qkvg_rows
    return {"name": f"{kernel.name} (rows={plan.rows})", "bytes": weight}


# The layer index changes every launch: a specialization on it (1, or a
# multiple of 16) would compile new binaries while serving. The warm word
# count is the next layer's, a runtime value too. The code range is the
# compiled binary's own, known only after the compile.
# The page-table row stride varies with request lengths, so keep it runtime.
@gluon.jit(
    launch_metadata=_launch_metadata,
    do_not_specialize=(
        "layer",
        "table_stride",
        "warm_quads",
        "code_offset",
        "code_quads",
    ),
)
def gluon_muse_spark_layer_gfx950(
    scratch,
    step,
    layer,
    bases,
    attn_ops,
    table_stride,
    moe_ops,
    tap,
    tap_col,
    tap_row,
    warm_ptr,
    warm_quads,
    code_offset,
    code_quads,
    proj_in,
    proj_out,
    w_next,
    trace_buf,
    stuck_rec,
    PLAN: gl.constexpr,
    RANK: gl.constexpr,
    SPLITS: gl.constexpr,
    WIN: gl.constexpr,
    ROPE: gl.constexpr,
    TAP: gl.constexpr,
    WHOLE_ROWS: gl.constexpr,
    CHECK_PARITY: gl.constexpr,
):
    """One decode step of one layer on rank ``RANK``.

    ``attn_ops`` (tensors only): (QKVG weight, attention input ``a``,
    ``seq_len``, KV write slot, page table, K cache, V cache, cos / sin,
    o_proj weight, residual slice ``h``, attention gate ``alpha``, ``beta``,
    pre-FFN gain).
    ``table_stride``: the page-table row stride, passed separately so changing
    request lengths cannot specialize the kernel.
    ``moe_ops``: (router columns, router bias, pooler rows, w13, w13 scales,
    their 4 strides, w2, w2 scales, their 4 strides, pre-expert gain,
    post-expert gain, post-expert projection rows, post-FFN gain, FFN gate
    ``alpha``, ``beta`` (all 8192 columns), next input gain (all 8192
    columns), residual output, next attention input).
    ``tap``, ``tap_col``, ``tap_row``: with ``TAP``, where the residual
    after the layer also goes, BF16 whole rows on each rank
    (``moe_tail.out_stage``).
    ``WHOLE_ROWS``: the residual output holds whole FP32 rows, not the owned
    columns. ``CHECK_PARITY``: fail-stop when this
    launch's parity repeats the last one's (``Mailbox.check_parity``).
    ``warm_ptr``, ``warm_quads``: the next launch's packed QKVG weights, read
    (``4 * warm_quads`` int32 words) only when ``PLAN`` has the warm stage.
    ``code_offset``, ``code_quads``: this binary's ``code_warm.text_range``.
    ``proj_in``: int32 ``[rows, qkvg.PROJ_WORDS]``, the q | k | v projection
    the previous launch's qnext tasks stored, read when ``PLAN`` has no qkvg
    stage. ``proj_out``, ``w_next``: where the qnext tasks store the next
    layer's projection, and its fused BF16 weight (with a qnext stage).
    """
    table = trace.allocate(trace_buf)
    peers = mailbox.load_peers(bases, RANK, layer & 1, WORLD)
    mb = mailbox.Mailbox(
        scratch, mailbox.launch_tag(step, layer), PLAN, table, peers, stuck_rec
    )
    if CHECK_PARITY:
        if gl.program_id(0) == 0:
            mb.check_parity(layer & 1)
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
    (
        w_qkvg,
        a_in,
        seq_len,
        slot,
        page_table,
        k_cache,
        v_cache,
        cos_sin,
        w_o,
        h_in,
        alpha1,
        beta1,
        pre_gain,
    ) = attn_ops
    (
        router_w,
        router_bias,
        w_pool,
        w13,
        w13s,
        s13_we,
        s13_se,
        s13_lin,
        s13_nb,
        w2,
        w2s,
        s2_we,
        s2_se,
        s2_lin,
        s2_nb,
        pre_exp_gain,
        pen_gain,
        w_post,
        post_gain,
        alpha2,
        beta2,
        in_gain,
        h_next,
        a_out,
    ) = moe_ops

    OWN: gl.constexpr = schedule.has_stage(PLAN, "qkvg")
    if OWN:
        task, active = schedule.stage_task(PLAN, "qkvg")
        if active:
            qkvg.qkvg_stage(mb, task, w_qkvg, a_in, HIDDEN, "qkvg")
    # With gate split over every CTA (_gate_halves) gate shares the head's
    # CTAs. Without its own qkvg the head is ready at once and every
    # attention task waits on it: it runs before gate. Otherwise (its own
    # qkvg: the head waits on the projection, gate runs meanwhile; or gate
    # off the head's CTAs) the head runs after gate.
    HEAD_FIRST: gl.constexpr = (not OWN) and qkvg.gate_halves(PLAN) == 2
    if HEAD_FIRST:
        task, active = schedule.stage_task(PLAN, "head")
        if active:
            _head(
                mb, task, seq_len, slot, k_cache, v_cache, cos_sin, proj_in, ROPE, OWN
            )
    task, active = schedule.stage_task(PLAN, "gate")
    if active:
        qkvg.gate_stage(mb, task, w_qkvg, a_in, HIDDEN, "gate")
    if not HEAD_FIRST:
        task, active = schedule.stage_task(PLAN, "head")
        if active:
            _head(
                mb, task, seq_len, slot, k_cache, v_cache, cos_sin, proj_in, ROPE, OWN
            )
    # Several requests' attention tasks may outnumber the CTAs: rounds.
    G: gl.constexpr = schedule.grid(PLAN)
    attn_task, attn_rounds = schedule.stage_rounds(PLAN, "attn")
    for i in range(attn_rounds):
        attention.attn_stage(
            mb,
            attn_task + i * G,
            seq_len,
            page_table,
            table_stride,
            k_cache,
            v_cache,
            cos_sin,
            proj_in,
            ROPE,
            OWN,
            SPLITS,
            WIN,
            "attn",
        )
    task, active = schedule.stage_task(PLAN, "merge")
    if active:
        attention.merge_stage(mb, task, SPLITS, "merge")
    task, active = schedule.stage_task(PLAN, "oproj")
    if active:
        attn_tail.oproj_stage(mb, task, w_o, RANK, "oproj")
    task, active = schedule.stage_task(PLAN, "tail1")
    if active:
        attn_tail.tail_stages(mb, task, h_in, alpha1, beta1, pre_gain, RANK)
    task, active = schedule.stage_task(PLAN, "router")
    if active:
        moe_route.router_stage(
            mb, task, router_w, pre_gain, h_in, alpha1, beta1, RANK, "router"
        )
    task, active = schedule.stage_task(PLAN, "topk")
    if active:
        moe_route.topk_stage(mb, task, router_bias, "topk")
    task, active = schedule.stage_task(PLAN, "pool")
    if active:
        moe_pool.pool_stage(mb, task, w_pool, RANK, "pool")
    task, active = schedule.stage_task(PLAN, "slt")
    if active:
        moe_experts.slot_table_stage(mb, "slt")
    task, active = schedule.stage_task(PLAN, "xq")
    if active:
        moe_experts.xq_stage(mb, task, pre_exp_gain, "xq")
    # A CTA's ex1 and ex2 tasks run in parts (see moe_experts): ex1's weights
    # once routing names the experts, the A operand meanwhile (from xq),
    # ex1's products, the K sums (exr),
    # then ex2's weights ahead of ex2's poll.
    task1, active1 = schedule.stage_task(PLAN, "ex1")
    task1 = gl.where(active1, task1, 0)
    task2, active2 = schedule.stage_task(PLAN, "ex2")
    task2 = gl.where(active2, task2, 0)
    ids, e1, e2 = moe_experts.ex1_poll_routes(mb, task1, task2, active1, "ex1")
    loads1 = moe_experts.ex1_issue(
        task1, active1, e1, w13, w13s, s13_we, s13_se, s13_lin, s13_nb
    )
    union = moe_experts.union_of(ids)
    # Every CTA, outside the task branches: it ends in a barrier.
    slot_lds = moe_experts.cta_slots(PLAN, union)
    pre1 = moe_experts.ex1_prefetch(PLAN, task1, active1, slot_lds, union, w13, s13_we)
    a1 = moe_experts.ex1_xq(mb, task1, active1, "ex1")
    if active1:
        moe_experts.ex1_finish(
            mb,
            task1,
            a1,
            loads1,
            e1,
            union,
            slot_lds,
            pre1,
            w13,
            w13s,
            s13_we,
            s13_se,
            s13_lin,
            s13_nb,
            "ex1",
        )
    task, active = schedule.stage_task(PLAN, "exr")
    if active:
        moe_experts.exr_stage(mb, task, union, slot_lds, "exr")
    loads2 = moe_experts.ex2_issue(
        mb, task2, active2, e2, w2, w2s, s2_we, s2_se, s2_lin, s2_nb, "ex2"
    )
    if active2:
        moe_experts.ex2_finish(
            mb,
            task2,
            loads2,
            e2,
            union,
            slot_lds,
            w2,
            w2s,
            s2_we,
            s2_se,
            s2_lin,
            s2_nb,
            RANK,
            "ex2",
        )
    if schedule.has_stage(PLAN, "warm"):
        task, active = schedule.stage_task(PLAN, "warm")
        if active:
            warm.warm_stage(
                mb,
                task,
                warm_ptr,
                warm_quads,
                schedule.stage_tasks(PLAN, "warm"),
                "warm",
            )
    task, active = schedule.stage_task(PLAN, "comb1")
    if active:
        moe_tail.comb_stages(mb, task, pen_gain, RANK)
    task, active = schedule.stage_task(PLAN, "unpool")
    if active:
        moe_tail.unpool_stages(
            mb,
            task,
            w_post,
            post_gain,
            alpha2,
            beta2,
            RANK,
        )
    task, active = schedule.stage_task(PLAN, "out")
    if schedule.has_stage(PLAN, "qnext") and schedule.rows(PLAN) > 8:
        # Past 8 rows out's 128 tasks share 32 CTAs with qnext's 160: those
        # CTAs issue their qnext weights before the out task (qnext reads
        # every out task's words, so out runs first), not after it.
        q_task, q_active = schedule.stage_task(PLAN, "qnext")
        if q_active:
            mb.stamp("qnext", trace.START)
            tiles = qkvg.qnext_tiles(w_next, q_task)
            if active:
                moe_tail.out_stage(
                    mb,
                    task,
                    post_gain,
                    beta2,
                    in_gain,
                    a_out,
                    h_next,
                    tap,
                    tap_col,
                    tap_row,
                    RANK,
                    TAP,
                    WHOLE_ROWS,
                    schedule.stage_tasks(PLAN, "out") // schedule.rows(PLAN),
                    schedule.has_stage(PLAN, "qnext"),
                    "out",
                )
            qkvg.qnext_finish(
                mb,
                q_task,
                tiles,
                proj_out,
                schedule.stage_tasks(PLAN, "out") // schedule.rows(PLAN),
                "qnext",
            )
        elif active:
            moe_tail.out_stage(
                mb,
                task,
                post_gain,
                beta2,
                in_gain,
                a_out,
                h_next,
                tap,
                tap_col,
                tap_row,
                RANK,
                TAP,
                WHOLE_ROWS,
                schedule.stage_tasks(PLAN, "out") // schedule.rows(PLAN),
                schedule.has_stage(PLAN, "qnext"),
                "out",
            )
    else:
        if active:
            moe_tail.out_stage(
                mb,
                task,
                post_gain,
                beta2,
                in_gain,
                a_out,
                h_next,
                tap,
                tap_col,
                tap_row,
                RANK,
                TAP,
                WHOLE_ROWS,
                schedule.stage_tasks(PLAN, "out") // schedule.rows(PLAN),
                schedule.has_stage(PLAN, "qnext"),
                "out",
            )
        if schedule.has_stage(PLAN, "qnext"):
            task, active = schedule.stage_task(PLAN, "qnext")
            if active:
                qkvg.qnext_stage(
                    mb,
                    task,
                    w_next,
                    proj_out,
                    schedule.stage_tasks(PLAN, "out") // schedule.rows(PLAN),
                    "qnext",
                )
    trace.flush(table, trace_buf)


def _launch_layer(
    scratch,
    step,
    layer,
    bases,
    attn_ops,
    table_stride,
    moe_ops,
    tap,
    tap_slot,
    warm_next,
    proj_in,
    proj_out,
    next_qkvg,
    trace_buf,
    stuck_rec,
    rank: int,
    full: bool,
    rows: int,
    chain: int,
    whole_rows: bool,
    check_parity: bool,
    compile_only: bool,
):
    """Launch one layer step of ``rows`` token rows, requests of ``chain``
    tokens (``full``: a
    full-attention, NoPE layer), or with ``compile_only`` only compile the
    launch. ``tap`` (None, or a BF16 ``[rows+, slots, 8192]`` buffer) gets
    the residual after the layer in slot ``tap_slot`` on every rank.
    ``warm_next`` (None, or the next launch's packed QKVG weights) is
    warmed into the memory-side cache after the experts. With
    ``whole_rows`` the residual output (in ``moe_ops``) is whole FP32
    ``[rows, 8192]`` rows, else the rank's owned ``[rows, 1024]`` columns.
    With ``check_parity`` the launch traps when its peer-region parity
    (``layer & 1``) repeats the last launch's (``Mailbox.check_parity``):
    for tests and benches, which must alternate.
    ``proj_in`` (None: the launch projects its own q | k | v) is the
    projection that the previous launch's qnext tasks stored; with
    ``proj_out`` (and the next layer's fused weight ``next_qkvg``) this
    launch's qnext tasks store the next layer's projection there."""
    plan = layer_plan(
        full=full,
        rows=rows,
        chain=chain,
        warm_next=warm_next is not None,
        own_qkvg=proj_in is None,
        next_qkvg=proj_out is not None,
        check_parity=check_parity,
    )
    if tap is None:
        tap_args = (attn_ops[0], 0, 0)
    else:
        tap_args = (tap, tap_slot * tap.shape[2], tap.stride(0))
    if warm_next is None:
        warm_args = (attn_ops[0], 0)
    else:
        if not warm_next.is_contiguous() or warm_next.nbytes % 16:
            raise ValueError("warm_next: contiguous, a whole number of 16-byte quads")
        # The warm stage reads int32 words: the same bytes, viewed.
        warm_args = (warm_next.view(-1).view(torch.int32), warm_next.nbytes // 16)

    # Unused projection operands take a placeholder of the same type.
    proj_args = (
        scratch.view(torch.int32) if proj_in is None else proj_in,
        scratch.view(torch.int32) if proj_out is None else proj_out,
        attn_ops[0] if next_qkvg is None else next_qkvg,
    )

    def kernel_args(code: tuple[int, int]) -> tuple:
        return (
            scratch,
            step,
            layer,
            bases,
            attn_ops,
            table_stride,
            moe_ops,
            *tap_args,
            *warm_args,
            *code,
            *proj_args,
            trace_buf,
            stuck_rec,
            plan.key,
            rank,
            FULL_SPLITS if full else SLIDING_SPLITS,
            0 if full else WINDOW,
            not full,
            tap is not None,
            whole_rows,
            check_parity,
        )

    # The device libraries of _extern.wave_uniform_i32 (attention.py).
    kernel = gluon_muse_spark_layer_gfx950.warmup(
        *kernel_args((0, 0)),
        num_warps=WAVES,
        grid=(plan.grid,),
        **extern_compile_options(),
    )
    if compile_only:
        return
    # The code range is a runtime argument: the same binary.
    gluon_muse_spark_layer_gfx950[(plan.grid,)](
        *kernel_args(code_warm.text_range(kernel)),
        num_warps=WAVES,
        **extern_compile_options(),
    )


def launch_gluon_muse_spark_layer_gfx950(
    weights,
    *,
    h,
    a,
    h_out,
    a_out,
    seq_len,
    slot,
    page_table,
    k_cache,
    v_cache,
    cos_sin,
    scratch,
    step,
    peer_bases,
    layer: int,
    chain: int,
    whole_rows: bool,
    tap,
    tap_slot,
    warm_next,
    proj_in,
    proj_out,
    trace_buf,
    stuck_rec,
    check_parity: bool,
    compile_only: bool,
):
    """One step of a packed layer (``weights.LayerWeights``) for the rows of
    one or more requests of ``chain`` tokens each: a decode row each, or a
    speculative verify's consecutive tokens (row ``t`` of request ``b`` at
    position ``seq_len[b chain] - 1 + t``). The row count ``rows`` comes
    from ``h``; ``rows`` and ``chain`` select the plan.

    Args:
        weights: From ``weights.pack_layer``.
        h: FP32 ``[rows, 1024]`` owned residual columns.
        h_out: The FP32 residual after the layer: whole ``[rows, 8192]``
            rows with ``whole_rows``, else ``[rows, 1024]`` owned columns.
        a, a_out: BF16 ``[rows, 8192]`` attention input of this layer and of
            the next one (normed with the next layer's input gain).
        seq_len, slot: int32 ``[rows]``: each row's length including its
            token, and the KV cache slot of the token.
        page_table: int32 ``[requests, pages]`` (or ``[pages]`` for one
            request): each request's page table (64 tokens a page, indexed
            by absolute position).
        k_cache, v_cache: BF16, or FP8 e4m3 holding unit-scale values,
            ``[slots, 2, 64]``.
        cos_sin: FP32 ``[positions, 64]`` rotary table (unused by a full,
            NoPE layer).
        scratch: int64 mailbox pairs on the Iris symmetric heap.
        step: int32 ``[1]`` step counter.
        peer_bases: int64 ``[8]`` heap bases of the group's ranks.
        layer: The decoder layer index (the tag and the peer copy).
        chain: Tokens a request: 1 for decode, ``k + 1`` for a verify.
        whole_rows: Every rank stores whole residual rows into ``h_out``
            (for a next layer on the regular path), not its owned columns.
        tap: None, or a BF16 ``[rows+, slots, 8192]`` buffer on each rank:
            the launch also stores the residual after the layer, rounded to
            BF16, into slot ``tap_slot`` of this rank's buffer (a draft's
            tap); every rank's launch stores the same values.
        tap_slot: The tap's slot, or None without ``tap``.
        warm_next: None, or the next launch's ``weights.LayerWeights``. Without
            ``proj_out`` this launch reads its QKVG weights into the
            memory-side cache after the experts, so the next launch's
            projections read the MALL.
        proj_in: None (the launch projects its own q | k | v at its start),
            or int32 ``[>= rows, qkvg.PROJ_WORDS]``: the projection that the
            previous launch stored with ``proj_out``. The launch's attention
            then starts at once.
        proj_out: None, or int32 ``[>= rows, qkvg.PROJ_WORDS]`` (past one
            row, with ``warm_next``): the launch ends with ``warm_next``'s
            q | k | v projection of its output rows, stored there for the
            next launch's ``proj_in``.
        trace_buf: None, or a trace buffer (``trace.py``).
        stuck_rec: None, or a host-visible ``int64 [STUCK_ROWS,
            mailbox.STUCK_WORDS]`` stuck-poll record buffer (a record per wave
            of the 256 CTAs of 8 waves, then the header;
            ``stuck.stuck_buffer(256, 8, ...)``): a poll that gives up writes
            what it waited for there before it traps (``stuck.decode`` reads
            it).
        check_parity: Trap when this launch's parity (``layer & 1``) repeats
            the last launch's on this rank (tests and benches; serving's
            launches alternate by construction, so it leaves the check off).
        compile_only: Only compile the launch for these operands. A rank's
            launch polls its peers' values for about a minute before it traps,
            so the ranks compile before any of them launches.
    """
    rows = h.shape[0] if h.dim() == 2 else 0
    owned, hidden = int(OWNED), int(HIDDEN)
    # The launch reads h and a while other CTAs already write the outputs.
    if (
        h.dtype != torch.float32
        or h.shape != (rows, owned)
        or not h.is_contiguous()
        or h.data_ptr() == h_out.data_ptr()
    ):
        raise ValueError(
            f"h: contiguous FP32 [rows, {owned}] that does not alias h_out"
        )
    width = hidden if whole_rows else owned
    if (
        h_out.dtype != torch.float32
        or h_out.shape != (rows, width)
        or not h_out.is_contiguous()
    ):
        raise ValueError(f"h_out: contiguous FP32 [rows, {width}]")
    if (
        a.dtype != torch.bfloat16
        or a.shape != (rows, hidden)
        or not a.is_contiguous()
        or a.data_ptr() == a_out.data_ptr()
    ):
        raise ValueError(
            f"a: contiguous BF16 [rows, {hidden}] that does not alias a_out"
        )
    for name, t in (("seq_len", seq_len), ("slot", slot)):
        if t.dtype != torch.int32 or t.numel() != rows:
            raise ValueError(f"{name}: int32 [{rows}], one per row")
    # FP8 caches hold unit-scale e4m3 values (the kernel applies no scale).
    if k_cache.dtype not in KV_DTYPES or v_cache.dtype != k_cache.dtype:
        raise ValueError("k_cache, v_cache: both BF16, or both FP8 e4m3 at unit scale")
    if page_table.dim() == 1:
        page_table = page_table[None]
    if (
        rows % chain
        or page_table.dim() != 2
        or page_table.shape[0] != rows // chain
        or page_table.stride(1) != 1
    ):
        raise ValueError(
            f"page_table: int32 [{rows // chain}, pages], one "
            f"row a request of {chain} tokens"
        )
    if (tap is None) != (tap_slot is None):
        raise ValueError("tap and tap_slot come together")
    for name, t in (("proj_in", proj_in), ("proj_out", proj_out)):
        if t is not None and (
            t.dtype != torch.int32
            or t.dim() != 2
            or t.shape[0] < rows
            or t.shape[1] != int(qkvg.PROJ_WORDS)
            or not t.is_contiguous()
        ):
            raise ValueError(
                f"{name}: contiguous int32 [>={rows}, {int(qkvg.PROJ_WORDS)}]"
            )
    if proj_out is not None and warm_next is None:
        raise ValueError("proj_out: the next layer's weights come in warm_next")
    if (
        proj_in is not None
        and proj_out is not None
        and (proj_in.data_ptr() == proj_out.data_ptr())
    ):
        raise ValueError("proj_in and proj_out: two buffers (the launch reads one)")
    if tap is not None and (
        tap.dtype != torch.bfloat16
        or tap.dim() != 3
        or tap.shape[0] < rows
        or tap.shape[2] != hidden
        or not tap.is_contiguous()
        or not 0 <= tap_slot < tap.shape[1]
    ):
        raise ValueError(
            f"tap: contiguous BF16 [>={rows}, slots, {hidden}], a slot in it"
        )
    if stuck_rec is not None and stuck_rec.shape[0] < STUCK_ROWS:
        raise ValueError(f"stuck_rec: int64 [>= {STUCK_ROWS}, STUCK_WORDS]")
    attn_ops, table_stride, moe_ops = weights.kernel_args(
        h=h,
        a=a,
        seq_len=seq_len,
        slot=slot,
        page_table=page_table,
        k_cache=k_cache,
        v_cache=v_cache,
        cos_sin=cos_sin,
        h_out=h_out,
        a_out=a_out,
    )
    _launch_layer(
        scratch,
        step,
        layer,
        peer_bases,
        attn_ops,
        table_stride,
        moe_ops,
        tap,
        tap_slot,
        None if warm_next is None or proj_out is not None else warm_next.qkvg,
        proj_in,
        proj_out,
        None if proj_out is None else warm_next.qkvg,
        trace_buf,
        stuck_rec,
        weights.rank,
        weights.full,
        rows,
        chain,
        whole_rows,
        check_parity,
        compile_only,
    )


__all__ = [
    "BASES",
    "MAX_ROWS",
    "launch_gluon_muse_spark_layer_gfx950",
    "FULL_SPLITS",
    "KV_DTYPES",
    "SLIDING_SPLITS",
    "gluon_muse_spark_layer_gfx950",
    "layer_plan",
]
