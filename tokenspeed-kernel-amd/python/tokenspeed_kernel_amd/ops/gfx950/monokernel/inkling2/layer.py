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

"""The Inkling2 mid local layer as one persistent launch a rank (stages in
``plan.local_layer_plan`` order; each module's docstring gives its math)."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._extern import extern_compile_options
from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mailbox, schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2 import (
    attention,
    conv,
    experts,
    head,
    inputs,
    mix,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2 import plan as ip
from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2 import (
    projection,
    quant,
    route,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import LAYER_SLOTS

NUM_WARPS = 8
SET_QUERIES = gl.constexpr(ip.SET_QUERIES)
WO_SLOTS = gl.constexpr(ip.WO_SLOTS)


@gluon.constexpr_function
def shared_rows(plan, full_splits):
    """The verify block rows a full attention task takes (0: a row a task,
    and for a local layer)."""
    # plan.shared_attention's rule (a function here would join the JIT key).
    if full_splits == 0 or plan.plan.chain != 8:
        return 0
    return ip.SHARED_ATTENTION[plan.plan.rows]


@gluon.constexpr_function
def head_sources(plan):
    """Whether the plan's head stage reads three sources' projections (a
    full layer, or a local one in a looped core's later passes)."""
    return any(
        stage.name == "head" and "kproj" in stage.reads for stage in plan.plan.stages
    )


@gluon.jit
def _sweeps(PLAN: gl.constexpr, mb, sweep, STAGE: gl.constexpr):
    """This CTA's tasks of a boundary layer's sweep stage ``STAGE``:
    ``sweep`` (the layer's MLP query) or ``sweep_rest`` (the set's other
    queries)."""
    if schedule.has_stage(PLAN, STAGE):
        G: gl.constexpr = schedule.grid(PLAN)
        FIRST: gl.constexpr = 0 if STAGE == "sweep" else 1
        QUERIES: gl.constexpr = 1 if STAGE == "sweep" else SET_QUERIES - 1
        task, rounds = schedule.stage_rounds(PLAN, STAGE)
        for i in range(rounds):
            mix.sweep_stage(
                mb,
                task + i * G,
                mix.Sweep(*sweep),
                "cand",
                "set_acc",
                "set_ms",
                QUERIES,
                FIRST,
                STAGE,
            )


# The layer index and the parity are runtime values: one compiled kernel
# serves every layer and both parities (an integer argument of 1 would
# otherwise specialize).
@gluon.jit(do_not_specialize=["layer", "parity"])
def gluon_inkling2_local_layer_gfx950(
    scratch,
    step,
    layer,
    parity,
    bases,
    terms,
    sources,
    cand_out,
    attn_term,
    mlp_term,
    attn_query,
    mlp_query,
    sweep,
    qkvr_w,
    kv_proj_w,
    tau,
    head_weights,
    head_cache,
    attn_cache,
    ring,
    wo_w,
    attn_taps,
    router_w,
    router_bias,
    up_bank,
    down_bank,
    mlp_taps,
    trace_buf,
    PLAN: gl.constexpr,
    RANK: gl.constexpr,
    NVFP4: gl.constexpr,
    BOUNDARY: gl.constexpr,
    FULL_SPLITS: gl.constexpr,
    RING_ROWS: gl.constexpr,
):
    """One decode (or verify) step of a mid local layer on rank ``RANK``,
    with NVFP4 (``NVFP4``) or MXFP8 experts.

    ``terms`` holds the previous layer's pending BF16 rows (2 or 3 global
    pointers ``[rows, HIDDEN]``), or for layer 0 (a plan with
    ``norm_attn``) the embedding rows. The launch writes this layer's terms:
    ``cand_out`` (``P``), ``attn_term`` (``A``) and ``mlp_term`` (``M``),
    the KV cache rows and the conv ring rows of its tokens. The tuple
    operands hold the fields of ``mix.Query``, ``head.Weights``,
    ``head.Cache``, ``attention.Cache``, ``conv.Ring`` (without the chain and
    the ring's shape: ``RING_ROWS`` rows a slot, fixed per server, of the
    plan's ``ring_width``) and ``experts.Bank`` (NVFP4) or ``experts.Mxfp8Bank``. ``parity`` (0 or 1) must alternate from one launch
    to the next on every rank.

    A full attention layer (``FULL_SPLITS`` KV splits, 0 for a local layer)
    takes three attention queries (``attn_query``: the Q, K and V sources'
    ``mix.Query`` fields), the K and V sources' projection tiles
    (``kv_proj_w``, ``()`` for a local layer) and the rows' FP32 ``tau``
    (None for a local layer).
    """
    table = trace.allocate(trace_buf)
    D: gl.constexpr = schedule.dims(PLAN)
    HIDDEN: gl.constexpr = D.hidden
    W: gl.constexpr = schedule.world(PLAN)
    CHAIN: gl.constexpr = schedule.chain(PLAN)
    # Per-row stages of more tasks than CTAs run in rounds.
    G: gl.constexpr = schedule.grid(PLAN)
    peers = mailbox.load_peers(bases, RANK, parity, W)
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, layer), PLAN, table, peers)
    rings = conv.Ring(ring[0], ring[1], ring[2], CHAIN, RING_ROWS, D.ring_width)

    FULL: gl.constexpr = FULL_SPLITS > 0
    if FULL & schedule.has_stage(PLAN, "mix_attn"):
        task, active = schedule.stage_task(PLAN, "mix_attn")
        if active:
            row = task // 3
            j = task % 3
            row_terms = ()
            for t in gl.static_range(len(terms)):
                row_terms += (terms[t] + row * HIDDEN,)
            if j == 0:
                q = mix.Query(*attn_query[0]).row_of(row, HIDDEN)
            elif j == 1:
                q = mix.Query(*attn_query[1]).row_of(row, HIDDEN)
            else:
                q = mix.Query(*attn_query[2]).row_of(row, HIDDEN)
            mix.full_attention_mix_stage(
                mb.row(row),
                row_terms,
                cand_out + row * HIDDEN,
                q,
                j,
                "cand",
                "mix_attn",
            )
    if (not FULL) & schedule.has_stage(PLAN, "mix_attn"):
        row, active = schedule.stage_task(PLAN, "mix_attn")
        if active:
            row_terms = ()
            for t in gl.static_range(len(terms)):
                row_terms += (terms[t] + row * HIDDEN,)
            mix.attention_mix_stage(
                mb.row(row),
                row_terms,
                cand_out + row * HIDDEN,
                mix.Query(*attn_query).row_of(row, HIDDEN),
                "x_attn",
                "cand",
                "mix_attn",
            )
    if schedule.has_stage(PLAN, "norm_attn"):
        row, active = schedule.stage_task(PLAN, "norm_attn")
        if active:
            mix.attention_norm_stage(
                mb.row(row),
                terms[0] + row * HIDDEN,
                cand_out + row * HIDDEN,
                mix.Query(*attn_query).gain,
                "x_attn",
                "cand",
                "norm_attn",
            )
    if schedule.has_stage(PLAN, "in_attn"):
        task, active = schedule.stage_task(PLAN, "in_attn")
        if active:
            inputs.attention_input_stage(mb, task, sources, "in_attn")
    if BOUNDARY:
        _sweeps(PLAN, mb, sweep, "sweep")
        _sweeps(PLAN, mb, sweep, "sweep_rest")
    projection.chained_stages(
        PLAN, mb, qkvr_w, "x_attn", "qkvr", "qkvr", HIDDEN, D.qkvr_splits, False
    )
    # The K and V sources' projections (a plan without their stages skips
    # them).
    if len(kv_proj_w) == 2:
        projection.chained_stages(
            PLAN,
            mb,
            kv_proj_w[0],
            "x_k",
            "kproj",
            "kproj",
            HIDDEN,
            D.qkvr_splits,
            False,
        )
        projection.chained_stages(
            PLAN,
            mb,
            kv_proj_w[1],
            "x_v",
            "vproj",
            "vproj",
            HIDDEN,
            D.qkvr_splits,
            False,
        )
    if FULL & schedule.has_stage(PLAN, "head"):
        task, rounds = schedule.stage_rounds(PLAN, "head")
        for i in range(rounds):
            head.full_head_stage(
                mb,
                task + i * G,
                head.Weights(*head_weights),
                head.Cache(*head_cache),
                rings,
                tau,
                "qkvr",
                "kproj",
                "vproj",
                "q",
                "kv",
                "rel",
                "head",
            )
    THREE: gl.constexpr = head_sources(PLAN)
    if (not FULL) & THREE & schedule.has_stage(PLAN, "head"):
        task, rounds = schedule.stage_rounds(PLAN, "head")
        for i in range(rounds):
            head.three_source_head_stage(
                mb,
                task + i * G,
                head.Weights(*head_weights),
                head.Cache(*head_cache),
                rings,
                "qkvr",
                "kproj",
                "vproj",
                "q",
                "kv",
                "rel",
                "head",
            )
    if (not FULL) & (not THREE) & schedule.has_stage(PLAN, "head"):
        task, rounds = schedule.stage_rounds(PLAN, "head")
        for i in range(rounds):
            head.head_stage(
                mb,
                task + i * G,
                head.Weights(*head_weights),
                head.Cache(*head_cache),
                rings,
                "qkvr",
                "q",
                "kv",
                "rel",
                "head",
            )
    a_cache = attention.Cache(*attn_cache)
    SHARED_ROWS: gl.constexpr = shared_rows(PLAN, FULL_SPLITS)
    SHARED: gl.constexpr = SHARED_ROWS > 0
    if SHARED & schedule.has_stage(PLAN, "attn"):
        task, rounds = schedule.stage_rounds(PLAN, "attn")
        attention.shared_attention_stage(
            mb,
            task,
            rounds,
            a_cache,
            "q",
            "kv",
            "rel",
            "attn_part",
            "attn_lse",
            FULL_SPLITS,
            SHARED_ROWS // 4,
            "attn",
        )
    if FULL & (not SHARED) & schedule.has_stage(PLAN, "attn"):
        task, rounds = schedule.stage_rounds(PLAN, "attn")
        attention.full_attention_stage(
            mb,
            task,
            rounds,
            a_cache,
            "q",
            "kv",
            "rel",
            "attn_part",
            "attn_lse",
            CHAIN,
            FULL_SPLITS,
            "attn",
        )
    if FULL & schedule.has_stage(PLAN, "merge"):
        task, rounds = schedule.stage_rounds(PLAN, "merge")
        for i in range(rounds):
            attention.full_merge_stage(
                mb,
                task + i * G,
                a_cache,
                "qkvr",
                "attn_part",
                "attn_lse",
                "o",
                "o_flag",
                FULL_SPLITS,
                "merge",
            )
    if (not FULL) & schedule.has_stage(PLAN, "attn"):
        task, rounds = schedule.stage_rounds(PLAN, "attn")
        attention.attention_stage(
            mb,
            task,
            rounds,
            a_cache,
            "q",
            "kv",
            "rel",
            "attn_part",
            "attn_lse",
            CHAIN,
            "attn",
        )
    if (not FULL) & schedule.has_stage(PLAN, "merge"):
        task, rounds = schedule.stage_rounds(PLAN, "merge")
        for i in range(rounds):
            attention.merge_stage(
                mb,
                task + i * G,
                a_cache,
                "qkvr",
                "attn_part",
                "attn_lse",
                "o",
                "o_flag",
                "merge",
            )
    if schedule.has_stage(PLAN, "wo"):
        task, active = schedule.stage_task(PLAN, "wo")
        if active:
            projection.partial_stage(
                mb,
                task,
                wo_w,
                "o",
                "o_flag",
                attention.merge_tasks(PLAN),
                None,
                "wo_part",
                "wo_flag",
                D.q_dim,
                HIDDEN,
                2,
                D.wo_splits,
                0,
                D.wo_splits,
                WO_SLOTS,
                RANK,
                projection.wo_scatter(PLAN),
                "wo",
            )
    if schedule.has_stage(PLAN, "fold_attn"):
        task, active = schedule.stage_task(PLAN, "fold_attn")
        if active:
            projection.scatter_fold_stage(
                mb,
                task,
                "wo_part",
                "wo_flag",
                "wo_gather",
                "wo_gather_flag",
                HIDDEN,
                32,
                RANK,
                "fold_attn",
            )
    if schedule.has_stage(PLAN, "red_attn"):
        task, active = schedule.stage_task(PLAN, "red_attn")
        if active:
            conv.reduce_conv_stage(
                mb,
                task,
                "wo_gather" if projection.wo_scatter(PLAN) else "wo_part",
                "wo_gather_flag" if projection.wo_scatter(PLAN) else "wo_flag",
                projection.wo_scatter(PLAN),
                "attn_sc",
                attn_term,
                attn_taps,
                rings,
                D.attn_channel0,
                HIDDEN,
                32,
                "red_attn",
            )
    if schedule.has_stage(PLAN, "in_mlp"):
        row, active = schedule.stage_task(PLAN, "in_mlp")
        if active:
            inputs.mlp_input_stage(mb, row, sources[0], "in_mlp")
    if schedule.has_stage(PLAN, "mix_mlp"):
        row, active = schedule.stage_task(PLAN, "mix_mlp")
        if active:
            mix.mlp_mix_stage(
                mb.row(row),
                mix.Query(*mlp_query).row_of(row, HIDDEN),
                "cand",
                "attn_sc",
                "x_mlp",
                "set_acc",
                "set_ms",
                BOUNDARY,
                "mix_mlp",
            )
    projection.chained_stages(
        PLAN,
        mb,
        router_w,
        "x_mlp",
        "router",
        "logits",
        HIDDEN,
        D.router_splits,
        True,
    )
    if schedule.has_stage(PLAN, "topk"):
        row, active = schedule.stage_task(PLAN, "topk")
        if active:
            route.topk_stage(mb.row(row), router_bias, "logits", "route", "topk")
    if schedule.has_stage(PLAN, "quant"):
        row, active = schedule.stage_task(PLAN, "quant")
        if active:
            if NVFP4:
                quant.quant_stage(
                    mb.row(row), "x_mlp", "xq", "xa", "xq_flag", HIDDEN, "quant"
                )
            else:
                quant.mxfp8_quant_stage(
                    mb.row(row), "x_mlp", "xq", "xq_flag", HIDDEN, "quant"
                )
    if schedule.has_stage(PLAN, "ex1") or schedule.has_stage(PLAN, "ex2"):
        # Once a launch: the routes stay the same from ex1 to ex2.
        routes = experts.local_routes(mb, "route", RANK)
    if schedule.has_stage(PLAN, "ex1"):
        task, active = schedule.stage_task(PLAN, "ex1")
        if active:
            experts.up_stage(
                mb,
                task,
                experts.bank(up_bank, NVFP4),
                routes,
                "xq",
                "xa" if NVFP4 else None,
                "xq_flag",
                "gate_up",
                "ex1",
                NVFP4,
            )
    if schedule.has_stage(PLAN, "swiglu"):
        task, rounds = schedule.stage_rounds(PLAN, "swiglu")
        for i in range(rounds):
            experts.swiglu_stage(
                mb,
                task + i * G,
                "route",
                "gate_up",
                "hq",
                "ha" if NVFP4 else None,
                "hq_flag",
                RANK,
                "swiglu",
                NVFP4,
            )
    if schedule.has_stage(PLAN, "ex2"):
        task, active = schedule.stage_task(PLAN, "ex2")
        if active:
            experts.down_stage(
                mb,
                task,
                experts.bank(down_bank, NVFP4),
                routes,
                "hq",
                "ha" if NVFP4 else None,
                "hq_flag",
                "ex2",
                "ex2",
                NVFP4,
            )
    if schedule.has_stage(PLAN, "fold"):
        task, active = schedule.stage_task(PLAN, "fold")
        if active:
            experts.fold_stage(
                mb, task, "route", "ex2", "fold_part", "fold_flag", 32, RANK, "fold"
            )
    if schedule.has_stage(PLAN, "red_moe"):
        task, active = schedule.stage_task(PLAN, "red_moe")
        if active:
            conv.reduce_conv_stage(
                mb,
                task,
                "fold_part",
                "fold_flag",
                False,
                "mlp_sc",
                mlp_term,
                mlp_taps,
                rings,
                D.mlp_channel0,
                HIDDEN,
                32,
                "red_moe",
            )
    trace.flush(table, trace_buf)


def launch_gluon_inkling2_local_layer_gfx950(
    plan_key,
    sources: tuple[torch.Tensor, ...],
    experts: str,
    boundary: bool,
    first: bool,
    full_splits: int,
    rank: int,
    scratch: torch.Tensor,
    step: torch.Tensor,
    layer: int,
    parity: int,
    bases: torch.Tensor,
    terms: tuple[torch.Tensor, ...],
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    queries: tuple[tuple, tuple],
    sweep: tuple,
    attention_weights: tuple,
    kv_proj_weights: tuple,
    tau: torch.Tensor | None,
    caches: tuple,
    ring: tuple,
    mlp_weights: tuple,
    trace_buf: torch.Tensor | None,
):
    """Launch one mid local layer step (:func:`gluon_inkling2_local_layer_gfx950`).

    Args:
        plan_key: ``plan.local_layer_plan(dims, rows, chain, experts,
            boundary, first).key``.
        experts: The routed experts' format, ``"nvfp4"`` or ``"mxfp8"``.
        boundary: Whether the layer starts an attention-residual block.
        first: Whether the layer is layer 0 (a boundary layer whose block
            starts from the embedding).
        full_splits: A full attention layer's KV splits
            (``plan.full_layer_plan(..., splits)``), 0 for a local layer.
        rank: This rank in the attention TP group.
        scratch: The plan's mailbox buffer on the Iris heap (int64).
        step: int32 ``[1]`` device step counter.
        layer, parity: The layer index (for the launch tag) and this
            launch's peer-region copy (alternates from launch to launch).
        bases: int64 ``[world]`` Iris heap bases.
        terms: The previous layer's pending BF16 rows (2 or 3); layer 0's
            embedding rows.
        sources: A later pass's part's normalized BF16 ``[rows, hidden]``
            inputs (``plan.later_pass_plan``): the Q, K and V sources of the
            attention part, the MoE input of the MoE part; ``()`` otherwise.
        outputs: BF16 ``[rows, HIDDEN]`` ``(P, A, M)`` this layer writes.
        queries: ``(attention, mlp)`` fields of ``mix.Query``. A boundary
            layer's MLP query reads its set partial from the sweep. A full
            layer's attention query is three: the Q, K and V sources'.
        sweep: A boundary layer's ``mix.Sweep`` fields (the stored blocks,
            their count, the new set's score weights and its partials'
            outputs); ``()`` for a mid layer.
        attention_weights: ``(qkvr tiles, head.Weights fields, wo tiles,
            attention output conv taps)``.
        kv_proj_weights: A full layer's ``(K source, V source)`` projection
            tiles (``plan.KV_PROJ`` rows each); ``()`` for a local layer.
        tau: A full layer's FP32 ``[rows]`` logit scales; None for a local
            layer.
        caches: ``(head.Cache fields, attention.Cache fields)``.
        ring: ``(ring, slots, positions)`` of ``conv.Ring``: the BF16 ring
            ``[slots, rows, dims.ring_width]``, its rows a slot the server's.
        mlp_weights: ``(router tiles, router bias, w13 bank fields, w2 bank
            fields, MLP output conv taps)``: ``experts.Bank`` fields for
            NVFP4, ``experts.Mxfp8Bank`` fields for MXFP8.
        trace_buf: Optional stage timestamps (``trace.py``).
    """
    # A step's launches tell their mailbox values apart by layer slot; a
    # larger index would alias the next step's tags.
    if not 0 <= layer < LAYER_SLOTS:
        raise ValueError(
            f"layer index {layer} is outside the {LAYER_SLOTS} launch tag "
            "slots of a step"
        )
    nvfp4 = experts == "nvfp4"
    # Only NVFP4 expert stages read or write the row maxima.
    for stage in plan_key.plan.stages:
        maxima = bool({"xa", "ha"} & set(stage.reads + stage.writes))
        if stage.name in ("quant", "ex1", "swiglu", "ex2") and maxima != nvfp4:
            raise ValueError(f"the plan is not one of {experts} experts")
    # Only a boundary layer's plan (or a slice of it) sweeps, and only layer
    # 0's normalizes its embedding.
    if not boundary and plan_key.plan.has_stage("sweep"):
        raise ValueError("a mid layer's launch of a boundary layer's plan")
    if plan_key.plan.has_stage("norm_attn" if not first else "mix_attn"):
        raise ValueError("layer 0's plan launches layer 0 only")
    full = full_splits > 0
    attn = [stage for stage in plan_key.plan.stages if stage.name == "attn"]
    if attn and attn[0].tasks != ip.attention_tasks(
        plan_key.plan.dims,
        plan_key.plan.rows,
        plan_key.plan.chain,
        full_splits if full else None,
    ):
        raise ValueError("the launch's split count is not its plan's")
    # Stages that project or read the K and V sources need their tiles.
    three = any({"x_k", "kproj"} & set(stage.reads) for stage in plan_key.plan.stages)
    # A full layer's head scales its rows by tau (an MoE part has no head).
    scaled = full and plan_key.plan.has_stage("head")
    if (
        (three and len(kv_proj_weights) != 2)
        or (scaled and tau is None)
        or (not full and tau is not None)
    ):
        raise ValueError(
            "a launch of three sources takes K and V tiles, a full layer's tau"
        )
    reads = {
        "in_attn": (3, plan_key.plan.has_stage("in_attn")),
        "in_mlp": (1, plan_key.plan.has_stage("in_mlp")),
    }
    count = sum(n for n, present in reads.values() if present)
    rows, hidden = plan_key.plan.rows, plan_key.plan.dims.hidden
    if len(sources) != count or any(
        tuple(x.shape) != (rows, hidden) or not x.is_contiguous() for x in sources
    ):
        raise ValueError(
            f"the plan reads {count} contiguous [{rows}, {hidden}] input rows"
        )
    qkvr_w, head_weights, wo_w, attn_taps = attention_weights
    head_cache, attn_cache = caches
    router_w, router_bias, up_bank, down_bank, mlp_taps = mlp_weights
    cand_out, attn_term, mlp_term = outputs
    if ring[0].shape[2] != plan_key.plan.dims.ring_width:
        raise ValueError("the conv ring's rows are not the plan's ring width")
    gluon_inkling2_local_layer_gfx950[(ip.GRID,)](
        scratch,
        step,
        layer,
        parity,
        bases,
        tuple(terms),
        tuple(sources),
        cand_out,
        attn_term,
        mlp_term,
        queries[0],
        queries[1],
        tuple(sweep),
        qkvr_w,
        tuple(kv_proj_weights),
        tau,
        head_weights,
        head_cache,
        attn_cache,
        ring,
        wo_w,
        attn_taps,
        router_w,
        router_bias,
        up_bank,
        down_bank,
        mlp_taps,
        trace_buf,
        plan_key,
        rank,
        nvfp4,
        boundary,
        full_splits,
        # The ring's rows a slot (the server's conv taps less one plus its
        # verify tokens) are fixed per server.
        ring[0].shape[1],
        num_warps=NUM_WARPS,
        **extern_compile_options(),
    )
