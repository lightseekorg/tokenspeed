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

"""One Qwen3.8-Flash-Next decoder layer as one launch, for 1 to ``MAX_ROWS``
token rows (decode, or the chained rows of a speculative verify: ``Plan.chain``).

Both layer kinds share the hyper-connection mixers (hc.py), the output
projection and residual combine (out_proj.py) and the MoE (moe.py):

    Gated DeltaNet layer:  [p1 p2] a1 a2 | b1 b3 (gdn.py)                | c1 | m1 m2 m3 m4 m5
    QSA attention layer:   a1 a2 | q1 q2 q3 q4 q4f q5 q6 (qsa.py) | c1 | m1 m2 m3 m4 m5 [f1 f2]

``p1 p2`` are the PLE stages of the layers that have one (ple.py). ``f1 f2``
are the final mixer of the model, which the launch of the last layer also runs.

A launch reads the input streams ``h_in`` and writes the output streams
``h_out`` (both BF16 ``[rows, 4 * HIDDEN]``). It also updates the caches of
the layer: the GDN conv and SSM state, the PLE state, or the KV cache and the
QSA indexer caches. In a verify, it also updates the verify scratch and
staging of TokenSpeed.

Placement: every CTA walks the stages in order. A task issues its weight loads
before it polls its inputs. So the weights of a stage stream while the stages
before it compute, but only on CTAs that finished their earlier tasks. The
stage bases put each stage on CTAs that are free early. For example, M1
shares the CTAs of A1, which finish long before the MLP half starts.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, replace

import torch
from tokenspeed_kernel_amd._scheduling import sched_compile_options
from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.megakernel import (
    mailbox,
    schedule,
    trace,
)
from tokenspeed_kernel_amd.ops.gfx950.megakernel.plan import Plan, Stage
from tokenspeed_kernel_amd.ops.gfx950.megakernel.qwen4_exp import (
    gdn,
    hc,
    moe,
    moe_bf16,
    out_proj,
    ple,
    qsa,
)
from tokenspeed_kernel_amd.ops.gfx950.megakernel.qwen4_exp.weights import (
    DeltaLayerWeights,
    MoeBf16Weights,
    QsaLayerWeights,
)

GRID = 256
NUM_WARPS = 8
# s_sleep units (64 clocks each) CTAs without a mixer task wait at the start.
HEAD_SLEEP = gl.constexpr(32)
# C1's first CTA. C1 runs on CTAs C1_BASE .. C1_BASE + 159.
C1_BASE = 96
# The routed and shared experts' formats a plan supports: MXFP4 weights and
# activations (moe.py) or BF16 (moe_bf16.py).
EXPERTS = ("mxfp4", "bf16")


def _tail(
    src: tuple[str, ...], final: bool, rows: int, experts: str
) -> tuple[Stage, ...]:
    """The output projection, the MLP mixer and the MoE. ``src`` names the
    mailbox region of the input streams, if any. ``experts`` is the experts'
    format (:data:`EXPERTS`).

    With ``final``, the model's final mixer runs after the MoE on the output
    streams (M5 also publishes them in region ``h``), on CTAs that are idle
    then, and writes the LM head's input.
    """
    final_mixer = (
        hc.stages("f_", {"f1": 57, "f2": 0}, rows, ("h",), False) if final else ()
    )
    if experts not in EXPERTS:
        raise ValueError(f"experts must be one of {EXPERTS}, got {experts!r}")
    mxfp4 = experts == "mxfp4"
    return (
        *out_proj.stages({"c1": C1_BASE}, "a_", src),
        # The MLP mixer also publishes its row as MXFP4 for MXFP4 experts.
        *hc.stages("m_", {"m1": 0, "m2": 176}, rows, ("w",), mxfp4),
        *(moe if mxfp4 else moe_bf16).stages(
            # M5 first takes the CTAs after M4's: their M5 weights stream
            # while M4 runs.
            {"m3": 28, "m4": 93, "m5": 96},
            rows,
            "m_",
            "w",
            ("h",) if final else (),
        ),
        *final_mixer,
    )


def _regions(attention, final: bool, experts: str):
    mxfp4 = experts == "mxfp4"
    final_regions = (
        hc.regions("f_", False) + (("h", int(hc.WIDTH) // 2),) if final else ()
    )
    return (
        hc.regions("a_", False)
        + attention
        + out_proj.REGIONS
        + hc.regions("m_", mxfp4)
        + (moe.REGIONS if mxfp4 else moe_bf16.REGIONS)
        + final_regions
    )


def experts_format(weights: DeltaLayerWeights | QsaLayerWeights) -> str:
    """The format (:data:`EXPERTS`) of a packed layer's experts."""
    return "bf16" if isinstance(weights.moe, MoeBf16Weights) else "mxfp4"


@functools.cache
def delta_plan(with_ple: bool, rows: int, chain: int, experts: str) -> Plan:
    """The plan of a Gated DeltaNet layer for ``rows`` token rows, ``chain``
    consecutive rows per request (``Plan.chain``), with ``experts`` in the
    format :data:`EXPERTS` names.

    With ``with_ple``, the PLE stages (ple.py) first turn the input streams
    into the streams the rest of the layer reads (mailbox region ``ph``).
    """
    src = ("ph",) if with_ple else ()
    stages = (
        *(ple.stages({"p1": 0, "p2": 176}, rows) if with_ple else ()),
        *hc.stages("a_", {"a1": 0, "a2": 176}, rows, src, False),
        # B1's slots map to its tasks so that the z tasks avoid A2's CTAs
        # (gdn.Z_FIRST).
        *gdn.stages({"b1": 0, "b3": 28}, rows, chain, "a_"),
        *_tail(src, False, rows, experts),
    )
    regions = _regions(gdn.REGIONS, False, experts) + (ple.REGIONS if with_ple else ())
    return Plan(grid=GRID, stages=stages, regions=regions, rows=rows, chain=chain)


@functools.cache
def qsa_plan(final: bool, rows: int, chain: int, experts: str, catch_up: bool) -> Plan:
    """The plan of a QSA full-attention layer for ``rows`` token rows,
    ``chain`` per request (``Plan.chain``), with ``experts`` in the format
    :data:`EXPERTS` names. With ``final``, the final mixer of the model runs
    after the layer (see :func:`_tail`).

    With ``catch_up`` (an MTP draft catch-up step, one request), every row
    runs up to Q2, which writes its K / V and index key. The stages from Q3
    on run for the request's live row only (:func:`live_plan`), as
    TokenSpeed's draft narrowing does.
    """
    if catch_up and rows != chain:
        raise ValueError(f"a catch-up plan serves one request, got {rows} / {chain}")
    attended = 1 if catch_up else rows
    # Q6 takes the CTAs before C1's when its tasks fit there (up to two
    # rows): C1's weights then stream while Q6 runs.
    q6 = 0 if int(qsa.MERGE_TASKS) * attended <= C1_BASE else 128
    attention = qsa.stages(
        {"q1": 28, "q2": 246, "q3": 0, "q4": 20, "q4f": 255, "q5": 64, "q6": q6},
        rows,
        attended,
        "a_",
    )
    stages = (
        *hc.stages("a_", {"a1": 0, "a2": 176}, rows, (), False),
        *attention,
        *_tail((), final, attended, experts),
    )
    return Plan(
        grid=GRID,
        stages=stages,
        regions=_regions(qsa.REGIONS, final, experts),
        rows=rows,
        chain=chain,
    )


def live_plan(plan: Plan, catch_up: bool) -> Plan:
    """The plan of the stages from Q3 on: ``plan``, or for a catch-up plan
    (:func:`qsa_plan`) one row, the request's live row. The stages and
    regions stay the same, so a mailbox of this plan at the live row
    (``Mailbox.row``) addresses that row's regions."""
    return replace(plan, rows=1, chain=1) if catch_up else plan


@gluon.constexpr_function
def _stream_src(plan):
    """Where the attention half reads the layer's streams: the PLE stages'
    region ``ph``, or "" for the global input."""
    return "ph" if plan.plan.has_stage("p1") else ""


@gluon.constexpr_function
def _stage_name(plan, index):
    return plan.plan.stages[index].name


@gluon.jit
def _attention_mixer(mb, streams, mixer, PLAN: gl.constexpr):
    task, active = schedule.stage_task(PLAN, "a1")
    if active:
        hc.reduce_stage(mb, task, streams, mixer, "a_", "a1")
    task, active = schedule.stage_task(PLAN, "a2")
    if active:
        hc.mix_stage(mb, task, streams, mixer, "a_", None, False, "a2")


@gluon.constexpr_function
def _bf16_experts(plan):
    """Whether the plan's experts are BF16 (``moe_bf16.py``)."""
    return any(name == "dnb" for name, _ in plan.plan.regions)


@gluon.constexpr_function
def _mxfp4_experts(plan):
    """Whether the plan's experts are MXFP4 (``moe.py``)."""
    return not _bf16_experts(plan)


@gluon.jit
def _expert_weights(experts, PLAN: gl.constexpr):
    """The MoE operand bundle of the plan's expert format."""
    if _bf16_experts(PLAN):
        weights = moe_bf16.Weights(*experts)
    else:
        weights = moe.Weights(*experts)
    return weights


@gluon.jit
def _expert_stages(mb, h_out, experts, OUT: gl.constexpr, PLAN: gl.constexpr):
    """M4 and M5. With MXFP4 experts, a CTA with a task of each runs them as
    one helper, which issues the M5 weight loads before M4 ends."""
    if _bf16_experts(PLAN):
        first, rounds = schedule.stage_rounds(PLAN, "m4")
        for i in range(rounds):
            moe_bf16.up_stage(mb, first + i * schedule.grid(PLAN), experts, "m_")
        task, active = schedule.stage_task(PLAN, "m5")
        if active:
            moe_bf16.down_stage(mb, task, experts, h_out, "m_", "w", OUT)
    else:
        _mxfp4_expert_stages(mb, h_out, experts, OUT, PLAN)


@gluon.jit
def _mxfp4_expert_stages(mb, h_out, experts, OUT: gl.constexpr, PLAN: gl.constexpr):
    up_task, up_active = schedule.stage_task(PLAN, "m4")
    down_task, down_active = schedule.stage_task(PLAN, "m5")
    if up_active:
        if down_active:
            moe.up_down_stage(
                mb, up_task, down_task, experts, h_out, "m_", "m_", "w", OUT
            )
        else:
            moe.up_stage(mb, up_task, experts, "m_")
    elif down_active:
        moe.down_stage(mb, down_task, experts, h_out, "m_", "w", OUT)


@gluon.jit
def _tail_stages(
    mb,
    streams,
    h_out,
    out_w,
    mlp_mixer,
    experts,
    final_mixer,
    mixed_out,
    PLAN: gl.constexpr,
):
    """The stages of :func:`_tail` (``streams``: the layer's input).
    The function reads ``final_mixer`` (the operand of ``hc.Mixer``) and
    ``mixed_out`` only when ``PLAN`` has the final mixer."""
    OUT: gl.constexpr = "h" if schedule.has_stage(PLAN, "f1") else ""
    task, active = schedule.stage_task(PLAN, "c1")
    if active:
        out_proj.out_stage(mb, task, out_w, streams, "", "a_", "c1")
    # The MLP mixer reads the streams that C1 publishes.
    mlp_in = hc.Streams(streams.ptr, streams.stride, "w")
    task, active = schedule.stage_task(PLAN, "m1")
    if active:
        hc.reduce_stage(mb, task, mlp_in, mlp_mixer, "m_", "m1")
    task, active = schedule.stage_task(PLAN, "m2")
    if active:
        hc.mix_stage(
            mb, task, mlp_in, mlp_mixer, "m_", None, _mxfp4_experts(PLAN), "m2"
        )
    task, active = schedule.stage_task(PLAN, "m3")
    if active:
        moe.router_stage(mb, task, experts, "m_", "m3")
    _expert_stages(mb, h_out, experts, OUT, PLAN)
    if schedule.has_stage(PLAN, "f1"):
        # The final mixer reads the output streams that M5 publishes.
        final_in = hc.Streams(h_out, hc.HIDDEN, "h")
        final = hc.Mixer(*final_mixer)
        task, active = schedule.stage_task(PLAN, "f1")
        if active:
            hc.reduce_stage(mb, task, final_in, final, "f_", "f1")
        task, active = schedule.stage_task(PLAN, "f2")
        if active:
            hc.mix_stage(mb, task, final_in, final, "f_", mixed_out, False, "f2")


@gluon.jit
def _ple_stages(mb, h_in, h_stride, weights, step, PLAN: gl.constexpr):
    """P1 and P2 (P2 runs one row per task, on that row's operands)."""
    task, active = schedule.stage_task(PLAN, "p1")
    if active:
        ple.projection_stage(mb, task, weights, step, "p1")
    streams = hc.Streams(h_in, h_stride, "")
    first, rounds = schedule.stage_rounds(PLAN, "p2")
    for i in range(rounds):
        row, task, tok = schedule.row_task(
            PLAN, first + i * schedule.grid(PLAN), ple.CONV_TASKS
        )
        ple.conv_stage(
            mb.row(row),
            task,
            tok,
            streams.row(row),
            weights,
            step.row(row, schedule.chain(PLAN)),
            "p2",
        )


@gluon.jit
def _delta_stages(mb, weights, state, PLAN: gl.constexpr):
    b1_task, b1_active = schedule.stage_task(PLAN, "b1")
    b3_task, b3_active = schedule.stage_task(PLAN, "b3")
    # B3 runs one task per (request, value head).
    b3_request = b3_task // gdn.VALUE_HEADS
    b3_head = b3_task % gdn.VALUE_HEADS
    # A CTA with a B1 and a B3 task loads the recurrent state with B1's
    # weights (projection_recurrence_stage). Past two rows the state's
    # registers next to B1's would spill, so B3 then loads its own state.
    fused = b3_active & (schedule.rows(PLAN) <= 2)
    if b1_active:
        if fused:
            gdn.projection_recurrence_stage(
                mb, b1_task, b3_request, b3_head, weights, state, "a_"
            )
        else:
            gdn.projection_stage(mb, b1_task, weights, state, "a_", "b1")
    if b3_active & ~(b1_active & fused):
        gdn.recurrence_stage(mb, b3_request, b3_head, weights, state, "b3")


@gluon.jit
def gluon_qwen4_exp_delta_layer_gfx950(
    scratch,
    step,
    layer,
    h_in,
    h_stride,
    h_out,
    attn_mixer,
    gdn_weights,
    gdn_state,
    out_w,
    mlp_mixer,
    experts,
    ple_weights,
    ple_step,
    trace_buf,
    PLAN: gl.constexpr,
):
    """One step of a Gated DeltaNet layer (see the module docstring).

    The tuple operands hold the fields of ``hc.Mixer`` (``attn_mixer``,
    ``mlp_mixer``), ``gdn.Weights``, ``gdn.State``, ``moe.Weights``
    (``experts``), ``ple.Weights`` and ``ple.Step``. The PLE operands are
    ``None`` unless ``PLAN`` has the PLE stages.
    """
    table = trace.allocate(trace_buf)
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, layer), PLAN, table)
    streams = hc.Streams(h_in, h_stride, _stream_src(PLAN))
    schedule.head_sleep(PLAN, _stage_name(PLAN, 0), _stage_name(PLAN, 1), HEAD_SLEEP)
    if schedule.has_stage(PLAN, "p1"):
        _ple_stages(
            mb, h_in, h_stride, ple.Weights(*ple_weights), ple.Step(*ple_step), PLAN
        )
    _attention_mixer(mb, streams, hc.Mixer(*attn_mixer), PLAN)
    _delta_stages(mb, gdn.Weights(*gdn_weights), gdn.State(*gdn_state), PLAN)
    if schedule.has_stage(PLAN, "p1"):
        # B1 depends on every P1 task, so no read of the input context page
        # (which can be the output page) is still pending.
        task, active = schedule.stage_task(PLAN, "b1")
        if task == 0:
            weights = ple.Weights(*ple_weights)
            step_ops = ple.Step(*ple_step)
            for r in gl.static_range(schedule.rows(PLAN)):
                ple.write_context(
                    weights,
                    step_ops.row(r, schedule.chain(PLAN)),
                    r % schedule.chain(PLAN),
                )
    _tail_stages(
        mb,
        streams,
        h_out,
        out_w,
        hc.Mixer(*mlp_mixer),
        _expert_weights(experts, PLAN),
        None,
        None,
        PLAN,
    )
    trace.flush(table, trace_buf)


@gluon.jit
def _prologue(mb, weights, cache, PLAN: gl.constexpr):
    """This CTA's Q2 task, if any."""
    CHAIN: gl.constexpr = schedule.chain(PLAN)
    task, active = schedule.stage_task(PLAN, "q2")
    if active:
        row, task, tok = schedule.row_task(PLAN, task, qsa.PROLOGUE_TASKS)
        qsa.prologue_stage(mb.row(row), task, tok, weights, cache.row(row, CHAIN), "q2")


@gluon.jit
def _attention_stages(mb, cache, TOK0, CHAIN0: gl.constexpr, PLAN: gl.constexpr):
    """Q3-Q6, one row per task, on the operands of that row. In a catch-up
    step, ``mb`` and ``cache`` are the views of the live row, ``TOK0`` is its
    token and ``CHAIN0`` the launch's chain. Else they are 0 and
    ``Plan.chain``."""
    G: gl.constexpr = schedule.grid(PLAN)
    CHAIN: gl.constexpr = schedule.chain(PLAN)
    first, rounds = schedule.stage_rounds(PLAN, "q3")
    for i in range(rounds):
        row, task, tok = schedule.row_task(PLAN, first + i * G, qsa.SCORE_TASKS)
        qsa.score_stage(mb.row(row), task, tok + TOK0, cache.row(row, CHAIN), "q3")
    task, active = schedule.stage_task(PLAN, "q4")
    if active:
        row, chunk, _ = schedule.row_task(PLAN, task, qsa.SELECT_CHUNKS)
        qsa.topk_stage(mb.row(row), chunk, cache.row(row, CHAIN).length(), "q4")
    task, active = schedule.stage_task(PLAN, "q4f")
    if active:
        qsa.topk_final_stage(mb.row(task), 0, cache.row(task, CHAIN).length(), "q4f")
    task, active = schedule.stage_task(PLAN, "q5")
    if active:
        row, split, tok = schedule.row_task(PLAN, task, qsa.ATTN_TASKS)
        qsa.attention_stage(
            mb.row(row), split, tok + TOK0, cache.row(row, CHAIN), CHAIN0, "q5"
        )
    task, active = schedule.stage_task(PLAN, "q6")
    if active:
        row, task, _ = schedule.row_task(PLAN, task, qsa.MERGE_TASKS)
        qsa.merge_stage(mb.row(row), task, "q6")


@gluon.jit
def gluon_qwen4_exp_qsa_layer_gfx950(
    scratch,
    step,
    layer,
    h_in,
    h_stride,
    h_out,
    attn_mixer,
    qsa_weights,
    cache,
    out_w,
    mlp_mixer,
    experts,
    final_mixer,
    mixed_out,
    trace_buf,
    PLAN: gl.constexpr,
    LIVE: gl.constexpr,
    ADVANCE: gl.constexpr,
):
    """One step of a QSA attention layer (see the module docstring).

    The tuple operands hold the fields of ``hc.Mixer`` (``attn_mixer``,
    ``mlp_mixer``, ``final_mixer``), ``qsa.Weights``, ``qsa.Cache`` and
    ``moe.Weights`` (``experts``). ``final_mixer`` and ``mixed_out`` are
    ``None`` unless ``PLAN`` has the final mixer of the model. ``LIVE`` is
    the plan of the stages from Q3 on (:func:`live_plan`). With
    ``ADVANCE``, the launch moves the step on (:func:`_advance_step`).
    """
    table = trace.allocate(trace_buf)
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, layer), PLAN, table)
    streams = hc.Streams(h_in, h_stride, "")
    weights = qsa.Weights(*qsa_weights)
    cache = qsa.Cache(*cache)
    schedule.head_sleep(PLAN, "a1", "a2", HEAD_SLEEP)
    _attention_mixer(mb, streams, hc.Mixer(*attn_mixer), PLAN)
    task, active = schedule.stage_task(PLAN, "q1")
    if active:
        qsa.projection_stage(mb, task, weights, "a_", "q1")
    _prologue(mb, weights, cache, PLAN)
    if _catch_up(PLAN, LIVE):
        # The stages from Q3 on see the live row as their row 0.
        live = _live_row(cache, schedule.chain(PLAN))
        live_mb = mailbox.Mailbox(mb.row(live).scratch, mb.tag, LIVE, table)
        _attention_stages(
            live_mb,
            cache.row(live, schedule.chain(PLAN)),
            live,
            schedule.chain(PLAN),
            LIVE,
        )
        _tail_stages(
            live_mb,
            streams.row(live),
            h_out,
            out_w,
            hc.Mixer(*mlp_mixer),
            _expert_weights(experts, LIVE),
            final_mixer,
            mixed_out,
            LIVE,
        )
    else:
        _attention_stages(mb, cache, 0, schedule.chain(PLAN), PLAN)
        _tail_stages(
            mb,
            streams,
            h_out,
            out_w,
            hc.Mixer(*mlp_mixer),
            _expert_weights(experts, PLAN),
            final_mixer,
            mixed_out,
            PLAN,
        )
    trace.flush(table, trace_buf)
    if ADVANCE:
        _advance_step(step, schedule.grid(PLAN))


@gluon.constexpr_function
def _catch_up(plan, live):
    """Whether the launch is a catch-up step (:func:`live_plan`)."""
    return plan.plan.rows != live.plan.rows


@gluon.jit
def _live_row(cache, CHAIN: gl.constexpr):
    """The live row of a catch-up step: the row of the request's last
    accepted token, which TokenSpeed's draft gathers. Token ``t`` sits at
    ``seq_len - CHAIN + t``, the last accepted one at ``accepted - 1``."""
    t = gl.load(cache.accepted) - gl.load(cache.seq_len) + CHAIN - 1
    return gl.minimum(gl.maximum(t, 0), CHAIN - 1)


@gluon.jit
def _advance_step(step, G: gl.constexpr):
    """Move ``step[0]`` on once every CTA of the launch is done: each CTA
    counts itself in ``step[1]``, and the last one resets the count and
    increments the step. Every CTA read the step when it began (the barrier
    covers the CTA's warps), so the launch's tag stays the same."""
    gl.barrier()
    done = gl.atomic_add(step + 1, 1, sem="acq_rel", scope="gpu")
    if done == G - 1:
        gl.store(step + 1, 0)
        gl.store(step, gl.load(step) + 1)


# Rows one launch serves: the GEMVs carry the rows in the MFMA rows,
# and the per-row stages' tasks fit the grid in a few rounds.
MAX_ROWS = 4


def _rows(h: torch.Tensor) -> int:
    rows = h.shape[0] if h.dim() == 2 else 0
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"a launch serves 1 to {MAX_ROWS} rows, got {rows}")
    return rows


def _check_streams(
    h: torch.Tensor,
    h_out: torch.Tensor,
    scratch: torch.Tensor,
    plan: Plan,
    out_rows: int,
) -> None:
    """``h`` has the plan's rows, ``h_out`` ``out_rows`` rows."""
    width, hidden = int(hc.WIDTH), int(hc.HIDDEN)
    for name, t, n, widths in (
        ("h", h, plan.rows, (width, hidden)),
        ("h_out", h_out, out_rows, (width,)),
    ):
        if (
            t.dtype != torch.bfloat16
            or t.dim() != 2
            or t.shape[0] != n
            or t.shape[1] not in widths
            or (not t.is_contiguous())
        ):
            raise ValueError(f"{name} must be contiguous BF16 [{n}, {widths}]")
    if h.data_ptr() == h_out.data_ptr():
        raise ValueError("h_out must not alias h: late stages still read h")
    if scratch.dtype != torch.int64 or scratch.numel() < plan.scratch_pairs:
        raise ValueError(f"scratch needs {plan.scratch_pairs} int64 pairs")


def _check_rows(plan: Plan, **operands: torch.Tensor) -> None:
    """Per-row operands hold one entry (or table row) per row."""
    for name, t in operands.items():
        if t.shape[0] != plan.rows:
            raise ValueError(f"{name} needs {plan.rows} rows, got {tuple(t.shape)}")


def _check_requests(plan: Plan, **operands: torch.Tensor) -> None:
    """Per-request operands hold one entry (or table row) per request
    (``plan.chain`` rows)."""
    requests = plan.rows // plan.chain
    for name, t in operands.items():
        if t.shape[0] != requests:
            raise ValueError(f"{name} needs {requests} requests, got {tuple(t.shape)}")


def _stream_stride(h: torch.Tensor) -> int:
    """Elements between the input streams: 0 when ``h`` is one row that
    stands for all four (``hc.stream_ptr``)."""
    return 0 if h.shape[-1] == int(hc.HIDDEN) else int(hc.HIDDEN)


@dataclass(frozen=True)
class PleStep:
    """The PLE block's per-step operands (``ple.Step``)."""

    token: torch.Tensor  # int [rows]: each row's token
    context: torch.Tensor  # int64 context pool [pages, 2] (oldest first)
    conv_state: torch.Tensor  # conv state pool [pages, 4 * HIDDEN, 9]
    page_in: torch.Tensor  # int32 [requests]: each request's input page
    # int32 [rows]: where the state after each row's token goes, a row of
    # context_out and conv_out (the output page in decode, the verify
    # scratch row in a verify). A value that is not positive means none.
    page_out: torch.Tensor
    context_out: torch.Tensor  # int64 [*, 2]: the context pool or scratch
    conv_out: torch.Tensor  # [*, 4 * HIDDEN, 9]: the conv pool or scratch

    def kernel_args(self) -> tuple:
        """The kernel operand of ``ple.Step``."""
        return (
            self.token,
            self.page_in,
            self.page_out,
            self.context,
            self.context.stride(0),
            self.conv_state,
            self.conv_state.stride(0),
            self.context_out,
            self.context_out.stride(0),
            self.conv_out,
            self.conv_out.stride(0),
        )


def launch_gluon_qwen4_exp_delta_layer_gfx950(
    h: torch.Tensor,
    h_out: torch.Tensor,
    weights: DeltaLayerWeights,
    *,
    chain: int,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    state_in: torch.Tensor,
    state_out: torch.Tensor,
    conv_out: torch.Tensor,
    conv_out_rows: torch.Tensor,
    replay: torch.Tensor | None,
    scratch: torch.Tensor,
    step: torch.Tensor,
    layer: int,
    trace_buf: torch.Tensor | None,
    ple_step: PleStep | None,
) -> None:
    """Run one Gated DeltaNet layer for ``rows`` (1 to ``MAX_ROWS``) token
    rows: ``chain`` consecutive tokens per request (1 for decode, ``k + 1``
    for a speculative verify, whose tokens chain the conv window and the
    recurrent state from row to row).

    Args:
        h: Input streams, BF16 ``[rows, 4 * HIDDEN]``, or ``[rows, HIDDEN]``:
            one row per request that stands for all four streams (the first
            layer's embedding).
        h_out: Output streams, BF16 ``[rows, 4 * HIDDEN]`` (may not alias ``h``).
        weights: The layer (``pack_delta_layer``).
        conv_state: Conv state pool ``[pages, QKV, 3]`` BF16. Pages may be
            strided. The inner planes must be dense.
        ssm_state: SSM state pool ``[pages, VALUE_HEADS, V, K]`` FP32. The same
            rule applies.
        state_in, state_out: int32 ``[requests]``: each request's input and
            output page (-1 for a padding request). The state after its last
            token goes to the output page (a verify passes -1: the launch
            writes no state).
        conv_out, conv_out_rows: Where each row's conv window after its
            token goes: ``[QKV, 3]`` at row ``conv_out_rows[row]`` (int32
            ``[rows]``, -1: none) of ``conv_out`` (``[*, QKV, 3]`` BF16, dense
            planes). In decode, these are ``conv_state`` and ``state_out``. In
            a verify, they are the verify scratch rows.
        replay: ``None``, or the layer's TokenSpeed replay payload ``[>= rows,
            QKV - KEY + 2 * VALUE_HEADS]`` (model dtype) that a verify fills
            with each row's ``[k | v | a | b]``.
        scratch: Mailbox scratch, int64, at least ``delta_plan().scratch_pairs``.
            All layers of a step share it. It needs no clearing.
        step: int32 ``[1]``, a counter that changes before every step. With
            ``layer`` (``< plan.LAYER_SLOTS``) it tags this launch's mailbox
            writes, so no write of an earlier launch can satisfy a poll.
        layer: The decoder layer index.
        trace_buf: Optional int64 ``[GRID, trace.EVENTS, NUM_WARPS]`` stage timestamps.
        ple_step: The PLE block's per-step operands, exactly when ``weights``
            has a PLE block.
    """
    if (weights.ple is None) != (ple_step is None):
        raise ValueError("ple_step must be given exactly for a layer with PLE")
    plan = delta_plan(weights.ple is not None, _rows(h), chain, experts_format(weights))
    _check_streams(h, h_out, scratch, plan, plan.rows)
    _check_rows(plan, conv_out_rows=conv_out_rows)
    _check_requests(plan, state_in=state_in, state_out=state_out)
    if (
        tuple(conv_state.stride()[1:]) != (3, 1)
        or tuple(conv_out.stride()[1:]) != (3, 1)
        or ssm_state.stride(-1) != 1
    ):
        raise ValueError("state pages must hold dense planes")
    width = int(gdn.QKV - gdn.KEY + 2 * gdn.VALUE_HEADS)
    if replay is not None and (
        replay.shape[0] < plan.rows
        or replay.shape[-1] != width
        or replay.stride(-1) != 1
    ):
        raise ValueError(
            f"replay payload needs [>= {plan.rows}, {width}], got {tuple(replay.shape)}"
        )
    if ple_step is not None:
        state = int(ple.STATE)
        if tuple(ple_step.conv_state.stride()[1:]) != (state, 1) or tuple(
            ple_step.conv_out.stride()[1:]
        ) != (state, 1):
            raise ValueError("PLE conv state pages must be dense")
        if ple_step.context.stride(-1) != 1 or ple_step.context_out.stride(-1) != 1:
            raise ValueError("PLE context pages must be dense")
        _check_rows(plan, token=ple_step.token, page_out=ple_step.page_out)
        _check_requests(plan, page_in=ple_step.page_in)
    gdn_state = (
        conv_state,
        conv_state.stride(0),
        conv_out,
        conv_out_rows,
        conv_out.stride(0),
        ssm_state,
        ssm_state.stride(0),
        replay,
        0 if replay is None else replay.stride(0),
        state_in,
        state_out,
    )
    gluon_qwen4_exp_delta_layer_gfx950[(plan.grid,)](
        scratch,
        step,
        layer,
        h,
        _stream_stride(h),
        h_out,
        weights.attn_mixer.kernel_args(),
        weights.gdn_args(),
        gdn_state,
        weights.out_w,
        weights.mlp_mixer.kernel_args(),
        weights.moe.kernel_args(),
        None if ple_step is None else weights.ple.kernel_args(),
        None if ple_step is None else ple_step.kernel_args(),
        trace_buf,
        plan.key,
        num_warps=NUM_WARPS,
        # The device library of _scheduling.wave_uniform_i32 (moe.py).
        **sched_compile_options(),
    )


def launch_gluon_qwen4_exp_qsa_layer_gfx950(
    h: torch.Tensor,
    h_out: torch.Tensor,
    weights: QsaLayerWeights,
    *,
    chain: int,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    seq_len: torch.Tensor,
    full_table: torch.Tensor,
    page_size: int,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    qsa_table: torch.Tensor,
    recent_table: torch.Tensor,
    compressed: torch.Tensor,
    raw_key: torch.Tensor,
    rope_position: torch.Tensor,
    staging: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
    draft: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    accepted: torch.Tensor | None,
    scratch: torch.Tensor,
    step: torch.Tensor,
    layer: int,
    mixed_out: torch.Tensor | None,
    advance_step: bool,
    trace_buf: torch.Tensor | None,
) -> None:
    """Run one QSA attention layer for ``rows`` (1 to ``MAX_ROWS``) token
    rows, ``chain`` consecutive tokens per request (1 for decode, ``k + 1``
    for a speculative verify). See the launcher of the delta layer.

    Args:
        h, h_out, scratch, step, layer, trace_buf: As in
            :func:`launch_gluon_qwen4_exp_delta_layer_gfx950`.
        weights: The layer (``pack_qsa_layer``).
        cos_sin: The rotary cache, FP32 ``[max_position, 64]``.
        positions: int64 ``[3, rows]`` MRoPE positions of each row's token.
        seq_len: int32 ``[requests]``: ``L + 1`` for a request's last token
            at ``L``. Token ``t`` of a request's ``chain`` rows sits at
            ``L - chain + 1 + t`` (TokenSpeed's verify layout).
        full_table, page_size: The requests' KV page tables (int32
            ``[requests, pages]``) and the page size in tokens.
        k_cache, v_cache: ``[slots, KV_HEADS, HEAD]`` BF16, token rows dense.
        qsa_table, recent_table: The requests' compressed-key (256 tokens a
            page) and recent-ring (64 tokens a page) page tables,
            ``[requests, pages]``.
        compressed: ``[pages, 64, 1, IDX_DIM]`` BF16 compressed keys.
        raw_key: ``[pages, 4, 1, IDX_DIM]`` BF16 raw index keys.
        rope_position: ``[pages, 3]`` int64 group-start positions.
        staging: None in decode (the index keys go to the recent ring). In a
            verify, the TokenSpeed ``QSAVerifyState`` rows ``(raw keys [>= rows,
            IDX_DIM] of this layer, positions [>= rows, 3] int64, logical
            positions [>= rows] int64, recent locations [>= rows] int32)``,
            which the launch writes instead of the ring (TokenSpeed commits the
            accepted ones).
        draft: None, or in an MTP draft step (``chain`` 1, one row per
            request) the requests' QSA draft scratch ``(raw keys [rows, 4, 1,
            IDX_DIM] BF16, group-start positions [rows, 3] int64, position
            tags [rows, 4] int64)``, which the launch writes instead of the
            ring. Compression reads a member from it where its tag matches.
        accepted: None, or in an MTP draft catch-up step (the draft's first
            step: one request's ``chain`` verified tokens, no staging) the
            request's int32 accepted length ``[1]``. A row at or past it is a
            rejected token: it writes its K / V but no index or compressed
            key. As TokenSpeed's draft narrowing, only the live row (the
            last accepted token's) runs the attention and the rest of the
            layer: ``h_out`` and ``mixed_out`` then hold that one row.
        mixed_out: The model's final mixer output, BF16 ``[out rows, HIDDEN]``
            (the rows of ``h_out``), when ``weights`` has the final mixer
            (the last layer). Else ``None``.
        advance_step: Whether the launch moves the step on when it ends (the
            last launch of a forward): ``step`` is then int32 ``[2]``, the
            step and a count that is 0 between launches.
    """
    final = weights.final_mixer is not None
    if advance_step and (
        step.dtype != torch.int32 or step.numel() < 2 or not step.is_contiguous()
    ):
        raise ValueError("advance_step needs step as int32 [step, count]")
    rows = _rows(h)
    catch_up = accepted is not None
    out_rows = rows // chain if catch_up else rows
    if final != (mixed_out is not None):
        raise ValueError("mixed_out must be given exactly when the final mixer is")
    if final and (
        mixed_out.dtype != torch.bfloat16
        or tuple(mixed_out.shape) != (out_rows, int(hc.HIDDEN))
        or not mixed_out.is_contiguous()
    ):
        raise ValueError(
            f"mixed_out must be contiguous BF16 [{out_rows}, {int(hc.HIDDEN)}]"
        )
    plan = qsa_plan(final, rows, chain, experts_format(weights), catch_up)
    if staging is not None:
        token_k, staged_positions, logical, recent = staging
        if (
            token_k.shape[0] < rows
            or token_k.stride(0) != int(qsa.IDX_DIM)
            or staged_positions.stride(0) != 3
            or logical.dtype != torch.int64
            or recent.dtype != torch.int32
        ):
            raise ValueError("staging needs dense rows of QSAVerifyState")
    if draft is not None:
        draft_k, draft_positions, draft_logical = draft
        if chain != 1 or staging is not None:
            raise ValueError("a draft step is one row per request, without staging")
        if (
            tuple(draft_k.shape) != (rows, int(qsa.COMPRESS), 1, int(qsa.IDX_DIM))
            or tuple(draft_positions.shape) != (rows, 3)
            or tuple(draft_logical.shape) != (rows, int(qsa.COMPRESS))
            or draft_positions.dtype != torch.int64
            or draft_logical.dtype != torch.int64
            or not all(
                t.is_contiguous() for t in (draft_k, draft_positions, draft_logical)
            )
        ):
            raise ValueError("draft needs the dense QSA draft scratch of the rows")
    if accepted is not None and (
        staging is not None
        or draft is not None
        or accepted.dtype != torch.int32
        or tuple(accepted.shape) != (rows // chain,)
    ):
        raise ValueError("accepted needs int32 [requests], without staging or draft")
    _check_streams(h, h_out, scratch, plan, out_rows)
    _check_rows(plan, positions=positions.T)
    _check_requests(
        plan,
        seq_len=seq_len,
        full_table=full_table,
        qsa_table=qsa_table,
        recent_table=recent_table,
    )
    head, idx = int(qsa.HEAD), int(qsa.IDX_DIM)
    for name, t, inner in (
        ("k_cache", k_cache, (head, 1)),
        ("v_cache", v_cache, (head, 1)),
        ("compressed", compressed, (idx, idx, 1)),
        ("raw_key", raw_key, (idx, idx, 1)),
        ("rope_position", rope_position, (1,)),
    ):
        if tuple(t.stride()[1:]) != inner:
            raise ValueError(f"{name} pages must be dense, got strides {t.stride()}")
    if k_cache.stride(0) != v_cache.stride(0):
        raise ValueError("k_cache and v_cache must share the slot stride")
    cache = (
        cos_sin,
        positions,
        positions.stride(0),
        seq_len,
        full_table,
        full_table.stride(0),
        page_size,
        k_cache,
        v_cache,
        k_cache.stride(0),
        qsa_table,
        qsa_table.stride(0),
        recent_table,
        recent_table.stride(0),
        compressed,
        compressed.stride(0),
        raw_key,
        raw_key.stride(0),
        rope_position,
        rope_position.stride(0),
        *(staging if staging is not None else (None,) * 4),
        *(draft if draft is not None else (None,) * 3),
        accepted,
        0,  # seq_offset: Cache.row sets each row's
    )
    gluon_qwen4_exp_qsa_layer_gfx950[(plan.grid,)](
        scratch,
        step,
        layer,
        h,
        _stream_stride(h),
        h_out,
        weights.attn_mixer.kernel_args(),
        weights.qsa_args(),
        cache,
        weights.out_w,
        weights.mlp_mixer.kernel_args(),
        weights.moe.kernel_args(),
        weights.final_mixer.kernel_args() if final else None,
        mixed_out,
        trace_buf,
        plan.key,
        live_plan(plan, catch_up).key,
        advance_step,
        num_warps=NUM_WARPS,
        # The device library of _scheduling.wave_uniform_i32 (moe.py).
        **sched_compile_options(),
    )


__all__ = [
    "GRID",
    "MAX_ROWS",
    "NUM_WARPS",
    "PleStep",
    "delta_plan",
    "experts_format",
    "gluon_qwen4_exp_delta_layer_gfx950",
    "gluon_qwen4_exp_qsa_layer_gfx950",
    "launch_gluon_qwen4_exp_delta_layer_gfx950",
    "launch_gluon_qwen4_exp_qsa_layer_gfx950",
    "live_plan",
    "qsa_plan",
]
