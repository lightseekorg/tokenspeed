# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright 2026 SGLang contributors
#
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


"""The Kimi-K3 MoE layer tail on gfx950: one persistent launch per layer.

For one decoder layer at TP8 and at most eight rows, one launch of 256
workgroups runs the layer's work after the attention core: the attention
output projection and its reduction across ranks, the MLP attention-residual
mix and the post-attention norm, the router, the shared expert, the latent
down projection, norm and up projection around the routed experts, and the
layer output (with ``next_mix``, also the next layer's attention input).

The workgroups hand off through tagged pairs in an uncached scratch and
through every rank's peer buffer, in the layout of ``contract.py``. Every
wait has a deadline, and a missed deadline stops the launch on every rank.

Modified from sgl-amd-mono (https://github.com/sgl-project/sgl-amd-mono, Apache
License 2.0) at b9f40eb562f449ce08c030df200f97c7ea716511:
``kimi_k3/gfx950/layer.py``.
"""

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.common import step_tag
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.down import (
    down_loads,
    load_inter,
    stage_down,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.experts import (
    matrix_layout,
    stage_ug,
    ug_loads,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.extern import (
    compile_options,
    load_peer_base,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.final import (
    stage_final,
    stage_up,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.latent import (
    load_latq,
    stage_lnorm,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.mix import (
    slice_loads,
    stage_gate,
    stage_mix,
    stage_slice_delta,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.next_mix import (
    stage_next_mix,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.oproj import (
    rearm_ar,
    stage_oproj,
    stage_reduce,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.projections import (
    latent,
    router,
    shared_gate_up,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.route import (
    load_route,
    stage_route,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.shared_experts import (
    run_sdown_first,
    run_sdown_rest,
    stage_h,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail.contract import (
    ATTENTION,
    EXPERTS,
    HIDDEN,
    LATENT,
    ROUTED,
    SHARED,
    TAG_OFFSET,
    TP,
    KimiK3MoeTailBuild,
    flags_offset,
    mix_scratch_layout,
    peer_bytes,
    peer_layout,
    scratch_alloc_bytes,
    scratch_layout,
    sent_half_bytes,
    sent_layout,
    sent_offset,
)

MIX_ROW_OFFSET = gl.constexpr(mix_scratch_layout()["mixrow"][0])
MIX_READY_OFFSET = gl.constexpr(mix_scratch_layout()["mixrdy"][0])


def _launch_metadata(grid, kernel, args):
    return {"name": f"{kernel.name} (rows={args['S']}, blocks={args['NB']})"}


@gluon.jit(
    launch_metadata=_launch_metadata,
    do_not_specialize=["rank", "slot", "blk_sm", "blk_sr"],
)
def gluon_kimi_k3_moe_tail_gfx950(
    core,
    w_o,
    prefix,
    prefix_in,
    blocks,
    ares_nw,
    ares_qk,
    in_nw,
    w_gate,
    w_ld,
    w_sgu,
    bias,
    w13,
    w13s,
    w2,
    w2s,
    w_sd,
    ln_w,
    w_up,
    output,
    scratch,
    peers,
    epoch,
    # The strides of ``blocks``, between ``epoch`` and ``rank``: with
    # ``epoch`` next to ``rank`` the compiler merges the first scalar argument
    # loads into one wider load, which changes the compiled code.
    blk_sm,
    blk_sr,
    rank,
    slot,
    # With NEXT_MIX: the next layer's mix weights, its FP32 block partial
    # and the attention input the launch writes; else unused.
    mix_wp,
    mix_nw,
    mix_m,
    mix_s,
    mix_acc,
    mixed,
    S: gl.constexpr,
    NB: gl.constexpr,
    RESET: gl.constexpr,
    EPS: gl.constexpr,
    OUT_EPS: gl.constexpr,
    LN_EPS: gl.constexpr,
    PART: gl.constexpr,
    WGT: gl.constexpr,
    MSQ: gl.constexpr,
    XRDY: gl.constexpr,
    XROW: gl.constexpr,
    XT: gl.constexpr,
    INTER: gl.constexpr,
    ISC: gl.constexpr,
    LMSQ: gl.constexpr,
    LNRDY: gl.constexpr,
    SHRDY: gl.constexpr,
    LN: gl.constexpr,
    SHP: gl.constexpr,
    SENT_LATR: gl.constexpr,
    SENT_FINAL: gl.constexpr,
    SGU: gl.constexpr,
    LPART: gl.constexpr,
    H: gl.constexpr,
    ROUTE: gl.constexpr,
    RLOGP: gl.constexpr,
    LATQ: gl.constexpr,
    PEER_HALF: gl.constexpr,
    SENT_OFFSET: gl.constexpr,
    SENT_HALF: gl.constexpr,
    FLAGS_OFFSET: gl.constexpr,
    NEXT_MIX: gl.constexpr,
    MIX_EPS: gl.constexpr,
):
    bid = gl.program_id(0)
    tag = step_tag(epoch, slot)
    # A peer base is the same in every lane: load_peer_base puts it in scalar
    # registers, without which each buffer access through it becomes a lane loop.
    own = load_peer_base(peers + rank)
    parity = slot % 2
    regions = ()
    for peer in gl.static_range(8):
        base = load_peer_base(peers + peer)
        regions += (base + SENT_OFFSET + parity * SENT_HALF,)
    ar_own = own + SENT_OFFSET + parity * SENT_HALF
    latr_regions, final_regions = (), ()
    for peer in gl.static_range(8):
        base = load_peer_base(peers + peer)
        latr_regions += (base + SENT_LATR + parity * SENT_HALF,)
        final_regions += (base + SENT_FINAL + parity * SENT_HALF,)
    latr_own = own + SENT_LATR + parity * SENT_HALF
    shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    VALS: gl.constexpr = triton.next_power_of_2(gl.cdiv(S * (NB + 1) * 32, 512) * 2048)
    CORE: gl.constexpr = triton.next_power_of_2(S * (1536 + 8))
    EARLY: gl.constexpr = triton.next_power_of_2(S * (7168 + 8))
    UNION_BYTES: gl.constexpr = triton.next_power_of_2(
        max((VALS + 1024) * 4, EARLY * 2)
    )
    union = gl.allocate_shared_memory(gl.uint8, [UNION_BYTES], shared)
    xl = union.reinterpret(gl.bfloat16, [CORE], shared)
    floats = union.reinterpret(gl.float32, [UNION_BYTES // 4], shared)
    vals = floats.slice(0, VALS)
    delta = floats.slice(VALS, 1024)
    red = gl.allocate_shared_memory(gl.float32, [2048], shared)
    route_rows = gl.allocate_shared_memory(
        gl.int32, [triton.next_power_of_2(S * 32)], shared
    )
    route_flag = gl.allocate_shared_memory(gl.int32, [1024], shared)
    route_expert = gl.allocate_shared_memory(gl.int32, [128], shared)
    route_kof = gl.allocate_shared_memory(gl.int32, [1024], shared)
    if bid >= 56:
        stage_oproj(core, w_o, xl, red, regions, rank, bid - 56, S)
        early_x = union.reinterpret(gl.bfloat16, [EARLY], shared)
        peer_parity = gl.load(epoch) % 2
        rlogp_regions = ()
        latq_regions = ()
        for peer in gl.static_range(8):
            # Scalar peer bases, as at the start of the launch.
            base = load_peer_base(peers + peer)
            rlogp_regions += (base + peer_parity * PEER_HALF + RLOGP,)
            latq_regions += (base + peer_parity * PEER_HALF + LATQ,)
        if bid >= 160 and bid < 188:
            router(
                w_gate,
                scratch + XT,
                early_x,
                red,
                rlogp_regions,
                rank,
                (bid - 160) // 4,
                (bid - 160) % 4,
                tag,
                S,
                (epoch, own, peers, rank, FLAGS_OFFSET, 6),
            )
        if bid >= 188 and bid < 244:
            latent(
                w_ld,
                scratch + XT,
                scratch + LPART,
                early_x,
                red,
                floats.slice(0, 1024),
                latq_regions,
                rank,
                (bid - 188) // 4,
                (bid - 188) % 4,
                tag,
                S,
                (epoch, own, peers, rank, FLAGS_OFFSET, 7),
            )
        if bid < 152:
            shared_gate_up(
                w_sgu,
                (scratch + XROW).to(gl.pointer_type(gl.bfloat16)),
                scratch + XRDY,
                scratch + SGU,
                early_x,
                red,
                bid - 56,
                tag,
                S,
                (epoch, own, peers, rank, FLAGS_OFFSET, 8),
            )
    if bid < 56:
        preloads = slice_loads(
            prefix_in, blocks, ares_nw, ares_qk, bid, blk_sm, blk_sr, S, NB
        )
        stage_reduce(ar_own, delta, bid, epoch, own, peers, rank, S, FLAGS_OFFSET)
        stage_slice_delta(
            prefix, scratch + PART, vals, delta, bid, tag, S, NB, RESET, preloads
        )
        if bid < S:
            stage_gate(
                scratch + PART,
                scratch + WGT,
                red,
                bid,
                tag,
                S,
                NB,
                EPS,
                (epoch, own, peers, rank, FLAGS_OFFSET, 4),
            )
        stage_mix(
            scratch + WGT,
            scratch + MSQ,
            scratch + XRDY,
            vals,
            red,
            in_nw,
            (scratch + XROW).to(gl.pointer_type(gl.bfloat16)),
            bid,
            tag,
            S,
            NB,
            OUT_EPS,
            scratch + XT,
            (epoch, own, peers, rank, FLAGS_OFFSET, 5),
        )
        rearm_ar(ar_own, bid, S)
        if bid < S:
            peer_parity = gl.load(epoch) % 2
            stage_route(
                own + peer_parity * PEER_HALF + RLOGP,
                floats,
                bias,
                scratch + ROUTE,
                red,
                bid,
                tag,
                S,
                (epoch, own, peers, rank, FLAGS_OFFSET, 9),
            )
        if bid >= 8 and bid < 8 + S:
            stage_h(
                scratch + SGU,
                scratch + H,
                bid - 8,
                tag,
                (epoch, own, peers, rank, FLAGS_OFFSET, 10),
            )
    peer_parity = gl.load(epoch) % 2
    integers = union.reinterpret(gl.int32, [UNION_BYTES // 4], shared)
    load_latq(
        own + peer_parity * PEER_HALF + LATQ,
        integers,
        tag,
        S,
        (epoch, own, peers, rank, FLAGS_OFFSET, 11),
    )
    nu = load_route(
        scratch + ROUTE,
        route_rows,
        route_flag,
        route_expert,
        route_kof,
        tag,
        S,
        (epoch, own, peers, rank, FLAGS_OFFSET, 12),
    )
    total = nu * 12
    task = 255 - bid
    ops_a = ug_loads(w13, w13s, route_expert, gl.minimum(task, total - 1))
    av = floats
    # Alternate full tasks so a prefetch can stay in its matrix operand registers.
    while task + 256 < total:
        ops_b = ug_loads(w13, w13s, route_expert, task + 256)
        stage_ug(
            integers,
            red,
            av,
            route_kof,
            scratch + INTER,
            scratch + ISC,
            task,
            ops_a,
            tag,
            S,
        )
        gl.amd.hint.sched_barrier()
        next_task = task + 512
        ops_a = ug_loads(w13, w13s, route_expert, gl.minimum(next_task, total - 1))
        stage_ug(
            integers,
            red,
            av,
            route_kof,
            scratch + INTER,
            scratch + ISC,
            task + 256,
            ops_b,
            tag,
            S,
        )
        gl.amd.hint.sched_barrier()
        task = next_task
    if task < total:
        stage_ug(
            integers,
            red,
            av,
            route_kof,
            scratch + INTER,
            scratch + ISC,
            task,
            ops_a,
            tag,
            S,
        )
    if bid < 224:
        ML: gl.constexpr = matrix_layout()
        BW: gl.constexpr = gl.DotOperandLayout(1, ML, 4)
        waves = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, BW)))
        down_ops = down_loads(w2, w2s, route_expert, bid, nu, gl.minimum(waves, nu - 1))
        gl.static_assert(ISC - INTER == S * 16 * 48 * 8)
        load_inter(
            scratch + INTER,
            integers,
            tag,
            S,
            (epoch, own, peers, rank, FLAGS_OFFSET, 16),
        )
        stage_down(
            w2,
            w2s,
            integers,
            red,
            route_rows,
            route_expert,
            route_kof,
            bid,
            nu,
            down_ops,
            latr_regions,
            rank,
            S,
        )
    shared_x = union.reinterpret(gl.bfloat16, [EARLY], shared)
    run_sdown_first(
        w_sd,
        scratch + H,
        scratch + SHP,
        scratch + SHRDY,
        shared_x,
        red,
        final_regions,
        rank,
        bid,
        tag,
        S,
        (epoch, own, peers, rank, FLAGS_OFFSET, 17),
    )
    run_sdown_rest(
        w_sd,
        scratch + H,
        scratch + SHP,
        scratch + SHRDY,
        shared_x,
        red,
        final_regions,
        rank,
        bid,
        tag,
        S,
        (epoch, own, peers, rank, FLAGS_OFFSET, 17),
    )
    if bid >= 56 and bid < 84:
        stage_lnorm(
            latr_own,
            ln_w,
            (scratch + LN).to(gl.pointer_type(gl.bfloat16)),
            scratch + LMSQ,
            scratch + LNRDY,
            red,
            bid - 56,
            tag,
            epoch,
            own,
            peers,
            rank,
            S,
            LN_EPS,
            FLAGS_OFFSET,
        )
    if bid < 56:
        stage_up(
            w_up,
            (scratch + LN).to(gl.pointer_type(gl.bfloat16)),
            scratch + SHP,
            scratch + LNRDY,
            scratch + SHRDY,
            shared_x,
            red,
            final_regions,
            rank,
            bid,
            tag,
            S,
            (epoch, own, peers, rank, FLAGS_OFFSET, 18),
        )
    if bid >= 200:
        stage_final(
            own + SENT_FINAL + parity * SENT_HALF,
            prefix,
            output,
            bid - 200,
            epoch,
            own,
            peers,
            rank,
            S,
            FLAGS_OFFSET,
            scratch + MIX_ROW_OFFSET,
            scratch + MIX_READY_OFFSET,
            tag,
            NEXT_MIX,
        )
    if NEXT_MIX:
        if bid >= 192 and bid < 192 + S:
            stage_next_mix(
                scratch + MIX_ROW_OFFSET,
                scratch + MIX_READY_OFFSET,
                mix_wp,
                mix_nw,
                mix_m,
                mix_s,
                mix_acc,
                mixed,
                bid - 192,
                tag,
                MIX_EPS,
                (epoch, own, peers, rank, FLAGS_OFFSET, 19),
            )


def compile_signature(key: KimiK3MoeTailBuild):
    """Return the kernel's argument types and compile-time constants for
    ``key``."""
    signature = dict.fromkeys(
        (
            "core",
            "w_o",
            "prefix",
            "prefix_in",
            "blocks",
            "ares_nw",
            "ares_qk",
            "in_nw",
            "w_gate",
            "w_ld",
            "w_sgu",
        ),
        "*bf16",
    )
    signature["bias"] = "*fp32"
    signature["w13"] = "*u8"
    signature["w13s"] = "*u8"
    signature["w2"] = "*u8"
    signature["w2s"] = "*u8"
    signature["w_sd"] = "*bf16"
    signature["ln_w"] = "*bf16"
    signature["w_up"] = "*bf16"
    signature["output"] = "*bf16"
    signature.update(
        scratch="*u8",
        peers="*i64",
        epoch="*i32",
        rank="i32",
        slot="i32",
        blk_sm="i32",
        blk_sr="i32",
        mix_wp="*bf16",
        mix_nw="*bf16",
        mix_m="*fp32",
        mix_s="*fp32",
        mix_acc="*fp32",
        mixed="*bf16",
    )
    layout = scratch_layout(key)
    play = peer_layout(key.tokens)
    constants = {
        "S": key.tokens,
        "NB": key.nblocks,
        "RESET": key.reset,
        "EPS": key.eps,
        "OUT_EPS": key.out_eps,
        "LN_EPS": key.ln_eps,
        **{
            name.upper(): layout[name][0]
            for name in (
                "part",
                "wgt",
                "msq",
                "xrdy",
                "xrow",
                "sgu",
                "lpart",
                "h",
                "route",
                "inter",
                "isc",
                "lmsq",
                "lnrdy",
                "shrdy",
                "ln",
                "shp",
            )
        },
        "XT": TAG_OFFSET,
        "RLOGP": play["rlogp"][0],
        "LATQ": play["latq"][0],
        "PEER_HALF": peer_bytes(key.tokens) // 2,
        "SENT_OFFSET": sent_offset() + sent_layout()["ars"][0],
        "SENT_LATR": sent_offset() + sent_layout()["latrs"][0],
        "SENT_FINAL": sent_offset() + sent_layout()["finals"][0],
        "SENT_HALF": sent_half_bytes(),
        "FLAGS_OFFSET": flags_offset(),
        "NEXT_MIX": key.next_mix,
        "MIX_EPS": 0.0 if key.mix_eps is None else key.mix_eps,
    }
    return signature, constants


def _check(tensor, dtype, shape, name):
    if (
        tensor.dtype != dtype
        or tuple(tensor.shape) != shape
        or not tensor.is_contiguous()
    ):
        raise ValueError(
            f"The MoE tail reads {name} as contiguous {dtype} {shape}, not "
            f"{tensor.dtype} {tuple(tensor.shape)} with strides {tensor.stride()}"
        )


def _check_blocks(blocks, rows, nblocks):
    if (
        blocks.dtype != torch.bfloat16
        or blocks.dim() != 3
        or blocks.shape[0] != rows
        or blocks.shape[1] < nblocks
        or blocks.shape[2] != HIDDEN
        or blocks.stride(2) != 1
        or blocks.stride(0) % 4
        or blocks.stride(1) % 4
    ):
        raise ValueError(
            f"The MoE tail reads the residual blocks as BF16 [{rows}, at least "
            f"{nblocks}, {HIDDEN}] with a unit inner stride and the other strides "
            f"multiples of 4, not {blocks.dtype} {tuple(blocks.shape)} with "
            f"strides {blocks.stride()}"
        )
    # The kernel loads four values of a block row at once, at a 32-bit byte
    # offset from the tensor's start (raw_bf4): every read byte must lie
    # below 2**31.
    last = (rows - 1) * blocks.stride(0) + (nblocks - 1) * blocks.stride(1) + HIDDEN
    if last * blocks.element_size() > 2**31:
        raise ValueError(
            "The MoE tail reads the residual blocks at 32-bit byte offsets, which "
            f"strides {blocks.stride()} exceed"
        )


def launch_gluon_kimi_k3_moe_tail_gfx950(
    key: KimiK3MoeTailBuild,
    *,
    core,
    w_o,
    prefix,
    prefix_in,
    blocks,
    ares_nw,
    ares_qk,
    in_nw,
    w_gate,
    w_ld,
    w_sgu,
    bias,
    w13,
    w13s,
    w2,
    w2s,
    w_sd,
    ln_w,
    w_up,
    output,
    scratch,
    peers,
    epoch,
    rank,
    slot,
    mix_wp,
    mix_nw,
    mix_m,
    mix_s,
    mix_acc,
    mixed,
    compile_only: bool,
):
    """Launch one layer's MoE tail on this rank, for the build ``key``.

    Every rank of the TP group must launch the same layers in the same order;
    a launch waits on its peers.

    Args:
        key: the compile-time build settings.
        core: the attention output, BF16 ``[rows, 1536]``.
        w_o: this rank's attention output projection, BF16 ``[7168, 1536]``.
        prefix: the running residual prefix the launch writes, BF16
            ``[rows, 7168]``.
        prefix_in: the incoming stream, BF16 ``[rows, 7168]``; a ``reset``
            build reads ``prefix`` itself here.
        blocks: the completed attention-residual block snapshots, BF16; row
            ``t`` of block ``j`` starts at ``t * stride(0) + j * stride(1)``.
        ares_nw, ares_qk: the MLP attention-residual norm and projection
            weights, BF16 ``[7168]``.
        in_nw: the post-attention norm weight, BF16 ``[7168]``.
        w_gate: the router weight, BF16 ``[896, 7168]``.
        w_ld: the latent down projection, BF16 ``[3584, 7168]``.
        w_sgu: this rank's shared-expert gate/up projection, BF16
            ``[1536, 7168]``.
        bias: the router's score bias, FP32 ``[896]``.
        w13, w13s: the routed experts' gate/up MXFP4 weights and E8M0
            scales, as uint8, in the N16 layout.
        w2, w2s: the routed experts' down MXFP4 weights and scales, likewise.
        w_sd: this rank's shared-expert down projection, BF16 ``[7168, 768]``.
        ln_w: the latent norm weight, BF16 ``[3584]``.
        w_up: this rank's rows of the latent up projection, BF16
            ``[896, 3584]``.
        output: the layer output the launch writes, BF16 ``[rows, 7168]``.
        scratch: this rank's uncached scratch, at least
            :func:`scratch_alloc_bytes` bytes; no clearing is needed between
            launches.
        peers: the eight ranks' peer buffer addresses, int64 ``[8]``.
        epoch: the step words (the same buffer the registration calls
            ``steps``), int32 ``[8]``: word 0 counts steps, words 1-2 hold
            each wait's time limit in 100 MHz clock ticks (low word first),
            and words 3-5 are error words (``_stages/sentinel.py``).
        rank: this rank, in ``[0, 8)``.
        slot: the layer's handoff slot, in ``[128, 256)``, distinct per layer.
        mix_wp, mix_nw, mix_m, mix_s, mix_acc, mixed: given exactly when
            ``key.next_mix``: the next layer's mix score weight and input
            norm weight (BF16 ``[7168]``), its FP32 block partial (maximum and
            sum ``[rows]``, accumulator ``[rows, 7168]``), and the next
            layer's attention input the launch writes (BF16 ``[rows, 7168]``).
        compile_only: only compile the kernel for these operands, without
            launching it. A launch waits on its peers, so the first launch of
            a build must not wait for one rank's compile; compile every
            rank's builds first.
    """
    rows, bf16, fp32, u8 = key.tokens, torch.bfloat16, torch.float32, torch.uint8
    if scratch.dtype != u8 or scratch.dim() != 1:
        raise ValueError(
            "The MoE tail's scratch must be a one-dimensional uint8 tensor"
        )
    if scratch.numel() < scratch_alloc_bytes():
        raise ValueError(
            f"The MoE tail's scratch holds {scratch.numel()} bytes, not at least "
            f"{scratch_alloc_bytes()}"
        )
    _check(epoch, torch.int32, (8,), "the step words")
    _check(peers, torch.int64, (TP,), "the ranks' peer buffer addresses")
    # A slot is one byte of every handoff's tag (step * 256 + slot + 1); the
    # range leaves slots below 128 to other launches sharing the step words.
    if not 128 <= slot < 256 or not 0 <= rank < TP:
        raise ValueError(
            f"The MoE tail requires a layer slot in [128, 256) and a rank in "
            f"[0, {TP}), not slot {slot} and rank {rank}"
        )
    for tensor, dtype, shape, name in (
        (core, bf16, (rows, ATTENTION), "the attention output"),
        (prefix, bf16, (rows, HIDDEN), "the residual prefix"),
        (prefix_in, bf16, (rows, HIDDEN), "the incoming stream"),
        (output, bf16, (rows, HIDDEN), "the output"),
        (w_o, bf16, (HIDDEN, ATTENTION), "the attention output projection"),
        (ares_nw, bf16, (HIDDEN,), "the MLP attention-residual norm weight"),
        (ares_qk, bf16, (HIDDEN,), "the MLP attention-residual projection"),
        (in_nw, bf16, (HIDDEN,), "the post-attention norm weight"),
        (w_gate, bf16, (EXPERTS, HIDDEN), "the router weight"),
        (bias, fp32, (EXPERTS,), "the router bias"),
        (w_ld, bf16, (LATENT, HIDDEN), "the latent down projection"),
        (ln_w, bf16, (LATENT,), "the latent norm weight"),
        (w_up, bf16, (HIDDEN // TP, LATENT), "this rank's latent up projection"),
        (w_sgu, bf16, (2 * SHARED, HIDDEN), "the shared gate/up projection"),
        (w_sd, bf16, (HIDDEN, SHARED), "the shared down projection"),
        (w13, u8, (EXPERTS, 2 * ROUTED, LATENT // 2), "the routed gate/up bank"),
        (w13s, u8, (EXPERTS * 2 * ROUTED, LATENT // 32), "its scales"),
        (w2, u8, (EXPERTS, LATENT, ROUTED // 2), "the routed down bank"),
        (w2s, u8, (EXPERTS * LATENT, 16), "its scales"),
    ):
        _check(tensor, dtype, shape, name)
    _check_blocks(blocks, rows, key.nblocks)
    mix = (mix_wp, mix_nw, mix_m, mix_s, mix_acc, mixed)
    if any((t is not None) != key.next_mix for t in mix):
        raise ValueError(
            "the next layer's mix operands are given exactly when the build "
            "writes its mix"
        )
    if key.next_mix:
        _check(mix_wp, bf16, (HIDDEN,), "the next mix's score weight")
        _check(mix_nw, bf16, (HIDDEN,), "the next input norm weight")
        _check(mix_m, fp32, (rows,), "the next mix's partial maximum")
        _check(mix_s, fp32, (rows,), "the next mix's partial sum")
        _check(mix_acc, fp32, (rows, HIDDEN), "the next mix's partial accumulator")
        _check(mixed, bf16, (rows, HIDDEN), "the next attention input")
    else:
        # Unused pointer arguments: typed stand-ins, never read.
        mix_wp = mix_nw = mixed = output
        mix_m = mix_s = mix_acc = bias
    _, constants = compile_signature(key)
    args = (
        core,
        w_o,
        prefix,
        prefix_in,
        blocks,
        ares_nw,
        ares_qk,
        in_nw,
        w_gate,
        w_ld,
        w_sgu,
        bias,
        w13,
        w13s,
        w2,
        w2s,
        w_sd,
        ln_w,
        w_up,
        output,
        scratch,
        peers,
        epoch,
        blocks.stride(0),
        blocks.stride(1),
        rank,
        slot,
        mix_wp,
        mix_nw,
        mix_m,
        mix_s,
        mix_acc,
        mixed,
    )
    options = dict(
        num_warps=8, num_stages=1, enable_fp_fusion=False, **compile_options()
    )
    if compile_only:
        gluon_kimi_k3_moe_tail_gfx950.warmup(*args, grid=(256,), **constants, **options)
        return
    gluon_kimi_k3_moe_tail_gfx950[(256,)](*args, **constants, **options)
