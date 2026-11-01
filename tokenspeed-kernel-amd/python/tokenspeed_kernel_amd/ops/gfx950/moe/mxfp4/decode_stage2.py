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


"""Stage-2 direct MXFP4 MFMA decode kernel for the gfx950 A4W4 MoE package.

Consumes the packed E2M1 intermediates written by stage 1, applies
gdot128-shuffled MXFP4 W2, and folds the routed-weight top-k combine into
the MFMA epilogue.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.decode_common import (
    _direct_mxfp4_load_tile,
    _direct_mxfp4_mfma,
    _direct_mxfp4_mfma_layouts,
)


# ---------------------------------------------------------------------------
# Experimental stage 2: MXFP4 inter x MXFP4 W2 direct MFMA + fused topk combine.
#
# This is the direct top-k variant of the reference combine GEMM.  It assumes the
# intermediate rows are already in (token, topk-slot) order, so it does not
# consume ragged metadata / scatter indices.  Keep it out of default dispatch
# until the matching direct stage1 path writes this row order with MXFP4 output.
# ---------------------------------------------------------------------------
@gluon.jit
def _stage2_mxfp4_direct_mfma_gluon(
    inter_ptr,  # (M*TOPK, I/2) uint8 e2m1 packed, slot order
    inter_scale_ptr,  # (Kscale_pad*32, ceil((M*TOPK)/32)) uint8 CDNA4 swizzled
    w2_ptr,  # (E, I/2 padded, D padded) uint8 gdot128-shuffled
    w2s_ptr,  # (E, Kscale*32, ceil(D/32)) uint8 CDNA4-swizzled
    topk_ids_ptr,  # (M, TOPK) int32
    topk_weights_ptr,  # (M, TOPK) float32
    out_ptr,  # (M, D) bf16
    expanded_idx_ptr,  # (M * TOPK) int32 when finalization is deferred
    M,
    D,
    expert_start,
    num_local_experts,
    N_PHYS: gl.constexpr,
    I_DIM,
    stride_im,
    stride_ik,
    stride_xslin,
    stride_xsnb,
    stride_we,
    stride_se,
    stride_slin,
    stride_snb,
    stride_om,
    stride_on,
    stride_tit,
    stride_tis,
    stride_twt,
    stride_tws,
    I_PACKED: gl.constexpr,
    TOPK: gl.constexpr,
    BLOCK_K: gl.constexpr,
    BLOCK_N: gl.constexpr,
    M_DUP: gl.constexpr,
    PIPELINE_K: gl.constexpr,
    DEFER_FINALIZE: gl.constexpr,
    SLOT_PARALLEL: gl.constexpr,
):
    BLOCK_K_PACKED: gl.constexpr = BLOCK_K // 2
    BLOCK_K_SCALE: gl.constexpr = BLOCK_K // 32
    gl.static_assert(
        BLOCK_K == 128 or BLOCK_K == 256,
        "direct MXFP4 stage2 BLOCK_K must contain one or two scaled-MFMA K tiles",
    )
    gl.static_assert(
        128 % BLOCK_N == 0,
        "direct MXFP4 stage2 BLOCK_N must divide the gdot128 128-wide W tile",
    )

    pid = gl.program_id(axis=0)
    num_n = gl.cdiv(D, BLOCK_N)
    route = pid // num_n
    if DEFER_FINALIZE:
        token = route // TOPK
        route_slot = route - token * TOPK
        ROUTES_PER_PROGRAM: gl.constexpr = 1
    else:
        token = route
        route_slot = route - route
        ROUTES_PER_PROGRAM: gl.constexpr = TOPK
    pid_n = pid % num_n

    layouts: gl.constexpr = _direct_mxfp4_mfma_layouts(M_DUP, BLOCK_N, BLOCK_K_SCALE, 1)
    mfma_layout: gl.constexpr = layouts[0]
    dot_a_layout: gl.constexpr = layouts[1]
    dot_b_layout: gl.constexpr = layouts[2]
    a_scale_layout: gl.constexpr = layouts[3]
    b_scale_layout: gl.constexpr = layouts[4]

    am = gl.arange(0, M_DUP, layout=gl.SliceLayout(1, dot_a_layout))[:, None]
    ak = gl.arange(0, BLOCK_K_PACKED, layout=gl.SliceLayout(0, dot_a_layout))[None, :]
    bk = gl.arange(0, BLOCK_K_PACKED, layout=gl.SliceLayout(1, dot_b_layout))[:, None]
    bn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, dot_b_layout))[None, :]
    asm = gl.arange(0, M_DUP, layout=gl.SliceLayout(1, a_scale_layout))[:, None]
    ask = gl.arange(0, BLOCK_K_SCALE, layout=gl.SliceLayout(0, a_scale_layout))[None, :]
    bsn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, b_scale_layout))[:, None]
    bsk = gl.arange(0, BLOCK_K_SCALE, layout=gl.SliceLayout(0, b_scale_layout))[None, :]

    n_cols = pid_n * BLOCK_N + bn
    n_cols_s = pid_n * BLOCK_N + bsn
    TOTAL_KT: gl.constexpr = gl.cdiv(I_PACKED, BLOCK_K_PACKED)
    acc_total = gl.zeros((M_DUP, BLOCK_N), dtype=gl.float32, layout=mfma_layout)

    if SLOT_PARALLEL and not DEFER_FINALIZE and token < M:
        # Short-K decode is bound by the dependent load -> MFMA chain, not by
        # bandwidth.  Issue every slot's K tile together into per-slot
        # accumulators; the epilogue keeps the sequential slot order below.
        owned_slots = ()
        gates = ()
        w_offs = ()
        s_offs = ()
        accs = ()
        for slot in gl.static_range(TOPK):
            expert = (
                gl.load(topk_ids_ptr + token * stride_tit + slot * stride_tis)
                - expert_start
            )
            owned = (expert >= 0) & (expert < num_local_experts)
            expert = gl.where(owned, expert, 0)
            owned_slots = owned_slots + (owned,)
            gates = gates + (
                gl.load(topk_weights_ptr + token * stride_twt + slot * stride_tws).to(
                    gl.float32
                ),
            )
            w_offs = w_offs + (expert.to(gl.int64) * stride_we,)
            s_offs = s_offs + (expert.to(gl.int64) * stride_se,)
            accs = accs + (
                gl.zeros((M_DUP, BLOCK_N), dtype=gl.float32, layout=mfma_layout),
            )
        for kt in gl.static_range(TOTAL_KT):
            tiles = ()
            for slot in gl.static_range(TOPK):
                row = token * TOPK + slot
                tiles = tiles + (
                    _direct_mxfp4_load_tile(
                        kt,
                        ak,
                        bk,
                        ask,
                        bsk,
                        am,
                        asm,
                        inter_ptr,
                        inter_scale_ptr,
                        w2_ptr,
                        w2s_ptr,
                        row.to(gl.int64) * stride_im,
                        w_offs[slot],
                        s_offs[slot],
                        n_cols,
                        n_cols_s,
                        row,
                        stride_ik,
                        stride_xslin,
                        stride_xsnb,
                        stride_slin,
                        stride_snb,
                        N_PHYS,
                        I_DIM,
                        D,
                        I_PACKED,
                        BLOCK_K_PACKED,
                        BLOCK_K_SCALE,
                    ),
                )
            updated = ()
            for slot in gl.static_range(TOPK):
                tile = tiles[slot]
                updated = updated + (
                    _direct_mxfp4_mfma(accs[slot], tile[0], tile[1], tile[2], tile[3]),
                )
            accs = updated
        for slot in gl.static_range(TOPK):
            partial = gl.where(owned_slots[slot], accs[slot], 0.0).to(
                out_ptr.dtype.element_ty
            )
            acc_total += (partial * gates[slot].to(partial.dtype)).to(gl.float32)
    elif token < M:
        for slot_offset in gl.static_range(0, ROUTES_PER_PROGRAM):
            slot = route_slot + slot_offset
            global_expert = gl.load(
                topk_ids_ptr + token * stride_tit + slot * stride_tis
            )
            expert = global_expert - expert_start
            owned = (expert >= 0) & (expert < num_local_experts)
            expert = gl.where(owned, expert, 0)
            if not DEFER_FINALIZE:
                gate = gl.load(
                    topk_weights_ptr + token * stride_twt + slot * stride_tws
                ).to(gl.float32)
            row = token * TOPK + slot
            if DEFER_FINALIZE:
                gl.store(
                    expanded_idx_ptr + row,
                    gl.where(owned, row, -1),
                    mask=(token < M) & (pid_n == 0),
                )
            x_row_off = row.to(gl.int64) * stride_im
            w_expert_off = expert.to(gl.int64) * stride_we
            s_expert_off = expert.to(gl.int64) * stride_se
            acc = gl.zeros((M_DUP, BLOCK_N), dtype=gl.float32, layout=mfma_layout)

            if owned and PIPELINE_K and TOTAL_KT > 1:
                # Hold one K tile in VGPRs while issuing the next tile's four
                # global loads.  Kimi stage2 has only four tiles, so this small
                # lookahead hides most VMEM waits without an LDS round trip.
                a, b, a_scale, b_scale = _direct_mxfp4_load_tile(
                    0,
                    ak,
                    bk,
                    ask,
                    bsk,
                    am,
                    asm,
                    inter_ptr,
                    inter_scale_ptr,
                    w2_ptr,
                    w2s_ptr,
                    x_row_off,
                    w_expert_off,
                    s_expert_off,
                    n_cols,
                    n_cols_s,
                    row,
                    stride_ik,
                    stride_xslin,
                    stride_xsnb,
                    stride_slin,
                    stride_snb,
                    N_PHYS,
                    I_DIM,
                    D,
                    I_PACKED,
                    BLOCK_K_PACKED,
                    BLOCK_K_SCALE,
                )
                for kt in range(0, TOTAL_KT - 1):
                    next_a, next_b, next_a_scale, next_b_scale = (
                        _direct_mxfp4_load_tile(
                            kt + 1,
                            ak,
                            bk,
                            ask,
                            bsk,
                            am,
                            asm,
                            inter_ptr,
                            inter_scale_ptr,
                            w2_ptr,
                            w2s_ptr,
                            x_row_off,
                            w_expert_off,
                            s_expert_off,
                            n_cols,
                            n_cols_s,
                            row,
                            stride_ik,
                            stride_xslin,
                            stride_xsnb,
                            stride_slin,
                            stride_snb,
                            N_PHYS,
                            I_DIM,
                            D,
                            I_PACKED,
                            BLOCK_K_PACKED,
                            BLOCK_K_SCALE,
                        )
                    )
                    acc = _direct_mxfp4_mfma(acc, a, b, a_scale, b_scale)
                    a, b, a_scale, b_scale = (
                        next_a,
                        next_b,
                        next_a_scale,
                        next_b_scale,
                    )
                acc = _direct_mxfp4_mfma(acc, a, b, a_scale, b_scale)
            elif owned:
                for kt in range(0, TOTAL_KT):
                    a, b, a_scale, b_scale = _direct_mxfp4_load_tile(
                        kt,
                        ak,
                        bk,
                        ask,
                        bsk,
                        am,
                        asm,
                        inter_ptr,
                        inter_scale_ptr,
                        w2_ptr,
                        w2s_ptr,
                        x_row_off,
                        w_expert_off,
                        s_expert_off,
                        n_cols,
                        n_cols_s,
                        row,
                        stride_ik,
                        stride_xslin,
                        stride_xsnb,
                        stride_slin,
                        stride_snb,
                        N_PHYS,
                        I_DIM,
                        D,
                        I_PACKED,
                        BLOCK_K_PACKED,
                        BLOCK_K_SCALE,
                    )
                    acc = _direct_mxfp4_mfma(acc, a, b, a_scale, b_scale)
            # Match the reference combine epilogue ordering.  Its GEMM kernel first
            # rounds each expert partial to the output dtype, multiplies by a
            # routed weight in that same dtype, stores BF16 partial rows, and
            # only then reduces top-k.  Keeping ``gate * acc`` in FP32 until
            # the final store changes thousands of Kimi decode elements by a
            # BF16 ULP even when routing, quantization, and stage 1 are exact.
            partial = acc.to(out_ptr.dtype.element_ty)
            if DEFER_FINALIZE:
                sm = gl.arange(
                    0,
                    M_DUP,
                    layout=gl.SliceLayout(1, mfma_layout),
                )[:, None]
                sn = gl.arange(
                    0,
                    BLOCK_N,
                    layout=gl.SliceLayout(0, mfma_layout),
                )[None, :]
                col = pid_n * BLOCK_N + sn
                gl.store(
                    out_ptr
                    + row.to(gl.int64) * stride_om
                    + col.to(gl.int64) * stride_on
                    + sm.to(gl.int64) * 0,
                    partial,
                    mask=(token < M) & (sm == 0) & (col < D),
                )
            else:
                routed_weight = gate.to(partial.dtype)
                acc_total += (partial * routed_weight).to(gl.float32)

    if not DEFER_FINALIZE:
        sm = gl.arange(0, M_DUP, layout=gl.SliceLayout(1, mfma_layout))[:, None]
        sn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))[None, :]
        col = pid_n * BLOCK_N + sn
        gl.store(
            out_ptr
            + token.to(gl.int64) * stride_om
            + col.to(gl.int64) * stride_on
            + sm.to(gl.int64) * 0,
            acc_total.to(out_ptr.dtype.element_ty),
            mask=(token < M) & (sm == 0) & (col < D),
        )


def invoke_stage2_mxfp4_mfma_decode_gluon(
    inter_states_mxfp4,  # (num_tokens*topk, I_r//2) uint8 e2m1, slot order
    inter_scale,  # CDNA4-swizzled e8m0 scales from _quantize_mxfp4_activation
    w2,  # (E, I_r//2 padded, D padded) uint8 gdot128-shuffled
    w2_scale,  # (E, Kscale*32, ceil(D/32)) uint8 CDNA4-swizzled
    topk_ids,
    topk_weights,
    out,  # (num_tokens, D) bf16
    topk: int,
    expert_start: int,
    do_finalize: bool,
    expanded_idx: torch.Tensor | None,
    BLOCK_N: int = 16,
    BLOCK_K: int = 128,
    M_DUP: int = 4,
    PIPELINE_K: bool = True,
    SLOT_PARALLEL: bool = False,
):
    assert inter_states_mxfp4.dtype == torch.uint8
    assert inter_scale.dtype == torch.uint8
    assert w2.dtype == torch.uint8 and w2_scale.dtype == torch.uint8
    assert out.dtype == torch.bfloat16
    _, Ih_phys, N_phys = w2.shape
    Ih = int(getattr(w2, "original_k_pk", Ih_phys))
    I_r = Ih * 2
    num_tokens = int(topk_ids.shape[0])
    D = int(getattr(w2, "original_n", N_phys))
    assert inter_states_mxfp4.shape == (num_tokens * topk, Ih)
    assert D <= N_phys
    expected_out_shape = (num_tokens, D) if do_finalize else (num_tokens * topk, D)
    assert out.shape == expected_out_shape
    if do_finalize:
        assert expanded_idx is None
    else:
        assert expanded_idx is not None
        assert expanded_idx.shape == (num_tokens * topk,)
        assert expanded_idx.dtype == torch.int32
    assert bool(getattr(w2, "is_shuffled_for_gluon_dot", False))
    assert int(getattr(w2, "gluon_dot_block_k_pk", 0)) == 128
    assert int(getattr(w2, "gluon_dot_block_n", 0)) == 128
    assert w2_scale.stride(-2) == 1
    assert inter_scale.stride(-2) == 1
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = topk_weights.to(torch.float32)
    expanded_idx_arg = topk_ids if expanded_idx is None else expanded_idx
    routes_per_token = topk if not do_finalize else 1
    grid = (num_tokens * routes_per_token * triton.cdiv(D, BLOCK_N),)
    _stage2_mxfp4_direct_mfma_gluon[grid](
        inter_states_mxfp4,
        inter_scale,
        w2,
        w2_scale,
        topk_ids,
        topk_weights,
        out,
        expanded_idx_arg,
        num_tokens,
        D,
        expert_start,
        int(w2.shape[0]),
        N_phys,
        I_r,
        inter_states_mxfp4.stride(0),
        inter_states_mxfp4.stride(1),
        inter_scale.stride(0),
        inter_scale.stride(1),
        w2.stride(0),
        w2_scale.stride(0),
        w2_scale.stride(1),
        w2_scale.stride(2),
        out.stride(0),
        out.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        I_PACKED=Ih,
        TOPK=topk,
        BLOCK_K=BLOCK_K,
        BLOCK_N=BLOCK_N,
        M_DUP=M_DUP,
        PIPELINE_K=PIPELINE_K,
        DEFER_FINALIZE=not do_finalize,
        SLOT_PARALLEL=SLOT_PARALLEL,
        num_warps=1,
    )
    return out


__all__ = [
    "invoke_stage2_mxfp4_mfma_decode_gluon",
]


def _route_tile_launch_metadata(grid, kernel, args):
    return {"name": f"{kernel.name} (routes={args['out'].shape[0]})"}


@gluon.jit(launch_metadata=_route_tile_launch_metadata)
def gluon_mxfp4_moe_stage2_route_tiles_gfx950(
    x,
    xscale,
    w,
    wscale,
    sorted_routes,
    descriptors,
    group_count,
    out,
    INTERMEDIATE: gl.constexpr,
    OUTPUT: gl.constexpr,
    XS0: gl.constexpr,
    XS1: gl.constexpr,
    WE: gl.constexpr,
    SE: gl.constexpr,
    SS0: gl.constexpr,
    SS1: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    BLOCK_M: gl.constexpr,
):
    """Read expert route tiles and return separate token/slot outputs."""
    gl.static_assert(
        INTERMEDIATE % BLOCK_K == 0, "route tiles require complete K tiles"
    )
    gl.static_assert(OUTPUT % BLOCK_N == 0, "route tiles require complete output tiles")
    K_PACKED: gl.constexpr = INTERMEDIATE // 2
    KP: gl.constexpr = BLOCK_K // 2
    KS: gl.constexpr = BLOCK_K // 32
    layouts: gl.constexpr = _direct_mxfp4_mfma_layouts(BLOCK_M, BLOCK_N, KS, 1)
    mfma: gl.constexpr = layouts[0]
    la: gl.constexpr = layouts[1]
    lb: gl.constexpr = layouts[2]
    lsa: gl.constexpr = layouts[3]
    lsb: gl.constexpr = layouts[4]
    am = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, la))[:, None]
    ak = gl.arange(0, KP, layout=gl.SliceLayout(0, la))[None, :]
    bk = gl.arange(0, KP, layout=gl.SliceLayout(1, lb))[:, None]
    bn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, lb))[None, :]
    asm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, lsa))[:, None]
    ask = gl.arange(0, KS, layout=gl.SliceLayout(0, lsa))[None, :]
    bsn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, lsb))[:, None]
    bsk = gl.arange(0, KS, layout=gl.SliceLayout(0, lsb))[None, :]
    groups = gl.load(group_count)
    for job in range(
        gl.program_id(0), groups * (OUTPUT // BLOCK_N), gl.num_programs(0)
    ):
        group = job // (OUTPUT // BLOCK_N)
        n = job % (OUTPUT // BLOCK_N)
        start = gl.load(descriptors + group * 3)
        count = gl.load(descriptors + group * 3 + 1)
        expert = gl.load(descriptors + group * 3 + 2)
        first = gl.load(sorted_routes + start)
        routes = gl.load(sorted_routes + start + am, am < count, other=first)
        scale_routes = gl.load(sorted_routes + start + asm, asm < count, other=first)
        xoff = routes.to(gl.int64) * K_PACKED
        woff = expert.to(gl.int64) * WE
        wsoff = expert.to(gl.int64) * SE
        ncols = n * BLOCK_N + bn
        nscols = n * BLOCK_N + bsn
        acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma)
        for kt in range(K_PACKED // KP):
            a, b, sa, sb = _direct_mxfp4_load_tile(
                kt,
                ak,
                bk,
                ask,
                bsk,
                am,
                asm,
                x,
                xscale,
                w,
                wscale,
                xoff,
                woff,
                wsoff,
                ncols,
                nscols,
                scale_routes,
                1,
                XS0,
                XS1,
                SS0,
                SS1,
                OUTPUT,
                INTERMEDIATE,
                OUTPUT,
                K_PACKED,
                KP,
                KS,
            )
            acc = _direct_mxfp4_mfma(acc, a, b, sa, sb)
        sm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mfma))[:, None]
        sn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma))[None, :]
        destinations = gl.load(sorted_routes + start + sm, sm < count, other=first)
        gl.store(
            out + destinations.to(gl.int64) * OUTPUT + n * BLOCK_N + sn,
            acc.to(gl.bfloat16),
            sm < count,
        )
