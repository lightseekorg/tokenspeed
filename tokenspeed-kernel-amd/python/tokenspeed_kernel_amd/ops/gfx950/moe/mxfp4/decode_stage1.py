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


"""Stage-1 direct MXFP4 MFMA decode kernel for the gfx950 A4W4 MoE package.

Consumes packed E2M1 activations, E8M0 activation scales, and
gdot128-shuffled MXFP4 W13, then fuses SwiGLU into the MFMA epilogue and
writes BF16 intermediates in (token, topk-slot) order.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.decode_common import (
    _direct_mxfp4_load_tile,
    _direct_mxfp4_mfma,
    _direct_mxfp4_mfma_layouts,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.quantize_gluon import (
    _mxfp4_quantize_tile,
    _mxfp4_store_cdna4_scale,
)


@gluon.constexpr_function
def _direct_swiglu_split_layout(
    block_m: int, block_n_full: int, num_warps: int
) -> gl.constexpr:
    del block_m, block_n_full
    threads_per_warp = 64
    return gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[4, threads_per_warp // 4],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )


@gluon.jit
def _direct_swiglu_reduce(
    acc,
    alpha: gl.constexpr,
    limit: gl.constexpr,
    beta: gl.constexpr,
    OUT_BLOCK_N: gl.constexpr,
):
    BLOCK_M: gl.constexpr = acc.shape[0]
    BLOCK_N_FULL: gl.constexpr = acc.shape[1]
    split_layout: gl.constexpr = _direct_swiglu_split_layout(
        BLOCK_M, BLOCK_N_FULL, gl.num_warps()
    )
    acc = gl.convert_layout(acc, split_layout)
    reshaped = acc.reshape((BLOCK_M, OUT_BLOCK_N, 2))
    gate, linear = gl.split(reshaped)
    if limit > 0.0:
        gate = gl.minimum(gate, limit)
        linear = gl.clamp(linear, -limit, limit)
    s = gate / (1.0 + gl.exp(-alpha * gate))
    return s * (linear + beta)


@gluon.jit
def _stage1_mxfp4_direct_mfma_gluon(
    hidden_ptr,  # (M, D/2) uint8 e2m1 packed, token order
    hidden_scale_ptr,  # (Kscale_pad*32, ceil(M/32)) uint8 CDNA4 swizzled
    w1_ptr,  # (E, D/2 padded, 2*I) uint8 gdot128-shuffled
    w1s_ptr,  # (E, Kscale*32, ceil((2*I)/32)) uint8 CDNA4-swizzled
    topk_ids_ptr,  # (M, TOPK) int32
    out_ptr,  # BF16 (M*TOPK, I), or packed uint8 (M*TOPK, I/2)
    out_scale_ptr,  # CDNA4-swizzled scales when QUANTIZE_OUTPUT
    M,
    D,
    TWO_I,
    expert_start,
    num_local_experts,
    N_PHYS: gl.constexpr,
    stride_xm,
    stride_xk,
    stride_xslin,
    stride_xsnb,
    stride_we,
    stride_se,
    stride_slin,
    stride_snb,
    stride_om,
    stride_on,
    stride_oslin,
    stride_osmb,
    stride_tit,
    stride_tis,
    K_PACKED: gl.constexpr,
    TOPK: gl.constexpr,
    BLOCK_K: gl.constexpr,
    BLOCK_N: gl.constexpr,
    M_DUP: gl.constexpr,
    QUANTIZE_OUTPUT: gl.constexpr,
    SWIGLU_ALPHA: gl.constexpr,
    SWIGLU_LIMIT: gl.constexpr,
    SWIGLU_BETA: gl.constexpr,
):
    BLOCK_K_PACKED: gl.constexpr = BLOCK_K // 2
    BLOCK_K_SCALE: gl.constexpr = BLOCK_K // 32
    OUT_BLOCK_N: gl.constexpr = BLOCK_N // 2
    gl.static_assert(
        BLOCK_K == 128 and BLOCK_K_PACKED == 64,
        "direct MXFP4 stage1 currently assumes one CDNA4 scaled-MFMA K tile",
    )
    gl.static_assert(BLOCK_N % 2 == 0, "SwiGLU stage1 needs even BLOCK_N")
    gl.static_assert(
        128 % BLOCK_N == 0,
        "direct MXFP4 stage1 BLOCK_N must divide the gdot128 128-wide W tile",
    )

    pid = gl.program_id(axis=0)
    num_n = gl.cdiv(TWO_I, BLOCK_N)
    slot_flat = pid // num_n
    pid_n = pid % num_n
    token = slot_flat // TOPK
    slot = slot_flat - token * TOPK
    global_expert = gl.load(topk_ids_ptr + token * stride_tit + slot * stride_tis)
    expert = global_expert - expert_start
    owned = (expert >= 0) & (expert < num_local_experts)
    expert = gl.where(owned, expert, 0)

    layouts: gl.constexpr = _direct_mxfp4_mfma_layouts(
        M_DUP, BLOCK_N, BLOCK_K_SCALE, gl.num_warps()
    )
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
    x_row_off = token.to(gl.int64) * stride_xm
    w_expert_off = expert.to(gl.int64) * stride_we
    s_expert_off = expert.to(gl.int64) * stride_se
    # Keep the trip count compile-time.  With runtime ``D`` Gluon emitted one
    # load/wait/MFMA group per K tile; the constexpr bound lets the backend
    # overlap adjacent groups and cuts the 56-tile Kimi stage1 wait chain.
    TOTAL_KT: gl.constexpr = gl.cdiv(K_PACKED, BLOCK_K_PACKED)
    acc = gl.zeros((M_DUP, BLOCK_N), dtype=gl.float32, layout=mfma_layout)

    if token < M and owned:
        for kt in range(0, TOTAL_KT):
            a, b, a_scale, b_scale = _direct_mxfp4_load_tile(
                kt,
                ak,
                bk,
                ask,
                bsk,
                am,
                asm,
                hidden_ptr,
                hidden_scale_ptr,
                w1_ptr,
                w1s_ptr,
                x_row_off,
                w_expert_off,
                s_expert_off,
                n_cols,
                n_cols_s,
                token,
                stride_xk,
                stride_xslin,
                stride_xsnb,
                stride_slin,
                stride_snb,
                N_PHYS,
                D,
                TWO_I,
                K_PACKED,
                BLOCK_K_PACKED,
                BLOCK_K_SCALE,
            )
            acc = _direct_mxfp4_mfma(acc, a, b, a_scale, b_scale)

        out_tile = _direct_swiglu_reduce(
            acc,
            SWIGLU_ALPHA,
            SWIGLU_LIMIT,
            SWIGLU_BETA,
            OUT_BLOCK_N,
        )
        sm = gl.arange(
            0,
            M_DUP,
            layout=gl.SliceLayout(1, out_tile.type.layout),
        )[:, None]
        if QUANTIZE_OUTPUT:
            gl.static_assert(
                OUT_BLOCK_N % 32 == 0,
                "direct stage1 quantization needs whole group-32 output tiles",
            )
            packed, scale_byte = _mxfp4_quantize_tile(out_tile)
            packed_rows = gl.arange(
                0,
                M_DUP,
                layout=gl.SliceLayout(1, packed.type.layout),
            )
            packed_cols = gl.arange(
                0,
                OUT_BLOCK_N // 2,
                layout=gl.SliceLayout(0, packed.type.layout),
            )
            out_col = pid_n * (OUT_BLOCK_N // 2) + packed_cols[None, :]
            gl.store(
                out_ptr
                + slot_flat.to(gl.int64) * stride_om
                + out_col.to(gl.int64) * stride_on
                + packed_rows[:, None].to(gl.int64) * 0,
                packed,
                mask=(packed_rows[:, None] == 0) & (out_col < (TWO_I // 4)),
            )

            scale_rows = gl.arange(
                0,
                M_DUP,
                layout=gl.SliceLayout(1, scale_byte.type.layout),
            )
            scale_groups = gl.arange(
                0,
                OUT_BLOCK_N // 32,
                layout=gl.SliceLayout(0, scale_byte.type.layout),
            )
            _mxfp4_store_cdna4_scale(
                out_scale_ptr,
                scale_byte,
                slot_flat + scale_rows[:, None] * 0,
                pid_n * (OUT_BLOCK_N // 32) + scale_groups[None, :],
                stride_oslin,
                stride_osmb,
                scale_rows[:, None] == 0,
                M_SWIZZLE=32,
                K_SWIZZLE=8,
            )
        else:
            sn = gl.arange(
                0,
                OUT_BLOCK_N,
                layout=gl.SliceLayout(0, out_tile.type.layout),
            )[None, :]
            out_col = pid_n * OUT_BLOCK_N + sn
            gl.store(
                out_ptr
                + slot_flat.to(gl.int64) * stride_om
                + out_col.to(gl.int64) * stride_on
                + sm.to(gl.int64) * 0,
                out_tile.to(out_ptr.dtype.element_ty),
                mask=(sm == 0) & (out_col < (TWO_I // 2)),
            )


def invoke_stage1_mxfp4_mfma_decode_gluon(
    hidden_states_mxfp4,  # (num_tokens, D//2) uint8 e2m1, token order
    hidden_scale,  # CDNA4-swizzled e8m0 scales from _quantize_mxfp4_activation
    w1,  # (E, D//2 padded, 2*I_r) uint8 gdot128-shuffled
    w1_scale,  # (E, Kscale*32, ceil(2*I_r/32)) uint8 CDNA4-swizzled
    topk_ids,
    out,  # (num_tokens*topk, I_r) bf16
    topk: int,
    expert_start: int,
    out_scale: torch.Tensor | None,
    BLOCK_N: int = 32,
    BLOCK_K: int = 128,
    M_DUP: int = 4,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    swiglu_beta: float = 1.0,
):
    assert hidden_states_mxfp4.dtype == torch.uint8
    assert hidden_scale.dtype == torch.uint8
    assert w1.dtype == torch.uint8 and w1_scale.dtype == torch.uint8
    num_tokens = int(hidden_states_mxfp4.shape[0])
    num_local_experts, Dh_phys, two_I = w1.shape
    Dh = int(getattr(w1, "original_k_pk", Dh_phys))
    D = Dh * 2
    I_r = two_I // 2
    assert two_I % 2 == 0
    assert hidden_states_mxfp4.shape == (num_tokens, Dh)
    quantize_output = out_scale is not None
    if quantize_output:
        assert BLOCK_N == 64
        assert out.dtype == torch.uint8
        assert out.shape == (num_tokens * topk, I_r // 2)
        assert out_scale is not None
        assert out_scale.dtype == torch.uint8
        assert out_scale.stride(-2) == 1
    else:
        assert out.dtype == torch.bfloat16
        assert out.shape == (num_tokens * topk, I_r)
    assert bool(getattr(w1, "is_shuffled_for_gluon_dot", False))
    assert int(getattr(w1, "gluon_dot_block_k_pk", 0)) == 128
    assert int(getattr(w1, "gluon_dot_block_n", 0)) == 128
    assert w1_scale.stride(-2) == 1
    assert hidden_scale.stride(-2) == 1
    topk_ids = topk_ids.to(torch.int32)
    out_scale_arg = out if out_scale is None else out_scale
    grid = (num_tokens * topk * triton.cdiv(two_I, BLOCK_N),)
    _stage1_mxfp4_direct_mfma_gluon[grid](
        hidden_states_mxfp4,
        hidden_scale,
        w1,
        w1_scale,
        topk_ids,
        out,
        out_scale_arg,
        num_tokens,
        D,
        two_I,
        expert_start,
        num_local_experts,
        w1.shape[2],
        hidden_states_mxfp4.stride(0),
        hidden_states_mxfp4.stride(1),
        hidden_scale.stride(0),
        hidden_scale.stride(1),
        w1.stride(0),
        w1_scale.stride(0),
        w1_scale.stride(1),
        w1_scale.stride(2),
        out.stride(0),
        out.stride(1),
        0 if out_scale is None else out_scale.stride(0),
        0 if out_scale is None else out_scale.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        K_PACKED=Dh,
        TOPK=topk,
        BLOCK_K=BLOCK_K,
        BLOCK_N=BLOCK_N,
        M_DUP=M_DUP,
        QUANTIZE_OUTPUT=quantize_output,
        SWIGLU_ALPHA=float(swiglu_alpha),
        SWIGLU_LIMIT=float(swiglu_limit),
        SWIGLU_BETA=float(swiglu_beta),
        num_warps=4 if BLOCK_N == 64 else 1,
    )
    return out


__all__ = [
    "invoke_stage1_mxfp4_mfma_decode_gluon",
]


def _route_tile_launch_metadata(grid, kernel, args):
    return {"name": f"{kernel.name} (routes={args['out'].shape[0]})"}


@gluon.jit(launch_metadata=_route_tile_launch_metadata)
def gluon_mxfp4_moe_stage1_route_tiles_gfx950(
    x,
    xscale,
    w,
    wscale,
    sorted_routes,
    descriptors,
    group_count,
    out,
    out_scale,
    HIDDEN: gl.constexpr,
    INTERMEDIATE: gl.constexpr,
    TOPK: gl.constexpr,
    X_ROW: gl.constexpr,
    XS0: gl.constexpr,
    XS1: gl.constexpr,
    WE: gl.constexpr,
    SE: gl.constexpr,
    SS0: gl.constexpr,
    SS1: gl.constexpr,
    OS0: gl.constexpr,
    OS1: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    BLOCK_M: gl.constexpr,
):
    """Batch one expert's route tile and store intermediates in token/slot order."""
    gl.static_assert(HIDDEN % BLOCK_K == 0, "route tiles require complete K tiles")
    gl.static_assert(
        BLOCK_N % 64 == 0 and (2 * INTERMEDIATE) % BLOCK_N == 0,
        "gate/up tiles must cover complete 32-value quantization groups",
    )
    D: gl.constexpr = HIDDEN
    TWO_I: gl.constexpr = 2 * INTERMEDIATE
    K_PACKED: gl.constexpr = HIDDEN // 2
    INTER_PACKED: gl.constexpr = INTERMEDIATE // 2
    KP: gl.constexpr = BLOCK_K // 2
    KS: gl.constexpr = BLOCK_K // 32
    layouts: gl.constexpr = _direct_mxfp4_mfma_layouts(
        BLOCK_M, BLOCK_N, KS, gl.num_warps()
    )
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
    for job in range(gl.program_id(0), groups * (TWO_I // BLOCK_N), gl.num_programs(0)):
        group = job // (TWO_I // BLOCK_N)
        n = job % (TWO_I // BLOCK_N)
        start = gl.load(descriptors + group * 3)
        count = gl.load(descriptors + group * 3 + 1)
        expert = gl.load(descriptors + group * 3 + 2)
        first = gl.load(sorted_routes + start)
        routes = gl.load(sorted_routes + start + am, am < count, other=first)
        scale_routes = gl.load(sorted_routes + start + asm, asm < count, other=first)
        xoff = (routes // TOPK).to(gl.int64) * X_ROW
        soff = scale_routes // TOPK
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
                soff,
                1,
                XS0,
                XS1,
                SS0,
                SS1,
                TWO_I,
                D,
                TWO_I,
                K_PACKED,
                KP,
                KS,
            )
            acc = _direct_mxfp4_mfma(acc, a, b, sa, sb)
        tile = _direct_swiglu_reduce(acc, 1.0, 0.0, 0.0, BLOCK_N // 2)
        packed, scale = _mxfp4_quantize_tile(tile)
        pr = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, packed.type.layout))
        pc = gl.arange(0, BLOCK_N // 4, layout=gl.SliceLayout(0, packed.type.layout))
        destination = gl.load(sorted_routes + start + pr, pr < count, other=first)
        gl.store(
            out
            + destination[:, None].to(gl.int64) * INTER_PACKED
            + n * (BLOCK_N // 4)
            + pc[None, :],
            packed,
            pr[:, None] < count,
        )
        sr = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, scale.type.layout))
        sc = gl.arange(0, BLOCK_N // 64, layout=gl.SliceLayout(0, scale.type.layout))
        scale_destination = gl.load(sorted_routes + start + sr, sr < count, other=first)
        _mxfp4_store_cdna4_scale(
            out_scale,
            scale,
            scale_destination[:, None],
            n * (BLOCK_N // 64) + sc[None, :],
            OS0,
            OS1,
            sr[:, None] < count,
            M_SWIZZLE=32,
            K_SWIZZLE=8,
        )
