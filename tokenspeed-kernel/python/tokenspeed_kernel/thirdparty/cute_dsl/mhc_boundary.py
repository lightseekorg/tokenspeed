# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from TensorRT-LLM (tensorrt_llm/_torch/kernels/blackwell/mhc/mhc_boundary.py),
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
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

"""DeepSeek-V4.1 shifted mHC boundary kernel on Blackwell (CuTe DSL).

The hidden dimension is split into 16 slices across a 16-CTA DSMEM cluster.
Each CTA updates its slice of the 4 residual streams, collapses the stream with
the lagged ``pre`` mix, RMS-normalizes into the sublayer input, and multiplies
the updated residual with its slice of the mixer projection on BF16 tensor
cores using BF16 hi+lo weight fragments (``prepare_boundary_weight``).
CTAs exchange their partial 24 mixes and residual/collapsed sums of squares via
cluster distributed shared memory (`st.async.shared::cluster` + cluster
mbarriers) in deterministic fixed rank order, with zero global workspace or
inter-replay split-barrier state, so execution is 100% CUDA-graph safe across
both Decode graphs and ``PrefillGraph``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import SmemAllocator

CLUSTER = 16
THREADS = 320
MMA_WARPS = 10
MULT = 4
MIXES = 24
# Floats per pushed token partial: 24 mixes, sum of squares of residual_cur, padding.
RECV = 28
TILE = 24
MAX_TOKENS = 1 << 20
# 9 bucketed tile sizes keep 1-wave Decode spread across all resident clusters
# (at most 24 * max_decode_wave_clusters(device), e.g. 336 on 112 SMs) while
# tile=24 also covers multi-wave PrefillGraph batches via dynamic sym_int.
TILES = (1, 2, 3, 4, 6, 8, 12, 16, 24)
PREFETCH = 4
# Measured per-tile schedules: (one 8-wide chunk per pass, L2 weight prefetch).
_TILE_CONFIG = {
    12: (True, False),
    16: (False, True),
}

_ATT = llvm.AsmDialect.AD_ATT


def _asm(result, args, body, constraints, side_effects=True, loc=None, ip=None):
    return llvm.inline_asm(
        result,
        [a.ir_value(loc=loc, ip=ip) for a in args],
        body,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=_ATT,
        loc=loc,
        ip=ip,
    )


def _fields(out, n, ty, wrap, loc, ip):
    return tuple(
        wrap(llvm.extractvalue(ty, out, [i], loc=loc, ip=ip)) for i in range(n)
    )


@dsl_user_op
def fp32x2_to_bf16x2(lo, hi, *, loc=None, ip=None):
    """Pack ``lo`` into bits [15:0] and ``hi`` into bits [31:16] as BF16."""
    return Uint32(
        _asm(
            T.i32(),
            [Float32(lo), Float32(hi)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def lds_v4_u32(addr, *, loc=None, ip=None):
    out = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int32(addr)],
        "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return _fields(out, 4, T.i32(), Uint32, loc, ip)


@dsl_user_op
def lds_v4_f32(addr, *, loc=None, ip=None):
    out = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Int32(addr)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return _fields(out, 4, T.f32(), Float32, loc, ip)


@dsl_user_op
def lds_f32(addr, *, loc=None, ip=None):
    return Float32(
        _asm(T.f32(), [Int32(addr)], "ld.shared.f32 $0, [$1];", "=f,r", loc=loc, ip=ip)
    )


@dsl_user_op
def ldsm_x4(addr, *, loc=None, ip=None):
    out = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int32(addr)],
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return _fields(out, 4, T.i32(), Uint32, loc, ip)


@dsl_user_op
def sts_f32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(addr), Float32(value)],
        "st.shared.f32 [$0], $1;",
        "r,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def sts_v2_f32(addr, a, b, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(addr), Float32(a), Float32(b)],
        "st.shared.v2.f32 [$0], {$1, $2};",
        "r,f,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def sts_v4_u32(addr, w, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(addr)] + [Uint32(v) for v in w],
        "st.shared.v4.u32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def sts_u32(addr, w, *, loc=None, ip=None):
    _asm(
        None, [Int32(addr), Uint32(w)], "st.shared.u32 [$0], $1;", "r,r", loc=loc, ip=ip
    )


@dsl_user_op
def ldg_v4_u32(addr, ordered=False, *, loc=None, ip=None):
    out = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(addr)],
        "ld.global.nc.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        side_effects=ordered,
        loc=loc,
        ip=ip,
    )
    return _fields(out, 4, T.i32(), Uint32, loc, ip)


@dsl_user_op
def ldg_v4_f32(addr, *, loc=None, ip=None):
    out = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Int64(addr)],
        "ld.global.nc.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return _fields(out, 4, T.f32(), Float32, loc, ip)


@dsl_user_op
def ldg_f32(addr, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(), [Int64(addr)], "ld.global.nc.f32 $0, [$1];", "=f,l", loc=loc, ip=ip
        )
    )


@dsl_user_op
def stg_v4_u32(addr, w, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr)] + [Uint32(v) for v in w],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stg_f32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr), Float32(value)],
        "st.global.f32 [$0], $1;",
        "l,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def bulk_g2s(dst, src, size, bar, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(dst), Int64(src), Int32(size), Int32(bar)],
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        "r,l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def bulk_prefetch_l2(src, size, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(src), Int32(size)],
        "cp.async.bulk.prefetch.L2.global [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_async_v4(addr, w, bar, rank, *, loc=None, ip=None):
    """Store 4 words at CTA ``rank``'s shared memory offset and signal ``bar``."""
    _asm(
        None,
        [Int32(addr)] + [Uint32(v) for v in w] + [Int32(bar), Int32(rank)],
        "{\n.reg .b32 ra, rb;\nmapa.shared::cluster.u32 ra, $0, $6;\nmapa.shared::cluster.u32 rb, $5, $6;\n"
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [ra], {$1, $2, $3, $4}, [rb];\n}",
        "r,r,r,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbar_init(bar, count, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(bar), Int32(count)],
        "mbarrier.init.shared::cta.b64 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbar_fence_init(*, loc=None, ip=None):
    _asm(None, [], "fence.mbarrier_init.release.cluster;", "", loc=loc, ip=ip)


@dsl_user_op
def mbar_expect_tx(bar, size, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(bar), Int32(size)],
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbar_wait(bar, phase, *, scope="cta", loc=None, ip=None):
    """Wait for ``phase``; ``scope="cluster"`` acquires st.async cluster writes."""
    sem = ".acquire.cluster" if scope == "cluster" else ""
    _asm(
        None,
        [Int32(bar), Int32(phase)],
        "{\n.reg .pred p;\nWAIT:\n"
        f"mbarrier.try_wait.parity{sem}.shared::cta.b64 p, [$0], $1;\n"
        "@!p bra WAIT;\n}",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mma_bf16(a, b, c, *, loc=None, ip=None):
    """``mma.sync`` m16n8k16 BF16 with FP32 accumulation."""
    out = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Uint32(v) for v in a] + [Uint32(v) for v in b] + [Float32(v) for v in c],
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, "
        "{$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return _fields(out, 4, T.f32(), Float32, loc, ip)


@dsl_user_op
def rcp(x, *, loc=None, ip=None):
    """Fast-path IEEE reciprocal refinement without slow out-of-range branch."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(x)],
            "{\n.reg .f32 r, e;\nrcp.approx.ftz.f32 r, $1;\nfma.rn.f32 e, $1, r, 0fBF800000;\n"
            "neg.f32 e, e;\nfma.rn.f32 $0, r, e, r;\n}",
            "=f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bf16x2_lo(word, *, loc=None, ip=None):
    return Float32(
        llvm.bitcast(
            T.f32(), (Uint32(word) << 16).ir_value(loc=loc, ip=ip), loc=loc, ip=ip
        )
    )


@dsl_user_op
def bf16x2_hi(word, *, loc=None, ip=None):
    high = Uint32(word) & Uint32(0xFFFF0000)
    return Float32(llvm.bitcast(T.f32(), high.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def f32_bits(value, *, loc=None, ip=None):
    return Uint32(
        llvm.bitcast(T.i32(), Float32(value).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    )


def sigmoid(v):
    """``1 / (1 + exp(-v))`` clamped below 2^126 for reciprocal stability."""
    return rcp(cute.arch.fmin(Float32(1.0) + cute.math.exp(-v), Float32(2.0**126)))


def residual_update(pw, cm, xw, rw):
    """BF16 pairs of ``bf16(post[s] * x + sum_j comb[j][s] * residual[j])``."""
    xf = [(bf16x2_lo(xw[p]), bf16x2_hi(xw[p])) for p in range(4)]
    rf = [
        [(bf16x2_lo(rw[j][p]), bf16x2_hi(rw[j][p])) for p in range(4)]
        for j in range(MULT)
    ]
    cur = []
    for s in range(MULT):
        words = []
        for p in range(4):
            nu = cute.arch.mul_packed_f32x2((pw[s], pw[s]), xf[p], ftz=True)
            for j in range(MULT):
                nu = cute.arch.fma_packed_f32x2(
                    (cm[j][s], cm[j][s]), rf[j][p], nu, ftz=True
                )
            words.append(fp32x2_to_bf16x2(nu[0], nu[1]))
        cur.append(words)
    return cur


def lagged_collapse(cur, pre, ssq_rc, ssq_li):
    """BF16 pairs of ``li = bf16(sum_j pre[j] * residual_cur[j])`` plus sums of squares."""
    li = []
    for p in range(4):
        acc = (Float32(0.0), Float32(0.0))
        for j in range(MULT):
            v = (bf16x2_lo(cur[j][p]), bf16x2_hi(cur[j][p]))
            ssq_rc = cute.arch.fma_packed_f32x2(v, v, ssq_rc)
            acc = cute.arch.fma_packed_f32x2((pre[j], pre[j]), v, acc)
        word = fp32x2_to_bf16x2(acc[0], acc[1])
        li.append(word)
        v = (bf16x2_lo(word), bf16x2_hi(word))
        ssq_li = cute.arch.fma_packed_f32x2(v, v, ssq_li)
    return li, ssq_rc, ssq_li


def _plan_layout(hidden: int, tile: int) -> dict[str, int]:
    """Compute shared-memory offsets and pass geometry for ``(hidden, tile)``."""
    hidden_slice = hidden // CLUSTER
    chunks = hidden_slice // 8
    parts = chunks // 2
    k_steps = MULT * hidden_slice // 16
    steps_per_warp = k_steps // MMA_WARPS
    n_tiles = (tile + 7) // 8
    cols = 40 if tile > 16 else 24
    owned = (tile + CLUSTER - 1) // CLUSTER
    li_words = (tile + 3) // 4 * 4
    one_chunk, l2_prefetch = _TILE_CONFIG.get(tile, (False, False))
    one = THREADS // max(2 * parts, 1)
    passes: list[tuple[int, int, int]] = []
    first = 0
    if one > 0:
        while first < tile:
            if one_chunk or tile - first <= one:
                passes.append((first, min(tile - first, one), 1))
            else:
                passes.append((first, min(tile - first, 2 * one), 2))
            first += passes[-1][1]
    staged = tile - passes[0][1] if passes else tile
    r_chunk = 8 * 32 + 16
    r_row = (hidden_slice // 32) * r_chunk + 16
    x_row = hidden_slice * 2

    off = 0

    def take(size: int, align: int = 16) -> int:
        nonlocal off
        off = (off + align - 1) // align * align
        start = off
        off += size
        return start

    o_r = take(max(tile * r_row, MMA_WARPS * MIXES * cols * 4), 128)
    o_x = take(tile * x_row)
    o_prc = take(tile * 2 * parts * 4)
    o_pli = take(tile * 2 * parts * 4)
    o_cta = take(tile * RECV * 4)
    o_cli = take(li_words * 4)
    o_rmix = take(owned * CLUSTER * RECV * 4)
    o_rli = take(CLUSTER * li_words * 4)
    o_tot = take(2 * RECV * 4)
    o_rows = take(2 * MULT * MULT * 4)
    o_rstd = take(li_words * 4)
    o_hcw = take(32 * 4)
    o_coef = take(max(staged, 1) * 24 * 4)
    o_bar = take(16, 8)
    return {
        "slice": hidden_slice,
        "chunks": chunks,
        "parts": parts,
        "k_steps": k_steps,
        "steps_per_warp": steps_per_warp,
        "n_tiles": n_tiles,
        "cols": cols,
        "owned": owned,
        "li_words": li_words,
        "l2_prefetch": int(l2_prefetch),
        "staged": staged,
        "r_chunk": r_chunk,
        "r_row": r_row,
        "x_row": x_row,
        "o_r": o_r,
        "o_x": o_x,
        "o_prc": o_prc,
        "o_pli": o_pli,
        "o_cta": o_cta,
        "o_cli": o_cli,
        "o_rmix": o_rmix,
        "o_rli": o_rli,
        "o_tot": o_tot,
        "o_rows": o_rows,
        "o_rstd": o_rstd,
        "o_hcw": o_hcw,
        "o_coef": o_coef,
        "o_bar": o_bar,
        "smem_bytes": off,
        "passes": passes,
    }


class MhcBoundaryKernel:
    """16-CTA DSMEM cluster shifted mHC boundary kernel."""

    def __init__(
        self, hidden: int, sinkhorn_iters: int, tile: int, use_pdl: bool
    ) -> None:
        if not self.supports(hidden, MULT, tile):
            raise ValueError(
                f"MhcBoundaryKernel does not support hidden={hidden}, tile={tile}"
            )
        if sinkhorn_iters < 1:
            raise ValueError("MhcBoundaryKernel requires sinkhorn_iters >= 1")
        self.tile = tile
        self.hidden = hidden
        self.sinkhorn_iters = sinkhorn_iters
        self.use_pdl = use_pdl
        plan = _plan_layout(hidden, tile)
        self.slice = plan["slice"]
        self.chunks = plan["chunks"]
        self.parts = plan["parts"]
        self.k_steps = plan["k_steps"]
        self.steps_per_warp = plan["steps_per_warp"]
        self.n_tiles = plan["n_tiles"]
        self.cols = plan["cols"]
        self.owned = plan["owned"]
        self.li_words = plan["li_words"]
        self.l2_prefetch = bool(plan["l2_prefetch"])
        self.passes: list[tuple[int, int, int]] = plan["passes"]
        self.staged = plan["staged"]
        self.r_chunk = plan["r_chunk"]
        self.r_row = plan["r_row"]
        self.x_row = plan["x_row"]
        self.o_r = plan["o_r"]
        self.o_x = plan["o_x"]
        self.o_prc = plan["o_prc"]
        self.o_pli = plan["o_pli"]
        self.o_cta = plan["o_cta"]
        self.o_cli = plan["o_cli"]
        self.o_rmix = plan["o_rmix"]
        self.o_rli = plan["o_rli"]
        self.o_tot = plan["o_tot"]
        self.o_rows = plan["o_rows"]
        self.o_rstd = plan["o_rstd"]
        self.o_hcw = plan["o_hcw"]
        self.o_coef = plan["o_coef"]
        self.o_bar = plan["o_bar"]
        self.smem_bytes = plan["smem_bytes"]
        self.ctas_per_sm = 2 if self.smem_bytes <= 112 * 1024 else 1

    @staticmethod
    def supports(hidden: int, hc_mult: int, tile: int = TILE) -> bool:
        if hc_mult != MULT or hidden <= 0 or hidden % 512 != 0 or tile <= 0:
            return False
        hidden_slice = hidden // CLUSTER
        if (MULT * hidden_slice // 16) % MMA_WARPS != 0:
            return False
        parts = (hidden_slice // 8) // 2
        if parts <= 0 or 2 * parts > THREADS:
            return False
        return _plan_layout(hidden, tile)["smem_bytes"] <= 224 * 1024

    @cute.jit
    def __call__(
        self,
        x,
        residual,
        post,
        comb,
        pre_lag,
        weight,
        scale,
        base,
        norm_w,
        residual_out,
        layer_input,
        pre_out,
        post_out,
        comb_out,
        rms_eps: Float32,
        hc_eps: Float32,
        sinkhorn_eps: Float32,
        post_mult: Float32,
        norm_eps: Float32,
        stream,
    ):
        tiles = (residual.shape[0] + self.tile - 1) // self.tile
        self.kernel(
            x,
            residual,
            post,
            comb,
            pre_lag,
            weight,
            scale,
            base,
            norm_w,
            residual_out,
            layer_input,
            pre_out,
            post_out,
            comb_out,
            rms_eps,
            hc_eps,
            sinkhorn_eps,
            post_mult,
            norm_eps,
        ).launch(
            # Deadlock-free launch: each CLUSTER=16 CTA cluster reduces solely
            # inside cluster DSMEM for its own disjoint token tile and never
            # waits on another cluster (monokernel deadlock rule), so multi-wave
            # launches remain safe at any SM count.
            grid=[tiles * CLUSTER, 1, 1],
            block=[THREADS, 1, 1],
            cluster=[CLUSTER, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=self.ctas_per_sm,
            preferred_smem_carveout=100,
        )

    @cute.kernel
    def kernel(
        self,
        x,
        residual,
        post,
        comb,
        pre_lag,
        weight,
        scale,
        base,
        norm_w,
        residual_out,
        layer_input,
        pre_out,
        post_out,
        comb_out,
        rms_eps: Float32,
        hc_eps: Float32,
        sinkhorn_eps: Float32,
        post_mult: Float32,
        norm_eps: Float32,
    ):
        H = self.hidden
        S = self.slice
        PARTS = self.parts
        tid = cute.arch.thread_idx()[0]
        bid = cute.arch.block_idx()[0]
        rank = bid % CLUSTER
        t0 = (bid // CLUSTER) * self.tile
        valid = cutlass.min(self.tile, residual.shape[0] - t0)
        p0 = self.passes[0][1]
        staged = self.staged

        smem = SmemAllocator()
        sbase = smem.allocate(self.smem_bytes, 128).toint()
        s_r = sbase + self.o_r
        s_x = sbase + self.o_x
        s_hcw = sbase + self.o_hcw
        s_coef = sbase + self.o_coef
        bar_in = sbase + self.o_bar
        bar_x = bar_in + 8
        hslice = Int64(rank * S)
        owned_here = (self.tile - rank + CLUSTER - 1) // CLUSTER
        g_frag = weight.iterator.toint() + Int64(rank) * Int64(self.k_steps * 1536)

        if tid == 0:
            mbar_init(bar_in, 1)
            mbar_init(bar_x, 1)
            mbar_fence_init()
            mbar_expect_tx(
                bar_in, cutlass.max(valid - p0, 0) * ((MULT + 1) * S * 2 + 24 * 4)
            )
            mbar_expect_tx(
                bar_x, owned_here * CLUSTER * RECV * 4 + CLUSTER * self.li_words * 4
            )
        cute.arch.sync_threads()
        cute.arch.cluster_arrive_relaxed()

        if (tid >= 256) & (tid < 256 + 3 + MIXES):
            i = tid - 256
            if i < 3:
                sts_f32(s_hcw + i * 4, ldg_f32(scale.iterator.toint() + i * 4))
            else:
                sts_f32(s_hcw + i * 4, ldg_f32(base.iterator.toint() + (i - 3) * 4))
        prefetchers = CLUSTER if self.l2_prefetch else 0
        if (tid < MMA_WARPS) & (bid < prefetchers):
            bulk_prefetch_l2(
                g_frag + Int64(tid * self.steps_per_warp * 1536),
                self.steps_per_warp * 1536,
            )

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        if (tid == THREADS - 1) & (valid > p0):
            n = valid - p0
            g0 = Int64(t0 + p0)
            bulk_g2s(s_coef, pre_lag.iterator.toint() + g0 * 16, n * 16, bar_in)
            bulk_g2s(
                s_coef + staged * 16, post.iterator.toint() + g0 * 16, n * 16, bar_in
            )
            bulk_g2s(
                s_coef + staged * 32, comb.iterator.toint() + g0 * 64, n * 64, bar_in
            )
        if tid < cutlass.max(valid - p0, 0) * (MULT + 1):
            bt = tid // (MULT + 1) + p0
            bj = tid % (MULT + 1)
            gtb = Int64(t0 + bt)
            if bj < MULT:
                bulk_g2s(
                    s_r + bt * self.r_row + bj * S * 2,
                    residual.iterator.toint() + ((gtb * MULT + bj) * H + hslice) * 2,
                    S * 2,
                    bar_in,
                )
            else:
                bulk_g2s(
                    s_x + bt * self.x_row,
                    x.iterator.toint() + (gtb * H + hslice) * 2,
                    S * 2,
                    bar_in,
                )

        for pi in cutlass.range_constexpr(len(self.passes)):
            first, count, per_thread = self.passes[pi]
            NP = 2 * PARTS // per_thread
            t = first + tid // NP
            part = tid % NP
            tc = cutlass.min(t, valid - 1)
            gt = Int64(t0 + tc)
            chunks = [part] if per_thread == 1 else [part, part + PARTS]
            rws = []
            xws = []
            if cutlass.const_expr(pi == 0):
                pre = ldg_v4_f32(pre_lag.iterator.toint() + gt * 16)
                pw = ldg_v4_f32(post.iterator.toint() + gt * 16)
                cm = [
                    ldg_v4_f32(comb.iterator.toint() + gt * 64 + j * 16)
                    for j in range(MULT)
                ]
                g_row = residual.iterator.toint() + (gt * MULT * H + hslice) * 2
                g_xrow = x.iterator.toint() + (gt * H + hslice) * 2
                for c in chunks:
                    rws.append(
                        [
                            ldg_v4_u32(g_row + Int64(j * H * 2 + c * 16))
                            for j in range(MULT)
                        ]
                    )
                    xws.append(ldg_v4_u32(g_xrow + Int64(c * 16)))
            else:
                if cutlass.const_expr(pi == 1):
                    mbar_wait(bar_in, 0)
                ci = cutlass.max(tc - p0, 0)
                pre = lds_v4_f32(s_coef + ci * 16)
                pw = lds_v4_f32(s_coef + staged * 16 + ci * 16)
                cm = [
                    lds_v4_f32(s_coef + staged * 32 + ci * 64 + j * 16)
                    for j in range(MULT)
                ]
                row = s_r + tc * self.r_row
                for c in chunks:
                    rws.append(
                        [lds_v4_u32(row + j * S * 2 + c * 16) for j in range(MULT)]
                    )
                    xws.append(lds_v4_u32(s_x + tc * self.x_row + c * 16))
                cute.arch.sync_threads()
            if (t < valid) & (tid < count * NP):
                ssq_rc = (Float32(0.0), Float32(0.0))
                ssq_li = (Float32(0.0), Float32(0.0))
                for ci_ in cutlass.range_constexpr(len(chunks)):
                    c = chunks[ci_]
                    r_addr = (
                        s_r + t * self.r_row + (c // 4) * self.r_chunk + (c % 4) * 4
                    )
                    cur = residual_update(pw, cm, xws[ci_], rws[ci_])
                    for s in cutlass.range_constexpr(MULT):
                        stg_v4_u32(
                            residual_out.iterator.toint()
                            + ((gt * MULT + s) * H + hslice + c * 8) * 2,
                            cur[s],
                        )
                    for s in cutlass.range_constexpr(MULT):
                        for w in cutlass.range_constexpr(4):
                            sts_u32(
                                r_addr + (2 * s + w // 2) * 32 + (w % 2) * 16, cur[s][w]
                            )
                    li, ssq_rc, ssq_li = lagged_collapse(cur, pre, ssq_rc, ssq_li)
                    sts_v4_u32(s_x + t * self.x_row + c * 16, li)
                slot = (part * self.tile + t) * 4
                sts_f32(sbase + self.o_prc + slot, ssq_rc[0] + ssq_rc[1])
                sts_f32(sbase + self.o_pli + slot, ssq_li[0] + ssq_li[1])
                if cutlass.const_expr(per_thread == 2):
                    slot = ((part + PARTS) * self.tile + t) * 4
                    sts_f32(sbase + self.o_prc + slot, Float32(0.0))
                    sts_f32(sbase + self.o_pli + slot, Float32(0.0))
        cute.arch.sync_threads()

        warp = tid // 32
        lane = tid % 32
        g = lane // 4
        q = lane % 4
        mrow = (lane % 8) + (lane // 16) * 8
        mk = ((lane // 8) % 2) * 8
        if warp < MMA_WARPS:
            acc = [[[Float32(0.0)] * 4 for _ in range(self.n_tiles)] for _ in range(3)]
            spw = self.steps_per_warp
            g_w = g_frag + Int64(warp * spw * 1536 + lane * 16)
            pending = [
                [ldg_v4_u32(g_w + st * 1536 + m * 512, True) for m in range(3)]
                for st in range(min(PREFETCH, spw))
            ]
            for step in cutlass.range_constexpr(spw):
                kk = warp * spw + step
                a = [tuple(words) for words in pending[0]]
                pending = pending[1:]
                if cutlass.const_expr(step + PREFETCH < spw):
                    nxt = g_w + (step + PREFETCH) * 1536
                    pending.append([ldg_v4_u32(nxt + m * 512, True) for m in range(3)])
                b = []
                for pair in cutlass.range_constexpr((self.n_tiles + 1) // 2):
                    tok = cutlass.min(pair * 16 + mrow, self.tile - 1)
                    m4 = ldsm_x4(
                        s_r
                        + tok * self.r_row
                        + (kk // 8) * self.r_chunk
                        + (kk % 8) * 32
                        + mk * 2
                    )
                    b.append((m4[0], m4[1]))
                    b.append((m4[2], m4[3]))
                for nt in cutlass.range_constexpr(self.n_tiles):
                    for mt in cutlass.range_constexpr(3):
                        acc[mt][nt] = list(mma_bf16(a[mt], b[nt], acc[mt][nt]))
            cute.arch.barrier(barrier_id=1, number_of_threads=MMA_WARPS * 32)
            for nt in cutlass.range_constexpr(self.n_tiles):
                col = nt * 8 + 2 * q
                for mi in cutlass.range_constexpr(3):
                    hi_t, hi_r = ((0, 0), (0, 2), (1, 0))[mi]
                    lo_t, lo_r = ((1, 2), (2, 0), (2, 2))[mi]
                    sts_v2_f32(
                        s_r + ((warp * MIXES + mi * 8 + g) * self.cols + col) * 4,
                        acc[hi_t][nt][hi_r] + acc[lo_t][nt][lo_r],
                        acc[hi_t][nt][hi_r + 1] + acc[lo_t][nt][lo_r + 1],
                    )
        cute.arch.sync_threads()

        s_cta = sbase + self.o_cta
        s_cli = sbase + self.o_cli
        for k in cutlass.range_constexpr((self.tile * 26 + THREADS - 1) // THREADS):
            i = tid + k * THREADS
            if i < self.tile * MIXES:
                n = i // self.tile
                tt = i % self.tile
                parts = [
                    lds_f32(s_r + ((w * MIXES + n) * self.cols + tt) * 4)
                    for w in range(MMA_WARPS)
                ]
                v = Float32(0.0)
                for w in cutlass.range_constexpr(MMA_WARPS):
                    v = v + parts[w]
                sts_f32(s_cta + (tt * RECV + n) * 4, v)
            elif i < self.tile * 26:
                ii = i - self.tile * MIXES
                kind = ii // self.tile
                tt = ii % self.tile
                src = sbase + self.o_prc + kind * (self.o_pli - self.o_prc) + tt * 4
                parts = [lds_f32(src + j * self.tile * 4) for j in range(2 * PARTS)]
                v = Float32(0.0)
                for j in cutlass.range_constexpr(2 * PARTS):
                    v = v + parts[j]
                if kind == 0:
                    sts_f32(s_cta + (tt * RECV + MIXES) * 4, v)
                else:
                    sts_f32(s_cli + tt * 4, v)
        cute.arch.sync_threads()

        cute.arch.cluster_wait()
        quads = RECV // 4
        li_quads = self.li_words // 4
        if tid < self.tile * quads:
            tt = tid // quads
            j = tid % quads
            dst = (
                sbase
                + self.o_rmix
                + (((tt // CLUSTER) * CLUSTER + rank) * RECV + j * 4) * 4
            )
            st_async_v4(
                dst, lds_v4_u32(s_cta + (tt * RECV + j * 4) * 4), bar_x, tt % CLUSTER
            )
        elif tid < self.tile * quads + CLUSTER * li_quads:
            i = tid - self.tile * quads
            j = i % li_quads
            dst = sbase + self.o_rli + (rank * self.li_words + j * 4) * 4
            st_async_v4(dst, lds_v4_u32(s_cli + j * 16), bar_x, i // li_quads)
        mbar_wait(bar_x, 0, scope="cluster")
        cute.arch.cluster_arrive_relaxed()
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        s_tot = sbase + self.o_tot
        s_rstd = sbase + self.o_rstd
        if tid < self.tile:
            parts = [
                lds_f32(sbase + self.o_rli + (p * self.li_words + tid) * 4)
                for p in range(CLUSTER)
            ]
            v = Float32(0.0)
            for p in cutlass.range_constexpr(CLUSTER):
                v = v + parts[p]
            sts_f32(s_rstd + tid * 4, cute.math.rsqrt(v / Float32(H) + norm_eps))
        item = tid - 32
        if (item >= 0) & (item < self.owned * 25):
            o = item // 25
            vi = item % 25
            parts = [
                lds_f32(sbase + self.o_rmix + ((o * CLUSTER + p) * RECV + vi) * 4)
                for p in range(CLUSTER)
            ]
            v = Float32(0.0)
            for p in cutlass.range_constexpr(CLUSTER):
                v = v + parts[p]
            sts_f32(s_tot + (o * RECV + vi) * 4, v)
        cute.arch.sync_threads()

        if tid < 32:
            self.coefficients(
                s_tot + (lane // 16) * RECV * 4,
                s_hcw,
                sbase + self.o_rows + (lane // 16) * 64,
                rank + (lane // 16) * CLUSTER,
                lane % 16,
                t0,
                valid,
                pre_out,
                post_out,
                comb_out,
                rms_eps,
                hc_eps,
                sinkhorn_eps,
                post_mult,
            )
        else:
            for k in cutlass.range_constexpr(
                (self.tile * self.chunks + THREADS - 33) // (THREADS - 32)
            ):
                u = tid - 32 + k * (THREADS - 32)
                tt = u // self.chunks
                c = u % self.chunks
                if tt < valid:
                    out = self.layer_input_words(s_x, s_rstd, norm_w, hslice, tt, c)
                    stg_v4_u32(
                        layer_input.iterator.toint()
                        + (Int64(t0 + tt) * H + hslice + c * 8) * 2,
                        out,
                    )
        cute.arch.cluster_wait()

    @cute.jit
    def layer_input_words(self, s_x, s_rstd, norm_w, hslice, tt, c):
        li = lds_v4_u32(s_x + tt * self.x_row + c * 16)
        nw = ldg_v4_u32(norm_w.iterator.toint() + hslice * 2 + c * 16)
        rstd = lds_f32(s_rstd + tt * 4)
        out = []
        for p in cutlass.range_constexpr(4):
            lo = (bf16x2_lo(li[p]) * rstd) * bf16x2_lo(nw[p])
            hi = (bf16x2_hi(li[p]) * rstd) * bf16x2_hi(nw[p])
            out.append(fp32x2_to_bf16x2(lo, hi))
        return out

    @cute.jit
    def coefficients(
        self,
        tot,
        s_hcw,
        s_rows,
        tt,
        r,
        t0,
        valid,
        pre_out,
        post_out,
        comb_out,
        rms_eps: Float32,
        hc_eps: Float32,
        eps: Float32,
        post_mult: Float32,
    ):
        rstd = cute.math.rsqrt(
            lds_f32(tot + MIXES * 4) / Float32(MULT * self.hidden) + rms_eps
        )
        gt = Int64(t0 + tt)
        if r < 2 * MULT:
            v = cute.math.fma(
                lds_f32(tot + r * 4) * rstd,
                lds_f32(s_hcw + (r // MULT) * 4),
                lds_f32(s_hcw + (3 + r) * 4),
            )
            if r < MULT:
                if tt < valid:
                    stg_f32(
                        pre_out.iterator.toint() + (gt * MULT + r) * 4,
                        sigmoid(v) + hc_eps,
                    )
            elif tt < valid:
                stg_f32(
                    post_out.iterator.toint() + (gt * MULT + r - MULT) * 4,
                    sigmoid(v) * post_mult,
                )
        elif r < 3 * MULT:
            i = r - 2 * MULT
            s2 = lds_f32(s_hcw + 2 * 4)
            row = [
                cute.math.fma(
                    lds_f32(tot + (8 + 4 * i + k) * 4) * rstd,
                    s2,
                    lds_f32(s_hcw + (11 + 4 * i + k) * 4),
                )
                for k in range(MULT)
            ]
            row_max = cute.arch.fmax(
                cute.arch.fmax(row[0], row[1]), cute.arch.fmax(row[2], row[3])
            )
            row = [cute.math.exp(v - row_max) for v in row]
            inv = rcp(row[0] + row[1] + row[2] + row[3])
            row = [cute.math.fma(v, inv, eps) for v in row]
            for k in cutlass.range_constexpr(MULT):
                sts_f32(s_rows + (4 * i + k) * 4, row[k])
        cute.arch.sync_warp()
        if r == 3 * MULT:
            out = self.sinkhorn(
                [lds_f32(s_rows + k * 4) for k in range(MULT * MULT)], eps
            )
            if tt < valid:
                for i in cutlass.range_constexpr(MULT):
                    stg_v4_u32(
                        comb_out.iterator.toint() + (gt * 16 + i * 4) * 4,
                        [f32_bits(v) for v in out[4 * i : 4 * i + 4]],
                    )

    @cute.jit
    def sinkhorn(self, c, eps: Float32):
        c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15 = (
            self.col_normalize(c, eps)
        )
        for _ in cutlass.range(self.sinkhorn_iters - 1, unroll=1):
            cc = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15]
            rr = []
            for i in cutlass.range_constexpr(MULT):
                rinv = rcp(
                    cc[4 * i] + cc[4 * i + 1] + cc[4 * i + 2] + cc[4 * i + 3] + eps
                )
                rr.extend([cc[4 * i + k] * rinv for k in range(MULT)])
            c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15 = (
                self.col_normalize(rr, eps)
            )
        return [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15]

    @staticmethod
    def col_normalize(c, eps):
        out = list(c)
        for k in range(MULT):
            inv = rcp((c[k] + c[4 + k]) + (c[8 + k] + c[12 + k]) + eps)
            for i in range(MULT):
                out[4 * i + k] = c[4 * i + k] * inv
        return out


def boundary_weight_shape(hidden: int) -> tuple[int, int, int, int, int]:
    """Return packed fragment shape for ``prepare_boundary_weight(fn)``."""
    return (CLUSTER, MULT * (hidden // CLUSTER) // 16, 3, 32, 4)


def prepare_boundary_weight(fn: torch.Tensor) -> torch.Tensor:
    """Pack ``[24, 4 * hidden]`` FP32 weights into BF16 hi+lo MMA fragments."""
    if fn.ndim != 2 or fn.shape[0] != MIXES or fn.dtype != torch.float32:
        raise ValueError("prepare_boundary_weight expects FP32 tensor [24, 4 * hidden]")
    if fn.shape[1] % MULT != 0:
        raise ValueError("prepare_boundary_weight width must be divisible by 4")
    hidden = fn.shape[1] // MULT
    if not MhcBoundaryKernel.supports(hidden, MULT, TILE):
        raise ValueError(f"Unsupported hidden={hidden} for mhc_boundary weight packing")
    hidden_slice = hidden // CLUSTER
    fn = fn.contiguous()
    hi = fn.to(torch.bfloat16)
    lo = (fn - hi.float()).to(torch.bfloat16)
    pairs = torch.cat([hi, lo]).view(torch.int32)
    dev = fn.device
    rank = torch.arange(CLUSTER, device=dev).view(-1, 1, 1, 1)
    kk = torch.arange(MULT * hidden_slice // 16, device=dev).view(1, -1, 1, 1)
    lane = torch.arange(32, device=dev).view(1, 1, -1, 1)
    g, q = lane // 4, lane % 4
    c, kb = kk // 8, kk % 8
    base = (kb // 2) * hidden + rank * hidden_slice + c * 32 + q * 8 + (kb % 2) * 4
    rows = torch.tensor([0, 8, 0, 8, 16, 24, 16, 24, 32, 40, 32, 40], device=dev).view(
        1, 1, 1, -1
    )
    upper = torch.tensor([0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2], device=dev).view(
        1, 1, 1, -1
    )
    frags = pairs[rows + g, (base + upper) // 2]
    return (
        frags.view(CLUSTER, MULT * hidden_slice // 16, 32, 3, 4)
        .transpose(2, 3)
        .contiguous()
    )


@functools.lru_cache(maxsize=16)
def max_decode_wave_clusters(device: torch.device | str | int) -> int:
    """Return clusters fitting concurrently in one wave at 2 CTAs per SM."""
    if isinstance(device, int):
        idx = device
    else:
        dev = torch.device(device)
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
    sm_count = torch.cuda.get_device_properties(idx).multi_processor_count
    return max(1, (2 * sm_count) // CLUSTER)


def max_decode_wave_tokens(device: torch.device | str | int) -> int:
    """Maximum token count fitting in one resident cluster wave on ``device``."""
    return max_decode_wave_clusters(device) * TILE


def boundary_tile(tokens: int, *, clusters: int) -> int:
    """Return the token tile for ``tokens`` (1-wave within ``clusters * TILE``)."""
    if tokens > clusters * TILE:
        return TILE
    return next(tile for tile in TILES if tile * clusters >= tokens)


def _mat(dtype, *shape):
    return cute.runtime.make_fake_compact_tensor(
        dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=16
    )


def _compile(
    device_index: int,
    hidden: int,
    sinkhorn_iters: int,
    tile: int,
    use_pdl: bool,
) -> Callable:
    tokens = cute.sym_int()
    bf16 = cutlass.BFloat16
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    with torch.cuda.device(device_index):
        return cute.compile(
            MhcBoundaryKernel(hidden, sinkhorn_iters, tile, use_pdl),
            _mat(bf16, tokens, hidden),
            _mat(bf16, tokens, MULT, hidden),
            _mat(Float32, tokens, MULT),
            _mat(Float32, tokens, MULT * MULT),
            _mat(Float32, tokens, MULT),
            _mat(Int32, *boundary_weight_shape(hidden)),
            _mat(Float32, 3),
            _mat(Float32, MIXES),
            _mat(bf16, hidden),
            _mat(bf16, tokens, MULT, hidden),
            _mat(bf16, tokens, hidden),
            _mat(Float32, tokens, MULT),
            _mat(Float32, tokens, MULT),
            _mat(Float32, tokens, MULT * MULT),
            Float32(0.0),
            Float32(0.0),
            Float32(0.0),
            Float32(0.0),
            Float32(0.0),
            stream,
            options="--opt-level 2 --enable-tvm-ffi",
        )


_KERNELS: dict[tuple[int, int, int, int, bool], Callable] = {}


def warmup_mhc_boundary(
    hidden: int,
    sinkhorn_iters: int,
    *,
    use_pdl: bool,
    device: torch.device | str = "cuda",
) -> None:
    """Pre-compile all bucketed ``MhcBoundaryKernel`` tiles before capture."""
    if not MhcBoundaryKernel.supports(hidden, MULT, TILE) or sinkhorn_iters < 1:
        return
    dev = torch.device(device)
    device_index = dev.index if dev.index is not None else torch.cuda.current_device()
    for tile in TILES:
        key = (device_index, hidden, sinkhorn_iters, tile, use_pdl)
        if key not in _KERNELS:
            _KERNELS[key] = _compile(*key)


def boundary_inputs_eligible(
    x: torch.Tensor,
    residual: torch.Tensor,
    pre_mix: torch.Tensor,
    post_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    sinkhorn_iters: int,
    norm_weight: torch.Tensor,
) -> bool:
    """Return True when all tensor shapes, alignments, and dims fit ``mhc_boundary``."""
    if not x.is_cuda or x.ndim != 2 or residual.ndim != 3 or sinkhorn_iters < 1:
        return False
    tokens, hidden = x.shape
    if not 0 < tokens <= MAX_TOKENS:
        return False
    if not MhcBoundaryKernel.supports(hidden, MULT, TILE):
        return False
    expected = (
        (x, (tokens, hidden), torch.bfloat16),
        (residual, (tokens, MULT, hidden), torch.bfloat16),
        (pre_mix, (tokens, MULT), torch.float32),
        (post_mix, (tokens, MULT), torch.float32),
        (comb_res_mix, (tokens, MULT, MULT), torch.float32),
        (weight, boundary_weight_shape(hidden), torch.int32),
        (scale, (3,), torch.float32),
        (base, (MIXES,), torch.float32),
        (norm_weight, (hidden,), torch.bfloat16),
    )
    return not any(
        t.shape != shape
        or t.dtype != dtype
        or t.device != x.device
        or not t.is_contiguous()
        or t.data_ptr() % 16 != 0
        for t, shape, dtype in expected
    )


def mhc_shifted_post_pre_boundary(
    x: torch.Tensor,
    residual: torch.Tensor,
    pre_mix: torch.Tensor,
    post_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    hc_norm_eps: float,
    hc_eps: float,
    num_sinkhorn_iters: int,
    norm_weight: torch.Tensor,
    norm_eps: float,
    *,
    use_pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Launch ``MhcBoundaryKernel`` with packed weights ``prepare_boundary_weight(fn)``.

    Returns ``(new_residual, layer_input, new_pre, new_post, new_comb)`` with
    shapes ``([T,4,H], [T,H], [T,4], [T,4], [T,4,4])``, or ``None`` if inputs
    are ineligible or compilation would occur inside CUDA graph capture.
    """
    if not boundary_inputs_eligible(
        x,
        residual,
        pre_mix,
        post_mix,
        comb_res_mix,
        weight,
        scale,
        base,
        num_sinkhorn_iters,
        norm_weight,
    ):
        return None
    tokens, hidden = x.shape
    device_index = (
        x.device.index if x.device.index is not None else torch.cuda.current_device()
    )
    key = (
        device_index,
        hidden,
        int(num_sinkhorn_iters),
        boundary_tile(tokens, clusters=max_decode_wave_clusters(device_index)),
        use_pdl,
    )
    kernel = _KERNELS.get(key)
    if kernel is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        kernel = _compile(*key)
        _KERNELS[key] = kernel
    new_residual = torch.empty_like(residual)
    layer_input = torch.empty(tokens, hidden, dtype=torch.bfloat16, device=x.device)
    new_pre = torch.empty(tokens, MULT, dtype=torch.float32, device=x.device)
    new_post = torch.empty(tokens, MULT, dtype=torch.float32, device=x.device)
    new_comb = torch.empty(tokens, MULT, MULT, dtype=torch.float32, device=x.device)
    kernel(
        x,
        residual,
        post_mix,
        comb_res_mix.view(tokens, MULT * MULT),
        pre_mix,
        weight,
        scale,
        base,
        norm_weight,
        new_residual,
        layer_input,
        new_pre,
        new_post,
        new_comb.view(tokens, MULT * MULT),
        float(hc_norm_eps),
        float(hc_eps),
        float(hc_eps),
        2.0,
        float(norm_eps),
    )
    return new_residual, layer_input, new_pre, new_post, new_comb
