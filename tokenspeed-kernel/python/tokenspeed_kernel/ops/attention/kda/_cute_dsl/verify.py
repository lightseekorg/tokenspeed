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

"""KDA speculative target verify for Blackwell, raw-gate replay payload.

One launch per layer does the whole verify after the gate GEMM: the causal
convolution (+SiLU) of the raw projections from the committed window, q/k L2
normalization, the per-token decay / contract / delta update, and the output
``q . h``; it also writes the replay payload's raw QKV and beta rows. Nothing
per position is stored: the replay recomputes the accepted prefix.

Persistent CTAs walk (request, head) items and prefetch the next item's state
tile and inputs into a double-buffered shared stage (bulk copies completing on
an mbarrier). Lane ``4g + q`` of warp ``w`` owns V rows ``32w + g + 8c`` and K
chunks ``q + 4j``, as in the batched replay, so both contractions are in-thread
packed FMAs plus a two-step butterfly.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import make_ptr

__all__ = ["kda_fused_verify"]

HEAD_DIM = 128
K = HEAD_DIM
_GMEM = cute.AddressSpace.gmem


class KdaFusedVerify:
    def __init__(
        self,
        draft_tokens: int,
        num_heads: int,
        state_dtype,
        qkv_stride: int,
        beta_stride: int,
        gate_stride: int,
        state_stride: int,
        payload_qkv_stride: int,
        payload_beta_stride: int,
        lower_bound: float,
    ):
        self.T = draft_tokens
        self.HV = num_heads
        self.state_dtype = state_dtype
        self.esz = state_dtype.width // 8
        self.s_qkv = qkv_stride
        self.s_beta = beta_stride
        self.s_g = gate_stride
        self.s_state = state_stride
        self.s_pay_qkv = payload_qkv_stride
        self.s_pay_beta = payload_beta_stride
        self.lb = lower_bound
        self.scale = K**-0.5

    @cute.jit
    def __call__(
        self,
        qkv: cute.Pointer,
        conv_w: cute.Pointer,
        conv_pool: cute.Pointer,
        conv_stride: Int64,
        gate: cute.Pointer,
        beta: cute.Pointer,
        a_log: cute.Pointer,
        dt_bias: cute.Pointer,
        state: cute.Pointer,
        read_indices: cute.Pointer,
        out: cute.Pointer,
        pay_qkv: cute.Pointer,
        pay_beta: cute.Pointer,
        batch: Int32,
        ctas: Int32,
        stream: CUstream,
    ):
        self.kernel(
            qkv,
            conv_w,
            conv_pool,
            conv_stride,
            gate,
            beta,
            a_log,
            dt_bias,
            state,
            read_indices,
            out,
            pay_qkv,
            pay_beta,
            batch,
        ).launch(grid=(ctas, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.jit
    def _bulk(
        self,
        dtype: cutlass.Constexpr,
        nbytes: cutlass.Constexpr,
        src: Int64,
        dst: cute.Tensor,
        mbar,
    ):
        atom = cute.make_copy_atom(
            cpasync.CopyBulkG2SOp(), dtype, num_bits_per_copy=nbytes * 8
        )
        gsrc = cute.make_tensor(
            cute.make_ptr(dtype, src, _GMEM, assumed_align=16),
            cute.make_layout(nbytes * 8 // dtype.width),
        )
        s_, d_ = cpasync.group_bulk_copy_modes(gsrc, dst)
        cute.copy(atom, s_, d_, mbar_ptr=mbar)

    @cute.jit
    def _prefetch(
        self,
        item: Int32,
        b: Int32,
        total: Int32,
        qkv: cute.Pointer,
        conv_w: cute.Pointer,
        conv_pool: cute.Pointer,
        conv_stride: Int64,
        gate: cute.Pointer,
        beta: cute.Pointer,
        a_log: cute.Pointer,
        dt_bias: cute.Pointer,
        state: cute.Pointer,
        mRead: cute.Tensor,
        sState,
        sQ,
        sK,
        sV,
        sG,
        sBias,
        sBeta,
        sAlog,
        sCq,
        sCk,
        sCv,
        sWq,
        sWk,
        sWv,
        atom4,
        mbar,
        tid: Int32,
    ):
        """Stage buffer ``b`` with item ``item``'s inputs, if it exists."""
        T, HV = self.T, self.HV
        P = HV * K
        SD = self.state_dtype
        if item < total:
            hv = item % HV
            n = item // HV
            rows = Int64(n) * T
            if tid == 0:
                read_page = Int64(mRead[n])
                bar = mbar + b
                inputs = T * (4 * K * 2) + K * 4 + 3 * K * 4 * 2
                a_qkv = qkv.toint()
                a_w = conv_w.toint()
                if read_page >= 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        bar, inputs + K * K * self.esz + 3 * K * 3 * 2
                    )
                    self._bulk(
                        SD,
                        K * K * self.esz,
                        state.toint()
                        + (read_page * self.s_state + Int64(hv * K * K)) * self.esz,
                        sState[b, None],
                        bar,
                    )
                    window = conv_pool.toint() + read_page * conv_stride * 2
                    self._bulk(
                        BFloat16,
                        K * 3 * 2,
                        window + Int64(hv * K * 6),
                        sCq[b, None],
                        bar,
                    )
                    self._bulk(
                        BFloat16,
                        K * 3 * 2,
                        window + Int64((P + hv * K) * 6),
                        sCk[b, None],
                        bar,
                    )
                    self._bulk(
                        BFloat16,
                        K * 3 * 2,
                        window + Int64((2 * P + hv * K) * 6),
                        sCv[b, None],
                        bar,
                    )
                else:
                    cute.arch.mbarrier_arrive_and_expect_tx(bar, inputs)
                self._bulk(
                    BFloat16, K * 4 * 2, a_w + Int64(hv * K * 8), sWq[b, None], bar
                )
                self._bulk(
                    BFloat16,
                    K * 4 * 2,
                    a_w + Int64((P + hv * K) * 8),
                    sWk[b, None],
                    bar,
                )
                self._bulk(
                    BFloat16,
                    K * 4 * 2,
                    a_w + Int64((2 * P + hv * K) * 8),
                    sWv[b, None],
                    bar,
                )
                self._bulk(
                    Float32,
                    K * 4,
                    dt_bias.toint() + Int64(hv * K * 4),
                    sBias[b, None],
                    bar,
                )
                for i in cutlass.range_constexpr(T):
                    row = rows + i
                    base = a_qkv + row * (self.s_qkv * 2)
                    self._bulk(
                        BFloat16, K * 2, base + Int64(hv * K * 2), sQ[b, i, None], bar
                    )
                    self._bulk(
                        BFloat16,
                        K * 2,
                        base + Int64((P + hv * K) * 2),
                        sK[b, i, None],
                        bar,
                    )
                    self._bulk(
                        BFloat16,
                        K * 2,
                        base + Int64((2 * P + hv * K) * 2),
                        sV[b, i, None],
                        bar,
                    )
                    self._bulk(
                        BFloat16,
                        K * 2,
                        gate.toint() + (row * self.s_g + Int64(hv * K)) * 2,
                        sG[b, i, None],
                        bar,
                    )
            # Beta rows and A_log are too narrow for a bulk copy: one 4 B
            # cp.async each, counted on the same mbarrier.
            if tid < T:
                src = cute.make_tensor(
                    cute.make_ptr(
                        BFloat16,
                        beta.toint()
                        + ((rows + tid) * self.s_beta + Int64(hv - hv % 2)) * 2,
                        _GMEM,
                        assumed_align=4,
                    ),
                    cute.make_layout(2),
                )
                cute.copy(atom4, src, sBeta[b, tid, None])
                cute.arch.cp_async_mbarrier_arrive_noinc(mbar + b)
            if tid == T:
                src = cute.make_tensor(
                    cute.make_ptr(
                        BFloat16, a_log.toint() + Int64(hv * 4), _GMEM, assumed_align=4
                    ),
                    cute.make_layout(2),
                )
                dst = cute.make_tensor(
                    cute.recast_ptr(sAlog.iterator + b, dtype=BFloat16),
                    cute.make_layout(2),
                )
                cute.copy(atom4, src, dst)
                cute.arch.cp_async_mbarrier_arrive_noinc(mbar + b)

    @cute.jit
    def _conv(self, sX, sC, sW, b: Int32, ch: Int32, read_ok, rOut):
        """Causal 4-tap convolution + SiLU of channel ``ch`` for every token."""
        w0 = sW[b, ch * 4 + 0].to(Float32)
        w1 = sW[b, ch * 4 + 1].to(Float32)
        w2 = sW[b, ch * 4 + 2].to(Float32)
        w3 = sW[b, ch * 4 + 3].to(Float32)
        s0 = Float32(0.0)
        s1 = Float32(0.0)
        s2 = Float32(0.0)
        if read_ok:
            s0 = sC[b, ch * 3 + 0].to(Float32)
            s1 = sC[b, ch * 3 + 1].to(Float32)
            s2 = sC[b, ch * 3 + 2].to(Float32)
        for i in cutlass.range_constexpr(self.T):
            x = sX[b, i, ch].to(Float32)
            acc = x * w3 + s0 * w0 + s1 * w1 + s2 * w2
            rOut[i] = acc * cute.math.div(
                Float32(1.0),
                Float32(1.0) + cute.math.exp(-acc, fastmath=True),
                full=True,
            )
            s0 = s1
            s1 = s2
            s2 = x

    @cute.kernel
    def kernel(
        self,
        qkv: cute.Pointer,
        conv_w: cute.Pointer,
        conv_pool: cute.Pointer,
        conv_stride: Int64,
        gate: cute.Pointer,
        beta: cute.Pointer,
        a_log: cute.Pointer,
        dt_bias: cute.Pointer,
        state: cute.Pointer,
        read_indices: cute.Pointer,
        out: cute.Pointer,
        pay_qkv: cute.Pointer,
        pay_beta: cute.Pointer,
        batch: Int32,
    ):
        T, HV = self.T, self.HV
        P = HV * K
        SD = self.state_dtype
        cta, _, _ = cute.arch.block_idx()
        ctas, _, _ = cute.arch.grid_dim()
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        g = lane // 4
        q = lane % 4
        total = batch * HV
        rows_total = batch * T
        mRead = cute.make_tensor(read_indices, cute.make_layout(batch))
        mOut = cute.make_tensor(out, cute.make_layout(rows_total * HV * K))
        mPayQkv = cute.make_tensor(
            pay_qkv, cute.make_layout(rows_total * self.s_pay_qkv)
        )
        mPayBeta = cute.make_tensor(
            pay_beta, cute.make_layout(rows_total * self.s_pay_beta)
        )

        smem = cutlass.utils.SmemAllocator()
        sState = smem.allocate_tensor(
            SD, cute.make_layout((2, K * K), stride=(K * K, 1)), byte_alignment=128
        )
        # [buffer, row, chunk of 8, element]
        sFrag = cute.make_tensor(
            sState.iterator,
            cute.make_layout((2, K, K // 8, 8), stride=(K * K, K, 8, 1)),
        )
        token_layout = cute.make_layout((2, T, K), stride=(T * K, K, 1))
        sQ = smem.allocate_tensor(BFloat16, token_layout, byte_alignment=16)
        sK = smem.allocate_tensor(BFloat16, token_layout, byte_alignment=16)
        sV = smem.allocate_tensor(BFloat16, token_layout, byte_alignment=16)
        sG = smem.allocate_tensor(BFloat16, token_layout, byte_alignment=16)
        window_layout = cute.make_layout((2, K * 3), stride=(K * 3, 1))
        sCq = smem.allocate_tensor(BFloat16, window_layout, byte_alignment=16)
        sCk = smem.allocate_tensor(BFloat16, window_layout, byte_alignment=16)
        sCv = smem.allocate_tensor(BFloat16, window_layout, byte_alignment=16)
        tap_layout = cute.make_layout((2, K * 4), stride=(K * 4, 1))
        sWq = smem.allocate_tensor(BFloat16, tap_layout, byte_alignment=16)
        sWk = smem.allocate_tensor(BFloat16, tap_layout, byte_alignment=16)
        sWv = smem.allocate_tensor(BFloat16, tap_layout, byte_alignment=16)
        sBias = smem.allocate_tensor(
            Float32, cute.make_layout((2, K), stride=(K, 1)), byte_alignment=16
        )
        sBeta = smem.allocate_tensor(
            BFloat16,
            cute.make_layout((2, T, 2), stride=(2 * T, 2, 1)),
            byte_alignment=16,
        )
        sAlog = smem.allocate_tensor(
            Float32, cute.make_layout((2, 1), stride=(1, 1)), byte_alignment=16
        )
        # Normalized q * scale, k and per-channel decay per token: [token, chunk, 8]
        chunked = cute.make_layout((T, K // 8, 8), stride=(K, 8, 1))
        sNq = smem.allocate_tensor(Float32, chunked, byte_alignment=16)
        sNk = smem.allocate_tensor(Float32, chunked, byte_alignment=16)
        sNa = smem.allocate_tensor(Float32, chunked, byte_alignment=16)
        flat = cute.make_layout((T, K), stride=(K, 1))
        fNq = cute.make_tensor(sNq.iterator, flat)
        fNk = cute.make_tensor(sNk.iterator, flat)
        fNa = cute.make_tensor(sNa.iterator, flat)
        sSS = smem.allocate_tensor(
            Float32,
            cute.make_layout((2, 4, T), stride=(4 * T, T, 1)),
            byte_alignment=16,
        )
        mbar = smem.allocate_array(Int64, 2, byte_alignment=8)
        atom4 = cute.make_copy_atom(cpasync.CopyG2SOp(), BFloat16, num_bits_per_copy=32)

        if tid == 0:
            # One expect-tx arrival, one cp.async arrival per beta row and one
            # for A_log.
            cute.arch.mbarrier_init(mbar, 2 + T)
            cute.arch.mbarrier_init(mbar + 1, 2 + T)
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        self._prefetch(
            cta, Int32(0), total, qkv, conv_w, conv_pool, conv_stride, gate, beta,
            a_log, dt_bias, state, mRead, sState, sQ, sK, sV, sG, sBias, sBeta, sAlog,
            sCq, sCk, sCv, sWq, sWk, sWv, atom4, mbar, tid,
        )  # fmt: skip

        it = Int32(0)
        for item in cutlass.range(cta, total, ctas):
            b = it % 2
            self._prefetch(
                item + ctas, 1 - b, total, qkv, conv_w, conv_pool, conv_stride, gate,
                beta, a_log, dt_bias, state, mRead, sState, sQ, sK, sV, sG, sBias,
                sBeta, sAlog, sCq, sCk, sCv, sWq, sWk, sWv, atom4, mbar, tid,
            )  # fmt: skip
            hv = item % HV
            n = item // HV
            read_ok = Int64(mRead[n]) >= 0
            cute.arch.mbarrier_wait(mbar + b, (it // 2) % 2)

            # Replay payload: this head's raw q|k|v channels and beta.
            for i in cutlass.range_constexpr(T):
                row = (Int64(n) * T + i) * self.s_pay_qkv
                mPayQkv[row + hv * K + tid] = sQ[b, i, tid]
                mPayQkv[row + P + hv * K + tid] = sK[b, i, tid]
                mPayQkv[row + 2 * P + hv * K + tid] = sV[b, i, tid]
            if tid < T:
                mPayBeta[(Int64(n) * T + tid) * self.s_pay_beta + hv] = sBeta[
                    b, tid, hv % 2
                ]

            # q/k side: thread t owns channel t.
            exp_a = cute.math.exp(sAlog[b, 0], fastmath=True)
            bias = sBias[b, tid]
            rq = cute.make_rmem_tensor(T, Float32)
            rk = cute.make_rmem_tensor(T, Float32)
            rsq = cute.make_rmem_tensor(T, Float32)
            rsk = cute.make_rmem_tensor(T, Float32)
            self._conv(sQ, sCq, sWq, b, tid, read_ok, rq)
            self._conv(sK, sCk, sWk, b, tid, read_ok, rk)
            for i in cutlass.range_constexpr(T):
                rsq[i] = cute.arch.warp_reduction_sum(rq[i] * rq[i])
                rsk[i] = cute.arch.warp_reduction_sum(rk[i] * rk[i])
                decay = self.lb * cute.math.div(
                    Float32(1.0),
                    Float32(1.0)
                    + cute.math.exp(
                        -(exp_a * (sG[b, i, tid].to(Float32) + bias)), fastmath=True
                    ),
                    full=True,
                )
                fNa[i, tid] = cute.math.exp(decay, fastmath=True)
            if lane == 0:
                for i in cutlass.range_constexpr(T):
                    sSS[0, warp, i] = rsq[i]
                    sSS[1, warp, i] = rsk[i]
            cute.arch.sync_threads()
            for i in cutlass.range_constexpr(T):
                sq = sSS[0, 0, i] + sSS[0, 1, i] + sSS[0, 2, i] + sSS[0, 3, i]
                sk = sSS[1, 0, i] + sSS[1, 1, i] + sSS[1, 2, i] + sSS[1, 3, i]
                fNq[i, tid] = (
                    rq[i]
                    * cute.math.rsqrt(sq + Float32(1e-6), fastmath=True)
                    * self.scale
                )
                fNk[i, tid] = rk[i] * cute.math.rsqrt(sk + Float32(1e-6), fastmath=True)

            # v side: lane 4g+q computes row 32w+g+8q; the group shares its four rows.
            rV = cute.make_rmem_tensor(cute.make_layout((T, 4), stride=(4, 1)), Float32)
            rB = cute.make_rmem_tensor(T, Float32)
            rv = cute.make_rmem_tensor(T, Float32)
            vq = 32 * warp + g + 8 * q
            self._conv(sV, sCv, sWv, b, vq, read_ok, rv)
            for i in cutlass.range_constexpr(T):
                rB[i] = cute.math.div(
                    Float32(1.0),
                    Float32(1.0)
                    + cute.math.exp(-sBeta[b, i, hv % 2].to(Float32), fastmath=True),
                    full=True,
                )
                for c in cutlass.range_constexpr(4):
                    rV[i, c] = cute.arch.shuffle_sync(rv[i], 4 * g + c)

            # State fragment [row c, chunk j, element]; a fresh page reads zeros.
            frag = cute.make_layout((4, 4, 8), stride=(32, 8, 1))
            rH = cute.make_rmem_tensor(frag, Float32)
            rIO = cute.make_rmem_tensor(frag, SD)
            if not read_ok:
                rIO.fill(0.0)
                for c in cutlass.range_constexpr(4):
                    for j in cutlass.range_constexpr(4):
                        cute.autovec_copy(
                            rIO[c, j, None],
                            sFrag[b, 32 * warp + g + 8 * c, q + 4 * j, None],
                        )
            for c in cutlass.range_constexpr(4):
                for j in cutlass.range_constexpr(4):
                    cute.autovec_copy(
                        sFrag[b, 32 * warp + g + 8 * c, q + 4 * j, None],
                        rIO[c, j, None],
                    )
            if cutlass.const_expr(SD.width == 16):
                # BF16 -> FP32 is a 16-bit shift of each half-word.
                wIO = cute.recast_tensor(rIO, Uint32)
                wH = cute.recast_tensor(rH, Uint32)
                for c in cutlass.range_constexpr(4):
                    for j in cutlass.range_constexpr(4):
                        for e in cutlass.range_constexpr(4):
                            word = wIO[c, j, e]
                            wH[c, j, 2 * e] = word << 16
                            wH[c, j, 2 * e + 1] = word & Uint32(0xFFFF0000)
            else:
                rH.store(rIO.load().to(Float32))
            cute.arch.sync_threads()

            rK = cute.make_rmem_tensor(cute.make_layout((4, 8), stride=(8, 1)), Float32)
            rX = cute.make_rmem_tensor(cute.make_layout((4, 8), stride=(8, 1)), Float32)
            rP = cute.make_rmem_tensor(cute.make_layout((4, 2), stride=(2, 1)), Float32)
            rU = cute.make_rmem_tensor(4, Float32)
            zero = (Float32(0.0), Float32(0.0))
            for i in cutlass.range_constexpr(T):
                # Decay, then contract with k.
                for j in cutlass.range_constexpr(4):
                    cute.autovec_copy(sNk[i, q + 4 * j, None], rK[j, None])
                    cute.autovec_copy(sNa[i, q + 4 * j, None], rX[j, None])
                rP.fill(0.0)
                for c in cutlass.range_constexpr(4):
                    for j in cutlass.range_constexpr(4):
                        for e in cutlass.range_constexpr(0, 8, 2):
                            h = cute.arch.fma_packed_f32x2(
                                (rH[c, j, e], rH[c, j, e + 1]),
                                (rX[j, e], rX[j, e + 1]),
                                zero,
                            )
                            rH[c, j, e] = h[0]
                            rH[c, j, e + 1] = h[1]
                            p = cute.arch.fma_packed_f32x2(
                                h, (rK[j, e], rK[j, e + 1]), (rP[c, 0], rP[c, 1])
                            )
                            rP[c, 0] = p[0]
                            rP[c, 1] = p[1]
                for c in cutlass.range_constexpr(4):
                    dot = rP[c, 0] + rP[c, 1]
                    dot = dot + cute.arch.shuffle_sync_bfly(dot, offset=1)
                    dot = dot + cute.arch.shuffle_sync_bfly(dot, offset=2)
                    rU[c] = (rV[i, c] - dot) * rB[i]
                # Delta update.
                for c in cutlass.range_constexpr(4):
                    for j in cutlass.range_constexpr(4):
                        for e in cutlass.range_constexpr(0, 8, 2):
                            h = cute.arch.fma_packed_f32x2(
                                (rU[c], rU[c]),
                                (rK[j, e], rK[j, e + 1]),
                                (rH[c, j, e], rH[c, j, e + 1]),
                            )
                            rH[c, j, e] = h[0]
                            rH[c, j, e + 1] = h[1]
                # Output: contract with q.
                for j in cutlass.range_constexpr(4):
                    cute.autovec_copy(sNq[i, q + 4 * j, None], rX[j, None])
                rP.fill(0.0)
                for c in cutlass.range_constexpr(4):
                    for j in cutlass.range_constexpr(4):
                        for e in cutlass.range_constexpr(0, 8, 2):
                            p = cute.arch.fma_packed_f32x2(
                                (rH[c, j, e], rH[c, j, e + 1]),
                                (rX[j, e], rX[j, e + 1]),
                                (rP[c, 0], rP[c, 1]),
                            )
                            rP[c, 0] = p[0]
                            rP[c, 1] = p[1]
                mine = Float32(0.0)
                for c in cutlass.range_constexpr(4):
                    o = rP[c, 0] + rP[c, 1]
                    o = o + cute.arch.shuffle_sync_bfly(o, offset=1)
                    o = o + cute.arch.shuffle_sync_bfly(o, offset=2)
                    if q == c:
                        mine = o
                mOut[((Int64(n) * T + i) * HV + hv) * K + vq] = mine.to(BFloat16)
            cute.arch.sync_threads()
            it = it + 1


_COMPILED: dict = {}


@functools.cache
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _check_row_operand(name: str, tensor: torch.Tensor, rows: int, width: int) -> None:
    if tensor.dtype != torch.bfloat16 or tensor.dim() != 2 or tensor.stride(-1) != 1:
        raise ValueError(
            f"{name} must be a 2-D bfloat16 tensor with unit column stride"
        )
    if tensor.shape[0] < rows or tensor.shape[1] < width:
        raise ValueError(
            f"{name} must cover [{rows}, {width}], got {tuple(tensor.shape)}"
        )


def kda_fused_verify(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    replay_mixed_qkv: torch.Tensor,
    replay_gate: torch.Tensor,
    replay_beta: torch.Tensor,
    *,
    num_heads: int,
    draft_token_num: int,
    lower_bound: float,
) -> torch.Tensor:
    """Verify a draft window per request and fill the raw-gate replay payload.

    The raw gate ``f_a_out @ f_b_weight.T`` is written straight into
    ``replay_gate``; one CuTe launch then computes the output and copies the
    raw QKV and beta rows into the payload. States are only read.

    Args:
        mixed_qkv: ``[rows, >= 3 * P]`` raw q|k|v projections, ``P = num_heads * 128``;
            rows are ``batch * draft_token_num`` tokens, request-major.
        conv_weights: ``[3 * P, 4]`` causal convolution taps, oldest first.
        conv_states: ``[pages, >= 3 * P, 3]`` committed convolution windows
            (q|k|v channels first).
        f_a_out: ``[rows, D]`` low-rank gate activation.
        f_b_weight: ``[P, D]`` second gate projection.
        beta_logits: ``[rows, >= num_heads]`` raw beta (sigmoid in-kernel).
        A_log: ``[num_heads]`` FP32 decay parameter.
        dt_bias: ``[P]`` FP32 gate bias.
        state_pool: ``[pages, >= num_heads, 128, 128]`` V-major BF16 or FP32 states.
        read_indices: ``[batch]`` int32 committed pages (-1 = fresh request).
        replay_mixed_qkv: ``[rows, 3 * P]`` payload view for the raw QKV rows.
        replay_gate: ``[rows, P]`` BF16 payload view for the raw gate.
        replay_beta: ``[rows, num_heads]`` payload view for the raw beta.
        num_heads: Local KDA heads.
        draft_token_num: Window tokens per request.
        lower_bound: Safe-gate lower bound.

    Returns:
        ``[rows, num_heads, 128]`` BF16 attention output.
    """
    T, HV = draft_token_num, num_heads
    P = HV * K
    batch = read_indices.numel()
    rows = batch * T
    out = torch.empty(rows, HV, K, dtype=torch.bfloat16, device=mixed_qkv.device)
    if batch == 0:
        return out
    if mixed_qkv.shape[0] != rows:
        raise ValueError(f"mixed_qkv must hold {rows} rows, got {mixed_qkv.shape[0]}")
    _check_row_operand("mixed_qkv", mixed_qkv, rows, 3 * P)
    _check_row_operand("beta_logits", beta_logits, rows, HV)
    _check_row_operand("replay_mixed_qkv", replay_mixed_qkv, rows, 3 * P)
    _check_row_operand("replay_gate", replay_gate, rows, P)
    _check_row_operand("replay_beta", replay_beta, rows, HV)
    # Bulk copies need 16 B aligned rows; beta is read two heads per 4 B copy.
    for name, tensor, unit in (
        ("mixed_qkv", mixed_qkv, 8),
        ("replay_gate", replay_gate, 8),
        ("beta_logits", beta_logits, 2),
    ):
        if tensor.stride(0) % unit or tensor.data_ptr() % (2 * unit):
            raise ValueError(f"{name} rows must be {2 * unit} B aligned")
    if conv_weights.shape != (3 * P, 4) or not conv_weights.is_contiguous():
        raise ValueError(f"conv_weights must be a contiguous [{3 * P}, 4] tensor")
    if (
        conv_states.shape[1] < 3 * P
        or conv_states.shape[2] != 3
        or conv_states.stride()[1:] != (3, 1)
    ):
        raise ValueError(
            f"conv_states must be [pages, >= {3 * P}, 3] with contiguous windows, got "
            f"shape {tuple(conv_states.shape)} strides {conv_states.stride()}"
        )
    if (
        state_pool.shape[1] < HV
        or state_pool.shape[2:] != (K, K)
        or state_pool.stride()[1:] != (K * K, K, 1)
    ):
        raise ValueError(
            f"state_pool must be [pages, >= {HV}, {K}, {K}] with contiguous head tiles, "
            f"got shape {tuple(state_pool.shape)} strides {state_pool.stride()}"
        )
    if read_indices.dtype != torch.int32 or not read_indices.is_contiguous():
        raise ValueError("read_indices must be a contiguous int32 tensor")
    if A_log.dtype != torch.float32 or dt_bias.dtype != torch.float32:
        raise ValueError("A_log and dt_bias must be FP32")

    torch.mm(f_a_out[:rows], f_b_weight.t(), out=replay_gate[:rows, :P])

    dtype = {torch.bfloat16: BFloat16, torch.float32: Float32}[state_pool.dtype]
    key = (
        T,
        HV,
        state_pool.dtype,
        mixed_qkv.stride(0),
        beta_logits.stride(0),
        replay_gate.stride(0),
        state_pool.stride(0),
        replay_mixed_qkv.stride(0),
        replay_beta.stride(0),
        float(lower_bound),
    )
    ptrs = (
        make_ptr(BFloat16, mixed_qkv.data_ptr(), _GMEM, assumed_align=16),
        make_ptr(BFloat16, conv_weights.data_ptr(), _GMEM, assumed_align=16),
        make_ptr(BFloat16, conv_states.data_ptr(), _GMEM, assumed_align=16),
        Int64(conv_states.stride(0)),
        make_ptr(BFloat16, replay_gate.data_ptr(), _GMEM, assumed_align=16),
        make_ptr(BFloat16, beta_logits.data_ptr(), _GMEM, assumed_align=4),
        make_ptr(Float32, A_log.data_ptr(), _GMEM, assumed_align=4),
        make_ptr(Float32, dt_bias.data_ptr(), _GMEM, assumed_align=16),
        make_ptr(dtype, state_pool.data_ptr(), _GMEM, assumed_align=16),
        make_ptr(Int32, read_indices.data_ptr(), _GMEM, assumed_align=4),
        make_ptr(BFloat16, out.data_ptr(), _GMEM, assumed_align=16),
        make_ptr(BFloat16, replay_mixed_qkv.data_ptr(), _GMEM, assumed_align=2),
        make_ptr(BFloat16, replay_beta.data_ptr(), _GMEM, assumed_align=2),
    )
    # Two BF16-state CTAs fit one SM's shared memory; an FP32 state tile does not.
    ctas_per_sm = 2 if state_pool.dtype == torch.bfloat16 else 1
    ctas = min(ctas_per_sm * _sm_count(mixed_qkv.device.index), batch * HV)
    stream = CUstream(torch.cuda.current_stream(mixed_qkv.device).cuda_stream)
    compiled = _COMPILED.get(key)
    if compiled is None:
        kernel = KdaFusedVerify(
            T,
            HV,
            dtype,
            mixed_qkv.stride(0),
            beta_logits.stride(0),
            replay_gate.stride(0),
            state_pool.stride(0),
            replay_mixed_qkv.stride(0),
            replay_beta.stride(0),
            float(lower_bound),
        )
        compiled = cute.compile(kernel, *ptrs, Int32(batch), Int32(ctas), stream)
        _COMPILED[key] = compiled
    compiled(*ptrs, Int32(batch), Int32(ctas), stream)
    return out
