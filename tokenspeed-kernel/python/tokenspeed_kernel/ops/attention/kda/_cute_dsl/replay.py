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

"""Batched KDA replay recurrence for Blackwell.

Persistent CTAs walk (layer, request, head) items. Each item's state tile and
its convolution, key, value, raw gate and beta inputs are prefetched into a
double-buffered shared-memory stage one item ahead (bulk copies completing on
an mbarrier). The decay gate ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``
is applied in-kernel to the BF16 raw gate the verify wrote into the payload. Within an item, warp ``w`` owns V rows ``[32w, 32w + 32)`` and
lane ``4g + q`` owns rows ``32w + g + 8c`` and the K chunks ``q + 4j`` (eight
elements each), so a token's contraction over K is in-thread packed FMAs plus
a two-step butterfly across the four lanes of a row group. The per-token math
and its order follow the Triton walk: decay, contract, delta update.
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

__all__ = ["kda_batched_replay"]

HEAD_DIM = 128
CTAS_PER_SM = 2
# Descriptor columns: qkv, conv_w, conv_pool, f_a, f_b, beta, A, dt, state, gate.
_QKV, _CONV_W, _CONV, _BETA, _A_LOG, _DT_BIAS, _STATE, _GATE = 0, 1, 2, 5, 6, 7, 8, 9
_GMEM = cute.AddressSpace.gmem


class KdaBatchedReplay:
    def __init__(
        self,
        draft_tokens: int,
        num_heads: int,
        state_dtype,
        qkv_stride: int,
        conv_stride: int,
        beta_stride: int,
        state_stride: int,
        gate_stride: int,
        lower_bound: float,
    ):
        self.T = draft_tokens
        self.HV = num_heads
        self.state_dtype = state_dtype
        self.esz = state_dtype.width // 8
        self.s_qkv = qkv_stride
        self.s_conv = conv_stride
        self.s_beta = beta_stride
        self.s_state = state_stride
        self.s_gate = gate_stride
        self.lb = lower_bound

    @cute.jit
    def __call__(
        self,
        addresses: cute.Pointer,
        group_indices: cute.Pointer,
        read_indices: cute.Pointer,
        write_indices: cute.Pointer,
        accepted_length: cute.Pointer,
        layers: Int32,
        batch: Int32,
        ctas: Int32,
        stream: CUstream,
    ):
        self.kernel(
            addresses,
            group_indices,
            read_indices,
            write_indices,
            accepted_length,
            layers,
            batch,
        ).launch(grid=(ctas, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.jit
    def _decode(self, item: Int32, batch: Int32, inv_batch: Float32):
        """Item -> (layer, request, head) without an integer divide by batch."""
        hv = item % self.HV
        nl = item // self.HV
        layer = Int32(Float32(nl) * inv_batch)
        if layer * batch > nl:
            layer = layer - 1
        if (layer + 1) * batch <= nl:
            layer = layer + 1
        return layer, nl - layer * batch, hv

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
        batch: Int32,
        inv_batch: Float32,
        mAddr: cute.Tensor,
        mGroup: cute.Tensor,
        mRead: cute.Tensor,
        sState,
        sXk,
        sXv,
        sG,
        sCk,
        sCv,
        sWk,
        sWv,
        sBeta,
        sBias,
        sAlog,
        beta_atom,
        mbar,
        tid: Int32,
    ):
        T, HV, K = self.T, self.HV, HEAD_DIM
        P = HV * K
        SD = self.state_dtype
        if item < total:
            layer, n, hv = self._decode(item, batch, inv_batch)
            nl = Int64(n)
            if tid == 0:
                read_page = Int64(mRead[mGroup[layer] * batch + n])
                a_qkv = Int64(mAddr[layer, _QKV])
                a_w = Int64(mAddr[layer, _CONV_W])
                a_gate = Int64(mAddr[layer, _GATE])
                bar = mbar + b
                inputs = T * (3 * K * 2) + 2 * K * 4 * 2 + K * 4
                if read_page >= 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        bar, inputs + K * K * self.esz + 2 * K * 3 * 2
                    )
                    self._bulk(
                        SD,
                        K * K * self.esz,
                        Int64(mAddr[layer, _STATE])
                        + (read_page * self.s_state + Int64(hv * K * K)) * self.esz,
                        sState[b, None],
                        bar,
                    )
                    conv = Int64(mAddr[layer, _CONV]) + read_page * self.s_conv * 2
                    self._bulk(
                        BFloat16,
                        K * 3 * 2,
                        conv + Int64((P + hv * K) * 6),
                        sCk[b, None],
                        bar,
                    )
                    self._bulk(
                        BFloat16,
                        K * 3 * 2,
                        conv + Int64((2 * P + hv * K) * 6),
                        sCv[b, None],
                        bar,
                    )
                else:
                    cute.arch.mbarrier_arrive_and_expect_tx(bar, inputs)
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
                for i in cutlass.range_constexpr(T):
                    row = nl * T + i
                    self._bulk(
                        BFloat16,
                        K * 2,
                        a_qkv + (row * self.s_qkv + Int64(P + hv * K)) * 2,
                        sXk[b, i, None],
                        bar,
                    )
                    self._bulk(
                        BFloat16,
                        K * 2,
                        a_qkv + (row * self.s_qkv + Int64(2 * P + hv * K)) * 2,
                        sXv[b, i, None],
                        bar,
                    )
                    self._bulk(
                        BFloat16,
                        K * 2,
                        a_gate + (row * self.s_gate + Int64(hv * K)) * 2,
                        sG[b, i, None],
                        bar,
                    )
                self._bulk(
                    Float32,
                    K * 4,
                    Int64(mAddr[layer, _DT_BIAS]) + Int64(hv * K * 4),
                    sBias[b, None],
                    bar,
                )
            # Beta rows and A_log are too narrow for a bulk copy: one 4 B
            # cp.async each, counted on the same mbarrier.
            if tid < T:
                row = nl * T + tid
                src = cute.make_tensor(
                    cute.make_ptr(
                        BFloat16,
                        Int64(mAddr[layer, _BETA])
                        + (row * self.s_beta + Int64(hv - hv % 2)) * 2,
                        _GMEM,
                        assumed_align=4,
                    ),
                    cute.make_layout(2),
                )
                cute.copy(beta_atom, src, sBeta[b, tid, None])
                cute.arch.cp_async_mbarrier_arrive_noinc(mbar + b)
            if tid == T:
                src = cute.make_tensor(
                    cute.make_ptr(
                        BFloat16,
                        Int64(mAddr[layer, _A_LOG]) + Int64(hv * 4),
                        _GMEM,
                        assumed_align=4,
                    ),
                    cute.make_layout(2),
                )
                dst = cute.make_tensor(
                    cute.recast_ptr(sAlog.iterator + b, dtype=BFloat16),
                    cute.make_layout(2),
                )
                cute.copy(beta_atom, src, dst)
                cute.arch.cp_async_mbarrier_arrive_noinc(mbar + b)

    @cute.kernel
    def kernel(
        self,
        addresses: cute.Pointer,
        group_indices: cute.Pointer,
        read_indices: cute.Pointer,
        write_indices: cute.Pointer,
        accepted_length: cute.Pointer,
        layers: Int32,
        batch: Int32,
    ):
        T, HV, K = self.T, self.HV, HEAD_DIM
        SD = self.state_dtype
        cta, _, _ = cute.arch.block_idx()
        ctas, _, _ = cute.arch.grid_dim()
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        g = lane // 4
        q = lane % 4
        total = layers * batch * HV
        inv_batch = cute.math.div(Float32(1.0), Float32(batch), full=True)

        mAddr = cute.make_tensor(
            addresses, cute.make_layout((layers, 10), stride=(10, 1))
        )
        mGroup = cute.make_tensor(group_indices, cute.make_layout(layers))
        mRead = cute.make_tensor(read_indices, cute.make_layout(layers * batch))
        mWrite = cute.make_tensor(write_indices, cute.make_layout(layers * batch))
        mAcc = cute.make_tensor(accepted_length, cute.make_layout(batch))

        smem = cutlass.utils.SmemAllocator()
        sState = smem.allocate_tensor(
            SD, cute.make_layout((2, K * K), stride=(K * K, 1)), byte_alignment=128
        )
        # [buffer, row, chunk of 8, element]
        sFrag = cute.make_tensor(
            sState.iterator,
            cute.make_layout((2, K, K // 8, 8), stride=(K * K, K, 8, 1)),
        )
        sXk = smem.allocate_tensor(
            BFloat16,
            cute.make_layout((2, T, K), stride=(T * K, K, 1)),
            byte_alignment=16,
        )
        sXv = smem.allocate_tensor(
            BFloat16,
            cute.make_layout((2, T, K), stride=(T * K, K, 1)),
            byte_alignment=16,
        )
        sG = smem.allocate_tensor(
            BFloat16,
            cute.make_layout((2, T, K), stride=(T * K, K, 1)),
            byte_alignment=16,
        )
        sBias = smem.allocate_tensor(
            Float32, cute.make_layout((2, K), stride=(K, 1)), byte_alignment=16
        )
        sAlog = smem.allocate_tensor(
            Float32, cute.make_layout((2, 1), stride=(1, 1)), byte_alignment=16
        )
        sCk = smem.allocate_tensor(
            BFloat16, cute.make_layout((2, K * 3), stride=(K * 3, 1)), byte_alignment=16
        )
        sCv = smem.allocate_tensor(
            BFloat16, cute.make_layout((2, K * 3), stride=(K * 3, 1)), byte_alignment=16
        )
        sWk = smem.allocate_tensor(
            BFloat16, cute.make_layout((2, K * 4), stride=(K * 4, 1)), byte_alignment=16
        )
        sWv = smem.allocate_tensor(
            BFloat16, cute.make_layout((2, K * 4), stride=(K * 4, 1)), byte_alignment=16
        )
        sBeta = smem.allocate_tensor(
            BFloat16,
            cute.make_layout((2, T, 2), stride=(2 * T, 2, 1)),
            byte_alignment=16,
        )
        # Normalized key and per-channel decay for each token: [token, chunk, 8]
        sK = smem.allocate_tensor(
            Float32,
            cute.make_layout((T, K // 8, 8), stride=(K, 8, 1)),
            byte_alignment=16,
        )
        sA = smem.allocate_tensor(
            Float32,
            cute.make_layout((T, K // 8, 8), stride=(K, 8, 1)),
            byte_alignment=16,
        )
        sSS = smem.allocate_tensor(
            Float32, cute.make_layout((K // 32, T), stride=(T, 1)), byte_alignment=16
        )
        fK = cute.make_tensor(sK.iterator, cute.make_layout((T, K), stride=(K, 1)))
        fA = cute.make_tensor(sA.iterator, cute.make_layout((T, K), stride=(K, 1)))
        mbar = smem.allocate_array(Int64, 2, byte_alignment=8)
        beta_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(), BFloat16, num_bits_per_copy=32
        )

        if tid == 0:
            # One expect-tx arrival, one cp.async arrival per beta row and one
            # for A_log.
            cute.arch.mbarrier_init(mbar, 2 + T)
            cute.arch.mbarrier_init(mbar + 1, 2 + T)
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        self._prefetch(
            cta, Int32(0), total, batch, inv_batch, mAddr, mGroup, mRead, sState,
            sXk, sXv, sG, sCk, sCv, sWk, sWv, sBeta, sBias, sAlog, beta_atom, mbar,
            tid,
        )  # fmt: skip

        it = Int32(0)
        for item in cutlass.range(cta, total, ctas):
            b = it % 2
            self._prefetch(
                item + ctas, 1 - b, total, batch, inv_batch, mAddr, mGroup, mRead,
                sState, sXk, sXv, sG, sCk, sCv, sWk, sWv, sBeta, sBias, sAlog,
                beta_atom, mbar, tid,
            )  # fmt: skip
            layer, n, hv = self._decode(item, batch, inv_batch)
            row = mGroup[layer] * batch + n
            read_page = Int64(mRead[row])
            write_page = Int64(mWrite[row])
            steps = mAcc[n]
            if steps < 0:
                steps = Int32(0)
            if steps > T:
                steps = Int32(T)
            read_ok = read_page >= 0
            cute.arch.mbarrier_wait(mbar + b, (it // 2) % 2)

            # Key side: thread t owns key channel t of every window token.
            w0 = sWk[b, tid * 4 + 0].to(Float32)
            w1 = sWk[b, tid * 4 + 1].to(Float32)
            w2 = sWk[b, tid * 4 + 2].to(Float32)
            w3 = sWk[b, tid * 4 + 3].to(Float32)
            s0 = Float32(0.0)
            s1 = Float32(0.0)
            s2 = Float32(0.0)
            if read_ok:
                s0 = sCk[b, tid * 3 + 0].to(Float32)
                s1 = sCk[b, tid * 3 + 1].to(Float32)
                s2 = sCk[b, tid * 3 + 2].to(Float32)
            exp_a = cute.math.exp(sAlog[b, 0], fastmath=True)
            bias = sBias[b, tid]
            rX = cute.make_rmem_tensor(T, Float32)
            rS = cute.make_rmem_tensor(T, Float32)
            for i in cutlass.range_constexpr(T):
                x = sXk[b, i, tid].to(Float32)
                acc = x * w3 + s0 * w0 + s1 * w1 + s2 * w2
                act = acc * cute.math.div(
                    Float32(1.0),
                    Float32(1.0) + cute.math.exp(-acc, fastmath=True),
                    full=True,
                )
                s0 = s1
                s1 = s2
                s2 = x
                rX[i] = act
                rS[i] = cute.arch.warp_reduction_sum(act * act)
                decay = self.lb * cute.math.div(
                    Float32(1.0),
                    Float32(1.0)
                    + cute.math.exp(
                        -(exp_a * (sG[b, i, tid].to(Float32) + bias)), fastmath=True
                    ),
                    full=True,
                )
                fA[i, tid] = cute.math.exp(decay, fastmath=True)
            if lane == 0:
                for i in cutlass.range_constexpr(T):
                    sSS[warp, i] = rS[i]
            cute.arch.sync_threads()
            for i in cutlass.range_constexpr(T):
                ss = Float32(0.0)
                for w in cutlass.range_constexpr(K // 32):
                    ss = ss + sSS[w, i]
                fK[i, tid] = rX[i] * cute.math.rsqrt(ss + Float32(1e-6), fastmath=True)

            # Value side: lane 4g + q computes row c = q; the group shares rows.
            rV = cute.make_rmem_tensor(cute.make_layout((T, 4), stride=(4, 1)), Float32)
            rB = cute.make_rmem_tensor(T, Float32)
            for i in cutlass.range_constexpr(T):
                beta = sBeta[b, i, hv % 2].to(Float32)
                rB[i] = cute.math.div(
                    Float32(1.0),
                    Float32(1.0) + cute.math.exp(-beta, fastmath=True),
                    full=True,
                )
            vq = 32 * warp + g + 8 * q
            u0 = sWv[b, vq * 4 + 0].to(Float32)
            u1 = sWv[b, vq * 4 + 1].to(Float32)
            u2 = sWv[b, vq * 4 + 2].to(Float32)
            u3 = sWv[b, vq * 4 + 3].to(Float32)
            t0 = Float32(0.0)
            t1 = Float32(0.0)
            t2 = Float32(0.0)
            if read_ok:
                t0 = sCv[b, vq * 3 + 0].to(Float32)
                t1 = sCv[b, vq * 3 + 1].to(Float32)
                t2 = sCv[b, vq * 3 + 2].to(Float32)
            for i in cutlass.range_constexpr(T):
                x = sXv[b, i, vq].to(Float32)
                acc = x * u3 + t0 * u0 + t1 * u1 + t2 * u2
                v = acc * cute.math.div(
                    Float32(1.0),
                    Float32(1.0) + cute.math.exp(-acc, fastmath=True),
                    full=True,
                )
                t0 = t1
                t1 = t2
                t2 = x
                for c in cutlass.range_constexpr(4):
                    rV[i, c] = cute.arch.shuffle_sync(v, 4 * g + c)

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
            rA = cute.make_rmem_tensor(cute.make_layout((4, 8), stride=(8, 1)), Float32)
            rP = cute.make_rmem_tensor(cute.make_layout((4, 2), stride=(2, 1)), Float32)
            rU = cute.make_rmem_tensor(4, Float32)
            zero = (Float32(0.0), Float32(0.0))
            for i in cutlass.range_constexpr(T):
                if i < steps:
                    for j in cutlass.range_constexpr(4):
                        cute.autovec_copy(sK[i, q + 4 * j, None], rK[j, None])
                        cute.autovec_copy(sA[i, q + 4 * j, None], rA[j, None])
                    rP.fill(0.0)
                    for c in cutlass.range_constexpr(4):
                        for j in cutlass.range_constexpr(4):
                            for e in cutlass.range_constexpr(0, 8, 2):
                                h = cute.arch.fma_packed_f32x2(
                                    (rH[c, j, e], rH[c, j, e + 1]),
                                    (rA[j, e], rA[j, e + 1]),
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

            if write_page >= 0:
                rIO.store(rH.load().to(SD))
                out = cute.make_tensor(
                    cute.make_ptr(
                        SD,
                        Int64(mAddr[layer, _STATE])
                        + (
                            write_page * self.s_state
                            + Int64(hv * K * K + (32 * warp + g) * K + q * 8)
                        )
                        * self.esz,
                        _GMEM,
                        assumed_align=16,
                    ),
                    cute.make_layout((4, 4, 8), stride=(8 * K, 32, 1)),
                )
                cute.autovec_copy(rIO, out)
            cute.arch.sync_threads()
            it = it + 1


_COMPILED: dict[tuple, object] = {}


@functools.cache
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def kda_batched_replay(
    addresses: torch.Tensor,
    group_indices: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    accepted_length: torch.Tensor,
    *,
    draft_token_num: int,
    num_heads: int,
    qkv_stride: int,
    conv_stride: int,
    beta_stride: int,
    state_stride: int,
    state_dtype: torch.dtype,
    gate_stride: int,
    lower_bound: float,
) -> None:
    """Replay the accepted window prefix of every descriptor layer.

    Reads each item's committed state and convolution window, replays
    ``accepted_length`` tokens and writes the state to ``write_indices``.
    The gate column holds the BF16 raw gate; the convolution windows are
    published by a separate launch.

    Args:
        addresses: ``[layers, 10]`` device pointer descriptors (int64 view).
        group_indices: ``[layers]`` int32 cache-group row of each layer.
        read_indices: ``[groups, batch]`` int32 source pages (-1 = fresh).
        write_indices: ``[groups, batch]`` int32 destination pages (-1 skips).
        accepted_length: ``[batch]`` int32 tokens to replay, clamped to
            ``[0, draft_token_num]``.
        draft_token_num: Window tokens per request.
        num_heads: Local KDA heads; the head dimension is 128.
        qkv_stride, conv_stride, beta_stride, state_stride, gate_stride:
            Row or page strides (elements) shared by every layer.
        state_dtype: ``torch.bfloat16`` or ``torch.float32`` state storage.
        lower_bound: Safe-gate lower bound.
    """
    batch = accepted_length.numel()
    layers = addresses.shape[0]
    if batch == 0 or layers == 0:
        return
    for name, tensor in (
        ("read_indices", read_indices),
        ("write_indices", write_indices),
        ("accepted_length", accepted_length),
    ):
        if tensor.dtype != torch.int32 or not tensor.is_contiguous():
            raise ValueError(f"{name} must be a contiguous int32 tensor")
    if read_indices.shape[-1] != batch or write_indices.shape[-1] != batch:
        raise ValueError("read_indices and write_indices must be [groups, batch]")
    dtype = {torch.bfloat16: BFloat16, torch.float32: Float32}[state_dtype]
    key = (
        draft_token_num,
        num_heads,
        state_dtype,
        qkv_stride,
        conv_stride,
        beta_stride,
        state_stride,
        gate_stride,
        float(lower_bound),
    )
    ptrs = (
        make_ptr(Int64, addresses.data_ptr(), _GMEM, assumed_align=8),
        make_ptr(Int32, group_indices.data_ptr(), _GMEM, assumed_align=4),
        make_ptr(Int32, read_indices.data_ptr(), _GMEM, assumed_align=4),
        make_ptr(Int32, write_indices.data_ptr(), _GMEM, assumed_align=4),
        make_ptr(Int32, accepted_length.data_ptr(), _GMEM, assumed_align=4),
    )
    ctas = min(
        CTAS_PER_SM * _sm_count(addresses.device.index), layers * batch * num_heads
    )
    stream = CUstream(torch.cuda.current_stream(addresses.device).cuda_stream)
    compiled = _COMPILED.get(key)
    if compiled is None:
        kernel = KdaBatchedReplay(
            draft_token_num,
            num_heads,
            dtype,
            qkv_stride,
            conv_stride,
            beta_stride,
            state_stride,
            gate_stride,
            float(lower_bound),
        )
        compiled = cute.compile(
            kernel, *ptrs, Int32(layers), Int32(batch), Int32(ctas), stream
        )
        _COMPILED[key] = compiled
    compiled(*ptrs, Int32(layers), Int32(batch), Int32(ctas), stream)
