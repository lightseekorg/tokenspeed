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

"""Direct-slot sparse prefill with native FP8 QK/PV and per-head Q scaling.

The low-level descriptor/MMA helpers imported below retain their MIT notices
in TokenSpeed. Each CTA owns one query and KV head; no request state is retained.
"""

import math

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float8E4M3FN, Float32, Int32, Int64, cute
from cutlass.cute.runtime import from_dlpack
from cutlass.experimental import primitives as prims
from cutlass.experimental.cuda import tensor_map as tmap
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_tma_gather.barrier import wait
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_tma_gather.handles import (
    MmaDesc,
    SmemTile,
)
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_tma_gather.mma import mma_ss
from tokenspeed_kernel.thirdparty.cute_dsl.tma import (
    TMA_CACHE_EVICT_LAST,
    tma_cache_hint,
    tma_gather4,
)

# Fixed FP8 geometry. Array elements below are bytes because E4M3 uses one byte.
_HEAD_DIM = 256
_TILE_TOKENS = 128
_PADDED_HEADS = 16
_TMA_BOX_BYTES = 128
_KV_BOX_BYTES = _TILE_TOKENS * _TMA_BOX_BYTES
_KV_STAGE_BYTES = _TILE_TOKENS * _HEAD_DIM
_Q_STAGE_BYTES = _PADDED_HEADS * _HEAD_DIM
# PV reads P with a 32-byte row pitch: 16 heads followed by 16 zero bytes.
_P_ROW_BYTES = 32
_P_STAGE_BYTES = _TILE_TOKENS * _P_ROW_BYTES
_SWIZZLE_ROWS = 8


class _Fp8SparsePrefill:
    """FP8 attention for 256-wide heads, with twelve Q heads per KV head.

    Each CTA owns one query/KV head and processes 128-token tiles with sixteen
    padded head columns. Shared memory holds separate 32 KiB K/V stages and
    4 KiB Q/P operands; TMEM holds the FP32 QK and PV fragments.
    """

    def __init__(self, kv_heads: int, pdl: bool):
        """Fix the KV-head count and PDL policy for this compiled specialization."""
        self.kv_heads: int = kv_heads
        self.pdl: bool = pdl

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        slots: cute.Tensor,
        positions: cute.Tensor,
        out: cute.Tensor,
        rows: Int32,
        pool_rows: Int32,
        slot_stride: Int64,
        pos_stride: Int64,
        scale: Float32,
        ks: Float32,
        vs: Float32,
        stream: CUstream,
    ):
        """Build runtime tensor maps and launch one CTA per query/KV head.

        Row counts, metadata strides and scales remain runtime inputs. ``scale``
        is the attention scale times log2(e); ``ks`` and ``vs`` are KV descales.
        The launch writes the supplied output on ``stream`` without changing Q/K/V.
        """
        dk = tmap.create_tensor_map_tiled(
            global_address=k.iterator.toint(),
            dtype=Float8E4M3FN,
            global_dims=(self.kv_heads * _HEAD_DIM, pool_rows),
            global_strides=(Int64(self.kv_heads * 16),),
            box_dims=(_TMA_BOX_BYTES, 1),
            swizzle=tmap.TensorMapSwizzle.s128b,
            l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
        )
        dv = tmap.create_tensor_map_tiled(
            global_address=v.iterator.toint(),
            dtype=Float8E4M3FN,
            global_dims=(self.kv_heads * _HEAD_DIM, pool_rows),
            global_strides=(Int64(self.kv_heads * 16),),
            box_dims=(_TMA_BOX_BYTES, 1),
            swizzle=tmap.TensorMapSwizzle.s128b,
            l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
        )
        self.kernel(
            q, dk, dv, slots, positions, out, slot_stride, pos_stride, scale, ks, vs
        ).launch(
            grid=(rows, self.kv_heads, 1),
            block=(256, 1, 1),
            stream=stream,
            use_pdl=self.pdl,
            preferred_smem_carveout=100,
        )

    @cute.jit
    def vector(self, base, offset, size: cutlass.Constexpr):
        """View shared FP8 bytes at an offset aligned to ``size``, without copying."""
        return cute.make_tensor(
            cute.make_ptr(
                Float8E4M3FN,
                base.data_ptr().toint() + Int64(offset),
                cute.AddressSpace.smem,
                assumed_align=size,
            ),
            cute.make_layout(size),
        )

    @cute.jit
    def prefetch(
        self,
        desc,
        raw,
        barrier,
        slot: Int64,
        group: Int32,
        tid: Int32,
        plane: cutlass.Constexpr,
    ):
        """Collectively enqueue a 128-token gather into the selected shared plane.

        Arm ``barrier`` for one K/V stage; nonpositive slots are zero-filled.
        The caller issues K then V, waiting on phases 0 then 1 each iteration,
        and must finish the plane's previous MMA reads before reusing it.
        """
        if tid == 0:
            prims.mbarrier_arrive_expect_tx(barrier, Int32(_KV_STAGE_BYTES))
        cute.arch.sync_threads()
        row = Int32(slot) if slot > 0 else Int32(-1)
        rows = cute.make_rmem_tensor(4, Int32)
        for quad in cutlass.range_constexpr(4):
            for i in cutlass.range_constexpr(4):
                rows[i] = cute.arch.shuffle_sync(row, quad * 4 + i)
            if tid % 32 == 0:
                hint = tma_cache_hint(TMA_CACHE_EVICT_LAST)
                token_base = tid // 32 * 16 + quad * 4
                for box in cutlass.range_constexpr(2):
                    tma_gather4(
                        Int32(
                            raw.data_ptr().toint()
                            + Int64(
                                plane * _KV_STAGE_BYTES
                                + token_base * _TMA_BOX_BYTES
                                + box * _KV_BOX_BYTES
                            )
                        ),
                        desc.get_ptr().toint(),
                        Int32(barrier.data_ptr().toint()),
                        group * _HEAD_DIM + box * _TMA_BOX_BYTES,
                        rows[0],
                        rows[1],
                        rows[2],
                        rows[3],
                        cache_hint=hint,
                    )

    @cute.jit
    def load_slot(
        self,
        slots: cute.Tensor,
        row: Int32,
        stride: Int64,
        token: Int32,
        complete: Int32,
        count: Int32,
    ):
        """Read a compact token's physical slot, or return -1 beyond ``count``.

        The first ``complete`` tokens read prefix columns directly; remaining
        tokens read tail columns starting at 2048. ``stride`` counts int32 slots.
        """
        slot = Int64(-1)
        if token < count:
            column = token if token < complete else 2048 + token - complete
            source = cute.make_tensor(
                cute.make_ptr(
                    Int32,
                    slots.iterator.toint() + (Int64(row) * stride + column) * 4,
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                ),
                cute.make_layout(1),
            )
            slot = Int64(source[0])
        return slot

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        k: cutlass.GridConstant[tmap.TensorMap],
        v: cutlass.GridConstant[tmap.TensorMap],
        slots: cute.Tensor,
        positions: cute.Tensor,
        out: cute.Tensor,
        slot_stride: Int64,
        pos_stride: Int64,
        scale: Float32,
        ks: Float32,
        vs: Float32,
    ):
        """Compute one query/KV head with FP8 QK/PV and FP32 online softmax.

        Overlap alternating K/V gathers with QK/PV MMA, then write BF16 output;
        fully masked rows produce zero. PDL waits precede input reads and the
        completion signal follows output stores and TMEM release.
        """
        row, group, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % 32
        warp = tid // 32
        head_group = tid // 128
        token_lane = tid % 128
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()
        # K and V each own a stage; TMA writes one while MMA reads the other.
        raw = cutlass.Array(
            Float8E4M3FN,
            2 * _KV_STAGE_BYTES,
            alignment=1024,
            space=cutlass.AddressSpace.smem,
        )
        sQ = cutlass.Array(
            Float8E4M3FN,
            _Q_STAGE_BYTES,
            alignment=1024,
            space=cutlass.AddressSpace.smem,
        )
        sP = cutlass.Array(
            Float8E4M3FN,
            _P_STAGE_BYTES,
            alignment=1024,
            space=cutlass.AddressSpace.smem,
        )
        q_scales = cutlass.Array(
            Float32, 16, alignment=16, space=cutlass.AddressSpace.smem
        )
        reductions = cutlass.Array(
            Float32, 64, alignment=16, space=cutlass.AddressSpace.smem
        )
        tm = cutlass.Array(Int32, 1, alignment=16, space=cutlass.AddressSpace.smem)
        done = cutlass.Array(Int64, 1, alignment=16, space=cutlass.AddressSpace.smem)
        loaded = cutlass.Array(Int64, 1, alignment=16, space=cutlass.AddressSpace.smem)
        if tid == 0:
            prims.mbarrier_init(done, 1)
            prims.mbarrier_init(loaded, 1)
        prims.fence_mbarrier_init()
        if warp == 0:
            prims.tcgen05_alloc(tm, Int32(64), group=prims.CTAGroup.CTA_1)
            prims.tcgen05_relinquish_alloc_permit(group=prims.CTAGroup.CTA_1)
        qbase = Int64(row) * (self.kv_heads * 12 * 256) + group * 12 * 256
        qlayout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout(
                (_PADDED_HEADS, _TMA_BOX_BYTES, 2),
                stride=(_TMA_BOX_BYTES, 1, _PADDED_HEADS * _TMA_BOX_BYTES),
            ),
        )
        qvec = cute.make_rmem_tensor(16, BFloat16)
        qquant = cute.make_rmem_tensor(16, Float8E4M3FN)
        head = tid // 16
        col = tid % 16 * 16
        qvec.fill(0)
        if head < 12:
            source = cute.make_tensor(
                cute.make_ptr(
                    BFloat16,
                    q.iterator.toint() + (qbase + head * 256 + col) * 2,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout(16),
            )
            cute.autovec_copy(source, qvec)
        maximum = Float32(0)
        for i in cutlass.range_constexpr(16):
            value = Float32(qvec[i])
            maximum = cute.arch.fmax(maximum, cute.arch.fmax(value, -value))
        for bit in cutlass.range_constexpr(4):
            maximum = cute.arch.fmax(
                maximum, cute.arch.shuffle_sync_bfly(maximum, offset=1 << bit)
            )
        q_scale = (
            cute.math.div(
                maximum, Float32(448), rounding=cute.math.RoundingMode.NEAREST_EVEN
            )
            if maximum > 0
            else Float32(1)
        )
        if tid % 16 == 0:
            q_scales.store(q_scale, head)
        for i in cutlass.range_constexpr(16):
            qquant[i] = Float8E4M3FN(
                cute.math.div(
                    Float32(qvec[i]),
                    q_scale,
                    rounding=cute.math.RoundingMode.NEAREST_EVEN,
                )
            )
        cute.autovec_copy(
            qquant, self.vector(sQ, qlayout((head, col % 128, col // 128)), 16)
        )
        playout = cute.make_composed_layout(
            cute.make_swizzle(1, 4, 3),
            0,
            cute.make_layout((_TILE_TOKENS, _P_ROW_BYTES), stride=(_P_ROW_BYTES, 1)),
        )
        zeros = cute.make_rmem_tensor(16, Float8E4M3FN)
        zeros.fill(0)
        if tid < 128:
            cute.autovec_copy(zeros, self.vector(sP, playout((tid, 16)), 16))
        prims.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()
        scales = cute.make_rmem_tensor(8, Float32)
        for h in cutlass.range_constexpr(8):
            scales[h] = q_scales.load(head_group * 8 + h)
        tbase = tm.load()
        # SW128 descriptors span eight 128-byte rows; P uses eight 32-byte rows.
        descq = SmemTile(
            base=sQ,
            elems_per_stage=_Q_STAGE_BYTES,
            leading_byte_offset=0,
            stride_byte_offset=_SWIZZLE_ROWS * _TMA_BOX_BYTES,
            layout=2,
        ).desc()
        desck = SmemTile(
            base=raw,
            elems_per_stage=_KV_STAGE_BYTES,
            leading_byte_offset=0,
            stride_byte_offset=_SWIZZLE_ROWS * _TMA_BOX_BYTES,
            layout=2,
        ).desc()
        descp = SmemTile(
            base=sP,
            elems_per_stage=_P_STAGE_BYTES,
            leading_byte_offset=0,
            stride_byte_offset=_SWIZZLE_ROWS * _P_ROW_BYTES,
            layout=6,
        ).desc()
        qk = MmaDesc(
            M=128,
            N=16,
            K=256,
            bpe_a=1,
            bpe_b=1,
            tile_k_hw=32,
            atranspose=False,
            btranspose=False,
            kind=prims.Tcgen05MMAKind.F8F6F4,
            idesc=prims.Tcgen05InstrDesc.build(
                c_dtype=Float32,
                a_dtype=Float8E4M3FN,
                b_dtype=Float8E4M3FN,
                n_dim=16,
                m_dim=128,
                k_dim=0,
            ),
        )
        pv = MmaDesc(
            M=128,
            N=16,
            K=128,
            bpe_a=1,
            bpe_b=1,
            tile_k_hw=32,
            atranspose=True,
            btranspose=True,
            kind=prims.Tcgen05MMAKind.F8F6F4,
            idesc=prims.Tcgen05InstrDesc.build(
                c_dtype=Float32,
                a_dtype=Float8E4M3FN,
                b_dtype=Float8E4M3FN,
                n_dim=16,
                m_dim=128,
                a_major=1,
                b_major=1,
                k_dim=0,
            ),
        )
        position_ptr = cute.make_tensor(
            cute.make_ptr(
                Int64,
                positions.iterator.toint() + Int64(row) * pos_stride * 8,
                cute.AddressSpace.gmem,
                assumed_align=8,
            ),
            cute.make_layout(1),
        )
        position = Int64(position_ptr[0])
        visible = position + 1 if position >= 0 else Int64(0)
        complete = Int32(visible // 4 if visible // 4 < 512 else Int64(512)) * 4
        count = complete + Int32(visible % 4)
        maxima = cute.make_rmem_tensor(8, Float32)
        sums = cute.make_rmem_tensor(8, Float32)
        output = cute.make_rmem_tensor((8, 2), Float32)
        probabilities = cute.make_rmem_tensor(8, Float8E4M3FN)
        alpha = cute.make_rmem_tensor(8, Float32)
        maxima.fill(-float("inf"))
        sums.fill(0)
        output.fill(0)
        slot = self.load_slot(slots, row, slot_stride, token_lane, complete, count)
        gather_slot = self.load_slot(
            slots, row, slot_stride, warp * 16 + lane % 16, complete, count
        )
        # Each loop completes two phases: loaded K=0/V=1 and done QK=0/PV=1.
        # TMEM columns 0:16 hold QK; 16:32 and 32:48 hold the two PV blocks.
        if count > 0:
            self.prefetch(k, raw, loaded, gather_slot, group, tid, 0)
        for start in range(0, count, 128):
            wait(loaded, Int32(0))
            cute.arch.sync_threads()
            self.prefetch(v, raw, loaded, gather_slot, group, tid, 1)
            if warp == 0:
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                mma_ss(
                    qk,
                    desck,
                    descq,
                    prims.make_tmem_ptr(tbase, cutlass.Int8),
                    accumulate=False,
                )
                if prims.elect_sync():
                    prims.tcgen05_commit(done, group=prims.CTAGroup.CTA_1)
            wait(done, Int32(0))
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            scores = prims.tcgen05_ld(
                "32x32b", prims.make_tmem_ptr(tbase + head_group * 8, Float32), num=8
            )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
            scaled = cute.make_rmem_tensor(8, Float32)
            for h in cutlass.range_constexpr(8):
                x = (
                    Float32(scores[h]) * (scale * ks * scales[h])
                    if slot > 0
                    else Float32(-float("inf"))
                )
                scaled[h] = x
                for bit in cutlass.range_constexpr(5):
                    x = cute.arch.fmax(
                        x, cute.arch.shuffle_sync_bfly(x, offset=1 << bit)
                    )
                if lane == 0:
                    reductions.store(x, warp * 8 + h)
            cute.arch.sync_threads()
            for h in cutlass.range_constexpr(8):
                maximum = maxima[h]
                for w in cutlass.range_constexpr(4):
                    maximum = cute.arch.fmax(
                        maximum, reductions.load(head_group * 32 + w * 8 + h)
                    )
                safe = Float32(0) if maximum == -float("inf") else maximum
                alpha[h] = cute.math.exp2(maxima[h] - safe, fastmath=True)
                p = cute.math.exp2(scaled[h] - safe, fastmath=True)
                probabilities[h] = Float8E4M3FN(p * Float32(448))
                sums[h] = sums[h] * alpha[h] + p
                maxima[h] = maximum
            cute.autovec_copy(
                probabilities, self.vector(sP, playout((token_lane, head_group * 8)), 8)
            )
            prims.fence_proxy("async.shared", space="cta")
            wait(loaded, Int32(1))
            cute.arch.sync_threads()
            next_slot = self.load_slot(
                slots, row, slot_stride, start + 128 + token_lane, complete, count
            )
            next_gather = self.load_slot(
                slots,
                row,
                slot_stride,
                start + 128 + warp * 16 + lane % 16,
                complete,
                count,
            )
            # QK has released the K stage. Refill it while PV reads the V stage.
            if start + 128 < count:
                self.prefetch(k, raw, loaded, next_gather, group, tid, 0)
            if warp == 0:
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                for block in cutlass.range_constexpr(2):
                    descv = SmemTile(
                        base=raw.subview(_KV_STAGE_BYTES + block * _KV_BOX_BYTES),
                        elems_per_stage=_KV_STAGE_BYTES,
                        leading_byte_offset=_KV_BOX_BYTES,
                        stride_byte_offset=_SWIZZLE_ROWS * _TMA_BOX_BYTES,
                        layout=2,
                    ).desc()
                    mma_ss(
                        pv,
                        descv,
                        descp,
                        prims.make_tmem_ptr(tbase + 16 + block * 16, cutlass.Int8),
                        accumulate=False,
                    )
                if prims.elect_sync():
                    prims.tcgen05_commit(done, group=prims.CTAGroup.CTA_1)
            wait(done, Int32(1))
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            for block in cutlass.range_constexpr(2):
                partial = prims.tcgen05_ld(
                    "32x32b",
                    prims.make_tmem_ptr(
                        tbase + 16 + block * 16 + head_group * 8, Float32
                    ),
                    num=8,
                )
                prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                for h in cutlass.range_constexpr(8):
                    output[h, block] = output[h, block] * alpha[h] + Float32(partial[h])
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            cute.arch.sync_threads()
            # PV and its TMEM reads are finished before the next V-stage write.
            slot = next_slot
            gather_slot = next_gather
        for h in cutlass.range_constexpr(8):
            total = sums[h]
            for bit in cutlass.range_constexpr(5):
                total += cute.arch.shuffle_sync_bfly(total, offset=1 << bit)
            if lane == 0:
                reductions.store(total, warp * 8 + h)
        cute.arch.sync_threads()
        for h in cutlass.range_constexpr(8):
            total = Float32(0)
            for w in cutlass.range_constexpr(4):
                total += reductions.load(head_group * 32 + w * 8 + h)
            denominator = total if total > 0 else Float32(1)
            for block in cutlass.range_constexpr(2):
                head = head_group * 8 + h
                if head < 12:
                    out[qbase + head * 256 + token_lane + block * 128] = BFloat16(
                        output[h, block] * (vs / (denominator * Float32(448)))
                    )
        cute.arch.sync_threads()
        if warp == 0:
            prims.tcgen05_dealloc(
                prims.make_tmem_ptr(tbase, cutlass.Int8),
                Int32(64),
                group=prims.CTAGroup.CTA_1,
            )

        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_launch_dependents()


# Cache of cute.compile results keyed by (device, KV-head count, PDL) so each
# static specialization compiles only once per process; later calls reuse it.
_compiled = {}


def launch_fp8_sparse_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    slots: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    *,
    scale: float,
    k_scale: float,
    v_scale: float,
    enable_pdl: bool,
) -> torch.Tensor:
    """Launch validated BF16 Q / FP8 KV prefill into the caller's output.

    All tensors must share a CUDA device. The caller validates the geometry,
    dtypes, and alignment before launch. KV heads must be one or two.

    Args:
        q: Dense BF16 queries shaped [rows, 12 * kv_heads, 256], aligned to
            16 bytes.
        k: Dense E4M3FN key cache shaped [pool_rows, kv_heads, 256], aligned
            to 16 bytes.
        v: Dense E4M3FN value cache with the same shape and alignment as k.
        slots: Int32 physical slots shaped [rows, 2051], aligned to 4 bytes,
            with contiguous columns and a nonnegative row stride. Columns
            hold 512 four-token blocks followed by three fixed tail slots.
            Nonpositive slots contribute no probability mass.
        positions: Int64 query positions shaped [rows], aligned to 8 bytes,
            with a nonnegative stride. Each position bounds the visible
            complete-block prefix and tail slots.
        out: Caller-owned dense BF16 output shaped like q, aligned to
            16 bytes, with storage separate from the inputs.
        scale: Scalar attention scale applied to FP32 query-key scores.
        k_scale: Scalar key cache descale.
        v_scale: Scalar value cache descale.
        enable_pdl: Whether to wait for producers and signal completion
            through programmatic dependent launch.

    Returns:
        The supplied out tensor, populated with attention results.
    """
    # TMA gather coordinates and the grid use signed 32-bit row counts.
    # Never derive pool rows from a flattened element count, which can overflow.
    if not 0 < k.shape[0] < (1 << 31) or not 0 <= q.shape[0] < (1 << 31):
        raise ValueError("FP8 QSA prefill requires fewer than 2**31 query/cache rows")
    if q.shape[0] == 0:
        return out
    tensors = [
        from_dlpack(x, assumed_align=alignment).mark_layout_dynamic()
        for x, alignment in (
            (q.view(-1), 16),
            (k, 16),
            (v, 16),
            (slots, 4),
            (positions, 8),
            (out.view(-1), 16),
        )
    ]
    with torch.cuda.device(q.device):
        args = (
            *tensors,
            Int32(q.shape[0]),
            Int32(k.shape[0]),
            Int64(slots.stride(0)),
            Int64(positions.stride(0)),
            Float32(math.log2(math.e) * scale),
            Float32(k_scale),
            Float32(v_scale),
            CUstream(torch.cuda.current_stream(q.device).cuda_stream),
        )
        key = (q.device.index, k.shape[1], enable_pdl)
        if key not in _compiled:
            _compiled[key] = cute.compile(
                _Fp8SparsePrefill(k.shape[1], enable_pdl), *args
            )
        _compiled[key](*args)
    return out
