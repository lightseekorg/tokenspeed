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

"""Four-token WY with the preceding accepted prefix and one state read."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack, make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op
from tokenspeed_kernel.ops.attention.gdn._cute_dsl.wy import _mma_tf32
from tokenspeed_kernel.platform import pdl_enabled


@dsl_user_op
def _tf32(x, *, loc=None, ip=None):
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "cvt.rn.tf32.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _product(a, b, lane: Int32):
    # Keep both residuals: cancellation in a propagated query can be strong.
    c = cute.make_rmem_tensor(4, Float32)
    c.fill(0.0)
    row, key = lane // 4, lane % 4
    for block in cutlass.range_constexpr(16):
        offset = key + block * 8
        av = cute.make_rmem_tensor(4, Float32)
        ah = cute.make_rmem_tensor(4, Float32)
        al = cute.make_rmem_tensor(4, Float32)
        av[0], av[1] = Float32(a[row, offset]), Float32(a[row + 8, offset])
        av[2], av[3] = Float32(a[row, offset + 4]), Float32(a[row + 8, offset + 4])
        for i in cutlass.range_constexpr(4):
            ah[i] = _tf32(av[i])
            al[i] = _tf32(av[i] - ah[i])
        b0, b1 = b[row, offset], b[row, offset + 4]
        bh0, bh1 = _tf32(b0), _tf32(b1)
        bl0, bl1 = _tf32(b0 - bh0), _tf32(b1 - bh1)
        c.store(_mma_tf32(ah[0], ah[1], ah[2], ah[3], bh0, bh1, c, True, True))
        c.store(_mma_tf32(ah[0], ah[1], ah[2], ah[3], bl0, bl1, c, True, True))
        c.store(_mma_tf32(al[0], al[1], al[2], al[3], bh0, bh1, c, True, True))
        c.store(_mma_tf32(al[0], al[1], al[2], al[3], bl0, bl1, c, True, True))
    return c.load()


class GdnDeferredWYKernel:
    def __init__(self, heads: int, vheads: int, normalize: bool, enable_pdl: bool):
        self.heads = heads
        self.vheads = vheads
        self.normalize = normalize
        self.enable_pdl = enable_pdl

    @cute.jit
    def __call__(
        self,
        operands: tuple,
        strides: tuple,
        batch: Int32,
        capacity: Int32,
        scale: Float32,
        state,
        stream: cuda.CUstream,
    ):
        layout = cute.make_layout((16, 128), stride=(128, 1))
        source = cute.make_tensor(
            state.iterator,
            cute.make_layout(
                (128, 128, self.vheads, state.shape[0]),
                stride=(128, 1, 16384, state.stride[0]),
            ),
        )
        load_atom, load_source = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), source, layout, (16, 128)
        )
        store_atom, store_source = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            source,
            cute.make_layout((32, 128), stride=(128, 1)),
            (32, 128),
        )
        self.run(
            operands,
            strides,
            capacity,
            scale,
            load_atom,
            load_source,
            store_atom,
            store_source,
        ).launch(
            grid=(batch * self.vheads, 4, 1),
            block=(128, 1, 1),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def run(
        self,
        operands: tuple,
        strides: tuple,
        capacity: Int32,
        scale: Float32,
        load_atom,
        load_source,
        store_atom,
        store_source,
    ):
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        tid = cute.arch.thread_idx()[0]
        warp, lane = tid // 32, tid % 32
        nh, tile, _ = cute.arch.block_idx()
        batch, hv = nh // self.vheads, nh % self.vheads
        head = hv // (self.vheads // self.heads)
        q, k, v, a, b, A, bias, indices, payload, rows, steps, pages, bank, out = tuple(
            cute.make_tensor(p, cute.make_layout(9223372036854775807)) for p in operands
        )
        (
            qs,
            ks,
            vs,
            ats,
            bts,
            payload_bank_stride,
            payload_layer_offset,
            page_bank_stride,
            page_layer_offset,
        ) = strides
        width = self.heads * 128 + self.vheads * 128 + 2 * self.vheads
        idx = Int64(indices[batch])
        code = Int32(rows[batch])
        accepted = Int32(0)
        old_bank, old_row = Int32(0), Int32(0)
        if code >= 0 and idx > 0:
            old_bank, old_row = code // capacity, code % capacity
            if (
                Int64(
                    pages[
                        Int64(old_bank) * page_bank_stride + page_layer_offset + old_row
                    ]
                )
                == idx
            ):
                accepted = Int32(steps[code])
                if accepted < 0:
                    accepted = Int32(0)
                if accepted > 4:
                    accepted = Int32(4)
        old_base = (
            Int64(old_bank) * payload_bank_stride
            + payload_layer_offset
            + Int64(old_row) * 4 * width
        )
        new_base = (
            Int64(bank[0]) * payload_bank_stride
            + payload_layer_offset
            + Int64(batch) * 4 * width
        )
        smem = cutlass.utils.SmemAllocator()
        state_layout = cute.make_layout((16, 128, 2), stride=(128, 1, 2048))
        sH = smem.allocate_tensor(cutlass.BFloat16, state_layout, byte_alignment=128)
        barriers = smem.allocate_tensor(Int64, cute.make_layout(2), byte_alignment=8)
        sCommit = smem.allocate_tensor(
            cutlass.BFloat16,
            cute.make_layout((32, 128), stride=(128, 1)),
            byte_alignment=128,
        )
        sK = smem.allocate_tensor(Float32, cute.make_layout((8, 128), stride=(132, 1)))
        sPhi = smem.allocate_tensor(
            Float32, cute.make_layout((4, 128), stride=(132, 1))
        )
        sV = smem.allocate_tensor(Float32, cute.make_layout((8, 32), stride=(36, 1)))
        sG = smem.allocate_tensor(Float32, cute.make_layout(8))
        sBeta = smem.allocate_tensor(Float32, cute.make_layout(8))
        sW = smem.allocate_tensor(Float32, cute.make_layout((4, 4)))
        sGram = smem.allocate_tensor(Float32, cute.make_layout((4, 4)))
        sCross = smem.allocate_tensor(Float32, cute.make_layout((4, 4)))
        sP = smem.allocate_tensor(Float32, cute.make_layout((8, 32), stride=(36, 1)))
        sU = smem.allocate_tensor(Float32, cute.make_layout((4, 32), stride=(36, 1)))
        # Two V16 transfers leave K intact for the packed projection MMA.
        for stage in cutlass.range_constexpr(2):
            if idx >= 0:
                if warp == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_init(barriers.iterator + stage, 1)
                    cute.arch.mbarrier_init_fence()
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            barriers.iterator + stage, 4096
                        )
                    source_tile = cute.local_tile(
                        load_source, (16, 128), (tile * 2 + stage, 0, hv, idx)
                    )
                    dst, src = cpasync.tma_partition(
                        load_atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sH[None, None, stage], 0, 2),
                        cute.group_modes(source_tile, 0, 2),
                    )
                    cute.copy(
                        load_atom, src, dst, tma_bar_ptr=barriers.iterator + stage
                    )
            else:
                for part in cutlass.range_constexpr(16):
                    offset = tid + part * 128
                    sH[offset // 128, offset % 128, stage] = cutlass.BFloat16(0)
        for half in cutlass.range_constexpr(2):
            token = warp
            kv = cute.make_rmem_tensor(4, Float32)
            qv = cute.make_rmem_tensor(4, Float32)
            qv.fill(0.0)
            knorm, qnorm = Float32(0), Float32(0)
            for part in cutlass.range_constexpr(4):
                col = lane + part * 32
                raw = Float32(0)
                if cutlass.const_expr(half == 0):
                    if token < accepted:
                        raw = Float32(
                            payload[old_base + token * width + head * 128 + col]
                        )
                else:
                    raw = Float32(
                        k[
                            Int64(batch) * ks[0]
                            + token * ks[1]
                            + head * ks[2]
                            + col * ks[3]
                        ]
                    )
                    qv[part] = Float32(
                        q[
                            Int64(batch) * qs[0]
                            + token * qs[1]
                            + head * qs[2]
                            + col * qs[3]
                        ]
                    )
                    qnorm += qv[part] * qv[part]
                kv[part] = raw
                knorm += raw * raw
            for shift in cutlass.range_constexpr(5):
                knorm += cute.arch.shuffle_sync_bfly(knorm, 1 << shift)
                if cutlass.const_expr(half == 1):
                    qnorm += cute.arch.shuffle_sync_bfly(qnorm, 1 << shift)
            ki, qi = Float32(1), scale
            if cutlass.const_expr(self.normalize):
                ki = cute.math.rsqrt(knorm + 1e-6, fastmath=True)
                qi *= cute.math.rsqrt(qnorm + 1e-6, fastmath=True)
            for part in cutlass.range_constexpr(4):
                sK[token + half * 4, lane + part * 32] = kv[part] * ki
                if cutlass.const_expr(half == 1):
                    sPhi[token, lane + part * 32] = qv[part] * qi
            valcol = tile * 32 + lane
            value = Float32(0)
            av, bv = Float32(0), Float32(0)
            if cutlass.const_expr(half == 0):
                if token < accepted:
                    value = Float32(
                        payload[
                            old_base
                            + token * width
                            + self.heads * 128
                            + hv * 128
                            + valcol
                        ]
                    )
                    av = Float32(
                        payload[
                            old_base
                            + token * width
                            + self.heads * 128
                            + self.vheads * 128
                            + hv
                        ]
                    )
                    bv = Float32(
                        payload[
                            old_base
                            + token * width
                            + self.heads * 128
                            + self.vheads * 128
                            + self.vheads
                            + hv
                        ]
                    )
            else:
                value = Float32(
                    v[
                        Int64(batch) * vs[0]
                        + token * vs[1]
                        + hv * vs[2]
                        + valcol * vs[3]
                    ]
                )
                av = Float32(a[Int64(batch) * ats[0] + token * ats[1] + hv * ats[2]])
                bv = Float32(b[Int64(batch) * bts[0] + token * bts[1] + hv * bts[2]])
            sV[token + half * 4, lane] = value
            if lane == 0:
                x = av + Float32(bias[hv])
                softplus = x
                if x <= 20:
                    softplus = cute.math.log(
                        1.0 + cute.math.exp(x, fastmath=True), fastmath=True
                    )
                sG[token + half * 4] = (
                    -cute.math.exp(Float32(A[hv]), fastmath=True) * softplus
                )
                sBeta[token + half * 4] = 1.0 / (
                    1.0 + cute.math.exp(-bv, fastmath=True)
                )
        cute.arch.sync_threads()
        phi = cute.make_rmem_tensor(4, Float32)
        for part in cutlass.range_constexpr(4):
            phi[part] = sPhi[warp, lane + part * 32]
        for back in cutlass.range_constexpr(4):
            j = 3 - back
            weight = Float32(0)
            if j <= warp:
                for part in cutlass.range_constexpr(4):
                    weight += phi[part] * sK[4 + j, lane + part * 32]
                for shift in cutlass.range_constexpr(5):
                    weight += cute.arch.shuffle_sync_bfly(weight, 1 << shift)
                weight *= sBeta[4 + j]
                decay = cute.math.exp(sG[4 + j], fastmath=True)
                for part in cutlass.range_constexpr(4):
                    phi[part] = decay * (
                        phi[part] - weight * sK[4 + j, lane + part * 32]
                    )
            if lane == 0:
                sW[warp, j] = weight
        # The old-key Gram and propagated-query products are only 4x4 each.
        for j in cutlass.range_constexpr(4):
            gram, cross = Float32(0), Float32(0)
            for part in cutlass.range_constexpr(4):
                key = sK[j, lane + part * 32]
                gram += sK[warp, lane + part * 32] * key
                cross += phi[part] * key
            for shift in cutlass.range_constexpr(5):
                gram += cute.arch.shuffle_sync_bfly(gram, 1 << shift)
                cross += cute.arch.shuffle_sync_bfly(cross, 1 << shift)
            if lane == 0:
                sGram[warp, j], sCross[warp, j] = gram, cross
        cute.arch.sync_threads()
        for part in cutlass.range_constexpr(4):
            sK[4 + warp, lane + part * 32] = phi[part]
        for stage in cutlass.range_constexpr(2):
            if idx >= 0:
                cute.arch.mbarrier_wait(barriers.iterator + stage, 0)
        cute.arch.sync_threads()
        if warp < 2:
            h = sH[None, None, warp]
            projection = _product(h, sK, lane)
            for j in cutlass.range_constexpr(4):
                row, col = lane // 4 + (j // 2) * 8, (lane % 4) * 2 + j % 2
                sP[col, warp * 16 + row] = projection[j]
        cute.arch.sync_threads()
        total_gate = Float32(0)
        for j in cutlass.range_constexpr(4):
            if j < accepted:
                total_gate += sG[j]
        decay = cute.math.exp(total_gate, fastmath=True)
        if tid < 32:
            u = cute.make_rmem_tensor(4, Float32)
            u.fill(0.0)
            prefix = Float32(0)
            for i in cutlass.range_constexpr(4):
                if i < accepted:
                    prefix += sG[i]
                    update = (
                        sV[i, tid] - cute.math.exp(prefix, fastmath=True) * sP[i, tid]
                    )
                    for j in cutlass.range_constexpr(i):
                        gate = Float32(0)
                        for step in cutlass.range_constexpr(j + 1, i + 1):
                            gate += sG[step]
                        update -= (
                            cute.math.exp(gate, fastmath=True) * sGram[i, j] * u[j]
                        )
                    u[i] = sBeta[i] * update
                suffix = Float32(0)
                for j in cutlass.range_constexpr(i + 1, 4):
                    if j < accepted:
                        suffix += sG[j]
                sU[i, tid] = u[i] * cute.math.exp(suffix, fastmath=True)
            for token in cutlass.range_constexpr(4):
                value = decay * sP[4 + token, tid]
                for j in cutlass.range_constexpr(4):
                    value += (
                        sCross[token, j] * sU[j, tid] + sW[token, j] * sV[4 + j, tid]
                    )
                out[
                    ((Int64(batch) * 4 + token) * self.vheads + hv) * 128
                    + tile * 32
                    + tid
                ] = out.element_type(value)
        cute.arch.sync_threads()
        if accepted > 0:
            for part in cutlass.range_constexpr(32):
                offset = tid + part * 128
                val, state_key = offset // 128, offset % 128
                state_value = decay * Float32(sH[val % 16, state_key, val // 16])
                for j in cutlass.range_constexpr(4):
                    state_value += sU[j, val] * sK[j, state_key]
                sCommit[val, state_key] = cutlass.BFloat16(state_value)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()
            if warp == 0:
                destination = cute.local_tile(
                    store_source, (32, 128), (tile, 0, hv, idx)
                )
                store_src, store_dst = cpasync.tma_partition(
                    store_atom,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sCommit, 0, 2),
                    cute.group_modes(destination, 0, 2),
                )
                cute.copy(store_atom, store_src, store_dst)
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=False)


_COMPILED: dict[tuple, object] = {}


def launch_gdn_deferred_wy(
    q,
    k,
    v,
    *,
    A_log,
    a,
    dt_bias,
    b,
    initial_state,
    initial_state_indices,
    scale,
    use_qk_l2norm,
    pending,
):
    """V32, two-stage TMA verify; counts, strides and bank selection stay dynamic."""
    batch, tokens, heads, dim = q.shape
    vheads = v.shape[2]
    if (
        tokens != 4
        or dim != 128
        or v.shape[-1] != 128
        or initial_state.dtype != torch.bfloat16
    ):
        raise ValueError("pending replay requires T4 and BF16 K128/V128 state")
    if initial_state.data_ptr() % 16 or initial_state.stride(0) * 2 % 16:
        raise ValueError("pending replay requires 16-byte aligned state rows")
    capacity = pending.steps.shape[1]
    out = q.new_empty((batch, 4, vheads, 128))
    tensors = (
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        initial_state_indices,
        pending.payload,
        pending.rows,
        pending.steps,
        pending.pages,
        pending.write_bank,
        out,
    )
    types = {
        torch.bfloat16: cutlass.BFloat16,
        torch.float16: cutlass.Float16,
        torch.float32: Float32,
        torch.int32: Int32,
        torch.int64: Int64,
    }
    pointers = tuple(
        make_ptr(
            types[x.dtype],
            x.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=x.element_size(),
        )
        for x in tensors
    )
    strides = tuple(tuple(Int64(s) for s in x.stride()) for x in (q, k, v, a, b)) + (
        Int64(pending.payload.stride(0)),
        Int64(pending.layer * pending.payload.stride(1)),
        Int64(pending.pages.stride(0)),
        Int64(pending.layer * pending.pages.stride(1)),
    )
    state = from_dlpack(initial_state, assumed_align=16).mark_layout_dynamic(
        leading_dim=3
    )
    pdl = pdl_enabled()
    key = (
        q.device.index,
        tuple(x.dtype for x in tensors),
        heads,
        vheads,
        use_qk_l2norm,
        pdl,
    )
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    args = (
        pointers,
        strides,
        Int32(batch),
        Int32(capacity),
        Float32(128**-0.5 if scale is None else scale),
        state,
        stream,
    )
    with torch.cuda.device(q.device):
        if key not in _COMPILED:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("deferred GDN WY must be warmed before capture")
            _COMPILED[key] = cute.compile(
                GdnDeferredWYKernel(heads, vheads, use_qk_l2norm, pdl), *args
            )
        _COMPILED[key](*args)
    return out
