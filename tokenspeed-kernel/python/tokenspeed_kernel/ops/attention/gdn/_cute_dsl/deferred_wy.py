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

"""V32 two-stage GDN WY and accepted replay with packed BF16 MMA.

One producer warp overlaps TMA reads and accepted-state stores with one WY
warp and two replay warps. A fifth warp prepares gates during normalization.
The state tile is read once; high/residual coefficient pairs retain FP32
accuracy in the packed four-term rank-four MMA.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack, make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op
from tokenspeed_kernel.platform import pdl_enabled


@dsl_user_op
def _ldmatrix1(pointer, *, loc=None, ip=None):
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 1),
        [pointer.ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {$0},[$1];",
        "=r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    values = [llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip) for i in range(1)]
    return cute.TensorSSA(
        vector.from_elements(ir.VectorType.get([1], T.i32()), values, loc=loc, ip=ip),
        1,
        Int32,
    )


@dsl_user_op
def _ldmatrix2(pointer, *, loc=None, ip=None):
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [pointer.ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {$0,$1},[$2];",
        "=r,=r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    values = [llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip) for i in range(2)]
    return cute.TensorSSA(
        vector.from_elements(ir.VectorType.get([2], T.i32()), values, loc=loc, ip=ip),
        2,
        Int32,
    )


@dsl_user_op
def _ldmatrix4(pointer, *, loc=None, ip=None):
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [pointer.ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0,$1,$2,$3},[$4];",
        "=r,=r,=r,=r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    values = [llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip) for i in range(4)]
    return cute.TensorSSA(
        vector.from_elements(ir.VectorType.get([4], T.i32()), values, loc=loc, ip=ip),
        4,
        Int32,
    )


@dsl_user_op
def _mma_bf16(a0, a1, a2, a3, b0, b1, c, *, loc=None, ip=None):
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [x.ir_value(loc=loc, ip=ip) for x in (a0, a1, a2, a3, b0, b1)]
        + [c[i].ir_value(loc=loc, ip=ip) for i in range(4)],
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0,$1,$2,$3},{$4,$5,$6,$7},{$8,$9},{$10,$11,$12,$13};",
        ",".join(["=f"] * 4 + ["r"] * 6 + ["f"] * 4),
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    values = [llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip) for i in range(4)]
    return cute.TensorSSA(
        vector.from_elements(ir.VectorType.get([4], T.f32()), values, loc=loc, ip=ip),
        4,
        Float32,
    )


@cute.jit
def _dot_bf16_hilo(a, high, low, lane: Int32, reduction: cutlass.Constexpr):
    accumulator = cute.make_rmem_tensor(4, Float32)
    accumulator.fill(0.0)
    for step in cutlass.range_constexpr(1 if reduction == 8 else reduction // 16):
        a_registers = cute.make_rmem_tensor(4, Int32)
        b_high = cute.make_rmem_tensor(2, Int32)
        b_low = cute.make_rmem_tensor(2, Int32)
        if cutlass.const_expr(reduction == 8):
            pair = cute.make_rmem_tensor(2, Int32)
            pair.store(_ldmatrix2(a.iterator.toint() + ((lane % 16) * a.stride[0]) * 2))
            a_registers[0], a_registers[1] = pair[0], pair[1]
            a_registers[2], a_registers[3] = Int32(0), Int32(0)
            single = cute.make_rmem_tensor(1, Int32)
            single.store(
                _ldmatrix1(high.iterator.toint() + (lane % 8) * high.stride[0] * 2)
            )
            b_high[0], b_high[1] = single[0], Int32(0)
            single.store(
                _ldmatrix1(low.iterator.toint() + (lane % 8) * low.stride[0] * 2)
            )
            b_low[0], b_low[1] = single[0], Int32(0)
        else:
            column = step * 16
            a_registers.store(
                _ldmatrix4(
                    a.iterator.toint()
                    + ((lane % 16) * a.stride[0] + (lane // 16) * 8 + column) * 2
                )
            )
            high_pointer = (
                high.iterator.toint()
                + ((lane % 8) * high.stride[0] + ((lane % 16) // 8) * 8 + column) * 2
            )
            low_pointer = (
                low.iterator.toint()
                + ((lane % 8) * low.stride[0] + ((lane % 16) // 8) * 8 + column) * 2
            )
            b_high.store(_ldmatrix2(high_pointer))
            b_low.store(_ldmatrix2(low_pointer))
        accumulator.store(
            _mma_bf16(
                a_registers[0],
                a_registers[1],
                a_registers[2],
                a_registers[3],
                b_high[0],
                b_high[1],
                accumulator.load(),
            )
        )
        accumulator.store(
            _mma_bf16(
                a_registers[0],
                a_registers[1],
                a_registers[2],
                a_registers[3],
                b_low[0],
                b_low[1],
                accumulator.load(),
            )
        )
    return accumulator.load()


@cute.jit
def _mixed_state_product(h, phi_high, phi_low, key_high, key_low, lane: Int32):
    accumulator = cute.make_rmem_tensor(4, Float32)
    accumulator.fill(0.0)
    row = lane % 8
    high_base = Int32(phi_high.iterator.toint())
    low_base = Int32(phi_low.iterator.toint())
    if row >= 4:
        high_base = Int32(key_high.iterator.toint())
        low_base = Int32(key_low.iterator.toint())
    high_base += (row % 4) * 136 * 2 + ((lane % 16) // 8) * 8 * 2
    low_base += (row % 4) * 136 * 2 + ((lane % 16) // 8) * 8 * 2
    for step in cutlass.range_constexpr(8):
        a = cute.make_rmem_tensor(4, Int32)
        bh = cute.make_rmem_tensor(2, Int32)
        bl = cute.make_rmem_tensor(2, Int32)
        a.store(
            _ldmatrix4(
                h.iterator.toint()
                + ((lane % 16) * 136 + (lane // 16) * 8 + step * 16) * 2
            )
        )
        bh.store(_ldmatrix2(high_base + step * 32))
        bl.store(_ldmatrix2(low_base + step * 32))
        accumulator.store(
            _mma_bf16(a[0], a[1], a[2], a[3], bh[0], bh[1], accumulator.load())
        )
        accumulator.store(
            _mma_bf16(a[0], a[1], a[2], a[3], bl[0], bl[1], accumulator.load())
        )
    return accumulator.load()


@dsl_user_op
def _shared_word(pointer, *, loc=None, ip=None):
    value = llvm.inline_asm(
        T.i32(),
        [pointer.ir_value(loc=loc, ip=ip)],
        "ld.shared.b32 $0,[$1];",
        "=r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Int32(value)


@dsl_user_op
def _write_delta_pair(high_ptr, low_ptr, x0, x1, *, loc=None, ip=None):
    result = llvm.inline_asm(
        T.i32(),
        [x.ir_value(loc=loc, ip=ip) for x in (high_ptr, low_ptr, x0, x1)],
        "{.reg .b32 hp,lp; .reg .b16 h0,h1; .reg .f32 hf0,hf1,l0,l1; cvt.rn.bf16x2.f32 hp,$4,$3; mov.b32 {h0,h1},hp; cvt.f32.bf16 hf0,h0; cvt.f32.bf16 hf1,h1; sub.f32 l0,$3,hf0; sub.f32 l1,$4,hf1; cvt.rn.bf16x2.f32 lp,l1,l0; st.shared.b32 [$1],hp; st.shared.b32 [$2],lp; mov.u32 $0,0;}",
        "=r,r,r,f,f,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def _store_shared_pair(pointer, lo, hi, *, loc=None, ip=None):
    value = llvm.inline_asm(
        T.i32(),
        [x.ir_value(loc=loc, ip=ip) for x in (pointer, lo, hi)],
        "{.reg .b32 p; cvt.rn.bf16x2.f32 p,$3,$2; st.shared.b32 [$1],p; mov.u32 $0,0;}",
        "=r,r,f,f,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Int32(value)


@dsl_user_op
def _transpose_values2(source, target, mask, *, loc=None, ip=None):
    result = llvm.inline_asm(
        T.i32(),
        [
            source.ir_value(loc=loc, ip=ip),
            target.ir_value(loc=loc, ip=ip),
            mask.ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b32 r0,r1; ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {r0,r1},[$1]; and.b32 r0,r0,$3; and.b32 r1,r1,$3; stmatrix.sync.aligned.m8n8.x2.shared.b16 [$2],{r0,r1}; mov.u32 $0,0; }",
        "=r,r,r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def _transpose_values4(source, target, mask, *, loc=None, ip=None):
    result = llvm.inline_asm(
        T.i32(),
        [
            source.ir_value(loc=loc, ip=ip),
            target.ir_value(loc=loc, ip=ip),
            mask.ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b32 r0,r1,r2,r3; ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {r0,r1,r2,r3},[$1]; and.b32 r0,r0,$3; and.b32 r1,r1,$3; and.b32 r2,r2,$3; and.b32 r3,r3,$3; stmatrix.sync.aligned.m8n8.x4.shared.b16 [$2],{r0,r1,r2,r3}; mov.u32 $0,0; }",
        "=r,r,r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def _load_raw_four(pointer, *, loc=None, ip=None):
    code = "{.reg .b32 w0,w1;.reg .b16 e0,e1,e2,e3;ld.shared.v2.b32 {w0,w1},[$4];mov.b32 {e0,e1},w0;mov.b32 {e2,e3},w1;"
    for i in range(4):
        code += f"cvt.f32.bf16 ${i},e{i};"
    code += "}"
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [pointer.ir_value(loc=loc, ip=ip)],
        code,
        "=f,=f,=f,=f,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    values = [llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip) for i in range(4)]
    return cute.TensorSSA(
        vector.from_elements(ir.VectorType.get([4], T.f32()), values, loc=loc, ip=ip),
        4,
        Float32,
    )


@dsl_user_op
def _store_normalized_four(pointer, x0, x1, x2, x3, *, loc=None, ip=None):
    value = llvm.inline_asm(
        T.i32(),
        [x.ir_value(loc=loc, ip=ip) for x in (pointer, x0, x1, x2, x3)],
        "{st.shared.v4.f32 [$1],{$2,$3,$4,$5};mov.u32 $0,0;}",
        "=r,r,f,f,f,f,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Int32(value)


class GdnDeferredWYKernel:
    def __init__(
        self, heads: int, vheads: int, normalize: bool, enable_pdl: bool, tiles: int
    ):
        self.heads = heads
        self.vheads = vheads
        self.normalize = normalize
        self.enable_pdl = enable_pdl
        self.value_tile = 32
        self.stages = 2
        self.heads_per_cta = 1
        self.tiles_per_head = 4
        self.tiles = tiles
        self.consumer_warps = 3
        self.wy_warps = 1
        self.replay_warps = 2
        self.block_threads = 160
        self.resident_blocks = 8
        self.store_group = 4

    @cute.jit
    def __call__(
        self,
        operands: tuple,
        strides: tuple,
        batch: Int32,
        capacity: Int32,
        scale: Float32,
        qkv_tensors,
        state_tensor,
        output_state_tensor,
        stream: cuda.CUstream,
    ):
        atoms, sources = (), ()
        # Q, current K, previous K, current V, previous V.
        for role in cutlass.range_constexpr(5):
            tensor = qkv_tensors[role]
            columns = 128 if role < 3 else self.value_tile
            if cutlass.const_expr(role == 2 or role == 4):
                source = cute.make_tensor(
                    tensor.iterator,
                    cute.make_layout(
                        (
                            tensor.shape[2],
                            tensor.shape[4],
                            tensor.shape[3],
                            tensor.shape[1],
                            tensor.shape[0],
                        ),
                        stride=(
                            tensor.stride[2],
                            tensor.stride[4],
                            tensor.stride[3],
                            tensor.stride[1],
                            tensor.stride[0],
                        ),
                    ),
                )
            else:
                source = cute.make_tensor(
                    tensor.iterator,
                    cute.make_layout(
                        (
                            tensor.shape[1],
                            tensor.shape[3],
                            tensor.shape[2],
                            tensor.shape[0],
                        ),
                        stride=(
                            tensor.stride[1],
                            tensor.stride[3],
                            tensor.stride[2],
                            tensor.stride[0],
                        ),
                    ),
                )
            atom, source = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                source,
                cute.make_layout((4, columns), stride=(columns, 1)),
                (4, columns),
            )
            atoms += (atom,)
            sources += (source,)
        h_layout = cute.make_layout((self.value_tile, 136), stride=(136, 1))
        state_source = cute.make_tensor(
            state_tensor.iterator,
            cute.make_layout(
                (
                    state_tensor.shape[2],
                    state_tensor.shape[3],
                    state_tensor.shape[1],
                    state_tensor.shape[0],
                ),
                stride=(
                    state_tensor.stride[2],
                    state_tensor.stride[3],
                    state_tensor.stride[1],
                    state_tensor.stride[0],
                ),
            ),
        )
        state_atom, state_source = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            state_source,
            h_layout,
            (self.value_tile, 136),
        )
        store_source = cute.make_tensor(
            output_state_tensor.iterator,
            cute.make_layout(
                (
                    output_state_tensor.shape[2],
                    output_state_tensor.shape[3],
                    output_state_tensor.shape[1],
                    output_state_tensor.shape[0],
                ),
                stride=(
                    output_state_tensor.stride[2],
                    output_state_tensor.stride[3],
                    output_state_tensor.stride[1],
                    output_state_tensor.stride[0],
                ),
            ),
        )
        store_atom, store_source = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            store_source,
            h_layout,
            (self.value_tile, 136),
        )
        self.run(
            operands,
            strides,
            capacity,
            scale,
            atoms,
            sources,
            state_atom,
            state_source,
            store_atom,
            store_source,
        ).launch(
            grid=(
                batch * (self.vheads // self.heads_per_cta) * (4 // self.tiles),
                1,
                1,
            ),
            block=(self.block_threads, 1, 1),
            stream=stream,
            use_pdl=self.enable_pdl,
            min_blocks_per_mp=self.resident_blocks,
        )

    @cute.jit
    def prefetch(
        self,
        tile: Int32,
        buffers,
        raw_v,
        barriers,
        atoms: tuple,
        sources: tuple,
        state_atom,
        state_source,
        batch: Int32,
        first_hv: Int32,
        index: Int64,
        old_row: Int32,
        old_bank: Int32,
        first_value_tile: Int32,
    ):
        warp = cute.arch.thread_idx()[0] // 32
        if warp == 0:
            stage = tile % self.stages
            hv = first_hv + tile // self.tiles_per_head
            value_tile = first_value_tile + tile % self.tiles_per_head
            bar = barriers.iterator + stage
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar, self.value_tile * (136 * 2 + 8 * 2)
                )
            state = cute.local_tile(
                state_source, (self.value_tile, 136), (value_tile, 0, hv, index)
            )
            target = buffers[None, None, stage]
            dst, src = cpasync.tma_partition(
                state_atom,
                0,
                cute.make_layout(1),
                cute.group_modes(target, 0, 2),
                cute.group_modes(state, 0, 2),
            )
            cute.copy(state_atom, src, dst, tma_bar_ptr=bar)
            for previous in cutlass.range_constexpr(2):
                role = 4 if previous == 0 else 3
                row_offset = 0 if previous == 0 else 4
                target = cute.make_tensor(
                    raw_v.iterator
                    + stage * 8 * self.value_tile
                    + row_offset * self.value_tile,
                    cute.make_layout((4, self.value_tile), stride=(self.value_tile, 1)),
                )
                if cutlass.const_expr(previous == 0):
                    source = cute.local_tile(
                        sources[role],
                        (4, self.value_tile),
                        (0, value_tile, hv, old_row, old_bank),
                    )
                else:
                    source = cute.local_tile(
                        sources[role], (4, self.value_tile), (0, value_tile, hv, batch)
                    )
                dst, src = cpasync.tma_partition(
                    atoms[role],
                    0,
                    cute.make_layout(1),
                    cute.group_modes(target, 0, 2),
                    cute.group_modes(source, 0, 2),
                )
                cute.copy(atoms[role], src, dst, tma_bar_ptr=bar)

    @cute.jit
    def prepare_queries(
        self,
        q,
        k,
        phi,
        weights,
        decay,
        beta,
        head: cutlass.Constexpr,
        warp: Int32,
        lane: Int32,
    ):
        values = cute.make_rmem_tensor(4, Float32)
        for part in cutlass.range_constexpr(4):
            values[part] = q[warp, lane + part * 32]
        for reverse in cutlass.range_constexpr(8):
            token = 7 - reverse
            weight = Float32(0)
            if token <= warp + 4:
                dot = Float32(0)
                for part in cutlass.range_constexpr(4):
                    dot += values[part] * k[token, lane + part * 32]
                for shift in cutlass.range_constexpr(5):
                    dot += cute.arch.shuffle_sync_bfly(dot, 1 << shift)
                weight = dot * beta[head, token]
                for part in cutlass.range_constexpr(4):
                    values[part] = decay[head, token] * (
                        values[part] - weight * k[token, lane + part * 32]
                    )
            if lane == 0:
                weights[warp, token] = weight
        for part in cutlass.range_constexpr(4):
            phi[warp, lane + part * 32] = values[part]
        cute.arch.barrier(barrier_id=1, number_of_threads=128)

    @cute.jit
    def prepare_basis(
        self,
        k,
        decay,
        beta,
        head: cutlass.Constexpr,
        basis,
        prefixes,
        warp: Int32,
        lane: Int32,
    ):
        values = cute.make_rmem_tensor(4, Float32)
        for part in cutlass.range_constexpr(4):
            values[part] = k[warp, lane + part * 32]
        for following in cutlass.range_constexpr(4):
            if following > warp:
                keys = cute.make_rmem_tensor(4, Float32)
                projection = Float32(0)
                for part in cutlass.range_constexpr(4):
                    keys[part] = k[following, lane + part * 32]
                    projection += values[part] * keys[part]
                for shift in cutlass.range_constexpr(5):
                    projection += cute.arch.shuffle_sync_bfly(projection, 1 << shift)
                weight = projection * beta[head, following]
                for part in cutlass.range_constexpr(4):
                    values[part] = decay[head, following] * (
                        values[part] - weight * keys[part]
                    )
        prefix = Float32(1)
        for t in cutlass.range_constexpr(4):
            if t <= warp:
                prefix *= decay[head, t]
        if lane == 0:
            prefixes[warp] = prefix
        for part in cutlass.range_constexpr(4):
            basis[warp, lane + part * 32] = beta[head, warp] * values[part]
        cute.arch.barrier(barrier_id=1, number_of_threads=128)

    @cute.jit
    def convert_coefficients(self, phi, weights, tid: Int32, k):
        queries = cute.make_rmem_tensor(4, Float32)
        keys = cute.make_rmem_tensor(4, Float32)
        for row in cutlass.range_constexpr(4):
            queries[row] = phi[row, tid]
            keys[row] = k[row, tid]
        replay_keys = cute.make_rmem_tensor(4, Float32)
        for row in cutlass.range_constexpr(4):
            replay_keys[row] = k[row + 4, tid]
        weight = Float32(0)
        if tid < 64:
            weight = weights[tid // 8, tid % 8]
        cute.arch.barrier(barrier_id=1, number_of_threads=128)
        qp = cute.recast_tensor(phi, cutlass.BFloat16)
        kp = cute.recast_tensor(k, cutlass.BFloat16)
        qh = cute.make_tensor(qp.iterator, cute.make_layout((4, 128), stride=(136, 1)))
        ql = cute.make_tensor(
            qp.iterator + 4 * 136, cute.make_layout((4, 128), stride=(136, 1))
        )
        kh = cute.make_tensor(kp.iterator, cute.make_layout((4, 128), stride=(136, 1)))
        kl = cute.make_tensor(
            kp.iterator + 4 * 136, cute.make_layout((4, 128), stride=(136, 1))
        )
        for row in cutlass.range_constexpr(4):
            high = cutlass.BFloat16(queries[row])
            qh[row, tid] = high
            ql[row, tid] = cutlass.BFloat16(queries[row] - Float32(high))
            high = cutlass.BFloat16(keys[row])
            kh[row, tid] = high
            kl[row, tid] = cutlass.BFloat16(keys[row] - Float32(high))
        rh = cute.make_tensor(
            kp.iterator + 8 * 136, cute.make_layout((128, 4), stride=(4, 1))
        )
        rl = cute.make_tensor(
            kp.iterator + 8 * 136 + 128 * 4, cute.make_layout((128, 4), stride=(4, 1))
        )
        for row in cutlass.range_constexpr(4):
            high = cutlass.BFloat16(replay_keys[row])
            rh[tid, row] = high
            rl[tid, row] = cutlass.BFloat16(replay_keys[row] - Float32(high))
        wp = cute.recast_tensor(weights, cutlass.BFloat16)
        if tid < 64:
            high = cutlass.BFloat16(weight)
            wp.iterator[tid] = high
            wp.iterator[tid + 64] = cutlass.BFloat16(weight - Float32(high))
        cute.arch.barrier(barrier_id=1, number_of_threads=128)

    @cute.jit
    def verify(
        self,
        h,
        v,
        phi,
        weights,
        out,
        batch: Int32,
        hv: Int32,
        value_tile: Int32,
        warp: Int32,
        lane: Int32,
        predictions,
        k,
    ):
        qp = cute.recast_tensor(phi, cutlass.BFloat16)
        kp = cute.recast_tensor(k, cutlass.BFloat16)
        qh = cute.make_tensor(qp.iterator, cute.make_layout((4, 128), stride=(136, 1)))
        ql = cute.make_tensor(
            qp.iterator + 4 * 136, cute.make_layout((4, 128), stride=(136, 1))
        )
        kh = cute.make_tensor(kp.iterator, cute.make_layout((4, 128), stride=(136, 1)))
        kl = cute.make_tensor(
            kp.iterator + 4 * 136, cute.make_layout((4, 128), stride=(136, 1))
        )
        wp = cute.recast_tensor(weights, cutlass.BFloat16)
        wh = cute.make_tensor(wp.iterator, cute.make_layout((8, 8), stride=(8, 1)))
        wl = cute.make_tensor(wp.iterator + 64, cute.make_layout((8, 8), stride=(8, 1)))
        for group in cutlass.range_constexpr(self.value_tile // (16 * self.wy_warps)):
            value_base = (warp + group * self.wy_warps) * 16
            hs = cute.make_tensor(
                h.iterator + value_base * 136,
                cute.make_layout((16, 128), stride=(136, 1)),
            )
            vs = cute.make_tensor(
                v.iterator + value_base * 8, cute.make_layout((16, 8), stride=(8, 1))
            )
            state_product = _mixed_state_product(hs, qh, ql, kh, kl, lane)
            states = cute.make_rmem_tensor(4, Float32)
            states.store(state_product)
            result = cute.make_rmem_tensor(4, Float32)
            result.store(state_product + _dot_bf16_hilo(vs, wh, wl, lane, 8))
            for element in cutlass.range_constexpr(4):
                token = (lane % 4) * 2 + element % 2
                value = value_base + lane // 4 + (element // 2) * 8
                if token < 4:
                    out[
                        ((Int64(batch) * 4 + token) * self.vheads + hv) * 128
                        + value_tile * self.value_tile
                        + value
                    ] = out.element_type(result[element])
                else:
                    predictions[token - 4, value] = states[element]

    @cute.jit
    def replay(
        self,
        h,
        v,
        prefixes,
        state_out,
        out_stride: Int64,
        destination: Int64,
        accepted: Int32,
        hv: Int32,
        value_tile: Int32,
        warp: Int32,
        lane: Int32,
        predictions,
        projections_ready,
        phase: Int32,
        k,
    ):
        cute.arch.mbarrier_wait(projections_ready, phase)
        value = warp * 16 + lane % 16
        first_rank = (lane // 16) * 2
        delta = cute.make_rmem_tensor(2, Float32)
        delta.fill(0.0)
        for part in cutlass.range_constexpr(2):
            rank = first_rank + part
            if rank < accepted:
                delta[part] = (
                    Float32(v[rank, value]) - prefixes[rank] * predictions[rank, value]
                )
        cute.arch.barrier(barrier_id=3, number_of_threads=64)
        address = v.iterator.toint() + (value * 4 + first_rank) * 2
        _write_delta_pair(
            address, address + self.value_tile * 4 * 2, delta[0], delta[1]
        )
        cute.arch.barrier(barrier_id=3, number_of_threads=64)
        pair = (lane % 2) * 2
        row = warp * 16 + lane // 4
        address = v.iterator.toint() + (row * 4 + pair) * 2
        if lane % 4 >= 2:
            address += self.value_tile * 4 * 2
        a0 = _shared_word(address)
        a1 = _shared_word(address + 8 * 4 * 2)
        words = cute.recast_tensor(k, cutlass.BFloat16)
        rh = words.iterator.toint() + 8 * 136 * 2
        rl = rh + 128 * 4 * 2
        scale = prefixes[3]
        for group in cutlass.range_constexpr(16 // self.store_group):
            bh = cute.make_rmem_tensor(self.store_group, Int32)
            bl = cute.make_rmem_tensor(self.store_group, Int32)
            accumulators = cute.make_rmem_tensor((4, self.store_group), Float32)
            for part in cutlass.range_constexpr(self.store_group):
                block = group * self.store_group + part
                key = block * 8 + lane // 4
                bh[part] = _shared_word(rh + (key * 4 + pair) * 2)
                bl[part] = _shared_word(rl + (key * 4 + pair) * 2)
                column = block * 8 + (lane % 4) * 2
                accumulators[0, part] = scale * Float32(h[row, column])
                accumulators[1, part] = scale * Float32(h[row, column + 1])
                accumulators[2, part] = scale * Float32(h[row + 8, column])
                accumulators[3, part] = scale * Float32(h[row + 8, column + 1])
            for part in cutlass.range_constexpr(self.store_group):
                accumulators[None, part].store(
                    _mma_bf16(
                        a0,
                        a1,
                        a0,
                        a1,
                        bh[part],
                        bl[part],
                        accumulators[None, part].load(),
                    )
                )
            if destination >= 0:
                for part in cutlass.range_constexpr(self.store_group):
                    block = group * self.store_group + part
                    column = block * 8 + (lane % 4) * 2
                    _store_shared_pair(
                        h.iterator.toint() + (row * 136 + column) * 2,
                        accumulators[0, part],
                        accumulators[1, part],
                    )
                    _store_shared_pair(
                        h.iterator.toint() + ((row + 8) * 136 + column) * 2,
                        accumulators[2, part],
                        accumulators[3, part],
                    )
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier(barrier_id=3, number_of_threads=64)

    @cute.kernel
    def run(
        self,
        operands: tuple,
        strides: tuple,
        capacity: Int32,
        scale: Float32,
        atoms: tuple,
        sources: tuple,
        state_atom,
        state_source,
        store_atom,
        store_source,
    ):
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        tid = cute.arch.thread_idx()[0]
        warp, lane = tid // 32, tid % 32
        group = cute.arch.block_idx()[0]
        first_value_tile = (group % (4 // self.tiles)) * self.tiles
        group = group // (4 // self.tiles)
        groups = self.vheads // self.heads_per_cta
        batch = group // groups
        first_hv = (group % groups) * self.heads_per_cta
        key_head = first_hv // (self.vheads // self.heads)
        tensors = tuple(
            cute.make_tensor(p, cute.make_layout(9223372036854775807)) for p in operands
        )
        (
            a,
            b,
            pa,
            pb,
            A_log,
            dt_bias,
            indices,
            pending_rows,
            accepts,
            pending_pages,
            state_out,
            out,
        ) = tensors
        ast, bst, past, pbst, out_stride, payload_bank_stride, page_bank_stride = (
            strides
        )
        index = Int64(indices[batch])
        code = Int32(pending_rows[batch])
        old_row, old_bank = Int32(0), Int32(0)
        accepted, destination = Int32(0), Int64(-1)
        if code >= 0 and index > 0:
            old_bank = Int32(code >= capacity)
            old_row = code - old_bank * capacity
            page = Int64(pending_pages[Int64(old_bank) * page_bank_stride + old_row])
            if page == index:
                accepted = Int32(accepts[code])
                if accepted < 0:
                    accepted = Int32(0)
                if accepted > 4:
                    accepted = Int32(4)
                if accepted > 0:
                    destination = page
        smem = cutlass.utils.SmemAllocator()
        raw_qk = smem.allocate_tensor(
            cutlass.BFloat16,
            cute.make_layout((4, 128, 3), stride=(128, 1, 512)),
            byte_alignment=128,
        )
        predictions = smem.allocate_tensor(
            Float32,
            cute.make_layout(
                (4, self.value_tile, self.stages),
                stride=(self.value_tile + 8, 1, 4 * (self.value_tile + 8)),
            ),
        )
        qk_bar = smem.allocate_tensor(Int64, cute.make_layout(1), byte_alignment=8)
        state = smem.allocate_tensor(
            cutlass.BFloat16,
            cute.make_layout(
                (self.value_tile, 136, self.stages),
                stride=(136, 1, self.value_tile * 136),
            ),
            byte_alignment=128,
        )
        raw_v = smem.allocate_tensor(
            cutlass.BFloat16,
            cute.make_layout(
                (8, self.value_tile, self.stages),
                stride=(self.value_tile, 1, 8 * self.value_tile),
            ),
            byte_alignment=128,
        )
        values = cute.make_tensor(
            raw_v.iterator,
            cute.make_layout(
                (8, self.value_tile, self.stages), stride=(1, 8, 8 * self.value_tile)
            ),
        )
        bars = smem.allocate_tensor(
            Int64, cute.make_layout(self.stages), byte_alignment=8
        )
        free_bars = smem.allocate_tensor(
            Int64, cute.make_layout(self.stages), byte_alignment=8
        )
        projections_ready = smem.allocate_tensor(
            Int64, cute.make_layout(self.stages), byte_alignment=8
        )
        gates_ready = smem.allocate_tensor(Int64, cute.make_layout(1), byte_alignment=8)
        coefficients_ready = smem.allocate_tensor(
            Int64, cute.make_layout(1), byte_alignment=8
        )
        # Raw Q/K is dead once all four warps have loaded their registers.
        q = cute.make_tensor(
            cute.recast_tensor(raw_qk, Float32).iterator,
            cute.make_layout((4, 136), stride=(136, 1)),
        )
        key_alignment = smem.allocate_tensor(Float32, cute.make_layout(16))
        k = smem.allocate_tensor(
            Float32, cute.make_layout((8, 136), stride=(136, 1)), byte_alignment=64
        )
        replay_basis = cute.make_tensor(
            k.iterator + 4 * 136,
            cute.make_layout((4, 128, 1), stride=(136, 1, 4 * 136)),
        )
        prefixes = smem.allocate_tensor(
            Float32, cute.make_layout((4, 1), stride=(1, 4))
        )
        phi = cute.make_tensor(
            q.iterator, cute.make_layout((4, 136, 1), stride=(136, 1, 4 * 136))
        )
        weights = smem.allocate_tensor(
            Float32, cute.make_layout((8, 8, self.heads_per_cta), stride=(8, 1, 64))
        )
        decay = smem.allocate_tensor(
            Float32, cute.make_layout((self.heads_per_cta, 8), stride=(8, 1))
        )
        beta = smem.allocate_tensor(
            Float32, cute.make_layout((self.heads_per_cta, 8), stride=(8, 1))
        )
        for item in cutlass.range_constexpr(
            (64 * self.heads_per_cta + self.block_threads - 1) // self.block_threads
        ):
            offset = tid + item * self.block_threads
            if offset < 64 * self.heads_per_cta:
                weights[(offset % 64) // 8, offset % 8, offset // 64] = Float32(0)
        if warp == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(qk_bar.iterator, 1)
                cute.arch.mbarrier_init(gates_ready.iterator, 1)
                cute.arch.mbarrier_init(coefficients_ready.iterator, 1)
                for stage in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(bars.iterator + stage, 1)
                    cute.arch.mbarrier_init(
                        free_bars.iterator + stage, self.consumer_warps
                    )
                    cute.arch.mbarrier_init(
                        projections_ready.iterator + stage, self.wy_warps
                    )
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        if warp == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    qk_bar.iterator, 3 * 4 * 128 * 2
                )
            for role in cutlass.range_constexpr(3):
                target = raw_qk[None, None, role]
                if cutlass.const_expr(role == 2):
                    source = cute.local_tile(
                        sources[role], (4, 128), (0, 0, key_head, old_row, old_bank)
                    )
                else:
                    source = cute.local_tile(
                        sources[role], (4, 128), (0, 0, key_head, batch)
                    )
                dst, src = cpasync.tma_partition(
                    atoms[role],
                    0,
                    cute.make_layout(1),
                    cute.group_modes(target, 0, 2),
                    cute.group_modes(source, 0, 2),
                )
                cute.copy(atoms[role], src, dst, tma_bar_ptr=qk_bar.iterator)
        for tile in cutlass.range_constexpr(self.stages):
            if cutlass.const_expr(tile < self.tiles):
                self.prefetch(
                    tile,
                    state,
                    raw_v,
                    bars,
                    atoms,
                    sources,
                    state_atom,
                    state_source,
                    batch,
                    first_hv,
                    index,
                    old_row,
                    old_bank,
                    first_value_tile,
                )
        if (warp == self.consumer_warps + 1) & (lane < 8):
            head, token = Int32(0), lane
            hv = first_hv + head
            gate, bt = Float32(0), Float32(0)
            if (token >= 4) | (token < accepted):
                av, bv = Float32(0), Float32(0)
                if token < 4:
                    av = Float32(
                        pa[
                            Int64(old_bank) * payload_bank_stride
                            + Int64(old_row) * past[0]
                            + token * past[1]
                            + hv * past[2]
                        ]
                    )
                    bv = Float32(
                        pb[
                            Int64(old_bank) * payload_bank_stride
                            + Int64(old_row) * pbst[0]
                            + token * pbst[1]
                            + hv * pbst[2]
                        ]
                    )
                else:
                    av = Float32(
                        a[Int64(batch) * ast[0] + (token - 4) * ast[1] + hv * ast[2]]
                    )
                    bv = Float32(
                        b[Int64(batch) * bst[0] + (token - 4) * bst[1] + hv * bst[2]]
                    )
                x = av + Float32(dt_bias[hv])
                sp = x
                if x <= 20.0:
                    sp = cute.math.log(
                        1.0 + cute.math.exp(x, fastmath=True), fastmath=True
                    )
                gate = -cute.math.exp(Float32(A_log[hv]), fastmath=True) * sp
                gate = cute.arch.fmax(gate, Float32(-1e30))
                bt = 1.0 / (1.0 + cute.math.exp(-bv, fastmath=True))
            decay[head, token] = cute.math.exp(gate, fastmath=True)
            beta[head, token] = bt
        cute.arch.mbarrier_wait(qk_bar.iterator, 0)
        if warp < 4:
            rq = cute.make_rmem_tensor(4, Float32)
            rk = cute.make_rmem_tensor(4, Float32)
            rp = cute.make_rmem_tensor(4, Float32)
            sum_q, sum_k, sum_p = Float32(0), Float32(0), Float32(0)
            address = raw_qk.iterator.toint() + (warp * 128 + lane * 4) * 2
            rq.store(_load_raw_four(address))
            rk.store(_load_raw_four(address + 4 * 128 * 2))
            rp.fill(0.0)
            if warp < accepted:
                rp.store(_load_raw_four(address + 8 * 128 * 2))
            for part in cutlass.range_constexpr(4):
                sum_q += rq[part] * rq[part]
                sum_k += rk[part] * rk[part]
                sum_p += rp[part] * rp[part]
            for shift in cutlass.range_constexpr(5):
                sum_q += cute.arch.shuffle_sync_bfly(sum_q, 1 << shift)
                sum_k += cute.arch.shuffle_sync_bfly(sum_k, 1 << shift)
                sum_p += cute.arch.shuffle_sync_bfly(sum_p, 1 << shift)
            iq, ik, ip = scale, Float32(1), Float32(1)
            if cutlass.const_expr(self.normalize):
                iq *= cute.math.rsqrt(sum_q + 1e-6, fastmath=True)
                ik = cute.math.rsqrt(sum_k + 1e-6, fastmath=True)
                ip = cute.math.rsqrt(sum_p + 1e-6, fastmath=True)
            # Raw inputs may be overwritten only after every normalization warp
            # has loaded its complete row into registers.
            cute.arch.barrier(barrier_id=1, number_of_threads=128)
            _store_normalized_four(
                q.iterator.toint() + (warp * 136 + lane * 4) * 4,
                rq[0] * iq,
                rq[1] * iq,
                rq[2] * iq,
                rq[3] * iq,
            )
            _store_normalized_four(
                k.iterator.toint() + ((warp + 4) * 136 + lane * 4) * 4,
                rk[0] * ik,
                rk[1] * ik,
                rk[2] * ik,
                rk[3] * ik,
            )
            _store_normalized_four(
                k.iterator.toint() + (warp * 136 + lane * 4) * 4,
                rp[0] * ip,
                rp[1] * ip,
                rp[2] * ip,
                rp[3] * ip,
            )
        cute.arch.sync_threads()
        # All head coefficients are ready before consumers start independent loops.
        if warp < 4:
            for head in cutlass.range_constexpr(self.heads_per_cta):
                self.prepare_queries(
                    q,
                    k,
                    phi[None, None, head],
                    weights[None, None, head],
                    decay,
                    beta,
                    head,
                    warp,
                    lane,
                )
            for head in cutlass.range_constexpr(self.heads_per_cta):
                self.prepare_basis(
                    k,
                    decay,
                    beta,
                    head,
                    replay_basis[None, None, head],
                    prefixes[None, head],
                    warp,
                    lane,
                )
            for head in cutlass.range_constexpr(self.heads_per_cta):
                self.convert_coefficients(
                    phi[None, None, head], weights[None, None, head], tid, k
                )
        cute.arch.sync_threads()
        if warp == 0:
            for tile in cutlass.range(self.tiles):
                stage = tile % self.stages
                head = tile // self.tiles_per_head
                value_tile = first_value_tile + tile % self.tiles_per_head
                hv = first_hv + head
                cute.arch.mbarrier_wait(
                    free_bars.iterator + stage, (tile // self.stages) % 2
                )
                if destination >= 0:
                    target = cute.local_tile(
                        store_source,
                        (self.value_tile, 136),
                        (value_tile, 0, hv, destination),
                    )
                    shared, target = cpasync.tma_partition(
                        store_atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(state[None, None, stage], 0, 2),
                        cute.group_modes(target, 0, 2),
                    )
                    with cute.arch.elect_one():
                        cute.copy(store_atom, shared, target)
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                if tile + self.stages < self.tiles:
                    self.prefetch(
                        tile + self.stages,
                        state,
                        raw_v,
                        bars,
                        atoms,
                        sources,
                        state_atom,
                        state_source,
                        batch,
                        first_hv,
                        index,
                        old_row,
                        old_bank,
                        first_value_tile,
                    )
            with cute.arch.elect_one():
                cute.arch.cp_async_bulk_wait_group(0)
        elif warp < 1 + self.wy_warps:
            wy_warp = warp - 1
            for tile in cutlass.range(self.tiles):
                stage = tile % self.stages
                head = tile // self.tiles_per_head
                value_tile = first_value_tile + tile % self.tiles_per_head
                hv = first_hv + head
                cute.arch.mbarrier_wait(
                    bars.iterator + stage, (tile // self.stages) % 2
                )
                if wy_warp == 0:
                    token = (lane % 4) * 2
                    mask = Int32(0)
                    if (token >= 4) | (token < accepted):
                        mask = mask | Int32(65535)
                    if (token + 1 >= 4) | (token + 1 < accepted):
                        mask = mask | Int32(-65536)
                    if cutlass.const_expr(self.value_tile == 16):
                        source = (
                            raw_v.iterator.toint()
                            + (
                                stage * 8 * self.value_tile
                                + (lane % 8) * self.value_tile
                                + ((lane % 16) // 8) * 8
                            )
                            * 2
                        )
                        target = (
                            values.iterator.toint()
                            + (stage * 8 * self.value_tile + (lane % 16) * 8) * 2
                        )
                        _transpose_values2(source, target, mask)
                    else:
                        for chunk in cutlass.range_constexpr(self.value_tile // 32):
                            source = (
                                raw_v.iterator.toint()
                                + (
                                    stage * 8 * self.value_tile
                                    + (lane % 8) * self.value_tile
                                    + (lane // 8) * 8
                                    + chunk * 32
                                )
                                * 2
                            )
                            target = (
                                values.iterator.toint()
                                + (
                                    stage * 8 * self.value_tile
                                    + (lane + chunk * 32) * 8
                                )
                                * 2
                            )
                            _transpose_values4(source, target, mask)
                if cutlass.const_expr(self.wy_warps == 1):
                    cute.arch.sync_warp()
                else:
                    cute.arch.barrier(
                        barrier_id=2, number_of_threads=32 * self.wy_warps
                    )
                self.verify(
                    state[None, None, stage],
                    values[None, None, stage],
                    phi[None, None, head],
                    weights[None, None, head],
                    out,
                    batch,
                    hv,
                    value_tile,
                    wy_warp,
                    lane,
                    predictions[None, None, stage],
                    k,
                )
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(projections_ready.iterator + stage)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(free_bars.iterator + stage)
        elif warp < 1 + self.consumer_warps:
            replay_warp = warp - 1 - self.wy_warps
            for tile in cutlass.range(self.tiles):
                stage = tile % self.stages
                head = tile // self.tiles_per_head
                value_tile = first_value_tile + tile % self.tiles_per_head
                hv = first_hv + head
                cute.arch.mbarrier_wait(
                    bars.iterator + stage, (tile // self.stages) % 2
                )
                self.replay(
                    state[None, None, stage],
                    values[None, None, stage],
                    prefixes[None, head],
                    state_out,
                    out_stride,
                    destination,
                    accepted,
                    hv,
                    value_tile,
                    replay_warp,
                    lane,
                    predictions[None, None, stage],
                    projections_ready.iterator + stage,
                    (tile // self.stages) % 2,
                    k,
                )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(free_bars.iterator + stage)


_COMPILED: dict[tuple, dict[tuple, object]] = {}
_DTYPES = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
    torch.float32: Float32,
    torch.int32: Int32,
    torch.int64: Int64,
}


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
    """Read a pending prefix by bank/row, commit it, and verify current T4 rows."""
    batch, tokens, heads, dim = q.shape
    vheads = v.shape[2]
    if (
        tokens != 4
        or dim != 128
        or v.shape[-1] != 128
        or initial_state.dtype != torch.bfloat16
    ):
        raise ValueError("pending replay requires T4 and BF16 K128/V128 state")
    row_stride = initial_state.stride(0)
    if (
        initial_state.stride()[1:] != (128 * 128, 128, 1)
        or row_stride < vheads * 128 * 128
    ):
        raise ValueError("pending replay requires dense K-last state rows")
    if initial_state.data_ptr() % 16 or row_stride * initial_state.element_size() % 16:
        raise ValueError("pending replay requires 16-byte aligned state rows")

    capacity = pending.steps.shape[1]
    out = q.new_empty((batch, 4, vheads, 128))
    state = from_dlpack(initial_state, assumed_align=16).mark_layout_dynamic(
        leading_dim=3
    )
    if (
        pending.payload.stride(-1) == 1
        and pending.payload.shape[-1] * pending.payload.element_size() % 16 == 0
        and A_log.dtype == dt_bias.dtype == torch.float32
        and all(
            x.stride(-1) == 1
            and x.data_ptr() % 16 == 0
            and all(s > 0 and s * x.element_size() % 16 == 0 for s in x.stride()[:-1])
            for x in (q, k, v)
        )
    ):
        kernel = GdnDeferredWYKernel
        tile_configs = ((1,), (4,))
        tile_config = (1,) if batch * vheads < 768 else (4,)
        width = pending.payload.shape[-1]
        bank_stride = pending.payload.stride(0)
        layer_offset = (
            pending.payload.storage_offset() + pending.layer * pending.payload.stride(1)
        )
        token_strides = (bank_stride, 4 * width, width, 128, 1)
        gate_strides = (bank_stride, 4 * width, width, 1)
        previous_k = pending.payload.as_strided(
            (2, capacity, 4, heads, 128), token_strides, layer_offset
        )
        previous_v = pending.payload.as_strided(
            (2, capacity, 4, vheads, 128), token_strides, layer_offset + heads * 128
        )
        gate_offset = layer_offset + (heads + vheads) * 128
        previous_a = pending.payload.as_strided(
            (2, capacity, 4, vheads), gate_strides, gate_offset
        )
        previous_b = pending.payload.as_strided(
            (2, capacity, 4, vheads), gate_strides, gate_offset + vheads
        )
        tensors = (
            a,
            b,
            previous_a,
            previous_b,
            A_log,
            dt_bias,
            initial_state_indices,
            pending.rows,
            pending.steps,
            pending.pages[:, pending.layer],
            initial_state,
            out,
        )
        strides = tuple(tuple(Int64(s) for s in x.stride()) for x in (a, b)) + (
            (Int64(4 * width), Int64(width), Int64(1)),
            (Int64(4 * width), Int64(width), Int64(1)),
            Int64(row_stride),
            Int64(bank_stride),
            Int64(pending.pages.stride(0)),
        )
        qkv = tuple(
            from_dlpack(x, assumed_align=16).mark_layout_dynamic(leading_dim=x.ndim - 1)
            for x in (q, k, previous_k, v, previous_v)
        )
        layouts = (qkv, state, state)
    else:
        from tokenspeed_kernel.ops.attention.gdn._cute_dsl._deferred_wy_fallback import (
            GdnDeferredWYStridedKernel,
        )

        kernel = GdnDeferredWYStridedKernel
        tile_configs = ((),)
        tile_config = ()
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
        strides = tuple(
            tuple(Int64(s) for s in x.stride()) for x in (q, k, v, a, b)
        ) + (
            Int64(pending.payload.stride(0)),
            Int64(pending.layer * pending.payload.stride(1)),
            Int64(pending.pages.stride(0)),
            Int64(pending.layer * pending.pages.stride(1)),
        )
        layouts = (state,)

    pointers = tuple(
        make_ptr(
            _DTYPES[x.dtype],
            x.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=x.element_size(),
        )
        for x in tensors
    )
    pdl = pdl_enabled()
    kernel_config = (heads, vheads, use_qk_l2norm, pdl)
    key = (
        kernel,
        q.device.index,
        *kernel_config,
        tuple(x.dtype for x in tensors),
    )
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    args = (
        pointers,
        strides,
        Int32(batch),
        Int32(capacity),
        Float32(128**-0.5 if scale is None else scale),
        *layouts,
        stream,
    )
    with torch.cuda.device(q.device):
        if key not in _COMPILED:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("deferred GDN WY must be warmed before capture")
            # Warm both fixed tile counts together. Small batches need more
            # CTAs to occupy the GPU, without specializing on their row counts.
            _COMPILED[key] = {
                config: cute.compile(kernel(*kernel_config, *config), *args)
                for config in tile_configs
            }
        _COMPILED[key][tile_config](*args)
    return out
