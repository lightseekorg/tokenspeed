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

"""Output-only GDN WY verification using warp-level TF32 MMA."""

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
from tokenspeed_kernel.ops.tuning import is_warming_up
from tokenspeed_kernel.platform import pdl_enabled


@dsl_user_op
def _copy_h_16bytes(dst, src, source_bytes, *, loc=None, ip=None):
    result = llvm.inline_asm(
        T.i32(),
        [
            dst.ir_value(loc=loc, ip=ip),
            src.ir_value(loc=loc, ip=ip),
            source_bytes.ir_value(loc=loc, ip=ip),
        ],
        "{ cp.async.cg.shared.global [$1], [$2], 16, $3; mov.u32 $0, 0; }",
        "=r,r,l,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def _load_four(pointer, is_bf16, *, loc=None, ip=None):
    kind = "bf16" if is_bf16 else "f16"
    code = "{ .reg .b32 lo, hi; .reg .b16 e0,e1,e2,e3;\n"
    code += "ld.global.v2.b32 {lo,hi}, [$4];\n"
    code += "mov.b32 {e0,e1},lo; mov.b32 {e2,e3},hi;\n"
    for i in range(4):
        code += f"cvt.f32.{kind} ${i},e{i};\n"
    code += "}"
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [pointer.ir_value(loc=loc, ip=ip)],
        code,
        "=f,=f,=f,=f,l,~{memory}",
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
def _mma_tf32(a0, a1, a2, a3, b0, b1, c, exact_a, exact_b, *, loc=None, ip=None):
    names = ("a0", "a1", "a2", "a3", "b0", "b1")
    code = f"{{\n.reg .b32 {','.join(names)};\n"
    for i, name in enumerate(names):
        exact = exact_a if i < 4 else exact_b
        code += (
            f"mov.b32 {name}, ${i+4};\n"
            if exact
            else f"cvt.rn.tf32.f32 {name}, ${i+4};\n"
        )
    code += "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
    code += "{$0,$1,$2,$3}, {a0,a1,a2,a3}, {b0,b1}, {$10,$11,$12,$13};\n}"
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [x.ir_value(loc=loc, ip=ip) for x in (a0, a1, a2, a3, b0, b1)]
        + [c[i].ir_value(loc=loc, ip=ip) for i in range(4)],
        code,
        ",".join(["=f"] * 4 + ["f"] * 10),
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


@dsl_user_op
def _mma_gram_tf32(k0, q0, k1, q1, c, *, loc=None, ip=None):
    code = "{ .reg .b32 k0,q0,k1,q1;\n"
    for i, name in enumerate(("k0", "q0", "k1", "q1")):
        code += f"cvt.rn.tf32.f32 {name}, ${i+4};\n"
    code += "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
    code += "{$0,$1,$2,$3}, {k0,q0,k1,q1}, {k0,k1}, {$8,$9,$10,$11}; }"
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [x.ir_value(loc=loc, ip=ip) for x in (k0, q0, k1, q1)]
        + [c[i].ir_value(loc=loc, ip=ip) for i in range(4)],
        code,
        ",".join(["=f"] * 4 + ["f"] * 8),
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
def _gram_eight(q, k, lane: Int32):
    partials = cute.make_rmem_tensor((4, 4), Float32)
    partials.fill(0.0)
    row = lane // 4
    for step in cutlass.range_constexpr(4):
        for accumulator in cutlass.range_constexpr(4):
            base = step * 8 + lane % 4 + accumulator * 32
            partials[None, accumulator].store(
                _mma_gram_tf32(
                    k[row, base],
                    q[row, base],
                    k[row, base + 4],
                    q[row, base + 4],
                    partials[None, accumulator],
                )
            )
    result = partials[None, 0].load()
    for accumulator in cutlass.range_constexpr(1, 4):
        result += partials[None, accumulator].load()
    return result


@cute.jit
def _dot_tf32(
    a: cute.Tensor,
    b: cute.Tensor,
    lane: Int32,
    column: Int32,
    reduction: cutlass.Constexpr,
    transpose_b: cutlass.Constexpr,
    accumulators: cutlass.Constexpr,
    initial,
):
    count = accumulators if reduction == 128 else 1
    partials = cute.make_rmem_tensor((4, count), Float32)
    partials.fill(0.0)
    if cutlass.const_expr(initial is not None):
        partials[None, 0].store(initial)
    row = lane // 4
    for step in cutlass.range_constexpr(reduction // (8 * count)):
        for accumulator in cutlass.range_constexpr(count):
            base = step * 8 + lane % 4 + accumulator * (reduction // count)
            if cutlass.const_expr(transpose_b):
                b0, b1 = Float32(b[base, column + row]), Float32(
                    b[base + 4, column + row]
                )
            else:
                b0, b1 = Float32(0), Float32(0)
                if cutlass.const_expr(a.shape[0] == 8):
                    if column + row < 8:
                        b0, b1 = Float32(b[column + row, base]), Float32(
                            b[column + row, base + 4]
                        )
                else:
                    b0, b1 = Float32(b[column + row, base]), Float32(
                        b[column + row, base + 4]
                    )
            a1, a3 = Float32(0), Float32(0)
            if cutlass.const_expr(a.shape[0] >= 16):
                a1, a3 = Float32(a[row + 8, base]), Float32(a[row + 8, base + 4])
            partials[None, accumulator].store(
                _mma_tf32(
                    Float32(a[row, base]),
                    a1,
                    Float32(a[row, base + 4]),
                    a3,
                    b0,
                    b1,
                    partials[None, accumulator],
                    a.element_type.width == 16,
                    b.element_type.width == 16,
                )
            )
    result = partials[None, 0].load()
    for accumulator in cutlass.range_constexpr(1, count):
        result += partials[None, accumulator].load()
    return result


@cute.jit
def _store_fragment(c: cute.Tensor, target: cute.Tensor, lane: Int32, column: Int32):
    row = lane // 4
    col = column + (lane % 4) * 2
    target[row, col], target[row, col + 1] = c[0], c[1]
    target[row + 8, col], target[row + 8, col + 1] = c[2], c[3]


class GdnWYKernel:
    """One CTA per request, value head and value tile.

    With M = beta * D_strict * KK and G the root-path gate sum, compute
    W = (D * QK) (I + M)^-1 diag(beta), then
    O = (exp(G) Q - W exp(G) K) H0^T + W V. This avoids a per-token
    state and a separate KH0 product. H0 is stored in K-last [V, K] order.
    """

    def __init__(
        self,
        heads: int,
        vheads: int,
        normalize: bool,
        tree: bool,
        replay: bool,
        enable_pdl: bool,
        accumulators: int,
        value_tile: int,
        token_bound: int,
        use_tma_qkv: bool,
        use_tma_state: bool,
        gram_rows: int,
        key_chunk: int,
    ):
        self.heads = heads
        self.vheads = vheads
        self.normalize = normalize
        self.tree = tree
        self.replay = replay
        self.enable_pdl = enable_pdl
        self.accumulators = accumulators
        self.value_tile = value_tile
        self.token_bound = token_bound
        self.use_tma_qkv = use_tma_qkv
        self.use_tma_state = use_tma_state
        self.gram_rows = gram_rows
        self.key_chunk = key_chunk
        self.reduction_bound = 8 if token_bound <= 8 else 16

    @cute.jit
    def __call__(
        self,
        operands: tuple,
        strides: tuple,
        batch: Int32,
        tokens: Int32,
        scale: Float32,
        qkv_tensors,
        state_tensor,
        stream: cuda.CUstream,
    ):
        atoms, sources = (), ()
        if cutlass.const_expr(self.use_tma_qkv):
            for operand in cutlass.range_constexpr(3):
                tensor = qkv_tensors[operand]
                columns = 128 if operand < 2 else self.value_tile
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
                    cute.make_layout((self.token_bound, columns), stride=(columns, 1)),
                    (self.token_bound, columns),
                )
                atoms += (atom,)
                sources += (source,)
        state_atom, state_source, h_layout = None, None, None
        if cutlass.const_expr(self.use_tma_state):
            h_stride = self.key_chunk + 128 // state_tensor.element_type.width
            h_layout = cute.make_layout(
                (self.value_tile, h_stride), stride=(h_stride, 1)
            )
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
                (self.value_tile, h_stride),
            )
        self.run(
            operands,
            strides,
            tokens,
            scale,
            atoms,
            sources,
            state_atom,
            state_source,
            h_layout,
        ).launch(
            grid=(batch * self.vheads, 128 // self.value_tile, 1),
            block=(128, 1, 1),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.jit
    def load_state(
        self,
        pool,
        pool_stride,
        idx,
        hv,
        tile,
        tid,
        warp,
        shared,
        barrier,
        atom,
        source,
        key_offset: cutlass.Constexpr,
        initialize: cutlass.Constexpr,
    ):
        h_stride = self.key_chunk + 128 // pool.element_type.width
        element_bytes = pool.element_type.width // 8
        col_base = tile * self.value_tile
        if cutlass.const_expr(self.use_tma_state):
            if warp == 0:
                if cutlass.const_expr(initialize):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_init(barrier.iterator, 1)
                    cute.arch.mbarrier_init_fence()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        barrier.iterator, self.value_tile * h_stride * element_bytes
                    )
                tile_source = source
                if cutlass.const_expr(key_offset != 0):
                    tile_source = cute.domain_offset((0, key_offset, 0, 0), source)
                tile_state = cute.local_tile(
                    tile_source, (self.value_tile, h_stride), (tile, 0, hv, idx)
                )
                dst, src = cpasync.tma_partition(
                    atom,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(shared, 0, 2),
                    cute.group_modes(tile_state, 0, 2),
                )
                cute.copy(atom, src, dst, tma_bar_ptr=barrier.iterator)
        else:
            source_row = Int64(0)
            if idx >= 0:
                source_row = idx
            h_base = (
                pool.iterator.toint()
                + (
                    source_row * pool_stride
                    + hv * 128 * 128
                    + col_base * 128
                    + key_offset
                )
                * element_bytes
            )
            if (h_base & 15) == 0:
                source_bytes = Int32(0)
                if idx >= 0:
                    source_bytes = Int32(16)
                vector_elements = 128 // pool.element_type.width
                copies = self.value_tile * self.key_chunk // (128 * vector_elements)
                for i in cutlass.range_constexpr(copies):
                    offset = (tid + i * 128) * vector_elements
                    row, column = offset // self.key_chunk, offset % self.key_chunk
                    dst = (
                        shared.iterator.toint()
                        + (row * h_stride + column) * element_bytes
                    )
                    src = h_base + (row * 128 + column) * element_bytes
                    _copy_h_16bytes(dst, src, source_bytes)
                    if cutlass.const_expr((i + 1) % 8 == 0):
                        cute.arch.cp_async_commit_group()
                if cutlass.const_expr(copies % 8 != 0):
                    cute.arch.cp_async_commit_group()
            else:
                for i in cutlass.range_constexpr(
                    self.value_tile * self.key_chunk // 128
                ):
                    offset = tid + i * 128
                    row, column = offset // self.key_chunk, offset % self.key_chunk
                    value = Float32(0)
                    if idx >= 0:
                        value = Float32(
                            pool[
                                idx * pool_stride
                                + hv * 128 * 128
                                + (col_base + row) * 128
                                + key_offset
                                + column
                            ]
                        )
                    shared[row, column] = shared.element_type(value)

    @cute.kernel
    def run(
        self,
        operands: tuple,
        strides: tuple,
        tokens: Int32,
        scale: Float32,
        atoms: tuple,
        sources: tuple,
        state_atom,
        state_source,
        h_layout,
    ):
        tid = cute.arch.thread_idx()[0]
        warp, lane = tid // 32, tid % 32
        nh, tile, _ = cute.arch.block_idx()
        batch, hv = nh // self.vheads, nh % self.vheads
        head = hv // (self.vheads // self.heads)
        col_base = tile * self.value_tile
        tensors = tuple(
            cute.make_tensor(p, cute.make_layout(9223372036854775807)) for p in operands
        )
        q, k, v, a, b, A_log, dt_bias, pool, indices, tree, replay, out = tensors
        qs, ks, vs, as_, bs, ps = strides
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        smem = cutlass.utils.SmemAllocator()
        # Short-window MMA reads eight padded token rows, never sixteen.
        workspace_rows = 8 if self.token_bound <= 8 else 16
        sQ = smem.allocate_tensor(
            Float32, cute.make_layout((workspace_rows, 128), stride=(132, 1))
        )
        sK = smem.allocate_tensor(
            Float32, cute.make_layout((workspace_rows, 128), stride=(132, 1))
        )
        h_stride = self.key_chunk + 128 // pool.element_type.width
        state_barrier = None
        if cutlass.const_expr(self.use_tma_state):
            # TMA zero-fills the columns beyond K128 into the ordinary row padding.
            sH = smem.allocate_tensor(pool.element_type, h_layout, byte_alignment=128)
            state_barrier = smem.allocate_tensor(
                Int64, cute.make_layout(1), byte_alignment=8
            )
        else:
            sH = smem.allocate_tensor(
                pool.element_type,
                cute.make_layout(
                    (self.value_tile, self.key_chunk), stride=(h_stride, 1)
                ),
                byte_alignment=16,
            )
        sV = smem.allocate_tensor(
            Float32,
            cute.make_layout(
                (workspace_rows, self.value_tile), stride=(self.value_tile + 4, 1)
            ),
        )
        sQK = smem.allocate_tensor(
            Float32,
            cute.make_layout(
                (self.gram_rows, self.gram_rows), stride=(self.gram_rows, 1)
            ),
        )
        sG = smem.allocate_tensor(Float32, cute.make_layout(16))
        sBeta = smem.allocate_tensor(Float32, cute.make_layout(16))
        sBits = smem.allocate_tensor(Int64, cute.make_layout(16))
        if cutlass.const_expr(self.use_tma_qkv):
            raw_layout = cute.make_layout((self.token_bound, 128), stride=(128, 1))
            raw_q = smem.allocate_tensor(q.element_type, raw_layout, byte_alignment=128)
            raw_k = smem.allocate_tensor(k.element_type, raw_layout, byte_alignment=128)
            raw_v = smem.allocate_tensor(
                v.element_type,
                cute.make_layout(
                    (self.token_bound, self.value_tile), stride=(self.value_tile, 1)
                ),
                byte_alignment=128,
            )
            qkv_barrier = smem.allocate_tensor(
                Int64, cute.make_layout(1), byte_alignment=8
            )
            if warp == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(qkv_barrier.iterator, 1)
                cute.arch.mbarrier_init_fence()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        qkv_barrier.iterator,
                        (2 * 128 + self.value_tile) * self.token_bound * 2,
                    )
                for i in cutlass.range_constexpr(3):
                    destination = raw_q if i == 0 else raw_k if i == 1 else raw_v
                    columns = 128 if i < 2 else self.value_tile
                    source_head = head if i < 2 else hv
                    source_tile = 0 if i < 2 else tile
                    tile_qk = cute.local_tile(
                        sources[i],
                        (self.token_bound, columns),
                        (0, source_tile, source_head, batch),
                    )
                    dst, src = cpasync.tma_partition(
                        atoms[i],
                        0,
                        cute.make_layout(1),
                        cute.group_modes(destination, 0, 2),
                        cute.group_modes(tile_qk, 0, 2),
                    )
                    cute.copy(atoms[i], src, dst, tma_bar_ptr=qkv_barrier.iterator)
        width = self.heads * 128 + self.vheads * 128 + 2 * self.vheads
        # Keep raw gate reads independent of the following Q/K normalization.
        if cutlass.const_expr(self.token_bound == 4 and self.value_tile == 64):
            prefetched_a = a.element_type(0)
            prefetched_b = b.element_type(0)
            prefetched_log = A_log.element_type(0)
            prefetched_bias = dt_bias.element_type(0)
            if tid < tokens:
                prefetched_a = a[Int64(batch) * as_[0] + tid * as_[1] + hv * as_[2]]
                prefetched_b = b[Int64(batch) * bs[0] + tid * bs[1] + hv * bs[2]]
                prefetched_log = A_log[hv]
                prefetched_bias = dt_bias[hv]
        if cutlass.const_expr(self.token_bound == 4):
            # Backward propagation fills four columns; MMA reads eight.
            for i in cutlass.range_constexpr(2):
                weight_offset = tid + i * 128
                sQK[weight_offset // 16, weight_offset % 16] = Float32(0)
        idx = Int64(indices[batch])
        self.load_state(
            pool,
            ps,
            idx,
            hv,
            tile,
            tid,
            warp,
            sH,
            state_barrier,
            state_atom,
            state_source,
            0,
            True,
        )
        if cutlass.const_expr(self.use_tma_qkv):
            cute.arch.sync_threads()
            cute.arch.mbarrier_wait(qkv_barrier.iterator, 0)
        for row_group in cutlass.range_constexpr(2 if self.token_bound <= 8 else 4):
            row = warp + row_group * 4
            rq = cute.make_rmem_tensor(4, Float32)
            rk = cute.make_rmem_tensor(4, Float32)
            rq.fill(0.0)
            rk.fill(0.0)
            if cutlass.const_expr(row_group * 4 < self.token_bound):
                if row < tokens:
                    if cutlass.const_expr(self.use_tma_qkv):
                        for part in cutlass.range_constexpr(4):
                            rq[part] = Float32(raw_q[row, lane * 4 + part])
                            rk[part] = Float32(raw_k[row, lane * 4 + part])
                    else:
                        q_offset = Int64(batch) * qs[0] + row * qs[1] + head * qs[2]
                        k_offset = Int64(batch) * ks[0] + row * ks[1] + head * ks[2]
                        q_address = q.iterator.toint() + q_offset * 2
                        k_address = k.iterator.toint() + k_offset * 2
                        if (
                            (qs[3] == 1)
                            & (ks[3] == 1)
                            & ((q_address & 7) == 0)
                            & ((k_address & 7) == 0)
                        ):
                            rq.store(
                                _load_four(
                                    q_address + lane * 8,
                                    q.element_type == cutlass.BFloat16,
                                )
                            )
                            rk.store(
                                _load_four(
                                    k_address + lane * 8,
                                    k.element_type == cutlass.BFloat16,
                                )
                            )
                        else:
                            for part in cutlass.range_constexpr(4):
                                qk_dimension = lane * 4 + part
                                rq[part] = Float32(q[q_offset + qk_dimension * qs[3]])
                                rk[part] = Float32(k[k_offset + qk_dimension * ks[3]])
            sum_q, sum_k = Float32(0), Float32(0)
            for part in cutlass.range_constexpr(4):
                qk_dimension = lane * 4 + part
                sum_q += rq[part] * rq[part]
                sum_k += rk[part] * rk[part]
                if cutlass.const_expr(self.replay):
                    if (
                        (row < tokens)
                        & (tile == 0)
                        & (hv % (self.vheads // self.heads) == 0)
                    ):
                        replay[
                            (Int64(batch) * tokens + row) * width
                            + head * 128
                            + qk_dimension
                        ] = replay.element_type(rk[part])
            if cutlass.const_expr(self.normalize and row_group * 4 < self.token_bound):
                for shift in cutlass.range_constexpr(5):
                    sum_q += cute.arch.shuffle_sync_bfly(sum_q, 1 << shift)
                    sum_k += cute.arch.shuffle_sync_bfly(sum_k, 1 << shift)
                sum_q = cute.math.rsqrt(sum_q + 1e-6, fastmath=True)
                sum_k = cute.math.rsqrt(sum_k + 1e-6, fastmath=True)
            else:
                sum_q, sum_k = Float32(1), Float32(1)
            for part in cutlass.range_constexpr(4):
                sQ[row, lane * 4 + part] = rq[part] * sum_q * scale
                sK[row, lane * 4 + part] = rk[part] * sum_k
        if tid < 16:
            gate, beta, bits = Float32(0), Float32(0), Int64(0)
            if tid < tokens:
                if cutlass.const_expr(self.token_bound == 4 and self.value_tile == 64):
                    av, bv = Float32(prefetched_a), Float32(prefetched_b)
                    gate_log, gate_bias = Float32(prefetched_log), Float32(
                        prefetched_bias
                    )
                else:
                    av = Float32(a[Int64(batch) * as_[0] + tid * as_[1] + hv * as_[2]])
                    bv = Float32(b[Int64(batch) * bs[0] + tid * bs[1] + hv * bs[2]])
                    gate_log, gate_bias = Float32(A_log[hv]), Float32(dt_bias[hv])
                x = av + gate_bias
                sp = x
                if x <= 20.0:
                    sp = cute.math.log(
                        1.0 + cute.math.exp(x, fastmath=True), fastmath=True
                    )
                gate = -cute.math.exp(gate_log, fastmath=True) * sp
                gate = cute.arch.fmax(gate, Float32(-1e30))
                beta = 1.0 / (1.0 + cute.math.exp(-bv, fastmath=True))
                if cutlass.const_expr(self.tree):
                    bits = tree[Int64(batch) * tokens + tid]
                else:
                    bits = (Int64(1) << (tid + 1)) - 1
                if cutlass.const_expr(self.replay):
                    if tile == 0:
                        r = (
                            (Int64(batch) * tokens + tid) * width
                            + self.heads * 128
                            + self.vheads * 128
                            + hv
                        )
                        replay[r], replay[r + self.vheads] = replay.element_type(
                            av
                        ), replay.element_type(bv)
            sG[tid], sBeta[tid], sBits[tid] = gate, beta, bits
        for vector_step in cutlass.range_constexpr(
            (8 if self.token_bound <= 8 else 16) * self.value_tile // (128 * 4)
        ):
            vector_offset = (tid + vector_step * 128) * 4
            vector_row, vector_col = (
                vector_offset // self.value_tile,
                vector_offset % self.value_tile,
            )
            vector_values = cute.make_rmem_tensor(4, Float32)
            vector_values.fill(0.0)
            if vector_row < tokens:
                if cutlass.const_expr(self.use_tma_qkv):
                    for vector_part in cutlass.range_constexpr(4):
                        vector_values[vector_part] = Float32(
                            raw_v[vector_row, vector_col + vector_part]
                        )
                else:
                    vector_base = (
                        Int64(batch) * vs[0]
                        + vector_row * vs[1]
                        + hv * vs[2]
                        + (col_base + vector_col) * vs[3]
                    )
                    vector_address = v.iterator.toint() + vector_base * 2
                    if (vs[3] == 1) & ((vector_address & 7) == 0):
                        vector_values.store(
                            _load_four(
                                vector_address, v.element_type == cutlass.BFloat16
                            )
                        )
                    else:
                        for vector_part in cutlass.range_constexpr(4):
                            vector_values[vector_part] = Float32(
                                v[vector_base + vector_part * vs[3]]
                            )
            for vector_part in cutlass.range_constexpr(4):
                sV[vector_row, vector_col + vector_part] = vector_values[vector_part]
                if cutlass.const_expr(self.replay):
                    if vector_row < tokens:
                        replay[
                            (Int64(batch) * tokens + vector_row) * width
                            + self.heads * 128
                            + hv * 128
                            + col_base
                            + vector_col
                            + vector_part
                        ] = replay.element_type(vector_values[vector_part])
        cute.arch.sync_threads()
        if cutlass.const_expr(self.token_bound == 4):
            # Back-propagate each output query through its ancestor updates.
            # This yields the same initial-state coefficient Phi and WY weights W.
            backward_row = warp
            backward_bits = sBits[backward_row]
            backward_phi = cute.make_rmem_tensor(4, Float32)
            for backward_part in cutlass.range_constexpr(4):
                backward_phi[backward_part] = sQ[
                    backward_row, lane + backward_part * 32
                ]
            for backward_step in cutlass.range_constexpr(4):
                backward_j = 3 - backward_step
                backward_weight = Float32(0)
                if ((backward_bits >> backward_j) & 1) != 0:
                    backward_dot = Float32(0)
                    for backward_part in cutlass.range_constexpr(4):
                        backward_dot += (
                            backward_phi[backward_part]
                            * sK[backward_j, lane + backward_part * 32]
                        )
                    for backward_shift in cutlass.range_constexpr(5):
                        backward_dot += cute.arch.shuffle_sync_bfly(
                            backward_dot, 1 << backward_shift
                        )
                    backward_weight = backward_dot * sBeta[backward_j]
                    backward_decay = cute.math.exp(sG[backward_j], fastmath=True)
                    for backward_part in cutlass.range_constexpr(4):
                        backward_phi[backward_part] = backward_decay * (
                            backward_phi[backward_part]
                            - backward_weight
                            * sK[backward_j, lane + backward_part * 32]
                        )
                if lane == 0:
                    sQK[backward_row, backward_j] = backward_weight
            for backward_part in cutlass.range_constexpr(4):
                sQ[backward_row, lane + backward_part * 32] = backward_phi[
                    backward_part
                ]
        else:
            sKK = smem.allocate_tensor(
                Float32,
                cute.make_layout(
                    (self.gram_rows, self.gram_rows), stride=(self.gram_rows, 1)
                ),
            )
            sD = smem.allocate_tensor(
                Float32,
                cute.make_layout(
                    (self.gram_rows, self.gram_rows), stride=(self.gram_rows, 1)
                ),
            )
            sDecay = smem.allocate_tensor(Float32, cute.make_layout(self.gram_rows))
            if cutlass.const_expr(self.gram_rows == 8):
                # One warp computes [K; Q] @ K.T while two warps form decays.
                if warp == 2:
                    gram_result = _gram_eight(sQ, sK, lane)
                    row, column = lane // 4, (lane % 4) * 2
                    sKK[row, column], sKK[row, column + 1] = (
                        gram_result[0],
                        gram_result[1],
                    )
                    sQK[row, column], sQK[row, column + 1] = (
                        gram_result[2],
                        gram_result[3],
                    )
            # Summing path segments avoids cancellation of large cumulative gates.
            for i in cutlass.range_constexpr(2 if self.gram_rows == 16 else 1):
                offset = tid + i * 128
                row, col = offset // self.gram_rows, offset % self.gram_rows
                active = True
                if cutlass.const_expr(self.gram_rows == 8):
                    active = tid < 64
                if active:
                    row_bits, col_bits = sBits[row], sBits[col]
                    segment = row_bits & ~col_bits
                    path_gate, segment_gate = Float32(0), Float32(0)
                    for step in cutlass.range_constexpr(self.token_bound):
                        g = sG[step]
                        if (row_bits & (Int64(1) << step)) != 0:
                            path_gate += g
                        if (segment & (Int64(1) << step)) != 0:
                            segment_gate += g
                    decay = Float32(0)
                    if ((row_bits >> col) & 1) != 0:
                        decay = cute.math.exp(segment_gate, fastmath=True)
                    sD[row, col] = decay
                    if col == 0:
                        sDecay[row] = cute.math.exp(path_gate, fastmath=True)
            if cutlass.const_expr(self.gram_rows == 16):
                if warp < 2:
                    c = _dot_tf32(
                        sK, sK, lane, warp * 8, 128, False, self.accumulators, None
                    )
                    _store_fragment(c, sKK, lane, warp * 8)
                else:
                    c = _dot_tf32(
                        sQ,
                        sK,
                        lane,
                        (warp - 2) * 8,
                        128,
                        False,
                        self.accumulators,
                        None,
                    )
                    _store_fragment(c, sQK, lane, (warp - 2) * 8)
            cute.arch.sync_threads()
            for i in cutlass.range_constexpr(2 if self.gram_rows == 16 else 1):
                offset = tid + i * 128
                row, col = offset // self.gram_rows, offset % self.gram_rows
                active = True
                if cutlass.const_expr(self.gram_rows == 8):
                    active = tid < 64
                if active:
                    sQK[row, col] = sQK[row, col] * sD[row, col]
                    sKK[row, col] = sKK[row, col] * sD[row, col] * sBeta[row]
            cute.arch.sync_threads()
            # W = (D * QK) @ (I + beta * D_strict * KK)^-1 @ diag(beta).
            # Each lane solves one output row from right to left in registers.
            if warp == 0:
                values = cute.make_rmem_tensor(self.token_bound, Float32)
                row = lane % self.gram_rows
                for j in cutlass.range_constexpr(self.token_bound):
                    values[j] = sQK[row, j]
                for step in cutlass.range_constexpr(self.token_bound):
                    i = self.token_bound - 1 - step
                    yi = values[i]
                    for j in cutlass.range_constexpr(i):
                        values[j] = values[j] - yi * sKK[i, j]
                if lane < self.gram_rows:
                    for j in cutlass.range_constexpr(self.token_bound):
                        sQK[row, j] = values[j] * sBeta[j]
            for i in cutlass.range_constexpr(workspace_rows):
                offset = tid + i * 128
                row, col = offset // 128, offset % 128
                sK[row, col] = sK[row, col] * sDecay[row]
            cute.arch.sync_threads()
            # Form the initial-state coefficient in place over the query tile.
            for tile_k in cutlass.range_constexpr(4):
                column = (warp + tile_k * 4) * 8
                c = _dot_tf32(
                    sQK,
                    sK,
                    lane,
                    column,
                    self.token_bound,
                    True,
                    self.accumulators,
                    None,
                )
                row, col = lane // 4, column + (lane % 4) * 2
                for j in cutlass.range_constexpr(4):
                    r = row + (j // 2) * 8
                    vc = col + (j % 2)
                    product = c[j]
                    if r < workspace_rows:
                        sQ[r, vc] = sQ[r, vc] * sDecay[r] - product
        if cutlass.const_expr(self.use_tma_state):
            cute.arch.mbarrier_wait(state_barrier.iterator, 0)
        else:
            cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        if cutlass.const_expr(self.token_bound <= 8):
            fragments = self.value_tile // 64
            buffered = fragments if self.key_chunk < 128 else 1
            for group in cutlass.range_constexpr(fragments // buffered):
                state_values = cute.make_rmem_tensor((4, buffered), Float32)
                value_values = cute.make_rmem_tensor((4, buffered), Float32)
                for key_stage in cutlass.range_constexpr(128 // self.key_chunk):
                    if cutlass.const_expr(key_stage != 0):
                        if cutlass.const_expr(self.use_tma_state):
                            cute.arch.mbarrier_wait(
                                state_barrier.iterator, key_stage % 2
                            )
                        else:
                            cute.arch.cp_async_wait_group(0)
                        cute.arch.sync_threads()
                    for fragment in cutlass.range_constexpr(buffered):
                        value_base = (warp + (group * buffered + fragment) * 4) * 16
                        h_rows = cute.make_tensor(
                            sH.iterator + value_base * h_stride,
                            cute.make_layout(
                                (16, self.key_chunk), stride=(h_stride, 1)
                            ),
                        )
                        q_rows = cute.local_tile(
                            sQ, (8, self.key_chunk), (0, key_stage)
                        )
                        initial = None
                        if cutlass.const_expr(key_stage != 0):
                            initial = state_values[None, fragment].load()
                        result = _dot_tf32(
                            h_rows,
                            q_rows,
                            lane,
                            0,
                            self.key_chunk,
                            False,
                            self.accumulators,
                            initial,
                        )
                        state_values[None, fragment].store(result)
                    if cutlass.const_expr(key_stage + 1 < 128 // self.key_chunk):
                        # Every reader releases the single state buffer before reuse.
                        cute.arch.sync_threads()
                        self.load_state(
                            pool,
                            ps,
                            idx,
                            hv,
                            tile,
                            tid,
                            warp,
                            sH,
                            state_barrier,
                            state_atom,
                            state_source,
                            (key_stage + 1) * self.key_chunk,
                            False,
                        )
                    if cutlass.const_expr(key_stage == 0):
                        # This product overlaps the next state transfer when streaming.
                        for fragment in cutlass.range_constexpr(buffered):
                            value_base = (warp + (group * buffered + fragment) * 4) * 16
                            v_rows = cute.make_tensor(
                                sV.iterator + value_base,
                                cute.make_layout(
                                    (16, 16), stride=(1, self.value_tile + 4)
                                ),
                            )
                            result = _dot_tf32(
                                v_rows,
                                sQK,
                                lane,
                                0,
                                self.reduction_bound,
                                False,
                                self.accumulators,
                                None,
                            )
                            value_values[None, fragment].store(result)
                for fragment in cutlass.range_constexpr(buffered):
                    value_base = (warp + (group * buffered + fragment) * 4) * 16
                    output_values = cute.make_rmem_tensor(4, Float32)
                    output_values.store(
                        state_values[None, fragment].load()
                        + value_values[None, fragment].load()
                    )
                    row, col = lane // 4, (lane % 4) * 2
                    for j in cutlass.range_constexpr(4):
                        token = col + j % 2
                        value = value_base + row + (j // 2) * 8
                        if token < tokens:
                            out[
                                ((Int64(batch) * tokens + token) * self.vheads + hv)
                                * 128
                                + col_base
                                + value
                            ] = out.element_type(output_values[j])
        else:
            for tile_v in cutlass.range_constexpr(self.value_tile // 32):
                column = tile_v * 32 + warp * 8
                c = _dot_tf32(sQ, sH, lane, column, 128, False, self.accumulators, None)
                d = _dot_tf32(
                    sQK,
                    sV,
                    lane,
                    column,
                    self.reduction_bound,
                    True,
                    self.accumulators,
                    None,
                )
                output_values = cute.make_rmem_tensor(4, Float32)
                output_values.store(c + d)
                row, col = lane // 4, column + (lane % 4) * 2
                for j in cutlass.range_constexpr(4):
                    r = row + (j // 2) * 8
                    vc = col + (j % 2)
                    if r < tokens:
                        out[
                            ((Int64(batch) * tokens + r) * self.vheads + hv) * 128
                            + col_base
                            + vc
                        ] = out.element_type(output_values[j])


_COMPILED = {}
_DUAL_ACCUMULATORS_MAX_HEADS = 64
_WIDE_VALUES_MIN_HEADS = 256
_STREAM_STATE_MAX_HEADS = 1024


def gdn_wy_launch_config(
    batch: int, vheads: int, token_bound: int, state_dtype: torch.dtype
) -> tuple[int, int, int, int]:
    """Return bounded launch choices for the serving geometry.

    Args:
        batch: Request count.
        vheads: Value heads on this rank.
        token_bound: The 4/8/16-token bucket.
        state_dtype: Recurrent-state pool dtype.

    Returns:
        Value tile, accumulator count, Gram rows and checkpoint key chunk.
    """
    value_tile = 128 if batch * vheads >= _WIDE_VALUES_MIN_HEADS else 64
    if token_bound == 16 and state_dtype == torch.float32:
        value_tile = 64
    accumulators = (
        2 if token_bound == 8 and batch * vheads <= _DUAL_ACCUMULATORS_MAX_HEADS else 1
    )
    gram_rows = 8 if token_bound == 8 else 16
    key_chunk = (
        64
        if token_bound == 4
        and _WIDE_VALUES_MIN_HEADS <= batch * vheads <= _STREAM_STATE_MAX_HEADS
        and state_dtype == torch.bfloat16
        else 128
    )
    return value_tile, accumulators, gram_rows, key_chunk


def _warmup_launch_configs(
    vheads: int, token_bound: int, state_dtype: torch.dtype
) -> tuple[tuple[int, int, int, int], ...]:
    batches = {
        1,
        _DUAL_ACCUMULATORS_MAX_HEADS // vheads + 1,
        (_WIDE_VALUES_MIN_HEADS + vheads - 1) // vheads,
        _STREAM_STATE_MAX_HEADS // vheads + 1,
    }
    return tuple(
        dict.fromkeys(
            gdn_wy_launch_config(batch, vheads, token_bound, state_dtype)
            for batch in sorted(batches)
        )
    )


def launch_gdn_wy(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float | None,
    use_qk_l2norm: bool,
    tree_ancestors: torch.Tensor | None,
    replay_payload: torch.Tensor | None,
) -> torch.Tensor:
    """Launch GDN WY verification with cached CuTe specializations.

    Tensor arguments follow ``cute_dsl_gdn_decode_mtp``. PyTorch dtypes map to
    CuTe pointer types; absent tree masks and replay destinations use existing
    tensors as placeholders. Return ``[B, T, HV, V]`` output with ``q.dtype``,
    including an empty output for an empty batch, without writing the state pool.

    ``_COMPILED`` caches one kernel per device, operand dtypes, model geometry,
    execution flags and bounded launch/transport choices. Token bounds use
    4/8/16 buckets; value tiles and accumulators follow batch/head thresholds.
    Actual batch/token counts, strides and scale remain runtime inputs. A cache
    miss compiles all Q/K/V and state transport combinations; ``kernel_warmup``
    also prepares reachable launch tiers for the selected token bucket.

    Warm every selected bucket/dtype/geometry before CUDA Graph capture;
    a missing specialization during capture raises ``RuntimeError``.
    """
    B, T, H, K = q.shape
    HV, V = v.shape[2:]
    out = q.new_empty((B, T, HV, V))
    if B == 0:
        return out
    tensors = (
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        initial_state,
        initial_state_indices,
        tree_ancestors if tree_ancestors is not None else initial_state_indices,
        replay_payload if replay_payload is not None else out,
        out,
    )
    dtypes = {
        torch.bfloat16: cutlass.BFloat16,
        torch.float16: cutlass.Float16,
        torch.float32: Float32,
        torch.int32: Int32,
        torch.int64: Int64,
    }
    ptrs = tuple(
        make_ptr(
            dtypes[x.dtype],
            x.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=x.element_size(),
        )
        for x in tensors
    )
    strides = tuple(tuple(Int64(s) for s in x.stride()) for x in (q, k, v, a, b)) + (
        Int64(initial_state.stride(0)),
    )
    pdl = pdl_enabled()
    use_tma_qkv = all(
        x.data_ptr() % 16 == 0
        and x.stride(3) == 1
        and all(s * x.element_size() % 16 == 0 for s in x.stride()[:3])
        for x in (q, k, v)
    )
    qkv_tensors = (
        tuple(
            from_dlpack(x, assumed_align=16).mark_layout_dynamic(leading_dim=3)
            for x in (q, k, v)
        )
        if use_tma_qkv
        else None
    )
    use_tma_state = (
        initial_state.data_ptr() % 16 == 0
        and initial_state.stride(0) * initial_state.element_size() % 16 == 0
    )
    state_tensor = (
        from_dlpack(initial_state, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        if use_tma_state
        else None
    )
    token_bound = 4 if T <= 4 else 8 if T <= 8 else 16
    value_tile, accumulators, gram_rows, key_chunk = gdn_wy_launch_config(
        B, HV, token_bound, initial_state.dtype
    )
    configs = ((value_tile, accumulators, gram_rows, key_chunk),)
    if is_warming_up():
        configs = _warmup_launch_configs(HV, token_bound, initial_state.dtype)
    key = (
        q.device.index,
        tuple(x.dtype for x in tensors),
        H,
        HV,
        use_qk_l2norm,
        tree_ancestors is not None,
        replay_payload is not None,
        pdl,
        use_tma_qkv,
        use_tma_state,
        accumulators,
        value_tile,
        token_bound,
        gram_rows,
        key_chunk,
    )
    with torch.cuda.device(q.device):
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        args = (
            ptrs,
            strides,
            Int32(B),
            Int32(T),
            Float32(K**-0.5 if scale is None else scale),
            qkv_tensors,
            state_tensor,
            stream,
        )
        for value_tile, accumulators, gram_rows, key_chunk in configs:
            config_key = (
                *key[:10],
                accumulators,
                value_tile,
                token_bound,
                gram_rows,
                key_chunk,
            )
            compiled = _COMPILED.get(config_key)
            if compiled is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "GDN WY must be warmed before CUDA Graph capture"
                    )
                # Graph replay may refresh from a contiguous input to a strided view.
                # Warm both transport variants together before any graph capture.
                for qkv_variant in (False, True):
                    for state_variant in (False, True):
                        variant_key = (
                            *config_key[:8],
                            qkv_variant,
                            state_variant,
                            *config_key[10:],
                        )
                        if variant_key in _COMPILED:
                            continue
                        variant_qkv, variant_state = None, None
                        if qkv_variant:
                            if use_tma_qkv:
                                variant_qkv = qkv_tensors
                            else:
                                warm_qkv = tuple(
                                    torch.empty(
                                        (1, token_bound, x.shape[2], 128),
                                        device=x.device,
                                        dtype=x.dtype,
                                    )
                                    for x in (q, k, v)
                                )
                                variant_qkv = tuple(
                                    from_dlpack(
                                        x, assumed_align=16
                                    ).mark_layout_dynamic(leading_dim=3)
                                    for x in warm_qkv
                                )
                        if state_variant:
                            if use_tma_state:
                                variant_state = state_tensor
                            else:
                                warm_state = torch.empty(
                                    (1, HV, 128, 128),
                                    device=initial_state.device,
                                    dtype=initial_state.dtype,
                                )
                                variant_state = from_dlpack(
                                    warm_state, assumed_align=16
                                ).mark_layout_dynamic(leading_dim=3)
                        variant_args = (*args[:5], variant_qkv, variant_state, stream)
                        _COMPILED[variant_key] = cute.compile(
                            GdnWYKernel(
                                H,
                                HV,
                                use_qk_l2norm,
                                tree_ancestors is not None,
                                replay_payload is not None,
                                pdl,
                                accumulators,
                                value_tile,
                                token_bound,
                                qkv_variant,
                                state_variant,
                                gram_rows,
                                key_chunk,
                            ),
                            *variant_args,
                        )
        _COMPILED[key](*args)
    return out
