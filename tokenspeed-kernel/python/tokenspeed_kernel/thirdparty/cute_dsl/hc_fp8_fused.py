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

"""Single-launch native FP8 mixer for prequantized activation and weights."""

import cutlass
import cutlass.cute as cute
import cutlass.utils as cute_utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cuda.bindings.driver import CUstream
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import CopyUniversalOp, tcgen05
from cutlass.cutlass_dsl import T
from cutlass.experimental.primitives import tcgen05_fence
from tokenspeed_kernel.thirdparty.cute_dsl.hc_fp8 import (
    Fp8GatedResidualGemm,
    _epilogue_barrier,
    _up_sigmoid_tanh,
)
from tokenspeed_kernel.thirdparty.cute_dsl.hc_fused import (
    _arrive_counter,
    _publish_activation_proxy,
)


@cute.experimental.jit
def _copy_residual_16(destination, source, size):
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(destination.toint()).ir_value(),
            source.toint().ir_value(),
            size.ir_value(),
        ],
        "cp.async.ca.shared.global [$0], [$1], 16, $2;",
        "r,l,r,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.experimental.jit
def _commit_residual():
    llvm.inline_asm(
        None,
        [],
        "cp.async.commit_group;",
        "~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.experimental.jit
def _wait_residual():
    llvm.inline_asm(
        None,
        [],
        "cp.async.wait_group 0;",
        "~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.experimental.jit
def _acquire(ptr):
    return cutlass.Int64(
        llvm.inline_asm(
            T.i64(),
            [ptr.toint().ir_value()],
            "ld.global.acquire.gpu.u64 $0, [$1];",
            "=l,l,~{memory}",
            has_side_effects=True,
            asm_dialect=0,
        )
    )


@cute.experimental.jit
def _release(ptr, value):
    llvm.inline_asm(
        None,
        [ptr.toint().ir_value(), value.ir_value()],
        "st.global.release.gpu.u64 [$0], $1;",
        "l,l,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.experimental.jit
def _grid_barrier(epochs, expected, wait_for_all):
    # Cooperative launch and measured capacity keep every participant resident.
    cute.arch.sync_threads()
    if cute.arch.thread_idx()[0] == 0:
        _arrive_counter(epochs.iterator)
        if wait_for_all:
            while _acquire(epochs.iterator) < expected:
                pass
    cute.arch.sync_threads()


@cute.experimental.jit
def _invalidate(barriers, count: cutlass.Constexpr):
    with cute.arch.elect_one():
        for index in cutlass.range_constexpr(count):
            address = cutlass.Int32((barriers + index).toint())
            llvm.inline_asm(
                None,
                [address.ir_value()],
                "mbarrier.inval.shared::cta.b64 [$0];",
                "r,~{memory}",
                has_side_effects=True,
                asm_dialect=0,
            )


@cute.experimental.jit
def _operand_storage(arena, first_layout, second_layout):
    dtype = cutlass.Float8E4M3FN
    first = cute.make_tensor(
        cute.make_ptr(
            dtype,
            arena.iterator.toint(),
            cute.AddressSpace.smem,
            assumed_align=1024,
            swizzle_=cute.get_swizzle_portion(first_layout),
        ),
        cute.get_nonswizzle_portion(first_layout),
    )
    second = cute.make_tensor(
        cute.make_ptr(
            dtype,
            arena.iterator.toint() + cute.cosize(first_layout),
            cute.AddressSpace.smem,
            assumed_align=1024,
            swizzle_=cute.get_swizzle_portion(second_layout),
        ),
        cute.get_nonswizzle_portion(second_layout),
    )
    return first, second


@cute.experimental.jit
def _operand_at(arena, layout, offset):
    return cute.make_tensor(
        cute.make_ptr(
            cutlass.Float8E4M3FN,
            arena.iterator.toint() + offset,
            cute.AddressSpace.smem,
            assumed_align=1024,
            swizzle_=cute.get_swizzle_portion(layout),
        ),
        cute.get_nonswizzle_portion(layout),
    )


@cute.experimental.jit
def _load_f32(ptr):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [ptr.toint().ir_value()],
            "ld.global.cg.f32 $0, [$1];",
            "=f,l,~{memory}",
            has_side_effects=True,
            asm_dialect=0,
        )
    )


@cute.experimental.jit
def _x_scale(scales, row):
    index = cutlass.Int64(0)
    if scales.shape[0] != 1:
        index = cutlass.Int64(row)
    return _load_f32(scales.iterator + index)


class _DownPhase(Fp8GatedResidualGemm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.producer_warp = 16
        self.mma_warp = 17

    @cute.experimental.jit
    def project(
        self,
        x,
        weight,
        partials,
        arena,
        barriers,
        tmem_base,
        output_tile,
        split,
        token_tile,
    ):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        dtype = cutlass.Float8E4M3FN
        mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (self.m, self.n),
        )
        tiler = (self.m, self.n, self.k)
        weight_layout = sm100_utils.make_smem_layout_a(mma, tiler, dtype, self.stages)
        activation_layout = sm100_utils.make_smem_layout_b(
            mma, tiler, dtype, self.stages
        )
        sw, sx = _operand_storage(arena, weight_layout, activation_layout)
        full, empty = barriers, barriers + self.stages
        done, read_done = empty + self.stages, empty + self.stages + 1
        if warp == self.mma_warp:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(full + stage, 1)
                    cute.arch.mbarrier_init(empty + stage, 1)
                cute.arch.mbarrier_init(done, 1)
                cute.arch.mbarrier_init(read_done, self.epilogue_threads)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        tcgen05_fence("after_thread_sync")
        w_view = cute.make_tensor(
            weight.iterator,
            cute.make_layout(
                (self.projection_rows, (64, self.inner // 64), 1),
                stride=(self.inner, (1, 64), 0),
            ),
        )
        x_view = cute.make_tensor(
            x.iterator,
            cute.make_layout(
                (x.shape[0], (64, self.inner // 64), 1),
                stride=(self.inner, (1, 64), 0),
            ),
        )
        gw = cute.local_tile(w_view, (self.m, self.k), (output_tile, None, 0))
        gx = cute.local_tile(x_view, (self.n, self.k), (token_tile, None, 0))
        map_w = cute_ext.get_cta_v_map_ab(w_view, tiler, mma, "A")
        map_x = cute_ext.get_cta_v_map_ab(x_view, tiler, mma, "B")
        stage_bytes = (self.m + self.n) * self.k
        if warp == self.producer_warp:
            for stage in cutlass.range_constexpr(self.stages):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(full + stage, stage_bytes)
                if cutlass.const_expr(self.weights_independent):
                    cute_ext.tma_load(
                        gw[None, None, split * self.tiles + stage],
                        sw[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_w,
                        update_expect_tx=False,
                    )
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
            for stage in cutlass.range_constexpr(self.stages):
                if cutlass.const_expr(not self.weights_independent):
                    cute_ext.tma_load(
                        gw[None, None, split * self.tiles + stage],
                        sw[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_w,
                        update_expect_tx=False,
                    )
                cute_ext.tma_load(
                    gx[None, None, split * self.tiles + stage],
                    sx[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
            for tile in cutlass.range(self.stages, self.tiles, unroll=1):
                stage = tile % self.stages
                cute.arch.mbarrier_wait(empty + stage, (tile // self.stages - 1) % 2)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(full + stage, stage_bytes)
                cute_ext.tma_load(
                    gw[None, None, split * self.tiles + tile],
                    sw[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_w,
                    update_expect_tx=False,
                )
                cute_ext.tma_load(
                    gx[None, None, split * self.tiles + tile],
                    sx[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
        elif warp == self.mma_warp:
            thread_mma = mma.get_slice(0)
            a_desc = thread_mma.make_fragment_A(sw)
            b_desc = thread_mma.make_fragment_B(sx)
            fragment = thread_mma.make_fragment_C(
                mma.partition_shape_C((self.m, self.n))
            )
            acc = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, tmem_base),
                fragment.layout,
            )
            mma.set(tcgen05.Field.ACCUMULATE, False)
            for tile in cutlass.range(self.tiles, unroll=1):
                stage = tile % self.stages
                cute.arch.mbarrier_wait(full + stage, tile // self.stages % 2)
                for block in cutlass.range_constexpr(cute.size(a_desc, mode=[2])):
                    cute.gemm(
                        mma,
                        acc,
                        a_desc[None, None, block, stage],
                        b_desc[None, None, block, stage],
                        acc,
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(empty + stage, None, tcgen05.CtaGroup.ONE)
            with cute.arch.elect_one():
                tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
            cute.arch.mbarrier_wait(read_done, 0)
        elif tid < self.epilogue_threads:
            cute.arch.mbarrier_wait(done, 0)
            fragment = mma.get_slice(0).make_fragment_C(
                mma.partition_shape_C((self.m, self.n))
            )
            acc = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, tmem_base),
                fragment.layout,
            )[((None, None), 0, 0)]
            reader_n = self.n // self.epilogue_groups
            tile_shape = (self.m, reader_n)
            group, lane = tid // 128, tid % 128
            acc = cute.local_tile(acc, tile_shape, (0, group))
            coordinates = cute.local_tile(
                cute.make_identity_tensor((self.m, self.n)),
                tile_shape,
                (0, group),
            )
            copy = tcgen05.make_tmem_copy(
                sm100_utils.get_tmem_load_op(
                    (self.m, reader_n, self.k),
                    cute_utils.LayoutEnum.COL_MAJOR,
                    cutlass.Float32,
                    cutlass.Float32,
                    tile_shape,
                    False,
                ),
                acc,
            )
            reader = copy.get_slice(lane)
            dummy = cute.make_tensor(
                partials.iterator, cute.make_layout(tile_shape, stride=(1, self.m))
            )
            values = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.rmem,
                cute_ext.make_t2r_rmem_layout(
                    copy, cute.flat_divide(dummy, tile_shape), lane
                ),
                alignment=32,
            )
            coords = reader.partition_D(coordinates)
            cute_ext.partition_and_copy(reader, acc, values)
            cute.arch.fence_view_async_tmem_load()
            for index in cutlass.range_constexpr(cute.size(values)):
                col, row = coords[index]
                local_row = cutlass.Int64(row)
                row = local_row + token_tile * self.n
                global_col = output_tile * self.m + col
                if row < x.shape[0] and global_col < self.projection_rows:
                    partials[split, local_row, global_col] = values[index]
            tcgen05_fence("before_thread_sync")
            cute.arch.mbarrier_arrive(read_done)


class Fp8FusedGatedResidualKernel:
    """Resident phases reuse one operand arena and one TMEM allocation."""

    def __init__(
        self,
        *,
        g,
        h,
        r,
        p,
        token_bucket,
        projection_scale,
        use_pdl,
        weights_independent,
        down_per_channel,
        up_per_channel
    ):
        self.g = g
        self.h = h
        self.r = r
        self.p = p
        self.n = token_bucket
        self.scale = projection_scale
        self.pdl = use_pdl
        self.weights_independent = weights_independent
        self.down_per_channel = down_per_channel
        self.up_per_channel = up_per_channel
        self.threads = 576
        self.up_hidden = 64 if r == 1024 else 32
        self.up_channels = self.up_hidden * g
        self.up_parts = 2 if token_bucket == 256 else 1
        # Small narrow tiles avoid padding tokens to the native M128 reader.
        self.token_major_up = r == 1024 or token_bucket == 256
        self.up_rows = 128 if self.token_major_up else token_bucket
        self.up_mn = (
            (self.up_rows, self.up_channels)
            if self.token_major_up
            else (self.up_channels, self.up_rows)
        )
        self.hidden_reader_columns = min(32, token_bucket // 4)
        self.hidden_reader_passes = 1
        self.reader_columns = self.up_hidden // 4
        self.up_k = 128 if r == 1024 else 64
        self.up_stages = 3
        self.up_tiles = r // self.up_k
        self.up_tail_tiles = self.up_stages if self.up_parts == 2 else 0
        self.hidden_tiles = h // self.up_hidden
        self.up_operand_bytes = (
            (self.up_channels + self.up_parts * self.up_rows)
            * self.up_k
            * self.up_stages
        )
        self.up_barrier_count = 2 * self.up_stages + self.up_parts + 1
        self.splits = 8 if r == 1024 else 16
        down_k = (512 if token_bucket == 16 else 256) if r == 1024 else 128
        if token_bucket >= 128:
            down_k = 256 if r == 1024 else 128
        down_stages = 2 if r == 1024 and token_bucket == 256 else 3
        down_epilogue_threads = 256 if token_bucket == 16 else 512
        self.down = _DownPhase(
            role="projection",
            hc_count=g,
            hidden_size=h,
            lowrank=r,
            projection_rows=p,
            token_tile=token_bucket,
            projection_tile=64,
            k_tile=down_k,
            stages=down_stages,
            split_k=self.splits,
            use_pdl=False,
            weights_independent=weights_independent,
            weight_scale_per_channel=down_per_channel,
            token_major_up=False,
            weight_scale_vectorized=False,
            up_hidden_tile=32,
            up_token_rows=self.up_rows,
            separate_operand_ready=False,
            epilogue_threads=down_epilogue_threads,
            shared_up_scales=False,
            interleave_tma=False,
        )
        self.down_tiles = (p + 63) // 64
        self.down_tasks = self.down_tiles * self.splits
        self.up_tasks = self.hidden_tiles
        self.columns = max(128, token_bucket, self.up_channels * self.up_parts)
        self.arena_bytes = max(
            (64 + token_bucket) * down_k * down_stages,
            self.up_operand_bytes,
            0 if self.token_major_up else token_bucket * self.up_channels * 4,
        )
        self.barrier_count = max(2 * down_stages + 2, self.up_barrier_count)
        # A conservative occupancy query includes alignment and epilogue scratch.
        self.shared_bytes = (
            self.arena_bytes
            + (0 if self.token_major_up else token_bucket * self.up_channels)
            + (2 * token_bucket + self.up_channels) * 4
            + self.barrier_count * 8
            + 2048
        )

    @cute.experimental.jit
    def __call__(
        self,
        x,
        xs,
        down,
        ds,
        up,
        us,
        partials,
        activation,
        alpha,
        epochs,
        out,
        inject,
        grid_ctas: cutlass.Int32,
        stream: CUstream,
    ):
        self.kernel(
            x, xs, down, ds, up, us, partials, activation, alpha, epochs, out, inject
        ).launch(
            grid=(grid_ctas, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            cooperative=True,
            use_pdl=self.pdl,
        )

    @cute.experimental.kernel
    def kernel(
        self, x, xs, down, ds, up, us, partials, activation, alpha, epochs, out, inject
    ):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        pid = cute.arch.block_idx()[0]
        ctas = cutlass.Int64(cute.arch.grid_dim()[0])
        arena = cute_ext.allocate(
            cutlass.Uint8,
            cute.AddressSpace.smem,
            cute.make_layout(self.arena_bytes),
            alignment=1024,
        )
        bars = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(self.barrier_count),
            alignment=8,
        )
        base = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        )
        epoch = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()
        if tid == 0:
            epoch[0] = _acquire(epochs.iterator + cutlass.Int64(pid) + 1) + 1
        if warp == 17:
            cute.arch.alloc_tmem(self.columns, base.iterator, is_two_cta=False)
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.sync_threads()
        generation = epoch[0]
        token_round = cutlass.Int64(0)
        global_base = cutlass.Int64(0)
        while global_base < x.shape[0]:
            task = cutlass.Int32(pid)
            while task < self.down_tasks:
                self.down.project(
                    x,
                    down,
                    partials,
                    arena,
                    bars.iterator,
                    base.iterator,
                    task // self.splits,
                    task % self.splits,
                    token_round,
                )
                cute.arch.sync_threads()
                if warp == 17:
                    _invalidate(bars.iterator, 2 * self.down.stages + 2)
                cute.arch.sync_threads()
                task = task + cutlass.Int32(ctas)
            _grid_barrier(epochs, ((generation - 1) * 3 + 1) * ctas, True)
            row = cutlass.Int64(pid)
            while row < self.n:
                self._epilogue(
                    partials,
                    xs,
                    ds,
                    activation,
                    alpha,
                    out,
                    inject,
                    global_base,
                    row,
                    x.shape[0],
                )
                row = row + ctas
            _publish_activation_proxy()
            _grid_barrier(epochs, ((generation - 1) * 3 + 2) * ctas, True)
            task = cutlass.Int32(pid)
            while task < self.up_tasks:
                self._up(
                    activation,
                    alpha,
                    x,
                    xs,
                    up,
                    us,
                    out,
                    arena,
                    bars.iterator,
                    base.iterator,
                    task,
                    global_base,
                )
                cute.arch.sync_threads()
                if warp == 17:
                    _invalidate(bars.iterator, self.up_barrier_count)
                cute.arch.sync_threads()
                task = task + cutlass.Int32(ctas)
            if tid == 0:
                _release(epochs.iterator + cutlass.Int64(pid) + 1, generation)
            # Another round needs a wait before scratch reuse. On the final
            # round, stream/PDL ordering protects the next invocation instead.
            _grid_barrier(
                epochs, generation * 3 * ctas, global_base + self.n < x.shape[0]
            )
            generation = generation + 1
            token_round = token_round + 1
            global_base = global_base + self.n
        if warp == 17:
            cute.arch.dealloc_tmem(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base.iterator),
                self.columns,
            )
        cute.arch.sync_threads()
        if cutlass.const_expr(self.pdl):
            if tid == 0:
                cute.arch.griddepcontrol_launch_dependents()

    @cute.experimental.jit
    def _partials(self, partial, row, col):
        pieces = cute.make_rmem_tensor((2, self.splits), cutlass.Float32)
        row = cutlass.Int64(row)
        stride = cutlass.Int64(partial.shape[1]) * self.p
        base = partial.iterator + row * self.p + col
        for rank in cutlass.range_constexpr(self.splits):
            result = llvm.inline_asm(
                llvm.StructType.get_literal([T.f32(), T.f32()]),
                [(base + rank * stride).toint().ir_value()],
                "ld.global.cg.v2.f32 {$0, $1}, [$2];",
                "=f,=f,l,~{memory}",
                has_side_effects=True,
                asm_dialect=0,
            )
            pieces[0, rank] = cutlass.Float32(llvm.extractvalue(T.f32(), result, [0]))
            pieces[1, rank] = cutlass.Float32(llvm.extractvalue(T.f32(), result, [1]))
        values = cute.make_rmem_tensor(2, cutlass.Float32)
        values.fill(0.0)
        for rank in cutlass.range_constexpr(self.splits):
            for index in cutlass.range_constexpr(2):
                values[index] = values[index] + pieces[index, rank]
        return values

    @cute.experimental.jit
    def _epilogue(
        self, partial, xs, ds, activation, alpha, out, inject, global_base, row, rows
    ):
        tid = cute.arch.thread_idx()[0]
        if tid < 512:
            col = tid * 2
            global_row = global_base + row
            rounded = cute.make_rmem_tensor(2, out.element_type)
            rounded.fill(0.0)
            if global_row < rows:
                if cutlass.const_expr(self.p > self.r):
                    if tid < self.g // 2:
                        values = self._partials(partial, row, self.r + col)
                        for i in cutlass.range_constexpr(2):
                            value = values[i] * _x_scale(xs, global_row)
                            value = (
                                value
                                * ds[self.r + col + i if self.down_per_channel else 0]
                            )
                            inject[global_row, col + i] = (value * self.scale).to(
                                out.element_type
                            )
                if col < self.r:
                    values = self._partials(partial, row, col)
                    for i in cutlass.range_constexpr(2):
                        value = values[i] * _x_scale(xs, global_row)
                        value = value * ds[col + i if self.down_per_channel else 0]
                        value = value * self.scale
                        value = value * cute.arch.rcp_approx(
                            1.0 + cute.exp(-value, fastmath=True)
                        )
                        rounded[i] = value.to(out.element_type)
            shared = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout(18),
                alignment=32,
            )
            maximum = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(2):
                maximum = cute.math.max(
                    maximum, cute.math.abs(rounded[i].to(cutlass.Float32))
                )
            maximum = cute.arch.warp_reduction_max(maximum, threads_in_group=32)
            if tid % 32 == 0:
                shared[tid // 32] = maximum
            _epilogue_barrier(512, cutlass.Int32(5))
            if tid < 32:
                maximum = cutlass.Float32(0.0)
                if tid < 16:
                    maximum = shared[tid]
                maximum = cute.arch.warp_reduction_max(maximum, threads_in_group=32)
                if tid == 0:
                    alpha[row] = cute.math.max(
                        maximum / cutlass.Float32(448.0),
                        cutlass.Float32(1.0 / (448.0 * 512.0)),
                    )
                    shared[17] = cute.math.min(
                        cutlass.Float32(448.0) / maximum,
                        cutlass.Float32(448.0 * 512.0),
                    )
            _epilogue_barrier(512, cutlass.Int32(5))
            if col < self.r:
                multiplier = shared[17]
                for i in cutlass.range_constexpr(2):
                    value = cute.math.min(
                        cute.math.max(
                            rounded[i].to(cutlass.Float32) * multiplier,
                            cutlass.Float32(-448.0),
                        ),
                        cutlass.Float32(448.0),
                    )
                    activation[row, col + i] = value.to(cutlass.Float8E4M3FN)
        cute.arch.sync_threads()

    @cute.experimental.jit
    def _reader(
        self,
        mma,
        base,
        original,
        tid,
        branch: cutlass.Constexpr,
        part: cutlass.Constexpr,
    ):
        layout = cute_ext.make_tmem_layout_acc(
            mma, (self.up_rows, self.up_channels), acc_stage=self.up_parts
        )
        accumulator = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base),
            layout,
        )[((None, None), 0, 0, part)]
        column = branch * self.up_hidden + (tid // 128) * self.reader_columns
        tile = (self.up_rows, self.reader_columns)
        accumulator = cute.make_tensor(
            accumulator.iterator + accumulator.layout((0, column)),
            cute.make_layout(
                (cute.shape(accumulator, mode=[0]), self.reader_columns),
                stride=(accumulator.stride[0], accumulator.stride[1]),
            ),
        )
        coords = cute.local_tile(
            cute.make_identity_tensor((self.up_rows, self.up_hidden)),
            tile,
            (0, tid // 128),
        )
        copy = tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (self.up_rows, self.up_channels, self.up_k),
                cute_utils.LayoutEnum.COL_MAJOR,
                cutlass.Float32,
                cutlass.Float32,
                tile,
                False,
            ),
            accumulator,
        )
        reader = copy.get_slice(tid % 128)
        dummy = cute.make_tensor(
            original.iterator, cute.make_layout(tile, stride=(1, self.up_rows))
        )
        values = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.rmem,
            cute_ext.make_t2r_rmem_layout(
                copy, cute.flat_divide(dummy, tile), tid % 128
            ),
            alignment=32,
        )
        return accumulator, reader, values, reader.partition_D(coords)

    @cute.experimental.jit
    def _hidden_reader(self, mma, base, original, tid, token_slice: cutlass.Constexpr):
        fragment = mma.get_slice(0).make_fragment_C(
            mma.partition_shape_C((128, self.up_rows))
        )
        accumulator = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base),
            fragment.layout,
        )[(None, None), 0, 0]
        tile = (128, self.hidden_reader_columns)
        group = tid // 128 + token_slice * 4
        accumulator = cute.local_tile(accumulator, tile, (0, group))
        coords = cute.local_tile(
            cute.make_identity_tensor((128, self.up_rows)), tile, (0, group)
        )
        copy = tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (128, self.up_rows, self.up_k),
                cute_utils.LayoutEnum.COL_MAJOR,
                cutlass.Float32,
                cutlass.Float32,
                tile,
                False,
            ),
            accumulator,
        )
        reader = copy.get_slice(tid % 128)
        dummy = cute.make_tensor(
            original.iterator, cute.make_layout(tile, stride=(1, 128))
        )
        values = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.rmem,
            cute_ext.make_t2r_rmem_layout(
                copy, cute.flat_divide(dummy, tile), tid % 128
            ),
            alignment=32,
        )
        return accumulator, reader, values, reader.partition_D(coords)

    @cute.experimental.jit
    def _prefetch_branch_x(
        self,
        target,
        coords,
        x,
        hidden_tile,
        global_base,
        part: cutlass.Constexpr,
        branch: cutlass.Constexpr,
    ):
        # M128's reader owns a contiguous hidden vector within one token.
        row, col = coords[0]
        global_row = global_base + cutlass.Int64(part * self.up_rows + row)
        if global_row < x.shape[0]:
            copy_bits = min(128, cute.size(target) * 8)
            pointer = cute.make_ptr(
                cutlass.Float8E4M3FN,
                (
                    x.iterator
                    + global_row * (self.g * self.h)
                    + branch * self.h
                    + hidden_tile * self.up_hidden
                    + col
                ).toint(),
                cute.AddressSpace.gmem,
                assumed_align=copy_bits // 8,
            )
            source = cute.make_tensor(pointer, cute.make_layout(cute.size(target)))
            atom = cute.make_copy_atom(
                CopyUniversalOp(), cutlass.Float8E4M3FN, num_bits_per_copy=copy_bits
            )
            cute.copy(atom, source, target)

    @cute.experimental.jit
    def _up(
        self,
        activation,
        alpha,
        x,
        xs,
        weight,
        ws,
        out,
        arena,
        bars,
        base,
        hidden_tile,
        global_base,
    ):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        cached = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout(2 * self.n + self.up_channels),
            alignment=16,
        )
        residual_tile = None
        if cutlass.const_expr(not self.token_major_up):
            residual_tile = cute_ext.allocate(
                cutlass.Float8E4M3FN,
                cute.AddressSpace.smem,
                cute.make_layout((self.up_rows, 128), stride=(128, 1)),
                alignment=16,
            )
        dtype = cutlass.Float8E4M3FN
        mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            self.up_mn,
        )
        tiler = (*self.up_mn, self.up_k)
        al = sm100_utils.make_smem_layout_a(mma, tiler, dtype, self.up_stages)
        bl = sm100_utils.make_smem_layout_b(mma, tiler, dtype, self.up_stages)
        sw = _operand_at(arena, bl, 0)
        sa0 = _operand_at(arena, al, cute.cosize(bl))
        sa1 = _operand_at(arena, al, cute.cosize(bl) + cute.cosize(al))
        full, empty = bars, bars + self.up_stages
        done = empty + self.up_stages
        read_done = done + self.up_parts
        if warp == 17:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.up_stages):
                    cute.arch.mbarrier_init(full + stage, 1)
                    cute.arch.mbarrier_init(empty + stage, 1)
                for part in cutlass.range_constexpr(self.up_parts):
                    cute.arch.mbarrier_init(done + part, 1)
                cute.arch.mbarrier_init(read_done, 512)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        tcgen05_fence("after_thread_sync")
        av = cute.make_tensor(
            activation.iterator,
            cute.make_layout(
                (activation.shape[0], (64, self.r // 64), 1),
                stride=(self.r, (1, 64), 0),
            ),
        )
        wv = cute.make_tensor(
            weight.iterator,
            cute.make_layout(
                (
                    (self.up_hidden, self.g, self.h // self.up_hidden),
                    (64, self.r // 64),
                    1,
                ),
                stride=((self.r, self.h * self.r, self.up_hidden * self.r), (1, 64), 0),
            ),
        )
        gx0 = cute.local_tile(av, (self.up_rows, self.up_k), (0, None, 0))
        gx1 = cute.local_tile(av, (self.up_rows, self.up_k), (1, None, 0))
        gw = cute.local_tile(wv, (self.up_channels, self.up_k), (hidden_tile, None, 0))
        if cutlass.const_expr(self.token_major_up):
            ga0, ga1, gb = gx0, gx1, gw
            ma = cute_ext.get_cta_v_map_ab(av, tiler, mma, "A")
            mb = cute_ext.get_cta_v_map_ab(wv, tiler, mma, "B")
        else:
            ga0, ga1, gb = gw, gw, gx0
            ma = cute_ext.get_cta_v_map_ab(wv, tiler, mma, "A")
            mb = cute_ext.get_cta_v_map_ab(av, tiler, mma, "B")
        if warp == 16:
            for tile in cutlass.range_constexpr(self.up_stages):
                slot = tile
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + slot,
                        (self.up_channels + self.up_rows * self.up_parts) * self.up_k,
                    )
                cute_ext.tma_load(
                    gb[None, None, tile],
                    sw[None, None, None, slot],
                    (full + slot).value,
                    cta_v_map=mb,
                    update_expect_tx=False,
                )
                cute_ext.tma_load(
                    ga0[None, None, tile],
                    sa0[None, None, None, slot],
                    (full + slot).value,
                    cta_v_map=ma,
                    update_expect_tx=False,
                )
                if cutlass.const_expr(self.up_parts == 2):
                    cute_ext.tma_load(
                        ga1[None, None, tile],
                        sa1[None, None, None, slot],
                        (full + slot).value,
                        cta_v_map=ma,
                        update_expect_tx=False,
                    )
            for tile in cutlass.range(self.up_stages, self.up_tiles, unroll=1):
                slot = tile % self.up_stages
                cute.arch.mbarrier_wait(empty + slot, (tile // self.up_stages - 1) % 2)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + slot,
                        (self.up_channels + self.up_rows * self.up_parts) * self.up_k,
                    )
                cute_ext.tma_load(
                    gb[None, None, tile],
                    sw[None, None, None, slot],
                    (full + slot).value,
                    cta_v_map=mb,
                    update_expect_tx=False,
                )
                cute_ext.tma_load(
                    ga0[None, None, tile],
                    sa0[None, None, None, slot],
                    (full + slot).value,
                    cta_v_map=ma,
                    update_expect_tx=False,
                )
                if cutlass.const_expr(self.up_parts == 2):
                    cute_ext.tma_load(
                        ga1[None, None, tile],
                        sa1[None, None, None, slot],
                        (full + slot).value,
                        cta_v_map=ma,
                        update_expect_tx=False,
                    )
        elif warp == 17:
            tm = mma.get_slice(0)
            ad0, bd = tm.make_fragment_A(sa0), tm.make_fragment_B(sw)
            ad1 = ad0
            if cutlass.const_expr(self.up_parts == 2):
                ad1 = tm.make_fragment_A(sa1)
            layout = cute_ext.make_tmem_layout_acc(
                mma, self.up_mn, acc_stage=self.up_parts
            )
            all_acc = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), layout
            )
            acc0 = all_acc[None, None, None, 0]
            acc1 = acc0
            if cutlass.const_expr(self.up_parts == 2):
                acc1 = all_acc[None, None, None, 1]
            mma.set(tcgen05.Field.ACCUMULATE, False)
            for tile in cutlass.range(self.up_tiles - self.up_tail_tiles, unroll=1):
                slot = tile % self.up_stages
                cute.arch.mbarrier_wait(full + slot, tile // self.up_stages % 2)
                for block in cutlass.range_constexpr(cute.size(ad0, mode=[2])):
                    cute.gemm(
                        mma,
                        acc0,
                        ad0[None, None, block, slot],
                        bd[None, None, block, slot],
                        acc0,
                    )
                    if cutlass.const_expr(self.up_parts == 2):
                        cute.gemm(
                            mma,
                            acc1,
                            ad1[None, None, block, slot],
                            bd[None, None, block, slot],
                            acc1,
                        )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(empty + slot, None, tcgen05.CtaGroup.ONE)
            if cutlass.const_expr(self.up_parts == 2):
                for tile in cutlass.range(
                    self.up_tiles - self.up_tail_tiles, self.up_tiles, unroll=1
                ):
                    slot = tile % self.up_stages
                    cute.arch.mbarrier_wait(full + slot, tile // self.up_stages % 2)
                    for block in cutlass.range_constexpr(cute.size(ad0, mode=[2])):
                        cute.gemm(
                            mma,
                            acc0,
                            ad0[None, None, block, slot],
                            bd[None, None, block, slot],
                            acc0,
                        )
                    # Retain these last operand slots for the other token part.
                with cute.arch.elect_one():
                    tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
                for tile in cutlass.range(
                    self.up_tiles - self.up_tail_tiles, self.up_tiles, unroll=1
                ):
                    slot = tile % self.up_stages
                    for block in cutlass.range_constexpr(cute.size(ad1, mode=[2])):
                        cute.gemm(
                            mma,
                            acc1,
                            ad1[None, None, block, slot],
                            bd[None, None, block, slot],
                            acc1,
                        )
                    with cute.arch.elect_one():
                        tcgen05.commit(empty + slot, None, tcgen05.CtaGroup.ONE)
                with cute.arch.elect_one():
                    tcgen05.commit(done + 1, None, tcgen05.CtaGroup.ONE)
            else:
                with cute.arch.elect_one():
                    tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
            cute.arch.mbarrier_wait(read_done, 0)
        elif tid < 512:
            if cutlass.const_expr(self.token_major_up):
                if tid < self.n:
                    global_row = global_base + cutlass.Int64(tid)
                    a_scale, x_scale = cutlass.Float32(0.0), cutlass.Float32(0.0)
                    if global_row < x.shape[0]:
                        a_scale = _load_f32(alpha.iterator + cutlass.Int64(tid))
                        x_scale = _x_scale(xs, global_row)
                    cached[tid] = a_scale
                    cached[self.n + tid] = x_scale
                if tid < self.up_channels:
                    channel = (
                        (tid // self.up_hidden) * self.h
                        + hidden_tile * self.up_hidden
                        + tid % self.up_hidden
                    )
                    cached[2 * self.n + tid] = ws[channel if self.up_per_channel else 0]
                _epilogue_barrier(512, cutlass.Int32(5))
                for part in cutlass.range_constexpr(self.up_parts):
                    _, _, template, coords = self._reader(mma, base, x, tid, 0, part)
                    sums = cute.make_rmem_tensor(cute.size(template), cutlass.Float32)
                    sums.fill(0.0)
                    x_ring = cute.make_rmem_tensor(
                        (cute.size(template), 2), cutlass.Float8E4M3FN
                    )
                    x_ring.fill(0.0)
                    self._prefetch_branch_x(
                        x_ring[None, 0], coords, x, hidden_tile, global_base, part, 0
                    )
                    for branch in cutlass.range_constexpr(self.g):
                        if cutlass.const_expr(branch + 1 < self.g):
                            self._prefetch_branch_x(
                                x_ring[None, (branch + 1) % 2],
                                coords,
                                x,
                                hidden_tile,
                                global_base,
                                part,
                                branch + 1,
                            )
                        cute.arch.mbarrier_wait(done + part, 0)
                        acc, reader, values, _ = self._reader(
                            mma, base, x, tid, branch, part
                        )
                        cute_ext.partition_and_copy(reader, acc, values)
                        cute.arch.fence_view_async_tmem_load()
                        for i in cutlass.range_constexpr(cute.size(values)):
                            row, col = coords[i]
                            local_row = part * self.up_rows + row
                            global_row = global_base + cutlass.Int64(local_row)
                            if global_row < x.shape[0]:
                                factor = (
                                    cached[local_row]
                                    * cached[2 * self.n + branch * self.up_hidden + col]
                                )
                                xv = (
                                    x_ring[i, branch % 2].to(cutlass.Float32)
                                    * cached[self.n + local_row]
                                )
                                sums[i] = sums[i] + xv * _up_sigmoid_tanh(
                                    values[i] * factor
                                )
                    packed = cute.make_rmem_tensor(cute.size(sums), out.element_type)
                    for i in cutlass.range_constexpr(cute.size(sums)):
                        packed[i] = (sums[i] / self.g).to(out.element_type)
                    row, col = coords[0]
                    global_row = global_base + cutlass.Int64(part * self.up_rows + row)
                    if global_row < x.shape[0]:
                        copy_bits = min(128, cute.size(sums) * 16)
                        pointer = cute.make_ptr(
                            out.element_type,
                            (
                                out.iterator
                                + global_row * self.h
                                + hidden_tile * self.up_hidden
                                + col
                            ).toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=copy_bits // 8,
                        )
                        target = cute.make_tensor(
                            pointer, cute.make_layout(cute.size(sums))
                        )
                        atom = cute.make_copy_atom(
                            CopyUniversalOp(),
                            out.element_type,
                            num_bits_per_copy=copy_bits,
                        )
                        cute.copy(atom, packed, target)
                tcgen05_fence("before_thread_sync")
                cute.arch.mbarrier_arrive(read_done)
            else:
                for copy_slice in cutlass.range_constexpr(
                    (self.up_rows * 128 + 8191) // 8192
                ):
                    linear = tid * 16 + copy_slice * 8192
                    if linear < self.up_rows * 128:
                        local_row = linear // 128
                        channel = linear % 128
                        global_row = global_base + cutlass.Int64(local_row)
                        source = (
                            x.iterator
                            + global_row * (self.g * self.h)
                            + (channel // 32) * self.h
                            + hidden_tile * 32
                            + channel % 32
                        )
                        size = cutlass.Int32(0)
                        if global_row < x.shape[0]:
                            size = cutlass.Int32(16)
                        _copy_residual_16(residual_tile.iterator + linear, source, size)
                _commit_residual()
                if tid < self.up_rows:
                    global_row = global_base + cutlass.Int64(tid)
                    activation_scale = cutlass.Float32(0.0)
                    input_scale = cutlass.Float32(0.0)
                    if global_row < x.shape[0]:
                        activation_scale = _load_f32(
                            alpha.iterator + cutlass.Int64(tid)
                        )
                        input_scale = _x_scale(xs, global_row)
                    cached[tid] = activation_scale
                    cached[self.up_rows + tid] = input_scale
                if tid < 128:
                    channel = (tid // 32) * self.h + hidden_tile * 32 + tid % 32
                    cached[2 * self.up_rows + tid] = ws[
                        channel if self.up_per_channel else 0
                    ]
                cute.arch.mbarrier_wait(done, 0)
                gates = cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Float32,
                        arena.iterator.toint(),
                        cute.AddressSpace.smem,
                        assumed_align=1024,
                    ),
                    cute.make_layout((self.up_rows, 128), stride=(128, 1)),
                )
                for token_slice in cutlass.range_constexpr(self.hidden_reader_passes):
                    acc, reader, values, coords = self._hidden_reader(
                        mma, base, x, tid, token_slice
                    )
                    cute_ext.partition_and_copy(reader, acc, values)
                    cute.arch.fence_view_async_tmem_load()
                    for i in cutlass.range_constexpr(cute.size(values)):
                        channel, row = coords[i]
                        gates[row, channel] = values[i]
                tcgen05_fence("before_thread_sync")
                _wait_residual()
                cute.arch.mbarrier_arrive(read_done)
        cute.arch.sync_threads()
        if cutlass.const_expr(not self.token_major_up):
            if tid < 512:
                gates = cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Float32,
                        arena.iterator.toint(),
                        cute.AddressSpace.smem,
                        assumed_align=1024,
                    ),
                    cute.make_layout((self.up_rows, 128), stride=(128, 1)),
                )
                col = tid % 32
                hidden = hidden_tile * 32 + col
                scales = cute.make_rmem_tensor(self.g, cutlass.Float32)
                for branch in cutlass.range_constexpr(self.g):
                    channel = branch * self.h + hidden
                    scales[branch] = cached[2 * self.up_rows + branch * 32 + col]
                for row_part in cutlass.range_constexpr(self.up_rows // 16):
                    row = tid // 32 + row_part * 16
                    global_row = global_base + cutlass.Int64(row)
                    if global_row < x.shape[0]:
                        activation_scale = cached[row]
                        input_scale = cached[self.up_rows + row]
                        total = cutlass.Float32(0.0)
                        for branch in cutlass.range_constexpr(self.g):
                            channel = branch * self.h + hidden
                            gate = gates[row, branch * 32 + col]
                            factor = activation_scale * scales[branch]
                            residual = (
                                residual_tile[row, branch * 32 + col].to(
                                    cutlass.Float32
                                )
                                * input_scale
                            )
                            total = total + residual * _up_sigmoid_tanh(gate * factor)
                        out[global_row, hidden] = (total / self.g).to(out.element_type)
