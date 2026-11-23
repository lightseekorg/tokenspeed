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
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05
from cutlass.experimental.primitives import tcgen05_fence
from tokenspeed_kernel.thirdparty.cute_dsl._fp8_mixer.common import (
    _acquire,
    _commit_residual,
    _copy_residual_16,
    _has_full_up_tiles,
    _load_f32,
    _make_down_operands,
    _operand_at,
    _release,
    _wait_residual,
    _x_scale,
)
from tokenspeed_kernel.thirdparty.cute_dsl._fp8_mixer.resident import (
    MixerOperands,
    QuantizationConfig,
    ResidentMixer,
    _mixer_cta,
    _mixer_shared,
)
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
def _grid_barrier(epochs, expected, wait_for_all):
    # Cooperative launch and measured capacity keep every participant resident.
    cute.arch.sync_threads()
    if cute.arch.thread_idx()[0] == 0:
        _arrive_counter(epochs.iterator)
        if wait_for_all:
            while _acquire(epochs.iterator) < expected:
                pass
    cute.arch.sync_threads()


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
        tiler = (self.m, self.n, self.k)
        mma, sw, sx = _make_down_operands(arena, tiler, self.stages)
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
            # The enclosing kernel already waited for input and weight producers.
            # Pair each stage's operands so its MMA can start before later loads.
            for stage in cutlass.range_constexpr(self.stages):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(full + stage, stage_bytes)
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


class Fp8FusedGatedResidualKernel(ResidentMixer):
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
        full_up_tiles = _has_full_up_tiles(r)
        self.up_hidden = 64 if full_up_tiles else 32
        self.up_channels = self.up_hidden * g
        self.up_parts = 2 if token_bucket == 256 else 1
        # Small narrow tiles avoid padding tokens to the native M128 reader.
        self.token_major_up = full_up_tiles or token_bucket == 256
        self.up_rows = (
            min(128, max(64, token_bucket)) if self.token_major_up else token_bucket
        )
        self.up_mn = (
            (self.up_rows, self.up_channels)
            if self.token_major_up
            else (self.up_channels, self.up_rows)
        )
        self.reader_groups = 4
        self.hidden_reader_columns = min(32, token_bucket // self.reader_groups)
        self.hidden_reader_passes = 1
        self.reader_columns = self.up_hidden // self.reader_groups
        self.up_k = 128 if full_up_tiles else 64
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
        self.splits = 8 if full_up_tiles else 16
        down_k = (512 if token_bucket == 16 else 256) if full_up_tiles else 128
        if token_bucket >= 128:
            down_k = 256 if full_up_tiles else 128
        down_stages = 2 if full_up_tiles and token_bucket == 256 else 3
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
        self.down_token_parts = self.quant_rows = 1
        self.quant_threads = 512
        self.quant_columns = 2
        self.quantization = QuantizationConfig(
            threads=self.quant_threads,
            maximum_threads=self.quant_threads,
            injection_start=0,
            column_stride=1,
            warp_max="shuffle",
            replicate_maximum=False,
            refined_scales=False,
            clamp="both",
            activation_store="native",
            release_writes=False,
        )
        self.mma_warp = 17
        self.down_barrier_count = 2 * self.down.stages + 2
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
        cta = _mixer_cta()
        tid, warp, pid, ctas = cta
        shared = _mixer_shared(self.arena_bytes, self.barrier_count, 0)
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()
        if tid == 0:
            shared.epoch[0] = _acquire(epochs.iterator + cutlass.Int64(pid) + 1) + 1
        if warp == 17:
            cute.arch.alloc_tmem(self.columns, shared.tmem, is_two_cta=False)
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.sync_threads()
        generation = shared.epoch[0]
        self._run_mixer_rounds(
            MixerOperands(
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
            ),
            cta,
            shared,
            None,
            None,
            generation,
            entry_prefills_down=False,
            prefetch_residual_before_down=False,
            skip_last_up_join=False,
            track_last_up_task=False,
        )
        if warp == 17:
            cute.arch.dealloc_tmem(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, shared.tmem),
                self.columns,
            )
        cute.arch.sync_threads()
        if cutlass.const_expr(self.pdl):
            if tid == 0:
                cute.arch.griddepcontrol_launch_dependents()

    @cute.experimental.jit
    def _project_down_tile(self, io, cta, shared, maps, state, tile, global_base):
        self.down.project(
            io.x,
            io.down,
            io.partials,
            shared.arena,
            shared.down_barriers,
            shared.tmem,
            tile.output_tile,
            tile.split,
            tile.token_tile,
        )
        cute.arch.sync_threads()

    @cute.experimental.jit
    def _begin_quantization(
        self, io, cta, shared, maps, state, generation, global_base
    ):
        _grid_barrier(io.epochs, ((generation - 1) * 3 + 1) * cta.count, True)
        row = cutlass.Int64(cta.pid)
        return generation, row, None

    @cute.experimental.jit
    def _finish_quantization(self, io, cta, state, generation, quant):
        _publish_activation_proxy()
        _grid_barrier(io.epochs, ((generation - 1) * 3 + 2) * cta.count, True)
        return None

    @cute.experimental.jit
    def _project_up_tile(
        self, io, shared, maps, state, tile, global_base, activation_target
    ):
        self._up(
            io.activation,
            io.alpha,
            io.x,
            io.xs,
            io.up,
            io.us,
            io.out,
            shared.arena,
            shared.up_barriers,
            shared.tmem,
            tile.hidden_tile,
            global_base,
        )

    @cute.experimental.jit
    def _finish_round(
        self, io, cta, shared, state, generation, activation_target, more_rounds
    ):
        if cta.tid == 0:
            _release(io.epochs.iterator + cutlass.Int64(cta.pid) + 1, generation)
        # Another round needs a wait before scratch reuse. On the final
        # round, stream/PDL ordering protects the next invocation instead.
        _grid_barrier(io.epochs, generation * 3 * cta.count, more_rounds)
        generation = generation + 1
        return generation

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
                cute.make_layout(
                    (self.up_rows, self.up_channels), stride=(self.up_channels, 1)
                ),
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
                self._mix_token_major_up(
                    mma,
                    base,
                    x,
                    xs,
                    alpha,
                    out,
                    cached,
                    tid,
                    hidden_tile,
                    global_base,
                    done,
                    read_done,
                    None,
                    row_scale_source="shared_element",
                    weight_cache_offset=2 * self.n,
                    vector_weight_scales=False,
                    prefetched_branches=0,
                    wait_once=False,
                    release_after_reads=False,
                )
                tcgen05_fence("before_thread_sync")
                cute.arch.mbarrier_arrive(read_done)
            else:
                tile_bytes = self.up_rows * self.up_channels
                copy_bytes = 512 * 16
                for copy_slice in cutlass.range_constexpr(
                    (tile_bytes + copy_bytes - 1) // copy_bytes
                ):
                    linear = tid * 16 + copy_slice * copy_bytes
                    if linear < tile_bytes:
                        local_row = linear // self.up_channels
                        channel = linear % self.up_channels
                        global_row = global_base + cutlass.Int64(local_row)
                        source = (
                            x.iterator
                            + global_row * (self.g * self.h)
                            + (channel // self.up_hidden) * self.h
                            + hidden_tile * self.up_hidden
                            + channel % self.up_hidden
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
                if tid < self.up_channels:
                    channel = (
                        (tid // self.up_hidden) * self.h
                        + hidden_tile * self.up_hidden
                        + tid % self.up_hidden
                    )
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
                    cute.make_layout(
                        (self.up_rows, self.up_channels), stride=(self.up_channels, 1)
                    ),
                )
                for token_slice in cutlass.range_constexpr(self.hidden_reader_passes):
                    acc, reader, values, coords = self._hidden_reader(
                        mma, base, x, tid, token_slice, None
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
                    cute.make_layout(
                        (self.up_rows, self.up_channels), stride=(self.up_channels, 1)
                    ),
                )
                col = tid % self.up_hidden
                hidden = hidden_tile * self.up_hidden + col
                scales = cute.make_rmem_tensor(self.g, cutlass.Float32)
                for branch in cutlass.range_constexpr(self.g):
                    channel = branch * self.h + hidden
                    scales[branch] = cached[
                        2 * self.up_rows + branch * self.up_hidden + col
                    ]
                for row_part in cutlass.range_constexpr(self.up_rows // 16):
                    row = tid // self.up_hidden + row_part * (512 // self.up_hidden)
                    global_row = global_base + cutlass.Int64(row)
                    if global_row < x.shape[0]:
                        activation_scale = cached[row]
                        input_scale = cached[self.up_rows + row]
                        total = cutlass.Float32(0.0)
                        for branch in cutlass.range_constexpr(self.g):
                            channel = branch * self.h + hidden
                            gate = gates[row, branch * self.up_hidden + col]
                            factor = activation_scale * scales[branch]
                            residual = (
                                residual_tile[row, branch * self.up_hidden + col].to(
                                    cutlass.Float32
                                )
                                * input_scale
                            )
                            total = total + residual * _up_sigmoid_tanh(gate * factor)
                        out[global_row, hidden] = (total / self.g).to(out.element_type)
