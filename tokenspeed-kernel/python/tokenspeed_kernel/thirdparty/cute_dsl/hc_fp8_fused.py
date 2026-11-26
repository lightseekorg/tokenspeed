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
from cuda.bindings.driver import CUstream
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05
from cutlass.experimental.primitives import tcgen05_fence
from tokenspeed_kernel.thirdparty.cute_dsl._fp8_mixer.common import (
    _acquire,
    _has_full_up_tiles,
    _init_mma_barriers,
    _make_down_fragments,
    _make_down_operands,
    _mma_barriers,
    _release,
    _run_cute_mma_ring,
    _tma_matrix_view,
    _wait_residual,
)
from tokenspeed_kernel.thirdparty.cute_dsl._fp8_mixer.resident import (
    MixerOperands,
    QuantizationConfig,
    ResidentDown,
    ResidentMixer,
    _mixer_cta,
    _mixer_shared,
    _read_down_partials,
)
from tokenspeed_kernel.thirdparty.cute_dsl.hc_fp8 import (
    _epilogue_barrier,
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


class _DownPhase(ResidentDown):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.producer_warp = 16
        self.mma_warp = 17

    @cute.experimental.jit
    def _load_down_stage(self, operands, full, stage, tile):
        for index in cutlass.range_constexpr(2):
            source, target, mapping = operands[index]
            cute_ext.tma_load(
                source[None, None, tile],
                target[None, None, None, stage],
                (full + stage).value,
                cta_v_map=mapping,
                update_expect_tx=False,
            )

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
        full, empty, done, read_done = _mma_barriers(
            barriers, self.stages, self.stages, 1
        )
        if warp == self.mma_warp:
            with cute.arch.elect_one():
                _init_mma_barriers(
                    barriers, self.stages, self.stages, 1, 1, 0, self.epilogue_threads
                )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        tcgen05_fence("after_thread_sync")
        w_view = _tma_matrix_view(weight, self.projection_rows, self.inner, self.inner)
        x_view = _tma_matrix_view(x, x.shape[0], self.inner, self.inner)
        gw = cute.local_tile(w_view, (self.m, self.k), (output_tile, None, 0))
        gx = cute.local_tile(x_view, (self.n, self.k), (token_tile, None, 0))
        map_w = cute_ext.get_cta_v_map_ab(w_view, tiler, mma, "A")
        map_x = cute_ext.get_cta_v_map_ab(x_view, tiler, mma, "B")
        if warp == self.producer_warp:
            self._feed_down_ring(
                ((gw, sw, map_w), (gx, sx, map_x)), full, empty, split, False, None
            )
        elif warp == self.mma_warp:
            acc, a_desc, b_desc = _make_down_fragments(
                mma.get_slice(0),
                tmem_base,
                sw,
                sx,
                mma.partition_shape_C((self.m, self.n)),
            )
            _run_cute_mma_ring(
                mma,
                acc,
                a_desc,
                b_desc,
                full,
                empty,
                self.tiles,
                self.stages,
                None,
                True,
            )
            with cute.arch.elect_one():
                tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
            cute.arch.mbarrier_wait(read_done, 0)
        elif tid < self.epilogue_threads:
            cute.arch.mbarrier_wait(done, 0)
            values, coords = _read_down_partials(
                mma,
                tmem_base,
                partials,
                tid,
                (self.m, self.n, self.k),
                self.epilogue_groups,
            )
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
    def _load_up_stage(self, sources, targets, mappings, full, tile, slot):
        ga0, ga1, gb = sources
        sa0, sa1, sw = targets
        ma, mb = mappings
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(
                full, (self.up_channels + self.up_rows * self.up_parts) * self.up_k
            )
        self._load_up_tma_operand(gb, sw, tile, slot, full, mb)
        self._load_up_tma_operand(ga0, sa0, tile, slot, full, ma)
        if cutlass.const_expr(self.up_parts == 2):
            self._load_up_tma_operand(ga1, sa1, tile, slot, full, ma)

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
        cached = self._up_scale_storage(2 * self.n)
        residual_tile = None
        if cutlass.const_expr(not self.token_major_up):
            residual_tile = self._up_residual_tile(self.up_channels, 16)
        mma, tiler, al, bl = self._up_operand_layouts()
        sw, sa0, sa1 = self._up_operand_storage(arena, al, bl)
        full, empty, done, read_done = _mma_barriers(
            bars, self.up_stages, self.up_stages, self.up_parts
        )
        if warp == 17:
            with cute.arch.elect_one():
                _init_mma_barriers(
                    bars, self.up_stages, self.up_stages, self.up_parts, 1, 0, 512
                )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        tcgen05_fence("after_thread_sync")
        ga0, ga1, gb, ma, mb = self._up_tma_operands(
            mma, tiler, activation, weight, hidden_tile
        )
        if warp == 16:
            for tile in cutlass.range_constexpr(self.up_stages):
                slot = tile
                self._load_up_stage(
                    (ga0, ga1, gb), (sa0, sa1, sw), (ma, mb), full + slot, tile, slot
                )
            for tile in cutlass.range(self.up_stages, self.up_tiles, unroll=1):
                slot = tile % self.up_stages
                cute.arch.mbarrier_wait(empty + slot, (tile // self.up_stages - 1) % 2)
                self._load_up_stage(
                    (ga0, ga1, gb), (sa0, sa1, sw), (ma, mb), full + slot, tile, slot
                )
        elif warp == 17:
            fragments = self._up_mma_fragments(mma, base, sa0, sa1, sw)
            mma.set(tcgen05.Field.ACCUMULATE, False)
            self._run_up_mma_tiles(
                mma,
                fragments,
                full,
                (empty, 0),
                None,
                self.up_tiles - self.up_tail_tiles,
                False,
            )
            if cutlass.const_expr(self.up_parts == 2):
                self._run_paired_up_tail(mma, fragments, full, empty, done, False)
            else:
                self._publish_up_done(done)
            cute.arch.mbarrier_wait(read_done, 0)
        elif tid < 512:
            if cutlass.const_expr(self.token_major_up):
                self._cache_up_row_scales(
                    alpha, xs, cached, x, global_base, tid, self.n, True
                )
                self._cache_up_weight_scales(
                    ws, cached, hidden_tile, tid, 2 * self.n, False
                )
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
                self._copy_hidden_residual(
                    x,
                    residual_tile,
                    hidden_tile,
                    global_base,
                    tid,
                    512,
                    self.up_channels,
                )
                self._cache_up_row_scales(
                    alpha, xs, cached, x, global_base, tid, self.up_rows, True
                )
                self._cache_up_weight_scales(
                    ws, cached, hidden_tile, tid, 2 * self.up_rows, False
                )
                cute.arch.mbarrier_wait(done, 0)
                gates = self._hidden_gate_tile(arena.iterator)
                values, coords = self._read_hidden_up(mma, base, x, tid, None)
                for i in cutlass.range_constexpr(cute.size(values)):
                    channel, row = coords[i]
                    gates[row, channel] = values[i]
                tcgen05_fence("before_thread_sync")
                _wait_residual()
                cute.arch.mbarrier_arrive(read_done)
        cute.arch.sync_threads()
        if cutlass.const_expr(not self.token_major_up):
            if tid < 512:
                gates = self._hidden_gate_tile(arena.iterator)
                self._mix_hidden_major_up(
                    gates,
                    residual_tile,
                    cached,
                    out,
                    x.shape[0],
                    tid,
                    hidden_tile,
                    global_base,
                    512,
                )
