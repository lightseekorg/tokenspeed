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

"""Resident CTA and token-round template for prequantized FP8 mixers."""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import CopyUniversalOp
from cutlass.experimental.primitives import tcgen05_fence
from tokenspeed_kernel.thirdparty.cute_dsl._fp8_mixer.common import (
    _decode_residual,
    _hidden_major_reader,
    _invalidate,
    _load_f32,
    _prefetch_branch_residual,
    _store_token_major_mix,
    _token_major_reader,
    _x_scale,
)
from tokenspeed_kernel.thirdparty.cute_dsl.hc_fused import _up_sigmoid_tanh


class MixerOperands(NamedTuple):
    x: cute.Tensor
    xs: cute.Tensor
    down: cute.Tensor
    ds: cute.Tensor
    up: cute.Tensor
    us: cute.Tensor
    partials: cute.Tensor
    activation: cute.Tensor
    alpha: cute.Tensor
    epochs: cute.Tensor
    out: cute.Tensor
    inject: cute.Tensor


class MixerCta(NamedTuple):
    tid: cutlass.Int32
    warp: cutlass.Int32
    pid: cutlass.Int32
    count: cutlass.Int64


class MixerShared(NamedTuple):
    arena: cute.Tensor
    down_barriers: cute.Pointer
    up_barriers: cute.Pointer
    tmem: cute.Pointer
    epoch: cute.Tensor


class DownTask(NamedTuple):
    output_tile: cutlass.Int32
    split: cutlass.Int32
    token_tile: cutlass.Int64
    entry_prefilled: cutlass.Boolean
    last: cutlass.Boolean


class UpTask(NamedTuple):
    hidden_tile: cutlass.Int32
    weights_prepared: cutlass.Boolean
    last: cutlass.Boolean


@cute.experimental.jit
def _mixer_cta():
    tid = cute.arch.thread_idx()[0]
    return MixerCta(
        tid,
        cute.arch.make_warp_uniform(tid // 32),
        cute.arch.block_idx()[0],
        cutlass.Int64(cute.arch.grid_dim()[0]),
    )


@cute.experimental.jit
def _mixer_shared(
    arena_bytes: cutlass.Constexpr,
    barrier_count: cutlass.Constexpr,
    up_barrier_offset: cutlass.Constexpr,
):
    arena = cute_ext.allocate(
        cutlass.Uint8,
        cute.AddressSpace.smem,
        cute.make_layout(arena_bytes),
        alignment=1024,
    )
    bars = cute_ext.allocate(
        cutlass.Int64,
        cute.AddressSpace.smem,
        cute.make_layout(barrier_count),
        alignment=8,
    )
    up_bars = bars.iterator
    if cutlass.const_expr(up_barrier_offset):
        up_bars = up_bars + up_barrier_offset
    base = cute_ext.allocate(
        cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
    )
    epoch = cute_ext.allocate(
        cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
    )
    return MixerShared(arena, bars.iterator, up_bars, base.iterator, epoch)


class ResidentMixer:
    """Compile-time mixer schedule with a shared resident task loop."""

    @cute.experimental.jit
    def _down_tile_coordinates(self, task, token_round):
        output_tile = task // self.splits
        input_tile = token_round
        if cutlass.const_expr(self.down_token_parts > 1):
            token_part = output_tile // self.down_tiles
            input_tile = token_round * self.down_token_parts + cutlass.Int64(token_part)
            output_tile = output_tile % self.down_tiles
        return output_tile, task % self.splits, input_tile

    @cute.experimental.jit
    def _run_mixer_rounds(
        self,
        io,
        cta,
        shared,
        maps,
        state,
        phase,
        entry_prefills_down: cutlass.Constexpr,
        prefetch_residual_before_down: cutlass.Constexpr,
        skip_last_up_join: cutlass.Constexpr,
        track_last_up_task: cutlass.Constexpr,
    ):
        """Walk resident work in split-K, quantization, and Up order.

        The plan supplies operand binding and publication. The template owns task
        assignment, row traversal and barrier retirement before scratch reuse.
        """
        global_base = cutlass.Int64(0)
        token_round = cutlass.Int64(0)
        task_stride = cutlass.Int32(cta.count)
        row_stride = cta.count * self.quant_rows
        while global_base < io.x.shape[0]:
            if cutlass.const_expr(prefetch_residual_before_down):
                self._prefetch_round_residual(io, cta, shared, maps, state, global_base)

            task = cutlass.Int32(cta.pid)
            while task < self.down_tasks:
                output_tile, split, input_tile = self._down_tile_coordinates(
                    task, token_round
                )
                entry_prefilled = cutlass.Boolean(False)
                if cutlass.const_expr(entry_prefills_down):
                    entry_prefilled = (global_base == 0) & (
                        task == cutlass.Int32(cta.pid)
                    )
                tile = DownTask(
                    output_tile,
                    split,
                    input_tile,
                    entry_prefilled,
                    task + task_stride >= self.down_tasks,
                )
                self._project_down_tile(io, cta, shared, maps, state, tile, global_base)
                if cta.warp == self.mma_warp:
                    _invalidate(shared.down_barriers, self.down_barrier_count)
                cute.arch.sync_threads()
                task = task + task_stride

            phase, row, quant = self._begin_quantization(
                io, cta, shared, maps, state, phase, global_base
            )
            while row < self.n:
                self._quantize_row(io, cta, state, quant, phase, global_base, row)
                row = row + row_stride
            activation_target = self._finish_quantization(io, cta, state, phase, quant)

            task = cutlass.Int32(cta.pid)
            while task < self.up_tasks:
                last = cutlass.Boolean(False)
                if cutlass.const_expr(track_last_up_task):
                    last = (task + task_stride >= self.up_tasks) & (
                        global_base + self.n >= io.x.shape[0]
                    )
                tile = UpTask(task, task == cutlass.Int32(cta.pid), last)
                self._project_up_tile(
                    io, shared, maps, state, tile, global_base, activation_target
                )
                reuse = cutlass.Boolean(True)
                if cutlass.const_expr(skip_last_up_join):
                    reuse = not last
                if reuse:
                    cute.arch.sync_threads()
                    if cta.warp == self.mma_warp:
                        _invalidate(shared.up_barriers, self.up_barrier_count)
                    cute.arch.sync_threads()
                task = task + task_stride

            more_rounds = global_base + self.n < io.x.shape[0]
            phase = self._finish_round(
                io, cta, shared, state, phase, activation_target, more_rounds
            )
            token_round = token_round + 1
            global_base = global_base + self.n

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
        return _token_major_reader(
            mma,
            base,
            original,
            tid,
            branch,
            part,
            (self.up_rows, self.up_channels, self.up_k),
            self.up_parts,
            self.up_hidden,
            self.reader_columns,
        )

    @cute.experimental.jit
    def _hidden_reader(
        self,
        mma,
        base,
        original,
        tid,
        token_slice: cutlass.Constexpr,
        part: cutlass.Constexpr,
    ):
        if cutlass.const_expr(part is None):
            fragment = mma.get_slice(0).make_fragment_C(
                mma.partition_shape_C((128, self.up_rows))
            )
            layout = fragment.layout
        else:
            layout = cute_ext.make_tmem_layout_acc(
                mma, self.up_mn, acc_stage=self.up_parts
            )
        accumulator = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), layout
        )
        if cutlass.const_expr(part is None):
            accumulator = accumulator[(None, None), 0, 0]
        else:
            accumulator = accumulator[((None, None), 0, 0, part)]
        group = tid // 128 + token_slice * self.reader_groups
        return _hidden_major_reader(
            accumulator,
            original,
            tid,
            group,
            (128, self.up_rows, self.up_k),
            self.hidden_reader_columns,
        )

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
        _prefetch_branch_residual(
            target,
            coords,
            x,
            hidden_tile,
            global_base,
            part,
            branch,
            self.up_rows,
            self.up_hidden,
            self.g,
            self.h,
        )

    @cute.experimental.jit
    def _mix_token_major_up(
        self,
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
        first_residual,
        row_scale_source: cutlass.Constexpr,
        weight_cache_offset: cutlass.Constexpr,
        vector_weight_scales: cutlass.Constexpr,
        prefetched_branches: cutlass.Constexpr,
        wait_once: cutlass.Constexpr,
        release_after_reads: cutlass.Constexpr,
    ):
        """Reduce the branches for either resident token-major Up layout."""
        for part in cutlass.range_constexpr(self.up_parts):
            if cutlass.const_expr(prefetched_branches and part == 0):
                template, coords, _ = first_residual
            else:
                _, _, template, coords = self._reader(mma, base, x, tid, 0, part)
            sums = cute.make_rmem_tensor(cute.size(template), cutlass.Float32)
            sums.fill(0.0)
            row_a_scale = cutlass.Float32(0.0)
            row_x_scale = cutlass.Float32(0.0)
            if cutlass.const_expr(row_scale_source != "shared_element"):
                row, _ = coords[0]
                local_row = part * self.up_rows + row
                if cutlass.const_expr(row_scale_source == "global"):
                    global_row = global_base + cutlass.Int64(local_row)
                    if global_row < x.shape[0]:
                        row_a_scale = _load_f32(
                            alpha.iterator + cutlass.Int64(local_row)
                        )
                        row_x_scale = _x_scale(xs, global_row)
                else:
                    row_a_scale = cached[local_row]
                    row_x_scale = cached[self.n + local_row]
            if cutlass.const_expr(prefetched_branches and part == 0):
                _, _, x_ring = first_residual
            else:
                x_ring = cute.make_rmem_tensor(
                    (cute.size(template), 2), cutlass.Float8E4M3FN
                )
                x_ring.fill(0.0)
                self._prefetch_branch_x(
                    x_ring[None, 0], coords, x, hidden_tile, global_base, part, 0
                )
            for branch in cutlass.range_constexpr(self.g):
                if cutlass.const_expr(
                    branch + 1 < self.g
                    and (part != 0 or branch + 1 >= prefetched_branches)
                ):
                    self._prefetch_branch_x(
                        x_ring[None, (branch + 1) % 2],
                        coords,
                        x,
                        hidden_tile,
                        global_base,
                        part,
                        branch + 1,
                    )
                decoded_x = _decode_residual(x_ring[None, branch % 2])
                if cutlass.const_expr(not wait_once or branch == 0):
                    cute.arch.mbarrier_wait(done + part, 0)
                accumulator, reader, values, _ = self._reader(
                    mma, base, x, tid, branch, part
                )
                cute_ext.partition_and_copy(reader, accumulator, values)
                cute.arch.fence_view_async_tmem_load()
                if cutlass.const_expr(
                    release_after_reads
                    and part + 1 == self.up_parts
                    and branch + 1 == self.g
                ):
                    tcgen05_fence("before_thread_sync")
                    cute.arch.mbarrier_arrive(read_done)
                channel_scales = None
                if cutlass.const_expr(vector_weight_scales):
                    _, col = coords[0]
                    scale_pointer = cute.make_ptr(
                        cutlass.Float32,
                        (
                            cached.iterator
                            + weight_cache_offset
                            + branch * self.up_hidden
                            + col
                        ).toint(),
                        cute.AddressSpace.smem,
                        assumed_align=16,
                    )
                    layout = cute.make_layout(cute.size(template))
                    channel_scales = cute.make_rmem_tensor(layout, cutlass.Float32)
                    cute.copy(
                        cute.make_copy_atom(
                            CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128
                        ),
                        cute.make_tensor(scale_pointer, layout),
                        channel_scales,
                    )
                for i in cutlass.range_constexpr(cute.size(values)):
                    row, col = coords[i]
                    local_row = part * self.up_rows + row
                    global_row = global_base + cutlass.Int64(local_row)
                    if global_row < x.shape[0]:
                        a_scale, x_scale = row_a_scale, row_x_scale
                        if cutlass.const_expr(row_scale_source == "shared_element"):
                            a_scale = cached[local_row]
                            x_scale = cached[self.n + local_row]
                        if cutlass.const_expr(vector_weight_scales):
                            weight_scale = channel_scales[i]
                        else:
                            weight_scale = cached[
                                weight_cache_offset + branch * self.up_hidden + col
                            ]
                        factor = a_scale * weight_scale
                        xv = decoded_x[i] * x_scale
                        sums[i] = sums[i] + xv * _up_sigmoid_tanh(values[i] * factor)
            _store_token_major_mix(
                out,
                sums,
                coords,
                hidden_tile,
                global_base,
                x.shape[0],
                part,
                self.up_rows,
                self.up_hidden,
                self.g,
                self.h,
            )
