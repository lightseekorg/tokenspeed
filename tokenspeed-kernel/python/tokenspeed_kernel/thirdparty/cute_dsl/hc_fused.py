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

"""Blackwell fused gated residual with distributed cluster reduction and TMA pipelines."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.utils as cute_utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cuda.bindings.driver import CUstream
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute import math as cute_math
from cutlass.cute.nvgpu import tcgen05
from cutlass.cutlass_dsl import T
from cutlass.experimental.primitives import Tcgen05InstrDesc, tcgen05_fence


@cute.jit
def _publish_activation_proxy():
    llvm.inline_asm(
        None,
        [],
        "fence.proxy.async.global;",
        "~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.jit
def _epilogue_barrier():
    nvvm.barrier_cta_sync(
        cutlass.Int32(1).ir_value(),
        thread_count=cutlass.Int32(128).ir_value(),
        aligned=False,
    )


@cute.jit
def _load_tail(ptr, valid, dtype: cutlass.Constexpr):
    """Load a valid half input as FP32, or return zero without accessing memory."""
    # Convert inside the asm instead of returning a partially defined 16-bit
    # register; masked-off loads must not depend on its previous contents.
    conversion = "cvt.f32.bf16" if dtype == cutlass.BFloat16 else "cvt.f32.f16"
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [ptr.toint().ir_value(), cutlass.Int32(valid).ir_value()],
            "{ .reg .pred p; .reg .b16 v; setp.ne.u32 p, $2, 0; mov.b16 v, 0; @p ld.global.b16 v, [$1]; "
            + conversion
            + " $0, v; }",
            "=f,l,r,~{memory}",
            has_side_effects=True,
            asm_dialect=0,
        )
    )


@cute.jit
def _store_tail(ptr, value, valid):
    llvm.inline_asm(
        None,
        [
            ptr.toint().ir_value(),
            llvm.bitcast(T.i16(), value.ir_value()),
            cutlass.Int32(valid).ir_value(),
        ],
        "{ .reg .pred p; setp.ne.u32 p, $2, 0; @p st.global.b16 [$0], $1; }",
        "l,h,r,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.jit
def _load_epoch_acquire(ptr):
    return cutlass.Int64(
        llvm.inline_asm(
            T.i64(),
            [ptr.toint().ir_value()],
            "ld.global.acquire.gpu.u64 $0, [$1];",
            "=l,l",
            has_side_effects=True,
            asm_dialect=0,
        )
    )


@cute.jit
def _store_epoch_release(ptr, value):
    llvm.inline_asm(
        None,
        [ptr.toint().ir_value(), value.ir_value()],
        "st.global.release.gpu.u64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.jit
def _mma_single_thread(acc, desc_a, desc_b, instruction, accumulate: cutlass.Constexpr):
    """Issue one native MMA; the caller must elect exactly one warp thread."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(acc.iterator.toint()).ir_value(),
            desc_a.ir_value(),
            desc_b.ir_value(),
            cutlass.Int32(instruction).ir_value(),
            cutlass.Int32(accumulate).ir_value(),
        ],
        "{ .reg .pred p; setp.ne.u32 p, $4, 0; "
        "tcgen05.mma.cta_group::1.kind::f16 [$0], $1, $2, $3, p; }",
        "r,l,l,r,r",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.jit
def _remote_partial(a, b, c, d, ptr, bar, peer):
    mapped = llvm.inline_asm(
        T.i32(),
        [ptr.toint().ir_value(), peer.ir_value()],
        "mapa.shared::cluster.u32 $0, $1, $2;",
        "=r,r,r",
        has_side_effects=False,
        asm_dialect=0,
    )
    mapped_bar = llvm.inline_asm(
        T.i32(),
        [bar.toint().ir_value(), peer.ir_value()],
        "mapa.shared::cluster.u32 $0, $1, $2;",
        "=r,r,r",
        has_side_effects=False,
        asm_dialect=0,
    )
    llvm.inline_asm(
        None,
        [mapped, a.ir_value(), b.ir_value(), c.ir_value(), d.ir_value(), mapped_bar],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];",
        "r,f,f,f,f,r,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.jit
def _cluster_arrive(bar):
    llvm.inline_asm(
        None,
        [bar.toint().ir_value()],
        "{ .reg .b32 dst; mapa.shared::cluster.u32 dst, $0, 0; "
        "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [dst]; }",
        "r,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.jit
def _cluster_wait(bar, phase):
    llvm.inline_asm(
        None,
        [bar.toint().ir_value(), cutlass.Int32(phase).ir_value()],
        "{ .reg .pred ready; wait_loop: "
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 ready, [$0], $1; "
        "@!ready bra wait_loop; }",
        "r,r,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


class FusedGatedResidualKernel:
    """Six warps: activation DMA/epilogue 0, epilogue 1-3, weights 4, MMA 5.

    Each CTA reduces M_proj / split_K columns in fixed split-rank order and
    writes post-scale/SiLU activation once. Independent weight and activation
    TMA producers share stage barriers. Up's three R320 stages are recycled
    between output tiles, only after their previous MMA reader has completed.
    Independent weights must be ready and immutable within the forward call.

    Rows come from the dynamic input layout; only the CTA tactic is compiled.
    ``single_tile`` promises at most one native token tile, so the same pipeline
    can elide its outer scheduling loop. The grid follows the runtime shape,
    capped by the compiled resident capacity.
    Round counts are scheduling buckets: keeping their loop bound static avoids
    excessive register lifetimes in this warp-specialized pipeline.
    ``full_tiles`` omits row predicates only for native-tile-aligned inputs.
    READY/CONSUMED halves use the workspace extent, never the active grid size.
    """

    def __init__(
        self,
        *,
        projection_tile: int,
        token_tile: int,
        projection_rows: int,
        split_k: int,
        projection_tiles: int,
        batch_tiles: int,
        workers: int,
        stages: int,
        final_tile: int,
        rounds: int,
        use_pdl: bool,
        scale: float,
        weights_independent: bool,
        single_tile: bool,
        full_tiles: bool,
    ):
        if split_k not in (1, 2, 4, 8, 16):
            raise ValueError("The fused HC tactic requires split-K in 1, 2, 4, 8, 16")
        if projection_tile not in (64, 128) or token_tile not in (8, 16, 32, 64):
            raise ValueError("Unsupported native MMA tile")
        if final_tile not in (16, 32):
            raise ValueError("Final hidden tile must be 16 or 32")
        if min(projection_tiles, batch_tiles, workers, stages, rounds) < 1:
            raise ValueError("Tile counts, workers and stages must be positive")
        self.single_tile = single_tile
        self.full_tiles = full_tiles
        self.p = projection_rows
        self.ps = 320
        self.split_k = split_k
        self.down_k = 128
        self.k_tiles = 10240 // split_k // self.down_k
        if stages > self.k_tiles:
            raise ValueError("Down stages must not exceed the per-rank K tile count")
        self.down_m = projection_tile
        self.n = token_tile
        self.slot_rows = max(16, token_tile)
        self.tmem_columns = max(32, token_tile)
        self.projection_tiles = projection_tiles
        self.batch_tiles = batch_tiles
        self.workers = workers
        self.rounds = rounds
        self.down_stages = stages
        self.final_tile = final_tile
        self.up_m = 4 * final_tile
        self.up_tiles = 2560 // final_tile
        self.clusters = (projection_rows + projection_tile * projection_tiles - 1) // (
            projection_tile * projection_tiles
        )
        self.group_ctas = self.clusters * split_k
        self.owner_cols = projection_tile // split_k
        self.pdl = use_pdl
        self.scale = scale
        self.weights_independent = weights_independent
        self.group = tcgen05.CtaGroup.ONE

    @cute.experimental.jit
    def __call__(self, x, w, u, activation, epochs, out, inject, stream: CUstream):
        up = cute.make_tensor(
            u.iterator,
            cute.make_layout(
                ((self.final_tile, 4, self.up_tiles), 320, 1),
                stride=((320, 2560 * 320, self.final_tile * 320), 1, 0),
            ),
        )
        # Inter-cluster handoffs require a resident cooperative grid. With one
        # projection cluster per worker, groups are independent and may run in
        # ordinary hardware-scheduled waves instead.
        workers = cute.ceil_div(x.shape[0], self.n * self.batch_tiles)
        if cutlass.const_expr(self.clusters > 1):
            workers = min(workers, self.workers)
        self.kernel(x, w, up, activation, epochs, out, inject).launch(
            grid=(self.clusters, self.split_k, workers),
            cluster=(1, self.split_k, 1),
            block=(192, 1, 1),
            stream=stream,
            use_pdl=self.pdl,
            cooperative=self.clusters > 1,
        )

    @cute.experimental.jit
    def _accumulator_tile(
        self,
        smem_ptr,
        acc_layout,
        tile_m,
        activation,
        epi_tid,
    ):
        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, smem_ptr), acc_layout
        )[((None, None), 0, 0, 0)]
        tile = (tile_m, self.n)
        t2r = tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (tile_m, self.n, 128),
                cute_utils.LayoutEnum.COL_MAJOR,
                cutlass.Float32,
                cutlass.Float32,
                tile,
                False,
            ),
            acc_view,
        )
        dummy = cute.make_tensor(
            activation.iterator, cute.make_layout(tile, stride=(1, self.ps))
        )
        rlayout = cute_ext.make_t2r_rmem_layout(
            t2r, cute.flat_divide(dummy, tile), epi_tid
        )
        values = cute_ext.allocate(
            cutlass.Float32, cute.AddressSpace.rmem, rlayout, alignment=32
        )
        thr = t2r.get_slice(epi_tid)
        coords = thr.partition_D(cute.make_identity_tensor(tile))
        return acc_view, thr, values, coords

    @cute.experimental.jit
    def _prefetch_x(
        self, x_regs, coords, x_values, pid, token, full_tile: cutlass.Constexpr
    ):
        # Compare native row offsets with the tile's remaining rows. Widen the
        # tile address before adding them to preserve linear 64-bit addressing.
        for i in cutlass.range_constexpr(cute.size(x_regs)):
            col, row = coords[i]
            global_row = row + token * self.n
            global_col = (
                (col // self.final_tile) * 2560
                + pid * self.final_tile
                + col % self.final_tile
            )
            if cutlass.const_expr(full_tile):
                value = x_values[global_row, global_col]
            else:
                value = _load_tail(
                    x_values.iterator
                    + cutlass.Int64(token) * (self.n * 10240)
                    + row * 10240
                    + global_col,
                    row < cutlass.Int32(x_values.shape[0]) - token * self.n,
                    x_values.element_type,
                )
            x_regs[i] = value.to(cutlass.Float32)

    @cute.experimental.jit
    def _read_up_with_prefetched_x(
        self,
        smem_ptr,
        acc_layout,
        activation,
        epi_tid,
        ready_bar,
        x_values,
        pid,
        token,
        phase,
    ):
        acc_view, thr, values, coords = self._accumulator_tile(
            smem_ptr, acc_layout, self.up_m, activation, epi_tid
        )
        x_regs = cute_ext.allocate(
            cutlass.Float32, cute.AddressSpace.rmem, values.layout, alignment=32
        )
        # Compute handed PDL readiness to the group before Down. Prefetch the
        # final gate operand while Up MMA proceeds on its separate SMEM inputs.
        self._prefetch_x(x_regs, coords, x_values, pid, token, self.full_tiles)
        cute.arch.mbarrier_wait(ready_bar, phase)
        cute_ext.partition_and_copy(thr, acc_view, values)
        cute.arch.fence_view_async_tmem_load()
        return values, coords, x_regs

    @cute.experimental.jit
    def _load_up(self, up, su, bar, map_u, pid):
        g = cute.local_tile(up, (self.up_m, 128), (pid, None, 0))
        for stage in cutlass.range_constexpr(3):
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar + stage, self.up_m * 128 * 2
                )
            cute_ext.tma_load(
                g[None, None, stage],
                su[None, None, None, stage],
                (bar + stage).value,
                cta_v_map=map_u,
                tma_operation_type=cute_ext.OperationTypeEnum.SM90_TMA_LOAD,
                update_expect_tx=False,
            )

    @cute.experimental.jit
    def _initialize_barriers(self, barriers, warp):
        """Initialize shared barriers once for all projection and token phases."""
        (
            down_full,
            down_empty,
            down_done,
            epi_done,
            up_full,
            control_ready,
            projection_ready,
            up_done,
            reduce_ready,
            cluster_done,
            up_empty,
            consumed,
        ) = barriers
        if warp == 4:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.down_stages):
                    cute.arch.mbarrier_init(down_full + stage, 2)
                if cutlass.const_expr(self.k_tiles > self.down_stages):
                    for stage in cutlass.range_constexpr(self.down_stages):
                        cute.arch.mbarrier_init(down_empty + stage, 1)
                cute.arch.mbarrier_init(down_done, 1)
                cute.arch.mbarrier_init(epi_done, 128)
                for stage in cutlass.range_constexpr(3):
                    cute.arch.mbarrier_init(up_full + stage, 2)
                cute.arch.mbarrier_init(control_ready, 1)
                cute.arch.mbarrier_init(projection_ready, 1)
                cute.arch.mbarrier_init(up_done, 1)
                cute.arch.mbarrier_init(reduce_ready, 1)
                cute.arch.mbarrier_init(cluster_done, self.split_k)
                if cutlass.const_expr(self.group_ctas < self.up_tiles):
                    cute.arch.mbarrier_init(up_empty, 1)
                if cutlass.const_expr(self.rounds * self.batch_tiles > 1):
                    cute.arch.mbarrier_init(consumed, self.split_k)
        # Remote DSM destinations must be initialized before any peer store.
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()

    @cute.experimental.kernel
    def kernel(self, x, w, up, activation, epochs, out, inject):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        cluster = cute.arch.block_idx()[0]
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        pid = cluster * self.split_k + split_rank
        worker = 0
        if cutlass.const_expr(self.clusters == 1 or self.workers > 1):
            worker = cute.arch.block_idx()[2]
        dtype = x.element_type
        down_mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            self.group,
            (self.down_m, self.n),
        )
        up_mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            self.group,
            (self.up_m, self.n),
        )
        down_tiler = (self.down_m, self.n, self.down_k)
        up_tiler = (self.up_m, self.n, 128)
        down_layout = sm100_utils.make_smem_layout_a(
            down_mma, down_tiler, dtype, self.down_stages
        )
        up_layout = sm100_utils.make_smem_layout_a(up_mma, up_tiler, dtype, 3)
        if cutlass.const_expr(self.n <= 16):
            sd = cute_ext.allocate(
                dtype, cute.AddressSpace.smem, down_layout, alignment=1024
            )
        else:
            weights_storage = cute_ext.allocate(
                dtype,
                cute.AddressSpace.smem,
                cute.make_layout(max(cute.cosize(down_layout), cute.cosize(up_layout))),
                alignment=1024,
            )
            sd = cute.make_tensor(
                cute.recast_ptr(
                    weights_storage.iterator,
                    swizzle_=cute.get_swizzle_portion(down_layout),
                    dtype=dtype,
                ),
                cute.get_nonswizzle_portion(down_layout),
            )
        sx = cute_ext.allocate(
            dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_b(
                down_mma, down_tiler, dtype, self.down_stages
            ),
            alignment=1024,
        )
        if cutlass.const_expr(self.n <= 16):
            su = cute_ext.allocate(
                dtype, cute.AddressSpace.smem, up_layout, alignment=1024
            )
        else:
            su = cute.make_tensor(
                cute.recast_ptr(
                    weights_storage.iterator,
                    swizzle_=cute.get_swizzle_portion(up_layout),
                    dtype=dtype,
                ),
                cute.get_nonswizzle_portion(up_layout),
            )
        sa = cute_ext.allocate(
            dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_b(up_mma, up_tiler, dtype, 3),
            alignment=1024,
        )
        if cutlass.const_expr(self.n <= 16):
            gate = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout((self.up_m, self.n)),
                alignment=128,
            )
        # Each rank receives [S source ranks, N tokens, M_proj / S columns].
        mailbox = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout(
                (self.down_m if self.n <= 16 else max(self.down_m, self.up_m)) * self.n
            ),
            alignment=128,
        )
        partial_tile = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout((self.down_m + 4) * self.n),
            alignment=128,
        )
        if cutlass.const_expr(self.n > 16):
            # Down's ordered reduction is fully acknowledged before Up gate
            # materialization; the two consumers have disjoint lifetimes.
            gate = cute.make_tensor(
                mailbox.iterator, cute.make_layout((self.up_m, self.n))
            )
        bars = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(
                2 * self.down_stages
                + 10
                + int(self.group_ctas < self.up_tiles)
                + int(self.rounds * self.batch_tiles > 1)
            ),
            alignment=8,
        ).iterator
        down_full = bars
        down_empty = down_full + self.down_stages
        down_done = down_empty + self.down_stages
        epi_done = down_done + 1
        up_full = epi_done + 1
        control_ready = up_full + 3
        projection_ready = control_ready + 1
        up_done = projection_ready + 1
        reduce_ready = up_done + 1
        cluster_done = reduce_ready + 1
        up_empty = cluster_done + 1
        consumed = up_empty + int(self.group_ctas < self.up_tiles)
        barriers = (
            down_full,
            down_empty,
            down_done,
            epi_done,
            up_full,
            control_ready,
            projection_ready,
            up_done,
            reduce_ready,
            cluster_done,
            up_empty,
            consumed,
        )
        tmem_base = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        ).iterator
        if warp == 5 and cutlass.const_expr(not self.single_tile):
            cute.arch.alloc_tmem(self.tmem_columns, tmem_base, is_two_cta=False)
            cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
        self._initialize_barriers(barriers, warp)
        token_tiles = cutlass.Int32(cute.ceil_div(x.shape[0], self.n))
        for job_round in cutlass.range(self.rounds, unroll_full=self.rounds == 1):
            for micro in cutlass.range_constexpr(self.batch_tiles):
                tile_iteration = job_round * self.batch_tiles + micro
                token = (job_round * self.workers + worker) * self.batch_tiles + micro
                if cutlass.const_expr(self.single_tile):
                    token = 0
                if (
                    cutlass.const_expr(self.rounds == 1 and self.batch_tiles == 1)
                    or token < token_tiles
                ):
                    self._process_tile(
                        x,
                        w,
                        up,
                        activation,
                        epochs,
                        out,
                        inject,
                        tid,
                        warp,
                        cluster,
                        split_rank,
                        pid,
                        worker,
                        token,
                        tile_iteration,
                        down_mma,
                        up_mma,
                        sd,
                        sx,
                        su,
                        sa,
                        gate,
                        mailbox,
                        partial_tile,
                        barriers,
                        tmem_base,
                    )
        if warp == 5 and cutlass.const_expr(not self.single_tile):
            tcgen05_fence("after_thread_sync")
            cute.arch.dealloc_tmem(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, tmem_base),
                self.tmem_columns,
                is_two_cta=False,
            )

    @cute.experimental.jit
    def _process_tile(
        self,
        x,
        w,
        up,
        activation,
        epochs,
        out,
        inject,
        tid,
        warp,
        cluster,
        split_rank,
        pid,
        worker,
        token,
        tile_iteration,
        down_mma,
        up_mma,
        sd,
        sx,
        su,
        sa,
        gate,
        mailbox,
        partial_tile,
        barriers,
        tmem_base,
    ):
        dtype = x.element_type
        down_tiler = (self.down_m, self.n, self.down_k)
        up_tiler = (self.up_m, self.n, 128)
        (
            down_full,
            down_empty,
            down_done,
            epi_done,
            up_full,
            control_ready,
            projection_ready,
            up_done,
            reduce_ready,
            cluster_done,
            up_empty,
            consumed,
        ) = barriers
        down_acc_layout = cute_ext.make_tmem_layout_acc(
            down_mma, (self.down_m, self.n), acc_stage=1
        )
        up_acc_layout = cute_ext.make_tmem_layout_acc(
            up_mma, (self.up_m, self.n), acc_stage=1
        )
        epoch_half = self.workers * self.clusters
        if cutlass.const_expr(self.clusters == 1):
            epoch_half = epochs.shape[0] // 2
        generation = cutlass.Int64(0)
        up_steps = (self.up_tiles + self.group_ctas - 1) // self.group_ctas
        cta_up_steps = (self.up_tiles + self.group_ctas - 1 - pid) // self.group_ctas
        tile_phase = tile_iteration % 2
        for projection_step in cutlass.range_constexpr(self.projection_tiles):
            projection = cluster + projection_step * self.clusters
            projection_iteration = (
                tile_iteration * self.projection_tiles + projection_step
            )
            projection_phase = projection_iteration % 2
            epi_phase = (
                tile_iteration * (self.projection_tiles + cta_up_steps)
                + projection_step
            ) % 2
            if warp == 4:
                # Both weight projections use this one producer. No activation
                # dependency precedes their initial loads in independent mode.
                if cutlass.const_expr(self.pdl and not self.weights_independent):
                    cute.arch.griddepcontrol_wait()
                map_d = cute_ext.get_cta_v_map_ab(w, down_tiler, down_mma, "A")
                map_u = cute_ext.get_cta_v_map_ab(up, up_tiler, up_mma, "A")
                gd = cute.local_tile(
                    w, (self.down_m, self.down_k), (projection, None, 0)
                )
                for stage in cutlass.range_constexpr(self.down_stages):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            down_full + stage, self.down_m * self.down_k * 2
                        )
                    cute_ext.tma_load(
                        gd[None, None, split_rank * self.k_tiles + stage],
                        sd[None, None, None, stage],
                        (down_full + stage).value,
                        cta_v_map=map_d,
                        tma_operation_type=cute_ext.OperationTypeEnum.SM90_TMA_LOAD,
                        update_expect_tx=False,
                    )
                if cutlass.const_expr(
                    self.n <= 16 and projection_step == self.projection_tiles - 1
                ):
                    if pid < self.up_tiles:
                        self._load_up(up, su, up_full, map_u, pid)
                # Any recycled Down slot waits only for its own empty phase.
                # Up weight requests have already been issued before that wait.
                for tile in cutlass.range_constexpr(self.down_stages, self.k_tiles):
                    stage = tile % self.down_stages
                    stage_uses = (
                        self.k_tiles + self.down_stages - 1 - stage
                    ) // self.down_stages
                    down_iteration = (
                        projection_iteration * stage_uses + tile // self.down_stages
                    )
                    cute.arch.mbarrier_wait(
                        down_empty + stage, (down_iteration - 1) % 2
                    )
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            down_full + stage, self.down_m * self.down_k * 2
                        )
                    cute_ext.tma_load(
                        gd[None, None, split_rank * self.k_tiles + tile],
                        sd[None, None, None, stage],
                        (down_full + stage).value,
                        cta_v_map=map_d,
                        tma_operation_type=cute_ext.OperationTypeEnum.SM90_TMA_LOAD,
                        update_expect_tx=False,
                    )
                if cutlass.const_expr(projection_step == self.projection_tiles - 1):
                    if cutlass.const_expr(self.n > 16):
                        # The shared weight allocation changes ownership only
                        # after the final Down MMA read has completed.
                        cute.arch.mbarrier_wait(down_done, projection_phase)
                        if pid < self.up_tiles:
                            self._load_up(up, su, up_full, map_u, pid)
                    for up_step in cutlass.range_constexpr(1, up_steps):
                        up_id = pid + up_step * self.group_ctas
                        if up_id < self.up_tiles:
                            up_iteration = tile_iteration * cta_up_steps + up_step
                            cute.arch.mbarrier_wait(up_empty, (up_iteration - 1) % 2)
                            map_u = cute_ext.get_cta_v_map_ab(up, up_tiler, up_mma, "A")
                            self._load_up(up, su, up_full, map_u, up_id)
            elif warp == 5:
                if cutlass.const_expr(self.single_tile and projection_step == 0):
                    cute.arch.alloc_tmem(self.tmem_columns, tmem_base, is_two_cta=False)
                    cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
                    if cutlass.const_expr(self.pdl):
                        cute.arch.griddepcontrol_wait()
                        cute.arch.griddepcontrol_launch_dependents()
                if cutlass.const_expr(projection_step == 0):
                    generation = (
                        _load_epoch_acquire(
                            epochs.iterator + worker * self.clusters + cluster
                        )
                        + 1
                    )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(control_ready)
                down_acc = cute.make_tensor(
                    cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, tmem_base),
                    down_acc_layout,
                )
                down_instruction = Tcgen05InstrDesc.build(
                    sparse_id2=0,
                    sparse_flag=0,
                    saturate=0,
                    sparse_format=0,
                    c_dtype=cutlass.Float32,
                    a_dtype=dtype,
                    b_dtype=dtype,
                    a_negate=0,
                    b_negate=0,
                    a_major=0,
                    b_major=0,
                    n_dim=self.n,
                    m_dim=self.down_m,
                    k_dim=0,
                    max_shift=0,
                )
                down_a = sd[None, None, 0, 0]
                down_b = sx[None, None, 0, 0]
                desc_a = tcgen05.smem_descriptor_to_int(
                    tcgen05.make_umma_smem_desc(
                        down_a.iterator, down_a.layout, "k", next_src=None
                    )
                )
                desc_b = tcgen05.smem_descriptor_to_int(
                    tcgen05.make_umma_smem_desc(
                        down_b.iterator, down_b.layout, "k", next_src=None
                    )
                )
                for tile in cutlass.range_constexpr(self.k_tiles):
                    stage = tile % self.down_stages
                    stage_uses = (
                        self.k_tiles + self.down_stages - 1 - stage
                    ) // self.down_stages
                    phase = (
                        projection_iteration * stage_uses + tile // self.down_stages
                    ) % 2
                    cute.arch.mbarrier_wait(down_full + stage, phase)
                    with cute.arch.elect_one():
                        for kb in cutlass.range_constexpr(self.down_k // 16):
                            idx = stage * (self.down_k // 16) + kb
                            _mma_single_thread(
                                down_acc[None, None, None, 0],
                                desc_a + (idx // 4) * self.down_m * 8 + (idx % 4) * 2,
                                desc_b + (idx // 4) * self.n * 8 + (idx % 4) * 2,
                                down_instruction,
                                tile != 0 or kb != 0,
                            )
                        if cutlass.const_expr(self.k_tiles > self.down_stages):
                            tcgen05.commit(down_empty + stage, None, self.group)
                with cute.arch.elect_one():
                    tcgen05.commit(down_done, None, self.group)
                cute.arch.mbarrier_wait(epi_done, epi_phase)
                if cutlass.const_expr(projection_step == self.projection_tiles - 1):
                    with cute.arch.elect_one():
                        _cluster_arrive(cluster_done)
                        if split_rank == 0:
                            _cluster_wait(cluster_done, tile_phase)
                            _store_epoch_release(
                                epochs.iterator + worker * self.clusters + cluster,
                                generation,
                            )
                    lane = tid % 32
                    ready = cutlass.Boolean(False)
                    while not ready:
                        local_ready = cutlass.Boolean(True)
                        if lane < self.clusters:
                            local_ready = (
                                _load_epoch_acquire(
                                    epochs.iterator + worker * self.clusters + lane
                                )
                                >= generation
                            )
                        ready = cute.arch.vote_all_sync(local_ready, mask=0xFFFFFFFF)
                    cute.arch.sync_warp(mask=0xFFFFFFFF)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(projection_ready)
                    for up_step in cutlass.range_constexpr(up_steps):
                        up_id = pid + up_step * self.group_ctas
                        if up_id < self.up_tiles:
                            up_phase = (tile_iteration * cta_up_steps + up_step) % 2
                            up_acc = cute.make_tensor(
                                cute.arch.retrieve_tmem_ptr(
                                    cutlass.Float32, 16, tmem_base
                                ),
                                up_acc_layout,
                            )[None, None, None, 0]
                            up_instruction = Tcgen05InstrDesc.build(
                                sparse_id2=0,
                                sparse_flag=0,
                                saturate=0,
                                sparse_format=0,
                                c_dtype=cutlass.Float32,
                                a_dtype=dtype,
                                b_dtype=dtype,
                                a_negate=0,
                                b_negate=0,
                                a_major=0,
                                b_major=0,
                                n_dim=self.n,
                                m_dim=self.up_m,
                                k_dim=0,
                                max_shift=0,
                            )
                            # K64 group bases are 1024-byte aligned, preserving
                            # swizzle/base/layout fields. The 14-bit start field
                            # uses 16-byte units, with no carry here.
                            up_a = su[None, None, 0, 0]
                            up_b = sa[None, None, 0, 0]
                            desc_a = tcgen05.smem_descriptor_to_int(
                                tcgen05.make_umma_smem_desc(
                                    up_a.iterator,
                                    up_a.layout,
                                    "k",
                                    next_src=None,
                                )
                            )
                            desc_b = tcgen05.smem_descriptor_to_int(
                                tcgen05.make_umma_smem_desc(
                                    up_b.iterator,
                                    up_b.layout,
                                    "k",
                                    next_src=None,
                                )
                            )
                            for stage in cutlass.range_constexpr(3):
                                cute.arch.mbarrier_wait(up_full + stage, up_phase)
                                with cute.arch.elect_one():
                                    for block in cutlass.range_constexpr(
                                        8 if stage < 2 else 4
                                    ):
                                        kb = stage * 8 + block
                                        _mma_single_thread(
                                            up_acc,
                                            desc_a
                                            + (kb // 4) * self.up_m * 8
                                            + (kb % 4) * 2,
                                            desc_b
                                            + (kb // 4) * self.n * 8
                                            + (kb % 4) * 2,
                                            up_instruction,
                                            kb != 0,
                                        )
                            with cute.arch.elect_one():
                                tcgen05.commit(up_done, None, self.group)
                                if cutlass.const_expr(self.group_ctas < self.up_tiles):
                                    tcgen05.commit(up_empty, None, self.group)
                            # Each phase acknowledges the preceding TMEM read
                            # before its storage is reused by the next step.
                            cute.arch.mbarrier_wait(
                                epi_done, (epi_phase + up_step + 1) % 2
                            )
                    if cutlass.const_expr(self.single_tile):
                        tcgen05_fence("after_thread_sync")
                        cute.arch.dealloc_tmem(
                            cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, tmem_base),
                            self.tmem_columns,
                            is_two_cta=False,
                        )
            elif warp < 4:
                epi_tid = tid
                cute.arch.mbarrier_wait(control_ready, projection_phase)
                if warp == 0:
                    map_x = cute_ext.get_cta_v_map_ab(x, down_tiler, down_mma, "B")
                    gx = cute.local_tile(x, (self.n, self.down_k), (token, None, 0))
                    for tile in cutlass.range_constexpr(self.k_tiles):
                        stage = tile % self.down_stages
                        if cutlass.const_expr(tile >= self.down_stages):
                            stage_uses = (
                                self.k_tiles + self.down_stages - 1 - stage
                            ) // self.down_stages
                            down_iteration = (
                                projection_iteration * stage_uses
                                + tile // self.down_stages
                            )
                            cute.arch.mbarrier_wait(
                                down_empty + stage, (down_iteration - 1) % 2
                            )
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                down_full + stage, self.n * self.down_k * 2
                            )
                        cute_ext.tma_load(
                            gx[None, None, split_rank * self.k_tiles + tile],
                            sx[None, None, None, stage],
                            (down_full + stage).value,
                            cta_v_map=map_x,
                            tma_operation_type=cute_ext.OperationTypeEnum.SM90_TMA_LOAD,
                            update_expect_tx=False,
                        )
                cute.arch.mbarrier_wait(down_done, projection_phase)
                down_acc, down_thr, down_values, down_coords = self._accumulator_tile(
                    tmem_base, down_acc_layout, self.down_m, activation, epi_tid
                )
                cute_ext.partition_and_copy(down_thr, down_acc, down_values)
                cute.arch.fence_view_async_tmem_load()
                # Every source scatters aligned four-column packets to S owners.
                if epi_tid == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        reduce_ready, self.down_m * self.n * 4
                    )
                for i in cutlass.range_constexpr(cute.size(down_values)):
                    col, row = down_coords[i]
                    partial_tile[row * (self.down_m + 4) + col] = down_values[i]
                _epilogue_barrier()
                for packet in cutlass.range_constexpr(
                    self.down_m * self.n // (4 * 128)
                ):
                    item = epi_tid + packet * 128
                    owner = item // (self.n * (self.owner_cols // 4))
                    row = item % self.n
                    pack = item // self.n % (self.owner_cols // 4)
                    offset = (
                        row * (self.down_m + 4) + owner * self.owner_cols + pack * 4
                    )
                    _remote_partial(
                        partial_tile[offset],
                        partial_tile[offset + 1],
                        partial_tile[offset + 2],
                        partial_tile[offset + 3],
                        mailbox.iterator
                        + split_rank * self.n * self.owner_cols
                        + row * self.owner_cols
                        + pack * 4,
                        reduce_ready,
                        owner,
                    )
                cute.arch.mbarrier_wait(reduce_ready, projection_phase)
                for part in cutlass.range_constexpr(
                    (self.n * self.owner_cols + 127) // 128
                ):
                    item = epi_tid + part * 128
                    row = item // self.owner_cols
                    col = (
                        projection * self.down_m
                        + split_rank * self.owner_cols
                        + item % self.owner_cols
                    )
                    if item < self.n * self.owner_cols:
                        value = mailbox[item]
                        for peer in cutlass.range_constexpr(1, self.split_k):
                            value = (
                                value + mailbox[peer * self.n * self.owner_cols + item]
                            )
                        value = value * self.scale
                        if col < 320:
                            value = value * cute_math.rcp(
                                1.0 + cute.exp(-value, fastmath=True),
                                fastmath=False,
                                approx=True,
                                rounding=None,
                                ftz=True,
                            )
                            activation[worker * self.slot_rows + row, col] = value.to(
                                dtype
                            )
                            _publish_activation_proxy()
                        elif col < self.p:
                            if cutlass.const_expr(self.full_tiles):
                                inject[token * self.n + row, col - 320] = value.to(
                                    dtype
                                )
                            else:
                                _store_tail(
                                    inject.iterator
                                    + (cutlass.Int64(token) * self.n + row) * 4
                                    + col
                                    - 320,
                                    value.to(dtype),
                                    row < cutlass.Int32(x.shape[0]) - token * self.n,
                                )
                # The complete 128-thread acknowledgement orders projection stores
                # before compute publishes this CTA into the cluster completion barrier.
                tcgen05_fence("before_thread_sync")
                cute.arch.mbarrier_arrive(epi_done)
                if cutlass.const_expr(projection_step == self.projection_tiles - 1):
                    epi_tid = tid
                    cute.arch.mbarrier_wait(projection_ready, tile_phase)
                    for up_step in cutlass.range_constexpr(up_steps):
                        up_id = pid + up_step * self.group_ctas
                        if up_id < self.up_tiles:
                            up_phase = (tile_iteration * cta_up_steps + up_step) % 2
                            if cutlass.const_expr(up_step > 0):
                                _epilogue_barrier()
                            if warp == 0:
                                if cutlass.const_expr(up_step > 0):
                                    cute.arch.mbarrier_wait(
                                        up_empty,
                                        (tile_iteration * cta_up_steps + up_step - 1)
                                        % 2,
                                    )
                                workspace_rows = self.workers * self.slot_rows
                                if cutlass.const_expr(self.clusters == 1):
                                    workspace_rows = activation.shape[0]
                                active = cute.make_tensor(
                                    activation.iterator,
                                    cute.make_layout(
                                        (workspace_rows, 320, 1),
                                        stride=(320, 1, 0),
                                    ),
                                )
                                map_a = cute_ext.get_cta_v_map_ab(
                                    active, up_tiler, up_mma, "B"
                                )
                                ga = cute.local_tile(
                                    active,
                                    (self.n, 128),
                                    (worker * (self.slot_rows // self.n), None, 0),
                                )
                                for stage in cutlass.range_constexpr(3):
                                    with cute.arch.elect_one():
                                        cute.arch.mbarrier_arrive_and_expect_tx(
                                            up_full + stage, self.n * 128 * 2
                                        )
                                    cute_ext.tma_load(
                                        ga[None, None, stage],
                                        sa[None, None, None, stage],
                                        (up_full + stage).value,
                                        cta_v_map=map_a,
                                        tma_operation_type=cute_ext.OperationTypeEnum.SM90_TMA_LOAD,
                                        update_expect_tx=False,
                                    )
                            x_values = cute.make_tensor(
                                x.iterator,
                                cute.make_layout(
                                    (x.shape[0], 10240), stride=(10240, 1)
                                ),
                            )
                            gates, up_coords, x_regs = self._read_up_with_prefetched_x(
                                tmem_base,
                                up_acc_layout,
                                activation,
                                epi_tid,
                                up_done,
                                x_values,
                                up_id,
                                token,
                                up_phase,
                            )
                            # All 128 readers complete tcgen05.wait::ld before
                            # releasing this phase. Gate/output no longer use TMEM.
                            tcgen05_fence("before_thread_sync")
                            cute.arch.mbarrier_arrive(epi_done)
                            for i in cutlass.range_constexpr(cute.size(gates)):
                                gate[up_coords[i][0], up_coords[i][1]] = x_regs[
                                    i
                                ] * cute.arch.rcp_approx(
                                    1.0 + cute.exp(-gates[i], fastmath=True)
                                )
                            _epilogue_barrier()
                            for part in cutlass.range_constexpr(
                                self.final_tile * self.n // 128
                            ):
                                elem = epi_tid + part * 128
                                j = elem % self.final_tile
                                row = elem // self.final_tile
                                value = cutlass.Float32(0)
                                for branch in cutlass.range_constexpr(4):
                                    value = (
                                        value + gate[branch * self.final_tile + j, row]
                                    )
                                if cutlass.const_expr(self.full_tiles):
                                    out[
                                        row + token * self.n,
                                        up_id * self.final_tile + j,
                                    ] = (value * 0.25).to(dtype)
                                else:
                                    _store_tail(
                                        out.iterator
                                        + (row + cutlass.Int64(token) * self.n) * 2560
                                        + up_id * self.final_tile
                                        + j,
                                        (value * 0.25).to(dtype),
                                        row
                                        < cutlass.Int32(x.shape[0]) - token * self.n,
                                    )

            if cutlass.const_expr(projection_step + 1 < self.projection_tiles):
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()
        if cutlass.const_expr(self.rounds * self.batch_tiles > 1):
            # Every local epilogue is finished before a worker can recycle its
            # activation slot. Publish CONSUMED separately from READY.
            cute.arch.sync_threads()
            if warp == 5:
                with cute.arch.elect_one():
                    _cluster_arrive(consumed)
                    if split_rank == 0:
                        _cluster_wait(consumed, tile_phase)
                        _store_epoch_release(
                            epochs.iterator
                            + epoch_half
                            + worker * self.clusters
                            + cluster,
                            generation,
                        )
                lane = tid % 32
                ready = cutlass.Boolean(False)
                while not ready:
                    local_ready = cutlass.Boolean(True)
                    if lane < self.clusters:
                        local_ready = (
                            _load_epoch_acquire(
                                epochs.iterator
                                + epoch_half
                                + worker * self.clusters
                                + lane
                            )
                            >= generation
                        )
                    ready = cute.arch.vote_all_sync(local_ready, mask=0xFFFFFFFF)
            cute.arch.sync_threads()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
