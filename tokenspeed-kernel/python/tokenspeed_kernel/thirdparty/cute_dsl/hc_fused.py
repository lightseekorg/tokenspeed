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
def _cluster_wait(bar):
    llvm.inline_asm(
        None,
        [bar.toint().ir_value()],
        "{ .reg .pred ready; wait_loop: "
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 ready, [$0], 0; "
        "@!ready bra wait_loop; }",
        "r,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


class FusedGatedResidualKernel:
    """Six warps: activation DMA/epilogue 0, epilogue 1-3, weights 4, MMA 5.

    Each CTA reduces four projection columns in fixed split-rank order and
    writes post-scale/SiLU activation once. Both GEMMs use stage full/empty
    barriers shared by independent weight and activation TMA producers.
    Independent weights must be ready and immutable within the forward call.
    """

    def __init__(
        self,
        rows: int,
        projection_rows: int,
        split_k: int,
        use_pdl: bool,
        scale: float,
        weights_independent: bool,
    ):
        if split_k != 16:
            raise ValueError("The fused HC tactic currently requires split-K=16")
        self.rows = rows
        self.n = 8 if rows <= 8 else 16
        self.p = projection_rows
        self.ps = 320
        self.split_k = split_k
        self.down_m = 64
        self.down_k = 128
        self.k_tiles = 10240 // split_k // self.down_k
        self.down_stages = min(self.k_tiles, 5)
        self.clusters = (projection_rows + self.down_m - 1) // self.down_m
        self.pdl = use_pdl
        self.scale = scale
        self.weights_independent = weights_independent
        self.group = tcgen05.CtaGroup.ONE

    @cute.experimental.jit
    def __call__(self, x, w, u, activation, epochs, out, inject, stream: CUstream):
        up = cute.make_tensor(
            u.iterator,
            cute.make_layout(
                ((32, 4, 80), 320, 1), stride=((320, 2560 * 320, 32 * 320), 1, 0)
            ),
        )
        # The inter-cluster handoff requires the full grid to be resident.
        # A cooperative launch enforces this also with concurrent streams.
        self.kernel(x, w, up, activation, epochs, out, inject).launch(
            grid=(self.clusters, self.split_k, 1),
            cluster=(1, self.split_k, 1),
            block=(192, 1, 1),
            smem=227 * 1024,
            stream=stream,
            use_pdl=self.pdl,
            cooperative=True,
        )

    @cute.experimental.jit
    def _read_acc(
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
        cute_ext.partition_and_copy(thr, acc_view, values)
        cute.arch.fence_view_async_tmem_load()
        coords = thr.partition_D(cute.make_identity_tensor(tile))
        return values, coords

    @cute.experimental.jit
    def _read_up_with_prefetched_x(
        self, smem_ptr, acc_layout, activation, epi_tid, ready_bar, x_values, pid
    ):
        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, smem_ptr), acc_layout
        )[((None, None), 0, 0, 0)]
        tile = (128, self.n)
        t2r = tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (128, self.n, 128),
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
        x_regs = cute_ext.allocate(
            cutlass.Float32, cute.AddressSpace.rmem, rlayout, alignment=32
        )
        thr = t2r.get_slice(epi_tid)
        coords = thr.partition_D(cute.make_identity_tensor(tile))
        # Compute handed PDL readiness to the group before Down. Prefetch the
        # final gate operand while Up MMA proceeds on its separate SMEM inputs.
        for i in cutlass.range_constexpr(cute.size(x_regs)):
            col, row = coords[i]
            value = cutlass.Float32(0)
            if row < self.rows:
                value = x_values[row, (col // 32) * 2560 + pid * 32 + col % 32].to(
                    cutlass.Float32
                )
            x_regs[i] = value
        cute.arch.mbarrier_wait(ready_bar, 0)
        cute_ext.partition_and_copy(thr, acc_view, values)
        cute.arch.fence_view_async_tmem_load()
        return values, coords, x_regs

    @cute.experimental.jit
    def _load_up(self, up, su, bar, empty, map_u, pid):
        g = cute.local_tile(up, (128, 128), (pid, None, 0))
        for stage in cutlass.range_constexpr(3):
            if cutlass.const_expr(self.ps > 3 * 128):
                cute.arch.mbarrier_wait(empty + stage, 1)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(bar + stage, 128 * 128 * 2)
            cute_ext.tma_load(
                g[None, None, stage],
                su[None, None, None, stage],
                (bar + stage).value,
                cta_v_map=map_u,
                tma_operation_type=cute_ext.OperationTypeEnum.SM90_TMA_LOAD,
                update_expect_tx=False,
            )

    @cute.experimental.kernel
    def kernel(self, x, w, up, activation, epochs, out, inject):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        cluster = cute.arch.block_idx()[0]
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        pid = cluster * self.split_k + split_rank
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
            (128, self.n),
        )
        down_tiler = (self.down_m, self.n, self.down_k)
        up_tiler = (128, self.n, 128)
        sd = cute_ext.allocate(
            dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_a(
                down_mma, down_tiler, dtype, self.down_stages
            ),
            alignment=1024,
        )
        sx = cute_ext.allocate(
            dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_b(
                down_mma, down_tiler, dtype, self.down_stages
            ),
            alignment=1024,
        )
        su = cute_ext.allocate(
            dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_a(up_mma, up_tiler, dtype, 3),
            alignment=1024,
        )
        sa = cute_ext.allocate(
            dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_b(up_mma, up_tiler, dtype, 3),
            alignment=1024,
        )
        gate = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout((128, self.n)),
            alignment=128,
        )
        # Each rank receives [16 source ranks, N tokens, 4 columns].
        # Up weights stay resident while all 16 CTAs reduce disjoint columns.
        mailbox = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout(self.split_k * 4 * self.n),
            alignment=128,
        )
        partial_tile = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout(68 * self.n),
            alignment=128,
        )
        bars = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(2 * self.down_stages + 13),
            alignment=8,
        ).iterator
        down_full = bars
        down_empty = down_full + self.down_stages
        down_done = down_empty + self.down_stages
        epi_done = down_done + 1
        up_full = epi_done + 1
        control_ready = up_full + 3
        projection_ready = control_ready + 1
        up_empty = projection_ready + 1
        up_done = up_empty + 3
        reduce_ready = up_done + 1
        cluster_done = reduce_ready + 1
        tmem_base = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        ).iterator
        if warp == 4:
            with cute.arch.elect_one():
                # Preserve layout and initialization order. A small GEMM K
                # never overwrites a slot, so its empty protocol is omitted.
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
                if cutlass.const_expr(self.ps > 3 * 128):
                    for stage in cutlass.range_constexpr(3):
                        cute.arch.mbarrier_init(up_empty + stage, 1)
                cute.arch.mbarrier_init(up_done, 1)
                cute.arch.mbarrier_init(reduce_ready, 1)
                cute.arch.mbarrier_init(cluster_done, self.split_k)
        # Every remote destination and its mbarrier must exist before a peer
        # can issue st.async.shared::cluster into it.
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()
        down_acc_layout = cute_ext.make_tmem_layout_acc(
            down_mma, (self.down_m, self.n), acc_stage=1
        )
        up_acc_layout = cute_ext.make_tmem_layout_acc(
            up_mma, (128, self.n), acc_stage=1
        )
        if warp == 4:
            # Both weight projections use this one producer. No activation
            # dependency precedes their initial loads in independent mode.
            if cutlass.const_expr(self.pdl and not self.weights_independent):
                cute.arch.griddepcontrol_wait()
            map_d = cute_ext.get_cta_v_map_ab(w, down_tiler, down_mma, "A")
            map_u = cute_ext.get_cta_v_map_ab(up, up_tiler, up_mma, "A")
            gd = cute.local_tile(w, (self.down_m, self.down_k), (cluster, None, 0))
            for stage in cutlass.range_constexpr(self.down_stages):
                if cutlass.const_expr(self.k_tiles > self.down_stages):
                    cute.arch.mbarrier_wait(down_empty + stage, 1)
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
            if pid < 80:
                self._load_up(up, su, up_full, up_empty, map_u, pid)
            # Any recycled Down slot waits only for its own empty phase.
            # Up weight requests have already been issued before that wait.
            for tile in cutlass.range_constexpr(self.down_stages, self.k_tiles):
                stage = tile % self.down_stages
                cute.arch.mbarrier_wait(
                    down_empty + stage, (tile // self.down_stages - 1) % 2
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
        elif warp == 5:
            cute.arch.alloc_tmem(32, tmem_base, is_two_cta=False)
            cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
            generation = _load_epoch_acquire(epochs.iterator + cluster) + 1
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
                phase = tile // self.down_stages % 2
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
            cute.arch.mbarrier_wait(epi_done, 0)
            with cute.arch.elect_one():
                _cluster_arrive(cluster_done)
                if split_rank == 0:
                    _cluster_wait(cluster_done)
                    _store_epoch_release(epochs.iterator + cluster, generation)
            lane = tid % 32
            ready = cutlass.Boolean(False)
            while not ready:
                local_ready = cutlass.Boolean(True)
                if lane < self.clusters:
                    local_ready = (
                        _load_epoch_acquire(epochs.iterator + lane) >= generation
                    )
                ready = cute.arch.vote_all_sync(local_ready, mask=0xFFFFFFFF)
            cute.arch.sync_warp(mask=0xFFFFFFFF)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(projection_ready)
            if pid < 80:
                up_acc = cute.make_tensor(
                    cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, tmem_base),
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
                    m_dim=128,
                    k_dim=0,
                    max_shift=0,
                )
                # A K64 group spans extent*128 bytes. Every group base is
                # 1024-byte aligned, so swizzle/base/layout fields are unchanged.
                # The 14-bit start field uses16-byte units, with no carry here.
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
                    cute.arch.mbarrier_wait(up_full + stage, 0)
                    with cute.arch.elect_one():
                        for block in cutlass.range_constexpr(8 if stage < 2 else 4):
                            kb = stage * 8 + block
                            _mma_single_thread(
                                up_acc,
                                desc_a + (kb // 4) * 128 * 8 + (kb % 4) * 2,
                                desc_b + (kb // 4) * self.n * 8 + (kb % 4) * 2,
                                up_instruction,
                                kb != 0,
                            )
                        if cutlass.const_expr(self.ps > 3 * 128):
                            tcgen05.commit(up_empty + stage, None, self.group)
                with cute.arch.elect_one():
                    tcgen05.commit(up_done, None, self.group)
                # Phase zero acknowledged down reads and DSM reduction. Phase one
                # acknowledges all up reads, before the independent gate work.
                cute.arch.mbarrier_wait(epi_done, 1)
            tcgen05_fence("after_thread_sync")
            cute.arch.dealloc_tmem(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, tmem_base),
                32,
                is_two_cta=False,
            )

        elif warp < 4:
            epi_tid = tid
            cute.arch.mbarrier_wait(control_ready, 0)
            if warp == 0:
                map_x = cute_ext.get_cta_v_map_ab(x, down_tiler, down_mma, "B")
                gx = cute.local_tile(x, (self.n, self.down_k), (0, None, 0))
                for tile in cutlass.range_constexpr(self.k_tiles):
                    stage = tile % self.down_stages
                    phase = (tile // self.down_stages + 1) % 2
                    if cutlass.const_expr(self.k_tiles > self.down_stages):
                        cute.arch.mbarrier_wait(down_empty + stage, phase)
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
            cute.arch.mbarrier_wait(down_done, 0)
            down_values, down_coords = self._read_acc(
                tmem_base, down_acc_layout, self.down_m, activation, epi_tid
            )
            # Every source scatters four-column slices to all 16 owners.
            # Owner r receives [source_rank, token, column_within_4].
            if epi_tid == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(
                    reduce_ready, self.split_k * 4 * self.n * 4
                )
            for i in cutlass.range_constexpr(cute.size(down_values)):
                col, row = down_coords[i]
                partial_tile[row * 68 + col] = down_values[i]
            _epilogue_barrier()
            for batch in cutlass.range_constexpr(self.n // 8):
                item = epi_tid + batch * 128
                owner = item // self.n
                row = item % self.n
                offset = row * 68 + owner * 4
                _remote_partial(
                    partial_tile[offset],
                    partial_tile[offset + 1],
                    partial_tile[offset + 2],
                    partial_tile[offset + 3],
                    mailbox.iterator + split_rank * self.n * 4 + row * 4,
                    reduce_ready,
                    owner,
                )
            cute.arch.mbarrier_wait(reduce_ready, 0)
            if epi_tid < self.rows * 4:
                row = epi_tid // 4
                col = cluster * 64 + split_rank * 4 + epi_tid % 4
                value = mailbox[epi_tid]
                for peer in cutlass.range_constexpr(1, self.split_k):
                    value = value + mailbox[peer * self.n * 4 + epi_tid]
                value = value * self.scale
                if col < 320:
                    value = value * cute_math.rcp(
                        1.0 + cute.exp(-value, fastmath=True),
                        fastmath=False,
                        approx=True,
                        rounding=None,
                        ftz=True,
                    )
                    activation[row, col] = value.to(dtype)
                    _publish_activation_proxy()
                elif col < self.p:
                    inject[row, col - 320] = value.to(dtype)
            # The complete 128-thread acknowledgement orders projection stores
            # before compute publishes this CTA into the cluster completion barrier.
            tcgen05_fence("before_thread_sync")
            cute.arch.mbarrier_arrive(epi_done)
            cute.arch.mbarrier_wait(projection_ready, 0)
            if pid < 80:
                if warp == 0:
                    active = cute.make_tensor(
                        activation.iterator,
                        cute.make_layout((16, 320, 1), stride=(320, 1, 0)),
                    )
                    map_a = cute_ext.get_cta_v_map_ab(active, up_tiler, up_mma, "B")
                    ga = cute.local_tile(active, (self.n, 128), (0, None, 0))
                    for stage in cutlass.range_constexpr(3):
                        if cutlass.const_expr(self.ps > 3 * 128):
                            cute.arch.mbarrier_wait(up_empty + stage, 1)
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
                    x.iterator, cute.make_layout((self.rows, 10240), stride=(10240, 1))
                )
                gates, up_coords, x_regs = self._read_up_with_prefetched_x(
                    tmem_base,
                    up_acc_layout,
                    activation,
                    epi_tid,
                    up_done,
                    x_values,
                    pid,
                )
                # All 128 readers have completed their tcgen05.wait::ld before
                # releasing this second phase. Gate/output use only registers
                # and shared memory, so TMEM can now be retired concurrently.
                tcgen05_fence("before_thread_sync")
                cute.arch.mbarrier_arrive(epi_done)
                for i in cutlass.range_constexpr(cute.size(gates)):
                    gate[up_coords[i][0], up_coords[i][1]] = x_regs[
                        i
                    ] * cute.arch.rcp_approx(1.0 + cute.exp(-gates[i], fastmath=True))
                _epilogue_barrier()
                for part in cutlass.range_constexpr(32 * self.n // 128):
                    elem = epi_tid + part * 128
                    j = elem % 32
                    row = elem // 32
                    if row < self.rows:
                        value = cutlass.Float32(0)
                        for branch in cutlass.range_constexpr(4):
                            value = value + gate[branch * 32 + j, row]
                        out[row, pid * 32 + j] = (value * 0.25).to(dtype)
