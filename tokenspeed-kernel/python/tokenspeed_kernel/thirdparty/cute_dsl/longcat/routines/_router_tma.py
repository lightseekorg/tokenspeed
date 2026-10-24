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

"""LongCat BF16 Router with SwapAB full-K TMA/UMMA.

Uses CuTe TMA/UMMA pipelines and TMEM allocation.
Output owners cover expert rows; the eight tokens occupy the narrow MMA N.
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100
from cutlass.cute.nvgpu import cpasync, tcgen05


class LongcatRouterTma:
    """One-CTA BF16 router with four K splits and a separate top-12 phase.

    Each CTA owns 128 expert rows and publishes one partial-logit slice.
    The four-warp top-k sums those slices in order before selecting routes.
    """

    def __init__(self):
        self.split_topk = True
        self.one_cta = True
        self.pair_local = False
        self.topk_local = False
        self.topk_candidate_count = 4
        self.topk_warps = 4
        self.k_splits = 4
        self.stages = 3
        self.tile = (128, 16, 256)
        self.cluster = (2, 1, 1)

    @cute.jit
    def prepare_device_routine(self, x, weight, logits):
        self.expert_count = weight.shape[0]
        if cutlass.const_expr(
            (
                self.tile[0] != (128)
                or self.cluster[0] != (2)
                or self.expert_count != 896
            )
        ):
            raise ValueError("Fused routing requires one 256-expert-tile cluster")
        # Split mode keeps the routing SMEM (row/partials) but skips the DSMEM
        # stores and in-routine top-k; router_topk_device_routine does that.
        # View, not a materialized transpose: logits[token, expert].
        out = cute.make_tensor(
            logits.iterator,
            cute.make_layout((self.expert_count, 8), stride=(1, self.expert_count)),
        )
        mma = cute.make_tiled_mma(
            tcgen05.MmaF16BF16Op(
                cutlass.BFloat16,
                cutlass.Float32,
                (self.tile[0], 16, 16),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
                tcgen05.OperandMajorMode.K,
                tcgen05.OperandMajorMode.K,
            )
        )
        # Each CTA owns its TMA and MMA pipeline inside the physical pair.
        cta_shape = (1, 1, 1)
        layout = cute.tiled_divide(cute.make_layout(cta_shape), (mma.thr_id,))
        sa = sm100.make_smem_layout_a(mma, self.tile, cutlass.BFloat16, self.stages)
        sb = sm100.make_smem_layout_b(mma, self.tile, cutlass.BFloat16, self.stages)
        copy_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        ta, ga = cute.nvgpu.make_tiled_tma_atom_A(
            copy_op,
            weight,
            cute.slice_(sa, (None, None, None, 0)),
            self.tile,
            mma,
            layout.shape,
        )
        tb, gb = cute.nvgpu.make_tiled_tma_atom_B(
            copy_op,
            x,
            cute.slice_(sb, (None, None, None, 0)),
            self.tile,
            mma,
            layout.shape,
        )

        @cute.struct
        class Storage:
            ab: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            acc: cute.struct.MemRange[cutlass.Int64, 2]
            dealloc: cutlass.Int64
            tmem: cutlass.Int32
            # Each CTA receives one token row through DSMEM. Keep this separate
            # from the live AB ring so routing never aliases GEMM operands.
            row: cute.struct.MemRange[cutlass.Float32, 896]
            route_values: cute.struct.MemRange[cutlass.Float32, 8]
            route_ids: cute.struct.MemRange[cutlass.Int32, self.topk_candidate_count]
            a: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(sa)], 128
            ]
            b: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(sb)], 128
            ]

        self.storage = Storage
        self.smem_bytes = Storage.size_in_bytes()
        return mma, layout, ta, ga, tb, gb, out, sa, sb

    @cute.jit
    def router_device_routine(
        self, mma, layout, ta, ga, tb, gb, out, la, lb, owner, rank, arena
    ):
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.warp_idx()
        k_tiles = 8192 // self.tile[2] // self.k_splits
        k_first = 0
        coord = (0, 0, 0, 0)
        part = 0
        block = owner * self.cluster[0] + rank
        expert_tile = block // self.k_splits
        if cutlass.const_expr(self.k_splits > 1):
            split = block % self.k_splits
            k_first = split * k_tiles
            out = cute.make_tensor(
                out.iterator + split * (self.expert_count * 8), out.layout
            )
        storage = self.storage(arena)
        sa = storage.a.get_tensor(la.outer, swizzle=la.inner)
        sb = storage.b.get_tensor(lb.outer, swizzle=lb.inner)
        if warp == 5:
            cpasync.prefetch_descriptor(ta)
            cpasync.prefetch_descriptor(tb)
        ma = None
        mb = None
        tx_bytes = (
            cute.size_in_bytes(cutlass.BFloat16, cute.select(la, mode=[0, 1, 2]))
            + cute.size_in_bytes(cutlass.BFloat16, cute.select(lb, mode=[0, 1, 2]))
        ) * (1)
        abp, abc = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab.data_ptr(),
            num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                size=cute.size(layout, mode=[1]) + cute.size(layout, mode=[2]) - 1,
            ),
            tx_count=tx_bytes,
            cta_layout_vmnk=layout,
            defer_sync=True,
        ).make_participants()
        ap, ac = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=4),
            cta_layout_vmnk=layout,
            defer_sync=True,
        ).make_participants()
        tm = utils.TmemAllocator(
            storage.tmem,
            barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=2, num_threads=160),
            allocator_warp_id=0,
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.dealloc,
        )
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()

        # Each pair owns a distinct expert tile. Activation multicast spans
        # four pairs by default, or stays within each pair with pair_local.
        gat = cute.local_tile(ga, self.tile, (expert_tile, 0, None), proj=(1, None, 1))
        gbt = cute.local_tile(gb, self.tile, (expert_tile, 0, None), proj=(None, 1, 1))
        gout = cute.local_tile(
            out, self.tile, (expert_tile, 0, None), proj=(1, 1, None)
        )
        ident = cute.local_tile(
            cute.make_identity_tensor(out.shape),
            self.tile,
            (expert_tile, 0, None),
            proj=(1, 1, None),
        )
        thread_mma = mma.get_slice(part)
        cga = thread_mma.partition_A(gat)
        cgb = thread_mma.partition_B(gbt)
        cgc = thread_mma.partition_C(gout)
        ci = thread_mma.partition_C(ident)
        a_shared, a_global = cpasync.tma_partition(
            ta,
            coord[2],
            cute.make_layout(cute.size(layout, mode=[2])),
            cute.group_modes(sa, 0, 3),
            cute.group_modes(cga, 0, 3),
        )
        b_shared, b_global = cpasync.tma_partition(
            tb,
            coord[1],
            cute.make_layout(cute.size(layout, mode=[1])),
            cute.group_modes(sb, 0, 3),
            cute.group_modes(cgb, 0, 3),
        )
        fa = mma.make_fragment_A(sa)
        fb = mma.make_fragment_B(sb)
        acc_shape = mma.partition_shape_C(self.tile[:2])
        acc_fake = mma.make_fragment_C(acc_shape)

        # All 160 participants rendezvous at the same named-barrier site.
        tm.allocate(32)
        if warp < 5:
            tm.wait_for_alloc()
        if warp == 5:
            for k in range(k_first, k_first + k_tiles):
                h = abp.acquire_and_advance()
                cute.copy(
                    ta,
                    a_global[None, k],
                    a_shared[None, h.index],
                    tma_bar_ptr=h.barrier,
                    mcast_mask=ma,
                )
                cute.copy(
                    tb,
                    b_global[None, k],
                    b_shared[None, h.index],
                    tma_bar_ptr=h.barrier,
                    mcast_mask=mb,
                )
            abp.tail()
        elif warp == 4:
            acc = cute.make_tensor(tm.retrieve_ptr(cutlass.Float32), acc_fake.layout)
            if part == 0:
                ah = ap.acquire_and_advance()
                for k in range(k_tiles):
                    h = abc.wait_and_advance()
                    mma.set(tcgen05.Field.ACCUMULATE, k != 0)
                    cute.gemm(
                        mma,
                        acc,
                        fa[None, None, None, h.index],
                        fb[None, None, None, h.index],
                        acc,
                    )
                    h.release()
                ah.commit()
        elif warp < 4:
            self.epilogue(tm, ac, acc_fake, cgc, ci, tid, storage)
        cute.arch.sync_threads()
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()

    @cute.jit
    def router_topk_device_routine(
        self,
        out: cute.Tensor,
        routing: tuple,
        token,
        tidx,
        arena,
        rounds=12,
        write=True,
    ):
        """Split-mode top-k for one token from the global logits.

        Runs four-warp top-k on a row staged from global memory.
        Plain global loads: the caller must make all
        four owners' logits stores visible first (release after
        router_device_routine, acquire before this call).

        Args:
            out: Expert-major view of the split router logits.
            routing: Bias, output ids, weights and identity-expert weights.
            token: Token index in [0, 8); selects the logits/ids/weights row.
            tidx: Thread index in [0, 128). All four aligned warps call this;
                named barrier id 3 synchronizes the 128 threads.
            arena: SMEM base laid out as self.storage (self.smem_bytes); only
                row, route_values and route_ids are touched.
            rounds: Selection rounds; fewer than 12 only for a warm-up pass.
            write: Runtime predicate on the ids / weights / zero stores; a
                warm-up pass (rounds=1, write=False) runs the same code on
                whatever the logits hold, to fetch it into the I-cache.
        """
        storage = self.storage(arena)
        row = storage.row.get_tensor(cute.make_layout((896,)))
        # Each thread stages exactly the elements the top-k helper makes it
        # read, so program order suffices and no extra barrier is needed.
        for i in cutlass.range_constexpr(7):
            value = out[tidx + i * 128, token]
            for s in cutlass.range_constexpr(1, self.k_splits):
                part = cute.make_tensor(
                    out.iterator + s * (self.expert_count * 8), out.layout
                )
                value = value + part[tidx + i * 128, token]
            row[tidx + i * 128] = value
        self.topk_four_warps(storage, routing, token, tidx, rounds, write)

    @cute.jit
    def epilogue(self, tm, ac, acc_fake, output, coords, tid, storage):
        ptr = tm.retrieve_ptr(cutlass.Float32)
        acc = cute.make_tensor(ptr, acc_fake.layout)
        h = ac.wait_and_advance()
        copy = tcgen05.make_tmem_copy(
            cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition.x8), cutlass.Float32
            ),
            acc,
        )
        thread = copy.get_slice(tid)
        src = thread.partition_S(acc)
        dst = thread.partition_D(output)
        idx = thread.partition_D(coords)
        reg = cute.make_rmem_tensor(dst.shape, cutlass.Float32)
        cute.copy(copy, src, reg)
        for i in cutlass.range_constexpr(cute.size(reg)):
            if cute.elem_less(idx[i], (self.expert_count, 8)):
                dst[i] = reg[i]
        # Single full-K output: the accumulator is retired, never recycled.
        tm.relinquish_alloc_permit()
        tm.free(ptr)

    @cute.jit
    def topk_four_warps(self, storage, routing, token, thread, rounds=12, write=True):
        # Four warps own one token's softmax+bias/top12.
        # ``rounds`` / ``write``: see router_topk_device_routine.
        bias, ids, weights, zero = routing
        store = thread == 0
        if cutlass.const_expr(write is not True):
            store = store & write
        lane = thread % 32
        warp = thread // 32
        barrier = pipeline.NamedBarrier(barrier_id=3, num_threads=128)
        row = storage.row.get_tensor(cute.make_layout((896,)))
        partial = storage.route_values.get_tensor(cute.make_layout((8,)))
        partial_ids = storage.route_ids.get_tensor(
            cute.make_layout((self.topk_candidate_count,))
        )
        scores = cute.make_rmem_tensor((7,), cutlass.Float32)
        biases = cute.make_rmem_tensor((7,), cutlass.Float32)
        maximum = cutlass.Float32(-float("inf"))
        for i in cutlass.range_constexpr(7):
            scores[i] = row[thread + i * 128]
            biases[i] = bias[thread + i * 128]
            maximum = cute.arch.fmax(maximum, scores[i])
        for shift in cutlass.range_constexpr(5):
            maximum = cute.arch.fmax(
                maximum, cute.arch.shuffle_sync_bfly(maximum, 1 << shift)
            )
        if lane == 0:
            partial[warp] = maximum
        barrier.arrive_and_wait()
        for i in cutlass.range_constexpr(4):
            maximum = cute.arch.fmax(maximum, partial[i])
        barrier.arrive_and_wait()
        total = cutlass.Float32(0)
        for i in cutlass.range_constexpr(7):
            scores[i] = cute.exp(scores[i] - maximum, fastmath=True)
            total += scores[i]
        total = cute.arch.warp_reduction_sum(total)
        if lane == 0:
            partial[warp] = total
        barrier.arrive_and_wait()
        total = cutlass.Float32(0)
        for i in cutlass.range_constexpr(4):
            total += partial[i]
        barrier.arrive_and_wait()
        for i in cutlass.range_constexpr(7):
            scores[i] = scores[i] / total + biases[i]
        zero_sum = cutlass.Float32(0)
        for k in range(rounds):
            best = cutlass.Float32(-float("inf"))
            expert = cutlass.Int32(896)
            selected_bias = cutlass.Float32(0)
            for i in cutlass.range_constexpr(7):
                if scores[i] > best:
                    best = scores[i]
                    expert = thread + i * 128
                    selected_bias = biases[i]
            for shift in cutlass.range_constexpr(5):
                other = cute.arch.shuffle_sync_bfly(best, 1 << shift)
                other_id = cute.arch.shuffle_sync_bfly(expert, 1 << shift)
                other_bias = cute.arch.shuffle_sync_bfly(selected_bias, 1 << shift)
                if (other > best) | ((other == best) & (other_id < expert)):
                    best = other
                    expert = other_id
                    selected_bias = other_bias
            if lane == 0:
                partial[warp] = best
                partial[warp + 4] = selected_bias
                partial_ids[warp] = expert
            barrier.arrive_and_wait()
            for i in cutlass.range_constexpr(4):
                other = partial[i]
                other_id = partial_ids[i]
                if (other > best) | ((other == best) & (other_id < expert)):
                    best = other
                    expert = other_id
                    selected_bias = partial[i + 4]
            barrier.arrive_and_wait()
            for i in cutlass.range_constexpr(7):
                if thread + i * 128 == expert:
                    scores[i] = -float("inf")
            value = (best - selected_bias) * 9.0
            if expert >= 768:
                zero_sum += value
            if store:
                if expert < 768:
                    ids[token, k] = expert
                    weights[token, k] = value
                else:
                    ids[token, k] = 0
                    weights[token, k] = 0.0
        if store:
            zero[token] = zero_sum
