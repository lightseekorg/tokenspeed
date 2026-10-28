# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


"""SM107 sparse attention with TMA gather4 and a persistent CLC scheduler.

Each work item is one or two consecutive queries and one KV head, with twelve
heads in sixteen padded columns per query. Each query uses two four-warp softmax
groups, sharing eight gather warps. MMA, Q/list loads and scheduling each have
one warp. Paired queries share a selected-block list, with independent causal masks. Two K/V
slots in the paired tactic leave room for wider Q/P operands; the single-query
tactic uses three. No CTA-wide barrier follows initialization.

QK/PV accumulate in FP32 and probabilities round to BF16 before PV. The
softmax masks missing pages as well as logical/causal bounds. Split outputs
are normalized FP32 O and natural-log LSE, with zero/-inf for empty rows.
"""

from typing import Callable, NamedTuple, Optional, Tuple

import cuda.bindings.driver as _cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass import pipeline
from cutlass._mlir.dialects import arith
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as prims
from cutlass.experimental.cuda import tensor_map as tmap
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_tma_gather.barrier import (
    MBarrier,
    Producer,
    wait,
)
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_tma_gather.common import (
    _bshd,
    _vec,
    row_max_for_exp2,
)
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_tma_gather.handles import (
    MmaDesc,
    SmemTile,
)
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_tma_gather.mma import mma_ss
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_tma_gather.scheduler import (
    Sched,
    read_clc_payload,
    read_tile_id_arrive,
    scheduler_warp_loop,
)
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_tma_gather.tma import bulk_copy, load_q
from tokenspeed_kernel.thirdparty.cute_dsl.conversion import cvt_f16x2_f32
from tokenspeed_kernel.thirdparty.cute_dsl.tma import (
    TMA_CACHE_EVICT_LAST,
    tma_cache_hint,
    tma_gather4,
)


class SparseBars(NamedTuple):
    mb_q_full: object
    mb_q_empty: object
    mb_ids_full: object
    mb_ids_empty: object
    mb_kv_full: object
    mb_kv_empty: object
    mb_s_full: object
    mb_s_empty: object
    mb_p_full: object
    mb_bmm2_done: object
    mb_tmem_dealloc: object


@cute.jit
def _select_i32(cond, a, b):
    return cutlass.Int32(
        arith.select(
            cond.ir_value(), cutlass.Int32(a).ir_value(), cutlass.Int32(b).ir_value()
        )
    )


@cute.jit
def _select_f32(cond, a, b):
    return cutlass.Float32(arith.select(cond.ir_value(), a.ir_value(), b.ir_value()))


@cute.jit
def _warp_reduce_max(x):
    for off in cutlass.range_constexpr(5):
        x = cute.math.max(
            x,
            cutlass.Float32(
                prims.shfl_sync(0xFFFFFFFF, x, 16 >> off, 31, kind=prims.Shfl.BFLY)
            ),
            ftz=True,
        )
    return x


@cute.jit
def _warp_reduce_sum(x):
    for off in cutlass.range_constexpr(5):
        x = x + cutlass.Float32(
            prims.shfl_sync(0xFFFFFFFF, x, 16 >> off, 31, kind=prims.Shfl.BFLY)
        )
    return x


class TmaGatherSparseAttentionKernel:
    """Static geometry and CuTe entry points for paged TMA-gather attention.

    Each instance owns one page-size/split tactic. Request shapes, strides
    and lengths remain runtime arguments to its compiled callable.
    """

    def __init__(self, *, page_size: int, split_kv: int, query_group_size: int):
        if page_size <= 0 or page_size % 4:
            raise ValueError("page_size must be a positive multiple of four")
        if split_kv not in (1, 4, 8, 16):
            raise ValueError("split_kv must be one of 1, 4, 8, 16")
        if query_group_size not in (1, 2):
            raise ValueError("query_group_size must be one or two")
        self.query_group_size = query_group_size
        self.page_size = page_size
        self.split_kv = split_kv

        # Fixed BF16 d256 geometry: twelve query heads use a sixteen-column tile.
        self.tile_n = 128
        self.tile_k = 256
        self.tile_o = self.tile_k
        self.query_head_tile = 16
        self.n_q = self.query_head_tile * query_group_size
        self.qh_per_kh = 12
        self.bpe = 2
        self.block_size = 4
        self.block_topk = 512
        self.blocks_per_tile = self.tile_n // self.block_size
        self.tile_k_hw = 16
        self.include_open_block = 1

        # Pipeline depths and warp roles retain the validated producer/consumer layout.
        # The wider Q/P operands leave room for two K/V slots in the paired tactic.
        self.stages_kv = 3 if query_group_size == 1 else 2
        self.stages_s = 2
        self.stages_q = 2
        self.stages_ids = 2
        self.scheduler_stages = 2
        # Eight columns per group keeps per-thread softmax state independent
        # of the number of queries sharing the K/V tile.
        self.softmax_groups = 2 * query_group_size
        self.softmax_warps_per_group = 4
        self.softmax_warps = self.softmax_groups * self.softmax_warps_per_group
        self.cols_per_group = self.n_q // self.softmax_groups
        self.gather_warps = 8
        self.gather_warp_base = self.softmax_warps
        self.mma_warp_id = self.gather_warp_base + self.gather_warps
        self.tmaldg_warp_id = self.mma_warp_id + 1
        self.sched_warp_id = self.tmaldg_warp_id + 1
        # One spare warp exits after CTA initialization.
        self.threads_per_cta = (self.sched_warp_id + 2) * 32
        self.one_lane = 1
        self.softmax_lanes = self.softmax_warps * 32
        self.kv_full_arrivers = self.gather_warps
        self.ids_empty_arrivers = self.softmax_warps + self.gather_warps
        self.read_tile_arrivers = self.ids_empty_arrivers + 2
        self.bar_tmem_threads = (self.softmax_warps + 1) * 32
        self.bar_softmax_threads = self.softmax_warps_per_group * 32

        # TMA boxes and stage storage are derived from the operand geometry.
        self.q_swz_bytes = 128
        self.k_swz_bytes = 128
        self.v_swz_bytes = 128
        self.p_swz_bytes = self.n_q * self.bpe
        self.blocks_per_warp = self.blocks_per_tile // self.gather_warps
        self.gather_boxes = self.tile_k * self.bpe // self.k_swz_bytes
        self.gather_box_elems = self.k_swz_bytes // self.bpe
        self.kv_tx_bytes_per_warp = (
            self.blocks_per_warp * self.block_size * self.tile_k * self.bpe
        )
        self.q_box_tokens = 1
        self.q_box_rows = self.qh_per_kh * self.q_box_tokens
        self.q_tx_bytes = self.q_box_rows * self.tile_k * self.bpe * query_group_size
        self.ids_tx_bytes = self.block_topk * 4
        self.ids_slot_bytes = self.ids_tx_bytes + 64
        self.qk_granu_elems = self.gather_box_elems
        self.qk_iters = self.tile_k // self.qk_granu_elems
        # Two score slots and two D128 output blocks; unrelated to warp groups.
        self.tmem_cols = self.n_q * (self.stages_s + self.tile_o // 128)
        self.s_acc_off = (0, self.n_q)
        self.o_off = (self.n_q * 2, self.n_q * 3)
        self.storage_dtype = cutlass.BFloat16
        self.mma_kind = prims.Tcgen05MMAKind.F16
        self.q_buffer_elems = self.n_q * self.tile_k
        self.kv_buffer_elems = self.tile_n * self.tile_k
        self.p_buffer_words = self.tile_n * self.n_q // 2
        self.sub_box_elems = self.tile_n * self.gather_box_elems
        mma_smem_bytes = (
            self.q_buffer_elems * self.stages_q * self.bpe
            + self.kv_buffer_elems * self.stages_kv * self.bpe
            + self.p_buffer_words * self.stages_s * 4
        )
        if mma_smem_bytes > 256 * 1024:
            raise ValueError("MMA operands exceed the 256 KiB descriptor window")
        self.ids_slot_words = self.ids_slot_bytes // 4
        swizzle_layouts = {128: 2, 64: 4, 32: 6}
        swizzle_bits = {128: 3, 64: 2, 32: 1}
        self.smem_layout_qk = swizzle_layouts[self.q_swz_bytes]
        self.smem_layout_p = swizzle_layouts[self.p_swz_bytes]
        self.p_smem_swizzle = cutlass.Swizzle(swizzle_bits[self.p_swz_bytes], 4, 3)
        self.p_row_words = self.n_q // 2
        self.stride_byte_offset_qk = 8 * self.k_swz_bytes
        self.leading_byte_offset_vt = self.tile_n * self.v_swz_bytes
        self.stride_byte_offset_vt = 8 * self.v_swz_bytes
        self.stride_byte_offset_p = 8 * self.p_swz_bytes
        self.o_blocks = self.tile_o // 128
        self.group_lanes = 32 * self.softmax_warps_per_group
        self.p_chunks = (self.cols_per_group * self.bpe) // 16
        self.p_group_words = self.cols_per_group // 2
        self.bar_tmem = 1
        self.bar_softmax = 2
        self.red_slots = self.stages_s + 1
        self.red_words = self.red_slots * self.softmax_warps * self.cols_per_group
        self.ln2 = 0.6931471805599453

    def _make_sparse_bars(self) -> SparseBars:
        """Allocate stage barriers with exact producer/consumer arrival counts."""

        def _alloc(n):
            return cutlass.Array(
                cutlass.Int64, n, alignment=16, space=cutlass.AddressSpace.smem
            )

        return SparseBars(
            mb_q_full=MBarrier(
                _alloc(self.stages_q),
                init_count=self.one_lane,
                producer=Producer.TMA_LOAD,
            ),
            mb_q_empty=MBarrier(
                _alloc(self.stages_q),
                init_count=self.one_lane,
                producer=Producer.MMA_COMMIT,
            ),
            mb_ids_full=MBarrier(
                _alloc(self.stages_ids),
                init_count=self.one_lane,
                producer=Producer.TMA_LOAD,
            ),
            mb_ids_empty=MBarrier(
                _alloc(self.stages_ids),
                init_count=self.ids_empty_arrivers,
                producer=Producer.THREAD,
            ),
            mb_kv_full=MBarrier(
                _alloc(self.stages_kv),
                init_count=self.kv_full_arrivers,
                producer=Producer.TMA_LOAD,
            ),
            mb_kv_empty=MBarrier(
                _alloc(self.stages_kv),
                init_count=self.one_lane,
                producer=Producer.MMA_COMMIT,
            ),
            mb_s_full=MBarrier(
                _alloc(self.stages_s),
                init_count=self.one_lane,
                producer=Producer.MMA_COMMIT,
            ),
            mb_s_empty=MBarrier(
                _alloc(self.stages_s),
                init_count=self.softmax_lanes,
                producer=Producer.THREAD,
            ),
            mb_p_full=MBarrier(
                _alloc(self.stages_s),
                init_count=self.softmax_lanes,
                producer=Producer.THREAD,
            ),
            mb_bmm2_done=MBarrier(
                _alloc(self.stages_s),
                init_count=self.one_lane,
                producer=Producer.MMA_COMMIT,
            ),
            mb_tmem_dealloc=MBarrier(
                _alloc(1),
                init_count=self.softmax_lanes,
                producer=Producer.THREAD,
            ),
        )

    @cute.jit
    def _item_bounds(
        self,
        tok,
        batch,
        item_row,
        eff_seqlen_q,
        block_lens_tensor: Optional[cute.Tensor],
        seq_kv_lens_tensor,
        seqlen_kv,
        split,
    ):
        """Compute the item's union of selected blocks and open tails.

        Adjacent queries have at most one distinct open block. Its extra list
        entry is masked independently by each softmax group, even when it has
        become a completed selected block for the later query.
        """
        eff_seqlen_kv = cutlass.Int32(
            cutlass.make_array_view(seq_kv_lens_tensor)[batch]
        )
        zero = cutlass.Int32(0)
        one = cutlass.Int32(1)
        bs = cutlass.Int32(self.block_size)
        pos = tok + (eff_seqlen_kv - eff_seqlen_q)
        n_vis = cute.math.max(cute.math.min(pos + one, eff_seqlen_kv), zero)
        first_n_vis = n_vis
        pos = pos + cutlass.Int32(self.query_group_size - 1)
        n_vis = cute.math.max(cute.math.min(pos + one, eff_seqlen_kv), zero)
        anchor = pos
        n_anchor = first_n_vis
        n_sel_default = cute.math.min(
            cutlass.Int32(self.block_topk), cute.math.max(anchor + one, zero) // bs
        )
        count = n_sel_default
        if cutlass.const_expr(block_lens_tensor is not None):
            given = cutlass.Int32(cutlass.make_array_view(block_lens_tensor)[item_row])
            count = cute.math.min(cute.math.max(given, zero), n_sel_default)
        tail_lo = n_anchor // bs
        n_tail = zero
        if cutlass.const_expr(self.include_open_block == 1):
            n_tail = _select_i32(
                ((n_vis % bs) != zero) | ((first_n_vis % bs) != zero), 1, 0
            )
        n_listed = count + n_tail
        dead = (
            (pos < zero)
            | (tok >= eff_seqlen_q)
            | (eff_seqlen_kv <= zero)
            | (n_listed == zero)
        )
        n_tiles = cute.math.max(
            cutlass.Int32(1),
            (n_listed + cutlass.Int32(self.blocks_per_tile - 1))
            // cutlass.Int32(self.blocks_per_tile),
        )
        tile0 = zero
        if cutlass.const_expr(self.split_kv > 1):
            per = n_tiles // cutlass.Int32(self.split_kv)
            rem = n_tiles % cutlass.Int32(self.split_kv)
            tile0 = split * per + cute.math.min(split, rem)
            n_sp = per + _select_i32(split < rem, 1, 0)
            dead = dead | (n_sp == zero)
            n_tiles = cute.math.max(cutlass.Int32(1), n_sp)
        return pos, count, n_tail, tail_lo, dead, n_tiles, eff_seqlen_kv, tile0

    @cute.jit
    def _block_id_at(self, idx, loaded, count, n_tail, tail_lo, dead):
        """The block id at list index ``idx`` of the current item: the staged word while ``idx < count``, the tail blocks
        ``tail_lo .. tail_lo + n_tail - 1`` at ``idx == count .. count + n_tail - 1`` (zero or one), -1 past that -- and -1
        for every index of a dead item.  ``loaded`` is the word the caller read unconditionally at an in-bounds address; this
        SELECT decides whether it means anything."""
        tail = _select_i32(
            (idx == count) & (n_tail != cutlass.Int32(0)), tail_lo, cutlass.Int32(-1)
        )
        blk = _select_i32(idx < count, loaded, tail)
        # Bound the block domain BEFORE forming 4 * blk + lane: a large positive Int32 id must not wrap into a live key.
        invalid = (
            dead
            | (blk < cutlass.Int32(0))
            | (blk > cutlass.Int32((2**31 - 1) // self.block_size))
        )
        return _select_i32(invalid, cutlass.Int32(-1), blk)

    @cute.jit
    def _decode_item(self, t0, t1, seq_kv_lens_tensor, n_batch, seqlen_q, seqlen_kv):
        """Decode the initial grid item or a subsequent CLC payload for every role."""
        split = cutlass.Int32(0)
        x = cute.arch.make_warp_uniform(t0)
        groups = seqlen_q // cutlass.Int32(self.query_group_size)
        tok = x
        if cutlass.const_expr(self.split_kv > 1):
            split = cute.arch.make_warp_uniform(x // groups)
            tok = x - split * groups
        tok = tok * cutlass.Int32(self.query_group_size)
        head = cute.arch.make_warp_uniform(t1 & cutlass.Int32(0xFFFF))
        batch = cute.arch.make_warp_uniform(
            (t1 >> cutlass.Int32(16)) & cutlass.Int32(0xFFFF)
        )
        eff_seqlen_q = seqlen_q
        item_row = batch * seqlen_q + tok
        kv_row_base = batch * seqlen_kv
        out_tok = tok
        out_batch = batch
        return (
            tok,
            head,
            batch,
            item_row,
            kv_row_base,
            eff_seqlen_q,
            out_tok,
            out_batch,
            split,
        )

    @cute.jit
    def _tma_issue_item(
        self,
        tma_q,
        sQ,
        sIds_raw,
        bars,
        ids_base,
        item_row,
        head,
        out_tok,
        out_batch,
        q_state: pipeline.PipelineState,
        qe_state: pipeline.PipelineState,
        i_state: pipeline.PipelineState,
        ie_state: pipeline.PipelineState,
    ):
        """Stage one selected-block row and its query in their next pipeline slots."""
        # --- the list (row 4 wait -> row 3 arm + copy)
        bars.mb_ids_empty[ie_state.index].wait(ie_state.phase)
        bars.mb_ids_full[i_state.index].arrive(
            n_bytes=self.ids_tx_bytes, pred=prims.elect_sync()
        )
        bulk_copy(
            sIds_raw.subview(i_state.index * cutlass.Int32(self.ids_slot_words)),
            ids_base + item_row * cutlass.Int32(self.block_topk),
            self.ids_tx_bytes,
            bars.mb_ids_full[i_state.index].smem_ptr,
        )
        # --- Q^T (the slot wait -> row 1 arm + 4 subtiles)
        bars.mb_q_empty[qe_state.index].wait(qe_state.phase)
        bars.mb_q_full[q_state.index].arrive(
            n_bytes=self.q_tx_bytes, pred=prims.elect_sync()
        )
        for query in cutlass.range_constexpr(self.query_group_size):
            load_q(
                sQ[q_state.index].base.subview(
                    query * self.query_head_tile * self.qk_granu_elems
                ),
                tma_q,
                bars.mb_q_full[q_state.index].smem_ptr,
                head * cutlass.Int32(self.qh_per_kh),
                out_tok + cutlass.Int32(query),
                out_batch,
                boxes=self.qk_iters,
                box_elems=self.qk_granu_elems,
                rows=self.n_q,
            )
        q_state.advance()
        qe_state.advance()
        i_state.advance()
        ie_state.advance()
        return q_state, qe_state, i_state, ie_state

    @cute.jit
    def _tmaldg_warp_group(
        self,
        tma_q_desc,
        sQ,
        sIds_raw,
        bars,
        sched,
        block_ids_tensor,
        seq_kv_lens_tensor,
        n_batch,
        seqlen_q,
        seqlen_kv,
        t0_init,
        t1_init,
        use_pdl: cutlass.Constexpr[bool],
    ):
        tma_q = tma_q_desc
        ids_base = Pointer(block_ids_tensor.iterator.raw_ptr(), dtype=cutlass.Int32)
        q_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.stages_q
        )
        qe_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.stages_q
        )
        i_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.stages_ids
        )
        ie_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.stages_ids
        )

        if cutlass.const_expr(use_pdl):
            # PDL (header, PDL SITES): this role's first global reads are the prologue's ids bulk copy (block_ids: the selector's / the
            # caller's list) and the Q^T box (the slab a projection / norm + RoPE launch wrote); the tensor-map prefetches above are kernel
            # parameters.  Nothing of the role is read or written above this wait.
            cute.arch.griddepcontrol_wait()
        # Prologue: the first item's loads (its coordinates are this CTA's grid position, decoded like every later payload).
        _tok, head, _batch, item_row, _kvb, _sq, out_tok, out_batch, _split = (
            self._decode_item(
                t0_init, t1_init, seq_kv_lens_tensor, n_batch, seqlen_q, seqlen_kv
            )
        )
        q_state, qe_state, i_state, ie_state = self._tma_issue_item(
            tma_q,
            sQ,
            sIds_raw,
            bars,
            ids_base,
            item_row,
            head,
            out_tok,
            out_batch,
            q_state,
            qe_state,
            i_state,
            ie_state,
        )

        is_valid = cutlass.Int32(1)
        sched_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.scheduler_stages
        )
        while is_valid > cutlass.Int32(0):
            # Credit the scheduler slot at the TOP of the item (row 14), then learn the NEXT item and prefetch its Q^T and list
            # under the current item's KV loop.  The _empty waits inside the issue bound this warp at one item ahead.
            read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.index))
            wait(
                sched.mb_scheduler.subview(sched_state.index),
                sched_state.phase,
            )
            t0, t1, nxt_v = read_clc_payload(
                sched, sched_state.index * cutlass.Int32(8), warp_broadcast=False
            )
            _tok, head, _batch, item_row, _kvb, _sq, out_tok, out_batch, _split = (
                self._decode_item(
                    t0, t1, seq_kv_lens_tensor, n_batch, seqlen_q, seqlen_kv
                )
            )
            is_valid = cute.arch.make_warp_uniform(nxt_v)
            sched_state.advance()
            if is_valid > cutlass.Int32(0):
                q_state, qe_state, i_state, ie_state = self._tma_issue_item(
                    tma_q,
                    sQ,
                    sIds_raw,
                    bars,
                    ids_base,
                    item_row,
                    head,
                    out_tok,
                    out_batch,
                    q_state,
                    qe_state,
                    i_state,
                    ie_state,
                )

        # Exit drains (P6): the last STAGES_IDS ids_empty arrives need a waiter; the carried states name exactly those phases.
        for _ in cutlass.range_constexpr(self.stages_ids):
            bars.mb_ids_empty[ie_state.index].wait(ie_state.phase)
            ie_state.advance()
        for _ in cutlass.range_constexpr(self.stages_q):
            bars.mb_q_empty[qe_state.index].wait(qe_state.phase)
            qe_state.advance()

    @cute.jit
    def _gather_block_ids(
        self, sIds_raw, slot_base, tile, w, count, n_tail, tail_lo, dead
    ):
        """This warp's BLOCKS_PER_WARP block ids of tile ``tile`` (the item's ABSOLUTE tile index: the split chunk's ``tile0 + i``):
        list indices ``32 tile + BLOCKS_PER_WARP w + q``.  The staged words are read with warp-uniform 16-B vector loads (the quad
        is 16-B aligned: BLOCK_TOPK % 4 == 0 and the slot start is) BEFORE the ring wait, at an address clamped into the slot; the
        SELECT (``_block_id_at``) substitutes the tail id(s) / -1 / the dead item's -1 per index.  Returns a Vector of
        BLOCKS_PER_WARP Int32."""
        first = tile * cutlass.Int32(self.blocks_per_tile) + w * cutlass.Int32(
            self.blocks_per_warp
        )
        ld_first = cute.math.min(
            first, cutlass.Int32(self.block_topk - self.blocks_per_warp)
        )
        ids = []
        for qq in cutlass.range_constexpr(self.blocks_per_warp // 4):
            v = sIds_raw.load(
                slot_base + ld_first + cutlass.Int32(qq * 4),
                vector_size=4,
                alignment=16,
            )
            for r in cutlass.range_constexpr(4):
                q = qq * 4 + r
                ids.append(
                    self._block_id_at(
                        first + cutlass.Int32(q),
                        cutlass.Int32(v[r]),
                        count,
                        n_tail,
                        tail_lo,
                        dead,
                    )
                )
        return cutlass.Vector.from_elements(tuple(ids), cutlass.Int32)

    @cute.jit
    def _physical_page(self, blk, table, batch, max_pages, num_pages, eff_seqlen_kv):
        """Use one page-validity predicate for both TMA reads and softmax weights."""
        key0 = blk * cutlass.Int32(self.block_size)
        logical_page = key0 // cutlass.Int32(self.page_size)
        index = cute.math.min(
            cute.math.max(logical_page, cutlass.Int32(0)), max_pages - cutlass.Int32(1)
        )
        page = cutlass.Int32(table[batch, index])
        valid = (
            (blk >= cutlass.Int32(0))
            & (key0 < eff_seqlen_kv)
            & (logical_page < max_pages)
        )
        # Page zero is TokenSpeed's null page. Reject invalid physical IDs before
        # issuing a TMA read, including the automatically appended incomplete tail.
        valid = valid & (page > cutlass.Int32(0)) & (page < num_pages)
        return _select_i32(valid, page, cutlass.Int32(-1))

    @cute.jit
    def _gather_block_pages(
        self, blks, block_table_tensor, batch, max_pages, num_pages, eff_seqlen_kv
    ):
        table = cutlass.make_array_view(block_table_tensor)
        pages = []
        for q in cutlass.range_constexpr(self.blocks_per_warp):
            pages.append(
                self._physical_page(
                    cutlass.Int32(blks[q]),
                    table,
                    batch,
                    max_pages,
                    num_pages,
                    eff_seqlen_kv,
                )
            )
        return cutlass.Vector.from_elements(tuple(pages), cutlass.Int32)

    @cute.jit
    def _gather_fill(
        self,
        desc,
        sKV_raw,
        stage_idx,
        mbar,
        w,
        blks,
        pages,
        kv_row_base,
        rows_per_page,
        head_rows,
        eff_seqlen_kv,
        col_base,
        hint,
    ):
        """Issue one warp's four selected blocks across four TMA column boxes."""
        stage_off = stage_idx * cutlass.Int32(self.kv_buffer_elems)
        quad0 = w * cutlass.Int32(self.blocks_per_warp)
        for q in cutlass.range_constexpr(self.blocks_per_warp):
            blk = cutlass.Int32(blks[q])
            key0 = blk * cutlass.Int32(self.block_size)
            page = cutlass.Int32(pages[q])
            in_page = key0 - (key0 // cutlass.Int32(self.page_size)) * cutlass.Int32(
                self.page_size
            )
            base = page * rows_per_page + head_rows + in_page
            live = page >= cutlass.Int32(0)
            rows = []
            for r in cutlass.range_constexpr(self.block_size):
                rows.append(
                    _select_i32(
                        live & ((key0 + cutlass.Int32(r)) < eff_seqlen_kv),
                        base + cutlass.Int32(r),
                        cutlass.Int32(-1),
                    )
                )
            quad_elems = (quad0 + cutlass.Int32(q)) * cutlass.Int32(
                self.block_size * self.gather_box_elems
            )
            for b in cutlass.range_constexpr(self.gather_boxes):
                tma_gather4(
                    cutlass.Int32(
                        sKV_raw.subview(
                            stage_off
                            + cutlass.Int32(b * self.sub_box_elems)
                            + quad_elems
                        )
                        .data_ptr()
                        .toint()
                    ),
                    desc.get_ptr().toint(),
                    cutlass.Int32(mbar.data_ptr().toint()),
                    col_base + cutlass.Int32(b * self.gather_box_elems),
                    rows[0],
                    rows[1],
                    rows[2],
                    rows[3],
                    cache_hint=hint,
                )

    @cute.jit
    def _gather_warp_group(
        self,
        tma_k_desc,
        tma_v_desc,
        sKV_raw,
        sIds_raw,
        bars,
        sched,
        block_lens_tensor,
        seq_kv_lens_tensor,
        seqlen_q,
        seqlen_kv,
        k_head_stride,
        v_head_stride,
        w,
        n_batch,
        t0_init,
        t1_init,
        block_table_tensor: Optional[cute.Tensor],
        max_pages,
        num_pages,
        k_rows_per_page,
        k_rows_per_head,
        v_rows_per_page,
        v_rows_per_head,
        use_pdl: cutlass.Constexpr[bool],
    ):
        if cutlass.const_expr(use_pdl):
            # PDL (header, PDL SITES): this role's first global reads are _item_bounds' lengths / counts, then the block table (paged arm)
            # and the K / V rows of the gathers -- all a predecessor's output; nothing is read above this wait.
            cute.arch.griddepcontrol_wait()
        hint = tma_cache_hint(TMA_CACHE_EVICT_LAST)
        # kv_state: the stage to fill next (producer of mb_kv_full) == the mb_kv_empty wait (pre-armed: the first STAGES_KV
        # fills pass fresh barriers); advanced per LOAD in the K(0), K(1), V(0), K(2), V(1), ... order the MMA warp consumes,
        # CARRIED across items (never recomputed from a per-item tile index -- barrier rows 5 / 6).
        kv_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.stages_kv
        )
        ids_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.stages_ids
        )
        tok, head, batch, item_row, kv_row_base, eff_seqlen_q, _ot, _ob, split = (
            self._decode_item(
                t0_init, t1_init, seq_kv_lens_tensor, n_batch, seqlen_q, seqlen_kv
            )
        )
        is_valid = cutlass.Int32(1)
        sched_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.scheduler_stages
        )
        while is_valid > cutlass.Int32(0):
            read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.index))
            _pos, count, n_tail, tail_lo, dead, n_tiles, eff_seqlen_kv, tile0 = (
                self._item_bounds(
                    tok,
                    batch,
                    item_row,
                    eff_seqlen_q,
                    block_lens_tensor,
                    seq_kv_lens_tensor,
                    seqlen_kv,
                    split,
                )
            )
            col_k = head * k_head_stride
            col_v = head * v_head_stride
            # PAGED arm: the per-item row terms of the head (0 for an NHD pool: its head lives in the column coordinate).
            head_rows_k = head * k_rows_per_head
            head_rows_v = head * v_rows_per_head
            slot_base = ids_state.index * cutlass.Int32(self.ids_slot_words)
            bars.mb_ids_full[ids_state.index].wait(ids_state.phase)

            # Ring order K(0), K(1), V(0), K(2), V(1), ...: K runs one tile ahead of V so BMM1(t+1) overlaps the softmax of t.
            # PAGED arm: the block-table lookups of a tile's quad ride with its ids (before the ring wait), K and V share them.
            blks_cur = self._gather_block_ids(
                sIds_raw, slot_base, tile0, w, count, n_tail, tail_lo, dead
            )
            pages_cur = self._gather_block_pages(
                blks_cur, block_table_tensor, batch, max_pages, num_pages, eff_seqlen_kv
            )
            bars.mb_kv_empty[kv_state.index].wait(kv_state.phase)
            bars.mb_kv_full[kv_state.index].arrive(
                n_bytes=self.kv_tx_bytes_per_warp, pred=prims.elect_sync()
            )
            if prims.elect_sync():
                self._gather_fill(
                    tma_k_desc,
                    sKV_raw,
                    kv_state.index,
                    bars.mb_kv_full[kv_state.index].smem_ptr,
                    w,
                    blks_cur,
                    pages_cur,
                    kv_row_base,
                    k_rows_per_page,
                    head_rows_k,
                    eff_seqlen_kv,
                    col_k,
                    hint,
                )
            kv_state.advance()
            for i in cutlass.range(0, n_tiles, 1, unroll=1):
                # The next tile's ids are read before this iteration's ring waits (an in-bounds address even past the last tile).
                t_next = cute.math.min(i + cutlass.Int32(1), n_tiles - cutlass.Int32(1))
                if cutlass.const_expr(self.split_kv > 1):
                    t_next = tile0 + t_next
                blks_next = self._gather_block_ids(
                    sIds_raw, slot_base, t_next, w, count, n_tail, tail_lo, dead
                )
                pages_next = self._gather_block_pages(
                    blks_next,
                    block_table_tensor,
                    batch,
                    max_pages,
                    num_pages,
                    eff_seqlen_kv,
                )
                if i + cutlass.Int32(1) < n_tiles:
                    bars.mb_kv_empty[kv_state.index].wait(kv_state.phase)
                    bars.mb_kv_full[kv_state.index].arrive(
                        n_bytes=self.kv_tx_bytes_per_warp, pred=prims.elect_sync()
                    )
                    if prims.elect_sync():
                        self._gather_fill(
                            tma_k_desc,
                            sKV_raw,
                            kv_state.index,
                            bars.mb_kv_full[kv_state.index].smem_ptr,
                            w,
                            blks_next,
                            pages_next,
                            kv_row_base,
                            k_rows_per_page,
                            head_rows_k,
                            eff_seqlen_kv,
                            col_k,
                            hint,
                        )
                    kv_state.advance()
                bars.mb_kv_empty[kv_state.index].wait(kv_state.phase)
                bars.mb_kv_full[kv_state.index].arrive(
                    n_bytes=self.kv_tx_bytes_per_warp, pred=prims.elect_sync()
                )
                if prims.elect_sync():
                    self._gather_fill(
                        tma_v_desc,
                        sKV_raw,
                        kv_state.index,
                        bars.mb_kv_full[kv_state.index].smem_ptr,
                        w,
                        blks_cur,
                        pages_cur,
                        kv_row_base,
                        v_rows_per_page,
                        head_rows_v,
                        eff_seqlen_kv,
                        col_v,
                        hint,
                    )
                kv_state.advance()
                blks_cur = blks_next
                pages_cur = pages_next

            # The last read of this item's list slot (row 4: ONE lane per consuming warp).
            if prims.elect_sync():
                bars.mb_ids_empty[ids_state.index].arrive()
            ids_state.advance()

            prims.bar_warp_sync(cute.arch.FULL_MASK)
            wait(
                sched.mb_scheduler.subview(sched_state.index),
                sched_state.phase,
            )
            t0, t1, nxt_v = read_clc_payload(
                sched, sched_state.index * cutlass.Int32(8), warp_broadcast=False
            )
            tok, head, batch, item_row, kv_row_base, eff_seqlen_q, _ot, _ob, split = (
                self._decode_item(
                    t0, t1, seq_kv_lens_tensor, n_batch, seqlen_q, seqlen_kv
                )
            )
            is_valid = cute.arch.make_warp_uniform(nxt_v)
            sched_state.advance()

        # Exit drain (P6): the MMA warp's last STAGES_KV kv_empty commits need a waiter; the carried state names their phases.
        for _ in cutlass.range_constexpr(self.stages_kv):
            bars.mb_kv_empty[kv_state.index].wait(kv_state.phase)
            kv_state.advance()

    @cute.jit
    def _mma_warp_group(
        self,
        sQ,
        sK,
        sVt,
        sP,
        tmem_ptr_i32,
        bars,
        sched,
        block_lens_tensor,
        seq_kv_lens_tensor,
        n_batch,
        seqlen_q,
        seqlen_kv,
        t0_init,
        t1_init,
        use_pdl: cutlass.Constexpr[bool],
    ):
        prims.tcgen05_alloc(
            tmem_ptr_i32, cutlass.Int32(self.tmem_cols), group=prims.CTAGroup.CTA_1
        )
        prims.tcgen05_relinquish_alloc_permit(group=prims.CTAGroup.CTA_1)
        prims.bar_warp_sync(cute.arch.FULL_MASK)
        prims.barrier_cta_arrive(self.bar_tmem, self.bar_tmem_threads)

        tmem_raw = prims.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)

        idesc_qk = prims.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32,
            a_dtype=self.storage_dtype,
            b_dtype=self.storage_dtype,
            n_dim=self.n_q,
            m_dim=self.tile_n,
            k_dim=0,
        )
        idesc_pv = prims.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32,
            a_dtype=self.storage_dtype,
            b_dtype=self.storage_dtype,
            n_dim=self.n_q,
            m_dim=128,
            a_major=1,
            b_major=1,
            k_dim=0,
        )
        # BMM1: S^T = K . Q^T -- K [128 keys x TILE_K] K-major, Q^T [N_Q x TILE_K] K-major.
        bmm1_desc = MmaDesc(
            M=self.tile_n,
            N=self.n_q,
            K=self.tile_k,
            bpe_a=self.bpe,
            bpe_b=self.bpe,
            tile_k_hw=self.tile_k_hw,
            btranspose=False,
            idesc=idesc_qk,
            kind=self.mma_kind,
            atranspose=False,
        )
        # BMM2: O^T = V^T . P^T -- V^T [128 d x 128 keys] MN-major (d contiguous), P^T [N_Q x 128 keys] MN-major.
        bmm2_desc = MmaDesc(
            M=128,
            N=self.n_q,
            K=self.tile_n,
            bpe_a=self.bpe,
            bpe_b=self.bpe,
            tile_k_hw=self.tile_k_hw,
            atranspose=True,
            btranspose=True,
            idesc=idesc_pv,
            kind=self.mma_kind,
        )

        # Every ring slot and parity below comes from a PipelineState CARRIED across work items (barrier rows 1, 2, 5-10):
        #   q_state      -- mb_q_full wait (consumer) and the slot of the mb_q_empty commit, once per item;
        #   kv_state     -- mb_kv_full wait + the slot of the mb_kv_empty commit, per LOAD in the gather warps' order;
        #   s_state      -- the S^T slot of BMM1(t) = the mb_s_full commit slot, per tile;
        #   s_empty_state-- mb_s_empty wait (pre-armed: BMM1(0), BMM1(1) pass fresh), per tile;
        #   p_state      -- mb_p_full wait, per tile;  bmm2_state -- the mb_bmm2_done commit slot, per tile.
        q_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.stages_q
        )
        kv_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.stages_kv
        )
        s_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.stages_s
        )
        s_empty_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.stages_s
        )
        p_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.stages_s
        )
        bmm2_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.stages_s
        )

        if cutlass.const_expr(use_pdl):
            # PDL (header, PDL SITES): the TMEM allocation, the named-barrier hand-off and the descriptor builds above are this role's
            # prologue; its only global reads are _item_bounds' lengths / counts, below (the operands arrive through SMEM).
            cute.arch.griddepcontrol_wait()
        tok, head, batch, item_row, _kvb, eff_seqlen_q, _ot, _ob, split = (
            self._decode_item(
                t0_init, t1_init, seq_kv_lens_tensor, n_batch, seqlen_q, seqlen_kv
            )
        )
        is_valid = cutlass.Int32(1)
        sched_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.scheduler_stages
        )
        while is_valid > cutlass.Int32(0):
            read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.index))
            _pos, _count, _n_tail, _tail_lo, _dead, n_tiles, _ekv, _tile0 = (
                self._item_bounds(
                    tok,
                    batch,
                    item_row,
                    eff_seqlen_q,
                    block_lens_tensor,
                    seq_kv_lens_tensor,
                    seqlen_kv,
                    split,
                )
            )

            bars.mb_q_full[q_state.index].wait(q_state.phase)
            desc_Q = sQ[q_state.index].desc()

            # Prologue: BMM1(0).  The item's LAST BMM1 (here when n_tiles == 1) also commits mb_q_empty (row 2): the commit tracks
            # every prior MMA of this thread, so it fires once the last read of sQ[slot] is done.
            bars.mb_kv_full[kv_state.index].wait(kv_state.phase)
            bars.mb_s_empty[s_empty_state.index].wait(s_empty_state.phase)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            mma_ss(
                bmm1_desc,
                sK[kv_state.index].desc(),
                desc_Q,
                tmem_raw.subview(
                    cutlass.Int32(self.s_acc_off[0])
                    + s_state.index * cutlass.Int32(self.n_q)
                ),
                accumulate=False,
            )
            elect_p = prims.elect_sync()
            bars.mb_s_full[s_state.index].arrive(pred=elect_p)
            bars.mb_kv_empty[kv_state.index].arrive(pred=elect_p)
            bars.mb_q_empty[q_state.index].arrive(
                pred=elect_p & (n_tiles == cutlass.Int32(1))
            )
            kv_state.advance()
            s_state.advance()
            s_empty_state.advance()

            for i in cutlass.range(0, n_tiles, 1, unroll=1):
                # BMM1(i + 1) ahead of BMM2(i): the softmax of tile i overlaps the next score tile, and the K slot is released as
                # early as possible.
                if i + cutlass.Int32(1) < n_tiles:
                    bars.mb_kv_full[kv_state.index].wait(kv_state.phase)
                    bars.mb_s_empty[s_empty_state.index].wait(s_empty_state.phase)
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    mma_ss(
                        bmm1_desc,
                        sK[kv_state.index].desc(),
                        desc_Q,
                        tmem_raw.subview(
                            cutlass.Int32(self.s_acc_off[0])
                            + s_state.index * cutlass.Int32(self.n_q)
                        ),
                        accumulate=False,
                    )
                    elect_p1 = prims.elect_sync()
                    bars.mb_s_full[s_state.index].arrive(pred=elect_p1)
                    bars.mb_kv_empty[kv_state.index].arrive(pred=elect_p1)
                    bars.mb_q_empty[q_state.index].arrive(
                        pred=elect_p1 & ((i + cutlass.Int32(2)) == n_tiles)
                    )
                    kv_state.advance()
                    s_state.advance()
                    s_empty_state.advance()

                # BMM2(i): V(i) is the next load of the ring, P^T(i) the softmax's publish.
                bars.mb_kv_full[kv_state.index].wait(kv_state.phase)
                bars.mb_p_full[p_state.index].wait(p_state.phase)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                desc_P = sP[p_state.index].desc()
                accum = cutlass.Boolean(i > cutlass.Int32(0))
                for blk in cutlass.range_constexpr(self.o_blocks):
                    mma_ss(
                        bmm2_desc,
                        sVt[kv_state.index]
                        .shifted(blk * 2 * self.tile_n * self.qk_granu_elems)
                        .desc(),
                        desc_P,
                        tmem_raw.subview(cutlass.Int32(self.o_off[blk])),
                        accumulate=accum,
                    )
                elect_p2 = prims.elect_sync()
                bars.mb_bmm2_done[bmm2_state.index].arrive(pred=elect_p2)
                bars.mb_kv_empty[kv_state.index].arrive(pred=elect_p2)
                kv_state.advance()
                p_state.advance()
                bmm2_state.advance()

            q_state.advance()

            prims.bar_warp_sync(cute.arch.FULL_MASK)
            wait(
                sched.mb_scheduler.subview(sched_state.index),
                sched_state.phase,
            )
            t0, t1, nxt_v = read_clc_payload(
                sched, sched_state.index * cutlass.Int32(8), warp_broadcast=False
            )
            tok, head, batch, item_row, _kvb, eff_seqlen_q, _ot, _ob, split = (
                self._decode_item(
                    t0, t1, seq_kv_lens_tensor, n_batch, seqlen_q, seqlen_kv
                )
            )
            is_valid = cute.arch.make_warp_uniform(nxt_v)
            sched_state.advance()

        bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
        prims.tcgen05_dealloc(
            prims.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8),
            cutlass.Int32(self.tmem_cols),
            group=prims.CTAGroup.CTA_1,
        )

    @cute.jit
    def _softmax_warp_group(
        self,
        tmem_ptr_i32,
        bars,
        sched,
        red_smem,
        sP_raw,
        sIds_raw,
        o_tensor,
        lse_tensor: Optional[cute.Tensor],
        block_lens_tensor: Optional[cute.Tensor],
        seq_kv_lens_tensor,
        n_batch,
        seqlen_q,
        seqlen_kv,
        t0_init,
        t1_init,
        scale_log2: cutlass.Float32,
        block_table_tensor,
        max_pages,
        num_pages,
        use_pdl: cutlass.Constexpr[bool],
    ):
        prims.barrier_cta_sync(
            barrier_id=self.bar_tmem, thread_count=self.bar_tmem_threads
        )
        tidx = cute.arch.thread_idx()[0]
        sm_warp = tidx // cutlass.Int32(32)  # 0 .. SOFTMAX_WARPS - 1 (all groups)
        # The column group (warp-uniform) and its first column; the key row of S^T / P^T = the d row of O^T is the lane WITHIN the group.
        # At ONE group the three are the constants 0 / 0 / tidx, so that rendering traces today's program.
        grp = (
            (sm_warp // cutlass.Int32(self.softmax_warps_per_group))
            if cutlass.const_expr(self.softmax_groups > 1)
            else cutlass.Int32(0)
        )
        col0 = (
            (grp * cutlass.Int32(self.cols_per_group))
            if cutlass.const_expr(self.softmax_groups > 1)
            else cutlass.Int32(0)
        )
        lane = (
            (tidx & cutlass.Int32(self.group_lanes - 1))
            if cutlass.const_expr(self.softmax_groups > 1)
            else tidx
        )
        bar_softmax = (
            (cutlass.Int32(self.bar_softmax) + grp)
            if cutlass.const_expr(self.softmax_groups > 1)
            else self.bar_softmax
        )
        tmem_base = tmem_ptr_i32.load()
        red_ptr = Pointer(red_smem.data_ptr(), dtype=cutlass.Float32)

        NEG_INF = cutlass.Float32(float("-inf"))
        ZERO = cutlass.Float32(0.0)
        ONE = cutlass.Float32(1.0)
        LN2_F = cutlass.Float32(self.ln2)
        TINY = cutlass.Float32(1e-30)

        # Carried states (barrier rows 3, 7-10): t_state = the per-tile slot / parity of mb_s_full (wait), mb_s_empty and
        # mb_p_full (this group's arrives); bmm2_state = the mb_bmm2_done wait, one per tile, one iteration behind; ids_state =
        # the mb_ids_full wait and the mb_ids_empty arrive slot, once per item.  None is ever recomputed from a per-item index.
        t_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.stages_s
        )
        bmm2_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.stages_s
        )
        ids_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.stages_ids
        )

        if cutlass.const_expr(use_pdl):
            # PDL (header, PDL SITES): the TMEM hand-off above is this role's prologue; its global reads (_item_bounds' lengths / counts)
            # and every global WRITE of the role (O / the partial slabs, the LSE) are below this wait.
            cute.arch.griddepcontrol_wait()
        tok, head, batch, item_row, _kvb, eff_seqlen_q, out_tok, out_batch, split = (
            self._decode_item(
                t0_init, t1_init, seq_kv_lens_tensor, n_batch, seqlen_q, seqlen_kv
            )
        )
        is_valid = cutlass.Int32(1)
        sched_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.scheduler_stages
        )
        while is_valid > cutlass.Int32(0):
            read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.index))
            pos, count, n_tail, tail_lo, dead, n_tiles, eff_seqlen_kv, tile0 = (
                self._item_bounds(
                    tok,
                    batch,
                    item_row,
                    eff_seqlen_q,
                    block_lens_tensor,
                    seq_kv_lens_tensor,
                    seqlen_kv,
                    split,
                )
            )
            query_tok = out_tok
            query_col0 = col0
            if cutlass.const_expr(self.query_group_size == 2):
                # Two groups own each query's sixteen padded head columns.
                query_tok = out_tok + col0 // cutlass.Int32(self.query_head_tile)
                query_col0 = col0 % cutlass.Int32(self.query_head_tile)
                pos = query_tok + eff_seqlen_kv - eff_seqlen_q
            query_vis = cute.math.max(
                cute.math.min(pos + cutlass.Int32(1), eff_seqlen_kv), cutlass.Int32(0)
            )
            query_count = cute.math.min(
                cutlass.Int32(self.block_topk),
                query_vis // cutlass.Int32(self.block_size),
            )
            slot_base = ids_state.index * cutlass.Int32(self.ids_slot_words)
            bars.mb_ids_full[ids_state.index].wait(ids_state.phase)

            # Running column state: max in the scaled log2 domain (-inf = no live key yet), the thread's partial sum.
            m_vec = cutlass.Vector.from_elements(
                tuple(NEG_INF for _ in range(self.cols_per_group)), cutlass.Float32
            )
            l_vec = cutlass.Vector.from_elements(
                tuple(ZERO for _ in range(self.cols_per_group)), cutlass.Float32
            )

            for i in cutlass.range(0, n_tiles, 1, unroll=1):
                par = t_state.index
                s_off = cutlass.Int32(self.s_acc_off[0]) + par * cutlass.Int32(self.n_q)
                if cutlass.const_expr(self.softmax_groups > 1):
                    s_off = s_off + col0  # the group's COLS columns of the slot

                bars.mb_s_full[par].wait(t_state.phase)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                s_raw = prims.tcgen05_ld(
                    "32x32b",
                    prims.make_tmem_ptr(tmem_base + s_off, cutlass.Float32),
                    num=self.cols_per_group,
                )
                # The LOAD wait is what ORDERS the arrive after the read (frost-kernels.md section 3).
                prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                bars.mb_s_empty[par].arrive()

                # Each group shares a position; paired groups mask independently.
                # Lane l of tile i holds key 4 blk + l % 4 with blk = the list entry 32 (tile0 + i) + l // 4 (tail / -1 by the SELECT).
                t_abs = i
                if cutlass.const_expr(self.split_kv > 1):
                    t_abs = tile0 + i
                idx = t_abs * cutlass.Int32(self.blocks_per_tile) + (
                    lane >> cutlass.Int32(2)
                )
                ld_idx = cute.math.min(idx, cutlass.Int32(self.block_topk - 1))
                word = cutlass.Int32(sIds_raw.load(slot_base + ld_idx))
                blk = self._block_id_at(idx, word, count, n_tail, tail_lo, dead)
                key_abs = blk * cutlass.Int32(self.block_size) + (
                    lane & cutlass.Int32(self.block_size - 1)
                )
                valid = (
                    (blk >= cutlass.Int32(0))
                    & (key_abs <= pos)
                    & (key_abs < eff_seqlen_kv)
                )
                if cutlass.const_expr(self.query_group_size == 2):
                    # A completed block may be selected for only the later query;
                    # the appended tail belongs only to queries with an open block.
                    valid = valid & (
                        (idx < query_count)
                        | (
                            (idx == count)
                            & ((query_vis % cutlass.Int32(self.block_size)) != 0)
                        )
                    )
                page = self._physical_page(
                    blk,
                    cutlass.make_array_view(block_table_tensor),
                    batch,
                    max_pages,
                    num_pages,
                    eff_seqlen_kv,
                )
                valid = valid & (page > cutlass.Int32(0))
                s_cols = []
                for j in cutlass.range_constexpr(self.cols_per_group):
                    s_cols.append(
                        _select_f32(
                            valid, cutlass.Float32(s_raw[j]) * scale_log2, NEG_INF
                        )
                    )

                # Column max over the tile's 128 keys: butterfly inside the warp, then the group's four warps exchange through
                # SMEM (slot = the tile's ring parity, so the single barrier per tile also orders the next reuse of the slot;
                # the slot is [SOFTMAX_WARPS][COLS] and group g reads warps 4 g .. 4 g + 3 of it after ITS named barrier).
                red_base = par * cutlass.Int32(
                    self.softmax_warps * self.cols_per_group
                ) + sm_warp * cutlass.Int32(self.cols_per_group)
                col_max = []
                for j in cutlass.range_constexpr(self.cols_per_group):
                    cm = _warp_reduce_max(s_cols[j])
                    (red_ptr + (red_base + cutlass.Int32(j))).store(cm)
                    col_max.append(cm)
                prims.barrier_cta_sync(
                    barrier_id=bar_softmax, thread_count=self.bar_softmax_threads
                )
                red_read = par * cutlass.Int32(self.softmax_warps * self.cols_per_group)
                if cutlass.const_expr(self.softmax_groups > 1):
                    red_read = red_read + grp * cutlass.Int32(
                        self.softmax_warps_per_group * self.cols_per_group
                    )
                tile_max = []
                for j in cutlass.range_constexpr(self.cols_per_group):
                    tm = col_max[j]
                    for w in cutlass.range_constexpr(self.softmax_warps_per_group):
                        tm = cute.math.max(
                            tm,
                            cutlass.Float32(
                                (
                                    red_ptr
                                    + (
                                        red_read
                                        + cutlass.Int32(w * self.cols_per_group + j)
                                    )
                                ).load()
                            ),
                            ftz=True,
                        )
                    tile_max.append(tm)

                # Online update per column (identical in every lane: alpha is uniform, so O^T's rescale is exact).
                m_new = []
                l_new = []
                alpha = []
                all_one = None
                p_cols = []
                for j in cutlass.range_constexpr(self.cols_per_group):
                    m_old_j = cutlass.Float32(m_vec[j])
                    m_new_j = cute.math.max(m_old_j, tile_max[j], ftz=True)
                    ms_old = row_max_for_exp2(m_old_j)
                    ms_new = row_max_for_exp2(m_new_j)
                    alpha_j = cute.math.exp2(
                        cute.math.min(ms_old - ms_new, ZERO, ftz=True), fastmath=True
                    )
                    p_j = cute.math.exp2(s_cols[j] - ms_new, fastmath=True)
                    l_new.append(cutlass.Float32(l_vec[j]) * alpha_j + p_j)
                    m_new.append(m_new_j)
                    alpha.append(alpha_j)
                    p_cols.append(p_j)
                    is_one = alpha_j == ONE
                    all_one = is_one if all_one is None else (all_one & is_one)
                m_vec = cutlass.Vector.from_elements(tuple(m_new), cutlass.Float32)
                l_vec = cutlass.Vector.from_elements(tuple(l_new), cutlass.Float32)

                # P^T(t) -> sP[par] as packed half pairs, row = the thread's key, swizzled the way the BMM2 B descriptor decodes
                # it.  WAR on the slot: BMM2(t-2) last read it, and its completion was waited at iteration t-1 (below), BEFORE
                # this store -- the store-then-wait order inside an iteration is exactly sufficient (header row 10).
                p_words = []
                for w in cutlass.range_constexpr(self.p_group_words):
                    p_words.append(
                        cvt_f16x2_f32(
                            p_cols[2 * w],
                            p_cols[2 * w + 1],
                            to_dtype=self.storage_dtype,
                        )
                    )
                p_row_base = par * cutlass.Int32(
                    self.p_buffer_words
                ) + lane * cutlass.Int32(self.p_row_words)
                if cutlass.const_expr(self.softmax_groups > 1):
                    # Each group writes disjoint columns of the same swizzled row.
                    p_row_base = p_row_base + grp * cutlass.Int32(self.p_group_words)
                for c in cutlass.range_constexpr(self.p_chunks):
                    chunk = cutlass.Vector.from_elements(
                        tuple(p_words[4 * c + k] for k in range(4)), cutlass.Int32
                    )
                    p_ptr = Pointer(
                        sP_raw.subview(p_row_base + cutlass.Int32(4 * c)).data_ptr(),
                        dtype=cutlass.Int32,
                    )
                    p_ptr.store_swizzled(chunk, self.p_smem_swizzle, alignment=16)
                prims.fence_proxy("async.shared", space="cta")

                # O^T(t-1) is complete once BMM2(t-1) commits; rescale this group's columns by alpha before BMM2(t) accumulates
                # on top.  Skipped when no column's max moved.
                if i > cutlass.Int32(0):
                    bars.mb_bmm2_done[bmm2_state.index].wait(bmm2_state.phase)
                    bmm2_state.advance()
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    if ~all_one:
                        for blk_o in cutlass.range_constexpr(self.o_blocks):
                            o_col = cutlass.Int32(self.o_off[blk_o])
                            if cutlass.const_expr(self.softmax_groups > 1):
                                o_col = o_col + col0
                            o_ptr = prims.make_tmem_ptr(
                                tmem_base + o_col, cutlass.Float32
                            )
                            o_vals = prims.tcgen05_ld(
                                "32x32b", o_ptr, num=self.cols_per_group
                            )
                            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                            o_scaled = cutlass.Vector.from_elements(
                                tuple(
                                    cutlass.Float32(o_vals[j]) * alpha[j]
                                    for j in range(self.cols_per_group)
                                ),
                                cutlass.Float32,
                            )
                            prims.tcgen05_st("32x32b", o_ptr, o_scaled)
                        prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
                # Publish P^T(t) (and the rescaled O^T) to the MMA warp.
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                bars.mb_p_full[par].arrive()
                t_state.advance()

            # The last read of this item's list slot (row 4: ONE lane per consuming warp).
            if prims.elect_sync():
                bars.mb_ids_empty[ids_state.index].arrive()
            ids_state.advance()

            # --- epilogue: the item's last BMM2 must land before O^T is read (one wait per tile: this is the last tile's).
            bars.mb_bmm2_done[bmm2_state.index].wait(bmm2_state.phase)
            bmm2_state.advance()
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)

            # Column sums over the 128 lanes (once per item), via the epilogue slot.
            red_epi = cutlass.Int32(
                self.stages_s * self.softmax_warps * self.cols_per_group
            ) + sm_warp * cutlass.Int32(self.cols_per_group)
            for j in cutlass.range_constexpr(self.cols_per_group):
                (red_ptr + (red_epi + cutlass.Int32(j))).store(
                    _warp_reduce_sum(cutlass.Float32(l_vec[j]))
                )
            prims.barrier_cta_sync(
                barrier_id=bar_softmax, thread_count=self.bar_softmax_threads
            )
            red_epi_read = cutlass.Int32(
                self.stages_s * self.softmax_warps * self.cols_per_group
            )
            if cutlass.const_expr(self.softmax_groups > 1):
                red_epi_read = red_epi_read + grp * cutlass.Int32(
                    self.softmax_warps_per_group * self.cols_per_group
                )

            lse_cols = []
            inv_cols = []
            dead_cols = []
            for j in cutlass.range_constexpr(self.cols_per_group):
                l_tot = ZERO
                for w in cutlass.range_constexpr(self.softmax_warps_per_group):
                    l_tot = l_tot + cutlass.Float32(
                        (
                            red_ptr
                            + (
                                red_epi_read
                                + cutlass.Int32(w * self.cols_per_group + j)
                            )
                        ).load()
                    )
                m_raw = cutlass.Float32(
                    m_vec[j]
                )  # -inf while the column has no live key
                m_nat = row_max_for_exp2(m_raw) * LN2_F
                row_dead = l_tot <= ZERO
                # The denominator floor sits inside the log and the reciprocal ONLY and is always followed by the row_dead
                # SELECT, so neither LSE nor O ever carries it (sdpa-invariants.md sections 2-4): keep these lines together.
                lse_j = m_nat + cute.math.log(cute.math.max(l_tot, TINY), fastmath=True)
                inv_j = ONE / cute.math.max(l_tot, TINY)
                lse_j = _select_f32(row_dead, NEG_INF, lse_j)
                inv_j = _select_f32(row_dead, ZERO, inv_j)
                lse_cols.append(lse_j)
                inv_cols.append(inv_j)
                dead_cols.append(row_dead)

            # Paired groups store to separate query rows. Padded head columns
            # never store; split partials retain the existing split-major layout.
            live_row = (
                query_tok < eff_seqlen_q
            )  # the row's liveness is the Q row's, not its key-space position's
            part_batch = out_batch
            if cutlass.const_expr(self.split_kv > 1):
                part_batch = out_batch + split * n_batch
            head_base = head * cutlass.Int32(self.qh_per_kh)
            # query_col0 is relative to the group's query, not the full MMA tile.
            n_live_cols = (
                (cutlass.Int32(self.q_box_rows) - query_col0)
                if cutlass.const_expr(self.softmax_groups > 1)
                else cutlass.Int32(self.q_box_rows)
            )
            if cutlass.const_expr(lse_tensor is not None):
                lse_arr = cutlass.make_array_view(lse_tensor)
                for j in cutlass.range_constexpr(self.cols_per_group):
                    if cutlass.const_expr(
                        self.softmax_groups > 1 or j < self.q_box_rows
                    ):
                        col = (
                            (col0 + cutlass.Int32(j))
                            if cutlass.const_expr(self.softmax_groups > 1)
                            else cutlass.Int32(j)
                        )
                        ok = (
                            (
                                (lane == col)
                                & live_row
                                & (cutlass.Int32(j) < n_live_cols)
                            )
                            if cutlass.const_expr(self.softmax_groups > 1)
                            else ((lane == col) & live_row)
                        )
                        if ok:
                            lse_arr[
                                part_batch,
                                head_base + query_col0 + cutlass.Int32(j),
                                query_tok,
                            ] = lse_cols[j]

            # O[q, d] = O^T[d, q] / l[q]: the thread index is d (and d + 128).
            oo = cutlass.make_array_view(o_tensor)
            D_V = cutlass.const_expr(o_tensor.shape[3])
            for blk_o in cutlass.range_constexpr(self.o_blocks):
                if cutlass.const_expr(blk_o * 128 < D_V):
                    d_idx = lane + cutlass.Int32(blk_o * 128)
                    o_col_e = cutlass.Int32(self.o_off[blk_o])
                    if cutlass.const_expr(self.softmax_groups > 1):
                        o_col_e = o_col_e + col0
                    o_vals = prims.tcgen05_ld(
                        "32x32b",
                        prims.make_tmem_ptr(tmem_base + o_col_e, cutlass.Float32),
                        num=self.cols_per_group,
                    )
                    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                    if (d_idx < cutlass.Int32(D_V)) & live_row:
                        for j in cutlass.range_constexpr(self.cols_per_group):
                            if cutlass.const_expr(
                                self.softmax_groups > 1 or j < self.q_box_rows
                            ):
                                val = _select_f32(
                                    dead_cols[j],
                                    ZERO,
                                    cutlass.Float32(o_vals[j]) * inv_cols[j],
                                )
                                if cutlass.const_expr(self.softmax_groups > 1):
                                    if (
                                        cutlass.Int32(j) < n_live_cols
                                    ):  # the group's columns at or past G are the zero tail: never stored
                                        oo[
                                            cutlass.Int64(part_batch),
                                            cutlass.Int64(query_tok),
                                            cutlass.Int64(
                                                head_base
                                                + query_col0
                                                + cutlass.Int32(j)
                                            ),
                                            cutlass.Int64(d_idx),
                                        ] = val.to(o_tensor.element_type)
                                else:
                                    oo[
                                        cutlass.Int64(part_batch),
                                        cutlass.Int64(query_tok),
                                        cutlass.Int64(head_base + cutlass.Int32(j)),
                                        cutlass.Int64(d_idx),
                                    ] = val.to(o_tensor.element_type)

            prims.bar_warp_sync(cute.arch.FULL_MASK)
            wait(
                sched.mb_scheduler.subview(sched_state.index),
                sched_state.phase,
            )
            t0, t1, nxt_v = read_clc_payload(
                sched, sched_state.index * cutlass.Int32(8), warp_broadcast=False
            )
            (
                tok,
                head,
                batch,
                item_row,
                _kvb,
                eff_seqlen_q,
                out_tok,
                out_batch,
                split,
            ) = self._decode_item(
                t0, t1, seq_kv_lens_tensor, n_batch, seqlen_q, seqlen_kv
            )
            is_valid = cute.arch.make_warp_uniform(nxt_v)
            sched_state.advance()

        bars.mb_tmem_dealloc.arrive()

    @cute.kernel
    def _kernel(
        self,
        tma_q_desc: cutlass.GridConstant[tmap.TensorMap],
        tma_k_desc: cutlass.GridConstant[tmap.TensorMap],
        tma_v_desc: cutlass.GridConstant[tmap.TensorMap],
        o_tensor: cute.Tensor,
        lse_tensor: Optional[cute.Tensor],
        block_ids_tensor: cute.Tensor,
        block_lens_tensor: Optional[cute.Tensor],
        seq_kv_lens_tensor: cute.Tensor,
        seqlen_q: cutlass.Int32,
        seqlen_kv: cutlass.Int32,
        k_head_stride: cutlass.Int32,
        v_head_stride: cutlass.Int32,
        scale_softmax_log2: cutlass.Float32,
        block_table_tensor: Optional[cute.Tensor],
        max_pages: cutlass.Int32,
        num_pages: cutlass.Int32,
        k_rows_per_page: cutlass.Int32,
        k_rows_per_head: cutlass.Int32,
        v_rows_per_page: cutlass.Int32,
        v_rows_per_head: cutlass.Int32,
        n_batch: cutlass.Int32,
        use_pdl: cutlass.Constexpr[bool],
    ) -> None:
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx = cute.arch.block_idx()[0]
        bidy = cute.arch.block_idx()[1]
        bidz = cute.arch.block_idx()[2]
        t0_init = bidx
        t1_init = bidy | (bidz << cutlass.Int32(16))

        # SMEM in the header table's order: the descriptor-read operands first (1024-B aligned), the register-addressed buffers last.
        sQ_raw = cutlass.Array(
            self.storage_dtype,
            self.stages_q * self.q_buffer_elems,
            alignment=1024,
            space=cutlass.AddressSpace.smem,
        )
        sP_raw = cutlass.Array(
            cutlass.Int32,
            self.stages_s * self.p_buffer_words,
            alignment=1024,
            space=cutlass.AddressSpace.smem,
        )
        sKV_raw = cutlass.Array(
            self.storage_dtype,
            self.stages_kv * self.kv_buffer_elems,
            alignment=1024,
            space=cutlass.AddressSpace.smem,
        )
        sIds_raw = cutlass.Array(
            cutlass.Int32,
            self.stages_ids * self.ids_slot_words,
            alignment=16,
            space=cutlass.AddressSpace.smem,
        )
        red_smem = cutlass.Array(
            cutlass.Float32,
            self.red_words,
            alignment=16,
            space=cutlass.AddressSpace.smem,
        )
        tmem_ptr_i32 = cutlass.Array(
            cutlass.Int32, 1, alignment=16, space=cutlass.AddressSpace.smem
        )

        # Q^T: the K-major B operand of BMM1 (N_Q rows x TILE_K per slot), one TMA sub-tile (N_Q rows x 128 B) per 64-wide granule.
        sQ = SmemTile(
            base=sQ_raw,
            elems_per_stage=self.q_buffer_elems,
            leading_byte_offset=0,
            stride_byte_offset=self.stride_byte_offset_qk,
            layout=self.smem_layout_qk,
        )
        # The gathered K/V ring, seen as the K-major A operand of BMM1 (TILE_N rows x TILE_K) ...
        sK = SmemTile(
            base=sKV_raw,
            elems_per_stage=self.kv_buffer_elems,
            leading_byte_offset=0,
            stride_byte_offset=self.stride_byte_offset_qk,
            layout=self.smem_layout_qk,
        )
        # ... and as the MN-major A operand of BMM2 (V^T: d contiguous, LBO steps the 64-wide d atoms).  Same bytes, two
        # descriptors -- the gather map writes under the same SW128 the descriptors decode.
        sVt = SmemTile(
            base=sKV_raw,
            elems_per_stage=self.kv_buffer_elems,
            leading_byte_offset=self.leading_byte_offset_vt,
            stride_byte_offset=self.stride_byte_offset_vt,
            layout=self.smem_layout_qk,
        )
        # P^T: the MN-major B operand of BMM2, STAGES_S buffers of TILE_N key rows.
        sP = SmemTile(
            base=sP_raw,
            elems_per_stage=self.p_buffer_words,
            leading_byte_offset=0,
            stride_byte_offset=self.stride_byte_offset_p,
            layout=self.smem_layout_p,
        )

        bars = self._make_sparse_bars()
        sched = Sched(
            mb_scheduler=cutlass.Array(
                cutlass.Int64,
                self.scheduler_stages,
                alignment=16,
                space=cutlass.AddressSpace.smem,
            ),
            mb_read_tile_id=cutlass.Array(
                cutlass.Int64,
                self.scheduler_stages,
                alignment=16,
                space=cutlass.AddressSpace.smem,
            ),
            tile_id_smem=cutlass.Array(
                cutlass.Int32,
                self.scheduler_stages * 8,
                alignment=16,
                space=cutlass.AddressSpace.smem,
            ),
            bidx_init=bidx,
            bidy_init=bidy,
            bidz_init=bidz,
        )

        # P4: ONE warp, ONE lane initialises every stage of every ring (.init() is per stage); fence + CTA sync outside.
        if warp_idx == 0:
            if prims.elect_sync():
                for s in cutlass.range_constexpr(self.stages_q):
                    bars.mb_q_full[s].init()
                    bars.mb_q_empty[s].init()
                for s in cutlass.range_constexpr(self.stages_ids):
                    bars.mb_ids_full[s].init()
                    bars.mb_ids_empty[s].init()
                for s in cutlass.range_constexpr(self.stages_kv):
                    bars.mb_kv_full[s].init()
                    bars.mb_kv_empty[s].init()
                for p in cutlass.range_constexpr(self.stages_s):
                    bars.mb_s_full[p].init()
                    bars.mb_s_empty[p].init()
                    bars.mb_p_full[p].init()
                    bars.mb_bmm2_done[p].init()
                bars.mb_tmem_dealloc.init()
                for s in cutlass.range_constexpr(self.scheduler_stages):
                    prims.mbarrier_init(sched.mb_scheduler.subview(s), self.one_lane)
                    prims.mbarrier_init(
                        sched.mb_read_tile_id.subview(s), self.read_tile_arrivers
                    )

        if cutlass.const_expr(self.q_box_rows < self.n_q):
            # Zero from the first query's padding to the end of each subtile.
            # Paired Q loads overwrite the second query's real rows before MMA;
            # both queries' padded rows remain zero across persistent items.
            if warp_idx < self.softmax_warps:
                zero4 = cutlass.Vector.from_elements(
                    tuple(cutlass.Int32(0) for _ in range(4)), cutlass.Int32
                )
                tail_chunks = (
                    self.n_q - self.q_box_rows
                ) * 8  # 16 B chunks per sub-tile
                n_chunks = self.stages_q * self.qk_iters * tail_chunks
                for it in cutlass.range_constexpr(
                    (n_chunks + self.softmax_lanes - 1) // self.softmax_lanes
                ):
                    chunk = cutlass.Int32(it * self.softmax_lanes) + tidx
                    if chunk < cutlass.Int32(n_chunks):
                        slot = chunk // cutlass.Int32(self.qk_iters * tail_chunks)
                        rem = chunk % cutlass.Int32(self.qk_iters * tail_chunks)
                        sub = rem // cutlass.Int32(tail_chunks)
                        within = rem % cutlass.Int32(tail_chunks)
                        elem_off = (
                            slot * cutlass.Int32(self.q_buffer_elems)
                            + sub * cutlass.Int32(self.n_q * self.qk_granu_elems)
                            + cutlass.Int32(self.q_box_rows * self.qk_granu_elems)
                            + within * cutlass.Int32(8)
                        )
                        Pointer(
                            sQ_raw.subview(elem_off).data_ptr(), dtype=cutlass.Int32
                        ).store(zero4, alignment=16)
                prims.fence_proxy("async.shared", space="cta")

        prims.fence_mbarrier_init()
        prims.barrier_cta_sync()
        # No CTA-wide barrier-0 sync may follow this one: the spare warp exits here.
        # PDL (header, PDL SITES): NO release here, or anywhere -- the CTA exit is the trigger of the dependent grid (a release at this
        # point made the split combine's CTAs resident through the core's whole run and cost 3.2-4.4 us per step at B <= 4 on the cc 10.7
        # class, the measured placement); each role's own wait follows in its body.

        if warp_idx < self.softmax_warps:
            self._softmax_warp_group(
                tmem_ptr_i32=tmem_ptr_i32,
                bars=bars,
                sched=sched,
                red_smem=red_smem,
                sP_raw=sP_raw,
                sIds_raw=sIds_raw,
                o_tensor=o_tensor,
                lse_tensor=lse_tensor,
                block_lens_tensor=block_lens_tensor,
                seq_kv_lens_tensor=seq_kv_lens_tensor,
                n_batch=n_batch,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                t0_init=t0_init,
                t1_init=t1_init,
                scale_log2=scale_softmax_log2,
                block_table_tensor=block_table_tensor,
                max_pages=max_pages,
                num_pages=num_pages,
                use_pdl=use_pdl,
            )
        elif warp_idx < self.gather_warp_base + self.gather_warps:
            self._gather_warp_group(
                tma_k_desc=tma_k_desc,
                tma_v_desc=tma_v_desc,
                sKV_raw=sKV_raw,
                sIds_raw=sIds_raw,
                bars=bars,
                sched=sched,
                block_lens_tensor=block_lens_tensor,
                seq_kv_lens_tensor=seq_kv_lens_tensor,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                k_head_stride=k_head_stride,
                v_head_stride=v_head_stride,
                w=warp_idx - cutlass.Int32(self.gather_warp_base),
                n_batch=n_batch,
                t0_init=t0_init,
                t1_init=t1_init,
                block_table_tensor=block_table_tensor,
                max_pages=max_pages,
                num_pages=num_pages,
                k_rows_per_page=k_rows_per_page,
                k_rows_per_head=k_rows_per_head,
                v_rows_per_page=v_rows_per_page,
                v_rows_per_head=v_rows_per_head,
                use_pdl=use_pdl,
            )
        elif warp_idx == self.mma_warp_id:
            self._mma_warp_group(
                sQ=sQ,
                sK=sK,
                sVt=sVt,
                sP=sP,
                tmem_ptr_i32=tmem_ptr_i32,
                bars=bars,
                sched=sched,
                block_lens_tensor=block_lens_tensor,
                seq_kv_lens_tensor=seq_kv_lens_tensor,
                n_batch=n_batch,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                t0_init=t0_init,
                t1_init=t1_init,
                use_pdl=use_pdl,
            )
        elif warp_idx == self.tmaldg_warp_id:
            prims.prefetch_tensormap(tma_q_desc.get_ptr())
            prims.prefetch_tensormap(tma_k_desc.get_ptr())
            prims.prefetch_tensormap(tma_v_desc.get_ptr())
            self._tmaldg_warp_group(
                tma_q_desc=tma_q_desc,
                sQ=sQ,
                sIds_raw=sIds_raw,
                bars=bars,
                sched=sched,
                block_ids_tensor=block_ids_tensor,
                seq_kv_lens_tensor=seq_kv_lens_tensor,
                n_batch=n_batch,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                t0_init=t0_init,
                t1_init=t1_init,
                use_pdl=use_pdl,
            )
        elif warp_idx == self.sched_warp_id:
            scheduler_warp_loop(sched, self.scheduler_stages)

    @cute.jit
    def __call__(
        self,
        q_ptr: cute.Pointer,
        k_ptr: cute.Pointer,
        v_ptr: cute.Pointer,
        o_ptr: cute.Pointer,
        lse_ptr: Optional[cute.Pointer],
        block_ids_ptr: cute.Pointer,
        block_lens_ptr: Optional[cute.Pointer],
        seq_kv_lens_ptr: cute.Pointer,
        problem_size: Tuple[int, int, int, int, int],
        q_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
        k_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
        v_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
        o_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
        lse_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
        scale_softmax_log2: cutlass.Float32,
        block_table_ptr: Optional[cute.Pointer],
        paged_geom: Tuple[int, int, int, int, int, int, int],
        use_pdl: cutlass.Constexpr[bool],
        stream: _cuda_driver.CUstream,
    ) -> None:
        """Bind the explicit pointer ABI and launch the sparse core."""
        B, QH, KH, SQ, SKV = problem_size
        (
            num_pages,
            max_pages,
            k_rows_per_page,
            k_rows_per_head,
            v_rows_per_page,
            v_rows_per_head,
            table_stride,
        ) = paged_geom
        n_o_batches = (
            B * self.split_kv
        )  # Split-major partial slabs use batch index b + split * B.
        q_tensor = _bshd(q_ptr, B, SQ, QH, self.tile_k, q_strides)
        o_tensor = _bshd(o_ptr, n_o_batches, SQ, QH, self.tile_o, o_strides)
        lse_tensor = None
        if cutlass.const_expr(lse_ptr is not None):
            l0, l1, l2 = lse_strides
            lse_tensor = cute.make_tensor(
                lse_ptr, cute.make_layout((n_o_batches, QH, SQ), stride=(l0, l1, l2))
            )
        # Each query owns one row of selected logical blocks.
        n_list_rows = B * SQ
        block_ids_tensor = _vec(block_ids_ptr, n_list_rows * self.block_topk)
        block_lens_tensor = None
        if cutlass.const_expr(block_lens_ptr is not None):
            block_lens_tensor = _vec(block_lens_ptr, n_list_rows)
        seq_kv_lens_tensor = _vec(seq_kv_lens_ptr, B)
        block_table_tensor = None
        if cutlass.const_expr(block_table_ptr is not None):
            block_table_tensor = cute.make_tensor(
                block_table_ptr,
                cute.make_layout((B, max_pages), stride=(table_stride, 1)),
            )

        def _tma_swz(byte_w: int):
            return (
                tmap.TensorMapSwizzle.s128b
                if byte_w == 128
                else (
                    tmap.TensorMapSwizzle.s64b
                    if byte_w == 64
                    else tmap.TensorMapSwizzle.s32b
                )
            )

        # Q^T box: ONE token x G heads of one KV group (row = head within the group), the 4-D map over (D, QH, SQ, B).
        stride_order = (3, 2, 1, 0)
        box_q = (1, self.q_box_tokens, self.q_box_rows, self.qk_granu_elems)
        tma_q_desc = tmap.create_tensor_map_tiled(
            global_address=q_tensor.iterator.toint(),
            dtype=q_tensor.element_type,
            global_dims=tuple(q_tensor.shape[i] for i in stride_order),
            global_strides=tuple(
                cutlass.Int64(q_tensor.stride[i]) * q_tensor.element_type.width // 128
                for i in stride_order[1:]
            ),
            box_dims=tuple(box_q[i] for i in stride_order),
            swizzle=_tma_swz(self.q_swz_bytes),
            l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
        )

        # K / V gather maps: ONE 2-D map per operand, rows = the tokens of every batch (B x SKV, pitch = the token stride;
        # PAGED arm: the pool's token rows, num_pages x rows_per_page, at the same pitch), columns = the token's full row
        # ((KH - 1) x head stride + D elements, contiguous); box (64 elements, 1 row) = one 128-B SW128 span.  The head and
        # the column box live in the COLUMN coordinate (gather warps: head x head_stride + box x 64), the batch / page term in
        # the ROW coordinate (batch x SKV + 4 blk + r; page x rows_per_page + head x rows_per_head + (4 blk) % PAGE_SIZE + r).
        def _gather_desc(ptr, strides, rows_per_page):
            _bs, ss, hs = strides
            rows = cutlass.Int32(num_pages) * cutlass.Int32(rows_per_page)
            inner = (cutlass.Int32(KH) - cutlass.Int32(1)) * cutlass.Int32(
                hs
            ) + cutlass.Int32(self.tile_k)
            return tmap.create_tensor_map_tiled(
                global_address=ptr.toint(),
                dtype=self.storage_dtype,
                global_dims=(inner, rows),
                global_strides=(cutlass.Int64(ss) * self.bpe // 16,),
                box_dims=(self.gather_box_elems, 1),
                swizzle=_tma_swz(self.k_swz_bytes),
                l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
            )

        tma_k_desc = _gather_desc(k_ptr, k_strides, k_rows_per_page)
        tma_v_desc = _gather_desc(v_ptr, v_strides, v_rows_per_page)

        grid_shape = (SQ // self.query_group_size * self.split_kv, KH, B)
        self._kernel(
            tma_q_desc,
            tma_k_desc,
            tma_v_desc,
            o_tensor,
            lse_tensor,
            block_ids_tensor,
            block_lens_tensor,
            seq_kv_lens_tensor,
            cutlass.Int32(SQ),
            cutlass.Int32(SKV),
            cutlass.Int32(k_strides[2]),
            cutlass.Int32(v_strides[2]),
            scale_softmax_log2,
            block_table_tensor,
            cutlass.Int32(max_pages),
            cutlass.Int32(num_pages),
            cutlass.Int32(k_rows_per_page),
            cutlass.Int32(k_rows_per_head),
            cutlass.Int32(v_rows_per_page),
            cutlass.Int32(v_rows_per_head),
            cutlass.Int32(B),
            use_pdl,
        ).launch(
            grid=grid_shape,
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
            use_pdl=use_pdl,  # the programmatic-stream-serialization attribute, paired with the kernel's griddepcontrol sites
        )

    def compile(self, *, enable_pdl: bool) -> Callable:
        """Compile this tactic with live shapes, strides and metadata operands."""
        gmem = cute.AddressSpace.gmem

        def P(dtype, align=16):
            return cute.runtime.make_ptr(
                dtype, 16, gmem, assumed_align=align
            )  # fake: type only

        i64_3 = (cutlass.Int64(0),) * 3
        return cute.compile(
            self,
            P(self.storage_dtype),
            P(self.storage_dtype),
            P(self.storage_dtype),
            (
                P(cutlass.Float32) if self.split_kv > 1 else P(self.storage_dtype)
            ),  # the fp32 partial slab under a split, else the caller's O
            P(cutlass.Float32, 4) if self.split_kv > 1 else None,
            P(cutlass.Int32, 16),
            None,
            P(cutlass.Int32, 4),
            (0, 0, 0, 0, 0),
            i64_3,
            i64_3,
            i64_3,
            i64_3,
            i64_3,
            cutlass.Float32(0.0),
            P(cutlass.Int32, 4),
            (0, 0, 0, 0, 0, 0, 0),
            enable_pdl,
            stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )
