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

"""Full-history BF16 paged DSA scores on Hopper."""

from functools import cache

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90
import torch
from cuda.bindings.driver import CUstream
from cutlass.cute.runtime import from_dlpack
from cutlass.tensor_utils import LayoutEnum
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

SM90_WGMMA_M = 64
SM90_WGMMA_K = 16  # BF16
_THREADS_PER_CTA = 256


@register_kernel(
    "attention",
    "dsa_paged_score",
    name="cute_dsl_dsa_paged_score_sm90",
    solution="cute_dsl",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(9, 0),
        max_arch_version=ArchVersion(9, 0),
    ),
    signatures=frozenset(
        {
            format_signature(
                q=dense_tensor_format(torch.bfloat16),
                index_k_cache=dense_tensor_format(torch.bfloat16),
                weights=dense_tensor_format(torch.float32),
                block_table=dense_tensor_format(torch.int32),
                context_lens=dense_tensor_format(torch.int32),
            )
        }
    ),
    traits={
        "index_heads": frozenset({4, 8, 16, 32, 64}),
        "head_dim": frozenset({128}),
        "page_size": frozenset({64, 128}),
    },
    priority=Priority.PERFORMANT,
)
def cute_dsl_dsa_paged_score_sm90(
    q,
    index_k_cache,
    weights,
    block_table,
    context_lens,
    max_seq_len,
    page_size,
    q_len_per_req,
    request_ids,
    out,
):
    rows, heads, dim = q.shape
    cache_length, key_dim = index_k_cache.shape
    if (
        key_dim != dim
        or cache_length % page_size
        or not index_k_cache.is_contiguous()
        or not q.is_contiguous()
    ):
        raise ValueError(
            "TMA requires 16-byte aligned packed Q and whole packed K pages"
        )
    if (
        weights.shape != (rows, heads)
        or weights.stride(-1) != 1
        or block_table.ndim != 2
        or block_table.stride(-1) != 1
        or context_lens.shape != (rows,)
        or not context_lens.is_contiguous()
    ):
        raise ValueError(
            "Expected row weights, a page table and one context length per query"
        )
    if request_ids is None:
        if q_len_per_req is None or q_len_per_req <= 0 or rows % q_len_per_req:
            raise ValueError("q_len_per_req must divide query rows")
        if block_table.shape[0] < rows // q_len_per_req:
            raise ValueError("Expected one page-table row per request")
    elif (
        q_len_per_req is not None
        or request_ids.shape != (rows,)
        or request_ids.dtype != torch.int32
        or not request_ids.is_contiguous()
    ):
        raise ValueError(
            "Explicit request_ids requires a contiguous int32 row map and q_len_per_req=None"
        )

    reqs = context_lens if request_ids is None else request_ids
    if not 0 <= max_seq_len < 2**31:
        raise ValueError("Score width and query rows must fit in int32")

    partitions = 1
    if rows and max_seq_len and cache_length and block_table.numel():
        sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
        max_wave = 16
        # The measured 78-SM Hopper geometry benefits from fewer CTA splits
        # at small row counts and more at 128–256 rows.
        if (
            sm_count == 78
            and page_size == 64
            and heads in (16, 32)
            and q_len_per_req in (1, 4, 8)
            and 64 * 1024 <= max_seq_len <= 256 * 1024
        ):
            if rows == 1:
                max_wave = 4
            elif rows < 8 or (rows == 8 and max_seq_len == 64 * 1024):
                max_wave = 8
            elif 128 <= rows <= 256:
                max_wave = 32
        target = (max_wave * sm_count + rows - 1) // rows
        tiles = (max_seq_len + SM90_WGMMA_M - 1) // SM90_WGMMA_M
        partitions = min(tiles, triton.next_power_of_2(target))

        # Tail columns are int32, including the rounded-up tile, CTA offset
        # and the final stride. The row-strided output address is still int64.
        if max_seq_len + SM90_WGMMA_M - 1 + partitions * _THREADS_PER_CTA >= 2**31:
            raise ValueError("Score tail coordinates must fit in int32")

    if out is None:
        out = torch.empty((rows, max_seq_len), device=q.device, dtype=torch.float32)
    elif (
        out.shape != (rows, max_seq_len)
        or out.dtype != torch.float32
        or out.device != q.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "Expected contiguous FP32 scores [rows, max_seq_len] on the query device"
        )

    if not out.numel():
        return out
    if not cache_length or not block_table.numel():
        return out.fill_(-float("inf"))

    with torch.cuda.device(q.device):
        _compile_score(heads, page_size, q.device.index)(
            *_tensors(index_k_cache, q, weights, block_table, reqs, context_lens, out),
            cutlass.Int32(q_len_per_req or 0),
            cutlass.Int32(partitions),
            CUstream(torch.cuda.current_stream(q.device).cuda_stream),
        )
    return out


def _tensors(key, query, weights, page_table, requests, context_lens, output):
    k = from_dlpack(key, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    q = from_dlpack(query, assumed_align=16).mark_layout_dynamic(leading_dim=2)
    rest = [
        from_dlpack(tensor).mark_layout_dynamic(leading_dim=tensor.ndim - 1)
        for tensor in (weights, page_table, requests, context_lens, output)
    ]
    return k, q, *rest


@cache
def _compile_score(heads, page_size, device_index):
    device = torch.device("cuda", device_index)
    key = torch.empty((page_size, 128), device=device, dtype=torch.bfloat16)
    query = torch.empty((2, heads, 128), device=device, dtype=torch.bfloat16)
    weights = torch.empty((2, heads), device=device, dtype=torch.float32)
    page_table = torch.empty((2, 2), device=device, dtype=torch.int32)
    requests = torch.empty((2,), device=device, dtype=torch.int32)
    context_lens = torch.empty_like(requests)
    output = torch.empty((2, SM90_WGMMA_M), device=device, dtype=torch.float32)
    return cute.compile(
        PagedDSAScore(heads=heads, page_size=page_size),
        *_tensors(key, query, weights, page_table, requests, context_lens, output),
        cutlass.Int32(0),
        cutlass.Int32(1),
        CUstream(0),
    )


class PagedDSAScore:
    """Score the visible Token K history in logical token order."""

    metadata_entries = 1  # One logical tile origin, or -64 for an invalid page.

    def __init__(
        self,
        heads: int,
        page_size: int,
    ):
        self.heads = heads
        self.dim = 128  # Index head dimension supported by this kernel.
        self.n = max(8, heads)
        self.stages = 4
        self.compute_warps = 128 // 32  # One WGMMA warpgroup.
        self.inner_loop = self.dim // SM90_WGMMA_K

        self.page_size = page_size

    @cute.jit
    def __call__(
        self,
        key: cute.Tensor,  # [num_pages * page_size, 128]
        query: cute.Tensor,  # [rows, nheads, 128]
        weights: cute.Tensor,  # [rows, nheads]
        page_table: cute.Tensor,  # [bsz, max_pages_among_reqs]
        requests: cute.Tensor,  # [rows]
        context_lens: cute.Tensor,  # [rows]
        output: cute.Tensor,  # [rows, max_seq_len]
        qolen: cutlass.Int32,
        partitions: cutlass.Int32,
        stream: CUstream,
    ):
        # WGMMA: Key [64,128] @ Query.T [128,self.n]
        # 1. input:  BF16
        # 2. accm_t: FP32
        mma = sm90.make_trivial_tiled_mma(
            cutlass.BFloat16,  # inputA.dtype
            cutlass.BFloat16,  # inputB.dtype
            cute.nvgpu.OperandMajorMode.K,  # inputA.k_major
            cute.nvgpu.OperandMajorMode.K,  # inputB.k_major
            cutlass.Float32,  # accumC
            (1, 1, 1),
            (SM90_WGMMA_M, self.n),  # m64-k16-n(nheads)
        )

        # cute.slice_(mma_tiler_mnk, (None, 0, None))
        # cute.append(a_smem_shape, self.stages)
        # k_smem_layout: [SM90_WGMMA_M, k, stages]
        k_smem_layout = sm90.make_smem_layout_a(
            LayoutEnum.ROW_MAJOR,
            (SM90_WGMMA_M, self.n, 128),
            cutlass.BFloat16,
            self.stages,
        )

        # q_smem_layout: [self.n, k, 1]
        q_smem_layout = sm90.make_smem_layout_b(
            LayoutEnum.ROW_MAJOR, (SM90_WGMMA_M, self.n, 128), cutlass.BFloat16, 1
        )

        # Build a TMA descriptor for one full 64-token K tile.
        k_tma_map, k_global_idx = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            key,  # k_hbm
            # [SM90_WGMMA_M, k, stage] -> [SM90_WGMMA_M, k], fixed at stage 0
            cute.slice_(k_smem_layout, (None, None, 0)),
            (SM90_WGMMA_M, self.dim),  # one complete K tile per stage
        )

        # Keep HBM Q in [rows,heads,128]; one copy selects one complete row.
        q_copy_layout = cute.slice_(q_smem_layout, (None, None, 0))
        q_tma_layout = cute.make_composed_layout(
            q_copy_layout.inner,
            0,
            cute.make_layout(
                (1, q_copy_layout.outer.shape[0], q_copy_layout.outer.shape[1]),
                stride=(
                    0,
                    q_copy_layout.outer.stride[0],
                    q_copy_layout.outer.stride[1],
                ),
            ),
        )
        q_tma_map, q_global_idx = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            query,  # q_hbm
            q_tma_layout,  # [1,self.n,128], same Q SMEM addresses
            (1, self.n, 128),  # copy-tile
        )

        self.kernel(
            k_tma_map,
            k_global_idx,
            q_tma_map,
            q_global_idx,
            weights,
            page_table,
            requests,
            context_lens,
            output,
            mma,
            k_smem_layout,
            q_smem_layout,
            qolen,
            partitions,
        ).launch(
            grid=(weights.shape[0], partitions, 1),  # num_tokens, cta_partitions
            block=(_THREADS_PER_CTA, 1, 1),
            stream=stream,
        )

    # ========================================================================
    # SECTION 3/6 — CTA resources
    # ========================================================================
    @cute.kernel
    def kernel(
        self,
        k_tma_map: cute.CopyAtom,
        key: cute.Tensor,
        q_tma_map: cute.CopyAtom,
        query: cute.Tensor,
        weights: cute.Tensor,
        page_table: cute.Tensor,
        requests: cute.Tensor,
        context_lens: cute.Tensor,
        output: cute.Tensor,
        mma: cute.TiledMma,
        k_smem_layout: cute.ComposedLayout,
        q_smem_layout: cute.ComposedLayout,
        qolen: cutlass.Int32,
        partitions: cutlass.Int32,
    ):
        row, cta_id_in_query, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()

        smem = utils.SmemAllocator()

        # k_smem: [SM90_WGMMA_M, k, stages]
        k_smem = smem.allocate_tensor(
            cutlass.BFloat16,
            k_smem_layout.outer,
            byte_alignment=1024,
            swizzle=k_smem_layout.inner,
        )

        # q_smem: [self.n, k]
        q_smem = smem.allocate_tensor(
            cutlass.BFloat16,
            q_smem_layout.outer,
            byte_alignment=1024,
            swizzle=q_smem_layout.inner,
        )
        metadata = smem.allocate_tensor(
            cutlass.Int64, cute.make_layout((self.stages, self.metadata_entries))
        )

        # Full/empty barriers; first K tile shares its full barrier with Q.
        barrier_storage = smem.allocate_array(cutlass.Int64, 2 * self.stages)
        full_barriers = barrier_storage
        empty_barriers = barrier_storage + self.stages
        if tid == 0:
            for stage in cutlass.range_constexpr(self.stages):
                cute.arch.mbarrier_init(full_barriers + stage, 1)
                cute.arch.mbarrier_init(empty_barriers + stage, self.compute_warps)

        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        # tile shape : [SM90_WGMMA_M, 128]
        # tile no.   : [num_k_tiles, num_dim_tiles]
        # selection  :    (None,            0)
        # k_tiles_hbm: [SM90_WGMMA_M, 128, num_k_tiles]
        k_tiles_hbm = cute.local_tile(
            key,  # [num_pages * page_size, 128]
            (SM90_WGMMA_M, self.dim),
            (None, 0),  # [:, 0]
        )

        # build tma copy map
        # smem_dst: [(SM90_WGMMA_M, 128), n_stages]
        # hbm_src : [(SM90_WGMMA_M, 128), num_k_tiles]
        k_tma_dst_smem, k_tma_src_hbm = cute.nvgpu.cpasync.tma_partition(
            k_tma_map,
            0,
            cute.make_layout(1),  # non-multi_cast
            cute.group_modes(k_smem, 0, 2),
            cute.group_modes(k_tiles_hbm, 0, 2),
        )

        # tile shape : [1, self.n, 128], 1 for row-placeholder
        # tile no.   : [num_tokens, num_head_tile, num_dim_tile]
        q_tiles_hbm = cute.local_tile(
            query,  # [rows, nheads, 128] TMA source coordinates
            (1, self.n, 128),
            (row, 0, 0),  # select row; keep all heads and index dimensions
        )

        # build tma copy map
        # smem_dst: [(nheads, 128), 1]
        # hbm_src : [(nheads, 128), 1]
        q_tma_dst_smem, q_tma_src_hbm = cute.nvgpu.cpasync.tma_partition(
            q_tma_map,
            0,
            cute.make_layout(1),
            cute.group_modes(q_smem, 0, 2),
            cute.group_modes(q_tiles_hbm, 0, 3),
        )

        mma_operand_slice = mma.get_slice(0)

        # k_smem: [SM90_WGMMA_M, 128, n_stages]
        # q_smem: [nheads, 128, 1]
        # layout changed by mma.partition_A
        k_mma_desc = mma.make_fragment_A(mma_operand_slice.partition_A(k_smem))
        q_mma_desc = mma.make_fragment_B(mma_operand_slice.partition_B(q_smem))

        accumulator_coords = mma.get_slice(tid).partition_C(
            cute.make_identity_tensor((SM90_WGMMA_M, self.n))
        )
        # accumulator_coords.shape: (SM90_WGMMA_M * self.n) / 128
        acc0 = cute.make_rmem_tensor(accumulator_coords.shape, cutlass.Float32)
        acc1 = cute.make_rmem_tensor(accumulator_coords.shape, cutlass.Float32)

        # query-view
        # output.shape[1] is max_histo_length
        tiles = self.tiles_in_row(output.shape[1], context_lens[row])
        self.fill_score_tail(output, tiles, partitions)

        # cta-in-query-view
        tiles_per_cta = cutlass.Int32(cute.ceil_div(tiles, partitions))

        tile_begin = cta_id_in_query * tiles_per_cta
        count = cutlass.Int32(0)
        if tile_begin < tiles:
            count = cutlass.min(tiles_per_cta, cutlass.Int32(tiles - tile_begin))

        if tid >= (self.compute_warps * 32):
            cute.arch.setmaxregister_decrease(48)
            if (tid // 32) == self.compute_warps:  # warp_idx==4
                self.produce(
                    k_tma_map,
                    k_tma_dst_smem,
                    k_tma_src_hbm,
                    q_tma_map,
                    q_tma_src_hbm,
                    q_tma_dst_smem,
                    full_barriers,
                    empty_barriers,
                    metadata,
                    tile_begin,
                    count,
                    page_table,
                    requests,
                    qolen,
                    cutlass.Int64(key.shape[0]),
                )
        else:
            cute.arch.setmaxregister_increase(248)
            self.consume(
                mma,
                k_mma_desc,
                q_mma_desc,
                acc0,
                acc1,
                full_barriers,
                empty_barriers,
                metadata,
                tile_begin,
                count,
                weights,
                context_lens,
                output,
            )
        cute.arch.sync_threads()

    @cute.jit
    def fill_score_tail(self, output, tiles, partitions):
        row, cta_id_in_query, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        blk_dim, _, _ = cute.arch.block_dim()

        # wgmma::m level write output[row, invalid_idx] with -inf
        tail_begin = tiles * SM90_WGMMA_M
        idx = cta_id_in_query * blk_dim + tid
        for column in cutlass.range(
            tail_begin + idx,
            cutlass.Int32(output.shape[1]),
            partitions * blk_dim,
            unroll=1,
        ):
            output[cutlass.Int64(row), cutlass.Int64(column)] = cutlass.Float32(
                float("-inf")
            )

    # ========================================================================
    # SECTION 4/6 — TMA producer: Map logical tokens to physical K slots
    # ========================================================================
    @cute.jit
    def tiles_in_row(self, width, context):
        # min(max_histo, own_histo)
        visible_tokens = cutlass.max(
            cutlass.Int32(0), cutlass.min(context, cutlass.Int32(width))
        )
        return cutlass.Int32(cute.ceil_div(visible_tokens, SM90_WGMMA_M))

    @cute.jit
    def produce(
        self,
        k_tma_map,
        k_tma_dst_smem,
        k_tma_src_hbm,
        q_tma_map,
        q_tma_src_hbm,
        q_tma_dst_smem,
        full_barriers,
        empty_barriers,
        metadata,
        tile_begin,
        count,
        page_table,
        requests,
        qolen, # for decode-spec
        cache_length,
    ):
        row, _, _ = cute.arch.block_idx()
        req_id = cutlass.Int64(0)
        if qolen > 0:
            req_id = cutlass.Int64(row // qolen)
        else:
            req_id = cutlass.Int64(requests[row])

        empty_phase = cutlass.Int32(0)
        for i in cutlass.range(count, unroll=1):
            slot = i % self.stages
            if i >= self.stages:
                empty_phase = (i // self.stages - 1) % 2
                cute.arch.mbarrier_wait(empty_barriers + slot, empty_phase)

            self.load_key(
                k_tma_map,
                k_tma_dst_smem,
                k_tma_src_hbm,
                full_barriers,
                metadata,
                slot,
                tile_begin + i,  # cta_id_in_query * tiles_per_cta + i
                page_table,
                req_id,
                cache_length,
                i == 0,
            )
            if i == 0:
                cute.copy(
                    q_tma_map,
                    q_tma_src_hbm,
                    q_tma_dst_smem[None, 0],
                    tma_bar_ptr=full_barriers,
                )

    @cute.jit
    def resolve_key_slot(
        self,
        token_begin,
        page_table,
        req_id,
        cache_length,
    ):
        # A 64-token tile stays in one physical page.
        logical_page = token_begin // self.page_size
        phy_begin = cache_length  # Out-of-pool TMA source for zero-fill.
        if (
            req_id >= 0
            and req_id < page_table.shape[0]
            and logical_page < page_table.shape[1]
        ):
            page = cutlass.Int64(page_table[req_id, logical_page])
            if page >= 0 and page < cache_length // self.page_size:
                phy_begin = page * self.page_size + token_begin % self.page_size
        return phy_begin

    @cute.jit
    def load_key(
        self,
        k_tma_map: cute.CopyAtom,
        k_tma_dst_smem: cute.Tensor,
        k_tma_src_hbm: cute.Tensor,
        full_barriers: cute.Pointer,
        metadata: cute.Tensor,
        slot: cutlass.Int32,
        tile: cutlass.Int32,
        page_table: cute.Tensor,
        req_id: cutlass.Int64,
        cache_length: cutlass.Int64,
        first_tile: cutlass.Boolean,
    ):
        token_begin = cutlass.Int64(tile) * SM90_WGMMA_M
        phy_begin = self.resolve_key_slot(token_begin, page_table, req_id, cache_length)

        element_bytes = cutlass.BFloat16.width // 8  # sizeof(BF16)
        transaction_bytes = SM90_WGMMA_M * self.dim * element_bytes  # a
        if first_tile:
            transaction_bytes += self.n * self.dim * element_bytes  # b

        with cute.arch.elect_one():
            origin = cutlass.Int64(-SM90_WGMMA_M)
            if phy_begin < cache_length:
                origin = token_begin
            metadata[slot, 0] = origin
            cute.arch.mbarrier_arrive_and_expect_tx(
                full_barriers + slot, transaction_bytes
            )

        cute.copy(
            k_tma_map,
            k_tma_src_hbm[None, phy_begin // SM90_WGMMA_M],
            k_tma_dst_smem[None, slot],
            tma_bar_ptr=full_barriers + slot,
        )

    # ========================================================================
    # SECTION 5/6 — WGMMA consumer: Overlap dual accumulators with CUDA core work
    # ========================================================================
    @cute.jit
    def load_weights(
        self,
        weights,
        row,
        tid,
    ):
        # Load this lane's head weights once per CTA, then reuse them for all K tiles.
        head_weights = cute.make_rmem_tensor(
            cute.make_layout((self.n // 8, 2)), cutlass.Float32
        )
        for h8 in cutlass.range_constexpr(self.n // 8):
            h = (tid % 4) * 2 + h8 * 8
            w0 = cutlass.Float32(0)
            w1 = cutlass.Float32(0)
            if h < self.heads:
                w0 = weights[cutlass.Int64(row), h]
                w1 = weights[cutlass.Int64(row), h + 1]
            head_weights[h8, 0] = w0
            head_weights[h8, 1] = w1
        return head_weights

    @cute.jit
    def consume(
        self,
        mma,
        k_mma_desc,
        q_mma_desc,
        acc0,
        acc1,
        full_barriers,
        empty_barriers,
        metadata,
        tile_begin,
        count,
        weights,
        context_lens,
        output,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        context = context_lens[row]

        head_weights = self.load_weights(weights, row, tid)

        # each cta trigger 2 wgmma for overlapping
        for pair in cutlass.range(cute.ceil_div(count, 2), unroll=1):
            i = pair * 2
            slot0 = i % self.stages
            slot1 = (i + 1) % self.stages
            is_double = (i + 1) < count

            # logical k-tile index processed by this CTA in this iter
            # each k-tile [SM90_WGMMA_M, 128]
            tile = tile_begin + i
            cute.arch.mbarrier_wait(full_barriers + slot0, i // self.stages % 2)

            cute.nvgpu.warpgroup.fence()
            mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
            for k in cutlass.range_constexpr(self.inner_loop):
                cute.gemm(
                    mma,
                    acc0,
                    k_mma_desc[None, None, k, slot0],  # at stage
                    q_mma_desc[None, None, k, 0],
                    acc0,
                )
                mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

            cute.nvgpu.warpgroup.commit_group()
            if is_double:
                cute.arch.mbarrier_wait(
                    full_barriers + slot1, (i + 1) // self.stages % 2
                )
                cute.nvgpu.warpgroup.fence()
                mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
                for k in cutlass.range_constexpr(self.inner_loop):
                    cute.gemm(
                        mma,
                        acc1,
                        k_mma_desc[None, None, k, slot1],
                        q_mma_desc[None, None, k, 0],
                        acc1,
                    )
                    mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(1)
            else:
                cute.nvgpu.warpgroup.wait_group(0)

            self.epilogue(
                acc0,
                head_weights,
                output,
                tile,
                metadata,
                slot0,
                context,
            )
            cute.arch.sync_warp()
            if tid % 32 == 0:
                cute.arch.mbarrier_arrive(empty_barriers + slot0)

            if is_double:
                cute.nvgpu.warpgroup.wait_group(0)
                self.epilogue(
                    acc1,
                    head_weights,
                    output,
                    tile + 1,
                    metadata,
                    slot1,
                    context,
                )
                cute.arch.sync_warp()
                if tid % 32 == 0:
                    cute.arch.mbarrier_arrive(empty_barriers + slot1)
        cute.nvgpu.warpgroup.wait_group(0)

    # ========================================================================
    # SECTION 6/6 — Epilogue: Reduce head scores to one score per token
    # ========================================================================
    @cute.jit
    def logical_token(self, metadata, slot, warp_id, lane_id, r):
        return metadata[slot, 0] + warp_id * 16 + lane_id // 4 + r * 8

    @cute.jit
    def epilogue(
        self,
        acc: cute.Tensor,
        head_weights: cute.Tensor,
        output: cute.Tensor,
        tile: cutlass.Int32,
        metadata: cute.Tensor,
        slot: cutlass.Int32,
        context: cutlass.Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane_id = tid % 32
        warp_id = tid // 32

        #  WGMMA::M × self.n / tids (128)
        flat = cute.make_tensor(acc.iterator, cute.make_layout(self.n // 2))
        v0 = cutlass.Float32(0)
        v1 = cutlass.Float32(0)

        # out: [WGMMA::M, self.n] per query
        for h8 in cutlass.range_constexpr(max(1, self.heads // 8)):
            # each lane works for 2-slot in 8-column-group
            # at same row inside WG::M
            w0 = head_weights[h8, 0]
            w1 = head_weights[h8, 1]
            v0 = (
                v0
                + cutlass.max(flat[4 * h8], 0.0) * w0
                + cutlass.max(flat[4 * h8 + 1], 0.0) * w1
            )
            v1 = (
                v1
                + cutlass.max(flat[4 * h8 + 2], 0.0) * w0
                + cutlass.max(flat[4 * h8 + 3], 0.0) * w1
            )
        v0 = v0 + cute.arch.shuffle_sync_bfly(v0, 1)
        v0 = v0 + cute.arch.shuffle_sync_bfly(v0, 2)
        v1 = v1 + cute.arch.shuffle_sync_bfly(v1, 1)
        v1 = v1 + cute.arch.shuffle_sync_bfly(v1, 2)

        # Each 4-lane group has both reduced scores; lanes 0/1 store one each.
        if lane_id % 4 < 2:
            r = lane_id % 4
            col = tile * SM90_WGMMA_M + warp_id * 16 + lane_id // 4 + r * 8
            if col < output.shape[1]:
                token = self.logical_token(metadata, slot, warp_id, lane_id, r)
                valid = token >= 0 and token < context
                value = v0
                if r == 1:
                    value = v1

                if not valid:
                    value = -float("inf")
                output[cutlass.Int64(row), cutlass.Int64(col)] = value
