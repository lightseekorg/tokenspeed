# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Vendored from vLLM's Kimi-K3.1 specialized decode (cutedsl/dsa/indexer.py): the
# FP8 tensor-parallel IndexerScore and IndexerSelect, without the MXFP4, DCP pitch,
# local top-N and exchange variants. Modified to reproduce DeepGEMM's FP32 paged
# MQA logits and an exact FP32 top-k:
# - The scorer's epilogue sums the heads in FP32 in DeepGEMM's SM100 order
#   (sm100_mqa_logits.cuh, FP32 reduce: x + |x|, two FMA chains over heads
#   (4i, 4i + 1) and (4i + 2, 4i + 3), the chains added, the two lanes added and
#   halved, times the key's FP32 scale) and stores FP32 logits with NaN and +inf
#   as -inf; head weights and key scales are FP32.
# - Each row keeps its accumulator until its last row's TMEM load (FP32 rows do
#   not fit the BF16 form's registers).
# - The select ranks 32-bit order-preserving keys (-0 as +0) in three radix
#   digits of 11, 11 and 10 bits and keeps 32-bit vector maxima; its bound uses
#   the maxima's top 22 bits.
# - Row ends come from per-request lengths (row j of q sees length - q + 1 + j
#   keys), clamped to the block table; row, request, page and table sizes are
#   runtime values, and the block table and the slot rows are runtime-strided.
# - The select never releases its dependents early: the sparse core reads the
#   counts before its PDL wait.

"""Fused decode indexer of a DSA layer: FP32 logits and an exact top-k.

Two PDL-chained kernels replace the paged MQA logits, the non-finite scrub, the
top-k and the slot mapping of a decode step:

  IndexerScore   persistent tcgen05 scorer over the 128-key tiles of every
                 request (key-balanced across the SMs): one [128 keys x q * 32]
                 FP8 MMA per tile scores all ``q`` rows of a request, so each key
                 page is read once per step. The epilogue forms DeepGEMM's FP32
                 logit of each (row, key) bit for bit, stores it into a
                 request-compact FP32 scratch and each 8 keys' largest 32-bit
                 order key into a scratch 1/8 its size. Requests whose rows all
                 fit the top-k are not scored.
  IndexerSelect  one CTA per row: the K-th largest of the row's 8-key maxima
                 bounds its K-th largest key from below, so only the keys at or
                 above that bound are candidates; an exact radix select over them
                 and one ordered pass write the selected keys' slots in key order
                 (ties at the threshold go to the lowest keys), -1 past the count.
                 A row of at most ``topk`` keys takes all of them without reading
                 logits.

Scratch: request r's rows are consecutive ``tiles(r) * 128``-key rows starting at
element ``q * 128 * (tiles of the requests before r)``, ``tiles(r) =
ceil(length(r) / 128)`` for a request longer than the top-k and 0 otherwise.
"""

import functools

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32, Int64, Uint8, Uint32
from cutlass.cute.nvgpu import cpasync, tcgen05
from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.common.primitives import (
    add_rn,
    asm,
    bits_float,
    fence_tmem,
    float_bits,
    fma,
    load_words,
)

PAGE = 64  # keys per index-K page
DIM = 128
ROW_BYTES = DIM + 4  # a key's FP8 bytes, then (in the page tail) its FP32 scale
HEADS = 32
TILE = 128  # keys per MMA tile (two pages)
ITEM_TILES = 16  # tiles per work item (the producer caches 32 page ids)
MAX_REQUESTS = 4096  # tile prefix sum in shared memory
STAGES = 6
EPI_GROUPS = 2  # TMEM accumulators, each tile's epilogue on one of them
EPI_WARPS = 4 * EPI_GROUPS
PRODUCER = EPI_WARPS
MMA_WARP = PRODUCER + 1
SCORE_THREADS = 32 * (MMA_WARP + 1)

SELECT_THREADS = 1024
SELECT_WARPS = SELECT_THREADS // 32
BINS = 2048  # the first two radix digits: bits 31..21, then 20..10
LOW_BINS = 1024  # the last digit: bits 9..0
# Vector maxima of a row kept in shared memory (rows of up to 128K keys; longer
# rows read the rest from the scorer's scratch).
TOPS = 16384
CANDIDATES = 6144  # keys at or above the bound (more: the exact select over the row)
HOT = 4096  # vectors holding a candidate (more: the exact select over the row)
NEG_INF_BITS = 0xFF800000
POS_INF_BITS = 0x7F800000


# --------------------------------------------------------------------------
# PTX helpers
# --------------------------------------------------------------------------
@asm()
def bulk_g2s(dst, src, barrier, size):
    code = "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
    code += "[$0], [$1], $3, [$2];"
    return code, "r,l,r,r,~{memory}", (dst, src, barrier, Int32(size))


@asm()
def store_b32(address, value):
    return "st.global.b32 [$0], $1;", "l,r,~{memory}", (Int64(address), Uint32(value))


@asm()
def shared_add(pointer, value):
    return "red.shared.add.u32 [$0], $1;", "r,r,~{memory}", (pointer, Uint32(value))


@asm(Int32)
def load_i32(address):
    return "ld.global.b32 $0, [$1];", "=r,l,~{memory}", (Int64(address),)


@asm(Uint32)
def load_u32(address):
    return "ld.global.b32 $0, [$1];", "=r,l,~{memory}", (Int64(address),)


@asm(Float32, pure=True)
def double_relu(x: Float32):
    """x + |x|, round to nearest even (DeepGEMM's FP32 relu, before the halving)."""
    code = "{ .reg .f32 a; abs.f32 a, $1; add.rn.f32 $0, $1, a; }"
    return code, "=f,f", (x,)


@asm(Float32, pure=True)
def mul_rn(a, b):
    return "mul.rn.f32 $0, $1, $2;", "=f,f,f", (Float32(a), Float32(b))


@asm(Uint32, pure=True)
def max_u32(a, b):
    return "max.u32 $0, $1, $2;", "=r,r,r", (Uint32(a), Uint32(b))


@cute.jit
def order_key(bits: Uint32) -> Uint32:
    """Order-preserving 32-bit key of FP32 ``bits`` (a larger value, a larger
    key); -0 ranks as +0, so equal values tie."""
    value = bits
    if value == Uint32(0x80000000):
        value = Uint32(0)
    flip = Uint32(0x80000000)
    if (value & Uint32(0x80000000)) != Uint32(0):
        flip = Uint32(0xFFFFFFFF)
    return value ^ flip


@cute.jit
def scrubbed_bits(value: Float32) -> Uint32:
    """FP32 bits of ``value`` with NaN and +inf as -inf."""
    bits = Uint32(float_bits(value))
    bad = Int32(0)
    if (bits & Uint32(0x7FFFFFFF)) > Uint32(POS_INF_BITS):
        bad = Int32(1)
    if bits == Uint32(POS_INF_BITS):
        bad = Int32(1)
    if bad != Int32(0):
        bits = Uint32(NEG_INF_BITS)
    return bits


@cute.jit
def table_entry(table: cute.Tensor, r: Int32, column: Int32) -> Int32:
    """``table[r, column]`` of an int32 block table, 64-bit addressed."""
    offset = Int64(r) * Int64(table.stride[0]) + Int64(column)
    return load_i32(table.iterator.toint() + offset * 4)


@cute.jit
def clamp_end(value: Int32, cap: Int32) -> Int32:
    return cutlass.min(cutlass.max(value, Int32(0)), cap)


@cute.jit
def request_tiles(seq_lens: cute.Tensor, r: Int32, cap: Int32, topk) -> Int32:
    """128-key tiles of request ``r``; 0 when all its rows fit the top-k (its
    last row sees the most keys)."""
    end = clamp_end(seq_lens[r], cap)
    tiles = Int32(0)
    if end > topk:
        tiles = (end + Int32(TILE - 1)) // Int32(TILE)
    return tiles


@cute.jit
def warp_inclusive(value: Uint32, lane: Int32) -> Uint32:
    for i in cutlass.range_constexpr(5):
        up = cute.arch.shuffle_sync_up(value, offset=1 << i, mask_and_clamp=0)
        if lane >= Int32(1 << i):
            value += up
    return value


@cute.jit
def warp_sum(value: Uint32) -> Uint32:
    for i in cutlass.range_constexpr(5):
        value += cute.arch.shuffle_sync_bfly(value, 1 << i)
    return value


@cute.jit
def warp_totals(scan, warp, lane):
    """(sum of ``scan`` over the warps before ``warp``, sum over all): every
    warp scans the per-warp words itself."""
    x = Uint32(0)
    if lane < SELECT_WARPS:
        x = scan[lane]
    inclusive = warp_inclusive(x, lane)
    return (
        cute.arch.shuffle_sync(inclusive - x, warp),
        cute.arch.shuffle_sync(inclusive, 31),
    )


@cute.jit
def block_exclusive(value: Uint32, scan, warp, lane):
    """(exclusive prefix over the CTA's threads in order, total); ``scan``:
    one word per warp."""
    inclusive = warp_inclusive(value, lane)
    if lane == 31:
        scan[warp] = inclusive
    cute.arch.sync_threads()
    prior, total = warp_totals(scan, warp, lane)
    cute.arch.sync_threads()
    return prior + inclusive - value, total


@cute.jit
def topdown(
    counts,
    bins: cutlass.Constexpr[int],
    need,
    info,
    scan,
    tid,
    warp,
    lane,
):
    """info[2] = the bin where the counts summed from the top bin down reach
    ``need``, info[3] = the count above it (the caller presets both)."""
    per = max(1, bins // SELECT_THREADS)
    top = bins - 1 - per * tid
    mine = Uint32(0)
    if top >= 0:
        for i in cutlass.range_constexpr(per):
            mine += counts[top - i]
    inclusive = warp_inclusive(mine, lane)
    if lane == 31:
        scan[warp] = inclusive
    cute.arch.sync_threads()
    prior, _ = warp_totals(scan, warp, lane)
    above = prior + inclusive - mine
    if above < need and above + mine >= need:
        running = above
        found = Int32(0)
        for i in cutlass.range_constexpr(per):
            count = counts[top - i]
            if found == 0 and running + count >= need:
                info[2] = top - i
                info[3] = Int32(running)
                found = Int32(1)
            running += count
    cute.arch.sync_threads()


@cute.jit
def vector_keys(address: Int64, v: Int32, length: Int32):
    """The order keys of vector ``v``'s eight FP32 logits; keys at or past
    ``length`` are 0 (below every logit's key)."""
    base = address + Int64(v) * 32
    low = load_words(base, count=4, mutable=True)
    high = load_words(base + 16, count=4, mutable=True)
    keys = []
    for e in cutlass.range_constexpr(8):
        key = order_key(Uint32(low[e] if e < 4 else high[e - 4]))
        if v * 8 + e >= length:
            key = Uint32(0)
        keys.append(key)
    return keys


@cute.jit
def vector_max(keys) -> Uint32:
    a = max_u32(max_u32(keys[0], keys[1]), max_u32(keys[2], keys[3]))
    b = max_u32(max_u32(keys[4], keys[5]), max_u32(keys[6], keys[7]))
    return max_u32(a, b)


# --------------------------------------------------------------------------
# Scorer
# --------------------------------------------------------------------------
@cute.jit
def _first_request(start: cute.Tensor, requests: Int32, g: Int32) -> Int32:
    """The request whose tiles contain global tile ``g`` (binary search)."""
    lo = Int32(0)
    hi = requests
    while lo < hi:
        mid = (lo + hi) // 2
        if start[mid + 1] > g:
            hi = mid
        else:
            lo = mid + 1
    return lo


@cute.jit
def _next_item(start: cute.Tensor, requests: Int32, g: Int32, r: Int32, hi: Int32):
    """The work item at global tile ``g`` of request ``r`` (the CTA range ends at
    ``hi``): (request, first key, tiles, next g, next r)."""
    r_end = start[r + 1]
    tiles = cutlass.min(cutlass.min(hi, r_end) - g, Int32(ITEM_TILES))
    k0 = (g - start[r]) * Int32(TILE)
    req = r
    g += tiles
    while r < requests and start[r + 1] <= g:
        r += 1
    return req, k0, tiles, g, r


def _mn_layout(t):
    """A tcgen05 accumulator as ((M), (N), stages)."""
    sh, sd = t.layout.shape, t.layout.stride
    new = cute.make_layout(
        ((sh[0][0], sh[1]), (sh[0][1], sh[2]), *sh[3:]),
        stride=((sd[0][0], sd[1]), (sd[0][1], sd[2]), *sd[3:]),
    )
    return cute.make_tensor(t.iterator, new)


class IndexerScore:
    """FP32 logits of every row (``q`` per request) into the request-compact
    scratch, and each 8 keys' largest order key into ``tops``.

    Arguments: ``query`` the FP8 index query bytes [rows * 32, 128]; ``weights``
    the FP32 head weights [rows * 32] (the query scale and score scale folded
    in); ``cache`` the packed index-K plane bytes [slots, 132] (per 64-slot page
    the FP8 keys, then one FP32 scale per key); ``table`` the int32 block table
    [requests, pages]; ``seq_lens`` the int32 request lengths [requests];
    ``logits`` / ``tops`` the FP32 / int32 scratch."""

    def __init__(self, q: int, topk: int, ctas: int, pdl: bool):
        assert 1 <= q <= 8
        self.q, self.topk, self.ctas, self.pdl = q, topk, ctas, pdl
        self.stages = STAGES
        self.n = q * HEADS
        self.tiler = (TILE, self.n, DIM)
        self.columns = max(32, 1 << (EPI_GROUPS * self.n - 1).bit_length())

    @cute.jit
    def __call__(
        self,
        query_bytes: cute.Tensor,
        weights: cute.Tensor,
        cache: cute.Tensor,
        table: cute.Tensor,
        seq_lens: cute.Tensor,
        logits: cute.Tensor,
        tops: cute.Tensor,
        stream,
    ):
        ab = cutlass.Float8E4M3FN
        vectors = query_bytes.shape[0]
        query = cute.make_tensor(
            cute.recast_ptr(query_bytes.iterator, dtype=ab),
            cute.make_layout((vectors, DIM, 1), stride=(DIM, 1, vectors * DIM)),
        )
        blocks = cache.shape[0] // PAGE
        keys = cute.make_tensor(
            cute.recast_ptr(cache.iterator, dtype=ab),
            cute.make_layout((PAGE, DIM, blocks), stride=(DIM, 1, PAGE * ROW_BYTES)),
        )
        scales = cute.make_tensor(
            cute.recast_ptr(cache.iterator + PAGE * DIM, dtype=Int32),
            cute.make_layout((blocks, PAGE), stride=(PAGE * ROW_BYTES // 4, 1)),
        )
        mma = sm100_utils.make_trivial_tiled_mma(
            ab,
            ab,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.ONE,
            self.tiler[:2],
        )
        vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (mma.thr_id.shape,))
        a_layout = sm100_utils.make_smem_layout_a(mma, self.tiler, ab, self.stages)
        b_layout = sm100_utils.make_smem_layout_b(mma, self.tiler, ab, 2)
        atom = sm100_utils.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, ab)
        page_layout = cute.tile_to_shape(atom, (PAGE, DIM, 2 * self.stages), (0, 1, 2))
        tma_k, tma_k_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            keys,
            cute.slice_(page_layout, (None, None, 0)),
            (PAGE, DIM),
        )
        b_one = cute.slice_(b_layout, (None, None, None, 0))
        tma_q, tma_q_tensor = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B((1, 1), mma.thr_id),
            query,
            b_one,
            self.tiler,
            mma,
            vmnk.shape,
        )
        self.q_bytes = cute.size_in_bytes(ab, b_one)
        self.kernel(
            mma,
            tma_k,
            tma_k_tensor,
            tma_q,
            tma_q_tensor,
            a_layout,
            b_layout,
            page_layout,
            scales,
            weights,
            table,
            seq_lens,
            logits,
            tops,
        ).launch(
            grid=[self.ctas, 1, 1],
            block=[SCORE_THREADS, 1, 1],
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.jit
    def rows_out(self, t2r, tAll, cD, sW, sEnd, rows, bases):
        """All rows of one tile: relu, the head weights and the key scale ->
        FP32 logits (DeepGEMM's order), and each 8 keys' largest order key. Row
        by row from TMEM; the accumulator is released (``release``: its empty
        barrier) once the last row's load has completed."""
        row0, pitch, key, kscale, qs, acc, lt = rows
        out_base, tops_base, release = bases
        Q = self.q
        # tcgen05.ld registers are valid after tcgen05.wait::ld: sum rows after it.
        frags = [cute.make_fragment_like(cD, Float32) for _ in range(Q)]
        cute.copy(t2r, tAll[(None, None, None, 0, 0, acc)], frags[0])
        cute.arch.fence_view_async_tmem_load()
        for tok in cutlass.range_constexpr(Q):
            if cutlass.const_expr(tok + 1 < Q):
                nxt = tAll[(None, None, None, 0, tok + 1, acc)]
                cute.copy(t2r, nxt, frags[tok + 1])
            if cutlass.const_expr(tok + 1 == Q):
                # Every load is complete (the waits above) and ordered before the
                # release that lets the next MMA overwrite the accumulator.
                fence_tmem(before=True)
                cute.arch.mbarrier_arrive(release)
            frag = frags[tok]
            weights = []
            for h in cutlass.range_constexpr(0, HEADS, 4):
                address = sW.iterator + qs * self.n + tok * HEADS + h
                weights += list(
                    load_words(
                        address,
                        count=4,
                        shared=True,
                        word_type="f32",
                        result_type=Float32,
                    )
                )
            # DeepGEMM's FP32 order: (x + |x|) * w chains, then lanes added, halved.
            s0x = Float32(0.0)
            s0y = Float32(0.0)
            s1x = Float32(0.0)
            s1y = Float32(0.0)
            for h in cutlass.range_constexpr(0, HEADS, 4):
                s0x = fma(double_relu(frag[h]), weights[h], s0x)
                s0y = fma(double_relu(frag[h + 1]), weights[h + 1], s0y)
                s1x = fma(double_relu(frag[h + 2]), weights[h + 2], s1x)
                s1y = fma(double_relu(frag[h + 3]), weights[h + 3], s1y)
            total = add_rn(add_rn(s0x, s1x), add_rn(s0y, s1y))
            bits = scrubbed_bits(mul_rn(mul_rn(total, Float32(0.5)), kscale))
            element = row0 + Int64(tok) * pitch + Int64(key)
            top = Uint32(0)
            if key < sEnd[tok, qs]:
                store_b32(out_base + element * 4, bits)
                top = order_key(bits)
            for b in cutlass.range_constexpr(3):
                top = max_u32(top, cute.arch.shuffle_sync_bfly(top, 1 << b))
            if lt % 8 == 0:
                store_b32(tops_base + (element >> 3) * 4, top)
            if cutlass.const_expr(tok + 1 < Q):
                cute.arch.fence_view_async_tmem_load()

    @cute.kernel
    def kernel(
        self,
        mma: cute.TiledMma,
        tma_k: cute.CopyAtom,
        mK: cute.Tensor,
        tma_q: cute.CopyAtom,
        mQ: cute.Tensor,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        page_layout: cute.ComposedLayout,
        scales: cute.Tensor,
        weights: cute.Tensor,
        table: cute.Tensor,
        seq_lens: cute.Tensor,
        logits: cute.Tensor,
        tops: cute.Tensor,
    ):
        Q = self.q
        ACCS = EPI_GROUPS
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()

        smem = utils.SmemAllocator()
        kv_full = smem.allocate_array(Int64, self.stages)
        kv_empty = smem.allocate_array(Int64, self.stages)
        sc_empty = smem.allocate_array(Int64, self.stages)
        q_full = smem.allocate_array(Int64, 2)
        q_empty = smem.allocate_array(Int64, 2)
        acc_full = smem.allocate_array(Int64, EPI_GROUPS)
        acc_empty = smem.allocate_array(Int64, EPI_GROUPS)
        tmem_hold = smem.allocate_array(Int32, 1)
        sEnd = smem.allocate_tensor(Int32, cute.make_layout((8, 2)), 16)
        sStart = smem.allocate_tensor(Int32, cute.make_layout((MAX_REQUESTS + 1,)), 16)
        sW = smem.allocate_tensor(Float32, cute.make_layout((self.n, 2)), 16)
        sSc = smem.allocate_tensor(Int32, cute.make_layout((TILE, self.stages)), 16)
        sA = smem.allocate_tensor(
            cutlass.Float8E4M3FN, a_layout.outer, 1024, swizzle=a_layout.inner
        )
        sB = smem.allocate_tensor(
            cutlass.Float8E4M3FN, b_layout.outer, 1024, swizzle=b_layout.inner
        )
        sPage = cute.make_tensor(sA.iterator, page_layout.outer)

        if tidx == 0:
            for i in cutlass.range_constexpr(self.stages):
                cute.arch.mbarrier_init(kv_full + i, 1)
                cute.arch.mbarrier_init(kv_empty + i, 1)
                cute.arch.mbarrier_init(sc_empty + i, 128)
            for i in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(q_full + i, 1)
                cute.arch.mbarrier_init(q_empty + i, 1 + EPI_WARPS)
            for i in cutlass.range_constexpr(EPI_GROUPS):
                cute.arch.mbarrier_init(acc_full + i, 1)
                cute.arch.mbarrier_init(acc_empty + i, 128)
            cute.arch.mbarrier_init_fence()
        if warp == PRODUCER:
            cpasync.prefetch_descriptor(tma_k)
            cpasync.prefetch_descriptor(tma_q)
        if warp == MMA_WARP:
            cute.arch.alloc_tmem(self.columns, tmem_hold)
            fence_tmem(before=True)

        # Every input may come from the kernels before: wait before the first read.
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()
        # The schedule: sStart[r] = first global tile of request r.
        requests = seq_lens.shape[0]
        cap = table.shape[1] * Int32(PAGE)
        for r in cutlass.range(tidx, requests, SCORE_THREADS, unroll=1):
            sStart[r + 1] = request_tiles(seq_lens, r, cap, self.topk)
        cute.arch.sync_threads()
        if warp == 0:
            carry = Int32(0)
            for base in cutlass.range(0, requests, 32, unroll=1):
                r = base + lane
                tiles = Int32(0)
                if r < requests:
                    tiles = sStart[r + 1]
                tiles = Int32(warp_inclusive(Uint32(tiles), lane))
                if r < requests:
                    sStart[r + 1] = carry + tiles
                carry += cute.arch.shuffle_sync(tiles, 31)
            if lane == 0:
                sStart[0] = Int32(0)
        cute.arch.sync_threads()
        fence_tmem()
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_launch_dependents()
        tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, 16, tmem_hold)
        acc_shape = mma.partition_shape_C(self.tiler[:2])
        acc_fake = mma.make_fragment_C(cute.append(acc_shape, EPI_GROUPS))
        tAcc = cute.make_tensor(tmem_ptr, acc_fake.layout)

        total = sStart[requests]
        lo = Int32((Int64(total) * Int64(bid)) // Int64(self.ctas))
        hi = Int32((Int64(total) * Int64(bid + 1)) // Int64(self.ctas))
        r_lo = _first_request(sStart, requests, lo)

        gQ = cute.local_tile(
            mQ, cute.slice_(self.tiler, (0, None, None)), (None, None, None)
        )
        thr_mma = mma.get_slice(0)
        tCgQ = thr_mma.partition_B(gQ)
        tQsQ, tQgQ = cpasync.tma_partition(
            tma_q,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgQ, 0, 3),
        )
        gK = cute.local_tile(mK, (PAGE, DIM), (None, None, None))
        tKsK, tKgK = cpasync.tma_partition(
            tma_k,
            0,
            cute.make_layout(1),
            cute.group_modes(sPage, 0, 2),
            cute.group_modes(gK, 0, 2),
        )
        it = Int32(0)  # tiles issued / consumed by this role
        j = Int32(0)  # items issued / consumed by this role
        g = lo
        r = r_lo
        max_page = table.shape[1] - 1
        blocks = scales.shape[0]
        if warp == PRODUCER:
            # One item of metadata ahead: the next item's dependent loads overlap
            # the current item's tiles.
            n_req = Int32(0)
            n_k0 = Int32(0)
            n_nt = Int32(0)
            n_page = Int32(0)
            n_end = Int32(0)
            if g < hi:
                n_req, n_k0, n_nt, g, r = _next_item(sStart, requests, g, r, hi)
                n_page = table_entry(
                    table, n_req, cutlass.min(n_k0 // PAGE + lane, max_page)
                )
                n_end = clamp_end(
                    seq_lens[n_req] - (Q - 1) + cutlass.min(lane, Int32(Q - 1)), cap
                )
            has_item = n_nt > 0
            while has_item:
                req = n_req
                ntile = n_nt
                my_page = n_page
                my_end = n_end
                n_nt = Int32(0)
                if g < hi:
                    n_req, n_k0, n_nt, g, r = _next_item(sStart, requests, g, r, hi)
                    n_page = table_entry(
                        table, n_req, cutlass.min(n_k0 // PAGE + lane, max_page)
                    )
                    n_end = clamp_end(
                        seq_lens[n_req] - (Q - 1) + cutlass.min(lane, Int32(Q - 1)),
                        cap,
                    )
                qs = j % 2
                cute.arch.mbarrier_wait(q_empty + qs, ((j // 2) & 1) ^ 1)
                if lane < Q:
                    sEnd[lane, qs] = my_end
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        q_full + qs, self.q_bytes + self.n * 4
                    )
                    bulk_g2s(
                        sW.iterator + qs * self.n,
                        weights.iterator + Int64(req) * Int64(Q * HEADS),
                        q_full + qs,
                        self.n * 4,
                    )
                cute.copy(
                    tma_q,
                    tQgQ[(None, req, 0, 0)],
                    tQsQ[(None, qs)],
                    tma_bar_ptr=q_full + qs,
                )
                j += 1
                for t in cutlass.range(ntile, unroll=1):
                    stage = it % self.stages
                    p0 = cute.arch.shuffle_sync(my_page, 2 * t)
                    p1 = cute.arch.shuffle_sync(my_page, 2 * t + 1)
                    # A partial last tile's second page may lie past the
                    # request's pages: keep its (unused) reads in bounds.
                    p0 = cutlass.min(cutlass.max(p0, Int32(0)), blocks - 1)
                    p1 = cutlass.min(cutlass.max(p1, Int32(0)), blocks - 1)
                    ph = ((it // self.stages) & 1) ^ 1
                    cute.arch.mbarrier_wait(kv_empty + stage, ph)
                    cute.arch.mbarrier_wait(sc_empty + stage, ph)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            kv_full + stage, TILE * DIM + TILE * 4
                        )
                        # 64-bit page offsets: past page 1,016,801 a page times
                        # the 2112-word page stride passes Int32.
                        bulk_g2s(
                            sSc.iterator + stage * TILE,
                            scales.iterator + Int64(p0) * Int64(scales.stride[0]),
                            kv_full + stage,
                            PAGE * 4,
                        )
                        bulk_g2s(
                            sSc.iterator + stage * TILE + PAGE,
                            scales.iterator + Int64(p1) * Int64(scales.stride[0]),
                            kv_full + stage,
                            PAGE * 4,
                        )
                    cute.copy(
                        tma_k,
                        tKgK[(None, 0, 0, p0)],
                        tKsK[(None, 2 * stage)],
                        tma_bar_ptr=kv_full + stage,
                    )
                    cute.copy(
                        tma_k,
                        tKgK[(None, 0, 0, p1)],
                        tKsK[(None, 2 * stage + 1)],
                        tma_bar_ptr=kv_full + stage,
                    )
                    it += 1
                has_item = n_nt > 0
        elif warp == MMA_WARP:
            tCrA = mma.make_fragment_A(sA)
            tCrB = mma.make_fragment_B(sB)
            while g < hi:
                _req, _k0, ntile, g, r = _next_item(sStart, requests, g, r, hi)
                qs = j % 2
                cute.arch.mbarrier_wait(q_full + qs, (j // 2) & 1)
                for _t in cutlass.range(ntile, unroll=1):
                    stage = it % self.stages
                    a = it % EPI_GROUPS
                    cute.arch.mbarrier_wait(acc_empty + a, ((it // EPI_GROUPS) & 1) ^ 1)
                    cute.arch.mbarrier_wait(kv_full + stage, (it // self.stages) & 1)
                    fence_tmem()  # after the epilogue's release of the accumulator
                    # Reconverge after the per-thread waits: cute.gemm and the
                    # commits below each elect one lane, and tcgen05.commit
                    # tracks only the MMAs its own lane issued.
                    cute.arch.sync_warp()
                    mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                        cute.gemm(
                            mma,
                            tAcc[(None, None, None, a)],
                            tCrA[(None, None, kb, stage)],
                            tCrB[(None, None, kb, qs)],
                            tAcc[(None, None, None, a)],
                        )
                        mma.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(kv_empty + stage)
                        tcgen05.commit(acc_full + a)
                    it += 1
                with cute.arch.elect_one():
                    tcgen05.commit(q_empty + qs)
                j += 1
        elif warp < EPI_WARPS:
            # Epilogue: warpgroup acc owns TMEM accumulator acc (every other tile).
            acc = warp // 4
            lt = tidx % 128
            ld_atom = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition(HEADS)), Float32
            )
            acc_cols = cute.flat_divide(_mn_layout(tAcc), (TILE, HEADS))
            t2r = tcgen05.make_tmem_copy(ld_atom, acc_cols[(None, None, 0, 0, 0)])
            thr = t2r.get_slice(lt)
            tAll = thr.partition_S(acc_cols)
            cD = thr.partition_D(cute.make_identity_tensor((TILE, HEADS)))
            out_base = logits.iterator.toint()
            tops_base = tops.iterator.toint()
            while g < hi:
                req, k0, ntile, g, r = _next_item(sStart, requests, g, r, hi)
                tiles = sStart[req + 1] - sStart[req]
                row0 = Int64(sStart[req]) * Int64(TILE * Q)
                pitch = Int64(tiles) * Int64(TILE)
                qs = j % 2
                cute.arch.mbarrier_wait(q_full + qs, (j // 2) & 1)
                for t in cutlass.range(ntile, unroll=1):
                    if it % ACCS == acc:
                        stage = it % self.stages
                        key = k0 + t * TILE + lt
                        phase = (it // self.stages) & 1
                        cute.arch.mbarrier_wait(kv_full + stage, phase)
                        kscale = bits_float(Uint32(sSc[lt, stage]))
                        cute.arch.mbarrier_arrive(sc_empty + stage)
                        cute.arch.mbarrier_wait(acc_full + acc, (it // ACCS) & 1)
                        # tcgen05.ld is .sync.aligned: reconverge the warp after
                        # the per-thread barrier wait.
                        cute.arch.sync_warp()
                        # Order the TMEM loads after the MMAs' commit.
                        fence_tmem()
                        rows = row0, pitch, key, kscale, qs, acc, lt
                        bases = out_base, tops_base, acc_empty + acc
                        self.rows_out(t2r, tAll, cD, sW, sEnd, rows, bases)
                    it += 1
                cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.mbarrier_arrive(q_empty + qs)
                j += 1

        cute.arch.sync_threads()
        if warp == MMA_WARP:
            fence_tmem()
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(tmem_ptr, self.columns)


# --------------------------------------------------------------------------
# Select
# --------------------------------------------------------------------------
class IndexerSelect:
    """The exact top-``topk`` of every row (one CTA per row) -> ``out`` [rows,
    topk]: the global slots (``page * 64 + offset`` through the block table) of
    the selected keys in key order, -1 past the row's count, and ``lens``
    [rows]: the counts.

    The K-th largest of the scorer's per-8 vector maxima bounds the K-th largest
    key from below (each of those vectors holds a key at least as large), so only
    the keys at or above it, about K of them, are candidates. Rows of fewer than
    2K vectors bound from their keys instead. A three-digit radix select over the
    candidates finds the threshold; rows whose candidates overflow the shared
    lists take the same select over every key."""

    def __init__(self, q: int, topk: int, pdl: bool):
        assert 1 <= q <= 8 and topk % SELECT_THREADS == 0 and 0 < topk <= 4096
        self.q, self.topk, self.pdl = q, topk, pdl

    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        tops: cute.Tensor,
        seq_lens: cute.Tensor,
        table: cute.Tensor,
        out: cute.Tensor,
        lens: cute.Tensor,
        stream,
    ):
        self.kernel(logits, tops, seq_lens, table, out, lens).launch(
            grid=(out.shape[0], 1, 1),
            block=(SELECT_THREADS, 1, 1),
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.jit
    def emit(self, out, table, request, row, pos, key):
        """Key ``key``'s global slot at position ``pos`` of the row."""
        page = table_entry(table, request, key >> 6)
        out[row, pos] = page * Int32(PAGE) + (key & Int32(PAGE - 1))

    @cute.jit
    def top_of(self, tops_s, tops_address, v):
        """Vector ``v``'s largest key: the first TOPS are in shared memory."""
        top = Uint32(0)
        if v < TOPS:  # noqa: SIM108 (a branch: the global load stays conditional)
            top = tops_s[v]
        else:
            top = load_u32(tops_address + Int64(v) * 4)
        return top

    @cute.jit
    def clear(self, hist, info, tid):
        """Zero the histogram and preset the threshold slots, once every thread
        has read the previous threshold."""
        cute.arch.sync_threads()
        for b in cutlass.range(tid, BINS, SELECT_THREADS, unroll=1):
            hist[b] = Uint32(0)
        if tid == 0:
            info[2] = Int32(0)
            info[3] = Int32(0)
        cute.arch.sync_threads()

    @cute.jit
    def digit(
        self,
        sm,
        use_list,
        listed,
        address,
        length,
        vectors,
        shift: cutlass.Constexpr[int],
        mask: cutlass.Constexpr[int],
        match_shift: cutlass.Constexpr[int],
        match,
        tid,
    ):
        """Histogram of the digit ``(key >> shift) & mask`` over the candidates
        (``use_list``) or every key of the row whose bits above ``match_shift``
        are ``match`` (all keys when ``match_shift`` is 32)."""
        hist, info, cand_key = sm[0], sm[3], sm[5]
        self.clear(hist, info, tid)
        if use_list:
            for i in cutlass.range(tid, listed, SELECT_THREADS, unroll=1):
                key = cand_key[i]
                hit = Int32(1)
                if cutlass.const_expr(match_shift < 32):
                    if (key >> Uint32(match_shift)) != match:
                        hit = Int32(0)
                if hit != Int32(0):
                    shared_add(hist.iterator + ((key >> Uint32(shift)) & mask), 1)
        else:
            for v in cutlass.range(tid, vectors, SELECT_THREADS, unroll=1):
                keys = vector_keys(address, v, length)
                for e in cutlass.range_constexpr(8):
                    key = keys[e]
                    hit = Int32(0)
                    if v * 8 + e < length:
                        hit = Int32(1)
                    if cutlass.const_expr(match_shift < 32):
                        if (key >> Uint32(match_shift)) != match:
                            hit = Int32(0)
                    if hit != Int32(0):
                        bin_ = (key >> Uint32(shift)) & mask
                        shared_add(hist.iterator + bin_, 1)
        cute.arch.sync_threads()

    @cute.jit
    def write_selected(
        self, key, index, n, threshold, take, carry_a, carry_t, lane, emit_args
    ):
        """One warp round of the ordered output: ``n`` (0/1) per lane of key
        ``key`` at ``index``; returns the carries."""
        out, table, request, row = emit_args
        n_a = Uint32(0)
        n_t = Uint32(0)
        if n != Uint32(0):
            if key > threshold:
                n_a = Uint32(1)
            elif key == threshold:
                n_t = Uint32(1)
        packed = (n_a << Uint32(16)) | n_t
        inclusive = warp_inclusive(packed, lane)
        exclusive = inclusive - packed
        total = cute.arch.shuffle_sync(inclusive, 31)
        pos_a = carry_a + (exclusive >> Uint32(16))
        pos_t = carry_t + (exclusive & Uint32(0xFFFF))
        if n_a != Uint32(0):
            pos = Int32(pos_a + cutlass.min(pos_t, take))
            self.emit(out, table, request, row, pos, index)
        if n_t != Uint32(0) and pos_t < take:
            pos = Int32(pos_a + pos_t)
            self.emit(out, table, request, row, pos, index)
        return carry_a + (total >> Uint32(16)), carry_t + (total & Uint32(0xFFFF))

    @cute.jit
    def warp_carries(self, counts, n_a, n_t, warp, lane):
        """This warp's keys above the threshold and ties at it -> the counts of
        the warps before it (in key order)."""
        n_a = warp_sum(n_a)
        n_t = warp_sum(n_t)
        if lane == 0:
            counts[0, warp] = n_a
            counts[1, warp] = n_t
        cute.arch.sync_threads()
        x_a = counts[0, lane]
        x_t = counts[1, lane]
        inc_a = warp_inclusive(x_a, lane)
        inc_t = warp_inclusive(x_t, lane)
        carry_a = cute.arch.shuffle_sync(inc_a - x_a, warp)
        carry_t = cute.arch.shuffle_sync(inc_t - x_t, warp)
        return carry_a, carry_t

    @cute.jit
    def emit_list(self, sm, listed, threshold, take, emit_args, warp, lane):
        """The selected candidates' slots in key order; each warp owns one
        contiguous share of the list."""
        counts, cand_index, cand_key = sm[2], sm[4], sm[5]
        share = (listed + SELECT_WARPS - 1) // SELECT_WARPS
        first = cutlass.min(warp * share, listed)
        last = cutlass.min(first + share, listed)
        n_a = Uint32(0)
        n_t = Uint32(0)
        for i in cutlass.range(first + lane, last, 32, unroll=1):
            key = cand_key[i]
            if key > threshold:
                n_a += Uint32(1)
            elif key == threshold:
                n_t += Uint32(1)
        carry_a, carry_t = self.warp_carries(counts, n_a, n_t, warp, lane)
        for start in cutlass.range(first, last, 32, unroll=1):
            i = start + lane
            key = Uint32(0)
            index = Int32(0)
            n = Uint32(0)
            if i < last:
                key = cand_key[i]
                index = cand_index[i]
                n = Uint32(1)
            carry_a, carry_t = self.write_selected(
                key, index, n, threshold, take, carry_a, carry_t, lane, emit_args
            )

    @cute.jit
    def emit_row(
        self, sm, address, length, vectors, threshold, take, emit_args, warp, lane
    ):
        """The exact select's ordered output over every key of the row: each warp
        owns one contiguous run of vectors, a lane one vector per round."""
        counts = sm[2]
        per_warp = (vectors + SELECT_WARPS - 1) // SELECT_WARPS
        w0 = cutlass.min(warp * per_warp, vectors)
        w1 = cutlass.min(w0 + per_warp, vectors)
        n_a = Uint32(0)
        n_t = Uint32(0)
        for v in cutlass.range(w0 + lane, w1, 32, unroll=1):
            keys = vector_keys(address, v, length)
            for e in cutlass.range_constexpr(8):
                if v * 8 + e < length:
                    if keys[e] > threshold:
                        n_a += Uint32(1)
                    elif keys[e] == threshold:
                        n_t += Uint32(1)
        carry_a, carry_t = self.warp_carries(counts, n_a, n_t, warp, lane)
        out, table, request, row = emit_args
        for base in cutlass.range(w0, w1, 32, unroll=1):
            v = base + lane
            mine = Int32(0)
            if v < w1:
                mine = Int32(1)
            # Lanes past the run read the run's last vector and count nothing.
            keys = vector_keys(address, cutlass.min(v, w1 - 1), length)
            v_a = Uint32(0)
            v_t = Uint32(0)
            for e in cutlass.range_constexpr(8):
                if mine != Int32(0) and v * 8 + e < length:
                    if keys[e] > threshold:
                        v_a += Uint32(1)
                    elif keys[e] == threshold:
                        v_t += Uint32(1)
            packed = (v_a << Uint32(16)) | v_t
            inclusive = warp_inclusive(packed, lane)
            exclusive = inclusive - packed
            total = cute.arch.shuffle_sync(inclusive, 31)
            pos_a = carry_a + (exclusive >> Uint32(16))
            pos_t = carry_t + (exclusive & Uint32(0xFFFF))
            if packed != Uint32(0):
                for e in cutlass.range_constexpr(8):
                    if v * 8 + e < length:
                        key = keys[e]
                        if key > threshold:
                            pos = Int32(pos_a + cutlass.min(pos_t, take))
                            self.emit(out, table, request, row, pos, v * 8 + e)
                            pos_a += Uint32(1)
                        elif key == threshold:
                            if pos_t < take:
                                pos = Int32(pos_a + pos_t)
                                self.emit(out, table, request, row, pos, v * 8 + e)
                            pos_t += Uint32(1)
            carry_a += total >> Uint32(16)
            carry_t += total & Uint32(0xFFFF)

    @cute.jit
    def bound(self, sm, address, tops_address, length, vectors, dense, tid, warp, lane):
        """A lower bound of the row's K-th largest key: the top 22 bits of the K-th
        largest vector maximum (``dense``: of the K-th largest key). Fills the
        vector maxima's shared copy."""
        hist, scan, _, info = sm[0], sm[1], sm[2], sm[3]
        tops_s = sm[7]
        K = self.topk
        self.clear(hist, info, tid)
        if dense:
            for v in cutlass.range(tid, vectors, SELECT_THREADS, unroll=1):
                keys = vector_keys(address, v, length)
                tops_s[v] = vector_max(keys)
                for e in cutlass.range_constexpr(8):
                    if v * 8 + e < length:
                        shared_add(hist.iterator + (keys[e] >> Uint32(21)), 1)
        else:
            for v in cutlass.range(tid, vectors, SELECT_THREADS, unroll=1):
                top = load_u32(tops_address + Int64(v) * 4)
                if v < TOPS:
                    tops_s[v] = top
                shared_add(hist.iterator + (top >> Uint32(21)), 1)
        cute.arch.sync_threads()
        topdown(hist, BINS, Uint32(K), info, scan, tid, warp, lane)
        b0 = Uint32(info[2])
        above = Uint32(info[3])
        self.clear(hist, info, tid)
        if dense:
            for v in cutlass.range(tid, vectors, SELECT_THREADS, unroll=1):
                keys = vector_keys(address, v, length)
                for e in cutlass.range_constexpr(8):
                    key = keys[e]
                    if v * 8 + e < length and (key >> Uint32(21)) == b0:
                        bin_ = (key >> Uint32(10)) & Uint32(BINS - 1)
                        shared_add(hist.iterator + bin_, 1)
        else:
            for v in cutlass.range(tid, vectors, SELECT_THREADS, unroll=1):
                top = self.top_of(tops_s, tops_address, v)
                if (top >> Uint32(21)) == b0:
                    bin_ = (top >> Uint32(10)) & Uint32(BINS - 1)
                    shared_add(hist.iterator + bin_, 1)
        cute.arch.sync_threads()
        topdown(hist, BINS, Uint32(K) - above, info, scan, tid, warp, lane)
        b1 = Uint32(info[2])
        return (b0 << Uint32(21)) | (b1 << Uint32(10))

    @cute.jit
    def gather(self, sm, address, tops_address, length, vectors, low, tid, warp, lane):
        """The candidates (keys >= ``low``) into the shared lists in key order;
        returns their count (more than CANDIDATES: they did not fit)."""
        scan, cand_index, cand_key = sm[1], sm[4], sm[5]
        hot_list, tops_s = sm[6], sm[7]
        # Lane runs (in key order); an odd run spreads the lanes' shared reads
        # over the banks.
        run = (vectors + SELECT_THREADS - 1) // SELECT_THREADS
        run = run + 1 - run % 2
        l0 = cutlass.min(tid * run, vectors)
        l1 = cutlass.min(l0 + run, vectors)
        count = Uint32(0)
        for v in cutlass.range(l0, l1, unroll=1):
            if self.top_of(tops_s, tops_address, v) >= low:
                count += Uint32(1)
        slot, hot = block_exclusive(count, scan, warp, lane)
        listed = Int32(CANDIDATES + 1)
        if hot <= Uint32(HOT):
            for v in cutlass.range(l0, l1, unroll=1):
                if self.top_of(tops_s, tops_address, v) >= low:
                    hot_list[slot] = v
                    slot += Uint32(1)
            cute.arch.sync_threads()
            # Count each run's keys >= low, then reload it to write them (no spill).
            hot_count = Int32(hot)
            per = (hot_count + SELECT_THREADS - 1) // SELECT_THREADS
            h0 = cutlass.min(tid * per, hot_count)
            h1 = cutlass.min(h0 + per, hot_count)
            count = Uint32(0)
            for u in cutlass.range_constexpr(HOT // SELECT_THREADS):
                if h0 + u < h1:
                    v = hot_list[h0 + u]
                    keys = vector_keys(address, v, length)
                    for e in cutlass.range_constexpr(8):
                        if keys[e] >= low and v * 8 + e < length:
                            count += Uint32(1)
            slot, total = block_exclusive(count, scan, warp, lane)
            listed = Int32(total)
            if total <= Uint32(CANDIDATES):
                for u in cutlass.range_constexpr(HOT // SELECT_THREADS):
                    if h0 + u < h1:
                        v = hot_list[h0 + u]
                        keys = vector_keys(address, v, length)
                        for e in cutlass.range_constexpr(8):
                            key = keys[e]
                            if key >= low and v * 8 + e < length:
                                cand_index[slot] = v * 8 + e
                                cand_key[slot] = key
                                slot += Uint32(1)
        cute.arch.sync_threads()
        return listed

    @cute.jit
    def select_row(self, sm, logits, tops, seq_lens, emit_args, j, length, cap):
        """The exact top-K of a row longer than K."""
        hist, scan, _, info = sm[0], sm[1], sm[2], sm[3]
        Q, K = self.q, self.topk
        _, _, request, _ = emit_args
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        # This row's logits: request ``request``'s row j of the scratch.
        if warp == 0:
            before = Int32(0)
            for base in cutlass.range(0, request, 32, unroll=1):
                r = base + lane
                tiles = Int32(0)
                if r < request:
                    tiles = request_tiles(seq_lens, r, cap, K)
                for i in cutlass.range_constexpr(5):
                    tiles += cute.arch.shuffle_sync_bfly(tiles, 1 << i)
                before += tiles
            if lane == 0:
                info[0] = before
                info[1] = request_tiles(seq_lens, request, cap, K)
        cute.arch.sync_threads()
        element = Int64(info[0]) * Int64(TILE * Q)
        element += Int64(j) * Int64(info[1]) * Int64(TILE)
        address = logits.iterator.toint() + element * 4
        tops_address = tops.iterator.toint() + (element >> 3) * 4
        vectors = (length + 7) // 8
        # Under 2K vectors the K-th largest maximum bounds loosely: bound from the
        # keys themselves.
        dense = vectors < 2 * K
        low = self.bound(
            sm, address, tops_address, length, vectors, dense, tid, warp, lane
        )
        listed = self.gather(
            sm, address, tops_address, length, vectors, low, tid, warp, lane
        )
        use_list = listed <= Int32(CANDIDATES)
        # The threshold: three radix digits over the candidates (or every key).
        need = Uint32(K)
        self.digit(
            sm, use_list, listed, address, length, vectors, 21, 0x7FF, 32, 0, tid
        )
        topdown(hist, BINS, need, info, scan, tid, warp, lane)
        c0 = Uint32(info[2])
        need -= Uint32(info[3])
        self.digit(
            sm, use_list, listed, address, length, vectors, 10, 0x7FF, 21, c0, tid
        )
        topdown(hist, BINS, need, info, scan, tid, warp, lane)
        c1 = Uint32(info[2])
        need -= Uint32(info[3])
        prefix = (c0 << Uint32(11)) | c1
        self.digit(
            sm, use_list, listed, address, length, vectors, 0, 0x3FF, 10, prefix, tid
        )
        topdown(hist, LOW_BINS, need, info, scan, tid, warp, lane)
        c2 = Uint32(info[2])
        need -= Uint32(info[3])
        threshold = (prefix << Uint32(10)) | c2
        # The ties at the threshold to take (the lowest keys).
        take = need
        if use_list:
            self.emit_list(sm, listed, threshold, take, emit_args, warp, lane)
        else:
            self.emit_row(
                sm, address, length, vectors, threshold, take, emit_args, warp, lane
            )

    @cute.kernel
    def kernel(self, logits, tops, seq_lens, table, out, lens):
        Q, K = self.q, self.topk
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        request = row // Q
        j = row - request * Q

        smem = utils.SmemAllocator()
        hist = smem.allocate_tensor(Uint32, cute.make_layout(BINS), 16)
        scan = smem.allocate_tensor(Uint32, cute.make_layout(SELECT_WARPS), 16)
        counts = smem.allocate_tensor(Uint32, cute.make_layout((2, SELECT_WARPS)), 16)
        info = smem.allocate_tensor(Int32, cute.make_layout(8), 16)
        cand_index = smem.allocate_tensor(Int32, cute.make_layout(CANDIDATES), 16)
        cand_key = smem.allocate_tensor(Uint32, cute.make_layout(CANDIDATES), 16)
        hot_list = smem.allocate_tensor(Int32, cute.make_layout(HOT), 16)
        tops_s = smem.allocate_tensor(Uint32, cute.make_layout(TOPS), 16)
        sm = hist, scan, counts, info, cand_index, cand_key, hot_list, tops_s

        # No early release: the sparse core reads the counts before its PDL wait.
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()
        cap = table.shape[1] * Int32(PAGE)
        length = clamp_end(seq_lens[request] - (Q - 1) + j, cap)
        emit_args = out, table, request, row
        if length <= K:
            # Short-context bypass: every key, no logits.
            for e in cutlass.range(tid, K, SELECT_THREADS, unroll=1):
                if e < length:
                    self.emit(out, table, request, row, e, e)
                else:
                    out[row, e] = Int32(-1)
            if tid == 0:
                lens[row] = length
        else:
            self.select_row(sm, logits, tops, seq_lens, emit_args, j, length, cap)
            if tid == 0:
                lens[row] = Int32(K)


def _fake(dtype, shape, stride, align):
    return cute.runtime.make_fake_tensor(
        dtype, shape, stride=stride, assumed_align=align
    )


@functools.cache
def compile_indexer(q: int, topk: int, ctas: int, pdl: bool):
    """The (score, select) kernels for ``q`` rows per request, ``topk`` selected
    keys and a ``ctas``-CTA persistent scorer. Rows, requests, slots, block-table
    width and strides, and the scratch sizes are runtime values: one compile
    serves every batch."""
    sym = cute.sym_int
    query = _fake(Uint8, (sym(), DIM), (DIM, 1), 16)
    weights = _fake(Float32, (sym(),), (1,), 16)
    cache = _fake(Uint8, (sym(), ROW_BYTES), (ROW_BYTES, 1), 16)
    table = _fake(Int32, (sym(), sym()), (cute.sym_int64(), 1), 4)
    seq_lens = _fake(Int32, (sym(),), (1,), 4)
    logits = _fake(Float32, (sym(),), (1,), 16)
    tops = _fake(Int32, (sym(),), (1,), 16)
    out = _fake(Int32, (sym(), topk), (cute.sym_int64(), 1), 4)
    lens = _fake(Int32, (sym(),), (1,), 4)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    options = "--enable-tvm-ffi"
    score = cute.compile(
        IndexerScore(q, topk, ctas, pdl),
        query,
        weights,
        cache,
        table,
        seq_lens,
        logits,
        tops,
        stream,
        options=options,
    )
    select = cute.compile(
        IndexerSelect(q, topk, pdl),
        logits,
        tops,
        seq_lens,
        table,
        out,
        lens,
        stream,
        options=options,
    )
    return score, select
