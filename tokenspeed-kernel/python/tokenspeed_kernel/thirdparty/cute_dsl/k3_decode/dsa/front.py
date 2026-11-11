# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Vendored from vLLM's Kimi-K3.1 specialized decode (cutedsl/dsa/front.py): the cluster
# IndexFront, whose 4-CTA cluster holds one indexer head's wq_b rows in K quarters,
# loads them before the PDL wait and reduces the head's partial projections at row
# owners. Modified to replace a TokenSpeed decode round's q_a / kv_a RMSNorm, stacked
# [q_b; wq_b] projection and index-K write with query quantization:
# - Each cluster also projects its share of the q_b rows (72 at TP8), so 32 clusters
#   cover the stacked weight at one CTA per SM.
# - The q_a RMSNorm reproduces RMSNormFusedParallel (thirdparty/cuda, built with
#   -use_fast_math) bit for bit: each CTA sends its quarter's warp-group sum and
#   half-group FMA chains to its peers over DSMEM, and every CTA rebuilds that kernel's
#   reduction tree. The kv_a RMSNorm runs in place in the same arithmetic.
# - The index query is quantized as quantize_index_query_weights does (TRT-LLM's 1x128
#   scale amax / 448, its IEEE reciprocal, FP32 weights folding the scale and the
#   score scale), not with a power-of-two scale into BF16 weights.
# - The index-K write repeats tokenspeed_triton's instruction sequence for
#   index_k_layernorm_fp8_scatter (in-thread pairs, butterflies, approximate square
#   root and full-range divisions) into the packed fp8_scaled plane.
# - The weights stream in 64-column K blocks (cp.async, L2 evict_first) with one
#   barrier per block; the partials overwrite the weight buffer and owners pull them
#   over DSMEM after a cluster barrier instead of receiving bulk copies.
# - The row count is a runtime value (up to 16 rows per compiled M tile).

"""Fused DSA decode front of a Kimi-K3.1 TP8 attention layer in one cluster launch:

  q_norm    = RMSNorm(q_a)                   RMSNormFusedParallel's bits
  kv_a      = RMSNorm(kv_a), in place        RMSNormFusedParallel's bits
  projected = q_norm @ [q_b; wq_b]^T         FP32 accumulation, rounded to BF16
  q_fp8, w  = quantize_index_query_weights(projected[:, q_b rows:], head weights)
  plane     = index_k_layernorm_fp8_scatter(key, slots)

Grid (4, 32), clusters of 4: cluster ``head`` owns index head ``head`` (its 128 wq_b
rows) and q_b rows ``[share * head, share * (head + 1))``; CTA ``part`` holds K
columns ``[384 * part, 384 * (part + 1))`` of those rows in shared memory, loaded
before the PDL wait. Eight warps run the norm, the MMAs and the reduction; a ninth
writes token ``4 * head + part``'s index key and normalizes its kv_a row.

RMSNormFusedParallel sums a q_a row's squares over 192 threads of 8 elements: an FMA
chain per thread, a butterfly per warp of 256 columns, and the six warp sums as
((w0 + w4) + w2) + ((w1 + w5) + w3). A K quarter holds one whole warp group and half
of another, so each CTA sends every peer its whole group's sum and its 16 half-group
chains, and every CTA rebuilds the same total. The quarter is normalized in place and
the BF16 MMAs (FP32 accumulators, K permuted alike in A and B) consume each 64-column
weight block as it lands. The [rows, 200] FP32 partials then overwrite the weight
buffer; after a cluster barrier the owner of a row (a quarter of the rows per CTA)
adds the four quarters in rank order, rounds to BF16 and quantizes the head.
"""

import functools

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32, Uint64
from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.common.primitives import (
    add_rn,
    asm,
    at,
    bf16x2,
    float_bits,
    fma,
    launch,
    load_bf16x8,
    load_words,
    mma_into,
    named_barrier,
    pack_fp8,
    smem_words,
    stage_weight,
    store_shared_words,
    store_words,
    unpack_bf16,
    wait,
    zeros,
)

CLUSTER = 4
K = 1536  # q_lora_rank
KQ = K // CLUSTER  # K columns per CTA
CHUNKS = KQ // 8  # 16-byte chunks of a row quarter
KBLOCK = 64  # K columns per weight block
KBLOCKS = KQ // KBLOCK
HEADS, DIM = 32, 128  # index heads, index head dim
INDEX_ROWS = HEADS * DIM
LATENT = 512  # kv_lora_rank
PAGE = 64  # index-K slots per page
ROW_BYTES = DIM + 4  # a slot's FP8 key bytes, then (in the page tail) its FP32 scale
MAIN_WARPS = 8
MAIN_THREADS = 32 * MAIN_WARPS
THREADS = MAIN_THREADS + 32
SS_ROW = 80  # bytes per row of a CTA's norm message: group sum, 16 chains, padding
SMEM_LIMIT = 232448  # sm_100 per-CTA shared memory
MAX_ROWS = 64


# RMSNormFusedParallel's arithmetic (nvcc -use_fast_math).
@asm(Float32, pure=True)
def fma_ftz(a, b, c):
    return "fma.rn.ftz.f32 $0,$1,$2,$3;", "=f,f,f,f", (a, b, c)


@asm(Float32, pure=True)
def add_ftz(a, b):
    return "add.ftz.f32 $0,$1,$2;", "=f,f,f", (a, b)


@asm(Float32, pure=True)
def mul_ftz(a, b):
    return "mul.ftz.f32 $0,$1,$2;", "=f,f,f", (a, b)


@asm(Float32, pure=True)
def norm_rsqrt(total, count, eps):
    """rsqrt(total / count + eps) as nvcc -use_fast_math compiles it, ``count`` a
    runtime operand as in the CUDA kernel."""
    code = "{ .reg .f32 m; div.approx.ftz.f32 m,$1,$2; add.ftz.f32 m,$3,m; "
    code += "rsqrt.approx.ftz.f32 $0,m; }"
    return code, "=f,f,f,f", (total, count, eps)


# tokenspeed_triton's index-K write and query quantization arithmetic.
@asm(Float32, pure=True)
def sub_rn(a, b):
    return "sub.rn.f32 $0,$1,$2;", "=f,f,f", (a, b)


@asm(Float32, pure=True)
def mul_rn(a, b):
    return "mul.rn.f32 $0,$1,$2;", "=f,f,f", (a, b)


@asm(Float32, pure=True)
def key_rstd(total, eps):
    """1 / sqrt(total / 128 + eps): an FMA, the approximate square root and the
    full-range division."""
    code = "{ .reg .f32 v,s,one; fma.rn.f32 v,$1,0f3C000000,$2; "
    code += "sqrt.approx.ftz.f32 s,v; mov.b32 one,0f3F800000; div.full.f32 $0,one,s; }"
    return code, "=f,f,f", (total, eps)


@asm(Float32, pure=True)
def key_scale(amax):
    """max(amax, 1e-10) * fp32(1 / 448)."""
    code = "{ .reg .f32 m; max.f32 m,$1,0f2EDBE6FF; mul.f32 $0,m,0f3B124925; }"
    return code, "=f,f", (amax,)


@asm(Float32, pure=True)
def div_full(a, b):
    return "div.full.f32 $0,$1,$2;", "=f,f,f", (a, b)


@asm(Float32, pure=True)
def div_rn(a, b):
    return "div.rn.f32 $0,$1,$2;", "=f,f,f", (a, b)


@asm(Float32, pure=True)
def rcp_rn(a):
    return "rcp.rn.f32 $0,$1;", "=f,f", (a,)


@asm(Float32, pure=True)
def clamp_fp8(value, limit):
    """sign(value) * min(|value|, limit): Triton's tl.clamp(value, -limit, limit)."""
    return "min.xorsign.abs.f32 $0,$1,$2;", "=f,f,f", (value, limit)


@asm(Float32, pure=True)
def max_f32(a, b):
    return "max.f32 $0,$1,$2;", "=f,f,f", (a, b)


@asm(Float32, pure=True)
def max3_f32(a, b, c):
    return "max.f32 $0,$1,$2,$3;", "=f,f,f,f", (a, b, c)


@asm(Float32, pure=True)
def abs_f32(a):
    return "abs.f32 $0,$1;", "=f,f", (a,)


@asm(Float32)
def redux_max(value):
    return "redux.sync.max.f32 $0,$1,0xffffffff;", "=f,f", (value,)


# Memory.
@asm()
def load_chunk(pointer, live):
    """The 16 bytes at global ``pointer`` as four words, zeros where ``live`` is 0."""
    code = "{ .reg .pred p; setp.ne.b32 p,$5,0; mov.b32 $0,0; mov.b32 $1,0; "
    code += "mov.b32 $2,0; mov.b32 $3,0; @p ld.global.cg.v4.b32 {$0,$1,$2,$3},[$4]; }"
    return code, "=r,=r,=r,=r,l,r,~{memory}", (pointer, Int32(live)), (Uint32,) * 4


@asm()
def load_bf16(pointer, live):
    """The BF16 at global ``pointer`` as FP32, zero where ``live`` is 0."""
    code = "{ .reg .pred p; .reg .b16 h,z; setp.ne.b32 p,$2,0; mov.b16 h,0; "
    code += "mov.b16 z,0; @p ld.global.cg.b16 h,[$1]; mov.b32 $0,{z,h}; }"
    return code, "=f,l,r,~{memory}", (pointer, Int32(live)), Float32


@asm()
def cp_async_chunk(dst, src):
    """A 16-byte cp.async (L2 only)."""
    return "cp.async.cg.shared.global [$0],[$1],16;", "r,l,~{memory}", (dst, src)


@asm()
def load_slot(pointer, *, wide):
    """The int64 (``wide``) or int32 slot at global ``pointer``, as int64."""
    if wide:
        return "ld.global.cg.b64 $0,[$1];", "=l,l,~{memory}", (pointer,), Int64
    code = "{ .reg .b32 s; ld.global.cg.b32 s,[$1]; cvt.s64.s32 $0,s; }"
    return code, "=l,l,~{memory}", (pointer,), Int64


@asm()
def cp_async_arrive(barrier):
    """Arrive on ``barrier`` once this thread's earlier cp.async copies land."""
    code = "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [$0];"
    return code, "r,~{memory}", (barrier,)


@asm()
def push_rows(pointer, barrier, peer, size):
    """Bulk copy ``size`` bytes at this CTA's shared ``pointer`` to the same address
    of CTA ``peer``, completing on that CTA's ``barrier``."""
    code = "{ .reg .b32 d,b; mapa.shared::cluster.u32 d,$0,$2; "
    code += "mapa.shared::cluster.u32 b,$1,$2; cp.async.bulk.shared::cluster"
    code += ".shared::cta.mbarrier::complete_tx::bytes [d],[$0],$3,[b]; }"
    return code, "r,r,r,r,~{memory}", (pointer, barrier, Int32(peer), Int32(size))


@asm()
def load_peer(pointer, peer):
    """Four FP32 at shared ``pointer`` of cluster CTA ``peer``."""
    code = "{ .reg .b32 a; mapa.shared::cluster.u32 a,$4,$5; "
    code += "ld.shared::cluster.v4.f32 {$0,$1,$2,$3},[a]; }"
    return code, "=f,=f,=f,=f,r,r,~{memory}", (pointer, Int32(peer)), (Float32,) * 4


def load_shared_f32(pointer):
    (value,) = load_words(
        pointer, count=1, shared=True, result_type=Float32, word_type="f32"
    )
    return value


def butterfly_ftz(value):
    """RMSNormFusedParallel's warp sum: offsets 16 down to 1, add.ftz."""
    for shift in (16, 8, 4, 2, 1):
        value = add_ftz(value, cute.arch.shuffle_sync_bfly(value, shift))
    return value


def butterfly_rn(value):
    """Triton's one-warp sum of a [1, 128] row: offsets 16 down to 1."""
    for shift in (16, 8, 4, 2, 1):
        value = add_rn(value, cute.arch.shuffle_sync_bfly(value, shift))
    return value


def butterfly_max(value):
    for shift in (16, 8, 4, 2, 1):
        value = max_f32(value, cute.arch.shuffle_sync_bfly(value, shift))
    return value


def square_chain(words):
    """One RMSNormFusedParallel thread's sum of squares of 8 BF16 (four words)."""
    total = Float32(0)
    for word in words:
        lo, hi = unpack_bf16(word)
        total = fma_ftz(hi, hi, fma_ftz(lo, lo, total))
    return total


def rms_scale(words, rr, gamma):
    """BF16 (x * rr) * gamma of 8 BF16 (four words), as RMSNormFusedParallel stores."""
    out = []
    for i, word in enumerate(words):
        lo, hi = unpack_bf16(word)
        low = mul_ftz(mul_ftz(lo, rr), gamma[2 * i])
        high = mul_ftz(mul_ftz(hi, rr), gamma[2 * i + 1])
        out.append(bf16x2(low, high))
    return tuple(out)


def swizzled(row, chunk):
    """Byte offset of 16-byte ``chunk`` of ``row`` within a K block or a row quarter:
    odd rows swap the 64-byte halves of every 128 bytes, so the fragment loads of a
    quarter warp (two rows, four chunks) hit distinct banks."""
    return (chunk ^ ((row & 1) << 2)) * 16


class DecodeFront:
    """The front of up to ``16 * tiles`` rows. ``query_rows``: the q_b rows ahead of
    the 4096 wq_b rows in the stacked weight; ``wide_slots``: int64 (else int32)
    slots; ``key_norm_fp32``: FP32 (else BF16) index-key LayerNorm parameters;
    ``pdl``: launch with programmatic dependent launch."""

    def __init__(
        self,
        tiles: int,
        query_rows: int,
        wide_slots: bool,
        key_norm_fp32: bool,
        pdl: bool,
    ):
        if not 1 <= tiles <= MAX_ROWS // 16:
            raise ValueError(f"tiles must be 1-{MAX_ROWS // 16}, got {tiles}")
        if query_rows <= 0 or query_rows % (8 * HEADS):
            raise ValueError(f"{query_rows} q_b rows do not split into 32 8-row shares")
        self.tiles, self.rows = tiles, 16 * tiles
        self.query_rows, self.share = query_rows, query_rows // HEADS
        self.n = DIM + self.share  # weight rows per CTA
        self.blocks = self.n // 8  # n8 MMA blocks per CTA
        self.slots = -(-self.blocks // MAIN_WARPS)  # n8 blocks per warp, at most
        self.wide_slots, self.key_norm_fp32, self.pdl = wide_slots, key_norm_fp32, pdl
        self.w_bytes = KBLOCKS * self.n * KBLOCK * 2
        self.x_bytes = self.rows * KQ * 2
        self.ss_bytes = CLUSTER * self.rows * SS_ROW
        self.g_bytes = KQ * 2  # this CTA's q_a RMSNorm weight columns
        if self.rows * self.n * 4 > self.w_bytes:
            raise ValueError("the partial sums do not fit the weight buffer")
        used = self.w_bytes + self.x_bytes + self.ss_bytes + self.g_bytes
        used += 4 * self.rows
        if used + 8 * (KBLOCKS + 2) + 512 > SMEM_LIMIT:
            raise ValueError(f"{query_rows} q_b rows at {self.rows} rows exceed SMEM")

    @cute.jit
    def __call__(self, *args):
        cluster = (CLUSTER, 1, 1)
        launch(self.kernel, args, (CLUSTER, HEADS, 1), THREADS, cluster, self.pdl)

    @cute.kernel
    def kernel(self, args):
        q_a, kv_a, key, head_w, q_gamma, kv_gamma, weight, k_gamma, k_beta = args[:9]
        cache, slots, q_norm, projected, q_fp8, w_out = args[9:15]
        d_q, d_kv, eps, key_eps, fp8_max, amax_floor, softmax_scale = args[15:]
        tid, _, _ = cute.arch.thread_idx()
        part, head, _ = cute.arch.block_idx()
        warp, lane = tid // 32, tid % 32
        group, quarter = lane // 4, lane % 4
        rows = q_a.shape[0]
        R, N, NB = self.rows, self.n, self.blocks
        smem = cutlass.utils.SmemAllocator()
        w_s = smem.allocate_array(Uint8, self.w_bytes, byte_alignment=128)
        x_s = smem.allocate_array(Uint8, self.x_bytes, byte_alignment=128)
        ss_s = smem.allocate_array(Uint8, self.ss_bytes, byte_alignment=128)
        rr = cute.make_tensor(
            smem.allocate_array(Float32, R, byte_alignment=16), cute.make_layout(R)
        )
        g_s = smem.allocate_array(Uint8, self.g_bytes, byte_alignment=16)
        bars = smem.allocate_array(Uint64, KBLOCKS + 2, byte_alignment=8)
        bar_ss, bar_g = bars + KBLOCKS, bars + KBLOCKS + 1
        if tid == 0:
            for j in cutlass.range_constexpr(KBLOCKS):
                cute.arch.mbarrier_init(bars + j, MAIN_THREADS)
            cute.arch.mbarrier_init(bar_ss, 1)
            cute.arch.mbarrier_init(bar_g, MAIN_THREADS)
            cute.arch.mbarrier_init_fence()
            peer_bytes = (CLUSTER - 1) * rows * SS_ROW
            cute.arch.mbarrier_arrive_and_expect_tx(bar_ss, peer_bytes)
        cute.arch.sync_threads()
        cute.arch.cluster_arrive_relaxed()
        if warp < MAIN_WARPS:
            # The q_a RMSNorm weight columns first (bar_g tracks only them), then
            # this CTA's K quarter of its rows block by block, before the PDL wait.
            if tid < CHUNKS:
                cp_async_chunk(g_s + 16 * tid, at(q_gamma, KQ * part + 8 * tid))
            cp_async_arrive(bar_g)
            per_block = N * KBLOCK // 8
            for j in cutlass.range_constexpr(KBLOCKS):
                for i in cutlass.range_constexpr(-(-per_block // MAIN_THREADS)):
                    c = tid + MAIN_THREADS * i
                    if c < per_block:
                        w_row, w_chunk = c // 8, c % 8
                        source = Int32(self.query_rows) + DIM * head + w_row
                        if w_row >= DIM:
                            source = Int32(self.share) * head + w_row - DIM
                        w_column = KQ * part + KBLOCK * j + 8 * w_chunk
                        w_offset = (j * N + w_row) * (KBLOCK * 2)
                        w_offset += swizzled(w_row, w_chunk)
                        stage_weight(w_s + w_offset, at(weight, source, w_column))
                cp_async_arrive(bars + j)
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

            # Each row's q_a quarter: this CTA's whole warp group of the norm (chunks
            # [0, 32) of an even part, [16, 48) of an odd one) and the 16 chunks of
            # the group it shares with its neighbour; rows past the batch are zeros.
            first = (part & 1) * 16
            second = 32 - 2 * first
            own_ss = ss_s + part * (R * SS_ROW)
            loaded = []
            for i in cutlass.range_constexpr(R // MAIN_WARPS):
                b = warp + MAIN_WARPS * i
                whole = first + lane
                half = second + lane
                row = at(q_a, b, KQ * part)
                xw = load_chunk(row + 8 * whole, b < rows)
                xh = load_chunk(row + 8 * half, (b < rows) & (lane < 16))
                loaded.append((b, whole, half, xw, xh))
            # The head weights of the rows this warp will own (lane 0's).
            owned = (rows + CLUSTER - 1) // CLUSTER
            count = rows - part * owned
            owner_rows = -(-(R // CLUSTER) // MAIN_WARPS)
            head_weights = []
            for i in cutlass.range_constexpr(owner_rows):
                local = warp + MAIN_WARPS * i
                live = (local < owned) & (local < count) & (lane == 0)
                pointer = at(head_w, part * owned + local, head)
                head_weights.append(load_bf16(pointer, live))
            for b, whole, half, xw, xh in loaded:
                x_row = x_s + b * (KQ * 2)
                store_shared_words(x_row + swizzled(b, whole), xw)
                group_sum = butterfly_ftz(square_chain(xw))
                chain = square_chain(xh)
                message = own_ss + b * SS_ROW
                if lane < 16:
                    store_shared_words(x_row + swizzled(b, half), xh)
                    store_shared_words(message + 4 + 4 * lane, (float_bits(chain),))
                if lane == 0:
                    store_shared_words(message, (float_bits(group_sum),))
            cute.arch.fence_proxy("async.shared", space="cta")
            named_barrier(1, MAIN_THREADS)
            cute.arch.cluster_wait()  # the peers' barriers are initialized
            if tid == 0:
                for p in cutlass.range_constexpr(1, CLUSTER):
                    push_rows(own_ss, bar_ss, (part + p) % CLUSTER, rows * SS_ROW)
            wait(bar_ss, 0)

            # Every row's reciprocal RMS: the half groups' butterflies (lanes 0-15
            # from the even part, 16-31 from the odd one), then the six warp sums
            # w0..w5 as RMSNormFusedParallel's cross-warp butterfly adds them.
            for i in cutlass.range_constexpr(R // MAIN_WARPS):
                b = warp + MAIN_WARPS * i
                if b < rows:
                    side = lane // 16
                    slot_chain = 4 + 4 * (lane % 16)
                    w1 = load_shared_f32(ss_s + (side * R + b) * SS_ROW + slot_chain)
                    w4 = ss_s + ((2 + side) * R + b) * SS_ROW + slot_chain
                    w4 = load_shared_f32(w4)
                    w1, w4 = butterfly_ftz(w1), butterfly_ftz(w4)
                    w0, w2, w3, w5 = (
                        load_shared_f32(ss_s + (p * R + b) * SS_ROW) for p in range(4)
                    )
                    total = add_ftz(add_ftz(w0, w4), w2)
                    total = add_ftz(total, add_ftz(add_ftz(w1, w5), w3))
                    if lane == 0:
                        rr[b] = norm_rsqrt(total, Float32(d_q), eps)
            named_barrier(1, MAIN_THREADS)

            # The quarter normalized in place (cluster 0 also stores q_norm).
            wait(bar_g, 0, scope="cta")
            for i in cutlass.range_constexpr(-(-R * CHUNKS // MAIN_THREADS)):
                c = tid + MAIN_THREADS * i
                if c < rows * CHUNKS:
                    nb, nj = c // CHUNKS, c % CHUNKS
                    x_chunk = x_s + nb * (KQ * 2) + swizzled(nb, nj)
                    gamma = load_bf16x8(g_s + 16 * nj, shared=True)
                    normed = rms_scale(smem_words(x_chunk), rr[nb], gamma)
                    store_shared_words(x_chunk, normed)
                    if head == 0:
                        store_words(at(q_norm, nb, KQ * part + 8 * nj), normed)
            named_barrier(1, MAIN_THREADS)

            # [R, 384] x [384, 8] per n8 block; warp w owns blocks w, w + 8, ... A
            # lane's K fragment is 8 contiguous columns in A and B alike.
            acc = zeros((self.tiles * self.slots, 4), Float32)
            for kc in cutlass.range_constexpr(KQ // 32):
                if cutlass.const_expr(kc % 2 == 0):
                    wait(bars + kc // 2, 0, scope="cta")
                frags = []
                for m in cutlass.range_constexpr(self.tiles):
                    r0 = 16 * m + group
                    xa = smem_words(
                        x_s + r0 * (KQ * 2) + swizzled(r0, 4 * kc + quarter)
                    )
                    r1 = r0 + 8
                    ya = smem_words(
                        x_s + r1 * (KQ * 2) + swizzled(r1, 4 * kc + quarter)
                    )
                    frags.append((xa, ya))
                block = (kc // 2) * N * (KBLOCK * 2)
                for n in cutlass.range_constexpr(self.slots):
                    if warp + MAIN_WARPS * n < NB:
                        wr = 8 * (warp + MAIN_WARPS * n) + group
                        z_chunk = 4 * (kc % 2) + quarter
                        z_row = w_s + block + wr * (KBLOCK * 2)
                        z = smem_words(z_row + swizzled(wr, z_chunk))
                        for m in cutlass.range_constexpr(self.tiles):
                            xa, ya = frags[m]
                            for s in cutlass.range_constexpr(2):
                                a = (xa[2 * s], ya[2 * s], xa[2 * s + 1], ya[2 * s + 1])
                                bb = (z[2 * s], z[2 * s + 1])
                                tile = acc[m * self.slots + n, None]
                                mma_into(tile, a, bb, bf16=True)
            named_barrier(1, MAIN_THREADS)  # the weights are consumed

            # The live rows' FP32 partials [rows, N] in place of the weights.
            for m in cutlass.range_constexpr(self.tiles):
                for n in cutlass.range_constexpr(self.slots):
                    if warp + MAIN_WARPS * n < NB:
                        s_column = 8 * (warp + MAIN_WARPS * n) + 2 * quarter
                        for h in cutlass.range_constexpr(2):
                            r = 16 * m + group + 8 * h
                            if r < rows:
                                tile = acc[m * self.slots + n, None]
                                pair = (tile[2 * h], tile[2 * h + 1])
                                s_words = tuple(float_bits(v) for v in pair)
                                s_pointer = w_s + (r * N + s_column) * 4
                                store_shared_words(s_pointer, s_words)
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

            # Owned rows: the four quarters in rank order, BF16, then the head's
            # quantization (quantize_index_query_weights) and the q_b share.
            for i in cutlass.range_constexpr(owner_rows):
                local = warp + MAIN_WARPS * i
                if (local < owned) & (local < count):
                    b = part * owned + local
                    sums = self.reduce4(w_s + (b * N + 4 * lane) * 4)
                    q_words = (bf16x2(sums[0], sums[1]), bf16x2(sums[2], sums[3]))
                    q_column = self.query_rows + DIM * head + 4 * lane
                    store_words(at(projected, b, q_column), q_words)
                    values = (*unpack_bf16(q_words[0]), *unpack_bf16(q_words[1]))
                    amax = max_f32(abs_f32(values[0]), abs_f32(values[1]))
                    amax = max3_f32(amax, abs_f32(values[2]), abs_f32(values[3]))
                    amax = max_f32(butterfly_max(amax), amax_floor)
                    inverse = div_rn(fp8_max, amax)
                    scale = rcp_rn(inverse)
                    codes = tuple(
                        clamp_fp8(mul_rn(v, inverse), fp8_max) for v in values
                    )
                    store_words(at(q_fp8, b, DIM * head + 4 * lane), (pack_fp8(codes),))
                    if lane == 0:
                        fold = mul_rn(mul_rn(head_weights[i], scale), softmax_scale)
                        store_words(at(w_out, b, head), (float_bits(fold),))
                    if lane < self.share // 4:
                        b_sums = self.reduce4(w_s + (b * N + DIM + 4 * lane) * 4)
                        b_words = (
                            bf16x2(b_sums[0], b_sums[1]),
                            bf16x2(b_sums[2], b_sums[3]),
                        )
                        store_words(
                            at(projected, b, self.share * head + 4 * lane), b_words
                        )
        else:
            k_w, k_b, kv_lo, kv_hi = self.aux_weights(k_gamma, k_beta, kv_gamma, lane)
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
            token = head * CLUSTER + part
            if token < rows:
                self.insert_key(key, slots, cache, k_w, k_b, key_eps, token, lane)
                self.kv_norm(kv_a, kv_lo, kv_hi, d_kv, eps, token, lane)
            cute.arch.cluster_wait()
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()  # no CTA exits while a peer may read its partials

    def reduce4(self, pointer):
        """The four K quarters' FP32 at shared ``pointer`` (four columns), added in
        rank order."""
        parts = [load_peer(pointer, p) for p in range(CLUSTER)]
        return [
            add_rn(add_rn(add_rn(parts[0][e], parts[1][e]), parts[2][e]), parts[3][e])
            for e in range(4)
        ]

    def aux_weights(self, k_gamma, k_beta, kv_gamma, lane):
        """Lane ``lane``'s index-key LayerNorm columns [4 lane, 4 lane + 4) and kv_a
        RMSNorm columns [8 lane, 8 lane + 8) and [256 + 8 lane, ...)."""
        if cutlass.const_expr(self.key_norm_fp32):
            options = dict(count=4, result_type=Float32, word_type="f32")
            k_w = load_words(at(k_gamma, 4 * lane), **options)
            k_b = load_words(at(k_beta, 4 * lane), **options)
        else:
            k_w, k_b = [
                [v for word in load_words(at(t, 4 * lane)) for v in unpack_bf16(word)]
                for t in (k_gamma, k_beta)
            ]
        kv_lo = load_bf16x8(at(kv_gamma, 8 * lane))
        kv_hi = load_bf16x8(at(kv_gamma, LATENT // 2 + 8 * lane))
        return k_w, k_b, kv_lo, kv_hi

    @cute.jit
    def insert_key(self, key, slots, cache, k_w, k_b, key_eps, token, lane):
        """index_k_layernorm_fp8_scatter of row ``token`` (one warp, lane ``lane``
        holding columns [4 lane, 4 lane + 4)): the FP32 LayerNorm rounded to BF16,
        then FP8 E4M3 with the scale max(amax, 1e-10) / 448."""
        x = []
        for word in load_words(at(key, token, 4 * lane), count=2, mutable=True):
            x += list(unpack_bf16(word))
        pair_sum = add_rn(add_rn(x[0], x[2]), add_rn(x[1], x[3]))
        mean = mul_rn(butterfly_rn(pair_sum), Float32(2.0**-7))
        c = [sub_rn(v, mean) for v in x]
        square = add_rn(
            fma(c[0], c[0], mul_rn(c[2], c[2])), fma(c[1], c[1], mul_rn(c[3], c[3]))
        )
        rstd = key_rstd(butterfly_rn(square), key_eps)
        y = [fma(mul_rn(c[e], rstd), k_w[e], k_b[e]) for e in range(4)]
        words = (bf16x2(y[0], y[1]), bf16x2(y[2], y[3]))
        y = [*unpack_bf16(words[0]), *unpack_bf16(words[1])]
        amax = max3_f32(
            max_f32(abs_f32(y[0]), abs_f32(y[1])), abs_f32(y[2]), abs_f32(y[3])
        )
        scale = key_scale(redux_max(amax))
        codes = pack_fp8(tuple(clamp_fp8(div_full(v, scale), Float32(448)) for v in y))
        slot = load_slot(at(slots, token), wide=self.wide_slots)
        page, offset = slot >> 6, slot & (PAGE - 1)
        base = cache.iterator + page * (PAGE * ROW_BYTES)
        store_words(base + offset * DIM + 4 * lane, (codes,))
        if lane == 0:
            store_words(base + PAGE * DIM + offset * 4, (float_bits(scale),))

    @cute.jit
    def kv_norm(self, kv_a, kv_lo, kv_hi, d_kv, eps, token, lane):
        """RMSNormFusedParallel's kv_a row ``token`` in place: lane ``lane`` is the
        kernel's thread ``lane`` of each of the two 256-column warps."""
        lo = load_words(at(kv_a, token, 8 * lane), count=4, mutable=True)
        hi = load_words(at(kv_a, token, LATENT // 2 + 8 * lane), count=4, mutable=True)
        total = add_ftz(
            butterfly_ftz(square_chain(lo)), butterfly_ftz(square_chain(hi))
        )
        rr = norm_rsqrt(total, Float32(d_kv), eps)
        store_words(at(kv_a, token, 8 * lane), rms_scale(lo, rr, kv_lo))
        store_words(at(kv_a, token, LATENT // 2 + 8 * lane), rms_scale(hi, rr, kv_hi))


def _fake(dtype, shape, stride, align):
    return cute.runtime.make_fake_tensor(
        dtype, shape, stride=stride, assumed_align=align
    )


@functools.cache
def compile_front(
    tiles: int, query_rows: int, wide_slots: bool, key_norm_fp32: bool, pdl: bool
):
    """The front of up to ``16 * tiles`` rows (``DecodeFront``'s arguments). Rows,
    input row strides and the plane's slot count are runtime values."""
    sym, sym64 = cute.sym_int, cute.sym_int64
    rows = sym()
    width = query_rows + INDEX_ROWS
    norm = Float32 if key_norm_fp32 else BFloat16
    args = (
        _fake(BFloat16, (rows, K), (sym64(), 1), 16),  # q_a
        _fake(BFloat16, (rows, LATENT), (sym64(), 1), 16),  # kv_a
        _fake(BFloat16, (rows, DIM), (sym64(), 1), 8),  # index keys
        _fake(BFloat16, (rows, HEADS), (sym64(), 1), 2),  # head weights
        _fake(BFloat16, (K,), (1,), 16),  # q_a RMSNorm weight
        _fake(BFloat16, (LATENT,), (1,), 16),  # kv_a RMSNorm weight
        _fake(BFloat16, (width, K), (K, 1), 16),  # [q_b; wq_b]
        _fake(norm, (DIM,), (1,), 16),  # index-key LayerNorm weight
        _fake(norm, (DIM,), (1,), 16),  # index-key LayerNorm bias
        _fake(Uint8, (sym(), ROW_BYTES), (ROW_BYTES, 1), 16),  # index-K plane
        _fake(Int64 if wide_slots else Int32, (rows,), (1,), 8 if wide_slots else 4),
        _fake(BFloat16, (rows, K), (K, 1), 16),  # q_norm
        _fake(BFloat16, (rows, width), (width, 1), 16),  # projected
        _fake(Uint8, (rows, INDEX_ROWS), (INDEX_ROWS, 1), 16),  # FP8 index query
        _fake(Float32, (rows, HEADS), (HEADS, 1), 16),  # folded head weights
        Int32(K),
        Int32(LATENT),
        *(Float32(0) for _ in range(5)),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    front = DecodeFront(tiles, query_rows, wide_slots, key_norm_fp32, pdl)
    return cute.compile(front, *args, options="--enable-tvm-ffi")
