# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# The cluster merge follows vLLM's Kimi-K3.1 specialized decode (cutedsl/dsa/
# attn_sparse.py, SparseDcpDecode): each CTA pushes its FP32 partial and running
# max and sum to the latent-column owners over DSMEM, which merge them, and P is
# E4M3 at 448 of the running row max. New here: tcgen05 FP8 MMAs with S and O^T
# in TMEM, 128-key passes alternating between two warpgroups, and 16-byte
# cp.async row gathers into the 128B-swizzled operand layout.

"""Sparse MLA decode of E4M3 queries over selected E4M3 latent cache rows."""

import math
from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint8, Uint32, Uint64
from cutlass.cute.runtime import make_fake_stream

from ..common.host import architecture, fake, stream, to_cute
from ..common.primitives import (
    alloc,
    asm,
    at,
    bf16x2,
    bits_float,
    copy_g2s,
    fence_tmem,
    launch,
    load_tmem,
    load_words,
    named_barrier,
    push_float,
    release_tmem,
    scatter_reduce,
    smem_desc,
    store_words,
    wait,
)

ROW = 576  # FP8 bytes per cache / query row (512 latent + 64 rope)
LATENT = 512
HEADS = 16  # query heads per CTA: the MMA N
KEYS = 128  # keys per pass: the MMA M
SLAB = KEYS * 128  # one 128-byte K block of a pass's key rows (128B-swizzled)
TILE_BYTES = 5 * SLAB  # 4 latent blocks and the 64-byte rope block
Q_SLAB = HEADS * 128
P_BYTES = HEADS * 128
GROUPS = 2  # warpgroup g takes passes g, g + 2, ... into key tile g
SOFT_THREADS = 128 * GROUPS
MMA_WARP = 4 * GROUPS
THREADS = 32 * (MMA_WARP + 1)
LOG2_P_SCALE = math.log2(448.0)  # E4M3 P at 448 of the row max
# kind::f8f6f4: FP32 D, E4M3 A and B, N = 16, M = 128; bit 15: A is MN-major.
QK_INSTR = 1 << 4 | (HEADS >> 3) << 17 | (KEYS >> 4) << 24
PV_INSTR = QK_INSTR | 1 << 15
# TMEM: S of group g at column 16 g; its O^T latent tile t at 32 + 64 g + 16 t.
O_COL = 32
TMEM_COLS = 256


@asm()
def mma_qk(d, a, b, barrier):
    """Issue S^T = K Q^T of a pass from the elected lane, then commit.

    A is the pass's five K-major 128B-swizzled key blocks, B the query's.
    """
    mma = "tcgen05.mma.cta_group::1.kind::f8f6f4"
    commit = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
    code = "{ .reg .pred e,p0,pt; .reg .b64 xa,xb; .reg .b32 id; "
    code += "elect.sync _|e,0xffffffff; setp.eq.b32 pt,0,0; setp.ne.b32 p0,0,0; "
    code += f"mov.b32 id,{QK_INSTR}; "
    for k in range(ROW // 32):
        a_off = k // 4 * (SLAB // 16) + k % 4 * 2
        b_off = k // 4 * (Q_SLAB // 16) + k % 4 * 2
        code += f"add.s64 xa,$1,{a_off}; add.s64 xb,$2,{b_off}; "
        code += f"@e {mma} [$0],xa,xb,id,{'pt' if k else 'p0'}; "
    code += f"@e {commit} [$3]; }}"
    return code, "r,l,l,r", (Uint32(d), Uint64(a), Uint64(b), barrier)


@asm()
def mma_pv(d, a, b, steps, barrier):
    """Issue O^T = V^T P^T over ``steps`` K32 key steps, then commit.

    A is the key rows' latent blocks read MN-major, B is P K-major; D holds the
    four 128-wide latent tiles.
    """
    mma = "tcgen05.mma.cta_group::1.kind::f8f6f4"
    commit = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
    code = "{ .reg .pred e,p0,pt,pk; .reg .b64 xa,xb; .reg .b32 dt,id; "
    code += "elect.sync _|e,0xffffffff; setp.eq.b32 pt,0,0; setp.ne.b32 p0,0,0; "
    code += f"mov.b32 id,{PV_INSTR}; "
    for t in range(LATENT // 128):
        code += f"add.u32 dt,$0,{16 * t}; "
        for j in range(KEYS // 32):
            a_off = t * (SLAB // 16) + j * (4096 // 16)
            b_off = j * 2
            code += f"setp.gt.s32 pk,$3,{j}; and.pred pk,pk,e; "
            code += f"add.s64 xa,$1,{a_off}; add.s64 xb,$2,{b_off}; "
            code += f"@pk {mma} [dt],xa,xb,id,{'pt' if j else 'p0'}; "
    code += f"@e {commit} [$4]; }}"
    return code, "r,l,l,r,r", (Uint32(d), Uint64(a), Uint64(b), Int32(steps), barrier)


@asm()
def store_byte_shared(pointer, value):
    return "st.shared.u8 [$0], $1;", "r,r,~{memory}", (pointer, Uint32(value))


@asm(Uint32, pure=True)
def e4m3_bits(value: Float32):
    """E4M3 (satfinite, RN) of ``value`` in the low byte."""
    code = "{ .reg .b16 pair; .reg .f32 z; mov.f32 z,0f00000000; "
    code += "cvt.rn.satfinite.e4m3x2.f32 pair,z,$1; cvt.u32.u16 $0,pair; }"
    return code, "=r,f", (value,)


@asm()
def copy_zfill(dst, src, size):
    """16-byte cp.async of ``size`` (0 or 16) source bytes, zero-filling the rest."""
    return (
        "cp.async.cg.shared.global [$0], [$1], 16, $2;",
        "r,l,r",
        (dst, src, Int32(size)),
    )


@asm()
def cp_async_arrive(barrier):
    return (
        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [$0];",
        "r,~{memory}",
        (barrier,),
    )


@asm()
def cp_async_wait_all():
    return "cp.async.wait_all;", "~{memory}", ()


@asm()
def group_sync(barrier_id):
    """``bar.sync`` of one warpgroup on a runtime barrier id."""
    return "bar.sync $0, 128;", "r,~{memory}", (Int32(barrier_id),)


@asm(Int32)
def probe(barrier, phase):
    """Return 1 once the phase of parity ``phase`` completed, else 0."""
    code = "{ .reg .pred p; "
    code += "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 p,[$1],$2; "
    code += "selp.u32 $0,1,0,p; }"
    return code, "=r,r,r,~{memory}", (barrier, Int32(phase))


@asm(Int32)
def load_i32(address):
    return "ld.global.b32 $0, [$1];", "=r,l,~{memory}", (Int64(address),)


@cute.jit
def scatter_max(values, lane, count: cutlass.Constexpr[int]):
    """Warp-reduce ``count`` values by max; lane pairs end up owning one each."""
    selected = Int32(0)
    for stage in cutlass.range_constexpr(5):
        offset = 16 >> stage
        remaining = max(count >> stage, 1)
        if cutlass.const_expr(remaining > 1):
            half = remaining // 2
            upper = (lane & offset) != 0
            for i in cutlass.range_constexpr(half):
                send = values[i] if upper else values[i + half]
                keep = values[i + half] if upper else values[i]
                values[i] = max(keep, cute.arch.shuffle_sync_bfly(send, offset))
            if upper:
                selected += half
        else:
            values[0] = max(values[0], cute.arch.shuffle_sync_bfly(values[0], offset))
    return values[0], selected


class SparseMlaDecode:
    """Attend 1 to 16 query heads of each row over the row's selected cache rows.

    A cluster of ``split`` CTAs serves one row. CTA s takes the row's live keys
    s, s + split, ... in passes of up to 128 keys, which its two warpgroups
    take in turn: one pass's gather, softmax, and O^T epilogue overlap the
    other's. The cluster merges its FP32 partials over DSMEM; CTA s owns latent
    columns s / ``split``.
    """

    def __init__(self, heads, split, has_lens, pdl):
        assert 1 <= heads <= HEADS and split in (1, 2, 4, 8)
        self.heads, self.split, self.has_lens, self.pdl = heads, split, has_lens, pdl
        self.cols = LATENT // split

    @cute.jit
    def __call__(
        self,
        query,
        kv,
        slots,
        lens,
        out,
        rows: Int32,
        width: Int32,
        scale_log2,
        out_scale,
        stream,
    ):
        args = (query, kv, slots, lens, out, width, scale_log2, out_scale, stream)
        launch(
            self.kernel,
            args,
            (self.split, rows, 1),
            THREADS,
            cluster=(self.split, 1, 1),
            pdl=self.pdl,
        )

    @cute.kernel
    def kernel(self, args):
        query, kv, slots, lens, out, width, scale_log2, out_scale = args
        split_n, cols, live = self.split, self.cols, self.heads
        smem = cutlass.utils.SmemAllocator()
        kvs = alloc(smem, Uint8, (GROUPS * TILE_BYTES,), 1024)
        qs = alloc(smem, Uint8, (5 * Q_SLAB,), 1024)
        ps = alloc(smem, Uint8, (GROUPS * P_BYTES,), 1024)
        inbox = alloc(smem, Float32, (split_n, HEADS, cols), 16)
        stats = alloc(smem, Float32, (split_n, 2, HEADS), 16)
        red = alloc(smem, Float32, (GROUPS, 2, 4, HEADS), 16)
        # Running max and sum per head of each warpgroup's passes.
        mstat = alloc(smem, Float32, (GROUPS, 2, HEADS), 16)
        tmem_slot = alloc(smem, Uint32, (1,), 4)
        bars = alloc(smem, cutlass.Uint64, (5 * GROUPS + 2,), 8)
        gbar = bars.iterator  # + g: the pass's key tile landed
        sbar = bars.iterator + GROUPS  # + g: S of the pass
        pbar = bars.iterator + 2 * GROUPS  # + g: P of the pass
        obar = (
            bars.iterator + 3 * GROUPS
        )  # + g: O^T of the pass (its key tile and P drained)
        cbar = bars.iterator + 4 * GROUPS  # + g: the epilogue read O^T
        qbar = bars.iterator + 5 * GROUPS
        inbound = bars.iterator + 5 * GROUPS + 1
        tid, _, _ = cute.arch.thread_idx()
        split, b, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        group, quad = warp // 4, warp % 4
        lt = tid % 128  # the softmax thread's key of a pass and TMEM lane
        if tid == 0:
            for g in cutlass.range_constexpr(GROUPS):
                cute.arch.mbarrier_init(gbar + g, 256)
                cute.arch.mbarrier_init(sbar + g, 1)
                cute.arch.mbarrier_init(pbar + g, 128)
                cute.arch.mbarrier_init(obar + g, 1)
                cute.arch.mbarrier_init(cbar + g, 128)
            cute.arch.mbarrier_init(qbar, 64)
            cute.arch.mbarrier_init(inbound, 1)
            cute.arch.mbarrier_init_fence()
            inbound_bytes = split_n * (live * cols + 8 * ((live + 3) // 4)) * 4
            cute.arch.mbarrier_arrive_and_expect_tx(inbound, inbound_bytes)
        if tid < GROUPS * HEADS:
            mstat[tid // HEADS, 0, tid % HEADS] = Float32(-math.inf)
            mstat[tid // HEADS, 1, tid % HEADS] = Float32(0.0)
        if warp == MMA_WARP:
            cute.arch.alloc_tmem(TMEM_COLS, tmem_slot.iterator)
            cute.arch.relinquish_tmem_alloc_permit()
        fence_tmem(before=True)
        cute.arch.sync_threads()
        fence_tmem()
        tmem = tmem_slot[0]
        # Every CTA's ``inbound`` is armed before any peer pushes (waited below).
        cute.arch.cluster_arrive_relaxed()
        # Nothing above reads the inputs: the kernel before may launch this early.
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        kv_base = kv.iterator.toint()
        row_slots = slots.iterator.toint() + Int64(b) * Int64(width) * 4
        # The first pass of each warpgroup: its key's slot, loaded with the row length.
        first = Int32(-1)
        pos0 = split + split_n * (KEYS * group + lt)
        if (warp < MMA_WARP) & (pos0 < width):
            first = load_i32(row_slots + Int64(pos0) * 4)
        n = width
        if cutlass.const_expr(self.has_lens):
            n = max(min(lens[b], width), Int32(0))
        mine = Int32(0)
        if n > split:
            mine = (n - split + split_n - 1) // split_n
        passes = (mine + KEYS - 1) // KEYS
        neg_inf = Float32(-math.inf)
        acc = cute.make_rmem_tensor((4, HEADS), Float32)
        acc.fill(0.0)
        alpha = cute.make_rmem_tensor(HEADS, Float32)
        if warp == MMA_WARP:
            # The query as five 128-byte K blocks of 16 rows; padding heads are zero.
            for i in cutlass.range_constexpr(HEADS * 36 // 32):
                flat = lane + 32 * i
                h, c = flat // 36, flat % 36
                dst = qs.iterator + (c // 8 * Q_SLAB + h // 8 * 1024 + h % 8 * 128)
                dst = dst + (c % 8 ^ h % 8) * 16
                if h < live:
                    copy_g2s(dst, at(query, b, h, 16 * c))
                else:
                    store_words(dst, (Uint32(0),) * 4, shared=True)
            cp_async_arrive(qbar)
            cute.arch.mbarrier_arrive(qbar)
            qd = smem_desc(qs.iterator, True)
            wait(qbar, 0, scope="cta")
            # Each pass's QK once its keys land, its PV once P is stored, as ready.
            qk = Int32(0)
            pv = Int32(0)
            while pv < passes:
                if qk < passes:
                    g, j = qk % GROUPS, qk // GROUPS
                    # Warp-uniform: the issue elects one lane of the converged warp.
                    if cute.arch.shuffle_sync(probe(gbar + g, j % 2), 0) != 0:
                        cute.arch.fence_proxy("async.shared", space="cta")
                        fence_tmem()
                        kd = smem_desc(kvs.iterator + g * TILE_BYTES, True)
                        mma_qk(tmem + 16 * g, kd, qd, sbar + g)
                        qk += 1
                if pv < qk:
                    g, j = pv % GROUPS, pv // GROUPS
                    ready = probe(pbar + g, j % 2)
                    if j > 0:
                        ready = ready & probe(cbar + g, (j - 1) % 2)  # O^T was read
                    if cute.arch.shuffle_sync(ready, 0) != 0:
                        cute.arch.fence_proxy("async.shared", space="cta")
                        fence_tmem()
                        count = min(mine - pv * KEYS, Int32(KEYS))
                        kd = smem_desc(kvs.iterator + g * TILE_BYTES, True)
                        pd = smem_desc(ps.iterator + g * P_BYTES, True)
                        mma_pv(
                            tmem + (O_COL + 64 * g),
                            kd,
                            pd,
                            (count + 31) // 32,
                            obar + g,
                        )
                        pv += 1
        elif warp < MMA_WARP:
            tile = kvs.iterator + group * TILE_BYTES
            for p in cutlass.range(group, passes, GROUPS, unroll=1):
                j = p // GROUPS
                count = min(mine - p * KEYS, Int32(KEYS))
                span = (count + 31) // 32 * 32
                slot = first
                if j > 0:
                    # This group's last PV MMAs have read the key tile and P.
                    wait(obar + group, (j - 1) % 2, scope="cta")
                    pos = split + split_n * (p * KEYS + lt)
                    slot = Int32(-1)
                    if pos < width:
                        slot = load_i32(row_slots + Int64(pos) * 4)
                if lt >= count:
                    slot = Int32(-1)
                valid = slot >= 0
                # 32 rows per warp; masked keys load zeros, so V is finite where P is 0.
                if 32 * quad < span:
                    for i in cutlass.range(36, unroll=12):
                        flat = lane + 32 * i
                        r, c = flat // 36, flat % 36
                        src_slot = cute.arch.shuffle_sync(slot, r)
                        k = 32 * quad + r
                        dst = tile + (c // 8 * SLAB + k // 8 * 1024 + k % 8 * 128)
                        dst = dst + (c % 8 ^ k % 8) * 16
                        size = cutlass.select_(src_slot >= 0, Int32(16), Int32(0))
                        src = kv_base + Int64(max(src_slot, Int32(0))) * ROW + c * 16
                        copy_zfill(dst, src, size)
                cp_async_arrive(gbar + group)
                cute.arch.mbarrier_arrive(gbar + group)
                # Softmax of this pass: thread lt holds key lt's 16 head scores.
                wait(sbar + group, j % 2, scope="cta")
                fence_tmem()
                s = load_tmem(tmem + (32 * quad << 16) + 16 * group, HEADS)
                vals = cute.make_rmem_tensor(HEADS, Float32)
                for h in cutlass.range_constexpr(HEADS):
                    vals[h] = cutlass.select_(valid, s[h] * scale_log2, neg_inf)
                top, head = scatter_max(vals, lane, HEADS)
                if lane % 2 == 0:
                    red[group, 0, quad, head] = top
                group_sync(1 + group)
                # Lanes 2 h and 2 h + 1 own head h: chunk max, running max, and rescale.
                chunk_max = red[group, 0, 0, head]
                for w in cutlass.range_constexpr(1, 4):
                    chunk_max = max(chunk_max, red[group, 0, w, head])
                m_old = mstat[group, 0, head]
                m_new = max(m_old, chunk_max)
                rescale = Float32(1.0)
                if m_old != neg_inf:
                    rescale = cute.math.exp2(m_old - m_new, fastmath=True)
                # P (E4M3 at 448 of the running max) as the K-major B operand.
                p_base = ps.iterator + group * P_BYTES
                for h in cutlass.range_constexpr(HEADS):
                    bias = cute.arch.shuffle_sync(m_new, 2 * h) - Float32(LOG2_P_SCALE)
                    prob = cute.math.exp2(s[h] * scale_log2 - bias, fastmath=True)
                    prob = cutlass.select_(valid, prob, Float32(0))
                    vals[h] = prob
                    offset = (
                        h // 8 * 1024 + h % 8 * 128 + (lt // 16 ^ h % 8) * 16 + lt % 16
                    )
                    store_byte_shared(p_base + offset, e4m3_bits(prob))
                cute.arch.mbarrier_arrive(pbar + group)
                total, head2 = scatter_reduce(vals, lane, HEADS)
                if lane % 2 == 0:
                    red[group, 1, quad, head2] = total
                group_sync(1 + group)
                if quad == 0:
                    if lane % 2 == 0:
                        total = red[group, 1, 0, head]
                        for w in cutlass.range_constexpr(1, 4):
                            total += red[group, 1, w, head]
                        mstat[group, 1, head] = mstat[group, 1, head] * rescale + total
                        mstat[group, 0, head] = m_new
                for h in cutlass.range_constexpr(HEADS):
                    alpha[h] = cute.arch.shuffle_sync(rescale, 2 * h)
                # O^T rows of this pass: latent 128 t + 32 quad + lane.
                wait(obar + group, j % 2, scope="cta")
                fence_tmem()
                for t in cutlass.range_constexpr(LATENT // 128):
                    o = load_tmem(
                        tmem + (32 * quad << 16) + (O_COL + 64 * group + 16 * t), HEADS
                    )
                    for h in cutlass.range_constexpr(HEADS):
                        acc[t, h] = acc[t, h] * alpha[h] + o[h]
                fence_tmem(before=True)
                cute.arch.mbarrier_arrive(cbar + group)
        # Group g keeps latent tiles 2 g and 2 g + 1; every PV is done past the barrier.
        if warp < MMA_WARP:
            named_barrier(3, SOFT_THREADS)
            # The tiles this group hands over, [head, latent] FP32 over its key tile.
            handoff = cute.make_tensor(
                cute.recast_ptr(kvs.iterator + group * TILE_BYTES, dtype=Float32),
                cute.make_layout((HEADS, LATENT // 2), stride=(LATENT // 2, 1)),
            )
            # Static indices: group 0 hands over tiles 2 and 3, group 1 tiles 0 and 1.
            first_group = group == 0
            for e in cutlass.range_constexpr(2):
                for h in cutlass.range_constexpr(live):
                    value = cutlass.select_(first_group, acc[2 + e, h], acc[e, h])
                    handoff[h, 128 * e + 32 * quad + lane] = value
            named_barrier(3, SOFT_THREADS)
            other = cute.make_tensor(
                cute.recast_ptr(kvs.iterator + (1 - group) * TILE_BYTES, dtype=Float32),
                cute.make_layout((HEADS, LATENT // 2), stride=(LATENT // 2, 1)),
            )
            for h in cutlass.range_constexpr(live):
                m_g = mstat[group, 0, h]
                m_o = mstat[1 - group, 0, h]
                top = max(m_g, m_o)
                w_g = Float32(0)
                w_o = Float32(0)
                if m_g != neg_inf:
                    w_g = cute.math.exp2(m_g - top, fastmath=True)
                if m_o != neg_inf:
                    w_o = cute.math.exp2(m_o - top, fastmath=True)
                for e in cutlass.range_constexpr(2):
                    kept = cutlass.select_(first_group, acc[e, h], acc[2 + e, h])
                    acc[e, h] = kept * w_g + other[h, 128 * e + 32 * quad + lane] * w_o
        cute.arch.cluster_wait()
        # Push this CTA's partial of the live heads to the column owners.
        if warp < MMA_WARP:
            for e in cutlass.range_constexpr(2):
                col = 128 * (2 * group + e) + 32 * quad + lane
                owner = col // cols
                for h in cutlass.range_constexpr(live):
                    push_float(
                        at(inbox, split, h, col % cols), acc[e, h], inbound, owner
                    )
        if warp == 0:
            # The merged running max and sum of both groups, pushed to every peer.
            quads_s = (live + 3) // 4
            for i in cutlass.range_constexpr(-(-2 * quads_s * split_n // 32)):
                flat = lane + 32 * i
                if flat < 2 * quads_s * split_n:
                    peer, part = flat // (2 * quads_s), flat % (2 * quads_s)
                    kind, q4 = part // quads_s, part % quads_s
                    m0 = load_words(at(mstat, 0, 0, 4 * q4), count=4, shared=True)
                    m1 = load_words(at(mstat, 1, 0, 4 * q4), count=4, shared=True)
                    l0 = load_words(at(mstat, 0, 1, 4 * q4), count=4, shared=True)
                    l1 = load_words(at(mstat, 1, 1, 4 * q4), count=4, shared=True)
                    merged = []
                    for e in cutlass.range_constexpr(4):
                        a, c = bits_float(m0[e]), bits_float(m1[e])
                        top = max(a, c)
                        wa = Float32(0)
                        wc = Float32(0)
                        if a != neg_inf:
                            wa = cute.math.exp2(a - top, fastmath=True)
                        if c != neg_inf:
                            wc = cute.math.exp2(c - top, fastmath=True)
                        lsum = bits_float(l0[e]) * wa + bits_float(l1[e]) * wc
                        merged.append(cutlass.select_(kind == 0, top, lsum))
                    push_float(
                        at(stats, split, kind, 4 * q4), tuple(merged), inbound, peer
                    )
        wait(inbound, 0)  # every source's partial of these columns landed
        self.merge(out, out_scale, inbox, stats, split, b, tid)
        cp_async_wait_all()  # a CTA without keys never waited for its query copies
        release_tmem(tmem, TMEM_COLS, warp, MMA_WARP)

    @cute.jit
    def merge(self, out, out_scale, inbox, stats, split, b, tid):
        """Merge this CTA's latent columns of the live heads over the cluster.

        Each thread takes four columns of one head per pass.
        """
        neg_inf = Float32(-math.inf)
        quads = self.cols // 4
        for i in cutlass.range_constexpr(-(-self.heads * quads // THREADS)):
            flat = tid + THREADS * i
            if flat < self.heads * quads:
                head, quad = flat // quads, flat % quads
                col = split * self.cols + 4 * quad
                top = stats[0, 0, head]
                for c in cutlass.range_constexpr(1, self.split):
                    top = max(top, stats[c, 0, head])
                total = Float32(0)
                merged = cute.make_rmem_tensor(4, Float32)
                merged.fill(0.0)
                for c in cutlass.range_constexpr(self.split):
                    words = load_words(
                        at(inbox, c, head, 4 * quad), count=4, shared=True
                    )
                    weight = Float32(0)
                    if stats[c, 0, head] != neg_inf:
                        weight = cute.math.exp2(stats[c, 0, head] - top, fastmath=True)
                    total += weight * stats[c, 1, head]
                    for e in cutlass.range_constexpr(4):
                        merged[e] += weight * bits_float(words[e])
                inverse = Float32(0)
                if total > Float32(0):
                    inverse = out_scale / total
                words = (
                    bf16x2(merged[0] * inverse, merged[1] * inverse),
                    bf16x2(merged[2] * inverse, merged[3] * inverse),
                )
                store_words(at(out, b, head, col), words)


@cache
def compile_sparse_mla(heads, split, has_lens, pdl, device_index):
    """Compile SparseMlaDecode for runtime row, slot-width and cache-row counts."""
    rows = cute.sym_int()
    query = fake((rows, heads, ROW), None, torch.uint8)
    kv = fake((cute.sym_int(), ROW), (ROW, 1), torch.uint8)
    slots = fake((cute.sym_int(),), (1,), torch.int32, 4)
    lens = fake((cute.sym_int(),), (1,), torch.int32, 4)
    out = fake((rows, heads, LATENT))
    kernel = SparseMlaDecode(heads, split, has_lens, pdl)
    arch = architecture(torch.device("cuda", device_index))
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            kernel,
            query,
            kv,
            slots,
            lens,
            out,
            Int32(1),
            Int32(1),
            Float32(1.0),
            Float32(1.0),
            make_fake_stream(),
            options=f"--gpu-arch={arch}",
        )


def sparse_mla_decode(query, kv, slots, lens, out, softmax_scale, *, split, pdl):
    """Attend each query row over its live selected cache rows.

    ``out[r, h] = softmax(query[r, h] . K[s]) V[s]`` over the live slots s of row r.

    Args:
        query: E4M3 [rows, heads, 576], contiguous, at most 16 heads.
        kv: E4M3 latent cache rows [slots, 576], contiguous.
        slots: int32 [rows, width] global cache rows; negative entries are masked.
        lens: int32 [rows] live prefix of each row's slots, or None for the width.
        out: BF16 [rows, heads, 512], contiguous.
        softmax_scale: Logit scale, including any key scale.
        split: CTAs per row (a cluster).
        pdl: Launch with programmatic dependent launch.

    Returns:
        ``out``. Rows without live slots come out zero.
    """
    rows, heads, _ = query.shape
    code = compile_sparse_mla(heads, split, lens is not None, pdl, query.device.index)
    code(
        to_cute(query.view(torch.uint8)),
        to_cute(kv.view(-1, ROW).view(torch.uint8)),
        to_cute(slots.view(-1), 4),
        to_cute(slots.view(-1) if lens is None else lens, 4),
        to_cute(out),
        Int32(rows),
        Int32(slots.shape[1]),
        Float32(softmax_scale * math.log2(math.e)),
        Float32(1.0),
        stream(query.device),
    )
    return out
