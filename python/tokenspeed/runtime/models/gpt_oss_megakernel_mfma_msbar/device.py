"""Device-side helpers shared by the megakernel phases."""

from __future__ import annotations

import triton
import triton.language as tl

_EPS = tl.constexpr(1e-5)


@triton.jit
def _barrier(
    bar_p, seq, NWG: tl.constexpr, NSHARD: tl.constexpr = 1, BARMODE: tl.constexpr = 0
):
    """Grid-wide barrier. Two implementations, selected by BARMODE.

    BARMODE==0 (single-counter, the read-poll baseline): all NWG workgroups
    atomic_add(release) a SINGLE int32, then spin on a coherent (`.cv`) vector
    load and issue ONE acquire on the loop-exit branch. The load poll is shared
    (every poller holds the line), so only the arrival RMWs and one acquire/L2
    invalidate serialise -- not the poll. NSHARD is ignored; bar_p needs 1 int.

    BARMODE==1 (master-slave two-phase, the fast path): the arrival atomic is
    SHARDED across NSHARD distinct counter lines (shard = pid % NSHARD), turning
    NWG-way contention on one cache line into NWG/NSHARD-way contention on each
    of NSHARD lines. Only pid 0 polls+sums the NSHARD shards and, once the total
    reaches (seq+1)*NWG, publishes a single release flag at bar_p+NSHARD. Every
    other workgroup spins on THAT one flag -- a cheap shared read, no O(NSHARD)
    sum. Memory ordering is preserved: each CTA's arrival RMW is `release` (its
    phase writes reach the coherence point before the count is visible), and the
    matching `acquire` on every loop-exit branch emits the device-scope
    `buffer_inv` that makes other XCDs' writes visible. bar_p needs NSHARD+1 ints.

    The poll MUST be a vector load: `bar_p` is a uniform address, and a scalar
    load is served from the scalar cache, which `buffer_inv` does not
    invalidate. Indexing with `arange` forces the vector path. The acquire MUST
    sit on the loop-exit path, not after the loop: it is what emits `buffer_inv
    sc1`, and only the control dependency of the exit branch stops the following
    data loads being hoisted above the invalidate. Placing it after the loop
    compiles, runs faster, and silently produces garbage.
    """
    if BARMODE == 1:
        tl.debug_barrier()
        pid = tl.program_id(0)
        tl.atomic_add(bar_p + (pid % NSHARD), 1, sem="release", scope="gpu")
        target = (seq + 1) * NWG
        sidx = tl.arange(0, NSHARD)
        one = tl.arange(0, 1)
        rel = bar_p + NSHARD
        if pid == 0:
            done = 0
            while done == 0:
                cur = tl.load(bar_p + sidx, cache_modifier=".cv")
                if tl.sum(cur) >= target:
                    if tl.atomic_add(bar_p, 0, sem="acquire", scope="gpu") >= 0:
                        done = 1
            tl.atomic_xchg(rel, seq + 1, sem="release", scope="gpu")
        else:
            done = 0
            while done == 0:
                v = tl.load(rel + one, cache_modifier=".cv")
                if tl.max(v) >= seq + 1:
                    if tl.atomic_add(rel, 0, sem="acquire", scope="gpu") >= seq + 1:
                        done = 1
        tl.debug_barrier()
        return
    tl.debug_barrier()
    tl.atomic_add(bar_p, 1, sem="release", scope="gpu")
    target = (seq + 1) * NWG
    idx = tl.arange(0, 1)
    done = 0
    while done == 0:
        cur = tl.load(bar_p + idx, cache_modifier=".cv")
        if tl.max(cur) >= target:
            if tl.atomic_add(bar_p, 0, sem="acquire", scope="gpu") >= target:
                done = 1
    tl.debug_barrier()


@triton.jit
def _rms_norm_vec(x_p, w_p, n: tl.constexpr, BLK: tl.constexpr):
    o = tl.arange(0, BLK)
    msk = o < n
    x = tl.load(x_p + o, mask=msk, other=0.0).to(tl.float32)
    ms = tl.sum(x * x) / n
    w = tl.load(w_p + o, mask=msk, other=0.0).to(tl.float32)
    return x * tl.rsqrt(ms + _EPS) * w


@triton.jit
def _mxfp4_row_dot(
    blk_p, scl_p, row, K: tl.constexpr, a_even, a_odd, BLK_K2: tl.constexpr
):
    """Precise path: dequantise one MXFP4 row to fp32 and dot it.

    BLK_K2 covers K/2 (=1440), NOT the full hidden width -- sizing this at 4096
    made ~64% of every load and multiply masked-out waste.

    The dequantisation is pure integer bit manipulation because this kernel is
    dominated by it: rocprofv3 measured 58.7 VALU instructions per VMEM
    instruction, i.e. ~30 lane-ops to consume a single 4-bit weight, and
    essentially all VALU in the megakernel is this function. Two identities
    remove most of them:

      * an E2M1 value with exponent e>0 IS an fp32 whose exponent field is
        (e-1+127) and whose mantissa is m<<22.  The magnitude code u = v & 7
        equals e*2+m, so `u << 22` lays e and m into exactly those fields in one
        shift -- no compare tree and no exp2.
      * multiplying by the E8M0 scale 2^(sb-127) is an ADD into that same
        exponent field, so folding `(sb-1) << 23` into the integer removes the
        per-element scale exp2 AND the per-element scale multiply outright.

    Only e==0 (the +-0 / +-0.5 codes) is not an fp32 in this sense and costs the
    one select. Results are bit-identical to the float formulation -- verified
    against it on all 5760 rows of a real layer-0 expert, 0 rows differing when
    compared as int32.
    """
    i = tl.arange(0, BLK_K2)
    msk = i < (K // 2)
    b = tl.load(blk_p + row.to(tl.int64) * (K // 2) + i, mask=msk, other=0)
    s = tl.load(scl_p + row.to(tl.int64) * (K // 32) + (i // 16), mask=msk, other=127)
    sbs = (s.to(tl.int32) - 1) << 23  # 2^(sb-128) as raw fp32 bits

    lo = (b & 0x0F).to(tl.int32)
    hi = (b >> 4).to(tl.int32)
    ul, uh = lo & 7, hi & 7
    bl = tl.where(ul >= 2, (ul << 22) + sbs, tl.where(ul == 1, sbs, 0)) | (
        (lo & 8) << 28
    )
    bh = tl.where(uh >= 2, (uh << 22) + sbs, tl.where(uh == 1, sbs, 0)) | (
        (hi & 8) << 28
    )
    return tl.sum(
        bl.to(tl.float32, bitcast=True) * a_even
        + bh.to(tl.float32, bitcast=True) * a_odd
    )


@triton.jit
def _quant_e8m0(x, BLK: tl.constexpr):
    """fp32 [BLK] -> (fp8e4m3 [BLK], E8M0 byte [BLK/32]) with per-32 shared exp.

    The clamp to +-448 is load-bearing: floor(log2(amax))-8 puts the scaled max
    in [256, 512) but e4m3's largest normal is 448, and anything above saturates
    to NaN and poisons the whole GEMV.
    """
    x2 = tl.reshape(x, (BLK // 32, 32))
    amax = tl.max(tl.abs(x2), axis=1)
    amax = tl.where(amax == 0.0, 1.0, amax)
    e = tl.floor(tl.log2(amax)) - 8.0
    e = tl.minimum(tl.maximum(e, -127.0), 127.0)
    q2 = x2 / tl.exp2(e)[:, None]
    q2 = tl.minimum(tl.maximum(q2, -448.0), 448.0)
    return tl.reshape(q2, (BLK,)), (e + 127.0)


@triton.jit
def _gemv_dot_scaled(
    w_p,
    ws_p,
    aq_p,
    as_p,
    out_p,
    bias_p,
    wgt,
    N: tl.constexpr,
    K: tl.constexpr,
    pid,
    npid,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MTILE: tl.constexpr,
    ACCUM: tl.constexpr,
):
    """Fast path: native scaled MFMA (v_mfma_scale_f32_16x16x128_f8f6f4).

    Operand convention copied from tokenspeed's own working _mm_mxfp4_kernel:
    every operand [non_K, K_packed] row-major, every scale [non_K, K/32] with
    other=127 (== 2^0). e2m1 packs 2 values/byte, e4m3 does not. The activation
    occupies row 0 of an MTILE-row operand; the rest is padding at batch 1.
    """
    half = K // 2
    nblk = K // 32
    offs_m = tl.arange(0, MTILE)
    n0 = pid * BLOCK_N
    while n0 < N:
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mn = offs_n < N
        acc = tl.zeros((BLOCK_N, MTILE), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            pk = k0 // 2 + tl.arange(0, BLOCK_K // 2)
            sk = k0 // 32 + tl.arange(0, BLOCK_K // 32)
            fk = k0 + tl.arange(0, BLOCK_K)
            w = tl.load(
                w_p + offs_n[:, None].to(tl.int64) * half + pk[None, :],
                mask=mn[:, None] & (pk[None, :] < half),
                other=0,
            )
            ws = tl.load(
                ws_p + offs_n[:, None].to(tl.int64) * nblk + sk[None, :],
                mask=mn[:, None] & (sk[None, :] < nblk),
                other=127,
            )
            a = tl.load(
                aq_p + offs_m[:, None] * 0 + fk[None, :],
                mask=(offs_m[:, None] == 0) & (fk[None, :] < K),
                other=0.0,
            )
            asc = tl.load(
                as_p + offs_m[:, None] * 0 + sk[None, :],
                mask=(offs_m[:, None] == 0) & (sk[None, :] < nblk),
                other=127,
            )
            acc = tl.dot_scaled(
                w, ws, "e2m1", a.trans(), asc, "e4m3", acc=acc, fast_math=True
            )
        v = tl.sum(tl.where(offs_m[None, :] == 0, acc, 0.0), axis=1)
        v += tl.load(bias_p + offs_n, mask=mn, other=0.0).to(tl.float32)
        if ACCUM:
            tl.store(
                out_p + offs_n,
                tl.load(out_p + offs_n, mask=mn, other=0.0) + wgt * v,
                mask=mn,
            )
        else:
            tl.store(out_p + offs_n, v, mask=mn)
        n0 += npid * BLOCK_N
