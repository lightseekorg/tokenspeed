"""M7: M6 with two MEASURED inefficiencies fixed. Same numerics, same contracts.

Both changes are grounded in `m6_profile.py` output (bs=1, 36 layers), not guessed:

  MoE       15.0 ms (57%)   <- dominant
  barriers   6.7 ms (25%)   at a MEASURED in-context slope of 8.41 us/barrier
  attention  2.4 ms  (9%)
  rest       2.3 ms  (9%)

Fix 1 -- expert loop order: (b, j) outer, rows inner.
  M6 loaded the activation even/odd halves INSIDE the row loop, so the [2048]
  vectors were re-loaded once per output row instead of once per (b, j) -- a
  5760x redundancy that M4 did not have. Hoisting them restores M4's structure
  and simultaneously restores the `_gemv_dot_scaled` call site, which M6 had
  silently dropped (DOT was wired only to the fp8 quant, so `dot_scaled` showed
  0% benefit in the M6 profile).

Fix 2 -- barrier fusion, 22/layer -> 11.
  * the four expert slots no longer take 3 barriers EACH (12 -> 3): with (b, j)
    inside a phase, one barrier per phase suffices
  * RoPE and KV-append merge: the k loop rotates AND appends in the same
    program, so no cross-program read occurs between them (this is what M4/M5
    did; M6 split them unnecessarily)
  * the hidden update folds into the down-proj epilogue: hs is pre-seeded with
    the residual in the top-4 phase, and down accumulates straight into it
  At 8.41 us/barrier x 36 layers that is a predicted ~3.3 ms.

Race note for the down phase: `j` is an inner loop with NO barrier between
iterations, so program `pid` must own the same output rows for every j. It does
-- the row loop is `r = pid; r += npid` in all iterations -- so no two programs
ever touch the same hs element. Do not switch that loop to a block mapping
without re-checking this.

Barriers (11/layer), all sequence numbers runtime-derived from the layer index:
  0 attn-norm | 1 QKV | 2 RoPE+KV-append | 3 attention | 4 O-proj+residual
  5 moe-norm  | 6 router | 7 top-4 + seed hs | 8 gate-up | 9 swiglu
  10 down (accumulates into hs)
"""

from __future__ import annotations

import triton
import triton.language as tl

from tokenspeed.runtime.models.gpt_oss_megakernel_mfma.device import (
    _barrier,
    _gemv_dot_scaled,
    _mxfp4_row_dot,
    _quant_e8m0,
    _rms_norm_vec,
)

H = 2880
I = 2880
NH = 64
NKV = 8
DH = 64
QKV = NH * DH + 2 * NKV * DH
E = 128
TOPK = 4
ALPHA = 1.702
BETA = 1.0
LIMIT = 7.0
NBAR = 8

_H = tl.constexpr(H)
_I = tl.constexpr(I)
_NH = tl.constexpr(NH)
_NKV = tl.constexpr(NKV)
_DH = tl.constexpr(DH)
_QKV = tl.constexpr(QKV)
_E = tl.constexpr(E)
_TOPK = tl.constexpr(TOPK)
_ALPHA = tl.constexpr(ALPHA)
_BETA = tl.constexpr(BETA)
_LIMIT = tl.constexpr(LIMIT)
_NBAR = tl.constexpr(NBAR)


@triton.jit
def mk_model_opt(
    hs_p,
    out_p,
    an_p,
    wqkv_p,
    bqkv_p,
    wo_p,
    bo_p,
    sinks_p,
    cos_p,
    sin_p,
    kbuf_p,
    vbuf_p,
    koff_p,
    voff_p,
    lgid_p,
    ptab_p,
    ps_p,
    wloc_p,
    win_p,
    posn_p,
    mn_p,
    rw_p,
    rb_p,
    gu_blk_p,
    gu_scl_p,
    gu_b_p,
    dn_blk_p,
    dn_scl_p,
    dn_b_p,
    fnorm_p,
    xnorm_p,
    qkv_p,
    attn_p,
    resid_p,
    rlog_p,
    gu_p,
    act_p,
    ynorm_p,
    tki_p,
    tkw_p,
    ynq_p,
    yns_p,
    actq_p,
    acts_p,
    bar_p,
    B,
    L,
    NWG: tl.constexpr,
    BLK_H: tl.constexpr,
    BLK_D: tl.constexpr,
    BLK_E: tl.constexpr,
    TAB_G: tl.constexpr,
    TAB_B: tl.constexpr,
    WLOC_G: tl.constexpr,
    BLK_K2: tl.constexpr,
    DOT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MTILE: tl.constexpr,
    EXTRA_BAR: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    oh = tl.arange(0, BLK_H)
    mh = oh < _H
    od = tl.arange(0, BLK_D)
    half = _DH // 2
    mhalf = od < half

    for layer in range(L):
        # precise path fuses gate-up with SwiGLU and drops one barrier
        _NB: tl.constexpr = _NBAR if DOT else _NBAR - 1
        b0 = layer * (_NB + EXTRA_BAR)
        li = layer.to(tl.int64)

        an = an_p + layer * _H
        mn = mn_p + layer * _H
        wq = wqkv_p + li * (_QKV * _H)
        bq = bqkv_p + layer * _QKV
        wo = wo_p + li * (_H * _NH * _DH)
        bo = bo_p + layer * _H
        sk = sinks_p + layer * _NH
        rw = rw_p + li * (_E * _H)
        rb = rb_p + layer * _E
        gb0 = gu_blk_p + li * _E * (2 * _I) * (_H // 2)
        gs0 = gu_scl_p + li * _E * (2 * _I) * (_H // 32)
        gbi0 = gu_b_p + li * _E * (2 * _I)
        db0 = dn_blk_p + li * _E * _H * (_I // 2)
        ds0 = dn_scl_p + li * _E * _H * (_I // 32)
        dbi0 = dn_b_p + li * _E * _H
        # koff_p/voff_p now hold ABSOLUTE addresses, bitcast to pointers
        kc = tl.load(koff_p + layer).to(tl.pointer_type(tl.bfloat16), bitcast=True)
        vc = tl.load(voff_p + layer).to(tl.pointer_type(tl.bfloat16), bitcast=True)
        gid = tl.load(lgid_p + layer)
        ps = tl.load(ps_p + gid)
        tabg = ptab_p + gid.to(tl.int64) * TAB_G
        wlocg = wloc_p + gid * WLOC_G
        win = tl.load(win_p + layer)

        # ---- 0+1 fused: attn RMSNorm (recomputed per program) + QKV --------
        # Each program recomputes xn for the token it processes, so no program
        # reads another program's norm write -> the norm barrier is removed.
        # xn is bit-identical to the old stored xnorm (both f32); xnorm_p is dead.
        for bb_ in range(B):
            bi = bb_.to(tl.int64)
            xn = _rms_norm_vec(hs_p + bi * _H, an, _H, BLK_H)
            rr = pid
            while rr < _QKV:
                w = tl.load(wq + rr.to(tl.int64) * _H + oh, mask=mh, other=0.0).to(
                    tl.float32
                )
                tl.store(
                    qkv_p + bi * _QKV + rr,
                    tl.sum(w * xn) + tl.load(bq + rr).to(tl.float32),
                )
                rr += npid
        _barrier(bar_p, b0 + 0, NWG)

        # ---- 2: RoPE and KV append, one phase ------------------------------
        # q rotates in place; the k loop rotates AND appends in the same
        # program, so nothing here reads another program's write.
        r = pid
        while r < B * _NH:
            bb = r // _NH
            hh = r % _NH
            pb = tl.load(posn_p + bb)
            c = tl.load(cos_p + pb * half + od, mask=mhalf, other=0.0)
            s = tl.load(sin_p + pb * half + od, mask=mhalf, other=0.0)
            qb = qkv_p + bb.to(tl.int64) * _QKV + hh * _DH
            a = tl.load(qb + od, mask=mhalf, other=0.0)
            bv = tl.load(qb + half + od, mask=mhalf, other=0.0)
            tl.store(qb + od, a * c - bv * s, mask=mhalf)
            tl.store(qb + half + od, bv * c + a * s, mask=mhalf)
            r += npid
        r = pid
        while r < B * _NKV:
            bb = r // _NKV
            kk = r % _NKV
            pb = tl.load(posn_p + bb)
            c = tl.load(cos_p + pb * half + od, mask=mhalf, other=0.0)
            s = tl.load(sin_p + pb * half + od, mask=mhalf, other=0.0)
            wl = tl.load(wlocg + bb).to(tl.int64)
            qb = qkv_p + bb.to(tl.int64) * _QKV + (_NH + kk) * _DH
            a = tl.load(qb + od, mask=mhalf, other=0.0)
            bv = tl.load(qb + half + od, mask=mhalf, other=0.0)
            n_lo = a * c - bv * s
            n_hi = bv * c + a * s
            tl.store(qb + od, n_lo, mask=mhalf)
            tl.store(qb + half + od, n_hi, mask=mhalf)
            kp = kc + (wl * _NKV + kk) * _DH
            tl.store(kp + od, n_lo, mask=mhalf)
            tl.store(kp + half + od, n_hi, mask=mhalf)
            r += npid
        r = pid
        while r < B * _NKV:
            bb = r // _NKV
            kk = r % _NKV
            wl = tl.load(wlocg + bb).to(tl.int64)
            val = tl.load(
                qkv_p + bb.to(tl.int64) * _QKV + (_NH + _NKV + kk) * _DH + od,
                mask=od < _DH,
                other=0.0,
            )
            tl.store(vc + (wl * _NKV + kk) * _DH + od, val, mask=od < _DH)
            r += npid
        _barrier(bar_p, b0 + 1, NWG)

        # ---- 3: attention with sinks ---------------------------------------
        r = pid
        while r < B * _NH:
            bb = r // _NH
            hh = r % _NH
            pb = tl.load(posn_p + bb)
            lo = tl.maximum(0, pb - win + 1)
            tab = tabg + bb.to(tl.int64) * TAB_B
            q = tl.load(
                qkv_p + bb.to(tl.int64) * _QKV + hh * _DH + od, mask=od < _DH, other=0.0
            )
            kvh = hh // (_NH // _NKV)
            sink = tl.load(sk + hh).to(tl.float32)
            m_i = sink
            l_i = 0.0
            acc = tl.zeros([BLK_D], dtype=tl.float32)
            # Blocked flash decode: BLOCK_T KV positions per iteration.
            # Measured: the scalar one-position-per-iteration loop cost
            # 26.5 ns per (head, layer, position) -- 66% of runtime at 1k
            # context -- because each position paid a full cross-lane tl.sum.
            # Blocking amortises the reduction and coalesces the K/V loads,
            # measured at 1.4 ns per (head, layer, position).
            t0 = lo
            while t0 <= pb:
                tt = t0 + tl.arange(0, BLOCK_T)
                m_t = tt <= pb
                pgs = tl.load(tab + tt // ps, mask=m_t, other=0)
                locs = (tl.maximum(pgs, 0) * ps + tt % ps).to(tl.int64)
                koff2 = (locs[:, None] * _NKV + kvh) * _DH + od[None, :]
                kk = tl.load(kc + koff2, mask=m_t[:, None], other=0.0)
                s2 = tl.sum(q[None, :] * kk, axis=1) * (1.0 / (_DH**0.5))
                s2 = tl.where(m_t, s2, -float("inf"))
                m_new = tl.maximum(m_i, tl.max(s2, axis=0))
                al = tl.exp(m_i - m_new)
                p = tl.where(m_t, tl.exp(s2 - m_new), 0.0)
                vv = tl.load(vc + koff2, mask=m_t[:, None], other=0.0)
                acc = acc * al + tl.sum(p[:, None] * vv, axis=0)
                l_i = l_i * al + tl.sum(p, axis=0)
                m_i = m_new
                t0 += BLOCK_T
            l_i += tl.exp(sink - m_i)
            tl.store(
                attn_p + bb.to(tl.int64) * (_NH * _DH) + hh * _DH + od,
                acc / l_i,
                mask=od < _DH,
            )
            r += npid
        _barrier(bar_p, b0 + 2, NWG)

        # ---- 4: O-proj + residual ------------------------------------------
        oq = tl.arange(0, BLK_H)
        mq = oq < (_NH * _DH)
        r = pid
        while r < B * _H:
            bb = r // _H
            rr = r % _H
            a_vec = tl.load(
                attn_p + bb.to(tl.int64) * (_NH * _DH) + oq, mask=mq, other=0.0
            )
            w = tl.load(wo + rr.to(tl.int64) * (_NH * _DH) + oq, mask=mq, other=0.0).to(
                tl.float32
            )
            v = tl.sum(w * a_vec) + tl.load(bo + rr).to(tl.float32)
            v += tl.load(hs_p + bb.to(tl.int64) * _H + rr).to(tl.float32)
            tl.store(resid_p + bb.to(tl.int64) * _H + rr, v)
            r += npid
        _barrier(bar_p, b0 + 3, NWG)

        # ---- 5+6 fused: moe RMSNorm (recomputed per program) + router ------
        # yn recomputed per program for its router dots -> no cross-program norm
        # read, norm barrier removed. ynorm_p/ynq_p still populated by pid 0
        # because gate-up reads them; yn is bit-identical to the old store (f32).
        for bb_ in range(B):
            bi = bb_.to(tl.int64)
            yn = _rms_norm_vec(resid_p + bi * _H, mn, _H, BLK_H)
            if pid == 0:
                tl.store(ynorm_p + bi * _H + oh, yn, mask=mh)
                if DOT:
                    yq, ye = _quant_e8m0(yn, BLK_H)
                    tl.store(ynq_p + bi * _H + oh, yq.to(tl.float8e4nv), mask=mh)
                    ob = tl.arange(0, BLK_H // 32)
                    tl.store(
                        yns_p + bi * (_H // 32) + ob,
                        ye.to(tl.uint8),
                        mask=ob < (_H // 32),
                    )
            ee = pid
            while ee < _E:
                w = tl.load(rw + ee.to(tl.int64) * _H + oh, mask=mh, other=0.0).to(
                    tl.float32
                )
                tl.store(
                    rlog_p + bi * _E + ee,
                    tl.sum(w * yn) + tl.load(rb + ee).to(tl.float32),
                )
                ee += npid
        _barrier(bar_p, b0 + 4, NWG)

        # ---- 7: top-4 per token, and seed hs with the residual -------------
        oe = tl.arange(0, BLK_E)
        b = pid
        while b < B:
            rl = tl.load(
                rlog_p + b.to(tl.int64) * _E + oe, mask=oe < _E, other=-float("inf")
            )
            NEG = -float("inf")
            cur = rl
            i0 = tl.argmax(cur, axis=0)
            v0 = tl.max(cur, axis=0)
            cur = tl.where(oe == i0, NEG, cur)
            i1 = tl.argmax(cur, axis=0)
            v1 = tl.max(cur, axis=0)
            cur = tl.where(oe == i1, NEG, cur)
            i2 = tl.argmax(cur, axis=0)
            v2 = tl.max(cur, axis=0)
            cur = tl.where(oe == i2, NEG, cur)
            i3 = tl.argmax(cur, axis=0)
            v3 = tl.max(cur, axis=0)
            vm = tl.maximum(tl.maximum(v0, v1), tl.maximum(v2, v3))
            x0 = tl.exp(v0 - vm)
            x1 = tl.exp(v1 - vm)
            x2 = tl.exp(v2 - vm)
            x3 = tl.exp(v3 - vm)
            zs = x0 + x1 + x2 + x3
            b4 = b.to(tl.int64) * _TOPK
            tl.store(tki_p + b4 + 0, i0)
            tl.store(tkw_p + b4 + 0, x0 / zs)
            tl.store(tki_p + b4 + 1, i1)
            tl.store(tkw_p + b4 + 1, x1 / zs)
            tl.store(tki_p + b4 + 2, i2)
            tl.store(tkw_p + b4 + 2, x2 / zs)
            tl.store(tki_p + b4 + 3, i3)
            tl.store(tkw_p + b4 + 3, x3 / zs)
            b += npid
        r = pid
        while r < B * _H:  # hs := residual; down accumulates
            tl.store(hs_p + r, tl.load(resid_p + r))
            r += npid
        # (fused 7->8: no barrier; gate-up recomputes top-4 locally from rlog)

        # ---- 7+8 fused: gate-up for every (token, expert-slot) -------------
        # (b, j) outer, rows inner: the activation halves load ONCE per (b, j)
        for b in range(B):
            bi = b.to(tl.int64)
            # local top-4 argmax recompute (bit-exact same seq as tki_p store)
            rlg = tl.load(rlog_p + bi * _E + oe, mask=oe < _E, other=-float("inf"))
            NEGf = -float("inf")
            cg = rlg
            j0 = tl.argmax(cg, axis=0)
            cg = tl.where(oe == j0, NEGf, cg)
            j1 = tl.argmax(cg, axis=0)
            cg = tl.where(oe == j1, NEGf, cg)
            j2 = tl.argmax(cg, axis=0)
            cg = tl.where(oe == j2, NEGf, cg)
            j3 = tl.argmax(cg, axis=0)
            oi = tl.arange(0, BLK_K2)
            y_e = tl.load(ynorm_p + bi * _H + 2 * oi, mask=2 * oi < _H, other=0.0)
            y_o = tl.load(
                ynorm_p + bi * _H + 2 * oi + 1, mask=2 * oi + 1 < _H, other=0.0
            )
            for j in range(_TOPK):
                if j == 0:
                    ei = j0.to(tl.int64)
                elif j == 1:
                    ei = j1.to(tl.int64)
                elif j == 2:
                    ei = j2.to(tl.int64)
                else:
                    ei = j3.to(tl.int64)
                gout = gu_p + (bi * _TOPK + j) * (2 * _I)
                if DOT:
                    _gemv_dot_scaled(
                        gb0 + ei * (2 * _I) * (_H // 2),
                        gs0 + ei * (2 * _I) * (_H // 32),
                        ynq_p + bi * _H,
                        yns_p + bi * (_H // 32),
                        gout,
                        gbi0 + ei * (2 * _I),
                        0.0,
                        2 * _I,
                        _H,
                        pid,
                        npid,
                        BLOCK_N,
                        BLOCK_K,
                        MTILE,
                        False,
                    )
                else:
                    # Rows 2*rr and 2*rr+1 are the gate/up pair for output rr.
                    # Owning the PAIR makes this program its own SwiGLU consumer,
                    # so the phase-8 barrier and the gu round-trip both vanish,
                    # and two independent row-dot loads are in flight before
                    # either reduction. Per-row arithmetic is unchanged, so the
                    # result is bit-identical to the split version.
                    gblk = gb0 + ei * (2 * _I) * (_H // 2)
                    gscl = gs0 + ei * (2 * _I) * (_H // 32)
                    gbse = gbi0 + ei * (2 * _I)
                    rr = pid
                    while rr < _I:
                        g = _mxfp4_row_dot(gblk, gscl, 2 * rr, _H, y_e, y_o, BLK_K2)
                        u = _mxfp4_row_dot(gblk, gscl, 2 * rr + 1, _H, y_e, y_o, BLK_K2)
                        g += tl.load(gbse + 2 * rr).to(tl.float32)
                        u += tl.load(gbse + 2 * rr + 1).to(tl.float32)
                        g = tl.minimum(g, _LIMIT)
                        u = tl.minimum(tl.maximum(u, -_LIMIT), _LIMIT)
                        tl.store(
                            act_p + (bi * _TOPK + j) * _I + rr,
                            (g * tl.sigmoid(_ALPHA * g)) * (u + _BETA),
                        )
                        rr += npid
        _barrier(bar_p, b0 + 5, NWG)
        # The DOT path still needs gate-up and SwiGLU as separate phases,
        # so it keeps 11 barriers/layer while the precise path uses 10.
        if DOT:

            # ---- 9: clamped SwiGLU over all (token, slot) ----------------------
            blk = tl.arange(0, BLK_D)
            base = pid * BLK_D
            while base < B * _TOPK * _I:
                ri = base + blk
                m = ri < B * _TOPK * _I
                sl = ri // _I  # (token, slot) index
                rr = ri % _I
                gp = gu_p + sl.to(tl.int64) * (2 * _I)
                g = tl.load(gp + 2 * rr, mask=m, other=0.0)
                u = tl.load(gp + 2 * rr + 1, mask=m, other=0.0)
                g = tl.minimum(g, _LIMIT)
                u = tl.minimum(tl.maximum(u, -_LIMIT), _LIMIT)
                glu = g * tl.sigmoid(_ALPHA * g)
                av = glu * (u + _BETA)
                tl.store(act_p + ri, av, mask=m)
                if DOT:
                    aq, ae = _quant_e8m0(av, BLK_D)
                    tl.store(actq_p + ri, aq.to(tl.float8e4nv), mask=m)
                    sb = base // 32 + tl.arange(0, BLK_D // 32)
                    tl.store(
                        acts_p + sb, ae.to(tl.uint8), mask=sb < (B * _TOPK * _I // 32)
                    )
                base += npid * BLK_D
            _barrier(bar_p, b0 + 6, NWG)

        # ---- 10: down-proj, accumulating straight into hs ------------------
        # j is an inner loop with no barrier between iterations, so program pid
        # must own the same rows for every j -- `r = pid; r += npid` does.
        for b in range(B):
            bi = b.to(tl.int64)
            for j in range(_TOPK):
                sl = bi * _TOPK + j
                ei = tl.load(tki_p + bi * _TOPK + j).to(tl.int64)
                wgt = tl.load(tkw_p + bi * _TOPK + j)
                if DOT:
                    _gemv_dot_scaled(
                        db0 + ei * _H * (_I // 2),
                        ds0 + ei * _H * (_I // 32),
                        actq_p + sl * _I,
                        acts_p + sl * (_I // 32),
                        hs_p + bi * _H,
                        dbi0 + ei * _H,
                        wgt,
                        _H,
                        _I,
                        pid,
                        npid,
                        BLOCK_N,
                        BLOCK_K,
                        MTILE,
                        True,
                    )
                else:
                    ai = tl.arange(0, BLK_K2)
                    a_e = tl.load(act_p + sl * _I + 2 * ai, mask=2 * ai < _I, other=0.0)
                    a_o = tl.load(
                        act_p + sl * _I + 2 * ai + 1, mask=2 * ai + 1 < _I, other=0.0
                    )
                    r = pid
                    while r < _H:
                        v = _mxfp4_row_dot(
                            db0 + ei * _H * (_I // 2),
                            ds0 + ei * _H * (_I // 32),
                            r,
                            _I,
                            a_e,
                            a_o,
                            BLK_K2,
                        )
                        v += tl.load(dbi0 + ei * _H + r).to(tl.float32)
                        tl.store(
                            hs_p + bi * _H + r, tl.load(hs_p + bi * _H + r) + wgt * v
                        )
                        r += npid
        _barrier(bar_p, b0 + _NB - 1, NWG)

        for xb in range(EXTRA_BAR):
            _barrier(bar_p, b0 + _NB + xb, NWG)

    # ---- final RMSNorm ------------------------------------------------------
    b = pid
    while b < B:
        fn = _rms_norm_vec(hs_p + b.to(tl.int64) * _H, fnorm_p, _H, BLK_H)
        tl.store(out_p + b.to(tl.int64) * _H + oh, fn, mask=mh)
        b += npid
