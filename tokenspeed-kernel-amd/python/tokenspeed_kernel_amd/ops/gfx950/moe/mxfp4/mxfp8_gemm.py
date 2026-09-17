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

"""M32 or M128, N128/K256 MXFP8 x MXFP4 expert GEMMs.

The two logical N64 fragments give each wave adjacent N16 gate/up (stage1)
or output-column (stage2) fragments, with two or eight M16 repeats. Stage1
uses async activation loads at both row counts; stage2 retains register loads
and phased LDS publication. Output exchange happens after FP32 arithmetic.
"""

from tokenspeed_kernel_amd._scheduling import sched_barrier
from tokenspeed_kernel_amd._triton import cdna4_async_copy, gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.n16_weights import (
    _n16_weight_offset,
)

_MMA = gl.constexpr(
    gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 128],
        transposed=False,
        warps_per_cta=[1, 4],
    )
)
_A = gl.constexpr(gl.DotOperandLayout(0, _MMA, 16))
_B = gl.constexpr(gl.DotOperandLayout(1, _MMA, 16))
_AS = gl.constexpr(gl.amd.cdna4.get_mfma_scale_layout(_A, [16, 4]))
_BS = gl.constexpr(gl.amd.cdna4.get_mfma_scale_layout(_B, [64, 4]))
_LOAD = gl.constexpr(gl.BlockedLayout([1, 16], [4, 16], [4, 1], [1, 0]))
_SHARED = gl.constexpr(gl.SwizzledSharedLayout(16, 1, 16, [1, 0]))
_PHYSICAL = gl.constexpr(gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))


@gluon.jit
def _load_b(w, K: gl.constexpr, KT: gl.constexpr, KH: gl.constexpr, NI: gl.constexpr):
    k = gl.arange(0, 64, layout=gl.SliceLayout(1, _B))
    n = gl.arange(0, 64, layout=gl.SliceLayout(0, _B))
    n = n // 16 * 32 + n % 16 + NI * 16
    offset = _n16_weight_offset(n[None, :], k[:, None], K // 2)
    # Keep one CTA descriptor; K progress is a final byte-offset add.
    return gl.amd.cdna4.buffer_load(w, offset.to(gl.int32) + (KT * 2 + KH) * 1024)


@gluon.jit
def _load_scales(sa, sb, K: gl.constexpr, KT: gl.constexpr, BM: gl.constexpr):
    row = gl.arange(0, 16, layout=gl.SliceLayout(1, _AS))
    kg = gl.arange(0, 4, layout=gl.SliceLayout(0, _AS))
    a_words = ()
    for mp in gl.static_range(BM // 32):
        offset = mp * (K // 256) * 64
        word = gl.amd.cdna4.buffer_load(
            sa,
            offset + kg[None, :] * 16 + row[:, None] + KT * 64,
        )
        a_words += (word,)
    n = gl.arange(0, 64, layout=gl.SliceLayout(1, _BS))
    kg_b = gl.arange(0, 4, layout=gl.SliceLayout(0, _BS))
    offset_b = (n[:, None] // 16) * (K // 256) * 64
    b_word = gl.amd.cdna4.buffer_load(
        sb,
        offset_b + kg_b[None, :] * 16 + n[:, None] % 16 + KT * 64,
    )
    return a_words, b_word


@gluon.jit
def _a_offsets(
    ids,
    m_base,
    M,
    TOPK: gl.constexpr,
    K: gl.constexpr,
    SLOT_MAJOR: gl.constexpr,
    BM: gl.constexpr,
    DMA: gl.constexpr,
):
    row = gl.arange(0, 16, layout=gl.SliceLayout(1, _LOAD))
    k = gl.arange(0, 256, layout=gl.SliceLayout(0, _LOAD))
    offsets = ()
    for part in gl.static_range(BM // 16):
        encoded = gl.load(ids + m_base + part * 16 + row).to(gl.uint32)
        token, slot = encoded & 0xFFFFFF, encoded >> 24
        valid = (token < M) & (slot < TOPK)
        token, slot = gl.where(valid, token, 0), gl.where(valid, slot, 0)
        source = token * TOPK + slot if SLOT_MAJOR else token
        col = k[None, :]
        if DMA:
            # DMA writes linear physical LDS addresses. XOR the source columns
            # instead, so the MFMA view sees the same XOR16 tile without shuffles.
            # XOR keeps each 16-byte group aligned; provide both DMA hints.
            col = gl.max_contiguous(
                gl.multiple_of(col ^ (row[:, None] * 16), [1, 16]), [1, 16]
            )
        offsets += (source[:, None].to(gl.int64) * K + col,)
    return offsets


@gluon.jit
def _copy_a(smem, x, offsets, KT: gl.constexpr, BUFFER_SAFE: gl.constexpr):
    x = x.to(gl.pointer_type(gl.uint8))
    physical = smem.reinterpret(layout=_PHYSICAL)
    for part in gl.static_range(len(offsets)):
        dest = physical.slice(part * 16, 16, 0)
        if BUFFER_SAFE:
            cdna4_async_copy.buffer_load_to_shared(
                dest,
                x,
                offsets[part].to(gl.int32) + KT * 256,
            )
        else:
            cdna4_async_copy.global_load_to_shared(
                dest,
                x + offsets[part] + KT * 256,
            )
    cdna4_async_copy.commit_group()


@gluon.jit
def _load_a(
    x,
    offsets,
    KT: gl.constexpr,
    BUFFER_SAFE: gl.constexpr,
    BEGIN: gl.constexpr,
    END: gl.constexpr,
):
    """Load physical portions [BEGIN, END) into a compact tuple."""
    x = x.to(gl.pointer_type(gl.uint8))
    a = ()
    for part in gl.static_range(BEGIN, END):
        if BUFFER_SAFE:
            value = gl.amd.cdna4.buffer_load(x, offsets[part].to(gl.int32) + KT * 256)
        else:
            value = gl.load(x + offsets[part] + KT * 256)
        a += (value,)
    return a


@gluon.jit
def _publish_a(smem, a, BEGIN: gl.constexpr, END: gl.constexpr):
    """Publish a compact tuple starting at physical portion BEGIN."""
    for part in gl.static_range(BEGIN, END):
        smem.slice(part * 16, 16, 0).store(a[part - BEGIN])


@gluon.jit
def _fragment_k64(smem, MI: gl.constexpr, KH: gl.constexpr, HALF: gl.constexpr):
    return smem.slice(MI * 16, 16, 0).slice(KH * 128 + HALF * 64, 64, 1).load(_A)


@gluon.jit
def _assemble_fragment(lo, hi):
    # Join whole per-lane K64 chunks, not alternating bytes from the two halves.
    half_layout: gl.constexpr = gl.SliceLayout(
        2, gl.BlockedLayout([1, 16, 2], [16, 4, 1], [1, 4, 1], [0, 1, 2])
    )
    lo = gl.convert_layout(lo, half_layout, assert_trivial=True)
    hi = gl.convert_layout(hi, half_layout, assert_trivial=True)
    joined = gl.join(lo, hi).permute(0, 2, 1).reshape(16, 128)
    return gl.convert_layout(joined, _A, assert_trivial=True).to(
        gl.float8e4nv, bitcast=True
    )


@gluon.jit
def _fragment(smem, MI: gl.constexpr, KH: gl.constexpr):
    return _assemble_fragment(
        _fragment_k64(smem, MI, KH, 0), _fragment_k64(smem, MI, KH, 1)
    )


@gluon.jit
def _full_fragments(smem, BM: gl.constexpr):
    a = ()
    for kh in gl.static_range(2):
        for mi in gl.static_range(BM // 16):
            a += (_fragment(smem, mi, kh),)
    return a


@gluon.jit
def _zeros(BM: gl.constexpr):
    acc = ()
    for mi in gl.static_range(BM // 16):
        acc += (
            (
                gl.zeros([16, 64], gl.float32, _MMA),
                gl.zeros([16, 64], gl.float32, _MMA),
            ),
        )
    return acc


@gluon.jit
def _bmajor_phase(acc, a, b, a_words, b_word, KH: gl.constexpr, NI: gl.constexpr):
    updated = ()
    bs = (b_word >> (8 * (2 * KH + NI))).to(gl.uint8)
    for mi in gl.static_range(len(acc)):
        scale = (a_words[mi // 2] >> (8 * (2 * KH + mi % 2))).to(gl.uint8)
        value = gl.amd.cdna4.mfma_scaled(
            a[KH * len(acc) + mi],
            scale,
            "e4m3",
            b,
            bs,
            "e2m1",
            acc[mi][NI],
        )
        if NI == 0:
            updated += ((value, acc[mi][1]),)
        else:
            updated += ((acc[mi][0], value),)
    return updated


@gluon.jit
def _stage1_interleaved(
    acc,
    old_a,
    old_b,
    old_as,
    old_bs,
    smem,
    w,
    sa,
    sb,
    K: gl.constexpr,
    NEXT: gl.constexpr,
    BM: gl.constexpr,
):
    # Four B-major phases. Distribute the next tile's M16 reads before
    # their consumers: two per phase at M128, one in phases 0/1 at M32.
    new_as, new_bs = _load_scales(sa, sb, K, NEXT, BM)
    READS: gl.constexpr = gl.cdiv(BM // 16, 4)
    new_a = ()
    new_b = ()
    for phase in gl.static_range(4):
        if phase == 1:
            new_b += (
                _load_b(w, K, NEXT, 0, 0),
                _load_b(w, K, NEXT, 0, 1),
            )
        elif phase == 2:
            new_b += (_load_b(w, K, NEXT, 1, 0),)
        elif phase == 3:
            new_b += (_load_b(w, K, NEXT, 1, 1),)
        sched_barrier()
        for mi in gl.static_range(phase * READS, min((phase + 1) * READS, BM // 16)):
            for kh in gl.static_range(2):
                new_a += (_fragment(smem, mi, kh),)
        sched_barrier()
        acc = _bmajor_phase(
            acc, old_a, old_b[phase], old_as, old_bs, phase // 2, phase % 2
        )
        sched_barrier()
    # The phase read order is M-major; the consumer tuple is K-half-major.
    a = ()
    for kh in gl.static_range(2):
        for mi in gl.static_range(BM // 16):
            a += (new_a[mi * 2 + kh],)
    return acc, a, new_b, new_as, new_bs


@gluon.jit
def _stage2_m32_pair(acc, smem, b_lo, b_hi, a_words, b_word, first_lo, first_hi):
    for kh in gl.static_range(2):
        b = b_lo if kh == 0 else b_hi
        fragments = ()
        for mi in gl.static_range(2):
            lo = (
                (first_lo if kh == 0 else first_hi)
                if mi == 0
                else _fragment_k64(smem, mi, kh, 0)
            )
            fragments += (_assemble_fragment(lo, _fragment_k64(smem, mi, kh, 1)),)
        updated = ()
        for mi in gl.static_range(2):
            scale = (a_words[0] >> (8 * (2 * kh + mi))).to(gl.uint8)
            c0 = gl.amd.cdna4.mfma_scaled(
                fragments[mi],
                scale,
                "e4m3",
                b[0],
                (b_word >> (16 * kh)).to(gl.uint8),
                "e2m1",
                acc[mi][0],
            )
            c1 = gl.amd.cdna4.mfma_scaled(
                fragments[mi],
                scale,
                "e4m3",
                b[1],
                (b_word >> (16 * kh + 8)).to(gl.uint8),
                "e2m1",
                acc[mi][1],
            )
            updated += ((c0, c1),)
        acc = updated
    return acc


@gluon.jit
def _stage2_prefix(
    acc,
    smem,
    b_lo,
    b_hi,
    a_words,
    b_word,
    first_lo,
    first_hi,
    next_slot,
    next_a,
    x,
    offsets,
    KT: gl.constexpr,
    BUFFER_SAFE: gl.constexpr,
    PUBLISH_NEXT: gl.constexpr,
):
    for kh in gl.static_range(2):
        if PUBLISH_NEXT and kh == 1:
            # Six portions publish after the prefix, not after the first MFMA.
            # Acquire them here while the second K half still has compute left.
            next_a_tail = _load_a(x, offsets, KT + 1, BUFFER_SAFE, 2, 8)
        b = b_lo if kh == 0 else b_hi
        updated = ()
        for mp in gl.static_range(4 if kh == 0 else 2):
            fragments = ()
            for im in gl.static_range(2):
                if mp == 0 and im == 0:
                    lo = first_lo if kh == 0 else first_hi
                else:
                    lo = _fragment_k64(smem, 2 * mp + im, kh, 0)
                hi = _fragment_k64(smem, 2 * mp + im, kh, 1)
                fragments += (_assemble_fragment(lo, hi),)
            for im in gl.static_range(2):
                a = fragments[im]
                scale = (a_words[mp] >> (8 * (2 * kh + im))).to(gl.uint8)
                c0 = gl.amd.cdna4.mfma_scaled(
                    a,
                    scale,
                    "e4m3",
                    b[0],
                    (b_word >> (16 * kh)).to(gl.uint8),
                    "e2m1",
                    acc[2 * mp + im][0],
                )
                if PUBLISH_NEXT and kh == 0 and mp == 0 and im == 0:
                    # Begin next-slot publication after the first MFMA.
                    _publish_a(next_slot, next_a, 0, 2)
                c1 = gl.amd.cdna4.mfma_scaled(
                    a,
                    scale,
                    "e4m3",
                    b[1],
                    (b_word >> (16 * kh + 8)).to(gl.uint8),
                    "e2m1",
                    acc[2 * mp + im][1],
                )
                updated += ((c0, c1),)
        if kh == 1:
            for mi in gl.static_range(4, 8):
                updated += (acc[mi],)
        acc = updated
    # Acquire every remaining current-slot fragment before permitting reuse.
    # Its last eight MFMAs then follow the next global-load prologue.
    tail_a = ()
    for mi in gl.static_range(4, 8):
        tail_a += (_fragment(smem, mi, 1),)
    if PUBLISH_NEXT:
        _publish_a(next_slot, next_a_tail, 2, 8)
    return acc, tail_a


@gluon.jit
def _stage2_tail(acc, a, b, a_words, b_word):
    updated = ()
    for mi in gl.static_range(4):
        updated += (acc[mi],)
    for mi in gl.static_range(4, 8):
        scale = (a_words[mi // 2] >> (8 * (2 + mi % 2))).to(gl.uint8)
        c0 = gl.amd.cdna4.mfma_scaled(
            a[mi - 4],
            scale,
            "e4m3",
            b[0],
            (b_word >> 16).to(gl.uint8),
            "e2m1",
            acc[mi][0],
        )
        c1 = gl.amd.cdna4.mfma_scaled(
            a[mi - 4],
            scale,
            "e4m3",
            b[1],
            (b_word >> 24).to(gl.uint8),
            "e2m1",
            acc[mi][1],
        )
        updated += ((c0, c1),)
    return updated


@gluon.jit
def _tanh(x):
    e = gl.exp2(-2.8853900817779268 * gl.abs(x))
    t = gl.extra.libdevice.fast_dividef(1.0 - e, 1.0 + e)
    return gl.where(x > 0.0, t, -t)


@gluon.jit
def _situ(gate, up, BETA: gl.constexpr, LINEAR_BETA: gl.constexpr):
    sigmoid = gl.extra.libdevice.fast_dividef(
        1.0, 1.0 + gl.exp2(-1.4426950408889634 * gate)
    )
    gate = BETA * _tanh(gate * (1.0 / BETA)) * sigmoid
    up = LINEAR_BETA * _tanh(up * (1.0 / LINEAR_BETA))
    return (gate * up).to(gl.bfloat16)


@gluon.jit(do_not_specialize=("M",))
def _mxfp8_stage1(
    x,
    sa,
    w,
    sb,
    ids,
    experts,
    valid_ids,
    z,
    M,
    TOPK: gl.constexpr,
    K: gl.constexpr,
    I: gl.constexpr,
    E: gl.constexpr,
    BETA: gl.constexpr,
    LINEAR_BETA: gl.constexpr,
    BUFFER_SAFE: gl.constexpr,
    SCHED_LIBRARY_HASH: gl.constexpr,  # Cache dependency; not a device operand.
    BM: gl.constexpr,
):
    gl.static_assert(BM == 32 or BM == 128)
    m_base = gl.program_id(1) * BM
    if m_base >= gl.load(valid_ids):
        return
    expert = gl.load(experts + gl.program_id(1))
    first = gl.load(ids + m_base).to(gl.uint32) & 0xFFFFFF
    if expert < 0 or expert >= E or first >= M:
        return
    n_base = gl.program_id(0) * 128
    w += expert.to(gl.int64) * (2 * I * (K // 2))
    sb += expert.to(gl.int64) * (2 * I * (K // 32))
    w += n_base.to(gl.int64) * (K // 2)
    sa = (sa + m_base.to(gl.int64) * (K // 32)).to(gl.pointer_type(gl.uint32))
    sb = (sb + n_base.to(gl.int64) * (K // 32)).to(gl.pointer_type(gl.uint32))
    a0 = gl.allocate_shared_memory(gl.uint8, [BM, 256], _SHARED)
    a1 = gl.allocate_shared_memory(gl.uint8, [BM, 256], _SHARED)
    meta_layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    row = gl.arange(0, BM, layout=meta_layout)
    route = gl.load(ids + m_base + row)
    tid = gl.allocate_shared_memory(
        gl.int32, [BM], gl.SwizzledSharedLayout(1, 1, 1, [0]), route
    )
    offsets = _a_offsets(ids, m_base, M, TOPK, K, False, BM, True)
    acc = _zeros(BM)
    sched_barrier()
    _copy_a(a0, x, offsets, 0, BUFFER_SAFE)
    sched_barrier()
    scales_a, scales_b = _load_scales(sa, sb, K, 0, BM)
    sched_barrier()
    if K > 256:
        _copy_a(a1, x, offsets, 1, BUFFER_SAFE)
    b = ()
    for kh in gl.static_range(2):
        for ni in gl.static_range(2):
            b += (_load_b(w, K, 0, kh, ni),)
    cdna4_async_copy.wait_group(0)
    sched_barrier()
    a = _full_fragments(a0, BM)
    sched_barrier()

    TAIL: gl.constexpr = 1 if K // 256 % 2 else 2
    for kt in gl.static_range(0, K // 256 - TAIL):
        read_slot = a1 if kt % 2 == 0 else a0
        write_slot = a0 if kt % 2 == 0 else a1
        sched_barrier()
        cdna4_async_copy.wait_group(0)
        sched_barrier()
        if kt + 2 < K // 256:
            _copy_a(write_slot, x, offsets, kt + 2, BUFFER_SAFE)
        acc, a, b, scales_a, scales_b = _stage1_interleaved(
            acc,
            a,
            b,
            scales_a,
            scales_b,
            read_slot,
            w,
            sa,
            sb,
            K,
            kt + 1,
            BM,
        )
    if TAIL == 2:
        _copy_a(a1, x, offsets, K // 256 - 1, BUFFER_SAFE)
        tail_b = ()
        for kh in gl.static_range(2):
            for ni in gl.static_range(2):
                tail_b += (_load_b(w, K, K // 256 - 1, kh, ni),)
        tail_as, tail_bs = _load_scales(sa, sb, K, K // 256 - 1, BM)
    for phase in gl.static_range(4):
        acc = _bmajor_phase(acc, a, b[phase], scales_a, scales_b, phase // 2, phase % 2)
    if TAIL == 2:
        cdna4_async_copy.wait_group(0)
        tail_a = _full_fragments(a1, BM)
        for phase in gl.static_range(4):
            acc = _bmajor_phase(
                acc, tail_a, tail_b[phase], tail_as, tail_bs, phase // 2, phase % 2
            )

    cdna4_async_copy.wait_group(0)
    # Alias retired A storage: gate/up interleaving emits only N64 BF16.
    c_shared = a0.reinterpret(
        gl.bfloat16, [2 * BM, 64], gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    )
    for mi in gl.static_range(BM // 16):
        c_shared.slice(mi * 16, 16, 0).store(
            _situ(acc[mi][0], acc[mi][1], BETA, LINEAR_BETA)
        )
    store_layout: gl.constexpr = gl.BlockedLayout([1, 2], [2, 32], [2, 2], [1, 0])
    c = c_shared.slice(0, BM, 0).load(store_layout)
    route = tid.load(gl.SliceLayout(1, store_layout)).to(gl.uint32)
    token, slot = route & 0xFFFFFF, route >> 24
    n = gl.program_id(0) * 64 + gl.arange(0, 64, layout=gl.SliceLayout(0, store_layout))
    gl.store(
        z + (token[:, None].to(gl.int64) * TOPK + slot[:, None]) * I + n[None, :],
        c,
        mask=(token[:, None] < M) & (slot[:, None] < TOPK),
    )


@gluon.jit(do_not_specialize=("M",))
def _mxfp8_stage2(
    x,
    sa,
    w,
    sb,
    ids,
    weights,
    experts,
    valid_ids,
    out,
    M,
    TOPK: gl.constexpr,
    K: gl.constexpr,
    N: gl.constexpr,
    E: gl.constexpr,
    OUT_STRIDE: gl.constexpr,
    BUFFER_A_SAFE: gl.constexpr,
    BUFFER_OUT_SAFE: gl.constexpr,
    SCHED_LIBRARY_HASH: gl.constexpr,  # Cache dependency; not a device operand.
    BM: gl.constexpr,
):
    gl.static_assert(BM == 32 or BM == 128)
    m_base = gl.program_id(1) * BM
    if m_base >= gl.load(valid_ids):
        return
    expert = gl.load(experts + gl.program_id(1))
    first = gl.load(ids + m_base).to(gl.uint32) & 0xFFFFFF
    if expert < 0 or expert >= E or first >= M:
        return
    n_base = gl.program_id(0) * 128
    w += expert.to(gl.int64) * (N * (K // 2))
    sb += expert.to(gl.int64) * (N * (K // 32))
    w += n_base.to(gl.int64) * (K // 2)
    sa = (sa + m_base.to(gl.int64) * (K // 32)).to(gl.pointer_type(gl.uint32))
    sb = (sb + n_base.to(gl.int64) * (K // 32)).to(gl.pointer_type(gl.uint32))
    a0 = gl.allocate_shared_memory(gl.uint8, [BM, 256], _SHARED)
    a1 = gl.allocate_shared_memory(gl.uint8, [BM, 256], _SHARED)
    meta_layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    row = gl.arange(0, BM, layout=meta_layout)
    route = gl.load(ids + m_base + row).to(gl.uint32)
    token, slot = route & 0xFFFFFF, route >> 24
    valid = (token < M) & (slot < TOPK)
    weight = gl.load(weights + m_base + row, mask=valid, other=0.0).to(gl.float32)
    offset_type: gl.constexpr = gl.int32 if BUFFER_OUT_SAFE else gl.int64
    row_bytes = gl.where(valid, token, 0).to(offset_type) * (OUT_STRIDE * 2)
    row_bytes = gl.where(valid, row_bytes, -1)
    tid = gl.allocate_shared_memory(
        offset_type, [BM], gl.SwizzledSharedLayout(1, 1, 1, [0]), row_bytes
    )
    tw = gl.allocate_shared_memory(
        gl.float32, [BM], gl.SwizzledSharedLayout(1, 1, 1, [0]), weight
    )
    offsets = _a_offsets(ids, m_base, M, TOPK, K, True, BM, False)
    acc = _zeros(BM)
    sched_barrier()
    b_lo = (_load_b(w, K, 0, 0, 0), _load_b(w, K, 0, 0, 1))
    scales_a, scales_b = _load_scales(sa, sb, K, 0, BM)
    sched_barrier()
    _publish_a(a0, _load_a(x, offsets, 0, BUFFER_A_SAFE, 0, BM // 16), 0, BM // 16)
    first_lo, first_hi = _fragment_k64(a0, 0, 0, 0), _fragment_k64(a0, 0, 1, 0)
    tail_a, tail_b, tail_as, tail_bs = None, None, None, None
    for kt in gl.static_range(0, K // 256):
        current = a0 if kt % 2 == 0 else a1
        next_slot = a1 if kt % 2 == 0 else a0
        next_a = None
        if kt + 1 < K // 256:
            next_a = _load_a(x, offsets, kt + 1, BUFFER_A_SAFE, 0, 2)
            next_as, next_bs = _load_scales(sa, sb, K, kt + 1, BM)
            next_b = (
                _load_b(w, K, kt + 1, 0, 0),
                _load_b(w, K, kt + 1, 0, 1),
            )
        b_hi = (_load_b(w, K, kt, 1, 0), _load_b(w, K, kt, 1, 1))
        if BM == 128 and kt > 0:
            acc = _stage2_tail(acc, tail_a, tail_b, tail_as, tail_bs)
        if kt == K // 256 - 1:
            epilogue_weights = ()
            for mi in gl.static_range(BM // 16):
                epilogue_weights += (
                    tw.slice(mi * 16, 16, 0).load(gl.SliceLayout(1, _MMA)),
                )
        if BM == 32:
            acc = _stage2_m32_pair(
                acc, current, b_lo, b_hi, scales_a, scales_b, first_lo, first_hi
            )
            if kt + 1 < K // 256:
                _publish_a(next_slot, next_a, 0, 2)
        else:
            acc, tail_a = _stage2_prefix(
                acc,
                current,
                b_lo,
                b_hi,
                scales_a,
                scales_b,
                first_lo,
                first_hi,
                next_slot,
                next_a,
                x,
                offsets,
                kt,
                BUFFER_A_SAFE,
                kt + 1 < K // 256,
            )
            tail_b, tail_as, tail_bs = b_hi, scales_a, scales_b
        if kt + 1 < K // 256:
            first_lo = _fragment_k64(next_slot, 0, 0, 0)
            first_hi = _fragment_k64(next_slot, 0, 1, 0)
            b_lo, scales_a, scales_b = next_b, next_as, next_bs

    if BM == 128:
        acc = _stage2_tail(acc, tail_a, tail_b, tail_as, tail_bs)
    c_shared = a0.reinterpret(
        gl.bfloat16, [BM, 128], gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    )
    for mi in gl.static_range(BM // 16):
        weight = epilogue_weights[mi]
        c0 = (acc[mi][0] * weight[:, None]).to(gl.bfloat16)
        c1 = (acc[mi][1] * weight[:, None]).to(gl.bfloat16)
        c = (
            gl.join(c0.reshape(16, 4, 16), c1.reshape(16, 4, 16))
            .permute(0, 1, 3, 2)
            .reshape(16, 128)
        )
        c_shared.slice(mi * 16, 16, 0).store(c)
    store_layout: gl.constexpr = gl.BlockedLayout([1, 2], [2, 32], [4, 1], [1, 0])
    c = c_shared.load(store_layout)
    row_bytes = tid.load(gl.SliceLayout(1, store_layout))
    n = n_base + gl.arange(0, 128, layout=gl.SliceLayout(0, store_layout))
    offset = (row_bytes[:, None] >> 1) + n[None, :]
    valid = row_bytes[:, None] >= 0
    if BUFFER_OUT_SAFE:
        gl.amd.cdna4.buffer_atomic_add(
            out, offset.to(gl.int32), c, mask=valid, sem="relaxed"
        )
    else:
        gl.atomic_add(out + offset, c, mask=valid, sem="relaxed")
