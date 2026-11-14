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

"""Attention of one Muse Spark rank: 16 q heads, 2 KV heads.

The launch's rows are the consecutive tokens of one or more requests
(``Plan.chain`` a request): one row a request for decode, ``k + 1`` for a
speculative verify of ``k`` draft tokens. Row ``t`` of a request is its
token at position ``L0 + t``, where ``L0`` is the request's own.

* head, 2 tasks per row: KV head ``g``. The per-head RMS norm of k in FP32
  (gain 1), then interleaved RoPE (GPT-J pairs) on sliding layers, then one
  rounding to the cache's dtype (BF16, or FP8 e4m3 at unit scale, as the
  regular path's prologue writes it); the rotated k and the raw v go to the
  KV cache slot of the row's token.
* attn, ``2 * SPLITS`` tasks a request: KV head ``g``, split ``s``, for every
  row of the request (its length and page table row are its own). A
  task forms the rotated q of its 8 q heads (q gain 5.473) of every row,
  one MFMA row per (row, head), and the launch's k and v itself, as head
  does, from the projection: no hand-off between them. It takes the
  launch's k and v from there, not from the cache: a CTA never reads
  another CTA's plain store of the same launch; it rounds them to the
  cache's dtype as head does, so they are the cached values. An FP8 cache's
  keys and values widen to BF16 (exactly) for the BF16 MFMAs, except on a
  full layer at long context (the key groups, from ``LONG_KEYS`` cached
  keys): there q rounds once from FP32 to the cache's FP8, as the regular
  path's prologue does, and scores the cached keys as stored with FP8 MFMAs
  (FP32 sums); P x V stays BF16. Row ``t`` attends keys
  ``max(0, L0 + t - WINDOW + 1) .. L0 + t`` (from 0 on a full layer). The
  splits share the cached keys before ``L0``, ``ceil(n / SPLITS)`` each
  (from the first key that row 0 attends), in chunks of 128 with an online
  base-2 softmax; a ring in LDS keeps two chunks' K and V copying while one
  is computed. On a full layer with 8 token rows from ``LONG_KEYS`` cached
  keys, the warps instead take 16 rows each in two key groups with their own
  softmax state (:func:`_group_keys`), and a BF16 cache's splits are whole
  pages, copied a page at a time. The last split then also takes the
  launch's tokens, as one more chunk. A task publishes FP32 ``acc / l`` and
  the LSE of the 8 q heads of its KV head of each row (an empty split
  publishes ``-inf``) as plain
  words, then one flag (``Mailbox.store_words`` / ``raise_flag``): at 8 rows
  a task writes 64 x 64 values, and tagged pairs would double the stores.
* merge, 16 tasks per row, one per q head: after the splits' flags, the LSE
  merge of the splits,
  BF16, then the regular path's gated head norm ``bf16(v rsqrt(mean(v^2) +
  eps) sigmoid(gate))``, published in ``z`` for the output projection.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._extern import wave_uniform_i32
from tokenspeed_kernel_amd._triton import cdna4_async_copy, gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mx, schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.dims import (
    GATE_ROW,
    HEAD_DIM,
    K_ROW,
    KV,
    KV_HEADS,
    Q_HEADS,
    QKVG,
    V_ROW,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.qkvg import PROJ_WORDS
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import Stage, placement

cdna4 = gl.amd.cdna4

HEAD = HEAD_DIM
GROUP = gl.constexpr(8)  # q heads per KV head
PAGE = gl.constexpr(64)  # tokens a KV page
KV_STRIDE = KV  # cache elements a slot: its 2 KV heads
# Keys a chunk, the same for either cache dtype: an FP8 chunk takes half the
# LDS bytes of a BF16 one.
BN = gl.constexpr(128)
# Chunks of K and V in LDS: two in flight while one is computed.
NBUF = gl.constexpr(3)
# Cached keys from which a full layer's splits take the key-group loop
# (_group_keys): below, its whole-page splits and once-a-split combine cost
# more than they save.
LONG_KEYS = gl.constexpr(32768)
WINDOW = 2048  # a sliding layer attends the last 2048 positions, its own included
EPS = gl.constexpr(1e-5)
Q_GAIN = gl.constexpr(5.473006248474121)  # qk_scale_factor as an FP32 norm gain
# exp2 softmax of (q . k) / sqrt(64).
QK_SCALE = gl.constexpr(0.125 * 1.4426950408889634)
HEAD_TASKS = KV_HEADS
MERGE_TASKS = Q_HEADS


def regions(splits: int) -> tuple[tuple[str, int], ...]:
    heads, kv, group, head = int(Q_HEADS), int(KV_HEADS), int(GROUP), int(HEAD)
    return (
        ("part", splits * kv * group * head),  # FP32 words (half the region)
        ("lse", splits * heads),  # FP32 words (half the region)
        ("pflag", splits * kv),  # a flag per attn task, in its request's first row
        ("z", heads * head // 2),  # BF16 pairs
    )


def stages(
    bases: dict[str, int], splits: int, rows: int, chain: int, own: bool
) -> tuple[Stage, ...]:
    """Head, attn and merge of ``rows`` token rows (requests of ``chain``
    tokens), placed by ``bases``. Head and attn poll the launch's qkvg
    tasks (``own``), or read the projection that the previous launch's
    qnext tasks stored."""
    head, attn, merge = placement(bases, ("head", "attn", "merge"))
    proj = ("qkvg",) if own else ()
    return (
        Stage("head", int(HEAD_TASKS) * rows, head, reads=proj),
        Stage(
            "attn",
            rows // chain * int(KV_HEADS) * splits,
            attn,
            reads=proj,
            writes=("part", "lse", "pflag"),
        ),
        Stage(
            "merge",
            int(MERGE_TASKS) * rows,
            merge,
            reads=("part", "lse", "pflag", "gate"),
            writes=("z",),
        ),
    )


@gluon.constexpr_function
def _pair_layout(rank, nw):
    """Index layouts of polls of projection pairs: the last axis (half 0
    even, half 0 odd, half 1 even, half 1 odd) in one thread. ``rank`` 2:
    ``[32 pairs, 4]``; 3: ``[16 heads, 32 pairs, 4]``."""
    if rank == 2:
        return gl.BlockedLayout([1, 4], [64, 1], [nw, 1], [1, 0])
    return gl.BlockedLayout([1, 1, 4], [2, 32, 1], [nw, 1, 1], [2, 1, 0])


@gluon.jit
def _pair_index(pair, e, N: gl.constexpr, row0):
    """Partial-sum indices of values ``row0 + 2 pair`` and ``+ 1`` of a
    projection region of ``N`` rows a K half (``e``: the last axis)."""
    return (e // 2) * N + row0 + 2 * pair + e % 2


@gluon.jit
def _pair_values(got, e, AX: gl.constexpr):
    """(even, odd) BF16-rounded values (FP32) from the polled partial sums
    of :func:`_pair_index` (``e``: their last axis ``AX``): half 0 plus
    half 1."""
    x = got.to(gl.float32, bitcast=True)
    lo = gl.sum(gl.where(e % 2 == 0, x, 0.0), axis=AX)
    hi = gl.sum(gl.where(e % 2 == 1, x, 0.0), axis=AX)
    return lo.to(gl.bfloat16).to(gl.float32), hi.to(gl.bfloat16).to(gl.float32)


@gluon.jit
def _norm_rope(lo, hi, gain, cos_sin, pos, j, ROPE: gl.constexpr):
    """Per-head RMS norm over the 64 values (pairs ``j``), then RoPE, in
    FP32."""
    inv = gl.rsqrt(gl.sum(lo * lo + hi * hi, axis=0) / HEAD + EPS)
    lo = lo * inv * gain
    hi = hi * inv * gain
    if ROPE:
        # The row's offset grows with the position: int64.
        rope = cos_sin + pos.to(gl.int64) * HEAD
        c = gl.load(rope + j)
        s = gl.load(rope + HEAD // 2 + j)
        lo, hi = gl.fma(lo, c, -(hi * s)), gl.fma(hi, c, lo * s)
    return lo, hi


@gluon.jit
def head_stage(
    mb,
    task,
    seq_len,
    slot,
    k_cache,
    v_cache,
    cos_sin,
    proj,
    ROPE: gl.constexpr,
    OWN: gl.constexpr,
    STAGE: gl.constexpr,
):
    """Head task ``task``: k head ``task % 2`` of row ``task // 2`` to the KV
    cache. The projection comes from the launch's qkvg tasks (``OWN``), or
    from ``proj`` (int32 ``[rows, PROJ_WORDS]``: the previous launch's
    qnext tasks stored it).

    Args:
        seq_len: int32 ``[rows]``: ``L + 1`` for the row's token at position
            ``L``.
        slot: int32 ``[rows]``: KV cache slot of the row's token.
        k_cache, v_cache: BF16 or FP8 e4m3 (unit scale) ``[slots, 2, 64]``.
        cos_sin: FP32 ``[positions, 64]``: ``cos`` of the 32 pairs, then ``sin``
            (unused when ``ROPE`` is false).
    """
    kv_dtype: gl.constexpr = k_cache.dtype.element_ty
    mb.stamp(STAGE, trace.START)
    row, g, _ = schedule.row_task(mb.plan, task, HEAD_TASKS)
    mb = mb.row(row)
    pl: gl.constexpr = _pair_layout(2, gl.num_warps())
    j = gl.arange(0, HEAD // 2, gl.SliceLayout(1, pl))
    e = gl.arange(0, 4, gl.SliceLayout(0, pl))[None, :]
    pos = gl.load(seq_len + row) - 1
    slot_at = gl.load(slot + row).to(gl.int64) * KV_STRIDE + 2 * j
    k_at = _pair_index(j[:, None], e, GATE_ROW, K_ROW + g * HEAD)
    v_at = _pair_index(j[:, None], e, GATE_ROW, V_ROW + g * HEAD)
    if OWN:
        kw, vw = mb.poll_many(("qkvg", "qkvg"), (k_at, v_at))
    else:
        kw = gl.load(proj + row * PROJ_WORDS + k_at)
        vw = gl.load(proj + row * PROJ_WORDS + v_at)
    mb.stamp(STAGE, trace.READY)
    lo, hi = _pair_values(kw, e, 1)
    lo, hi = _norm_rope(lo, hi, 1.0, cos_sin, pos, j, ROPE)
    # The cache keeps the rotated k and the raw v, each rounded once.
    at = slot_at + g * HEAD
    gl.store(k_cache + at, lo.to(kv_dtype))
    gl.store(k_cache + at + 1, hi.to(kv_dtype))
    vlo, vhi = _pair_values(vw, e, 1)
    gl.store(v_cache + at, vlo.to(kv_dtype))
    gl.store(v_cache + at + 1, vhi.to(kv_dtype))
    mb.stamp(STAGE, trace.END)


@gluon.constexpr_function
def _layouts(nw, QM):
    """(qk MFMA, pv MFMA, K operand, V load) layouts for ``QM`` MFMA rows.

    With 32 rows or more (3+ token rows) the warps split the rows in two and
    the chunk's keys in ``nw / 2``: the softmax's row max and sum then cross
    half the warps, for keys loaded by two warps each. Splitting the rows in
    four loses more to the duplicated key loads than it gains. One
    or two token rows are 16 MFMA rows, a single row tile."""
    rw = 2 if QM >= 32 else 1
    qk_mma = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[rw, nw // rw],
    )
    pv_mma = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 16],
        transposed=True,
        warps_per_cta=[rw, nw // rw],
    )
    k_layout = gl.DotOperandLayout(1, qk_mma, 8)
    v_load = gl.BlockedLayout([1, 8], [8, 8], [nw, 1], [1, 0])
    return qk_mma, pv_mma, k_layout, v_load


@gluon.constexpr_function
def _copy_layout(nw, bits):
    """Rows of a ``[BN, 64]`` chunk of K or V for the direct-to-LDS copies:
    16 bytes a thread, the widest copy gfx950 has (it has no 8-byte one,
    which 8 FP8 elements would need)."""
    per = 128 // bits
    return gl.BlockedLayout(
        [1, per], [64 // (HEAD // per), HEAD // per], [nw, 1], [1, 0]
    )


@gluon.jit
def _chunk_keys(c0, hi, v_load: gl.constexpr):
    """Key positions of chunk ``c0 ..``, one per row of ``v_load``. Positions
    from ``hi`` repeat the split's last key (or key 0): an async copy leaves
    masked lanes' LDS unwritten, and stale bits could be NaN in V, where
    ``0 * NaN`` would poison the sum. Their scores are masked to -inf."""
    j = c0 + gl.arange(0, BN, gl.SliceLayout(1, v_load))
    return gl.maximum(gl.minimum(j, hi - 1), 0)


@gluon.jit
def _load_pages(table, c0, hi, v_load: gl.constexpr):
    """The pages of chunk ``c0 ..``'s keys.

    A wave's loads, the copies to LDS included, retire in issue order
    (``vmcnt``): reading a page loaded after some chunk's copies would wait
    for those copies too. So the pages load before the copies of the chunk
    ahead of the one they locate, and a chunk's copies wait only for older
    loads. A plain load: a volatile one waited for every outstanding load
    (``vmcnt(0)``) right after it."""
    return gl.load(table + _chunk_keys(c0, hi, v_load) // PAGE)


@gluon.jit
def _issue_kv(
    k_ring, v_ring, slot, page, k_cache, v_cache, g, c0, hi, dv, v_load: gl.constexpr
):
    """Copy chunk ``c0 ..``'s keys and values ``[BN, 64]`` (the cache's dtype)
    at ``page`` (:func:`_load_pages`) into ring slot ``slot``, as one commit
    group, straight to LDS: the compiler's waits for register loads do not
    drain it, so the ring keeps two chunks in flight across the loop."""
    # A chunk past the split's end copies the split's last key again (its
    # scores are masked) instead of skipping its copies: a copy under a branch
    # would make the compiler wait for every outstanding load after it.
    j = _chunk_keys(c0, hi, v_load)
    row = (page.to(gl.int64) * PAGE + j % PAGE) * KV_STRIDE + g * HEAD
    at = row[:, None] + dv[None, :]
    cdna4_async_copy.global_load_to_shared(k_ring.index(slot), k_cache + at)
    cdna4_async_copy.global_load_to_shared(v_ring.index(slot), v_cache + at)
    cdna4_async_copy.commit_group()


@gluon.jit
def softmax_step(
    q, keys, values, valid, m, l, acc, p_layout: gl.constexpr, scale: gl.constexpr
):
    """One chunk of the online base-2 softmax: ``keys [D, N]`` and
    ``values [N, D]`` (an operand layout and LDS), where ``valid [QM, N]``."""
    pv_mma: gl.constexpr = acc.type.layout
    scores = gl.zeros([q.shape[0], keys.shape[1]], gl.float32, valid.type.layout)
    scores = gl.amd.cdna4.mfma(q, keys, scores)
    scores = gl.where(valid, scores * scale, float("-inf"))
    m_new = gl.maximum(m, gl.max(scores, axis=1))
    m_safe = gl.where(m_new == float("-inf"), 0.0, m_new)
    alpha = gl.exp2(m - m_safe)
    p = gl.exp2(scores - m_safe[:, None])
    l = l * alpha + gl.sum(p, axis=1)
    pv = gl.zeros(acc.shape, gl.float32, pv_mma)
    pv = gl.amd.cdna4.mfma(gl.convert_layout(p.to(gl.bfloat16), p_layout), values, pv)
    acc = acc * gl.convert_layout(alpha, gl.SliceLayout(1, pv_mma))[:, None] + pv
    return m_new, l, acc


@gluon.constexpr_function
def _group_layouts(nw):
    """(qk MFMA, pv MFMA) layouts of ``[key group, row, ...]`` tensors for
    :func:`_group_keys`: two key groups (the batch dimension) of ``nw / 2``
    warps, each warp 16 MFMA rows, so a warp's softmax max, sum and
    probabilities stay in the warp: no LDS round trip or barrier a chunk."""
    qk3 = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[2, nw // 2, 1],
    )
    pv3 = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 16],
        transposed=True,
        warps_per_cta=[2, nw // 2, 1],
    )
    return qk3, pv3


@gluon.jit
def _key_quarters(x):
    """``x [G, QM, 64]`` as its four ``[G, QM, 16]`` key quarters, in key
    order. A quarter is one K step of the PV MFMA; the 4 quarters sit in
    each thread's registers, so this only renames registers."""
    G: gl.constexpr = x.shape[0]
    QM: gl.constexpr = x.shape[1]
    y = gl.reshape(
        gl.permute(gl.reshape(x, [G, QM, 4, 16]), [0, 1, 3, 2]), [G, QM, 16, 2, 2]
    )
    even, odd = gl.split(y)  # quarter % 2
    q0, q2 = gl.split(even)  # quarter // 2
    q1, q3 = gl.split(odd)
    return q0, q1, q2, q3


@gluon.jit
def _group_qk(q_w, k_slot, qk3: gl.constexpr, k3: gl.constexpr, FP8QK: gl.constexpr):
    """The QK products ``[G, QM, KW]`` of every key group's own keys: ``q_w``
    times the chunk's keys in LDS ``k_slot [BN, 64]``."""
    G: gl.constexpr = q_w.shape[0]
    QM: gl.constexpr = q_w.shape[1]
    KW: gl.constexpr = BN // G
    keys = k_slot.reshape([G, KW, HEAD]).permute([0, 2, 1]).load(k3)
    if not FP8QK:
        keys = keys.to(gl.bfloat16)  # a BF16 cache's keys as they are
    scores = gl.zeros([G, QM, KW], gl.float32, qk3)
    return gl.amd.cdna4.mfma(q_w, keys, scores)


@gluon.jit
def _group_softmax(scores, valid, m, l, MASK: gl.constexpr):
    """The online softmax of every key group's own keys from their QK
    products ``scores [G, QM, KW]`` (``valid [G, QM, KW]`` used only when
    ``MASK``): the new running max and sum, the probabilities ``p`` and the
    running output's scale ``alpha``. :func:`_group_pv` finishes the step."""
    if MASK:
        scores = gl.where(valid, scores * QK_SCALE, float("-inf"))
    else:
        scores = scores * QK_SCALE
    m_new = gl.maximum(m, gl.max(scores, axis=2))
    m_safe = gl.where(m_new == float("-inf"), 0.0, m_new)
    alpha = gl.exp2(m - m_safe)
    p = gl.exp2(scores - m_safe[:, :, None])
    l = l * alpha + gl.sum(p, axis=2)
    return m_new, l, p, alpha


@gluon.jit
def _group_pv(p, alpha, v_slot, acc, p_layout: gl.constexpr):
    """``acc alpha + p V`` for the values in LDS ``v_slot [G, KW, 64]`` (the
    cache's dtype).

    The values load a K step (16 keys) at a time, each right before its
    MFMA, instead of the whole chunk at once: a quarter of the registers
    hold them. The 4 MFMAs accumulate in key order, as one K = 64 MFMA
    does."""
    pv3: gl.constexpr = acc.type.layout
    v3: gl.constexpr = gl.DotOperandLayout(1, pv3, 4)
    pv = gl.zeros(acc.shape, gl.float32, pv3)
    quarters = _key_quarters(p.to(gl.bfloat16))
    for t in gl.static_range(4):
        values = v_slot.slice(16 * t, 16, dim=1).load(v3).to(gl.bfloat16)
        pv = gl.amd.cdna4.mfma(gl.convert_layout(quarters[t], p_layout), values, pv)
    return acc * gl.convert_layout(alpha, gl.SliceLayout(2, pv3))[:, :, None] + pv


@gluon.constexpr_function
def _k_ring_layout(bits):
    """The K ring's layout. A lane reads 8 dims of one key for the QK MFMA,
    so with 64-byte FP8 rows the lanes of keys ``j``, ``j + 4``, ``j + 8``,
    ``j + 12`` hit the same LDS banks (4-way conflicts). The FP8 rows' 16-byte blocks are
    XOR-swizzled by ``(key // 4) % 4``; the copies place them so. A BF16 row
    is 128 bytes and keeps the plain layout (its whole-page copies)."""
    if bits == 8:
        return gl.SwizzledSharedLayout(16, 4, 4, [1, 0])
    return gl.SwizzledSharedLayout(1, 1, 1, [1, 0])


@gluon.constexpr_function
def _page_copy_layout(nw):
    """A ``[PAGE, 64]`` BF16 tile of one page: 16 bytes a thread."""
    return gl.BlockedLayout([1, 8], [64 // nw, 8], [nw, 1], [1, 0])


@gluon.jit
def _page_of(table, c, hi):
    """The page-table entry of key ``c``, clamped to the split's last key
    (or key 0)."""
    return gl.load(table + gl.maximum(gl.minimum(c, hi - 1), 0) // PAGE)


@gluon.jit
def _issue_page(ring, cache, page, c, hi, g, offs, half: gl.constexpr, slot):
    """Copy the page of keys ``c .. c + 64`` (entry ``page``) into half
    ``half`` of ring slot ``slot``, through a descriptor at the page: the
    page's address stays in SGPRs and ``offs`` is the same for every page.
    The descriptor ends at key ``hi``, so later keys read as zero (their
    scores are masked) at no cost."""
    # The clamp compiles to a VALU v_med3: back to an SGPR, or every copy
    # would loop over the lanes' descriptors.
    n = wave_uniform_i32(gl.minimum(gl.maximum(hi - c, 0), PAGE))
    at = cache + wave_uniform_i32(page).to(gl.int64) * (PAGE * KV_STRIDE)
    ELEM_BYTES: gl.constexpr = cache.dtype.element_ty.primitive_bitwidth // 8
    desc = cdna4.buffer.make_buffer_descriptor(at, n * (KV_STRIDE * ELEM_BYTES))
    cdna4.buffer.load_to_shared(
        ring.index(slot).slice(half * PAGE, PAGE), desc, offs, soffset=g * HEAD
    )


@gluon.jit
def _issue_chunk(k_ring, v_ring, slot, pa, pb, k_cache, v_cache, g, c0, hi, offs):
    """Chunk ``c0 ..``'s two pages (entries ``pa``, ``pb``) of K and V into
    ring slot ``slot``, as one commit group."""
    _issue_page(k_ring, k_cache, pa, c0, hi, g, offs, 0, slot)
    _issue_page(v_ring, v_cache, pa, c0, hi, g, offs, 0, slot)
    _issue_page(k_ring, k_cache, pb, c0 + PAGE, hi, g, offs, 1, slot)
    _issue_page(v_ring, v_cache, pb, c0 + PAGE, hi, g, offs, 1, slot)
    cdna4_async_copy.commit_group()


@gluon.jit
def _group_keys(
    q,
    q8,
    k_ring,
    v_ring,
    table,
    k_cache,
    v_cache,
    g,
    lo,
    hi,
    pa,
    pb,
    offs,
    pages,
    dv,
    cl: gl.constexpr,
    m,
    l,
    acc,
    PAGES: gl.constexpr,
):
    """A full layer's cached keys ``lo .. hi`` from ``LONG_KEYS`` (the ring
    holds chunks 0 and 1; ``pa``, ``pb`` (``PAGES``) or ``pages`` locate chunk
    2). Key group ``w`` takes keys ``KW w ..`` of each chunk with its own
    softmax state; the groups combine once at the end. A BF16 cache's chunks
    copy a page at a time (``PAGES``, whole-page splits); an FP8 page's ``[64,
    64]`` tile is half of what 8 warps copy at 16 bytes a thread, so FP8 keeps
    the per-key copies. Only the split's last chunk is masked. Returns ``m``,
    ``l`` and ``acc`` (in the layouts of the arguments) with the keys folded
    in."""
    nw: gl.constexpr = gl.num_warps()
    QM: gl.constexpr = q.shape[0]
    G: gl.constexpr = 2
    KW: gl.constexpr = BN // G
    qk3: gl.constexpr = _group_layouts(nw)[0]
    pv3: gl.constexpr = _group_layouts(nw)[1]
    FP8QK: gl.constexpr = k_ring.dtype.primitive_bitwidth == 8
    QKW: gl.constexpr = 16 if FP8QK else 8
    q3l: gl.constexpr = gl.DotOperandLayout(0, qk3, QKW)
    k3: gl.constexpr = gl.DotOperandLayout(1, qk3, QKW)
    p3: gl.constexpr = gl.DotOperandLayout(0, pv3, 4)
    # Every group's copy of q: one conversion, then a broadcast in registers.
    if FP8QK:
        # q rounded once from FP32 to the cache's FP8 (as the regular path's
        # prologue does for an FP8 cache), for FP8 x FP8 products.
        q_w = gl.convert_layout(q8, gl.SliceLayout(0, q3l))
        q_w, _ = gl.broadcast(
            q_w[None, :, :], gl.zeros([G, QM, HEAD], k_ring.dtype, q3l)
        )
    else:
        q_w = gl.convert_layout(q, gl.SliceLayout(0, q3l))
        q_w = q_w[None, :, :] + gl.zeros([G, QM, HEAD], gl.bfloat16, q3l)
    ww = gl.arange(0, G, gl.SliceLayout(1, gl.SliceLayout(2, qk3)))[:, None, None]
    jw = gl.arange(0, KW, gl.SliceLayout(0, gl.SliceLayout(1, qk3)))[None, None, :]
    m_w = gl.full([G, QM], float("-inf"), gl.float32, gl.SliceLayout(2, qk3))
    l_w = gl.zeros([G, QM], gl.float32, gl.SliceLayout(2, qk3))
    acc_w = gl.zeros([G, QM, HEAD], gl.float32, pv3)
    # Chunk 0's QK before the loop (chunks 0 and 1 are in flight). Iteration
    # i then runs chunk i's softmax, chunk i + 1's QK and chunk i's PV, so
    # chunk i + 1's QK MFMAs overlap chunk i's softmax-to-PV chain. Every
    # running state (max, sum, output) still takes the chunks in order.
    cdna4_async_copy.wait_group(1)
    gl.barrier()
    scores = _group_qk(q_w, k_ring.index(0), qk3, k3, FP8QK)
    i = 0
    for c0 in range(lo, hi, BN):
        valid = c0 + ww * KW + jw < hi  # [G, 1, KW]: every row attends every key
        # As attn_stage's loop: every wave is done with the slot the next
        # copy refills (chunk i - 1's: its keys were read for the QK in
        # iteration i - 2, its values by the PV in iteration i - 1), then
        # chunk i + 3's pages and chunk i + 2's copies.
        gl.barrier()
        if PAGES:
            na = _page_of(table, c0 + 3 * BN, hi)
            nb = _page_of(table, c0 + 3 * BN + PAGE, hi)
            _issue_chunk(
                k_ring,
                v_ring,
                (i + 2) % NBUF,
                pa,
                pb,
                k_cache,
                v_cache,
                g,
                c0 + 2 * BN,
                hi,
                offs,
            )
            pa = na
            pb = nb
        else:
            pages3 = _load_pages(table, c0 + 3 * BN, hi, cl)
            _issue_kv(
                k_ring,
                v_ring,
                (i + 2) % NBUF,
                pages,
                k_cache,
                v_cache,
                g,
                c0 + 2 * BN,
                hi,
                dv,
                cl,
            )
            pages = pages3
        # Only the split's last chunk has keys past its end.
        if c0 + BN > hi:
            m_w, l_w, p, alpha = _group_softmax(scores, valid, m_w, l_w, True)
        else:
            m_w, l_w, p, alpha = _group_softmax(scores, valid, m_w, l_w, False)
        if c0 + BN < hi:
            # Chunk i + 1 has landed once only chunk i + 2's copies are
            # outstanding; every wave sees every wave's copies after the
            # barrier. (lo and hi are uniform: all waves take this branch.)
            cdna4_async_copy.wait_group(1)
            gl.barrier()
            scores = _group_qk(q_w, k_ring.index((i + 1) % NBUF), qk3, k3, FP8QK)
        acc_w = _group_pv(
            p, alpha, v_ring.index(i % NBUF).reshape([G, KW, HEAD]), acc_w, p3
        )
        i += 1
    cdna4_async_copy.wait_group(0)
    # The groups' states into one: their LSE merge, once.
    m_max = gl.max(m_w, axis=0)
    m_safe = gl.where(m_max == float("-inf"), 0.0, m_max)
    scale = gl.exp2(m_w - m_safe[None, :])
    l_sum = gl.sum(l_w * scale, axis=0)
    acc_w = acc_w * gl.convert_layout(scale, gl.SliceLayout(2, pv3))[:, :, None]
    acc_sum = gl.sum(acc_w, axis=0)
    return (
        gl.convert_layout(m_max, m.type.layout),
        gl.convert_layout(l_sum, l.type.layout),
        gl.convert_layout(acc_sum, acc.type.layout),
    )


@gluon.jit
def _ring_keys(
    q,
    k_ring,
    v_ring,
    table,
    k_cache,
    v_cache,
    g,
    lo,
    hi,
    pages,
    dv,
    cl: gl.constexpr,
    row_first,
    m,
    l,
    acc,
    k_layout: gl.constexpr,
    v_layout: gl.constexpr,
    p_layout: gl.constexpr,
):
    """The cached keys ``lo .. hi`` in chunks of ``BN`` through the ring (it
    holds chunks 0 and 1, ``pages`` locates chunk 2), every warp on its share
    of each chunk. Returns ``m``, ``l`` and ``acc`` with the keys folded in."""
    qk_mma: gl.constexpr = row_first.type.layout.parent
    i = 0
    for c0 in range(lo, hi, BN):
        js = c0 + gl.arange(0, BN, gl.SliceLayout(0, qk_mma))
        valid = (js[None, :] < hi) & (js[None, :] >= row_first[:, None])
        # Every wave is done with the slot the next copy refills (chunk
        # i - 1's); then the pages of chunk i + 3 and chunk i + 2's copies
        # (at pages loaded before chunk i + 1's) go before this chunk's math.
        gl.barrier()
        pages3 = _load_pages(table, c0 + 3 * BN, hi, cl)
        _issue_kv(
            k_ring,
            v_ring,
            (i + 2) % NBUF,
            pages,
            k_cache,
            v_cache,
            g,
            c0 + 2 * BN,
            hi,
            dv,
            cl,
        )
        pages = pages3
        # Chunk i's group is complete once at most two are outstanding.
        cdna4_async_copy.wait_group(2)
        gl.barrier()
        keys = k_ring.index(i % NBUF).permute([1, 0]).load(k_layout)
        values = v_ring.index(i % NBUF).load(v_layout)
        m, l, acc = softmax_step(
            q,
            keys.to(gl.bfloat16),
            values.to(gl.bfloat16),
            valid,
            m,
            l,
            acc,
            p_layout,
            QK_SCALE,
        )
        i += 1
    # No copy may still write LDS when the launch moves on.
    cdna4_async_copy.wait_group(0)
    return m, l, acc


@gluon.jit
def attn_stage(
    mb,
    task,
    seq_len,
    table,
    table_stride,
    k_cache,
    v_cache,
    cos_sin,
    proj,
    ROPE: gl.constexpr,
    OWN: gl.constexpr,
    SPLITS: gl.constexpr,
    LAYER_WINDOW: gl.constexpr,
    STAGE: gl.constexpr,
):
    """Attn task ``task`` = (request ``task // (2 SPLITS)``, KV head, split)
    of every row of the request. The projection comes as head_stage takes
    it (``proj``, ``OWN``).

    Args:
        seq_len: int32 ``[rows]``, as head_stage takes it (the request's
            first row's is read).
        table: int32 page tables of the requests, one row each (``PAGE``
            tokens a page), ``table_stride`` apart.
        cos_sin, ROPE: As head_stage takes them.
        LAYER_WINDOW: Attended positions of a sliding layer (``WINDOW``), 0 on
            a full layer.
    """
    C: gl.constexpr = schedule.chain(mb.plan)
    gl.static_assert(C <= 16, "the launch's keys are one chunk of 16")
    # One MFMA row per (row, q head): rows past C * GROUP are zero.
    QM: gl.constexpr = max(16, triton.next_power_of_2(C * GROUP))
    KT: gl.constexpr = 16  # the launch's keys, rows past C zero
    mb.stamp(STAGE, trace.START)
    req = task // (KV_HEADS * SPLITS)
    g = task % (KV_HEADS * SPLITS) // SPLITS
    s = task % SPLITS
    # The request's rows and its page table row.
    base = req * C
    table = table + req.to(gl.int64) * table_stride
    nw: gl.constexpr = gl.num_warps()
    qk_mma: gl.constexpr = _layouts(nw, QM)[0]
    pv_mma: gl.constexpr = _layouts(nw, QM)[1]
    k_layout: gl.constexpr = _layouts(nw, QM)[2]
    q_layout: gl.constexpr = gl.DotOperandLayout(0, qk_mma, 8)
    p_layout: gl.constexpr = gl.DotOperandLayout(0, pv_mma, 4)
    v_layout: gl.constexpr = gl.DotOperandLayout(1, pv_mma, 4)

    L0 = gl.load(seq_len + base) - 1
    # The cached keys of the splits: from row 0's first key, before L0.
    if LAYER_WINDOW == 0:
        first = 0
    else:
        first = gl.maximum(L0 - LAYER_WINDOW + 1, 0)
    per = (L0 - first + SPLITS - 1) // SPLITS
    # Key groups (_group_keys): 16 MFMA rows a warp, so 64 rows (8 token
    # rows) at 8 warps, on a full layer.
    GROUPED: gl.constexpr = (LAYER_WINDOW == 0) and (QM == 16 * (nw // 2))
    # A BF16 cache's chunks copy a page at a time (whole-page splits).
    PAGES: gl.constexpr = k_cache.dtype.element_ty.primitive_bitwidth == 16
    if GROUPED:
        # A long context takes the key groups (a BF16 cache with whole pages
        # a split, so a chunk is two whole pages). Uniform (an SGPR), so the
        # paths branch on a scalar.
        long = wave_uniform_i32((L0 >= LONG_KEYS).to(gl.int32)) != 0
        if PAGES:
            per = gl.where(long, (per + PAGE - 1) // PAGE * PAGE, per)
    lo = first + s * per
    hi = gl.minimum(lo + per, L0)
    if GROUPED:
        # In SGPRs: the copies' descriptors derive from them.
        lo = wave_uniform_i32(lo)
        hi = wave_uniform_i32(hi)

    # The first two chunks' copies go before the poll: they do not depend on
    # this launch, and they arrive while the projections run.
    kv_dtype: gl.constexpr = k_cache.dtype.element_ty
    plain: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    k_ring = gl.allocate_shared_memory(
        kv_dtype, [NBUF, BN, HEAD], _k_ring_layout(kv_dtype.primitive_bitwidth)
    )
    v_ring = gl.allocate_shared_memory(kv_dtype, [NBUF, BN, HEAD], plain)
    cl: gl.constexpr = _copy_layout(nw, kv_dtype.primitive_bitwidth)
    dv = gl.arange(0, HEAD, gl.SliceLayout(0, cl))
    if GROUPED:
        pcl: gl.constexpr = _page_copy_layout(nw)
        offs = (
            gl.arange(0, PAGE, gl.SliceLayout(1, pcl))[:, None] * KV_STRIDE
            + gl.arange(0, HEAD, gl.SliceLayout(0, pcl))[None, :]
        )
        pa = hi * 0
        pb = hi * 0
        pages = gl.zeros([BN], gl.int32, gl.SliceLayout(1, cl))
        if PAGES:
            if long:
                p0a = _page_of(table, lo, hi)
                p0b = _page_of(table, lo + PAGE, hi)
                p1a = _page_of(table, lo + BN, hi)
                p1b = _page_of(table, lo + BN + PAGE, hi)
                pa = _page_of(table, lo + 2 * BN, hi)
                pb = _page_of(table, lo + 2 * BN + PAGE, hi)
                _issue_chunk(
                    k_ring, v_ring, 0, p0a, p0b, k_cache, v_cache, g, lo, hi, offs
                )
                _issue_chunk(
                    k_ring, v_ring, 1, p1a, p1b, k_cache, v_cache, g, lo + BN, hi, offs
                )
            else:
                page0 = _load_pages(table, lo, hi, cl)
                page1 = _load_pages(table, lo + BN, hi, cl)
                pages = _load_pages(table, lo + 2 * BN, hi, cl)
                _issue_kv(k_ring, v_ring, 0, page0, k_cache, v_cache, g, lo, hi, dv, cl)
                _issue_kv(
                    k_ring, v_ring, 1, page1, k_cache, v_cache, g, lo + BN, hi, dv, cl
                )
        else:
            page0 = _load_pages(table, lo, hi, cl)
            page1 = _load_pages(table, lo + BN, hi, cl)
            pages = _load_pages(table, lo + 2 * BN, hi, cl)
            _issue_kv(k_ring, v_ring, 0, page0, k_cache, v_cache, g, lo, hi, dv, cl)
            _issue_kv(
                k_ring, v_ring, 1, page1, k_cache, v_cache, g, lo + BN, hi, dv, cl
            )
    else:
        page0 = _load_pages(table, lo, hi, cl)
        page1 = _load_pages(table, lo + BN, hi, cl)
        pages = _load_pages(table, lo + 2 * BN, hi, cl)
        _issue_kv(k_ring, v_ring, 0, page0, k_cache, v_cache, g, lo, hi, dv, cl)
        _issue_kv(k_ring, v_ring, 1, page1, k_cache, v_cache, g, lo + BN, hi, dv, cl)

    # q of the 8 heads of every row, and the launch's k and v. A thread
    # holds one pair, 32 lanes a head: the norm sums in head_stage's order.
    q3: gl.constexpr = _pair_layout(3, nw)
    ql: gl.constexpr = gl.SliceLayout(2, q3)
    qr = gl.arange(0, QM, gl.SliceLayout(1, ql))[:, None]
    qp = gl.arange(0, HEAD // 2, gl.SliceLayout(0, ql))[None, :]
    eq = gl.arange(0, 4, gl.SliceLayout(0, gl.SliceLayout(1, q3)))[None, None, :]
    q_tok = gl.minimum(qr // GROUP, C - 1)
    q_head = qr % GROUP
    kr = gl.arange(0, KT, gl.SliceLayout(1, ql))[:, None]
    k_tok = gl.minimum(kr, C - 1)
    if ROPE:
        # The RoPE rows load before the poll: a load after a poll would
        # start only when the poll returns.
        # Row L0's offset grows with the position (int64); the launch's
        # rows past it are in-tile offsets.
        rope = cos_sin + L0.to(gl.int64) * HEAD
        cq = gl.load(rope + q_tok * HEAD + qp)
        sq = gl.load(rope + q_tok * HEAD + HEAD // 2 + qp)
        ck = gl.load(rope + k_tok * HEAD + qp)
        sk = gl.load(rope + k_tok * HEAD + HEAD // 2 + qp)
    q_at = _pair_index(
        qp[:, :, None], eq, GATE_ROW, ((g * GROUP + q_head) * HEAD)[:, :, None]
    )
    k_at = _pair_index(qp[:, :, None], eq, GATE_ROW, K_ROW + g * HEAD)
    v_at = _pair_index(qp[:, :, None], eq, GATE_ROW, V_ROW + g * HEAD)
    if OWN:
        got_q, got_k, got_v = mb.poll_many(
            ("qkvg", "qkvg", "qkvg"),
            (
                mb.row_offset(base + q_tok)[:, :, None] + q_at,
                mb.row_offset(base + k_tok)[:, :, None] + k_at,
                mb.row_offset(base + k_tok)[:, :, None] + v_at,
            ),
        )
    else:
        got_q = gl.load(proj + ((base + q_tok) * PROJ_WORDS)[:, :, None] + q_at)
        got_k = gl.load(proj + ((base + k_tok) * PROJ_WORDS)[:, :, None] + k_at)
        got_v = gl.load(proj + ((base + k_tok) * PROJ_WORDS)[:, :, None] + v_at)
    mb.stamp(STAGE, trace.READY)
    q_lo, q_hi = _pair_values(got_q, eq, 2)
    inv_q = gl.rsqrt(gl.sum(q_lo * q_lo + q_hi * q_hi, axis=1) / HEAD + EPS)[:, None]
    q_lo = q_lo * inv_q * Q_GAIN
    q_hi = q_hi * inv_q * Q_GAIN
    k_lo, k_hi = _pair_values(got_k, eq, 2)
    inv_k = gl.rsqrt(gl.sum(k_lo * k_lo + k_hi * k_hi, axis=1) / HEAD + EPS)[:, None]
    k_lo = k_lo * inv_k
    k_hi = k_hi * inv_k
    if ROPE:
        q_lo, q_hi = gl.fma(q_lo, cq, -(q_hi * sq)), gl.fma(q_hi, cq, q_lo * sq)
        k_lo, k_hi = gl.fma(k_lo, ck, -(k_hi * sk)), gl.fma(k_hi, ck, k_lo * sk)
    q = gl.reshape(gl.join(q_lo, q_hi), [QM, HEAD])
    q = gl.where(
        gl.arange(0, QM, gl.SliceLayout(1, q.type.layout))[:, None] < C * GROUP,
        q,
        0.0,
    )
    q8 = q.to(kv_dtype)  # rounded once from FP32 (an FP8 cache's long path)
    q = gl.convert_layout(q.to(gl.bfloat16), q_layout)
    v_lo, v_hi = _pair_values(got_v, eq, 2)
    # The launch's k and v as the cache holds them: rounded once to its dtype.
    cur_k = gl.allocate_shared_memory(gl.bfloat16, [KT, HEAD], plain)
    cur_v = gl.allocate_shared_memory(gl.bfloat16, [KT, HEAD], plain)
    cur_k.store(
        gl.reshape(gl.join(k_lo, k_hi), [KT, HEAD]).to(kv_dtype).to(gl.bfloat16)
    )
    cur_v.store(
        gl.reshape(gl.join(v_lo, v_hi), [KT, HEAD]).to(kv_dtype).to(gl.bfloat16)
    )

    # Row t's first attended key (a sliding layer's window).
    qi = gl.arange(0, QM, gl.SliceLayout(1, qk_mma))
    qi_tok = gl.minimum(qi // GROUP, C - 1)
    if LAYER_WINDOW == 0:
        row_first = gl.zeros_like(qi_tok)
    else:
        row_first = L0 + qi_tok - LAYER_WINDOW + 1
    m = gl.full([QM], float("-inf"), gl.float32, gl.SliceLayout(1, qk_mma))
    l = gl.zeros([QM], gl.float32, gl.SliceLayout(1, qk_mma))
    acc = gl.zeros([QM, HEAD], gl.float32, pv_mma)
    if GROUPED:
        if long:
            m, l, acc = _group_keys(
                q,
                q8,
                k_ring,
                v_ring,
                table,
                k_cache,
                v_cache,
                g,
                lo,
                hi,
                pa,
                pb,
                offs,
                pages,
                dv,
                cl,
                m,
                l,
                acc,
                PAGES,
            )
        else:
            m, l, acc = _ring_keys(
                q,
                k_ring,
                v_ring,
                table,
                k_cache,
                v_cache,
                g,
                lo,
                hi,
                pages,
                dv,
                cl,
                row_first,
                m,
                l,
                acc,
                k_layout,
                v_layout,
                p_layout,
            )
    else:
        m, l, acc = _ring_keys(
            q,
            k_ring,
            v_ring,
            table,
            k_cache,
            v_cache,
            g,
            lo,
            hi,
            pages,
            dv,
            cl,
            row_first,
            m,
            l,
            acc,
            k_layout,
            v_layout,
            p_layout,
        )
    # The launch's tokens: token u is the key at L0 + u; row t attends u <= t.
    gl.barrier()
    u = gl.arange(0, KT, gl.SliceLayout(0, qk_mma))
    own = (u[None, :] <= qi_tok[:, None]) & (u[None, :] < C) & (s == SPLITS - 1)
    m, l, acc = softmax_step(
        q,
        cur_k.permute([1, 0]).load(gl.DotOperandLayout(1, qk_mma, 8)),
        cur_v.load(v_layout),
        own,
        m,
        l,
        acc,
        p_layout,
        QK_SCALE,
    )
    mb.stamp(STAGE, trace.COMPUTED)
    l_acc = gl.convert_layout(l, gl.SliceLayout(1, pv_mma))
    out = acc / gl.where(l_acc > 0, l_acc, 1.0)[:, None]
    lse = gl.where(l > 0, m + gl.log2(gl.where(l > 0, l, 1.0)), float("-inf"))
    r = gl.arange(0, QM, gl.SliceLayout(1, pv_mma))
    c = gl.arange(0, HEAD, gl.SliceLayout(0, pv_mma))
    part = ((s * KV_HEADS + g) * GROUP + r[:, None] % GROUP) * HEAD + c[None, :]
    # Plain words (row r's copy starts at word 2 row_offset(r)), then the flag.
    mb.store_words(
        "part",
        2 * mb.row_offset(base + r[:, None] // GROUP) + part,
        out,
        # (c >= 0) is always true: it broadcasts the row mask to the tile.
        mask=(r < C * GROUP)[:, None] & (c >= 0)[None, :],
    )
    mb.store_words(
        "lse",
        2 * mb.row_offset(base + qi // GROUP) + s * Q_HEADS + g * GROUP + qi % GROUP,
        lse,
        mask=qi < C * GROUP,
    )
    mb.raise_flag("pflag", mb.row_offset(base) + g * SPLITS + s)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def merge_stage(mb, task, SPLITS: gl.constexpr, STAGE: gl.constexpr):
    """Merge task ``task`` = q head ``h = task % 16`` of row ``task // 16``:
    the LSE merge, then the gated head norm of the regular path's output
    projection."""
    mb.stamp(STAGE, trace.START)
    row, h, tok = schedule.row_task(mb.plan, task, MERGE_TASKS)
    # The splits' flags sit in the request's first row (row - tok).
    flag0 = -tok * schedule.row_pairs(mb.plan)
    mb = mb.row(row)
    g = h // GROUP
    hh = h % GROUP
    nw: gl.constexpr = gl.num_warps()
    ml: gl.constexpr = gl.BlockedLayout([1, 1], [2, 32], [nw, 1], [1, 0])
    s = gl.arange(0, SPLITS, gl.SliceLayout(1, ml))
    c = gl.arange(0, HEAD, gl.SliceLayout(0, ml))
    part = ((s[:, None] * KV_HEADS + g) * GROUP + hh) * HEAD + c[None, :]
    gl2: gl.constexpr = gl.BlockedLayout([1, 2], [64, 1], [nw, 1], [1, 0])
    gc = gl.arange(0, HEAD, gl.SliceLayout(1, gl2))[:, None]
    gh = gl.arange(0, 2, gl.SliceLayout(0, gl2))[None, :]
    # One tagged poll for the flags and the gate words: two polls in a row
    # would add a round trip on the merge's critical path.
    _, got_g = mb.poll_many(
        ("pflag", "gate"),
        (flag0 + g * SPLITS + s, gh * (QKVG - GATE_ROW) + h * HEAD + gc),
    )
    # Every split's flag is in: the words load after the poll returned.
    gl.barrier()
    got_p = mb.load_words("part", part)
    got_l = mb.load_words("lse", s * Q_HEADS + h)
    mb.stamp(STAGE, trace.READY)
    lse = got_l.to(gl.float32, bitcast=True)
    top = gl.max(lse, axis=0)
    w = gl.where(lse == float("-inf"), 0.0, gl.exp2(lse - top))
    p = got_p.to(gl.float32, bitcast=True)
    out = gl.sum(p * w[:, None], axis=0) / gl.sum(w, axis=0)
    a = out.to(gl.bfloat16).to(gl.float32)
    inv = gl.rsqrt(gl.sum(a * a, axis=0) / HEAD + EPS)
    gate = gl.sum(got_g.to(gl.float32, bitcast=True), axis=1)
    gate = gl.convert_layout(gate.to(gl.bfloat16).to(gl.float32), c.type.layout)
    z = a * inv * mx.sigmoid(gate)
    mb.stamp(STAGE, trace.COMPUTED)
    mx.put_bf16(mb, "z", h * (HEAD // 2), z)
    mb.stamp(STAGE, trace.END)
