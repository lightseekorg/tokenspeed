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

"""Windowed relative attention of a local layer and its split merge with the
output gate and sink (TokenSpeed ``rel_mha_decode_with_kvcache`` (Triton
split-KV decode) and ``rel_mha_gate_sink``).

The row of a token at position ``p`` attends the keys ``i`` of its request at
positions ``start .. p``, ``start = L - min(L, 1024)``, ``L = p + 1``. Over
the ``n = L - start`` keys, split ``s`` of 8 takes offsets
``s * per .. (s + 1) * per`` with ``per = 32 ceil(ceil(n / 8) / 32)``, at
most 128. For head ``h``:

    score_i = (q_h . k_i) / sqrt(128) + rel[h, L - 1 - i]
    O_s = sum_i bf16(exp(score_i - m)) v_i / sum_i exp(score_i - m)
    lse_s = m + log(sum_i exp(score_i - m))

The merge combines the used splits in LSE and gates the result:
``o = bf16((O sigmoid(gate)) sigmoid(lse - sink))``.

The heads are the plan's (``Dims``): ``Hq`` query heads, ``Hkv`` KV heads
and ``group = Hq / Hkv`` query heads a KV head (a rank of Inkling 2 Small
at TP4: 12, 2 and 6). Both stages equal the served regular path bit for
bit (the Triton ``_rel_mha_decode_grouped_stage1_kernel`` and
``_rel_mha_decode_stage2_kernel`` at a 128-token KV page, then
``_gate_sink_kernel``): a split is one of that kernel's KV blocks, a KV
head's query heads the first rows of its head block (:func:`head_block`:
8 rows at a group of 6, 4 at 3), its two products are the same MFMA 4x4x4
chains (16 blocks along N, ``K`` in ascending steps of 4), its row sum has
the same layout (which follows the head block), the merge runs online in
split order, and the sigmoids use libdevice ``exp``. The regular kernel's
KV block is the page: at another page size its blocks and row sums change,
and the two agree only to the last bits.

An attention task computes one split of a KV head of a row (``8 Hkv`` tasks
a row, in rounds past the grid). A CTA's tasks stream their cached keys and
values through two LDS tiles with direct-to-LDS copies: the copies of a
task issue before the previous task computes, and the first task's before
its polls. A task writes the tile rows that are not copied: zeros past the
split, the mailbox rows of the launch's own tokens. A merge task merges the
query heads of a KV head, a wave each.

The keys and values of the request's tokens in this launch come from their
rows' mailboxes (the head stage's ``kv``), not from the cache: another CTA's
cache write in this launch need not be visible yet.

A full attention layer (:func:`full_attention_stage`, :func:`full_merge_stage`)
attends every key ``0 .. p`` over the regular decode's split count at the
served context length (32), ``per = 32 ceil(ceil(L / 32) / 32)`` keys a split.
A split is one or more KV blocks of 128 keys from its first key, combined as
the regular kernel's loop does: ``m' = max(m, max score)``, ``acc' = acc
exp(m - m') + bf16(exp(score - m')) v`` (the value MFMA chain onto the
rescaled sum) and ``s' = fma(s, exp(m - m'), sum exp(score - m'))``. The
relative logits cover distances below 1024; farther keys add no bias. An
unused split publishes nothing.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import aggregate
from tokenspeed_kernel_amd._triton import cdna4_async_copy as async_copy
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mx, schedule, trace

# Every Inkling2 geometry's; the head counts, the head block and the
# projection row's offsets are the plan's (schedule.dims, head_block).
DIM = gl.constexpr(128)
WINDOW = gl.constexpr(1024)
SPLITS = gl.constexpr(8)
BN = gl.constexpr(128)  # keys of a full split
SM_SCALE = gl.constexpr(0.08838834764831845)  # 128 ** -0.5


@gluon.constexpr_function
def head_block(plan):
    """Rows of the regular grouped decode kernel's head block for a KV
    head's query heads: ``BLOCK_H = min(8, next_power_of_2(group))``
    (``_rel_mha_decode_fwd``), the group's heads in its first rows, the
    others masked. One block a KV head: a group of up to 8 heads."""
    group = plan.plan.dims.group
    if group > 8:
        raise ValueError(f"a KV head's {group} query heads span several head blocks")
    rows = 1
    while rows < group:
        rows *= 2
    return rows


@gluon.constexpr_function
def merge_tasks(plan):
    """Merge tasks a row: one a KV head (its query heads), each raising its
    flag of the row's ``o`` (the output projection waits for all)."""
    return plan.plan.dims.kv_heads


@aggregate
class Cache:
    """The layer's KV cache and the launch's per-row positions."""

    k_cache: gl.tensor  # BF16 [slots, KV heads, DIM]
    v_cache: gl.tensor
    positions: gl.tensor  # int64 [rows]
    table: gl.tensor  # int32 [requests, table_stride] KV page table
    table_stride: gl.tensor | gl.constexpr
    page_size: gl.tensor | gl.constexpr


@gluon.jit
def _split(L, s):
    """``(start, first, end)``: the row's window start and the key offsets
    of split ``s``."""
    n = gl.minimum(L, WINDOW)
    per = ((n + SPLITS - 1) // SPLITS + 31) // 32 * 32
    first = s * per
    end = gl.minimum(first + per, n)
    return L - n, first, end


@gluon.constexpr_function
def _layouts(nw, hb):
    """The regular kernel's layouts for its ``hb``-row head block: both
    ``tl.dot`` products as MFMA 4x4x4 with 16 blocks along N (A broadcast
    from one block per instruction, ``K`` in ascending steps of 4), their
    operands from LDS tiles as it stages them, and the row sum of the
    probabilities in its blocked layout of ``[hb, 128]`` over its 4 warps:
    ``hb / 2`` adjacent columns a thread, then the row's lanes (4 columns
    and 32 lanes at 8 rows, 2 columns and 64 lanes at 4). Warps only
    replicate output tiles, which leaves each element's chain as it is."""
    if hb not in (4, 8):
        raise ValueError(f"no regular decode row sum layout of a {hb}-row head block")
    columns = hb // 2
    lanes = 128 // columns  # of a 128-key block
    mma = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[4, 64, 64],
        transposed=False,
        warps_per_cta=[2, nw // 2],
    )
    return (
        mma,
        # [keys, DIM] K and V tiles: 16-byte rows of a key, coalesced.
        gl.BlockedLayout([1, 8], [4, 16], [nw, 1], [1, 0]),
        gl.SwizzledSharedLayout(4, 1, 4, [1, 0]),
        gl.BlockedLayout([1, columns], [64 // lanes, lanes], [nw, 1], [1, 0]),
    )


@gluon.jit
def _slots(cache, request, position, ok):
    page = gl.load(
        cache.table
        + request * cache.table_stride
        + gl.where(ok, position, 0) // cache.page_size,
        mask=ok,
        other=0,
    )
    return gl.where(
        ok, page.to(gl.int64) * cache.page_size + position % cache.page_size, 0
    )


@gluon.jit
def _task(cache, task, CHAIN: gl.constexpr, plan: gl.constexpr):
    """``(row, g, s, request, L, launch0, tok)`` of attention task ``task``
    (row, KV head ``g``, split ``s``): the row's length ``L`` and the
    position ``launch0`` of its request's first token in this launch.

    The split rotates with the row's block of a round's rows (16 at 2 KV
    heads on 256 CTAs): a CTA's tasks of successive rounds (a round's rows
    apart) take successive splits, so the window's last split, which writes
    the launch's own keys into its tile, falls on a CTA once, not in every
    round."""
    D: gl.constexpr = schedule.dims(plan)
    KV_HEADS: gl.constexpr = D.kv_heads
    TASKS: gl.constexpr = KV_HEADS * SPLITS  # a row's
    ROUND_ROWS: gl.constexpr = schedule.grid(plan) // TASKS
    row = task // TASKS
    g = (task % TASKS) // SPLITS
    s = (task + row // ROUND_ROWS) % SPLITS
    tok = row % CHAIN
    # Positions fit int32; the cache offsets that grow with them are int64.
    L = gl.load(cache.positions + row).to(gl.int32) + 1
    return row, g, s, row // CHAIN, L, L - 1 - tok, tok


@gluon.jit
def _issue_split(cache, k_buf, v_buf, task, CHAIN: gl.constexpr, plan: gl.constexpr):
    """Copy the cached keys and values of attention task ``task``'s split
    into ``k_buf`` and ``v_buf`` (``[BN, DIM]``, the tile rows at the
    split's key offsets) with direct-to-LDS copies, as one commit group.
    The rows of other keys (past the split, or the launch's own tokens)
    are not copied: :func:`_attention_task` writes them."""
    D: gl.constexpr = schedule.dims(plan)
    KV_HEADS: gl.constexpr = D.kv_heads
    HB: gl.constexpr = head_block(plan)
    _, g, s, request, L, launch0, _ = _task(cache, task, CHAIN, plan)
    start, first, end = _split(L, s)
    kv_load: gl.constexpr = _layouts(gl.num_warps(), HB)[1]
    pk = start + first + gl.arange(0, BN, gl.SliceLayout(1, kv_load))
    cached = (pk < start + end) & (pk < launch0)
    slot = _slots(cache, request, pk, cached)
    d = gl.arange(0, DIM, gl.SliceLayout(0, kv_load))
    at_kv = g * DIM + slot[:, None] * (KV_HEADS * DIM) + d[None, :]
    async_copy.global_load_to_shared(k_buf, cache.k_cache + at_kv, mask=cached[:, None])
    async_copy.global_load_to_shared(v_buf, cache.v_cache + at_kv, mask=cached[:, None])
    async_copy.commit_group()


@gluon.jit
def _key_rows(keys, L, s, launch0):
    """``(cached, own)`` of the tile rows ``keys`` (offsets in split ``s``
    of a row at length ``L``): whether the tile row is a cached key's, and
    the launch token whose key the row is (-1 for none)."""
    start, first, end = _split(L, s)
    pk = start + first + keys
    ok = pk < start + end
    return ok & (pk < launch0), gl.where(ok & (pk >= launch0), pk - launch0, -1)


@gluon.jit
def _rel_index(L, g, start, first, plan: gl.constexpr):
    """``[HB, BN]`` element indices of the relative logits of each (head,
    key) of a split: distance ``L - 1 - i``."""
    D: gl.constexpr = schedule.dims(plan)
    HEADS: gl.constexpr = D.q_heads
    GROUP: gl.constexpr = D.group
    HB: gl.constexpr = head_block(plan)
    mma: gl.constexpr = _layouts(gl.num_warps(), HB)[0]
    sr = gl.arange(0, HB, gl.SliceLayout(1, mma))
    sc = start + first + gl.arange(0, BN, gl.SliceLayout(0, mma))
    dist = gl.minimum(gl.maximum(L - 1 - sc, 0), WINDOW - 1)
    rh = gl.minimum(g * GROUP + sr, HEADS - 1)
    return rh[:, None] * WINDOW + dist[None, :].to(gl.int32)


@gluon.jit
def _attention_task(
    mb,
    task,
    cache,
    k_buf,
    v_buf,
    q: gl.constexpr,
    kv: gl.constexpr,
    rel: gl.constexpr,
    part: gl.constexpr,
    lse: gl.constexpr,
    CHAIN: gl.constexpr,
    STAGE: gl.constexpr,
):
    """Attention task ``task`` = (row, KV head ``g``, split ``s``): the
    regular kernel's program for its one KV block, the split, whose cached
    keys and values :func:`_issue_split` copied into ``k_buf`` and
    ``v_buf`` (the newest commit group may still run). Publishes the
    split's output and LSE."""
    D: gl.constexpr = schedule.dims(mb.plan)
    HEADS: gl.constexpr = D.q_heads
    KV_HEADS: gl.constexpr = D.kv_heads
    GROUP: gl.constexpr = D.group
    HB: gl.constexpr = head_block(mb.plan)
    row, g, s, _, L, launch0, tok = _task(cache, task, CHAIN, mb.plan)
    view = mb.row(row)
    nw: gl.constexpr = gl.num_warps()
    mma: gl.constexpr = _layouts(nw, HB)[0]
    tile: gl.constexpr = _layouts(nw, HB)[2]
    row_sum: gl.constexpr = _layouts(nw, HB)[3]
    start, first, end = _split(L, s)
    # The launch's own tokens are not in the cache yet: the split that holds
    # them (n_own > 0) also polls their k and v and writes them into its
    # tile below.
    o0 = launch0 - start - first  # the tile row of the first launch token
    n_own = gl.minimum(end - first, o0 + CHAIN) - gl.maximum(o0, 0)
    # q of the group's heads (the head block's other rows zero), the
    # relative logits and,
    # for that split, k and v of the request's tokens in this launch (rows
    # -tok .. 0 from this one): one round trip.
    ql: gl.constexpr = gl.BlockedLayout([1, 2], [2, 32], [nw, 1], [1, 0])
    qr = gl.arange(0, HB, gl.SliceLayout(1, ql))[:, None]
    qp = gl.arange(0, DIM // 2, gl.SliceLayout(0, ql))[None, :]
    head = gl.minimum(g * GROUP + qr, HEADS - 1)
    rel_index = _rel_index(L, g, start, first, mb.plan)
    NP: gl.constexpr = triton.next_power_of_2(CHAIN)
    tl_: gl.constexpr = gl.BlockedLayout([NP, 1], [1, 64], [1, nw], [1, 0])
    u = gl.arange(0, NP, gl.SliceLayout(1, tl_))[:, None]
    p = gl.arange(0, DIM // 2, gl.SliceLayout(0, tl_))[None, :]
    rows = view.row_offset(gl.where(u <= tok, u - tok, 0))
    line: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    tok_k = gl.allocate_shared_memory(gl.bfloat16, [NP * DIM], line)
    tok_v = gl.allocate_shared_memory(gl.bfloat16, [NP * DIM], line)
    if n_own > 0:
        got_q, got_r, got_k, got_v = view.poll_many(
            (q, rel, kv, kv),
            (
                head * (DIM // 2) + qp,
                rel_index // 2,
                rows + g * (DIM // 2) + p,
                rows + (KV_HEADS + g) * (DIM // 2) + p,
            ),
        )
        tok_k.store(gl.reshape(_bf16_values(got_k).to(gl.bfloat16), [NP * DIM]))
        tok_v.store(gl.reshape(_bf16_values(got_v).to(gl.bfloat16), [NP * DIM]))
    else:
        got_q, got_r = view.poll_many(
            (q, rel), (head * (DIM // 2) + qp, rel_index // 2)
        )
    mb.stamp(STAGE, trace.READY)
    bias = mx.bf16_half(got_r, rel_index)
    qv = _bf16_values(got_q)
    q_rows = gl.arange(0, HB, gl.SliceLayout(1, qv.type.layout))[:, None]
    qv = gl.where(q_rows < GROUP, qv, 0.0).to(gl.bfloat16)
    q_smem = gl.allocate_shared_memory(gl.bfloat16, [HB, DIM], tile)
    q_smem.store(qv)
    # This split's copies are complete in this lane; the barrier makes every
    # lane's copies and stores visible.
    async_copy.wait_group(1)
    gl.barrier()
    # The own tokens' tile rows take their mailbox rows, in the (at most
    # two) 8-row blocks that hold them. The tile rows past the split need
    # no zeros: their scores are masked, and their values (finite, stale or
    # zero, see attention_stage) multiply probabilities of zero, which adds
    # nothing.
    if n_own > 0:
        kv_load: gl.constexpr = _layouts(nw, HB)[1]
        b0 = gl.maximum(o0, 0) // 8
        # The tile's 8-row blocks: the same rows and swizzle (by row % 4).
        blocks: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 4, [1, 0])
        k_blocks = k_buf.reinterpret(shape=[BN // 8, 8, DIM], layout=blocks)
        v_blocks = v_buf.reinterpret(shape=[BN // 8, 8, DIM], layout=blocks)
        for i in gl.static_range(2):
            blk = gl.minimum(b0 + i, BN // 8 - 1)
            # The second block only when the own rows reach into it.
            if (i == 0) | (b0 + i <= gl.minimum(BN // 8 - 1, (o0 + CHAIN - 1) // 8)):
                r = blk * 8 + gl.arange(0, 8, gl.SliceLayout(1, kv_load))
                own = _key_rows(r, L, s, launch0)[1]
                keys = k_blocks.index(blk).load(kv_load)
                values = v_blocks.index(blk).load(kv_load)
                for t in gl.static_range(CHAIN):
                    k_t = tok_k.slice(t * DIM, DIM).load(gl.SliceLayout(0, kv_load))
                    keys = gl.where((own == t)[:, None], k_t[None, :], keys)
                    v_t = tok_v.slice(t * DIM, DIM).load(gl.SliceLayout(0, kv_load))
                    values = gl.where((own == t)[:, None], v_t[None, :], values)
                gl.barrier()
                k_blocks.index(blk).store(keys)
                v_blocks.index(blk).store(values)
        gl.barrier()
    a_op: gl.constexpr = gl.DotOperandLayout(0, mma, 4)
    b_op: gl.constexpr = gl.DotOperandLayout(1, mma, 4)
    # The products in two chained halves of K (the regular kernel's MFMA
    # order), so half an operand is live at a time.
    KH: gl.constexpr = 64
    k_cols = k_buf.permute([1, 0])
    scores = gl.zeros([HB, BN], gl.float32, mma)
    for c in gl.static_range(DIM // KH):
        scores = gl.amd.cdna4.mfma(
            q_smem.slice(c * KH, KH, dim=1).load(a_op),
            k_cols.slice(c * KH, KH, dim=0).load(b_op),
            scores,
        )
    pk = start + first + gl.arange(0, BN, gl.SliceLayout(0, mma))
    valid = pk < start + end
    # Scale, then add the bias: one FMA, as the regular kernel contracts it.
    scores = gl.where(valid[None, :], gl.fma(scores, SM_SCALE, bias), float("-inf"))
    m = gl.max(scores, axis=1)
    m_safe = gl.where(m == float("-inf"), 0.0, m)
    pr = gl.exp(scores - m_safe[:, None])
    l = gl.sum(gl.convert_layout(pr, row_sum), axis=1)
    p_smem = gl.allocate_shared_memory(gl.bfloat16, [HB, BN], tile)
    p_smem.store(pr.to(gl.bfloat16))
    gl.barrier()
    acc = gl.zeros([HB, DIM], gl.float32, mma)
    for c in gl.static_range(BN // KH):
        acc = gl.amd.cdna4.mfma(
            p_smem.slice(c * KH, KH, dim=1).load(a_op),
            v_buf.slice(c * KH, KH, dim=0).load(b_op),
            acc,
        )
    l_acc = gl.convert_layout(l, gl.SliceLayout(1, mma))
    out = acc / gl.where(l_acc > 0, l_acc, 1.0)[:, None]
    m_l = gl.convert_layout(m, gl.SliceLayout(1, row_sum))
    split_lse = gl.where(l > 0, m_l + gl.log(gl.where(l > 0, l, 1.0)), float("-inf"))
    r = gl.arange(0, HB, gl.SliceLayout(1, mma))
    c = gl.arange(0, DIM, gl.SliceLayout(0, mma))
    at = ((s * KV_HEADS + g) * GROUP + r[:, None]) * DIM + c[None, :]
    view.put(part, at, out.to(gl.int32, bitcast=True), mask=(r < GROUP)[:, None])
    lr = gl.arange(0, HB, gl.SliceLayout(1, row_sum))
    view.put(
        lse,
        s * HEADS + g * GROUP + lr,
        split_lse.to(gl.int32, bitcast=True),
        mask=lr < GROUP,
    )
    # The next task's copies, or the next stage on this CTA, write the LDS
    # after every wave's reads and stores.
    gl.barrier()


@gluon.jit
def attention_stage(
    mb,
    task,
    rounds,
    cache,
    q: gl.constexpr,
    kv: gl.constexpr,
    rel: gl.constexpr,
    part: gl.constexpr,
    lse: gl.constexpr,
    CHAIN: gl.constexpr,
    STAGE: gl.constexpr,
):
    """This CTA's attention tasks ``task + i * grid`` for ``i < rounds``
    (:func:`_attention_task`). Their keys and values stream through two
    LDS tiles: the copies of a task issue before the previous task
    computes, and those of the first before its polls. A round past the
    grid then costs its compute, not its loads' latency."""
    HB: gl.constexpr = head_block(mb.plan)
    mb.stamp(STAGE, trace.START)
    G: gl.constexpr = schedule.grid(mb.plan)
    # A copy writes a lane's 16 bytes contiguously: swizzle 8-value vectors.
    tile: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 4, [1, 0])
    k_bufs = gl.allocate_shared_memory(gl.bfloat16, [2, BN, DIM], tile)
    v_bufs = gl.allocate_shared_memory(gl.bfloat16, [2, BN, DIM], tile)
    if rounds > 0:
        # The value tiles start zero: the copies write cached keys' rows
        # only, so every value row stays finite.
        kv_load: gl.constexpr = _layouts(gl.num_warps(), HB)[1]
        zeros = gl.zeros([BN, DIM], gl.bfloat16, kv_load)
        for i in gl.static_range(2):
            v_bufs.index(i).store(zeros)
        gl.barrier()
        _issue_split(cache, k_bufs.index(0), v_bufs.index(0), task, CHAIN, mb.plan)
        for i in range(rounds):
            # The next task's copies (the last task's again after the last).
            following = task + gl.minimum(i + 1, rounds - 1) * G
            _issue_split(
                cache,
                k_bufs.index((i + 1) % 2),
                v_bufs.index((i + 1) % 2),
                following,
                CHAIN,
                mb.plan,
            )
            _attention_task(
                mb,
                task + i * G,
                cache,
                k_bufs.index(i % 2),
                v_bufs.index(i % 2),
                q,
                kv,
                rel,
                part,
                lse,
                CHAIN,
                STAGE,
            )
        async_copy.wait_group(0)
        gl.barrier()
    mb.stamp(STAGE, trace.END)


@gluon.jit
def _bf16_values(words):
    """FP32 ``[..., 2 n]`` of BF16 pairs ``words [..., n]``."""
    lo = (words << 16).to(gl.float32, bitcast=True)
    hi = (words & -65536).to(gl.float32, bitcast=True)
    shape: gl.constexpr = [words.shape[0], 2 * words.shape[1]]
    return gl.reshape(gl.join(lo, hi), shape)


@gluon.jit
def _sigmoid(x):
    """``1 / (1 + exp(-x))`` as ``rel_mha_gate_sink`` computes it off
    NVIDIA: libdevice ``exp``, round-to-nearest division."""
    return gl.div_rn(1.0, 1.0 + gl.extra.libdevice.exp(-x))


@gluon.jit
def merge_stage(
    mb,
    task,
    cache,
    qkvr: gl.constexpr,
    part: gl.constexpr,
    lse: gl.constexpr,
    o: gl.constexpr,
    o_flag: gl.constexpr,
    STAGE: gl.constexpr,
):
    """Merge task ``task`` = (row, KV head ``g``): for each of the group's
    query heads (a wave each), the regular merge of the row's used splits
    (online, in split order), then the output gate and sink. The heads'
    BF16 pairs go to ``o`` as plain words with flag ``g`` of the row in
    ``o_flag`` (the output projection loads every row's ``o``)."""
    D: gl.constexpr = schedule.dims(mb.plan)
    HEADS: gl.constexpr = D.q_heads
    KV_HEADS: gl.constexpr = D.kv_heads
    GROUP: gl.constexpr = D.group
    GATE_OFF: gl.constexpr = D.gate_off
    SINK_OFF: gl.constexpr = D.sink_off
    HB: gl.constexpr = head_block(mb.plan)
    mb.stamp(STAGE, trace.START)
    row = task // KV_HEADS
    g = task % KV_HEADS
    view = mb.row(row)
    L = gl.load(cache.positions + row) + 1
    n = gl.minimum(L, WINDOW)
    per = ((n + SPLITS - 1) // SPLITS + 31) // 32 * 32
    used = (n + per - 1) // per
    NW: gl.constexpr = gl.num_warps()
    # [head, split, column]: a wave a head (rows past the group's heads repeat
    # its last head and publish nothing), every split of a column in one
    # thread.
    ml: gl.constexpr = gl.BlockedLayout(
        [1, SPLITS, 2], [1, 1, 64], [NW, 1, 1], [2, 1, 0]
    )
    hl: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, ml))
    hh = gl.arange(0, HB, hl)
    h = g * GROUP + gl.minimum(hh, GROUP - 1)
    s = gl.arange(0, SPLITS, gl.SliceLayout(0, gl.SliceLayout(2, ml)))
    c = gl.arange(0, DIM, gl.SliceLayout(0, gl.SliceLayout(1, ml)))
    sp = gl.minimum(s, used - 1)
    h3 = h[:, None, None]
    at = ((sp[None, :, None] * KV_HEADS + g) * GROUP + h3 % GROUP) * DIM + c[
        None, None, :
    ]
    cl: gl.constexpr = gl.SliceLayout(1, ml)
    hc = gl.convert_layout(h, gl.SliceLayout(1, cl))[:, None]
    cc = gl.arange(0, DIM, gl.SliceLayout(0, cl))[None, :]
    got_p, got_l, got_g, got_s = view.poll_many(
        (part, lse, qkvr, qkvr),
        (
            at,
            sp[None, :] * HEADS
            + gl.convert_layout(h, gl.SliceLayout(1, gl.SliceLayout(2, ml)))[:, None],
            (GATE_OFF + hc * DIM + cc) // 2,
            (SINK_OFF + hc + cc * 0) // 2,
        ),
    )
    mb.stamp(STAGE, trace.READY)
    split_lse = got_l.to(gl.float32, bitcast=True)
    values = got_p.to(gl.float32, bitcast=True)
    # The regular merge: splits 0 .. 7 in order, each rescaling the running
    # sums. Its compiler contracts ``acc * old + scale * tv`` into
    # ``fma(acc, old, scale * tv)``; spelled out, since LLVM would fuse the
    # other product here.
    e_sum = gl.zeros([HB, DIM], gl.float32, cl)
    e_max = gl.full([HB, DIM], float("-inf"), gl.float32, cl)
    acc = gl.zeros([HB, DIM], gl.float32, cl)
    for i in gl.static_range(SPLITS):
        # Split i's values and LSE, the only finite terms of a max.
        on_i = (s == i)[None, :, None]
        tv = gl.max(gl.where(on_i, values, float("-inf")), axis=1)
        lse_i = gl.max(gl.where((s == i)[None, :], split_lse, float("-inf")), axis=1)
        lse_i = gl.convert_layout(lse_i, gl.SliceLayout(1, cl))[:, None]
        n_e_max = gl.maximum(lse_i, e_max)
        old_scale = gl.exp(e_max - n_e_max)
        split_scale = gl.exp(lse_i - n_e_max)
        on = i < used
        acc = gl.where(on, gl.fma(acc, old_scale, split_scale * tv), acc)
        e_sum = gl.where(on, gl.fma(e_sum, old_scale, split_scale), e_sum)
        e_max = gl.where(on, n_e_max, e_max)
    out = gl.where(e_sum > 0, acc / gl.maximum(e_sum, 1.0e-30), 0.0)
    full_lse = gl.where(e_sum > 0, e_max + gl.log(e_sum), float("-inf"))
    gate = mx.bf16_half(got_g, GATE_OFF + hc * DIM + cc)
    sink = mx.bf16_half(got_s, SINK_OFF + hc + cc * 0)
    gated = out * _sigmoid(gate)
    gated = gated * _sigmoid(full_lse - sink)
    mb.stamp(STAGE, trace.COMPUTED)
    # Each wave's head: BF16 pairs h * DIM / 2 ...
    pairs: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [NW, 1], [1, 0])
    v = gl.convert_layout(gated.to(gl.bfloat16), pairs)
    word = mx.pack_bf16_pairs(gl.reshape(v, [HB, DIM // 2, 2]))
    wl: gl.constexpr = word.type.layout
    hw = g * GROUP + gl.arange(0, HB, gl.SliceLayout(1, wl))
    pw = gl.arange(0, DIM // 2, gl.SliceLayout(0, wl))
    live = gl.arange(0, HB, gl.SliceLayout(1, wl)) < GROUP
    view.store_words(
        o, (hw * (DIM // 2))[:, None] + pw[None, :], word, mask=live[:, None]
    )
    view.raise_flag(o_flag, g)
    mb.stamp(STAGE, trace.END)


# A full attention layer: no window, every key of the request, over
# FULL_SPLITS splits of the regular kernel, each one or more KV blocks of BN
# keys (the regular kernel's blocks start at the split's first key). The
# relative logits reach REL_EXTENT keys back; farther keys add no bias.
REL_EXTENT = gl.constexpr(1024)


@gluon.jit
def _full_split(L, s, SPLITS: gl.constexpr):
    """``(first, end, blocks)`` of split ``s`` of a row at length ``L``: its
    key offsets ``first .. end`` and KV blocks (0 for an unused split)."""
    per = ((L + SPLITS - 1) // SPLITS + 31) // 32 * 32
    first = s * per
    end = gl.minimum(first + per, L)
    return first, end, gl.maximum(end - first + BN - 1, 0) // BN


@gluon.jit
def _full_task(
    cache, task, CHAIN: gl.constexpr, SPLITS: gl.constexpr, plan: gl.constexpr
):
    """``(row, g, s, request, L, launch0, tok)`` of full attention task
    ``task`` (row, KV head ``g``, split ``s``), as :func:`_task`."""
    D: gl.constexpr = schedule.dims(plan)
    KV_HEADS: gl.constexpr = D.kv_heads
    PER_ROW: gl.constexpr = KV_HEADS * SPLITS
    row = task // PER_ROW
    g = (task % PER_ROW) // SPLITS
    s = task % SPLITS
    tok = row % CHAIN
    L = gl.load(cache.positions + row).to(gl.int32) + 1
    return row, g, s, row // CHAIN, L, L - 1 - tok, tok


@gluon.jit
def _full_issue(
    cache,
    k_buf,
    v_buf,
    task,
    block,
    live,
    CHAIN: gl.constexpr,
    SPLITS: gl.constexpr,
    plan: gl.constexpr,
):
    """Copy the cached keys and values of KV block ``block`` of full task
    ``task`` into ``k_buf`` and ``v_buf`` (none unless ``live``), one
    commit group."""
    D: gl.constexpr = schedule.dims(plan)
    KV_HEADS: gl.constexpr = D.kv_heads
    HB: gl.constexpr = head_block(plan)
    _, g, s, request, L, launch0, _ = _full_task(cache, task, CHAIN, SPLITS, plan)
    first, end, _ = _full_split(L, s, SPLITS)
    kv_load: gl.constexpr = _layouts(gl.num_warps(), HB)[1]
    pk = first + block * BN + gl.arange(0, BN, gl.SliceLayout(1, kv_load))
    cached = (pk < end) & (pk < launch0) & live
    slot = _slots(cache, request, pk, cached)
    d = gl.arange(0, DIM, gl.SliceLayout(0, kv_load))
    at_kv = g * DIM + slot[:, None] * (KV_HEADS * DIM) + d[None, :]
    async_copy.global_load_to_shared(k_buf, cache.k_cache + at_kv, mask=cached[:, None])
    async_copy.global_load_to_shared(v_buf, cache.v_cache + at_kv, mask=cached[:, None])
    async_copy.commit_group()


@gluon.jit
def _full_block(
    view,
    k_buf,
    v_buf,
    q_smem,
    tok_k,
    tok_v,
    g,
    L,
    launch0,
    b0key,
    end,
    e_max,
    e_sum,
    acc,
    rel: gl.constexpr,
    CHAIN: gl.constexpr,
):
    """One KV block of keys ``b0key .. b0key + BN`` (``end`` the split's
    end) of the regular kernel's loop: the scores, the online rescale and
    the value product chained onto the rescaled accumulator."""
    D: gl.constexpr = schedule.dims(view.plan)
    HEADS: gl.constexpr = D.q_heads
    GROUP: gl.constexpr = D.group
    HB: gl.constexpr = head_block(view.plan)
    nw: gl.constexpr = gl.num_warps()
    mma: gl.constexpr = _layouts(nw, HB)[0]
    tile: gl.constexpr = _layouts(nw, HB)[2]
    row_sum: gl.constexpr = _layouts(nw, HB)[3]
    # This block's copies are complete in this lane; the barrier makes
    # every lane's copies visible.
    async_copy.wait_group(1)
    gl.barrier()
    o0 = launch0 - b0key  # the tile row of the first launch token
    if (o0 < BN) & (o0 + CHAIN > 0):
        kv_load: gl.constexpr = _layouts(nw, HB)[1]
        b8 = gl.maximum(o0, 0) // 8
        blocks: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 4, [1, 0])
        k_blocks = k_buf.reinterpret(shape=[BN // 8, 8, DIM], layout=blocks)
        v_blocks = v_buf.reinterpret(shape=[BN // 8, 8, DIM], layout=blocks)
        for i in gl.static_range(2):
            blk = gl.minimum(b8 + i, BN // 8 - 1)
            if (i == 0) | (b8 + i <= gl.minimum(BN // 8 - 1, (o0 + CHAIN - 1) // 8)):
                r = blk * 8 + gl.arange(0, 8, gl.SliceLayout(1, kv_load))
                pk = b0key + r
                own = gl.where((pk < end) & (pk >= launch0), pk - launch0, -1)
                keys = k_blocks.index(blk).load(kv_load)
                values = v_blocks.index(blk).load(kv_load)
                for t in gl.static_range(CHAIN):
                    k_t = tok_k.slice(t * DIM, DIM).load(gl.SliceLayout(0, kv_load))
                    keys = gl.where((own == t)[:, None], k_t[None, :], keys)
                    v_t = tok_v.slice(t * DIM, DIM).load(gl.SliceLayout(0, kv_load))
                    values = gl.where((own == t)[:, None], v_t[None, :], values)
                gl.barrier()
                k_blocks.index(blk).store(keys)
                v_blocks.index(blk).store(values)
        gl.barrier()
    a_op: gl.constexpr = gl.DotOperandLayout(0, mma, 4)
    b_op: gl.constexpr = gl.DotOperandLayout(1, mma, 4)
    KH: gl.constexpr = 64
    k_cols = k_buf.permute([1, 0])
    scores = gl.zeros([HB, BN], gl.float32, mma)
    for c in gl.static_range(DIM // KH):
        scores = gl.amd.cdna4.mfma(
            q_smem.slice(c * KH, KH, dim=1).load(a_op),
            k_cols.slice(c * KH, KH, dim=0).load(b_op),
            scores,
        )
    sr = gl.arange(0, HB, gl.SliceLayout(1, mma))
    pk = b0key + gl.arange(0, BN, gl.SliceLayout(0, mma))
    dist = L - 1 - pk
    near = dist < REL_EXTENT
    # The relative logits of the keys within REL_EXTENT of the row (a block
    # past them polls nothing); the others add zero, as the regular kernel's
    # masked load does.
    if L - 1 - b0key < REL_EXTENT + BN:
        rh = gl.minimum(g * GROUP + sr, HEADS - 1)
        rel_index = (
            rh[:, None] * REL_EXTENT
            + gl.minimum(gl.maximum(dist, 0), REL_EXTENT - 1)[None, :]
        )
        (got_r,) = view.poll_many((rel,), (rel_index // 2,))
        bias = gl.where(near[None, :], mx.bf16_half(got_r, rel_index), 0.0)
    else:
        bias = gl.zeros([HB, BN], gl.float32, mma)
    valid = pk < end
    scores = gl.where(valid[None, :], gl.fma(scores, SM_SCALE, bias), float("-inf"))
    n_e_max = gl.maximum(gl.max(scores, axis=1), e_max)
    # Rows past the group's heads hold no key (all -inf): keep them finite.
    m_safe = gl.where(n_e_max == float("-inf"), 0.0, n_e_max)
    re_scale = gl.exp(e_max - m_safe)
    pr = gl.exp(scores - m_safe[:, None])
    p_smem = gl.allocate_shared_memory(gl.bfloat16, [HB, BN], tile)
    p_smem.store(pr.to(gl.bfloat16))
    gl.barrier()
    acc = acc * re_scale[:, None]
    for c in gl.static_range(BN // KH):
        acc = gl.amd.cdna4.mfma(
            p_smem.slice(c * KH, KH, dim=1).load(a_op),
            v_buf.slice(c * KH, KH, dim=0).load(b_op),
            acc,
        )
    l = gl.sum(gl.convert_layout(pr, row_sum), axis=1)
    re_l = gl.convert_layout(re_scale, gl.SliceLayout(1, row_sum))
    e_sum = gl.fma(e_sum, re_l, l)
    # The next block's copies write the LDS after every wave's reads.
    gl.barrier()
    return n_e_max, e_sum, acc


@gluon.jit
def full_attention_stage(
    mb,
    task,
    rounds,
    cache,
    q: gl.constexpr,
    kv: gl.constexpr,
    rel: gl.constexpr,
    part: gl.constexpr,
    lse: gl.constexpr,
    CHAIN: gl.constexpr,
    SPLITS: gl.constexpr,
    STAGE: gl.constexpr,
):
    """This CTA's full attention tasks ``task + i * grid`` for ``i <
    rounds``: task (row, KV head ``g``, split ``s``) runs the regular
    kernel's program of that split over every KV block of it, keys and
    values streaming through two LDS tiles (a block's copies issue before
    the previous block computes), and publishes the split's output and LSE.
    An unused split publishes nothing (the merge skips it)."""
    D: gl.constexpr = schedule.dims(mb.plan)
    HEADS: gl.constexpr = D.q_heads
    KV_HEADS: gl.constexpr = D.kv_heads
    GROUP: gl.constexpr = D.group
    HB: gl.constexpr = head_block(mb.plan)
    mb.stamp(STAGE, trace.START)
    G: gl.constexpr = schedule.grid(mb.plan)
    nw: gl.constexpr = gl.num_warps()
    mma: gl.constexpr = _layouts(nw, HB)[0]
    tile: gl.constexpr = _layouts(nw, HB)[2]
    row_sum: gl.constexpr = _layouts(nw, HB)[3]
    kv_tile: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 4, [1, 0])
    k_bufs = gl.allocate_shared_memory(gl.bfloat16, [2, BN, DIM], kv_tile)
    v_bufs = gl.allocate_shared_memory(gl.bfloat16, [2, BN, DIM], kv_tile)
    q_smem = gl.allocate_shared_memory(gl.bfloat16, [HB, DIM], tile)
    NP: gl.constexpr = triton.next_power_of_2(CHAIN)
    line: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    tok_k = gl.allocate_shared_memory(gl.bfloat16, [NP * DIM], line)
    tok_v = gl.allocate_shared_memory(gl.bfloat16, [NP * DIM], line)
    if rounds > 0:
        kv_load: gl.constexpr = _layouts(nw, HB)[1]
        zeros = gl.zeros([BN, DIM], gl.bfloat16, kv_load)
        for i in gl.static_range(2):
            v_bufs.index(i).store(zeros)
        gl.barrier()
        _full_issue(
            cache,
            k_bufs.index(0),
            v_bufs.index(0),
            task,
            0,
            True,
            CHAIN,
            SPLITS,
            mb.plan,
        )
        unit = 0
        for i in range(rounds):
            t = task + i * G
            row, g, s, _, L, launch0, tok = _full_task(cache, t, CHAIN, SPLITS, mb.plan)
            view = mb.row(row)
            first, end, nb = _full_split(L, s, SPLITS)
            # q of the group's heads (the other rows zero) and, for the split
            # that holds the launch's own tokens, their k and v.
            ql: gl.constexpr = gl.BlockedLayout([1, 2], [2, 32], [nw, 1], [1, 0])
            qr = gl.arange(0, HB, gl.SliceLayout(1, ql))[:, None]
            qp = gl.arange(0, DIM // 2, gl.SliceLayout(0, ql))[None, :]
            head = gl.minimum(g * GROUP + qr, HEADS - 1)
            tl_: gl.constexpr = gl.BlockedLayout([NP, 1], [1, 64], [1, nw], [1, 0])
            u = gl.arange(0, NP, gl.SliceLayout(1, tl_))[:, None]
            p = gl.arange(0, DIM // 2, gl.SliceLayout(0, tl_))[None, :]
            rows = view.row_offset(gl.where(u <= tok, u - tok, 0))
            if (nb > 0) & (end > launch0):
                got_q, got_k, got_v = view.poll_many(
                    (q, kv, kv),
                    (
                        head * (DIM // 2) + qp,
                        rows + g * (DIM // 2) + p,
                        rows + (KV_HEADS + g) * (DIM // 2) + p,
                    ),
                )
                tok_k.store(gl.reshape(_bf16_values(got_k).to(gl.bfloat16), [NP * DIM]))
                tok_v.store(gl.reshape(_bf16_values(got_v).to(gl.bfloat16), [NP * DIM]))
            else:
                (got_q,) = view.poll_many((q,), (head * (DIM // 2) + qp,))
            mb.stamp(STAGE, trace.READY)
            qv = _bf16_values(got_q)
            q_rows = gl.arange(0, HB, gl.SliceLayout(1, qv.type.layout))[:, None]
            q_smem.store(gl.where(q_rows < GROUP, qv, 0.0).to(gl.bfloat16))
            e_max = gl.full([HB], float("-inf"), gl.float32, gl.SliceLayout(1, mma))
            e_sum = gl.zeros([HB], gl.float32, gl.SliceLayout(1, row_sum))
            acc = gl.zeros([HB, DIM], gl.float32, mma)
            for b in range(nb):
                # The next block's copies: this task's next block, or the
                # next task's first (none past the last).
                last = b + 1 == nb
                live = (~last) | (i + 1 < rounds)
                # Past the last task: this task again, copying nothing.
                nxt_t = gl.where(last & live, t + G, t)
                nxt_b = gl.where(last, 0, b + 1)
                _full_issue(
                    cache,
                    k_bufs.index((unit + 1) % 2),
                    v_bufs.index((unit + 1) % 2),
                    nxt_t,
                    nxt_b,
                    live,
                    CHAIN,
                    SPLITS,
                    mb.plan,
                )
                e_max, e_sum, acc = _full_block(
                    view,
                    k_bufs.index(unit % 2),
                    v_bufs.index(unit % 2),
                    q_smem,
                    tok_k,
                    tok_v,
                    g,
                    L,
                    launch0,
                    first + b * BN,
                    end,
                    e_max,
                    e_sum,
                    acc,
                    rel,
                    CHAIN,
                )
                unit += 1
            if nb == 0:
                # An unused split: the next task's first block, as a block
                # of this one would have issued.
                _full_issue(
                    cache,
                    k_bufs.index((unit + 1) % 2),
                    v_bufs.index((unit + 1) % 2),
                    gl.where(i + 1 < rounds, t + G, t),
                    0,
                    i + 1 < rounds,
                    CHAIN,
                    SPLITS,
                    mb.plan,
                )
                async_copy.wait_group(1)
                unit += 1
            else:
                l_acc = gl.convert_layout(e_sum, gl.SliceLayout(1, mma))
                out = acc / gl.where(l_acc > 0, l_acc, 1.0)[:, None]
                m_l = gl.convert_layout(e_max, gl.SliceLayout(1, row_sum))
                split_lse = gl.where(
                    e_sum > 0,
                    m_l + gl.log(gl.where(e_sum > 0, e_sum, 1.0)),
                    float("-inf"),
                )
                r = gl.arange(0, HB, gl.SliceLayout(1, mma))
                c = gl.arange(0, DIM, gl.SliceLayout(0, mma))
                at = ((s * KV_HEADS + g) * GROUP + r[:, None]) * DIM + c[None, :]
                view.put(
                    part, at, out.to(gl.int32, bitcast=True), mask=(r < GROUP)[:, None]
                )
                lr = gl.arange(0, HB, gl.SliceLayout(1, row_sum))
                view.put(
                    lse,
                    s * HEADS + g * GROUP + lr,
                    split_lse.to(gl.int32, bitcast=True),
                    mask=lr < GROUP,
                )
        async_copy.wait_group(0)
        gl.barrier()
    mb.stamp(STAGE, trace.END)


@gluon.jit
def full_merge_stage(
    mb,
    task,
    cache,
    qkvr: gl.constexpr,
    part: gl.constexpr,
    lse: gl.constexpr,
    o: gl.constexpr,
    o_flag: gl.constexpr,
    SPLITS: gl.constexpr,
    STAGE: gl.constexpr,
):
    """:func:`merge_stage` of a full attention layer: the row's used splits
    of ``SPLITS`` over its whole length, merged in split order (polled 8
    splits at a time, those of the used ones), then the output gate and
    sink."""
    D: gl.constexpr = schedule.dims(mb.plan)
    HEADS: gl.constexpr = D.q_heads
    KV_HEADS: gl.constexpr = D.kv_heads
    GROUP: gl.constexpr = D.group
    GATE_OFF: gl.constexpr = D.gate_off
    SINK_OFF: gl.constexpr = D.sink_off
    HB: gl.constexpr = head_block(mb.plan)
    mb.stamp(STAGE, trace.START)
    row = task // KV_HEADS
    g = task % KV_HEADS
    view = mb.row(row)
    L = (gl.load(cache.positions + row) + 1).to(gl.int32)
    per = ((L + SPLITS - 1) // SPLITS + 31) // 32 * 32
    used = (L + per - 1) // per
    NW: gl.constexpr = gl.num_warps()
    CH: gl.constexpr = SPLITS if SPLITS < 8 else 8
    ml: gl.constexpr = gl.BlockedLayout([1, CH, 2], [1, 1, 64], [NW, 1, 1], [2, 1, 0])
    hl: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, ml))
    hh = gl.arange(0, HB, hl)
    h = g * GROUP + gl.minimum(hh, GROUP - 1)
    s = gl.arange(0, CH, gl.SliceLayout(0, gl.SliceLayout(2, ml)))
    c = gl.arange(0, DIM, gl.SliceLayout(0, gl.SliceLayout(1, ml)))
    h3 = h[:, None, None]
    cl: gl.constexpr = gl.SliceLayout(1, ml)
    hc = gl.convert_layout(h, gl.SliceLayout(1, cl))[:, None]
    cc = gl.arange(0, DIM, gl.SliceLayout(0, cl))[None, :]
    got_g, got_s = view.poll_many(
        (qkvr, qkvr),
        ((GATE_OFF + hc * DIM + cc) // 2, (SINK_OFF + hc + cc * 0) // 2),
    )
    e_sum = gl.zeros([HB, DIM], gl.float32, cl)
    e_max = gl.full([HB, DIM], float("-inf"), gl.float32, cl)
    acc = gl.zeros([HB, DIM], gl.float32, cl)
    for chunk in gl.static_range(SPLITS // CH):
        if chunk * CH < used:
            sp = gl.minimum(chunk * CH + s, used - 1)
            at = ((sp[None, :, None] * KV_HEADS + g) * GROUP + h3 % GROUP) * DIM + c[
                None, None, :
            ]
            hs = gl.convert_layout(h, gl.SliceLayout(1, gl.SliceLayout(2, ml)))
            got_p, got_l = view.poll_many(
                (part, lse), (at, sp[None, :] * HEADS + hs[:, None])
            )
            split_lse = got_l.to(gl.float32, bitcast=True)
            values = got_p.to(gl.float32, bitcast=True)
            # The regular merge, splits in order (as merge_stage).
            for i in gl.static_range(CH):
                on_i = (s == i)[None, :, None]
                tv = gl.max(gl.where(on_i, values, float("-inf")), axis=1)
                lse_i = gl.max(
                    gl.where((s == i)[None, :], split_lse, float("-inf")), axis=1
                )
                lse_i = gl.convert_layout(lse_i, gl.SliceLayout(1, cl))[:, None]
                n_e_max = gl.maximum(lse_i, e_max)
                old_scale = gl.exp(e_max - n_e_max)
                split_scale = gl.exp(lse_i - n_e_max)
                on = chunk * CH + i < used
                acc = gl.where(on, gl.fma(acc, old_scale, split_scale * tv), acc)
                e_sum = gl.where(on, gl.fma(e_sum, old_scale, split_scale), e_sum)
                e_max = gl.where(on, n_e_max, e_max)
    mb.stamp(STAGE, trace.READY)
    out = gl.where(e_sum > 0, acc / gl.maximum(e_sum, 1.0e-30), 0.0)
    full_lse = gl.where(e_sum > 0, e_max + gl.log(e_sum), float("-inf"))
    gate = mx.bf16_half(got_g, GATE_OFF + hc * DIM + cc)
    sink = mx.bf16_half(got_s, SINK_OFF + hc + cc * 0)
    gated = out * _sigmoid(gate)
    gated = gated * _sigmoid(full_lse - sink)
    mb.stamp(STAGE, trace.COMPUTED)
    pairs: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [NW, 1], [1, 0])
    v = gl.convert_layout(gated.to(gl.bfloat16), pairs)
    word = mx.pack_bf16_pairs(gl.reshape(v, [HB, DIM // 2, 2]))
    wl: gl.constexpr = word.type.layout
    hw = g * GROUP + gl.arange(0, HB, gl.SliceLayout(1, wl))
    pw = gl.arange(0, DIM // 2, gl.SliceLayout(0, wl))
    live = gl.arange(0, HB, gl.SliceLayout(1, wl)) < GROUP
    view.store_words(
        o, (hw * (DIM // 2))[:, None] + pw[None, :], word, mask=live[:, None]
    )
    view.raise_flag(o_flag, g)
    mb.stamp(STAGE, trace.END)


# The shared-row full attention (a verify block's 8 rows a task): the tile
# rows are (row, head) pairs, row r's HB head slots (its head block,
# head_block) at HB r .. HB r + HB - 1: MT = RB HB tile rows, a pass of 4
# rows MH = 4 HB of them (a block's two passes share its keys).
RB = gl.constexpr(8)  # rows a task: a verify block


@gluon.jit
def _per(L, SPLITS: gl.constexpr):
    """Keys a split of a row at length ``L`` (the regular kernel's)."""
    return ((L + SPLITS - 1) // SPLITS + 31) // 32 * 32


@gluon.jit
def _block_task(
    cache, task, SPLITS: gl.constexpr, PASSES: gl.constexpr, plan: gl.constexpr
):
    """``(request, g, s, row0, La, Lz, launch0, uniform)`` of shared-row task
    ``task``: request, KV head ``g``, split ``s`` and the task's rows
    ``row0 .. row0 + 4 PASSES - 1`` of the request's verify block (all 8
    with 2 passes, else half), their first and last lengths, the position
    of the block's first token (the launch's first own token), and whether
    the rows share one split length (else each row runs alone,
    :func:`_row_alone`)."""
    D: gl.constexpr = schedule.dims(plan)
    KV_HEADS: gl.constexpr = D.kv_heads
    PARTS: gl.constexpr = 2 // PASSES
    unit = task // PARTS
    row0 = (task % PARTS) * 4
    request = unit // (KV_HEADS * SPLITS)
    g = (unit // SPLITS) % KV_HEADS
    s = unit % SPLITS
    base = cache.positions + request * RB
    La = gl.load(base + row0).to(gl.int32) + 1
    Lz = gl.load(base + row0 + 4 * PASSES - 1).to(gl.int32) + 1
    launch0 = gl.load(base).to(gl.int32)
    return request, g, s, row0, La, Lz, launch0, _per(La, SPLITS) == _per(Lz, SPLITS)


@gluon.jit
def _block_issue(
    cache,
    buf,
    task,
    block,
    live,
    V: gl.constexpr,
    SPLITS: gl.constexpr,
    PASSES: gl.constexpr,
    plan: gl.constexpr,
):
    """Copy the keys (or with ``V`` the values) of KV block ``block`` of
    shared-row task ``task`` (its split's keys up to its last row's end, but
    the launch's own tokens) into ``buf``, one commit group; none unless
    ``live`` and the task's rows share their split length."""
    D: gl.constexpr = schedule.dims(plan)
    KV_HEADS: gl.constexpr = D.kv_heads
    HB: gl.constexpr = head_block(plan)
    request, g, s, _, _, Lz, launch0, uniform = _block_task(
        cache, task, SPLITS, PASSES, plan
    )
    per = _per(Lz, SPLITS)
    first = s * per
    end = gl.minimum(first + per, Lz)
    kv_load: gl.constexpr = _layouts(gl.num_warps(), HB)[1]
    pk = first + block * BN + gl.arange(0, BN, gl.SliceLayout(1, kv_load))
    cached = (pk < end) & (pk < launch0) & live & uniform
    slot = _slots(cache, request, pk, cached)
    d = gl.arange(0, DIM, gl.SliceLayout(0, kv_load))
    at_kv = g * DIM + slot[:, None] * (KV_HEADS * DIM) + d[None, :]
    src = cache.v_cache if V else cache.k_cache
    async_copy.global_load_to_shared(buf, src + at_kv, mask=cached[:, None])
    async_copy.commit_group()


@gluon.jit
def _row_lengths(cache, request, rbase, layout: gl.constexpr, plan: gl.constexpr):
    """``L`` of each tile row's row (``[MH]`` in ``layout``) of a pass over
    rows ``rbase .. rbase + 3`` of the request's verify block."""
    HB: gl.constexpr = head_block(plan)
    MH: gl.constexpr = 4 * HB
    m = gl.arange(0, MH, layout)
    return gl.load(cache.positions + request * RB + rbase + m // HB).to(gl.int32) + 1


@gluon.jit
def _shared_block(
    mb,
    request,
    k_buf,
    v_buf,
    probs,
    q_stage,
    tok_k,
    tok_v,
    g,
    launch0,
    b0key,
    first,
    per,
    e_max,
    e_sum,
    acc,
    cache,
    row0,
    rel: gl.constexpr,
    PASSES: gl.constexpr,
):
    """One KV block of keys ``b0key ..`` of a shared-row task: each tile
    row's (row, head) scores, its row's mask and relative logits, the
    online rescale and the value product, as :func:`_full_block` for each
    of the 8 rows. The probabilities pass through ``probs`` (FP32 LDS): the
    row sums and the value product's operand load them in their layouts."""
    HB: gl.constexpr = head_block(mb.plan)
    nw: gl.constexpr = gl.num_warps()
    mma: gl.constexpr = _layouts(nw, HB)[0]
    row_sum: gl.constexpr = _layouts(nw, HB)[3]
    # This block's keys have landed in this lane (its values and the next
    # keys, the two newest groups, may run: the values land during the
    # scores); the barrier makes every lane's copies visible.
    async_copy.wait_group(2)
    gl.barrier()
    o0 = launch0 - b0key
    own_block = (o0 < BN) & (o0 + RB > 0)
    if own_block:
        # The launch's own tokens' rows of both tiles: the values first.
        async_copy.wait_group(1)
        gl.barrier()
        kv_load: gl.constexpr = _layouts(nw, HB)[1]
        b8 = gl.maximum(o0, 0) // 8
        blocks: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 4, [1, 0])
        k_blocks = k_buf.reinterpret(shape=[BN // 8, 8, DIM], layout=blocks)
        v_blocks = v_buf.reinterpret(shape=[BN // 8, 8, DIM], layout=blocks)
        for i in gl.static_range(2):
            blk = gl.minimum(b8 + i, BN // 8 - 1)
            if (i == 0) | (b8 + i <= gl.minimum(BN // 8 - 1, (o0 + RB - 1) // 8)):
                r = blk * 8 + gl.arange(0, 8, gl.SliceLayout(1, kv_load))
                pk = b0key + r
                own = gl.where((pk >= launch0) & (pk < launch0 + RB), pk - launch0, -1)
                keys = k_blocks.index(blk).load(kv_load)
                values = v_blocks.index(blk).load(kv_load)
                for t in gl.static_range(RB):
                    k_t = tok_k.slice(t * DIM, DIM).load(gl.SliceLayout(0, kv_load))
                    keys = gl.where((own == t)[:, None], k_t[None, :], keys)
                    v_t = tok_v.slice(t * DIM, DIM).load(gl.SliceLayout(0, kv_load))
                    values = gl.where((own == t)[:, None], v_t[None, :], values)
                gl.barrier()
                k_blocks.index(blk).store(keys)
                v_blocks.index(blk).store(values)
        gl.barrier()
    out_max = ()
    out_sum = ()
    out_acc = ()
    for half in gl.static_range(PASSES):
        n_max, n_sum, n_acc = _shared_half(
            mb,
            request,
            k_buf,
            v_buf,
            probs,
            q_stage,
            g,
            b0key,
            first,
            per,
            e_max[half],
            e_sum[half],
            acc[half],
            cache,
            rel,
            half,
            row0 + 4 * half,
            half == 0,
        )
        out_max += (n_max,)
        out_sum += (n_sum,)
        out_acc += (n_acc,)
    return out_max, out_sum, out_acc


@gluon.jit
def _shared_half(
    mb,
    request,
    k_buf,
    v_buf,
    probs,
    q_stage,
    g,
    b0key,
    first,
    per,
    e_max,
    e_sum,
    acc,
    cache,
    rel: gl.constexpr,
    half: gl.constexpr,
    rbase,
    WAIT_V: gl.constexpr,
):
    """Pass ``half`` of a shared-row block: tile rows ``MH half ..`` of the
    staged q, the (row, head) pairs of rows ``rbase .. rbase + 3`` of the
    request's verify block."""
    D: gl.constexpr = schedule.dims(mb.plan)
    HEADS: gl.constexpr = D.q_heads
    GROUP: gl.constexpr = D.group
    HB: gl.constexpr = head_block(mb.plan)
    MH: gl.constexpr = 4 * HB
    nw: gl.constexpr = gl.num_warps()
    mma: gl.constexpr = _layouts(nw, HB)[0]
    row_sum: gl.constexpr = _layouts(nw, HB)[3]
    b_op: gl.constexpr = gl.DotOperandLayout(1, mma, 4)
    a_op: gl.constexpr = gl.DotOperandLayout(0, mma, 4)
    KH: gl.constexpr = 64
    k_cols = k_buf.permute([1, 0])
    q_half = q_stage.slice(half * MH, MH, dim=0)
    scores = gl.zeros([MH, BN], gl.float32, mma)
    for c in gl.static_range(DIM // KH):
        scores = gl.amd.cdna4.mfma(
            q_half.slice(c * KH, KH, dim=1).load(a_op),
            k_cols.slice(c * KH, KH, dim=0).load(b_op),
            scores,
        )
    m = gl.arange(0, MH, gl.SliceLayout(1, mma))
    L = _row_lengths(cache, request, rbase, gl.SliceLayout(1, mma), mb.plan)
    end = gl.minimum(first + per, L)
    pk = b0key + gl.arange(0, BN, gl.SliceLayout(0, mma))
    # The relative logits when the block holds a key within REL_EXTENT of
    # any of the pass's rows: the shortest row's is the nearest.
    if gl.min(L, axis=0) - 1 - b0key < REL_EXTENT + BN:
        dist = L[:, None] - 1 - pk[None, :]
        head = gl.minimum(g * GROUP + m % HB, HEADS - 1)
        rel_index = head[:, None] * REL_EXTENT + gl.minimum(
            gl.maximum(dist, 0), REL_EXTENT - 1
        )
        at = mb.row_offset(request * RB + rbase + m // HB)[:, None] + rel_index // 2
        (got_r,) = mb.poll_many((rel,), (at,))
        bias = gl.where(dist < REL_EXTENT, mx.bf16_half(got_r, rel_index), 0.0)
    else:
        bias = gl.zeros([MH, BN], gl.float32, mma)
    valid = pk[None, :] < end[:, None]
    scores = gl.where(valid, gl.fma(scores, SM_SCALE, bias), float("-inf"))
    n_e_max = gl.maximum(gl.max(scores, axis=1), e_max)
    m_safe = gl.where(n_e_max == float("-inf"), 0.0, n_e_max)
    re_scale = gl.exp(e_max - m_safe)
    probs.store(gl.exp(scores - m_safe[:, None]))
    if WAIT_V:
        # The block's values (the first pass's product reads them first).
        async_copy.wait_group(1)
    gl.barrier()
    acc = acc * re_scale[:, None]
    for c in gl.static_range(BN // KH):
        p_c = probs.slice(c * KH, KH, dim=1).load(a_op).to(gl.bfloat16)
        acc = gl.amd.cdna4.mfma(p_c, v_buf.slice(c * KH, KH, dim=0).load(b_op), acc)
    l = gl.sum(probs.load(row_sum), axis=1)
    re_l = gl.convert_layout(re_scale, gl.SliceLayout(1, row_sum))
    e_sum = gl.fma(e_sum, re_l, l)
    # The next pass's probabilities, or the next values' copies, write the
    # LDS after every wave's reads.
    gl.barrier()
    return n_e_max, e_sum, acc


@gluon.jit
def _row_alone(
    mb,
    task,
    cache,
    k_buf,
    v_buf,
    q,
    kv: gl.constexpr,
    rel: gl.constexpr,
    part: gl.constexpr,
    lse: gl.constexpr,
    SPLITS: gl.constexpr,
    PASSES: gl.constexpr,
):
    """A shared-row task whose rows split differently: each of its rows'
    split as :func:`full_attention_stage`'s task, its blocks copied and
    computed one at a time."""
    D: gl.constexpr = schedule.dims(mb.plan)
    KV_HEADS: gl.constexpr = D.kv_heads
    request, g, s, row0, _, _, _, _ = _block_task(cache, task, SPLITS, PASSES, mb.plan)
    for r in range(row0, row0 + 4 * PASSES):
        # The per-row task of (row, g, s).
        t = ((request * RB + r) * KV_HEADS + g) * SPLITS + s
        _single_row_task(mb, t, cache, k_buf, v_buf, q, kv, rel, part, lse, SPLITS)


@gluon.jit
def _single_row_task(
    mb,
    t,
    cache,
    k_buf,
    v_buf,
    q: gl.constexpr,
    kv: gl.constexpr,
    rel: gl.constexpr,
    part: gl.constexpr,
    lse: gl.constexpr,
    SPLITS: gl.constexpr,
):
    """Per-row task ``t`` (row-major: row, KV head, split) of a full layer,
    unpipelined: each block's copies wait before it computes."""
    D: gl.constexpr = schedule.dims(mb.plan)
    HEADS: gl.constexpr = D.q_heads
    KV_HEADS: gl.constexpr = D.kv_heads
    GROUP: gl.constexpr = D.group
    HB: gl.constexpr = head_block(mb.plan)
    nw: gl.constexpr = gl.num_warps()
    mma: gl.constexpr = _layouts(nw, HB)[0]
    tile: gl.constexpr = _layouts(nw, HB)[2]
    row_sum: gl.constexpr = _layouts(nw, HB)[3]
    PER_ROW: gl.constexpr = KV_HEADS * SPLITS
    row = t // PER_ROW
    g = (t % PER_ROW) // SPLITS
    s = t % SPLITS
    tok = row % RB
    L = gl.load(cache.positions + row).to(gl.int32) + 1
    launch0 = L - 1 - tok
    request = row // RB
    view = mb.row(row)
    first, end, nb = _full_split(L, s, SPLITS)
    if nb > 0:
        ql: gl.constexpr = gl.BlockedLayout([1, 2], [2, 32], [nw, 1], [1, 0])
        qr = gl.arange(0, HB, gl.SliceLayout(1, ql))[:, None]
        qp = gl.arange(0, DIM // 2, gl.SliceLayout(0, ql))[None, :]
        head = gl.minimum(g * GROUP + qr, HEADS - 1)
        tl_: gl.constexpr = gl.BlockedLayout([RB, 1], [1, 64], [1, nw], [1, 0])
        u = gl.arange(0, RB, gl.SliceLayout(1, tl_))[:, None]
        p = gl.arange(0, DIM // 2, gl.SliceLayout(0, tl_))[None, :]
        rows = view.row_offset(gl.where(u <= tok, u - tok, 0))
        got_q, got_k, got_v = view.poll_many(
            (q, kv, kv),
            (
                head * (DIM // 2) + qp,
                rows + g * (DIM // 2) + p,
                rows + (KV_HEADS + g) * (DIM // 2) + p,
            ),
        )
        line: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
        tok_k = gl.allocate_shared_memory(gl.bfloat16, [RB * DIM], line)
        tok_v = gl.allocate_shared_memory(gl.bfloat16, [RB * DIM], line)
        tok_k.store(gl.reshape(_bf16_values(got_k).to(gl.bfloat16), [RB * DIM]))
        tok_v.store(gl.reshape(_bf16_values(got_v).to(gl.bfloat16), [RB * DIM]))
        qv = _bf16_values(got_q)
        q_rows = gl.arange(0, HB, gl.SliceLayout(1, qv.type.layout))[:, None]
        q_smem = gl.allocate_shared_memory(gl.bfloat16, [HB, DIM], tile)
        q_smem.store(gl.where(q_rows < GROUP, qv, 0.0).to(gl.bfloat16))
        e_max = gl.full([HB], float("-inf"), gl.float32, gl.SliceLayout(1, mma))
        e_sum = gl.zeros([HB], gl.float32, gl.SliceLayout(1, row_sum))
        acc = gl.zeros([HB, DIM], gl.float32, mma)
        kv_load: gl.constexpr = _layouts(nw, HB)[1]
        for b in range(nb):
            pk = first + b * BN + gl.arange(0, BN, gl.SliceLayout(1, kv_load))
            cached = (pk < end) & (pk < launch0)
            slot = _slots(cache, request, pk, cached)
            d = gl.arange(0, DIM, gl.SliceLayout(0, kv_load))
            at_kv = g * DIM + slot[:, None] * (KV_HEADS * DIM) + d[None, :]
            async_copy.global_load_to_shared(
                k_buf, cache.k_cache + at_kv, mask=cached[:, None]
            )
            async_copy.global_load_to_shared(
                v_buf, cache.v_cache + at_kv, mask=cached[:, None]
            )
            async_copy.commit_group()
            # _full_block waits for all but the newest group: an empty one.
            async_copy.commit_group()
            e_max, e_sum, acc = _full_block(
                view,
                k_buf,
                v_buf,
                q_smem,
                tok_k,
                tok_v,
                g,
                L,
                launch0,
                first + b * BN,
                end,
                e_max,
                e_sum,
                acc,
                rel,
                RB,
            )
        _publish_split(view, part, lse, s, g, e_max, e_sum, acc, HB)


@gluon.jit
def _publish_split(
    view,
    part: gl.constexpr,
    lse: gl.constexpr,
    s,
    g,
    e_max,
    e_sum,
    acc,
    ROWS: gl.constexpr,
):
    """Publish a split's output and LSE of ``ROWS`` tile rows of one row's
    heads (rows past the group's heads publish nothing)."""
    D: gl.constexpr = schedule.dims(view.plan)
    HEADS: gl.constexpr = D.q_heads
    KV_HEADS: gl.constexpr = D.kv_heads
    GROUP: gl.constexpr = D.group
    HB: gl.constexpr = head_block(view.plan)
    nw: gl.constexpr = gl.num_warps()
    mma: gl.constexpr = _layouts(nw, HB)[0]
    row_sum: gl.constexpr = _layouts(nw, HB)[3]
    l_acc = gl.convert_layout(e_sum, gl.SliceLayout(1, mma))
    out = acc / gl.where(l_acc > 0, l_acc, 1.0)[:, None]
    m_l = gl.convert_layout(e_max, gl.SliceLayout(1, row_sum))
    split_lse = gl.where(
        e_sum > 0, m_l + gl.log(gl.where(e_sum > 0, e_sum, 1.0)), float("-inf")
    )
    r = gl.arange(0, ROWS, gl.SliceLayout(1, mma))
    c = gl.arange(0, DIM, gl.SliceLayout(0, mma))
    at = ((s * KV_HEADS + g) * GROUP + r[:, None]) * DIM + c[None, :]
    view.put(part, at, out.to(gl.int32, bitcast=True), mask=(r < GROUP)[:, None])
    lr = gl.arange(0, ROWS, gl.SliceLayout(1, row_sum))
    view.put(
        lse,
        s * HEADS + g * GROUP + lr,
        split_lse.to(gl.int32, bitcast=True),
        mask=lr < GROUP,
    )


@gluon.jit
def _publish_half(
    mb,
    request,
    rbase,
    part: gl.constexpr,
    lse: gl.constexpr,
    s,
    g,
    first,
    e_max,
    e_sum,
    acc,
    cache,
):
    """Publish a pass's split outputs and LSEs (rows ``rbase .. rbase +
    3`` of the request's verify block), each row's if split
    ``s`` holds keys of it."""
    D: gl.constexpr = schedule.dims(mb.plan)
    HEADS: gl.constexpr = D.q_heads
    KV_HEADS: gl.constexpr = D.kv_heads
    GROUP: gl.constexpr = D.group
    HB: gl.constexpr = head_block(mb.plan)
    MH: gl.constexpr = 4 * HB
    nw: gl.constexpr = gl.num_warps()
    mma: gl.constexpr = _layouts(nw, HB)[0]
    row_sum: gl.constexpr = _layouts(nw, HB)[3]
    L = _row_lengths(cache, request, rbase, gl.SliceLayout(1, mma), mb.plan)
    l_acc = gl.convert_layout(e_sum, gl.SliceLayout(1, mma))
    out = acc / gl.where(l_acc > 0, l_acc, 1.0)[:, None]
    m_l = gl.convert_layout(e_max, gl.SliceLayout(1, row_sum))
    split_lse = gl.where(
        e_sum > 0, m_l + gl.log(gl.where(e_sum > 0, e_sum, 1.0)), float("-inf")
    )
    mr = gl.arange(0, MH, gl.SliceLayout(1, mma))
    c = gl.arange(0, DIM, gl.SliceLayout(0, mma))
    live_r = ((mr % HB) < GROUP) & (first < L)
    at = (
        mb.row_offset(request * RB + rbase + mr // HB)[:, None]
        + ((s * KV_HEADS + g) * GROUP + mr % HB)[:, None] * DIM
        + c[None, :]
    )
    mb.put(part, at, out.to(gl.int32, bitcast=True), mask=live_r[:, None])
    ml = gl.arange(0, MH, gl.SliceLayout(1, row_sum))
    Ll = _row_lengths(cache, request, rbase, gl.SliceLayout(1, row_sum), mb.plan)
    mb.put(
        lse,
        mb.row_offset(request * RB + rbase + ml // HB)
        + s * HEADS
        + g * GROUP
        + ml % HB,
        split_lse.to(gl.int32, bitcast=True),
        mask=((ml % HB) < GROUP) & (first < Ll),
    )


@gluon.jit
def shared_attention_stage(
    mb,
    task,
    rounds,
    cache,
    q: gl.constexpr,
    kv: gl.constexpr,
    rel: gl.constexpr,
    part: gl.constexpr,
    lse: gl.constexpr,
    SPLITS: gl.constexpr,
    PASSES: gl.constexpr,
    STAGE: gl.constexpr,
):
    """:func:`full_attention_stage` with a verify block's 8 rows a task:
    task (request, KV head ``g``, split ``s``) streams the split's keys and
    values once for its 8 rows. Its tile rows are the rows' (row, head)
    pairs, each with its row's length mask, relative logits and running
    sums, every element's MFMA chain and row sum as in the regular kernel's
    program of its row. A task whose rows split differently (a row length
    crossing a split-length step) runs each row alone (:func:`_row_alone`).
    With one pass (``PASSES``) a task takes half a block's rows: twice the
    tasks for few requests, each block's keys read twice.

    The keys stream through two LDS tiles (a block's issue before the
    previous block computes), the values through one (a block's issue after
    the previous block's value product): the tile rows' FP32
    probabilities take the LDS of a second value tile."""
    D: gl.constexpr = schedule.dims(mb.plan)
    HEADS: gl.constexpr = D.q_heads
    KV_HEADS: gl.constexpr = D.kv_heads
    GROUP: gl.constexpr = D.group
    HB: gl.constexpr = head_block(mb.plan)
    MT: gl.constexpr = RB * HB
    MH: gl.constexpr = 4 * HB
    mb.stamp(STAGE, trace.START)
    G: gl.constexpr = schedule.grid(mb.plan)
    nw: gl.constexpr = gl.num_warps()
    mma: gl.constexpr = _layouts(nw, HB)[0]
    tile: gl.constexpr = _layouts(nw, HB)[2]
    row_sum: gl.constexpr = _layouts(nw, HB)[3]
    kv_tile: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 4, [1, 0])
    k_bufs = gl.allocate_shared_memory(gl.bfloat16, [2, BN, DIM], kv_tile)
    v_buf = gl.allocate_shared_memory(gl.bfloat16, [BN, DIM], kv_tile)
    flat: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    probs = gl.allocate_shared_memory(gl.float32, [MH, BN], flat)
    q_stage = gl.allocate_shared_memory(gl.bfloat16, [MT, DIM], tile)
    line: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    tok_k = gl.allocate_shared_memory(gl.bfloat16, [RB * DIM], line)
    tok_v = gl.allocate_shared_memory(gl.bfloat16, [RB * DIM], line)
    if rounds > 0:
        kv_load: gl.constexpr = _layouts(nw, HB)[1]
        v_buf.store(gl.zeros([BN, DIM], gl.bfloat16, kv_load))
        gl.barrier()
        _block_issue(
            cache, k_bufs.index(0), task, 0, True, False, SPLITS, PASSES, mb.plan
        )
        _block_issue(cache, v_buf, task, 0, True, True, SPLITS, PASSES, mb.plan)
        unit = 0
        for i in range(rounds):
            t = task + i * G
            request, g, s, row0, _, Lz, launch0, uniform = _block_task(
                cache, t, SPLITS, PASSES, mb.plan
            )
            view = mb.row(request * RB)
            per = _per(Lz, SPLITS)
            first = s * per
            nb = gl.maximum(gl.minimum(first + per, Lz) - first + BN - 1, 0) // BN
            nxt = gl.where(i + 1 < rounds, t + G, t)
            if uniform & (nb > 0):
                # q of the 8 rows' group heads (the other head slots zero) and the
                # launch's own tokens' k and v.
                ql: gl.constexpr = gl.BlockedLayout([1, 2], [2, 32], [nw, 1], [1, 0])
                mq = gl.arange(0, PASSES * MH, gl.SliceLayout(1, ql))[:, None]
                qp = gl.arange(0, DIM // 2, gl.SliceLayout(0, ql))[None, :]
                hq = gl.minimum(g * GROUP + mq % HB, HEADS - 1)
                tl_: gl.constexpr = gl.BlockedLayout([RB, 1], [1, 64], [1, nw], [1, 0])
                u = gl.arange(0, RB, gl.SliceLayout(1, tl_))[:, None]
                p = gl.arange(0, DIM // 2, gl.SliceLayout(0, tl_))[None, :]
                rows = view.row_offset(u)
                got_q, got_k, got_v = view.poll_many(
                    (q, kv, kv),
                    (
                        view.row_offset(row0 + mq // HB) + hq * (DIM // 2) + qp,
                        rows + g * (DIM // 2) + p,
                        rows + (KV_HEADS + g) * (DIM // 2) + p,
                    ),
                )
                mb.stamp(STAGE, trace.READY)
                tok_k.store(gl.reshape(_bf16_values(got_k).to(gl.bfloat16), [RB * DIM]))
                tok_v.store(gl.reshape(_bf16_values(got_v).to(gl.bfloat16), [RB * DIM]))
                qv = _bf16_values(got_q)
                q_rows = gl.arange(0, PASSES * MH, gl.SliceLayout(1, qv.type.layout))[
                    :, None
                ]
                q_stage.slice(0, PASSES * MH, dim=0).store(
                    gl.where(q_rows % HB < GROUP, qv, 0.0).to(gl.bfloat16)
                )
                e_max = ()
                e_sum = ()
                acc = ()
                for half in gl.static_range(PASSES):
                    e_max += (
                        gl.full(
                            [MH], float("-inf"), gl.float32, gl.SliceLayout(1, mma)
                        ),
                    )
                    e_sum += (gl.zeros([MH], gl.float32, gl.SliceLayout(1, row_sum)),)
                    acc += (gl.zeros([MH, DIM], gl.float32, mma),)
                for b in range(nb):
                    last = b + 1 == nb
                    live = (~last) | (i + 1 < rounds)
                    nt = gl.where(last, nxt, t)
                    nbk = gl.where(last, 0, b + 1)
                    _block_issue(
                        cache,
                        k_bufs.index((unit + 1) % 2),
                        nt,
                        nbk,
                        live,
                        False,
                        SPLITS,
                        PASSES,
                        mb.plan,
                    )
                    e_max, e_sum, acc = _shared_block(
                        mb,
                        request,
                        k_bufs.index(unit % 2),
                        v_buf,
                        probs,
                        q_stage,
                        tok_k,
                        tok_v,
                        g,
                        launch0,
                        first + b * BN,
                        first,
                        per,
                        e_max,
                        e_sum,
                        acc,
                        cache,
                        row0,
                        rel,
                        PASSES,
                    )
                    _block_issue(
                        cache, v_buf, nt, nbk, live, True, SPLITS, PASSES, mb.plan
                    )
                    unit += 1
                # Each row's split output, if the split holds keys of it.
                for half in gl.static_range(PASSES):
                    _publish_half(
                        mb,
                        request,
                        row0 + 4 * half,
                        part,
                        lse,
                        s,
                        g,
                        first,
                        e_max[half],
                        e_sum[half],
                        acc[half],
                        cache,
                    )
            else:
                # The copies issued for this task were empty (no keys, or
                # rows that split differently).
                async_copy.wait_group(0)
                gl.barrier()
                if ~uniform:
                    # Each row by its own split length (its splits may hold
                    # keys where the last row's do not).
                    _row_alone(
                        mb,
                        t,
                        cache,
                        k_bufs.index(unit % 2),
                        v_buf,
                        q,
                        kv,
                        rel,
                        part,
                        lse,
                        SPLITS,
                        PASSES,
                    )
                    async_copy.wait_group(0)
                    gl.barrier()
                live = i + 1 < rounds
                _block_issue(
                    cache,
                    k_bufs.index((unit + 1) % 2),
                    nxt,
                    0,
                    live,
                    False,
                    SPLITS,
                    PASSES,
                    mb.plan,
                )
                _block_issue(cache, v_buf, nxt, 0, live, True, SPLITS, PASSES, mb.plan)
                unit += 1
        async_copy.wait_group(0)
        gl.barrier()
    mb.stamp(STAGE, trace.END)
