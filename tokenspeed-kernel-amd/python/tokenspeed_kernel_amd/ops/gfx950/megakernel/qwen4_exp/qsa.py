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

"""QSA full-attention decode stages (TokenSpeed ``Qwen4ExpAttentionDecoderLayer``
with its ``QSAIndexer``, one token at position ``L`` per row).

Each Q1 task serves every row of the launch, so Q1 reads each weight once.
Q2-Q6 run one row per task. There, ``mb`` is the view of that row, and the
per-step operands (``seq_len``, ``positions``, page tables) are that row's:
the positions and the staging have one entry a row, the lengths and the
page tables one a request (a row's length is its request's, less the tokens
of the request that follow the row's).

* projection (Q1), 218 tasks of 64 rows: ``qkv_proj`` (``[q | gate]`` per
  head, then k and v) and ``index_qk_proj`` (4 index q heads, then the index
  key), BF16 weights (320 KiB a task) on the BF16 mixed row, BF16 out.
* prologue (Q2), 31 tasks:
  - 24 q heads and 2 k heads: Gemma RMS norm (``1 + w`` in FP32), then NeoX
    MRoPE on dims 0-63 of the FP32 normed values. A k task also writes the
    rotated k and the raw v to the KV cache slot of position ``L``.
  - 4 index q heads: Gemma RMS norm with the BF16 ``1 + w``, then MRoPE.
  - the index key: writes the raw key to its recent ring slot (and the
    group's rope position at a group start). When ``L`` completes a 4-token
    block, the task mean-pools the raw keys of the block. Then it normalizes
    the result, rotates it at the group start and stores the compressed key.
* block scores (Q3), 256 tasks: ``sum_h relu(q_h . k_b)`` of every complete
  block, only when there are more than ``TOPK_BLOCKS``.
* top-k (Q4), ``SELECT_CHUNKS`` tasks: each selects the ``TOPK_BLOCKS`` best
  blocks of its chunk of ``SELECT_CHUNK`` blocks. The task polls the scores
  once into LDS and runs four 8-bit radix passes there. A tie goes to the
  lower block. With one chunk, this is the selection. Else the candidates
  of each chunk go to top-k final (Q4F), 1 task, which selects the best of
  them. In both cases the ids land in ascending order (TokenSpeed's
  selection). A block of the global top-k is in the top-k of its chunk, so
  the two levels select exactly.
* attention (Q5), ``KV_HEADS * SPLITS`` tasks: the attended tokens are the
  tokens of the selected blocks, then the tokens of the partial block. Q5
  splits them as TokenSpeed splits them (16-token tiles over 32 splits).
  Each task publishes its FP32 ``acc / l`` and base-2 ``lse`` for the 12 q
  heads of its KV head.
* merge (Q6), 48 tasks of 128 dims of a head: LSE merge, BF16, then
  ``bf16(a * sigmoid(gate))`` published as BF16 for ``o_proj``.

The current token's k and v come from mailboxes, not from the cache. The
cache write of another CTA in the same launch need not be visible yet.

A launch can verify ``Plan.chain`` tokens per request (TokenSpeed's MTP
verify). Each row is one token, at its own ``seq_len``. Every value that a
token of this launch produced comes from the row of that token:

* the members and the first position of a compression block (Q2),
* the compressed key of a block that an earlier token completed (Q3),
* the k / v of the earlier tokens of the request (Q5).

The launch writes the KV and compressed caches as in decode. The index key
goes to the verify staging of TokenSpeed instead of the recent ring.
TokenSpeed fills the recent ring for the accepted tokens.

An MTP draft step (one row per request) stages its index key in the QSA
indexer's draft scratch instead of the ring, tagged with its position. A
block member that an earlier draft step staged comes from there.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import aggregate, gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.megakernel import bf16, mx, schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.megakernel.plan import Stage, placement
from tokenspeed_kernel_amd.ops.gfx950.megakernel.qwen4_exp.dims import EPS, HIDDEN

HEADS = gl.constexpr(24)
KV_HEADS = gl.constexpr(2)
GROUP = gl.constexpr(12)  # q heads per KV head
HEAD = gl.constexpr(256)
Q_GATE = gl.constexpr(12288)  # [q | gate] per head
K_OFF = gl.constexpr(12288)
V_OFF = gl.constexpr(12800)
QKV_N = gl.constexpr(13312)
IDX_HEADS = gl.constexpr(4)
IDX_DIM = gl.constexpr(128)
IDX_Q_OFF = gl.constexpr(13312)
IDX_K_OFF = gl.constexpr(13824)
PROJ_N = gl.constexpr(13952)
ROWS = gl.constexpr(64)  # projection rows of a Q1 task
QKV_TASKS = gl.constexpr(208)  # QKV_N / ROWS
PROJ_TASKS = gl.constexpr(218)  # PROJ_N / ROWS
PROLOGUE_TASKS = gl.constexpr(31)  # 24 q + 2 k + 4 index q + index key
ROT_HALF = gl.constexpr(32)  # NeoX pairs (i, i + 32) over the 64 rotary dims
COMPRESS = gl.constexpr(4)
TOPK_BLOCKS = gl.constexpr(512)
MAX_BLOCKS = gl.constexpr(65536)  # 262144-token context / COMPRESS
SELECT_CHUNK = gl.constexpr(8192)  # blocks a Q4 task selects from
SELECT_CHUNKS = gl.constexpr(8)  # Q4 tasks per row: MAX_BLOCKS / SELECT_CHUNK
SELECT_TILE = gl.constexpr(4096)  # keys per pass of the select loops
HIST_COPIES = gl.constexpr(16)  # radix histogram copies (see _radix_select)
SCORE_TASKS = gl.constexpr(255)  # every CTA but the one of Q4F
SCORE_CHUNK = gl.constexpr(128)  # blocks a Q3 task scores per pass
SPLITS = gl.constexpr(32)
TILE = gl.constexpr(16)
SPLIT_TOKENS = gl.constexpr(128)  # bound of one split's tokens (5 tiles)
ATTN_TASKS = gl.constexpr(64)  # KV_HEADS * SPLITS
MERGE_TASKS = gl.constexpr(48)  # HEADS * 2 halves of 128 dims
QK_SCALE = gl.constexpr(0.09016844005556021)  # HEAD ** -0.5 * log2(e)

REGIONS = (
    ("qp", 6976),  # BF16 pairs of the projections
    ("qn", 3072),  # normed, rotated q (BF16 pairs)
    ("kn", 256),  # normed, rotated k of the current token (BF16 pairs)
    ("iq", 256),  # normed, rotated index q (BF16 pairs)
    ("ck", 64),  # compressed key the current token completes (BF16 pairs)
    ("isc", 65536),  # FP32 block scores
    ("csc", 4096),  # Q4 candidates' ordered score keys [chunk, TOPK_BLOCKS]
    ("cid", 4096),  # their block ids (-1: unused)
    ("sel", 512),  # selected block ids
    ("part", 196608),  # FP32 split outputs [split, kv head, head, dim]
    ("lse", 768),  # FP32 split base-2 LSE [split, head]
    ("obf", 3072),  # BF16 pairs of the gated attention output
)


def stages(
    bases: dict[str, int], rows: int, attended: int, X: str
) -> tuple[Stage, ...]:
    """Q1-Q6, placed by ``bases``. Q1 and Q2 run for ``rows`` rows, Q3-Q6
    (the attention) for ``attended`` rows: ``rows``, or one row per request
    in an MTP draft catch-up step. ``X`` is the region prefix of the mixed
    row (the row of the attention mixer)."""
    q1, q2, q3, q4, q4f, q5, q6 = placement(
        bases, ("q1", "q2", "q3", "q4", "q4f", "q5", "q6")
    )
    return (
        Stage("q1", int(PROJ_TASKS), q1, reads=(X + "xbf",), writes=("qp",)),
        Stage(
            "q2",
            int(PROLOGUE_TASKS) * rows,
            q2,
            reads=("qp",),
            writes=("qn", "kn", "iq", "ck"),
        ),
        Stage(
            "q3", int(SCORE_TASKS) * attended, q3, reads=("iq", "ck"), writes=("isc",)
        ),
        Stage(
            "q4",
            int(SELECT_CHUNKS) * attended,
            q4,
            reads=("isc",),
            writes=("csc", "cid", "sel"),
        ),
        Stage("q4f", attended, q4f, reads=("csc", "cid"), writes=("sel",)),
        Stage(
            "q5",
            int(ATTN_TASKS) * attended,
            q5,
            reads=("sel", "qn", "kn", "qp"),
            writes=("part", "lse"),
        ),
        Stage(
            "q6",
            int(MERGE_TASKS) * attended,
            q6,
            reads=("part", "lse", "qp"),
            writes=("obf",),
        ),
    )


@aggregate
class Weights:
    """A QSA attention layer's weights (``weights.QsaLayerWeights``)."""

    qkv_w: gl.tensor  # qkv_proj, BF16 [QKV_N, HIDDEN] (weights.bf16_tiles)
    idx_w: gl.tensor  # index_qk_proj, BF16 [PROJ_N - QKV_N, HIDDEN]
    q_norm_w: gl.tensor  # Gemma weight [HEAD] (raw w)
    k_norm_w: gl.tensor  # Gemma weight [HEAD] (raw w)
    iq_norm_w: gl.tensor  # the indexer's BF16 1 + w [IDX_DIM]
    ik_norm_w: gl.tensor  # the indexer's BF16 1 + w [IDX_DIM]


@aggregate
class Cache:
    """A launch's attention caches and per-row operands, seen from one row
    (:meth:`row`): ``positions`` and the staging rows point at that row's
    entry; ``seq_len``, the page tables and ``accepted`` at its request's
    (row ``r`` is token ``r % chain`` of request ``r // chain``)."""

    cos_sin: gl.tensor  # FP32 [max_position, 64] ([cos(32) | sin(32)])
    # int64 position of axis 0 of the row. Axis a is a * pos_stride later.
    positions: gl.tensor
    pos_stride: gl.tensor | gl.constexpr
    # int32: the request's L + 1 for its last token at L (:meth:`length`).
    seq_len: gl.tensor
    full_table: gl.tensor  # the row's KV page table
    full_stride: gl.tensor | gl.constexpr
    page_size: gl.tensor | gl.constexpr
    k_cache: gl.tensor  # [slots, KV_HEADS, HEAD], kv_stride elements a slot
    v_cache: gl.tensor
    kv_stride: gl.tensor | gl.constexpr
    qsa_table: gl.tensor  # compressed-key page table (256 tokens a page)
    qsa_stride: gl.tensor | gl.constexpr
    recent_table: gl.tensor  # recent-ring page table (64 tokens a page)
    recent_stride: gl.tensor | gl.constexpr
    # Compressed keys, 64 rows of IDX_DIM a page, compressed_stride a page.
    compressed: gl.tensor
    compressed_stride: gl.tensor | gl.constexpr
    # Raw index keys, 4 ring slots of IDX_DIM a page, raw_stride a page.
    raw_key: gl.tensor
    raw_stride: gl.tensor | gl.constexpr
    # int64 [3] group-start positions a page, rope_stride a page.
    rope_position: gl.tensor
    rope_stride: gl.tensor | gl.constexpr
    # None in decode (the index keys go to the recent ring). In a verify, the
    # row of TokenSpeed's QSAVerifyState that the launch writes instead of the
    # ring: raw key [IDX_DIM], positions [3], logical position, recent location.
    staged_k: gl.tensor | gl.constexpr
    staged_positions: gl.tensor | gl.constexpr
    staged_logical: gl.tensor | gl.constexpr
    staged_recent: gl.tensor | gl.constexpr
    # None, or in an MTP draft step (one row per request) the request's draft
    # scratch that the launch writes instead of the ring: raw keys [COMPRESS,
    # IDX_DIM] by position % COMPRESS, int64 group-start positions [3] and
    # int64 position tags [COMPRESS] (TokenSpeed's QSA draft staging).
    draft_k: gl.tensor | gl.constexpr
    draft_positions: gl.tensor | gl.constexpr
    draft_logical: gl.tensor | gl.constexpr
    # None, or in an MTP draft catch-up step (the draft's first step, over the
    # verified rows) the request's int32 accepted length: a row at or past it
    # is a rejected token. It writes its K / V, but no index or compressed key.
    accepted: gl.tensor | gl.constexpr
    # What the row adds to its request's seq_len: t - chain + 1 for token t
    # (0 before :meth:`row`).
    seq_offset: gl.tensor | gl.constexpr

    @gluon.jit
    def length(self):
        """``L + 1`` for the row's token at ``L``."""
        return gl.load(self.seq_len) + self.seq_offset

    @gluon.jit
    def row(self, r, CHAIN: gl.constexpr):
        """This cache as seen by row ``r`` (token ``r % CHAIN`` of request
        ``r // CHAIN``). The view of a row is a cache of one request, whose
        row 0 with ``CHAIN`` 1 is that row again."""
        request = r // CHAIN
        staged_k = self.staged_k
        staged_positions = self.staged_positions
        staged_logical = self.staged_logical
        staged_recent = self.staged_recent
        if self.staged_k is not None:
            staged_k = self.staged_k + r * IDX_DIM
            staged_positions = self.staged_positions + r * 3
            staged_logical = self.staged_logical + r
            staged_recent = self.staged_recent + r
        draft_k = self.draft_k
        draft_positions = self.draft_positions
        draft_logical = self.draft_logical
        if self.draft_k is not None:
            draft_k = self.draft_k + r * (COMPRESS * IDX_DIM)
            draft_positions = self.draft_positions + r * 3
            draft_logical = self.draft_logical + r * COMPRESS
        accepted = self.accepted
        if self.accepted is not None:
            accepted = self.accepted + request
        return Cache(
            self.cos_sin,
            self.positions + r,
            self.pos_stride,
            self.seq_len + request,
            self.full_table + request * self.full_stride,
            self.full_stride,
            self.page_size,
            self.k_cache,
            self.v_cache,
            self.kv_stride,
            self.qsa_table + request * self.qsa_stride,
            self.qsa_stride,
            self.recent_table + request * self.recent_stride,
            self.recent_stride,
            self.compressed,
            self.compressed_stride,
            self.raw_key,
            self.raw_stride,
            self.rope_position,
            self.rope_stride,
            staged_k,
            staged_positions,
            staged_logical,
            staged_recent,
            draft_k,
            draft_positions,
            draft_logical,
            accepted,
            self.seq_offset + r % CHAIN - CHAIN + 1,
        )


@gluon.jit
def projection_stage(mb, task, weights, X: gl.constexpr, STAGE: gl.constexpr):
    """Q1 task ``task``: 64 projection rows of ``qkv_proj`` or
    ``index_qk_proj`` (``X``: region prefix of the mixed row)."""
    mb.stamp(STAGE, trace.START)
    tiles = _projection_issue(task, weights)
    gl.amd.hint.sched_barrier()
    _projection_finish(mb, task, tiles, X, STAGE)


@gluon.jit
def _projection_issue(task, weights):
    """Issue Q1 task ``task``'s weight loads: 4 16-row blocks, each split
    over two waves along K (320 KiB per task)."""
    is_idx = task >= QKV_TASKS
    w = gl.where(is_idx, weights.idx_w, weights.qkv_w)
    nb0 = gl.where(is_idx, task - QKV_TASKS, task) * (ROWS // 16)
    return bf16.load_weight_tiles(
        w,
        nb0=nb0,
        K=HIDDEN,
        kit0=0,
        KITS=HIDDEN // 32,
        SPLIT=2,
        NW=4,
        GROUP=4,
        GROUP_STRIDE=0,
    )


@gluon.jit
def _projection_finish(mb, task, tiles, X: gl.constexpr, STAGE: gl.constexpr):
    """Q1 task ``task`` after its weight loads: GEMV and publish, for every
    row."""
    R: gl.constexpr = schedule.rows(mb.plan)
    x_smem, _ = bf16.load_rows(mb, X + "xbf", 0, HIDDEN, R)
    mb.stamp(STAGE, trace.READY)
    projs = bf16.gemv(x_smem, tiles, FIRST=0, ROWS=R)
    mb.stamp(STAGE, trace.COMPUTED)
    mx.put_bf16_rows(mb, "qp", task * (ROWS // 2), projs)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def _rope_tables(cos_sin, pos_ptr, pos_stride, d):
    """MRoPE ``(cos, sin)`` of dims ``d`` (interleaved sections [11, 11, 10]).

    Pair ``i = d % 32`` reads the position of axis 1 when ``i % 3 == 1``,
    axis 2 when ``i % 3 == 2`` and ``i < 30``, else axis 0. For text the
    three positions are equal.
    """
    i = d % ROT_HALF
    axis = gl.where(i % 3 == 1, 1, gl.where((i % 3 == 2) & (i < 30), 2, 0))
    pos = gl.load(pos_ptr + axis * pos_stride)
    row = cos_sin + pos * (2 * ROT_HALF)
    return gl.load(row + i), gl.load(row + ROT_HALF + i)


@gluon.jit
def _partner(d):
    """The NeoX partner of rotary dim ``d`` (``d`` itself past dim 63)."""
    return gl.where(
        d < ROT_HALF, d + ROT_HALF, gl.where(d < 2 * ROT_HALF, d - ROT_HALF, d)
    )


@gluon.jit
def _gemma_rotate(x, xp, w1, w1p, cos, sin, d, N: gl.constexpr):
    """TokenSpeed's ``_gqa_prologue``: ``(x * rsqrt(mean(x^2) + eps)) * (1 + w)``
    in FP32, then NeoX rotation of the FP32 normed values (FMA form)."""
    inv = gl.rsqrt(gl.sum(x * x, axis=0) * (1.0 / N) + EPS)
    n = x * inv * w1
    np_ = xp * inv * w1p
    lo = gl.fma(n, cos, -np_ * sin)
    hi = gl.fma(n, cos, np_ * sin)
    return gl.where(d < ROT_HALF, lo, gl.where(d < 2 * ROT_HALF, hi, n))


@gluon.jit
def _index_norm_rotate(x, xp, w, wp, cos, sin, d):
    """TokenSpeed's indexer norm and rope: ``1 / sqrt(s / 128 + eps)`` with
    ``s`` summed in three parts (dims 0-31, 32-63, 64-127), the BF16
    ``1 + w``, then the plain rotation."""
    s1 = gl.sum(gl.where(d < ROT_HALF, x * x, 0.0), axis=0)
    s2 = gl.sum(gl.where((d >= ROT_HALF) & (d < 2 * ROT_HALF), x * x, 0.0), axis=0)
    s3 = gl.sum(gl.where(d >= 2 * ROT_HALF, x * x, 0.0), axis=0)
    scale = 1.0 / gl.sqrt((s1 + s2 + s3) / IDX_DIM + EPS)
    n = x * scale * w
    np_ = xp * scale * wp
    lo = n * cos - np_ * sin
    hi = n * cos + np_ * sin
    return gl.where(d < ROT_HALF, lo, gl.where(d < 2 * ROT_HALF, hi, n))


@gluon.jit
def prologue_stage(mb, task, tok, weights, cache, STAGE: gl.constexpr):
    """Q2 task ``task`` of a row that is token ``tok`` of its request (see
    the module docstring). ``mb`` and ``cache`` are the views of the row."""
    mb.stamp(STAGE, trace.START)
    L = cache.length() - 1
    if task < HEADS + KV_HEADS:
        _qk_head(mb, task, L, weights, cache, STAGE)
    elif task < HEADS + KV_HEADS + IDX_HEADS:
        _index_q_head(mb, task - HEADS - KV_HEADS, weights, cache, STAGE)
    else:
        _index_key(mb, L, tok, weights, cache, STAGE)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def _qk_head(mb, task, L, weights, cache, STAGE: gl.constexpr):
    """A q head (``task < HEADS``) or a k head, with the KV cache write."""
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    d = gl.arange(0, HEAD, lanes)
    dp = _partner(d)
    is_k = task >= HEADS
    base = gl.where(is_k, K_OFF + (task - HEADS) * HEAD, task * 2 * HEAD)
    norm_w = gl.where(is_k, weights.k_norm_w, weights.q_norm_w)
    w1 = gl.load(norm_w + d).to(gl.float32) + 1.0
    w1p = gl.load(norm_w + dp).to(gl.float32) + 1.0
    cos, sin = _rope_tables(cache.cos_sin, cache.positions, cache.pos_stride, d)
    gl.amd.hint.sched_barrier()
    vbase = V_OFF + gl.maximum(task - HEADS, 0) * HEAD
    got, gotp, got_v = mb.poll_many(
        ("qp", "qp", "qp"), ((base + d) // 2, (base + dp) // 2, (vbase + d) // 2)
    )
    mb.stamp(STAGE, trace.READY)
    x = mx.bf16_half(got, base + d)
    xp = mx.bf16_half(gotp, base + dp)
    o = _gemma_rotate(x, xp, w1, w1p, cos, sin, d, HEAD).to(gl.bfloat16)
    mb.stamp(STAGE, trace.COMPUTED)
    if is_k:
        kh = task - HEADS
        mx.put_bf16(mb, "kn", kh * (HEAD // 2), o.to(gl.float32))
        page_size = cache.page_size
        page = gl.load(cache.full_table + L // page_size)
        loc = gl.where(page > 0, page * page_size + L % page_size, 0).to(gl.int64)
        gl.store(cache.k_cache + loc * cache.kv_stride + kh * HEAD + d, o)
        v = mx.bf16_half(got_v, vbase + d)
        gl.store(
            cache.v_cache + loc * cache.kv_stride + kh * HEAD + d, v.to(gl.bfloat16)
        )
    else:
        mx.put_bf16(mb, "qn", task * (HEAD // 2), o.to(gl.float32))


@gluon.jit
def _index_q_head(mb, j, weights, cache, STAGE: gl.constexpr):
    """Index q head ``j``."""
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    d = gl.arange(0, IDX_DIM, lanes)
    dp = _partner(d)
    w = gl.load(weights.iq_norm_w + d).to(gl.float32)
    wp = gl.load(weights.iq_norm_w + dp).to(gl.float32)
    cos, sin = _rope_tables(cache.cos_sin, cache.positions, cache.pos_stride, d)
    gl.amd.hint.sched_barrier()
    base = IDX_Q_OFF + j * IDX_DIM
    got, gotp = mb.poll_many(("qp", "qp"), ((base + d) // 2, (base + dp) // 2))
    mb.stamp(STAGE, trace.READY)
    x = mx.bf16_half(got, base + d)
    xp = mx.bf16_half(gotp, base + dp)
    o = _index_norm_rotate(x, xp, w, wp, cos, sin, d)
    mb.stamp(STAGE, trace.COMPUTED)
    mx.put_bf16(mb, "iq", j * (IDX_DIM // 2), o)


@gluon.jit
def _index_key(mb, L, tok, weights, cache, STAGE: gl.constexpr):
    """The index key: compression, then the recent ring, the verify staging
    or the draft scratch.

    A block member (and the first position of the block) at or after the
    first token of the request in the launch, ``L - tok``, comes from the row
    of that token (this launch). Earlier members come from the ring, which
    earlier steps wrote. In a draft step, a member that an earlier draft step
    staged comes from the draft scratch, where its position tag matches.
    """
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    d = gl.arange(0, IDX_DIM, lanes)
    dp = _partner(d)
    positions = cache.positions
    pos_stride = cache.pos_stride
    rpage = gl.load(cache.recent_table + L // 64)
    ring = gl.where(rpage > 0, rpage, 0).to(gl.int64)
    recent_ok = rpage > 0
    qpage = gl.load(cache.qsa_table + L // 256)
    qsa_ok = qpage > 0
    completes = (L + 1) % COMPRESS == 0
    # Members L-3 .. L-1 of the block from the ring (earlier steps).
    ring_row = cache.raw_key + ring * cache.raw_stride
    k0 = gl.load(ring_row + ((L + 1) % COMPRESS) * IDX_DIM + d).to(gl.float32)
    k1 = gl.load(ring_row + ((L + 2) % COMPRESS) * IDX_DIM + d).to(gl.float32)
    k2 = gl.load(ring_row + ((L + 3) % COMPRESS) * IDX_DIM + d).to(gl.float32)
    k0p = gl.load(ring_row + ((L + 1) % COMPRESS) * IDX_DIM + dp).to(gl.float32)
    k1p = gl.load(ring_row + ((L + 2) % COMPRESS) * IDX_DIM + dp).to(gl.float32)
    k2p = gl.load(ring_row + ((L + 3) % COMPRESS) * IDX_DIM + dp).to(gl.float32)
    w = gl.load(weights.ik_norm_w + d).to(gl.float32)
    wp = gl.load(weights.ik_norm_w + dp).to(gl.float32)
    # The rope position of the block is the position of its first token. It
    # comes from the ring page (written at L - 3) or from the positions of the
    # row of that token in this launch.
    head_here = tok >= COMPRESS - 1
    start = gl.where(
        head_here,
        positions - (COMPRESS - 1),
        cache.rope_position + ring * cache.rope_stride,
    )
    if cache.draft_k is not None:
        k0, k0p = _draft_member(cache, L - 3, k0, k0p, d, dp)
        k1, k1p = _draft_member(cache, L - 2, k1, k1p, d, dp)
        k2, k2p = _draft_member(cache, L - 1, k2, k2p, d, dp)
        head_tag = gl.load(cache.draft_logical + (L + 1) % COMPRESS)
        start = gl.where(
            (L >= COMPRESS - 1) & (head_tag == L - (COMPRESS - 1)),
            cache.draft_positions,
            start,
        )
    cos, sin = _rope_tables(cache.cos_sin, start, gl.where(head_here, pos_stride, 1), d)
    gl.amd.hint.sched_barrier()
    # The raw keys of the members that can be rows of this launch, the last
    # min(chain, COMPRESS) members. Member L - M + 1 + m is at row offset
    # m - M + 1. A member before the launch reads the current row.
    M: gl.constexpr = _launch_members(schedule.chain(mb.plan))
    ml: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [1, gl.num_warps()], [1, 0])
    m = gl.arange(0, M, gl.SliceLayout(1, ml))[:, None]
    dm = gl.arange(0, IDX_DIM, gl.SliceLayout(0, ml))[None, :]
    back = gl.where(m - (M - 1) >= -tok, m - (M - 1), 0)
    got, gotp = mb.poll_many(
        ("qp", "qp"),
        (
            mb.row_offset(back) + (IDX_K_OFF + dm) // 2,
            mb.row_offset(back) + (IDX_K_OFF + _partner(dm)) // 2,
        ),
    )
    mb.stamp(STAGE, trace.READY)
    here = mx.bf16_half(got, IDX_K_OFF + dm)
    here_p = mx.bf16_half(gotp, IDX_K_OFF + _partner(dm))
    k3 = _member(here, m, M - 1, lanes)
    k3p = _member(here_p, m, M - 1, lanes)
    if M > 1:
        k2 = gl.where(tok >= 1, _member(here, m, M - 2, lanes), k2)
        k2p = gl.where(tok >= 1, _member(here_p, m, M - 2, lanes), k2p)
    if M > 2:
        k1 = gl.where(tok >= 2, _member(here, m, M - 3, lanes), k1)
        k1p = gl.where(tok >= 2, _member(here_p, m, M - 3, lanes), k1p)
    if M > 3:
        k0 = gl.where(tok >= 3, _member(here, m, M - 4, lanes), k0)
        k0p = gl.where(tok >= 3, _member(here_p, m, M - 4, lanes), k0p)
    pooled = (((k0 + k1) + k2) + k3) / 4.0
    pooled_p = (((k0p + k1p) + k2p) + k3p) / 4.0
    ck = _index_norm_rotate(pooled, pooled_p, w, wp, cos, sin, d).to(gl.bfloat16)
    keep = L >= 0
    if cache.accepted is not None:
        keep = L < gl.load(cache.accepted)
    write = completes & qsa_ok & recent_ok & keep
    mb.stamp(STAGE, trace.COMPUTED)
    mx.put_bf16(mb, "ck", 0, gl.where(write, ck.to(gl.float32), 0.0))
    qsa_loc = qpage * 256 + L % 256
    row = (qsa_loc // 256) * 64 + (qsa_loc % 256) // COMPRESS
    dst = (
        cache.compressed
        + (row // 64).to(gl.int64) * cache.compressed_stride
        + (row % 64) * IDX_DIM
    )
    gl.store(dst + d, ck, mask=(d >= 0) & write)
    a = gl.arange(0, 4, lanes)
    pos = gl.load(positions + gl.minimum(a, 2) * pos_stride)
    if cache.draft_k is not None:
        # A draft step writes no ring: its keys are speculative. The scratch
        # keeps them for the next draft step's compression.
        slot = L % COMPRESS
        gl.store(cache.draft_k + slot * IDX_DIM + d, k3.to(gl.bfloat16), mask=recent_ok)
        gl.store(
            cache.draft_logical + slot + a,
            (L + a * 0).to(gl.int64),
            mask=(a < 1) & recent_ok,
        )
        gl.store(
            cache.draft_positions + a,
            pos,
            mask=(a < 3) & (slot == 0) & recent_ok,
        )
    elif cache.staged_k is None:
        gl.store(ring_row + (L % COMPRESS) * IDX_DIM + d, k3.to(gl.bfloat16), mask=keep)
        # A block's first token writes the block's rope position to the ring
        # page. (``start`` can point at a row of this launch instead.)
        header = cache.rope_position + ring * cache.rope_stride
        gl.store(header + a, pos, mask=(a < 3) & (L % COMPRESS == 0) & keep)
    else:
        # A verify writes no ring: TokenSpeed commits the accepted tokens'
        # staged keys after acceptance.
        gl.store(cache.staged_k + d, k3.to(gl.bfloat16))
        gl.store(cache.staged_positions + a, pos, mask=a < 3)
        gl.store(cache.staged_logical + a, (L + a * 0).to(gl.int64), mask=a < 1)
        recent_loc = gl.where(recent_ok, rpage * 64 + L % 64, 0)
        gl.store(cache.staged_recent + a, (recent_loc + a * 0).to(gl.int32), mask=a < 1)


@gluon.jit
def _draft_member(cache, e, k, kp, d, dp):
    """Block member at position ``e`` (and its NeoX partners): the draft
    scratch's key when its tag is ``e``, else ``k`` / ``kp`` (the ring's)."""
    slot = (e + COMPRESS) % COMPRESS
    tagged = (e >= 0) & (gl.load(cache.draft_logical + slot) == e)
    row = cache.draft_k + slot * IDX_DIM
    k = gl.where(tagged, gl.load(row + d).to(gl.float32), k)
    kp = gl.where(tagged, gl.load(row + dp).to(gl.float32), kp)
    return k, kp


@gluon.constexpr_function
def _launch_members(chain):
    """Block members that can be rows of this launch, a power of two."""
    return 1 << (min(chain, COMPRESS) - 1).bit_length()


@gluon.jit
def _member(x, m, M: gl.constexpr, layout: gl.constexpr):
    """Row ``M`` of ``x [COMPRESS, IDX_DIM]`` in ``layout``."""
    return gl.convert_layout(gl.sum(gl.where(m == M, x, 0.0), axis=0), layout)


@gluon.constexpr_function
def _score_layout(num_warps):
    """``[SCORE_CHUNK blocks, IDX_DIM]`` compressed keys, 8 dims per lane."""
    return gl.BlockedLayout([1, 8], [4, 16], [num_warps, 1], [1, 0])


@gluon.jit
def score_stage(mb, task, tok, cache, STAGE: gl.constexpr):
    """Q3 task ``task`` of a row that is token ``tok`` of its request: scores
    of a contiguous run of complete blocks.

    The run's first ``SCORE_CHUNK`` compressed keys need no mailbox input,
    so they load before the task polls the index query.
    """
    mb.stamp(STAGE, trace.START)
    L = cache.length() - 1
    complete = (L + 1) // COMPRESS
    if complete > TOPK_BLOCKS:
        per = (complete + SCORE_TASKS - 1) // SCORE_TASKS
        first = task * per
        last = gl.minimum(complete, first + per)
        kl: gl.constexpr = _score_layout(gl.num_warps())
        d = gl.arange(0, IDX_DIM, gl.SliceLayout(0, kl))
        h = gl.arange(0, IDX_HEADS, gl.SliceLayout(1, kl))
        b, k = _score_keys(first, last, cache)
        gl.amd.hint.sched_barrier()
        # The tokens of this launch complete at most one block: a request has
        # at most COMPRESS tokens in a launch. The task reads that block from
        # the mailbox of the token that completes it. The cache write of that
        # block need not be visible yet.
        end = L - (L + 1) % COMPRESS
        here = end >= L - tok
        back = gl.where(here, end - L, 0)
        qw, cw = mb.poll_many(
            ("iq", "ck"),
            ((h[:, None] * IDX_DIM + d[None, :]) // 2, mb.row_offset(back) + d // 2),
        )
        q = mx.bf16_half(qw, d[None, :])
        ck = mx.bf16_half(cw, d)
        mb.stamp(STAGE, trace.READY)
        fresh = gl.where(here, (end + 1) // COMPRESS - 1, -1)
        _score_put(mb, b, last, k, q, ck, fresh)
        for c in range(first + SCORE_CHUNK, last, SCORE_CHUNK):
            b, k = _score_keys(c, last, cache)
            _score_put(mb, b, last, k, q, ck, fresh)
        mb.stamp(STAGE, trace.COMPUTED)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def _score_keys(c, last, cache):
    """Blocks ``c .. c + SCORE_CHUNK - 1`` (below ``last``) and their FP32
    compressed keys."""
    kl: gl.constexpr = _score_layout(gl.num_warps())
    b = c + gl.arange(0, SCORE_CHUNK, gl.SliceLayout(1, kl))
    d = gl.arange(0, IDX_DIM, gl.SliceLayout(0, kl))
    ok = b < last
    page = gl.load(cache.qsa_table + b // 64, mask=ok, other=0)
    src = (
        cache.compressed
        + page.to(gl.int64) * cache.compressed_stride
        + (b % 64) * IDX_DIM
    )
    k = gl.load(src[:, None] + d[None, :], mask=ok[:, None], other=0.0)
    return b, k.to(gl.float32)


@gluon.jit
def _score_put(mb, b, last, k, q, ck, fresh):
    """Publish ``sum_h relu(q_h . k_b)`` of blocks ``b`` (below ``last``).
    Block ``fresh`` takes the mailbox key ``ck``."""
    kl: gl.constexpr = _score_layout(gl.num_warps())
    h = gl.arange(0, IDX_HEADS, gl.SliceLayout(1, kl))
    k = gl.where((b == fresh)[:, None], ck[None, :], k)
    score = gl.zeros([SCORE_CHUNK], gl.float32, gl.SliceLayout(1, kl))
    for hh in gl.static_range(IDX_HEADS):
        qh = gl.sum(gl.where(h[:, None] == hh, q, 0.0), axis=0)
        score += gl.maximum(gl.sum(k * qh[None, :], axis=1), 0.0)
    mb.put("isc", b, score.to(gl.int32, bitcast=True), mask=b < last)


@gluon.constexpr_function
def _select_layouts(num_warps):
    """(layout of ``[SELECT_TILE]`` keys with 8 per lane, layout of ``[256]``
    histogram bins)."""
    return (
        gl.BlockedLayout([8], [64], [num_warps], [0]),
        gl.BlockedLayout([1], [64], [num_warps], [0]),
    )


@gluon.jit
def _ordered_key(values):
    # Monotonic map from FP32 to uint32: larger floats give larger keys.
    bits = values.to(gl.uint32, bitcast=True)
    return bits ^ gl.where((bits >> 31) != 0, 0xFFFFFFFF, 0x80000000).to(gl.uint32)


@gluon.jit
def _radix_select(keys_smem, N: gl.constexpr, count):
    """``(prefix, remaining, bits)`` for the ``TOPK_BLOCKS`` largest of the
    first ``count`` keys in LDS.

    Key 0 marks an unused entry, and used keys are above 0. The function does
    not read the tiles past ``count``. On their high ``bits`` bits, the top
    keys are the keys above ``prefix`` and the first ``remaining`` keys equal
    to it (in index order).

    The function runs up to four 8-bit radix passes over LDS. Each pass has
    its own histogram. The function zeroes all histograms first, so no pass
    waits to clear one. The passes stop when the top keys include every key
    that is left in the chosen bin.
    """
    NW: gl.constexpr = gl.num_warps()
    kl: gl.constexpr = _select_layouts(NW)[0]
    bl: gl.constexpr = _select_layouts(NW)[1]
    # HIST_COPIES copies of each bin, picked by lane. The top digits of the
    # scores are mostly equal. With one copy, the atomics of a wave would
    # serialize on one bin (64-way).
    C: gl.constexpr = HIST_COPIES
    copies: gl.constexpr = gl.BlockedLayout([1, C], [64, 1], [NW, 1], [1, 0])
    cols = gl.arange(0, SELECT_TILE, kl)
    bins = gl.arange(0, 256, bl)
    copy = (cols // 8) % C
    hist_smem = gl.allocate_shared_memory(
        gl.int32, [4 * 256, C], gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    )
    flat_hist = hist_smem.reshape([4 * 256 * C])
    for start in gl.static_range(0, 4 * 256 * C, SELECT_TILE):
        flat_hist.slice(start, SELECT_TILE).store(gl.zeros([SELECT_TILE], gl.int32, kl))
    gl.barrier()
    prefix = gl.to_tensor(0).to(gl.uint32)
    remaining = gl.to_tensor(TOPK_BLOCKS)
    bits = gl.to_tensor(0)
    done = gl.to_tensor(0)
    for p in gl.static_range(4):
        if done == 0:
            for start in gl.static_range(0, N, SELECT_TILE):
                if start < count:
                    keys = keys_smem.slice(start, SELECT_TILE).load(kl)
                    _count_digits(keys, flat_hist, prefix, p, copy, C)
            gl.barrier()
            counts = gl.convert_layout(
                gl.sum(hist_smem.slice(p * 256, 256).load(copies), axis=1), bl
            )
            # at_least[b]: matching keys with digit >= b. The chosen digit is
            # the highest b with at_least[b] >= remaining. The keys above it
            # are at_least[digit + 1], the largest at_least below remaining.
            # The chosen bin holds at_least[digit] - above keys.
            at_least = gl.cumsum(counts, axis=0, reverse=True)
            enough = at_least >= remaining
            digit_plus, above, through = gl.reduce(
                (
                    gl.where(enough, bins + 1, 0),
                    gl.where(enough, 0, at_least),
                    gl.where(enough, at_least, 1 << 30),
                ),
                0,
                _select_combine,
            )
            prefix = prefix | ((digit_plus - 1).to(gl.uint32) << (24 - 8 * p))
            remaining = remaining - above
            bits = bits * 0 + (8 * p + 8)
            # Done when the rest of the chosen bin is exactly what is left.
            done = (through - above == remaining).to(gl.int32)
    return prefix, remaining, bits


@gluon.jit
def _count_digits(keys, flat_hist, prefix, p: gl.constexpr, copy, C: gl.constexpr):
    """Radix pass ``p``: count the 8-bit digits of the keys that match the
    digits chosen so far into histogram ``p`` (``C`` copies per bin)."""
    match = keys != 0
    if p > 0:
        match = match & ((keys >> (32 - 8 * p)) == (prefix >> (32 - 8 * p)))
    digits = ((keys >> (24 - 8 * p)) & 0xFF).to(gl.int32)
    flat_hist.atomic_scatter_add(
        gl.full_like(digits, 1), (p * 256 + digits) * C + copy, 0, mask=match
    )


@gluon.jit
def _select_combine(a0, a1, a2, b0, b1, b2):
    return gl.maximum(a0, b0), gl.maximum(a1, b1), gl.minimum(a2, b2)


@gluon.jit
def _emit_top(
    mb,
    keys_smem,
    ids_smem,
    N: gl.constexpr,
    count,
    prefix,
    remaining,
    bits,
    KEYS: gl.constexpr,
    IDS: gl.constexpr,
    base,
):
    """Publish the top keys (see :func:`_radix_select`) of the first
    ``count`` in index order: ids to ``IDS[base ..]`` and, unless ``KEYS``
    is "", keys to ``KEYS[base ..]``.

    Returns the number published.
    """
    kl: gl.constexpr = _select_layouts(gl.num_warps())[0]
    threshold = prefix >> (32 - bits)
    written = gl.to_tensor(0)
    equal_seen = gl.to_tensor(0)
    for start in gl.static_range(0, N, SELECT_TILE):
        if start < count:
            keys = keys_smem.slice(start, SELECT_TILE).load(kl)
            valid = keys != 0
            high = keys >> (32 - bits)
            equal = valid & (high == threshold)
            equal_rank = equal_seen + gl.cumsum(equal.to(gl.int32), axis=0) - 1
            take = (valid & (high > threshold)) | (equal & (equal_rank < remaining))
            equal_seen += gl.sum(equal.to(gl.int32), axis=0)
            position = base + written + gl.cumsum(take.to(gl.int32), axis=0) - 1
            ids = ids_smem.slice(start, SELECT_TILE).load(kl)
            mb.put(IDS, position, ids, mask=take)
            if KEYS != "":
                mb.put(KEYS, position, keys.to(gl.int32, bitcast=True), mask=take)
            written += gl.sum(take.to(gl.int32), axis=0)
    return written


@gluon.jit
def topk_stage(mb, task, length, STAGE: gl.constexpr):
    """Q4 task ``task`` = chunk ``c`` of ``SELECT_CHUNK`` blocks: the chunk's
    top ``TOPK_BLOCKS`` blocks (ties: the lower block) as candidates for
    Q4F, or, when the chunk holds every block, the final selection."""
    mb.stamp(STAGE, trace.START)
    n = length // COMPRESS
    first = task * SELECT_CHUNK
    if (n > TOPK_BLOCKS) & (first < n):
        kl: gl.constexpr = _select_layouts(gl.num_warps())[0]
        flat: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
        keys_smem = gl.allocate_shared_memory(gl.uint32, [SELECT_CHUNK], flat)
        ids_smem = gl.allocate_shared_memory(gl.int32, [SELECT_CHUNK], flat)
        cols = gl.arange(0, SELECT_TILE, kl)
        # Poll the scores of the chunk once. The radix passes read LDS. Key 0
        # marks the entries past the last block.
        count = gl.minimum(n - first, SELECT_CHUNK)
        for start in gl.static_range(0, SELECT_CHUNK, SELECT_TILE):
            if start < count:
                b = first + start + cols
                got = mb.poll("isc", gl.minimum(b, n - 1))
                keys = _ordered_key(got.to(gl.float32, bitcast=True))
                keys_smem.slice(start, SELECT_TILE).store(gl.where(b < n, keys, 0))
                ids_smem.slice(start, SELECT_TILE).store(b)
        gl.barrier()
        mb.stamp(STAGE, trace.READY)
        prefix, remaining, bits = _chunk_threshold(keys_smem, count)
        mb.stamp(STAGE, trace.COMPUTED)
        if n <= SELECT_CHUNK:
            _emit_top(
                mb,
                keys_smem,
                ids_smem,
                SELECT_CHUNK,
                count,
                prefix,
                remaining,
                bits,
                "",
                "sel",
                0,
            )
        else:
            _emit_candidates(
                mb, task, keys_smem, ids_smem, count, prefix, remaining, bits
            )
    mb.stamp(STAGE, trace.END)


@gluon.jit
def _chunk_threshold(keys_smem, blocks):
    """:func:`_radix_select` of a chunk, or (take all) when it holds at most
    ``TOPK_BLOCKS`` blocks."""
    if blocks > TOPK_BLOCKS:
        prefix, remaining, bits = _radix_select(keys_smem, SELECT_CHUNK, blocks)
    else:
        prefix = gl.to_tensor(0).to(gl.uint32)
        remaining = gl.to_tensor(TOPK_BLOCKS)
        bits = gl.to_tensor(32)
    return prefix, remaining, bits


@gluon.jit
def _emit_candidates(mb, task, keys_smem, ids_smem, count, prefix, remaining, bits):
    """Publish chunk ``task``'s candidates for Q4F, every slot once per
    launch: a chunk with fewer than ``TOPK_BLOCKS`` blocks marks the rest
    unused (key 0)."""
    base = task * TOPK_BLOCKS
    written = _emit_top(
        mb,
        keys_smem,
        ids_smem,
        SELECT_CHUNK,
        count,
        prefix,
        remaining,
        bits,
        "csc",
        "cid",
        base,
    )
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    slot = gl.arange(0, TOPK_BLOCKS, lanes)
    unused = slot >= written
    mb.put("cid", base + slot, gl.full_like(slot, -1), mask=unused)
    mb.put("csc", base + slot, gl.zeros_like(slot), mask=unused)


@gluon.jit
def topk_final_stage(mb, task, length, STAGE: gl.constexpr):
    """Q4F: the best ``TOPK_BLOCKS`` of the Q4 chunks' candidates, in
    ascending block order (the candidates are in block order), when there
    is more than one chunk."""
    mb.stamp(STAGE, trace.START)
    n = length // COMPRESS
    if n > SELECT_CHUNK:
        count = (n + SELECT_CHUNK - 1) // SELECT_CHUNK * TOPK_BLOCKS
        kl: gl.constexpr = _select_layouts(gl.num_warps())[0]
        flat: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
        N: gl.constexpr = SELECT_CHUNKS * TOPK_BLOCKS
        keys_smem = gl.allocate_shared_memory(gl.uint32, [N], flat)
        ids_smem = gl.allocate_shared_memory(gl.int32, [N], flat)
        cols = gl.arange(0, SELECT_TILE, kl)
        for start in gl.static_range(0, N, SELECT_TILE):
            c = start + cols
            got_k, got_i = mb.poll_many(
                ("csc", "cid"), (gl.minimum(c, count - 1), gl.minimum(c, count - 1))
            )
            keys = got_k.to(gl.uint32, bitcast=True)
            keys_smem.slice(start, SELECT_TILE).store(gl.where(c < count, keys, 0))
            ids_smem.slice(start, SELECT_TILE).store(got_i)
        gl.barrier()
        mb.stamp(STAGE, trace.READY)
        prefix, remaining, bits = _radix_select(keys_smem, N, count)
        mb.stamp(STAGE, trace.COMPUTED)
        _emit_top(
            mb, keys_smem, ids_smem, N, count, prefix, remaining, bits, "", "sel", 0
        )
    mb.stamp(STAGE, trace.END)


@gluon.jit
def _attend_tokens(mb, L, tok, j, n, nsel, complete, cache):
    """``(slot, valid, launch, position)`` of attended tokens ``j``: their KV
    slots, and whether each is a token of this launch (position ``L - tok``
    or later: its K / V come from its row's mailbox, not the cache).

    Token ``j < 4 * nsel`` is ``4 * sel[j // 4] + j % 4``. Later tokens are
    the tokens of the partial block, ``4 * complete ..``.
    """
    ok = j < n
    if complete > TOPK_BLOCKS:
        blk = mb.poll("sel", gl.minimum(j // COMPRESS, TOPK_BLOCKS - 1))
    else:
        blk = j // COMPRESS
    head = j < COMPRESS * nsel
    pos = gl.where(
        head, blk * COMPRESS + j % COMPRESS, COMPRESS * complete + (j - COMPRESS * nsel)
    )
    ok = ok & (pos >= 0) & (pos <= L) & (gl.where(head, blk < complete, True))
    page_size = cache.page_size
    page = gl.load(
        cache.full_table + gl.where(ok, pos, 0) // page_size, mask=ok, other=0
    )
    ok = ok & (page > 0)
    slot = (page * page_size + pos % page_size).to(gl.int64)
    return gl.where(ok, slot, 0), ok, ok & (pos >= L - tok), pos


@gluon.jit
def attention_stage(mb, task, tok, cache, CHAIN: gl.constexpr, STAGE: gl.constexpr):
    """Q5 task ``task`` = (KV head ``g``, split ``s``) of a row that is token
    ``tok`` of its request, of a launch of ``CHAIN`` tokens per request (the
    launch's ``Plan.chain``, also when ``mb`` is the view of one row)."""
    mb.stamp(STAGE, trace.START)
    kv = _attention_issue(mb, task, tok, cache)
    gl.amd.hint.sched_barrier()
    _attention_finish(mb, task, tok, kv, CHAIN, STAGE)
    # The next stage on this CTA issues its weight loads after every wave's
    # stores here. Else the late waves' stores queue behind those loads, and
    # Q6 waits about 10 us for them (4 rows).
    gl.barrier()


@gluon.constexpr_function
def _attention_layouts(nw):
    """(qk MFMA, pv MFMA, K operand, V load) layouts of Q5."""
    qk_mma = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, nw]
    )
    pv_mma = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16], transposed=True, warps_per_cta=[1, nw]
    )
    k_layout = gl.DotOperandLayout(1, qk_mma, 8)
    v_load = gl.BlockedLayout([1, 8], [2, 32], [nw, 1], [1, 0])
    return qk_mma, pv_mma, k_layout, v_load


@gluon.jit
def _attention_issue(mb, task, tok, cache):
    """Issue Q5 task ``task``'s K / V gather from the cache.

    Returns ``(keys [HEAD, BN], values [BN, HEAD], valid, the launch masks
    and the positions of the keys and of the values, and the position of the
    request's first token in this launch)``. The entries of the tokens of
    this launch are zero (they come from the mailbox).
    """
    g = task // SPLITS
    s = task % SPLITS
    L = cache.length() - 1
    complete = (L + 1) // COMPRESS
    nsel = gl.minimum(complete, TOPK_BLOCKS)
    n = COMPRESS * nsel + (L + 1 - COMPRESS * complete)
    tiles = (n + TILE - 1) // TILE
    per = (tiles + SPLITS - 1) // SPLITS * TILE
    first = s * per
    k_layout: gl.constexpr = _attention_layouts(gl.num_warps())[2]
    v_load: gl.constexpr = _attention_layouts(gl.num_warps())[3]
    BN: gl.constexpr = SPLIT_TOKENS

    # Token slots in the K (columns) and V (rows) load layouts.
    jk = first + gl.arange(0, BN, gl.SliceLayout(0, k_layout))
    jv = first + gl.arange(0, BN, gl.SliceLayout(1, v_load))
    end = gl.minimum(n, first + per)
    slot_k, ok_k, cur_k, pos_k = _attend_tokens(
        mb, L, tok, gl.where(jk < end, jk, n), n, nsel, complete, cache
    )
    slot_v, ok_v, cur_v, pos_v = _attend_tokens(
        mb, L, tok, gl.where(jv < end, jv, n), n, nsel, complete, cache
    )
    dk = gl.arange(0, HEAD, gl.SliceLayout(1, k_layout))
    dv = gl.arange(0, HEAD, gl.SliceLayout(0, v_load))
    keys = gl.load(
        cache.k_cache + g * HEAD + slot_k[None, :] * cache.kv_stride + dk[:, None],
        mask=(ok_k & ~cur_k)[None, :],
        other=0.0,
    )
    values = gl.load(
        cache.v_cache + g * HEAD + slot_v[:, None] * cache.kv_stride + dv[None, :],
        mask=(ok_v & ~cur_v)[:, None],
        other=0.0,
    )
    return keys, values, ok_k, cur_k, pos_k, cur_v, pos_v, L - tok


@gluon.jit
def _attention_finish(mb, task, tok, kv, CHAIN: gl.constexpr, STAGE: gl.constexpr):
    """Q5 task ``task`` after :func:`_attention_issue`: poll q and the k / v of
    the request's tokens in this launch (rows ``-tok .. 0`` from this one),
    attend, publish the split's output and LSE."""
    keys, values, ok_k, cur_k, pos_k, cur_v, pos_v, start = kv
    N: gl.constexpr = CHAIN
    NP: gl.constexpr = triton.next_power_of_2(N)
    g = task // SPLITS
    s = task % SPLITS
    nw: gl.constexpr = gl.num_warps()
    qk_mma: gl.constexpr = _attention_layouts(nw)[0]
    pv_mma: gl.constexpr = _attention_layouts(nw)[1]
    q_layout: gl.constexpr = gl.DotOperandLayout(0, qk_mma, 8)
    k_layout: gl.constexpr = _attention_layouts(nw)[2]
    p_layout: gl.constexpr = gl.DotOperandLayout(0, pv_mma, 4)
    v_layout: gl.constexpr = gl.DotOperandLayout(1, pv_mma, 4)
    v_load: gl.constexpr = _attention_layouts(nw)[3]
    BN: gl.constexpr = SPLIT_TOKENS

    # q of the 12 heads of the KV head (rows 12 .. 15 are zero), and k, v of
    # the tokens of the request in the launch ([token, HEAD]). Token u is row
    # u - tok from this one, and later tokens read this row. The poll uses
    # plain blocked layouts: the asm of the poll wants one value per register,
    # which MFMA operand layouts do not give.
    # The polls read pairs (two BF16 values), one load per pair: indexing
    # the values loads each pair twice and holds both in registers.
    ql: gl.constexpr = gl.BlockedLayout([1, 2], [2, 32], [nw, 1], [1, 0])
    tl_: gl.constexpr = gl.BlockedLayout([NP, 1], [1, 64], [1, nw], [1, 0])
    qr = gl.arange(0, 16, gl.SliceLayout(1, ql))[:, None]
    qp = gl.arange(0, HEAD // 2, gl.SliceLayout(0, ql))[None, :]
    u = gl.arange(0, NP, gl.SliceLayout(1, tl_))[:, None]
    p = gl.arange(0, HEAD // 2, gl.SliceLayout(0, tl_))[None, :]
    rows = mb.row_offset(gl.where(u <= tok, u - tok, 0))
    head = gl.minimum(g * GROUP + qr, HEADS - 1)
    got_q, got_k, got_v = mb.poll_many(
        ("qn", "kn", "qp"),
        (
            head * (HEAD // 2) + qp,
            rows + g * (HEAD // 2) + p,
            rows + (V_OFF + g * HEAD) // 2 + p,
        ),
    )
    mb.stamp(STAGE, trace.READY)
    q = _bf16_values(got_q)
    q_rows = gl.arange(0, 16, gl.SliceLayout(1, q.type.layout))[:, None]
    q = gl.where(q_rows < GROUP, q, 0.0).to(gl.bfloat16)
    q = gl.convert_layout(q, q_layout)
    # The tokens' k and v go through LDS once; each token's row then loads
    # in the layout it merges into. (Per-token register reductions and
    # layout conversions cost an LDS round trip each and spilled registers.)
    line: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    tok_k = gl.allocate_shared_memory(gl.bfloat16, [NP * HEAD], line)
    tok_v = gl.allocate_shared_memory(gl.bfloat16, [NP * HEAD], line)
    tok_k.store(gl.reshape(_bf16_values(got_k).to(gl.bfloat16), [NP * HEAD]))
    tok_v.store(gl.reshape(_bf16_values(got_v).to(gl.bfloat16), [NP * HEAD]))
    gl.barrier()
    # Each column's token of this launch (-1: from the cache).
    tok_col = gl.where(cur_k, pos_k - start, -1)
    for t in gl.static_range(N):
        k_t = tok_k.slice(t * HEAD, HEAD).load(gl.SliceLayout(1, k_layout))
        keys = gl.where((tok_col == t)[None, :], k_t[:, None], keys)
    tok_row = gl.where(cur_v, pos_v - start, -1)
    for t in gl.static_range(N):
        v_t = tok_v.slice(t * HEAD, HEAD).load(gl.SliceLayout(0, v_load))
        values = gl.where((tok_row == t)[:, None], v_t[None, :], values)

    scores = gl.zeros([16, BN], gl.float32, qk_mma)
    scores = gl.amd.cdna4.mfma(q, keys, scores)
    valid = gl.convert_layout(ok_k, gl.SliceLayout(0, qk_mma))
    scores = gl.where(valid[None, :], scores * QK_SCALE, float("-inf"))
    m = gl.max(scores, axis=1)
    m_safe = gl.where(m == float("-inf"), 0.0, m)
    p = gl.exp2(scores - m_safe[:, None])
    l = gl.sum(p, axis=1)
    p_op = gl.convert_layout(p.to(gl.bfloat16), p_layout)
    v_smem = gl.allocate_shared_memory(
        gl.bfloat16, [BN, HEAD], gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    )
    v_smem.store(values)
    gl.barrier()
    v_op = v_smem.load(v_layout)
    acc = gl.zeros([16, HEAD], gl.float32, pv_mma)
    acc = gl.amd.cdna4.mfma(p_op, v_op, acc)
    mb.stamp(STAGE, trace.COMPUTED)
    l_acc = gl.convert_layout(l, gl.SliceLayout(1, pv_mma))
    out = acc / gl.where(l_acc > 0, l_acc, 1.0)[:, None]
    lse = gl.where(l > 0, m + gl.log2(gl.where(l > 0, l, 1.0)), float("-inf"))
    r = gl.arange(0, 16, gl.SliceLayout(1, pv_mma))
    c = gl.arange(0, HEAD, gl.SliceLayout(0, pv_mma))
    part = ((s * KV_HEADS + g) * GROUP + r[:, None]) * HEAD + c[None, :]
    mb.put("part", part, out.to(gl.int32, bitcast=True), mask=(r < GROUP)[:, None])
    rl = gl.arange(0, 16, gl.SliceLayout(1, qk_mma))
    mb.put(
        "lse",
        s * HEADS + g * GROUP + rl,
        lse.to(gl.int32, bitcast=True),
        mask=rl < GROUP,
    )
    mb.stamp(STAGE, trace.END)


@gluon.jit
def _bf16_values(words):
    """FP32 ``[..., 2 n]`` of the BF16 pairs ``words [..., n]``: value ``2 p``
    is the low half of pair ``p``."""
    lo = (words << 16).to(gl.float32, bitcast=True)
    hi = (words & -65536).to(gl.float32, bitcast=True)
    shape: gl.constexpr = [words.shape[0], 2 * words.shape[1]]
    return gl.reshape(gl.join(lo, hi), shape)


@gluon.jit
def merge_stage(mb, task, STAGE: gl.constexpr):
    """Q6 task ``task`` = (head ``h``, dims ``128 * half ..``)."""
    mb.stamp(STAGE, trace.START)
    h = task // 2
    half = task % 2
    g = h // GROUP
    hh = h % GROUP
    ml: gl.constexpr = gl.BlockedLayout([1, 4], [2, 32], [gl.num_warps(), 1], [1, 0])
    s = gl.arange(0, SPLITS, gl.SliceLayout(1, ml))
    c = half * 128 + gl.arange(0, 128, gl.SliceLayout(0, ml))
    part = ((s[:, None] * KV_HEADS + g) * GROUP + hh) * HEAD + c[None, :]
    gate = (h * 2 * HEAD + HEAD + c) // 2
    got_p, got_l, got_g = mb.poll_many(
        ("part", "lse", "qp"), (part, s * HEADS + h, gate)
    )
    mb.stamp(STAGE, trace.READY)
    lse = got_l.to(gl.float32, bitcast=True)
    top = gl.max(lse, axis=0)
    w = gl.exp2(lse - top)
    out = gl.sum(got_p.to(gl.float32, bitcast=True) * w[:, None], axis=0) / gl.sum(
        w, axis=0
    )
    a = out.to(gl.bfloat16).to(gl.float32)
    gated = a * mx.sigmoid(mx.bf16_half(got_g, HEAD + c))
    mb.stamp(STAGE, trace.COMPUTED)
    mx.put_bf16(mb, "obf", (h * HEAD + half * 128) // 2, gated)
    mb.stamp(STAGE, trace.END)


__all__ = [
    "ATTN_TASKS",
    "Cache",
    "MERGE_TASKS",
    "PROJ_TASKS",
    "PROLOGUE_TASKS",
    "REGIONS",
    "SCORE_TASKS",
    "Weights",
    "attention_stage",
    "merge_stage",
    "projection_stage",
    "prologue_stage",
    "score_stage",
    "topk_final_stage",
    "topk_stage",
]
