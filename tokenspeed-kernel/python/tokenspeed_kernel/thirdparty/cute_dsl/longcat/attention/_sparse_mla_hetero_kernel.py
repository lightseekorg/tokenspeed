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

"""LongCat heterogeneous sparse-MLA decode kernel (SM103, verify-8, BS=1).

One launch computes, for every verify token ``t`` and this rank's local MLA
heads, the sparse attention over that token's Top-K selection:

    S[h, j] = sm_scale * <q_abs[t, h, :576], kv_cache[slot[t, j], :576]>
    P       = softmax_row(S)
    out[t, h, :512] = sum_j P[h, j] * kv_cache[slot[t, j], :512]

Grid ``(ctas_per_cluster, clusters, 1)`` uses clusters
``(ctas_per_cluster, 1, 1)``. With ``splits=S`` CTAs per token, each CTA
owns ``ceil(2048 / (32*S))`` tiles, and ``S / ctas_per_cluster`` clusters
(quarters) cooperate on a token. CTA ``bidx`` in quarter ``q`` owns split
``q * ctas_per_cluster + bidx``. The default is 16 splits and 4 CTAs per
cluster (128 CTAs total); see ``FORCED_TOKENS`` for row ownership.

Forced rows are gathered before selection publication; phase B waits for
the selection. Each tile uses ``cp.async.bulk`` with mbarrier tx-count.
A 2-stage ``PipelineAsync`` overlaps QK(tile t+1) with PV(tile t):

* W0-1 (64 threads): QK and register online softmax, 16 KV columns each.
  Four-lane shuffles reduce rows; ``sStat``/``BAR_PROD`` exchange maxima.
  Producers publish packed P A-fragments and alpha with the pipeline stage.
* W2: idle until the merge.
* W3-6 (128 threads): rescale register O by alpha, then PV; one 16-byte P
  load per k-step, V via ``ldmatrix.x2.trans``.
* W7-9: FP8 conversion, Q as tile -1 then ``sK16[2]``; BF16 idle.
* W10: gather issuer. W11: one Q bulk copy per real head plus slot prefetch.

K readiness uses ``mbarFull[t % ring_tiles]``, phase
``(t // ring_tiles) & 1``. BF16 W0 broadcasts its wait via ``BAR_PROD``;
FP8 converters hand off via two ``BAR_CONV`` passes. PV releases each
``sK16`` buffer through ``sKFree`` after its last V read; raw-ring reuse
waits for the last raw reader (FP8 converters or BF16 PV).

A single-round DSMEM reduce-scatter merges O/m/l within each cluster.
Each CTA writes its ``512 / ctas_per_cluster`` O columns to
``partial[token][quarter]``; CTA 0 writes m/l. Every CTA release-adds the
token's monotonic counter, then acquires ``(gen + 1) * S`` before reading
partials. Split ``s < local_heads`` merges head ``s`` across quarters,
normalizes its latent row and optionally projects through ``w_vc_t[s]``;
each head's weights are read by one CTA. Device-side generation supports
CUDA-graph replay; engine callers may supply a generation value instead.

The engine-callable device routine takes a 128-byte-aligned SMEM arena;
use the instance's ``arena_bytes`` for its configured geometry.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import (
    BFloat16,
    Float8E4M3FN,
    Float16,
    Float32,
    Int32,
    Int64,
    Uint8,
    Uint32,
)
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    PipelineAsync,
    PipelineUserType,
    make_pipeline_state,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._primitives import (
    cpasync_bulk_g2s,
    exp2_f32,
    fp8x4_to_fp16x4,
    ld_shared_cluster_f32,
    ld_shared_cluster_f32x4,
    mapa_shared_cluster_u32,
    mma_sync,
    mma_sync_bf16_ssa,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention.selection_exchange import (
    MAX_TAGGED_SLOT,
    TAG_MASK,
    TAG_SHIFT,
    VALUE_MASK,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._collective_primitives import (
    fence_proxy_async_global,
    load_acquire_sys_u32,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._primitives import (
    atomic_add_acq_rel_gpu_u32,
    load_acquire_gpu_u32,
    prefetch_global_l2,
    red_add_release_gpu_u32,
)
from tokenspeed_kernel.thirdparty.cute_dsl.topk.top_k.single_pass_multi_cta_radix_topk import (
    fence_acq_rel_gpu,
)

# Shape constants (LongCat 2.0 decode, BS=1, verify 8)

NUM_TOKENS = 8
TOPK = 2048
HEAD_LATENT = 512
# Offset adjacent FP32 rows by eight banks for the PV warps' paired stores.
MERGE_PITCH = HEAD_LATENT + 8
HEAD_ROPE = 64
ROW_ELEMS = HEAD_LATENT + HEAD_ROPE  # 576
# Module constants describe the 16-split default; instances use split_geometry.
SPLITS_PER_TOKEN = 16
# CTAs per cluster divide the latent columns; quarters per token count the
# cluster partials. These dimensions need not be equal.
CTAS_PER_CLUSTER = 4
QUARTERS_PER_TOKEN = SPLITS_PER_TOKEN // CTAS_PER_CLUSTER
CLUSTERS = NUM_TOKENS * SPLITS_PER_TOKEN // CTAS_PER_CLUSTER
ARRIVALS_PER_TOKEN = (
    QUARTERS_PER_TOKEN * CTAS_PER_CLUSTER
)  # == SPLITS_PER_TOKEN, per token per step on gCounter
if SPLITS_PER_TOKEN % CTAS_PER_CLUSTER or HEAD_LATENT % (CTAS_PER_CLUSTER * 4):
    # Each CTA's column slice must contain whole 16-B merge chunks.
    raise ValueError(
        f"CTAS_PER_CLUSTER {CTAS_PER_CLUSTER} must divide SPLITS_PER_TOKEN "
        f"{SPLITS_PER_TOKEN} and quarter HEAD_LATENT {HEAD_LATENT} evenly"
    )
TILE_KV = 32
TILES_PER_CTA = 4  # ceil(TOPK / (SPLITS_PER_TOKEN * TILE_KV))
M_ROWS = 16  # local_heads (9/10) padded
SLOTS_PER_CTA = TILES_PER_CTA * TILE_KV  # 128


def split_geometry(splits: int, ctas_per_cluster: int = CTAS_PER_CLUSTER) -> dict:
    """Per-token CTA geometry for ``splits`` CTAs per verify token.

    Args:
        splits: CTAs sharing one token's ``TOPK`` rows; a multiple of
            ``ctas_per_cluster``.
        ctas_per_cluster: CTAs per physical cluster (``CTAS_PER_CLUSTER``
            by default).

    Returns:
        ``tiles`` (per CTA, ceil -- the last one may be ragged), ``slots``
        (ring rows per CTA), ``quarters`` (clusters per token), ``clusters``
        (grid y) and ``ctas``.  At 16 splits these are the module constants.
    """
    if splits <= 0 or splits % ctas_per_cluster:
        raise ValueError(
            f"splits {splits} must be a positive multiple of "
            f"ctas_per_cluster {ctas_per_cluster}"
        )
    tiles = -(-TOPK // (splits * TILE_KV))
    return {
        "tiles": tiles,
        "slots": tiles * TILE_KV,
        "quarters": splits // ctas_per_cluster,
        "clusters": NUM_TOKENS * splits // ctas_per_cluster,
        "ctas": NUM_TOKENS * splits,
    }


PIPE_STAGES = 2  # sP/sStat pipeline depth (QK(t+1) overlaps PV(t))
# Full-ring configurations need no sRawFree; smaller rings recycle slots.
RING_TILES = 4

# Canonical gSlots order (indexer P3, _forced_index): 16 initial positions,
# 1024 local-window positions, then score-selected positions. Both forced
# groups are ascending; f_len = min(causal length, FORCED_TOKENS).
# Contiguous map (16 splits only): phase-A rows [0, 64) map to canonical
# 16 + 64*split + r; phase-B rows [64, 127) to
# f_len + 63*split + (r - 64); row 127 is initial token split.
# Strided map: row r maps to r*splits + split. Phase A stays forced because
# 64*splits <= FORCED_TOKENS; phase B includes the remaining forced rows
# (canonical 1024..1039 at 16 splits, including current-step positions).
# Forced indices are masked at f_len, selected indices at valid length.
INITIAL_TOKENS = 16
LOCAL_TOKENS = 1024
FORCED_TOKENS = INITIAL_TOKENS + LOCAL_TOKENS  # 1040
PHASE_A_TILES = 2  # ring tiles gathered before the selection
# Strided rows: tiles from here on are skipped when none of their rows is
# valid (short rows).  Engine mode learns the length with the selection, so
# tile ``PHASE_A_TILES`` is always run -- its arrival is what publishes
# ``sValidLen`` to every warp; standalone mode knows it at entry (tile 0 on).
SKIP_FROM = PHASE_A_TILES + 1
PHASE_A_ROWS = PHASE_A_TILES * TILE_KV  # 64 local-window rows per CTA
PHASE_B_ROWS = SLOTS_PER_CTA - PHASE_A_ROWS - 1  # 63 selected rows per CTA
INITIAL_ROW = SLOTS_PER_CTA - 1  # ring row of the CTA's initial token

# Shared-memory pitches.  Raw FP8 rows are 576 B, 16-bit compute rows
# 1152 B; both padded so the 8 rows of one ldmatrix matrix hit eight
# distinct 4-byte banks (592 B = 148 words, 1168 B = 292 words).
RAW_PITCH_FP8_ELEMS = 592  # 1-byte elements
ELEMS_16BIT_PITCH = 584  # 584 16-bit elements = 1168 B

HEAD_PROJ = 128  # w_vc output width of one head
WVC_PITCH = HEAD_LATENT * 2 + 16  # SMEM row pitch of the staged w_vc_t slice
PROJ_THREADS = 256  # threads of the head-CTA that run the w_vc projection
PROJ_COLS_PER_GROUP = 4  # output columns one 8-lane group owns (128 / 32)
PROJ_K_SLICES = 4

# Partial scratch: [token][quarter] = m(16) | l(16) | unnormalized O(16*512).
PART_M = 0
PART_L = M_ROWS
PART_O = 2 * M_ROWS
PARTIAL_FLOATS = PART_O + M_ROWS * HEAD_LATENT

# Warp roles.
W_QK0 = 0  # producer: QK MMA + register softmax, n-tile 0
W_QK1 = 1  # producer: QK MMA + register softmax, n-tile 1
W_STATS = 2  # idle (statistics live in the producer registers)
W_PV0 = 3
W_PV_END = 6  # inclusive; consumers: PV MMA (O in registers)
W_CONV0 = 7  # FP8 tile conversion (Q as tile -1, then sK16[2]); BF16: idle
W_CONV_END = 9  # inclusive
W_GATHER = 10  # slot table + tile gather issue
W_LOAD = 11  # Q bulk loader + slot L2 prefetch

PRODUCER_THREADS = 64  # W0-1
CONSUMER_THREADS = 128  # W3-6 (PV; also the sKFree arrive count)
PV_THREADS = 128  # W3-6 (sKFree arrive count)
CONV_THREADS = 96  # W7-9

# CTA-local named barriers; participant counts must match exactly to avoid
# bar.sync deadlock. IDs 0-3 are reserved/mbarrier-space on SM103;
# cutlass.pipeline uses 8.
#   4  W0-1       64: BF16 K-ready broadcast and producer max exchange.
#   6  W0-1,7-9  160: FP8 Q/K16 hand-off, two passes per Q/tile conversion.
#   7  all       384: DSMEM merge staging.
#   9  all       384: partial publication, counter wait and per-head merge.
#  10  W7-9       96: raw Q/K readiness broadcast to FP8 converters.
BAR_PROD = 4
BAR_CONV = 6
BAR_MERGE = 7
BAR_FINAL = 9
BAR_RAW = 10

# Score clamp: keeps exp2 arguments finite when every column of a row is
# masked (m stays >= CLAMP_M; masked scores sit at MASKED_SCORE so that
# exp2(MASKED_SCORE - m) == 0 under the clamp).
CLAMP_M = -30000.0
MASKED_SCORE = -60000.0

LOG2E = 1.4426950408889634


def _is_fp8(dtype) -> bool:
    return cutlass.const_expr(dtype is Float8E4M3FN)


def _build_smem_map(
    is_fp8: bool,
    ring_tiles: int = RING_TILES,
    splits: int = SPLITS_PER_TOKEN,
    ctas_per_cluster: int = CTAS_PER_CLUSTER,
) -> dict:
    """Byte offsets and total size of the 128-byte-aligned SMEM arena.

    Args:
        is_fp8: adds raw Q staging, ``sK16[2]`` and ``sKFree[2]``;
            BF16 aliases ``sK16`` onto ``sK``.
        ring_tiles: raw gather slots; adds ``sRawFree`` if slots recycle.
        splits: CTAs per token (sizes the per-CTA slot and mask tables).
        ctas_per_cluster: CTAs sharing the DSMEM merge (sizes ``sCross``).

    Returns:
        Dict of section name -> byte offset, plus ``"total"``.
    """
    geo = split_geometry(splits, ctas_per_cluster)
    off = 0
    sections = {}

    def alloc(name: str, nbytes: int, align: int) -> None:
        nonlocal off
        off = (off + align - 1) // align * align
        sections[name] = off
        off += nbytes

    # The engine issues early gathers at arena offset 0.
    sk_bytes = (
        ring_tiles
        * TILE_KV
        * (RAW_PITCH_FP8_ELEMS if is_fp8 else ELEMS_16BIT_PITCH * 2)
    )
    alloc("sK", sk_bytes, 128)
    # Projection broadcasts the merged latent row through the dead gather ring.
    sections["sLatent"] = sections["sK"]
    ring_end = sections["sK"] + sk_bytes
    if is_fp8:
        alloc("sQRaw", M_ROWS * RAW_PITCH_FP8_ELEMS, 16)
    alloc("sQ", M_ROWS * ELEMS_16BIT_PITCH * 2, 16)
    if is_fp8:
        alloc("sK16", 2 * TILE_KV * ELEMS_16BIT_PITCH * 2, 16)
    else:
        sections["sK16"] = sections["sK"]  # alias: BF16 reads tiles in place
    alloc("sP", PIPE_STAGES * 2 * 32 * 4 * 4, 16)
    alloc("sStat", PIPE_STAGES * 2 * M_ROWS * 2 * 4, 16)
    alloc("sAlpha", PIPE_STAGES * M_ROWS * 4, 16)
    alloc("sMrow", M_ROWS * 4, 16)
    alloc("sLrow", M_ROWS * 4, 16)
    alloc("sMergeM", M_ROWS * 4, 16)
    alloc("sMergeL", M_ROWS * 4, 16)
    alloc("sMerge", M_ROWS * MERGE_PITCH * 4, 16)
    # Keep merged m/l separate from sources peers may still be pulling.
    alloc("sCross", M_ROWS * ctas_per_cluster * 4, 16)
    alloc("sMergeMB", M_ROWS * 4, 16)
    alloc("sMergeLB", M_ROWS * 4, 16)
    alloc("sSlots", geo["slots"] * 4, 16)
    alloc("sMaskBytes", geo["slots"] * 4, 16)
    # Gather publishes valid length via phase-B mbarriers (release arm/acquire wait).
    alloc("sValidLen", 16, 16)
    alloc("mbarFull", ring_tiles * 8, 8)
    alloc("sQFull", 8, 8)
    if is_fp8:
        alloc("sKFree", 2 * 8, 8)
    if ring_tiles < geo["tiles"]:
        # Last raw readers release each slot: FP8 converters or BF16 PV warps.
        alloc("sRawFree", ring_tiles * 8, 8)
    alloc("pipe", 2 * PIPE_STAGES * 8, 8)
    alloc("mbarWvc", 8, 8)
    # After the DSMEM entry barrier the raw ring and Q/K16 staging are dead.
    # Stage w_vc_t[head] (128 x 1 KB) behind sLatent in region A, then in
    # region B (Q/K16, sP/sStat/sAlpha). WVC_PITCH = 1 KB + 16 keeps the
    # eight ldmatrix row addresses in distinct bank groups.
    sections["sWvcA"] = sections["sK"] + HEAD_LATENT * 4
    rows_a = (ring_end - sections["sWvcA"]) // WVC_PITCH
    sections["wvcRowsA"] = min(rows_a, HEAD_PROJ)
    sections["sWvcB"] = sections["sQRaw"] if is_fp8 else sections["sQ"]
    # Projection tracing checks combined staging capacity.
    sections["wvcRowsB"] = (sections["sMrow"] - sections["sWvcB"]) // WVC_PITCH
    # The BF16 projection operand reuses the dead slot/mask tables.
    sections["sLatent16"] = sections["sSlots"]
    assert (
        sections["sMaskBytes"] + geo["slots"] * 4 - sections["sSlots"]
        >= HEAD_LATENT * 2
    )
    sections["total"] = off
    return sections


_SMEM_MAP_FP8 = _build_smem_map(True)
_SMEM_MAP_BF16 = _build_smem_map(False)


def dynamic_smem_bytes(kv_dtype) -> int:
    """Dynamic shared-memory bytes per CTA at the default split/ring geometry.

    Args:
        kv_dtype: ``Float8E4M3FN`` or ``BFloat16`` — the KV-cache row dtype
            the kernel is compiled for.

    Returns:
        Total byte count (callers must provide a 128-byte-aligned base).
    """
    if kv_dtype is Float8E4M3FN:
        return _SMEM_MAP_FP8["total"]
    return _SMEM_MAP_BF16["total"]


def _smem_ptr(dtype, arena_i32, offset, *, assumed_align):
    """Trace-time pointer into the routine's shared-memory arena."""
    return cute.make_ptr(
        dtype,
        arena_i32 + Int32(offset),
        cute.AddressSpace.smem,
        assumed_align=assumed_align,
    )


def _smem_tensor(dtype, arena_i32, offset, shape, stride=None, *, assumed_align=16):
    """Trace-time tensor view with explicit element strides into the arena."""
    return cute.make_tensor(
        _smem_ptr(dtype, arena_i32, offset, assumed_align=assumed_align),
        cute.make_layout(shape, stride=stride),
    )


def _load_shared_words(ptr, nwords, instruction, constraints, *, loc, ip):
    """Trace one shared-memory load and unpack its Uint32 result words."""
    addr = ptr.toint(loc=loc, ip=ip)
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * nwords),
        [addr.ir_value(loc=loc, ip=ip)],
        instruction,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    packed = vector.from_elements(
        ir.VectorType.get([nwords], T.i32(), loc=loc),
        [
            llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)
            for i in range(nwords)
        ],
        loc=loc,
    )
    return cute.TensorSSA(packed, nwords, Uint32)


def _pack_16bitx2(lo, hi, instruction, *, loc, ip):
    """Trace one f32-pair conversion into a packed 16-bit word."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Float32(lo).ir_value(loc=loc, ip=ip),
                cutlass.Float32(hi).ir_value(loc=loc, ip=ip),
            ],
            instruction,
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def pack_bf16x2(
    lo: cutlass.Float32, hi: cutlass.Float32, *, loc=None, ip=None
) -> Uint32:
    """Pack two Float32 into one bf16x2 word (``cvt.rn.bf16x2.f32``).

    ``lo`` lands in the low half (even element), ``hi`` in the high half.
    """
    return _pack_16bitx2(lo, hi, "cvt.rn.bf16x2.f32 $0, $2, $1;", loc=loc, ip=ip)


@dsl_user_op
def pack_f16x2(
    lo: cutlass.Float32, hi: cutlass.Float32, *, loc=None, ip=None
) -> Uint32:
    """Pack two Float32 into one f16x2 word (``cvt.rn.f16x2.f32``).

    ``lo`` lands in the low half (even element), ``hi`` in the high half.
    """
    return _pack_16bitx2(lo, hi, "cvt.rn.f16x2.f32 $0, $2, $1;", loc=loc, ip=ip)


@dsl_user_op
def ld_shared_u32x4(ptr: cute.Pointer, *, loc=None, ip=None):
    """Load 16 B from this CTA's shared memory as four Uint32 (LDS.128)."""
    return _load_shared_words(
        ptr,
        4,
        "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_shared_u32x4(ptr: cute.Pointer, packed, *, loc=None, ip=None) -> None:
    """Store one 16 B fragment (four Uint32) to this CTA's shared memory."""
    words = [packed[i].ir_value(loc=loc, ip=ip) for i in range(4)]
    llvm.inline_asm(
        None,
        [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip), *words],
        "st.shared.v4.u32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ldmatrix_b16_x4(ptr: cute.Pointer, *, loc=None, ip=None):
    """``ldmatrix.sync.aligned.m8n8.x4.shared.b16`` — four 8x8 b16 matrices.

    Lanes 0-31 provide one row address each: matrix ``l // 8``, row ``l % 8``.
    """
    return _load_shared_words(
        ptr,
        4,
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ldmatrix_b16_x2(ptr: cute.Pointer, *, loc=None, ip=None):
    """``ldmatrix.sync.aligned.m8n8.x2.shared.b16`` — two 8x8 b16 matrices.

    Lanes 0-15 provide row addresses (matrix ``l // 8``, row ``l % 8``).
    """
    return _load_shared_words(
        ptr,
        2,
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {$0, $1}, [$2];",
        "=r,=r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ldmatrix_b16_x2_trans(ptr: cute.Pointer, *, loc=None, ip=None):
    """``ldmatrix...x2.trans`` — B-fragments straight from row-major V rows.

    Lanes 0-7 address the k0-7 rows, lanes 8-15 the k8-15 rows; each 16-byte
    row spans the n window.  Transposed fragments land directly in the
    ``mma.sync`` B-register layout.
    """
    return _load_shared_words(
        ptr,
        2,
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {$0, $1}, [$2];",
        "=r,=r,r",
        loc=loc,
        ip=ip,
    )


def _proj_ldmatrix(a_base: Int32, b_base: Int32, ks):
    """Trace-time helper: the projection's A x4 and the two B x4 fragments
    of 32-wide k slab ``ks`` (byte bases of this lane's row addresses)."""

    def ld(addr):
        return ldmatrix_b16_x4(
            cute.make_ptr(BFloat16, addr, cute.AddressSpace.smem, assumed_align=16)
        )

    return (
        ld(a_base + Int32(ks * 64)),
        ld(b_base + Int32(ks * 64)),
        ld(b_base + Int32(ks * 64 + 32)),
    )


class LongcatSparseMlaHeteroKernel:
    """Sparse-MLA verify-8 decode kernel for the heterogeneous attention path.

    Args:
        local_heads: MLA heads owned by this rank (8, 9 or 10); folded into the
            MMA M dimension and padded to 16 rows.
        kv_dtype: ``Float8E4M3FN`` or ``BFloat16`` — KV-cache row dtype; Q
            follows the same precision per the production contract.
        sm_scale: effective softmax scale (LongCat folds the YaRN mscale into
            ``layer.scaling``; pass that value, not ``1/sqrt(192)``).
        k_scale: per-tensor FP8 cache scale folded into the score scale
            (production keeps ``k_scale == 1.0`` for the DSA path).
        splits: CTAs per verify token (``SPLITS_PER_TOKEN`` = 16, chain 128,
            by default).  A multiple of ``ctas_per_cluster`` and at least
            ``local_heads`` (the per-head finalize gives each head one CTA
            of the token), at most 16 (phase A must be forced rows only).
            12 is chain 96: 6 tiles per CTA, with rows past ``TOPK`` masked.
        ring_tiles: raw gather ring slots, ``1..tiles`` (default
            ``min(RING_TILES, tiles)``).  Below the CTA's tile count a slot
            is re-armed for tile ``t + ring_tiles`` once tile ``t``'s last
            reader frees it.
        row_map: which canonical rows a split owns (module comment at
            ``FORCED_TOKENS``).  ``"contiguous"`` is the 16-split layout
            (the default at 16 splits, illegal otherwise); ``"strided"`` is
            ``r * splits + s`` (the default at any other split count).  The
            two partition the same rows differently, so outputs agree to
            fp32 merge tolerance, not bit for bit.  An enclosing engine
            stages phase A itself and must use the same map.
        ctas_per_cluster: CTAs per physical cluster (``CTAS_PER_CLUSTER`` =
            4 by default, or 2).  It is how many CTAs share the DSMEM merge
            (so ``HEAD_LATENT`` is cut into that many column slices), while
            ``splits / ctas_per_cluster`` clusters cooperate on a token (the
            partials the per-head merge reads back); CTA ``r`` of a token's
            cluster ``q`` has split ``q * ctas_per_cluster + r``.
    """

    def __init__(
        self,
        *,
        local_heads: int,
        kv_dtype,
        sm_scale: float,
        k_scale: float = 1.0,
        splits: int = SPLITS_PER_TOKEN,
        ring_tiles: int | None = None,
        row_map: str | None = None,
        ctas_per_cluster: int = CTAS_PER_CLUSTER,
    ) -> None:
        # Branch 1 uses 8 heads; branch 0 uses 9/10. All fit M_ROWS.
        if local_heads not in (8, 9, 10):
            raise ValueError(f"local_heads must be 8, 9 or 10, got {local_heads}")
        self.local_heads = local_heads
        self.kv_dtype = kv_dtype
        self.is_fp8 = kv_dtype is Float8E4M3FN
        self.mma_dtype = Float16 if self.is_fp8 else BFloat16
        self.score_scale = float(sm_scale) * float(k_scale) * LOG2E
        self.out_scale = float(k_scale)
        if ctas_per_cluster not in (2, 4) or HEAD_LATENT % (ctas_per_cluster * 4):
            raise ValueError(
                f"ctas_per_cluster must be 2 or 4 and quarter HEAD_LATENT "
                f"{HEAD_LATENT} evenly, got {ctas_per_cluster}"
            )
        self.ctas_per_cluster = int(ctas_per_cluster)
        geo = split_geometry(splits, self.ctas_per_cluster)
        if splits < local_heads:
            raise ValueError(
                f"splits {splits} < local_heads {local_heads}: the per-head "
                "finalize needs one CTA of the token per head"
            )
        if PHASE_A_ROWS * splits > FORCED_TOKENS:
            # Pre-selection phase A must contain only forced rows.
            raise ValueError(f"splits {splits} > 16 puts selected rows in phase A")
        self.splits = int(splits)
        self.tiles_per_cta = geo["tiles"]
        self.slots_per_cta = geo["slots"]
        self.quarters = geo["quarters"]
        self.clusters = geo["clusters"]
        if ring_tiles is None:
            ring_tiles = min(RING_TILES, self.tiles_per_cta)
        if not 1 <= ring_tiles <= self.tiles_per_cta:
            raise ValueError(
                f"ring_tiles must be in [1, {self.tiles_per_cta}], got {ring_tiles}"
            )
        self.ring_tiles = int(ring_tiles)
        if row_map is None:
            row_map = "contiguous" if splits == SPLITS_PER_TOKEN else "strided"
        if row_map not in ("contiguous", "strided"):
            raise ValueError(f"row_map must be contiguous or strided, got {row_map}")
        if row_map == "contiguous" and splits != SPLITS_PER_TOKEN:
            raise ValueError("the contiguous row map exists at 16 splits only")
        self.row_map = row_map
        smem = _build_smem_map(
            self.is_fp8, self.ring_tiles, self.splits, self.ctas_per_cluster
        )
        self.arena_bytes = smem.pop("total")
        self.off_sQRaw = 0
        self.off_sKFree = 0
        self.off_sRawFree = 0
        for name, offset in smem.items():
            setattr(self, f"off_{name}", offset)

    @classmethod
    def smem_map(cls, kv_dtype) -> dict:
        """SMEM byte offsets for one KV dtype at the default geometry.

        Args:
            kv_dtype: ``Float8E4M3FN`` or ``BFloat16``.

        Returns:
            Mapping of section name to byte offset inside the sub-arena
            (``"sK"`` is the gather ring, always at offset 0) plus
            ``"total"``.  An enclosing engine uses it to place its own
            storage around the ring.
        """
        return dict(_SMEM_MAP_FP8 if kv_dtype is Float8E4M3FN else _SMEM_MAP_BF16)

    @classmethod
    def smem_bytes(cls, kv_dtype) -> int:
        """Shared-memory bytes per CTA at the default geometry.

        The engine (worker P5) calls this before carving its SMEM arena; the
        returned sub-arena must start on a 128-byte boundary.

        Args:
            kv_dtype: ``Float8E4M3FN`` or ``BFloat16`` — the KV-cache row
                dtype the routine is compiled for.

        Returns:
            Total byte count of the routine's SMEM sub-arena.
        """
        return dynamic_smem_bytes(kv_dtype)

    @cute.jit
    def sparse_mla_device_routine(
        self,
        gQ: cute.Tensor,
        gKv: cute.Tensor,
        gSlots: cute.Tensor,
        gLens: cute.Tensor,
        gPartial: cute.Tensor,
        gCounter: cute.Tensor,
        gGen: cute.Tensor,
        smem_base_ptr: cute.Pointer,
        cta: Int32,
        token: Int32,
        quarter: Int32,
        bidx: Int32,
        tidx: Int32,
        gDone=None,
        gather_mbar=None,
        generation=None,
        gWvc=None,
        gProj=None,
        forced_len=None,
        gSelFlag=None,
        sel_target=None,
        sel_tag=None,
        gate=None,
        gDefer=None,
        gBlockTable=None,
        gPositions=None,
        gOutCacheLoc=None,
        page_shift: cutlass.Constexpr = 6,
    ) -> None:
        """Run one sparse-MLA CTA with 384 threads and a caller-owned arena.

        ``generation`` (engine mode): the step generation as a value; the
        routine then neither reads nor bumps ``gGen``, so a CTA delayed past
        another token's completion cannot observe the next generation.

        ``gSlots[token]`` uses the canonical layout at ``FORCED_TOKENS``;
        score masks are derived from canonical indices in registers.
        ``cta`` is the global CTA id; ``bidx`` its rank within the cluster;
        ``token``/``quarter`` identify the cluster's work; ``tidx`` is 0..383.

        Args:
            gQ: ``[NUM_TOKENS, local_heads, 576]`` absorbed queries.
            gKv: ``[total_slots, 576]`` physical-slot KV rows.
            gSlots: ``[NUM_TOKENS, TOPK]`` int32 slot ids (-1 = masked).
            gLens: ``[NUM_TOKENS]`` int32 valid selection lengths.
            gPartial: ``[NUM_TOKENS, quarters, PARTIAL_FLOATS]`` fp32 merge
                scratch (``quarters = splits / ctas_per_cluster``).
            gCounter: ``[NUM_TOKENS]`` int32 arrival counters (+splits per
                token per step, one per CTA of the token).
            gGen: ``[1]`` int32 device-side generation (kernel-maintained).
            smem_base_ptr: 128-B-aligned arena of ``self.arena_bytes`` bytes.
            gDone: optional uint32 ``[1]`` monotonic head-completion counter
                for an enclosing engine. Each head-CTA release-publishes its
                output with one arrival: ``NUM_TOKENS * local_heads`` per step.
            gather_mbar: optional SMEM pointer to ``ring_tiles`` int64
                mbarriers owned by an enclosing engine that has initialized
                all of them and armed + issued the phase-A tiles (ring rows
                [0, 64), the local-window rows) plus ring row 127 (the
                initial token) itself, before the selection exists.  The
                routine gathers phase B after acquiring the selection (flag
                or tag below). In contiguous mode it stages rows [64, 127)
                from ``gSlots`` and arms tiles 2-3 for 32 rows each. The
                engine must complete phase 0 of ``gather_mbar[0..1]`` with
                all ``TILE_KV`` rows of each tile
                (masked rows included) landed in the ring, and land exactly
                one row (127) on ``gather_mbar[3]``.  Under
                ``row_map="strided"`` no initial row is pre-issued: the
                routine issues all phase-B rows, waiting on slot reuse.
            forced_len: ``min(causal length, FORCED_TOKENS)`` of this token
                as a value (engine mode: derived from the positions, since
                ``gLens`` is not readable before the selection arrives).
                ``None`` derives it from ``gLens[token]``, which then must
                be valid at entry (standalone mode).
            gSelFlag / sel_target: engine selection readiness;
                every gather lane system-acquires ``gSelFlag[0] >= sel_target``.
            sel_tag: engine mode alternative to the flag: the step's Lamport
                tag (``generation & TAG_MASK``, see ``selection_exchange``).
                Gather lanes acquire their ``gSlots``/``gLens`` words until
                every top byte matches, then decode the low 24 bits.
                Exactly one of ``gSelFlag`` / ``sel_tag`` may be given.
            gBlockTable / gPositions / gOutCacheLoc: optional worker-engine
                metadata for canonical short rows. When the entire verify
                batch ends at or below TOPK, reconstruct the same slot order
                directly. The caller must acquire the current KV producers
                before entry and preserve selection publication before its
                downstream selection copy. Page size is 1 << page_shift.
            gate / gDefer: standalone mode, entered before
                its inputs exist (an engine phase issued before its gate):
                ``gate`` lists ``(ticket, target)`` pairs that cover ``gQ``
                and the KV rows of this step's tokens, whose slots are
                ``gDefer`` (``int32[8]``); the rest of the selection is in
                place at entry.  The gather warp issues every other row of
                the first ``ring_tiles`` tiles at once, acquire-spins on the
                gate, then issues the deferred rows; the Q loader waits for
                the gate before its copies.
        """
        lane = tidx % 32
        warp_id = tidx // 32
        split = quarter * self.ctas_per_cluster + bidx
        if cutlass.const_expr(generation is not None):
            gen = Int32(generation)
        else:
            gen = Int32(gGen[0])
        if cutlass.const_expr(forced_len is not None):
            f_len = Int32(forced_len)
        else:
            f_len = Int32(gLens[token])
            if f_len > Int32(FORCED_TOKENS):
                f_len = Int32(FORCED_TOKENS)
        # Contiguous-map bases; _canonical_index handles both row maps.
        a_base = Int32(INITIAL_TOKENS) + split * Int32(PHASE_A_ROWS)
        b_base = f_len + split * Int32(PHASE_B_ROWS) - Int32(PHASE_A_ROWS)

        local_heads = cutlass.const_expr(self.local_heads)
        is_fp8 = _is_fp8(self.kv_dtype)
        mma_dtype = self.mma_dtype
        ring_tiles = cutlass.const_expr(self.ring_tiles)
        tiles = cutlass.const_expr(self.tiles_per_cta)
        slots_per_cta = cutlass.const_expr(self.slots_per_cta)
        recycle = cutlass.const_expr(ring_tiles < tiles)
        score_scale = cutlass.Float32(self.score_scale)

        # Shared-memory views.
        # Padded-stride views are built with make_ptr + make_tensor (recast
        # does not divide padded strides).
        arena_i32 = smem_base_ptr.toint()

        sQ = _smem_tensor(
            mma_dtype,
            arena_i32,
            self.off_sQ,
            (M_ROWS, ROW_ELEMS),
            (ELEMS_16BIT_PITCH, 1),
        )
        sQ_u32 = _smem_tensor(
            Uint32,
            arena_i32,
            self.off_sQ,
            (M_ROWS, ROW_ELEMS // 2),
            (ELEMS_16BIT_PITCH // 2, 1),
        )
        raw_pitch = cutlass.const_expr(
            RAW_PITCH_FP8_ELEMS if is_fp8 else ELEMS_16BIT_PITCH
        )
        row_bytes = cutlass.const_expr(ROW_ELEMS if is_fp8 else ROW_ELEMS * 2)
        sK = _smem_tensor(
            self.kv_dtype,
            arena_i32,
            self.off_sK,
            (ring_tiles, TILE_KV, ROW_ELEMS),
            (TILE_KV * raw_pitch, raw_pitch, 1),
            assumed_align=128,
        )
        raw_row_words = cutlass.const_expr(
            RAW_PITCH_FP8_ELEMS // 4 if is_fp8 else ELEMS_16BIT_PITCH // 2
        )
        # Bulk-copy destination shared with the engine's early gathers.
        sKRaw = _smem_tensor(
            Uint32,
            arena_i32,
            self.off_sK,
            (ring_tiles * TILE_KV * raw_row_words,),
            assumed_align=128,
        )
        if cutlass.const_expr(is_fp8):
            sKRaw_u32 = _smem_tensor(
                Uint32,
                arena_i32,
                self.off_sK,
                (ring_tiles, TILE_KV, ROW_ELEMS // 4),
                (TILE_KV * raw_row_words, raw_row_words, 1),
                assumed_align=128,
            )
            sQRaw_u32 = _smem_tensor(
                Uint32,
                arena_i32,
                self.off_sQRaw,
                (M_ROWS, raw_row_words),
                (raw_row_words, 1),
            )
            sK16 = _smem_tensor(
                mma_dtype,
                arena_i32,
                self.off_sK16,
                (PIPE_STAGES, TILE_KV, ROW_ELEMS),
                (TILE_KV * ELEMS_16BIT_PITCH, ELEMS_16BIT_PITCH, 1),
            )
            sK16_u32 = _smem_tensor(
                Uint32,
                arena_i32,
                self.off_sK16,
                (PIPE_STAGES, TILE_KV, ROW_ELEMS // 2),
                (TILE_KV * (ELEMS_16BIT_PITCH // 2), ELEMS_16BIT_PITCH // 2, 1),
            )
        else:
            sKRaw_u32 = sK
            sQRaw_u32 = sQ_u32
            sK16 = sK
            sK16_u32 = sK
        sP = _smem_tensor(
            Uint32,
            arena_i32,
            self.off_sP,
            (PIPE_STAGES, 2, 32, 4),
            (2 * 32 * 4, 32 * 4, 4, 1),
        )
        sStat = _smem_tensor(
            Float32,
            arena_i32,
            self.off_sStat,
            (PIPE_STAGES, 2, M_ROWS, 2),
            (2 * M_ROWS * 2, M_ROWS * 2, 2, 1),
        )
        sAlpha = _smem_tensor(
            Float32,
            arena_i32,
            self.off_sAlpha,
            (PIPE_STAGES, M_ROWS),
            (M_ROWS, 1),
        )
        sMergeM = _smem_tensor(Float32, arena_i32, self.off_sMergeM, (M_ROWS,))
        sMergeL = _smem_tensor(Float32, arena_i32, self.off_sMergeL, (M_ROWS,))
        sMerge = _smem_tensor(
            Float32,
            arena_i32,
            self.off_sMerge,
            (M_ROWS, HEAD_LATENT),
            (MERGE_PITCH, 1),
        )
        # Preserve source m/l while peers still pull them.
        sMergeMB = _smem_tensor(Float32, arena_i32, self.off_sMergeMB, (M_ROWS,))
        sMergeLB = _smem_tensor(Float32, arena_i32, self.off_sMergeLB, (M_ROWS,))
        sCross = _smem_tensor(
            Float32,
            arena_i32,
            self.off_sCross,
            (M_ROWS, self.ctas_per_cluster),
            (self.ctas_per_cluster, 1),
        )
        sSlots = _smem_tensor(Int32, arena_i32, self.off_sSlots, (slots_per_cta,))
        sValidLen = _smem_tensor(Int32, arena_i32, self.off_sValidLen, (4,))
        external_gather = cutlass.const_expr(gather_mbar is not None)
        # Strided valid rows form a prefix; masked tail tiles contribute
        # P == 0 and alpha == 1, so they can be skipped.
        skip_from = cutlass.const_expr(SKIP_FROM if external_gather else 1)
        skip_tail = cutlass.const_expr(self.row_map == "strided" and tiles > skip_from)
        # Contiguous engine mode lands ring row 127 on tile 3 before the
        # selection, which a reused slot would overwrite.  (Strided engine
        # mode recycles: phase B below waits for each reused slot.)
        assert not (
            external_gather and recycle and self.row_map == "contiguous"
        ), "contiguous engine mode needs a full ring"
        if cutlass.const_expr(external_gather):
            mbarFull = gather_mbar
        else:
            mbarFull = _smem_ptr(Int64, arena_i32, self.off_mbarFull, assumed_align=8)
        sQFull = _smem_ptr(Int64, arena_i32, self.off_sQFull, assumed_align=8)
        mbarWvc = _smem_ptr(Int64, arena_i32, self.off_mbarWvc, assumed_align=8)
        if cutlass.const_expr(is_fp8):
            sKFree = _smem_ptr(Int64, arena_i32, self.off_sKFree, assumed_align=8)
        else:
            sKFree = mbarFull  # unused on BF16; never dereferenced
        if cutlass.const_expr(recycle):
            sRawFree = _smem_ptr(Int64, arena_i32, self.off_sRawFree, assumed_align=8)
        else:
            sRawFree = None
        pipe_mbar = _smem_ptr(Int64, arena_i32, self.off_pipe, assumed_align=8)

        # Prologue before PipelineAsync's CTA barrier: W10 initializes
        # mbarriers/slots; W11 zeros padded Q rows.
        if warp_id == W_GATHER:
            if lane == 0:
                if cutlass.const_expr(not external_gather):
                    for i in cutlass.range_constexpr(ring_tiles):
                        cute.arch.mbarrier_init(mbarFull + i, 1)
                cute.arch.mbarrier_init(sQFull, 1)
                cute.arch.mbarrier_init(mbarWvc, 1)
                if cutlass.const_expr(is_fp8):
                    cute.arch.mbarrier_init(sKFree + 0, PV_THREADS)
                    cute.arch.mbarrier_init(sKFree + 1, PV_THREADS)
                if cutlass.const_expr(recycle):
                    # FP8 converters or BF16 PV release raw slots. QK reads
                    # precede BF16 PV via pipeline commit/wait.
                    for i in cutlass.range_constexpr(ring_tiles):
                        cute.arch.mbarrier_init(
                            sRawFree + i, CONV_THREADS if is_fp8 else PV_THREADS
                        )
                cute.arch.mbarrier_init_fence()
            if cutlass.const_expr(not external_gather):
                # Standalone slots and length are ready at entry. Issue their
                # independent loads together; clamp masked indices to TOPK-1.
                n_rows = cutlass.const_expr(slots_per_cta // 32)
                r_slot = cute.make_rmem_tensor((n_rows,), Int32)
                for j in cutlass.range_constexpr(n_rows):
                    idx_rd = self._canonical_index(
                        lane + Int32(j * 32), split, a_base, b_base
                    )
                    if idx_rd > Int32(TOPK - 1):
                        idx_rd = Int32(TOPK - 1)
                    r_slot[j] = Int32(gSlots[token, idx_rd])
                valid_len = Int32(gLens[token])
                if lane == 0:
                    sValidLen[0] = valid_len
                    if cutlass.const_expr(skip_tail):
                        rows = Int32(0)
                        if valid_len > split:
                            rows = (
                                valid_len - split + Int32(self.splits - 1)
                            ) // Int32(self.splits)
                        n_live = (rows + Int32(TILE_KV - 1)) // Int32(TILE_KV)
                        if n_live < Int32(1):
                            n_live = Int32(1)
                        if n_live > Int32(tiles):
                            n_live = Int32(tiles)
                        sValidLen[1] = n_live
                for j in cutlass.range_constexpr(n_rows):
                    r = lane + Int32(j * 32)
                    idx = self._canonical_index(r, split, a_base, b_base)
                    limit = self._row_limit(r, idx, f_len, valid_len)
                    slot = r_slot[j]
                    if idx >= limit:
                        slot = Int32(-1)
                    if slot < 0:
                        slot = Int32(-1)
                    sSlots[r] = slot
        if warp_id == W_LOAD:
            # Ungathered/unconverted padding rows must be finite for MMA.
            pad_words = cutlass.const_expr((M_ROWS - local_heads) * (ROW_ELEMS // 2))
            for i in range(lane, pad_words, 32):
                row = Int32(local_heads) + i // (ROW_ELEMS // 2)
                sQ_u32[row, i % (ROW_ELEMS // 2)] = Uint32(0)

        # Pipeline init publishes prologue writes through a full-CTA barrier.
        pipe = PipelineAsync.create(
            barrier_storage=pipe_mbar,
            num_stages=PIPE_STAGES,
            producer_group=CooperativeGroup(Agent.Thread, PRODUCER_THREADS),
            consumer_group=CooperativeGroup(Agent.Thread, CONSUMER_THREADS),
            defer_sync=False,
        )

        # W10 issues the first lap after the CTA barrier.
        if cutlass.const_expr(not external_gather):
            if warp_id == W_GATHER:
                deferred = self.issue_gather(
                    gKv,
                    sSlots.iterator,
                    sKRaw.iterator,
                    mbarFull,
                    lane,
                    None,
                    0,
                    ring_tiles,
                    Int32(sValidLen[1]) if skip_tail else None,
                    gDefer,
                )
                if cutlass.const_expr(gate is not None):
                    # This step's rows: written by the gate's producers with
                    # generic stores, read here by bulk copies.
                    if lane == 0:
                        self.spin_gate(gate)
                    cute.arch.sync_warp()
                    fence_proxy_async_global()
                    self.issue_deferred(
                        gKv, sSlots.iterator, sKRaw.iterator, mbarFull, lane, deferred
                    )
                cute.arch.sync_warp()

        # W11 bulk-loads real Q rows into sQRaw (FP8) / sQ (BF16), leaving
        # padding zero, then prefetches KV slots.
        if warp_id == W_LOAD:
            if cutlass.const_expr(gate is not None):
                if lane == 0:
                    self.spin_gate(gate)
                cute.arch.sync_warp()
            if lane == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(
                    sQFull, Int32(local_heads * row_bytes)
                )
            if lane < local_heads:
                row = lane
                if cutlass.const_expr(is_fp8):
                    dst = _smem_ptr(
                        Uint32, arena_i32, self.off_sQRaw, assumed_align=16
                    ) + row * (RAW_PITCH_FP8_ELEMS // 4)
                else:
                    dst = sQ.iterator + row * ELEMS_16BIT_PITCH
                src = gQ.iterator + Int64(
                    token * local_heads * ROW_ELEMS + row * ROW_ELEMS
                )
                # The query rows were written by generic stores (P3/P4 of
                # the same launch in engine mode): fence before the bulk read.
                fence_proxy_async_global()
                cpasync_bulk_g2s(src, dst, sQFull, row_bytes)
            if lane == 0:
                cute.arch.mbarrier_wait(sQFull, 0)
            if cutlass.const_expr(not external_gather):
                for i in range(lane, slots_per_cta, 32):
                    slot = Int32(sSlots[i])
                    if slot >= 0:
                        _ = prefetch_global_l2(gKv.iterator + Int64(slot) * ROW_ELEMS)

        if cutlass.const_expr(not is_fp8):
            # BF16: sQ is final once the bulk copies land; one CTA barrier
            # publishes the W11 wait to the producer warps.
            cute.arch.barrier()
        else:
            # FP8: the conversion warps treat Q as "tile -1"; producers (and
            # only they need sQ) are released through the BAR_CONV chain.
            if warp_id >= W_CONV0 and warp_id <= W_CONV_END:
                gtid = tidx - W_CONV0 * 32  # 0..95
                if warp_id == W_CONV0 and lane == 0:
                    cute.arch.mbarrier_wait(sQFull, 0)
                cute.arch.barrier(barrier_id=BAR_RAW, number_of_threads=CONV_THREADS)
                # rows < local_heads x 36 16B chunks; 96 lanes.
                for c in range(gtid, local_heads * 36, CONV_THREADS):
                    row = c // 36
                    ck = c % 36
                    self._convert_chunk(
                        sQRaw_u32,
                        row * (RAW_PITCH_FP8_ELEMS // 4) + ck * 4,
                        sQ_u32,
                        row * (ELEMS_16BIT_PITCH // 2) + ck * 8,
                    )
                cute.arch.barrier(barrier_id=BAR_CONV, number_of_threads=160)
                cute.arch.barrier(barrier_id=BAR_CONV, number_of_threads=160)

        if cutlass.const_expr(recycle and not external_gather):
            # Reuse waits must follow the Q hand-off: BF16 PV readers cannot
            # reach their releases until W10 has joined the CTA barrier.
            if warp_id == W_GATHER:
                self.issue_gather(
                    gKv,
                    sSlots.iterator,
                    sKRaw.iterator,
                    mbarFull,
                    lane,
                    sRawFree,
                    ring_tiles,
                    tiles,
                    Int32(sValidLen[1]) if skip_tail else None,
                )
                cute.arch.sync_warp()

        if cutlass.const_expr(external_gather):
            # Engine phase B acquires selection readiness in every lane.
            # Write sValidLen before arming tiles: release arm/acquire wait
            # publishes it to producers only after tile 2 lands.
            if warp_id == W_GATHER:
                # Each lane owns one row per phase-B tile. Ragged indices
                # read the tagged word at TOPK-1 and are masked by index.
                nb = cutlass.const_expr(tiles - PHASE_A_TILES)
                r_idx = cute.make_rmem_tensor((nb,), Int32)
                r_rd = cute.make_rmem_tensor((nb,), Int32)
                r_slot = cute.make_rmem_tensor((nb,), Int32)
                for b in cutlass.range_constexpr(nb):
                    idx_b = self._canonical_index(
                        Int32((PHASE_A_TILES + b) * TILE_KV) + lane,
                        split,
                        a_base,
                        b_base,
                    )
                    r_idx[b] = idx_b
                    idx_rd = idx_b
                    if idx_rd > Int32(TOPK - 1):
                        idx_rd = Int32(TOPK - 1)
                    r_rd[b] = idx_rd
                valid_len = Int32(0)
                known_short = Int32(0)
                if cutlass.const_expr(gPositions is not None):
                    if gPositions[NUM_TOKENS - 1] < Int32(TOPK):
                        known_short = Int32(1)
                        valid_len = gPositions[token] + Int32(1)
                        shift = valid_len - Int32(FORCED_TOKENS)
                        if shift < Int32(0):
                            shift = Int32(0)
                        first_cur = gPositions[0]
                        for b in cutlass.range_constexpr(nb):
                            idx = r_idx[b]
                            pos = idx
                            if idx >= Int32(FORCED_TOKENS):
                                pos = idx - Int32(FORCED_TOKENS - INITIAL_TOKENS)
                            elif idx >= Int32(INITIAL_TOKENS):
                                pos = idx + shift
                            slot = Int32(-1)
                            if idx < valid_len:
                                if pos >= first_cur:
                                    slot = Int32(gOutCacheLoc[pos - first_cur])
                                else:
                                    slot = (
                                        gBlockTable[0, pos >> Int32(page_shift)]
                                        << Int32(page_shift)
                                    ) | (pos & Int32((1 << page_shift) - 1))
                            r_slot[b] = slot
                if known_short == Int32(0):
                    if cutlass.const_expr(sel_tag is not None):
                        # Poll all lane words together; each matching top
                        # byte validates its own payload during publication.
                        tag = Uint32(sel_tag) & Uint32(TAG_MASK)
                        p_len = cute.make_ptr(
                            Uint32,
                            (gLens.iterator + token).toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        )
                        word_ptrs = [
                            cute.make_ptr(
                                Uint32,
                                (
                                    gSlots.iterator + token * Int32(TOPK) + r_rd[b]
                                ).toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=4,
                            )
                            for b in range(nb)
                        ]
                        r_word = cute.make_rmem_tensor((nb,), Uint32)
                        w_len = load_acquire_sys_u32(p_len)
                        for b in cutlass.range_constexpr(nb):
                            r_word[b] = load_acquire_sys_u32(word_ptrs[b])
                        pending = Int32(1)
                        while pending != Int32(0):
                            pending = Int32(0)
                            if (w_len >> Uint32(TAG_SHIFT)) != tag:
                                pending = Int32(1)
                            for b in cutlass.range_constexpr(nb):
                                if (r_word[b] >> Uint32(TAG_SHIFT)) != tag:
                                    pending = Int32(1)
                            if pending != Int32(0):
                                w_len = load_acquire_sys_u32(p_len)
                                for b in cutlass.range_constexpr(nb):
                                    r_word[b] = load_acquire_sys_u32(word_ptrs[b])
                        valid_len = self._untag_word(w_len)
                        for b in cutlass.range_constexpr(nb):
                            r_slot[b] = self._untag_word(r_word[b])
                    else:
                        if cutlass.const_expr(gSelFlag is not None):
                            target = Uint32(sel_target)
                            flag = load_acquire_sys_u32(gSelFlag.iterator)
                            while (flag - target).to(Int32) < Int32(0):
                                flag = load_acquire_sys_u32(gSelFlag.iterator)
                        # Load length and lane-owned slots together; no SMEM
                        # slot table is needed for these copies.
                        valid_len = Int32(gLens[token])
                        for b in cutlass.range_constexpr(nb):
                            r_slot[b] = Int32(gSlots[token, r_rd[b]])
                n_active = Int32(tiles)
                if cutlass.const_expr(skip_tail):
                    rows = Int32(0)
                    if valid_len > split:
                        rows = (valid_len - split + Int32(self.splits - 1)) // Int32(
                            self.splits
                        )
                    n_active = (rows + Int32(TILE_KV - 1)) // Int32(TILE_KV)
                    if n_active < Int32(SKIP_FROM):
                        n_active = Int32(SKIP_FROM)
                    if n_active > Int32(tiles):
                        n_active = Int32(tiles)
                if lane == 0:
                    sValidLen[0] = valid_len
                    sValidLen[1] = n_active
                if cutlass.const_expr(self.row_map == "strided"):
                    # Phase B includes current-step KV rows. The engine
                    # acquired P4's generic stores before P5; fence the async
                    # proxy before bulk copies read them.
                    fence_proxy_async_global()
                for b in cutlass.range_constexpr(nb):
                    t = PHASE_A_TILES + b
                    live = cutlass.Boolean(True)
                    if cutlass.const_expr(skip_tail):
                        if Int32(t) >= Int32(skip_from):
                            live = n_active > Int32(t)
                    if live:
                        ring_slot = t % ring_tiles
                        if cutlass.const_expr(t >= ring_tiles):
                            # Wait for the last reader of t-ring_tiles; W10
                            # owes it no further barrier (see issue_gather).
                            cute.arch.mbarrier_wait(
                                sRawFree + ring_slot, ((t // ring_tiles) - 1) & 1
                            )
                        if lane == 0:
                            # The first arm also releases ``sValidLen`` to the
                            # producers' acquire on this tile.
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                mbarFull + ring_slot, Int32(TILE_KV * row_bytes)
                            )
                        r = Int32(t * TILE_KV) + lane
                        idx = r_idx[b]
                        slot = r_slot[b]
                        limit = self._row_limit(r, idx, f_len, valid_len)
                        if idx >= limit:
                            slot = Int32(-1)
                        if slot < 0:
                            # Masked row: land a finite row so P == 0 kills it.
                            slot = Int32(0)
                        src = gKv.iterator + Int64(slot) * ROW_ELEMS
                        dst = (
                            sKRaw.iterator
                            + (ring_slot * TILE_KV + lane) * raw_row_words
                        )
                        if cutlass.const_expr(self.row_map == "strided"):
                            cpasync_bulk_g2s(src, dst, mbarFull + ring_slot, row_bytes)
                        else:
                            # Row 127 (the initial token) is the engine's.
                            if r != Int32(INITIAL_ROW):
                                cpasync_bulk_g2s(
                                    src, dst, mbarFull + ring_slot, row_bytes
                                )
                cute.arch.sync_warp()

        # Producers W0-1.
        if warp_id <= W_QK1:
            if cutlass.const_expr(is_fp8):
                # The BAR_CONV pair waits for converted Q (tile -1).
                cute.arch.barrier(barrier_id=BAR_CONV, number_of_threads=160)
                cute.arch.barrier(barrier_id=BAR_CONV, number_of_threads=160)
            stat_w = warp_id - W_QK0  # 0 or 1
            nt0 = stat_w * 2
            rowA = lane // 4
            prod_state = make_pipeline_state(PipelineUserType.Producer, PIPE_STAGES)
            rA_u32 = cute.make_rmem_tensor((4,), Uint32)
            rB_u32 = cute.make_rmem_tensor((2,), Uint32)
            rA = cute.recast_tensor(rA_u32, mma_dtype)
            rB = cute.recast_tensor(rB_u32, mma_dtype)
            rS = cute.make_rmem_tensor((4, 2), Float32)
            # For rowA/rowA+8, both warps share m_run after max exchange;
            # l_run covers this warp's 16 columns and is summed after the loop.
            m_run0 = cutlass.Float32(CLAMP_M)
            m_run1 = cutlass.Float32(CLAMP_M)
            l_run0 = cutlass.Float32(0.0)
            l_run1 = cutlass.Float32(0.0)
            for t in range(tiles):
                live = cutlass.Boolean(True)
                if cutlass.const_expr(skip_tail):
                    if Int32(t) >= Int32(skip_from):
                        live = Int32(sValidLen[1]) > Int32(t)
                if live:
                    pipe.producer_acquire(prod_state)
                    stage = prod_state.index
                    if cutlass.const_expr(is_fp8):
                        # Two-pass hand-off waits for converted tile t.
                        cute.arch.barrier(barrier_id=BAR_CONV, number_of_threads=160)
                        cute.arch.barrier(barrier_id=BAR_CONV, number_of_threads=160)
                    else:
                        if warp_id == W_QK0 and lane == 0:
                            cute.arch.mbarrier_wait(
                                mbarFull + (t % ring_tiles), (t // ring_tiles) & 1
                            )
                        cute.arch.barrier(
                            barrier_id=BAR_PROD, number_of_threads=PRODUCER_THREADS
                        )
                    rS.fill(0.0)
                    # S = Q @ K^T over 576 dims; b-frag == plain a-frag of the
                    # K rows (mma.sync B col-major == B^T row-major).
                    if cutlass.const_expr(is_fp8):
                        ksrc = sK16[t % PIPE_STAGES, None, None]
                    else:
                        ksrc = sK[t % ring_tiles, None, None]
                    for k in cutlass.range_constexpr(36):
                        a_row = (lane // 8 % 2) * 8 + lane % 8
                        a_k = k * 16 + (lane // 8 // 2) * 8
                        a_addr = sQ.iterator + Int32(a_row * ELEMS_16BIT_PITCH + a_k)
                        rA_u32.store(ldmatrix_b16_x4(a_addr))
                        for nt in cutlass.range_constexpr(2):
                            b_row = (nt0 + nt) * 8 + lane % 8
                            b_k = k * 16 + (lane % 16) // 8 * 8
                            b_addr = ksrc.iterator + Int32(
                                b_row * ELEMS_16BIT_PITCH + b_k
                            )
                            rB_u32.store(ldmatrix_b16_x2(b_addr))
                            rS[None, nt] = mma_sync(rA, rB, rS[None, nt])
                    # Scale, mask, and per-warp row max (rows rowA / rowA+8):
                    # 4-lane butterfly over the lanes sharing a row group.
                    m0 = cutlass.Float32(CLAMP_M)
                    m1 = cutlass.Float32(CLAMP_M)
                    # Phase B reads sValidLen after the tile wait acquired
                    # the gather warp's release arm; phase A uses only f_len.
                    row_base = t * TILE_KV
                    vlen = Int32(0)
                    if t >= PHASE_A_TILES:
                        vlen = Int32(sValidLen[0])
                    for nt in cutlass.range_constexpr(2):
                        for i in cutlass.range_constexpr(4):
                            col = (nt0 + nt) * 8 + (lane % 4) * 2 + (i % 2)
                            r = row_base + col
                            idx = self._canonical_index(r, split, a_base, b_base)
                            limit = self._row_limit(r, idx, f_len, vlen)
                            s_val = rS[i, nt] * score_scale
                            if idx >= limit:
                                s_val = cutlass.Float32(MASKED_SCORE)
                            rS[i, nt] = s_val
                            if i < 2:
                                m0 = cute.arch.fmax(m0, s_val)
                            else:
                                m1 = cute.arch.fmax(m1, s_val)
                    for off in cutlass.range_constexpr(2):
                        stride = 1 << off
                        o0 = cute.arch.shuffle_sync_bfly(
                            m0, offset=stride, mask=-1, mask_and_clamp=31
                        )
                        o1 = cute.arch.shuffle_sync_bfly(
                            m1, offset=stride, mask=-1, mask_and_clamp=31
                        )
                        m0 = cute.arch.fmax(m0, o0)
                        m1 = cute.arch.fmax(m1, o1)
                    if lane % 4 == 0:
                        sStat[stage, stat_w, rowA, 0] = m0
                        sStat[stage, stat_w, rowA + 8, 0] = m1
                    cute.arch.barrier(
                        barrier_id=BAR_PROD, number_of_threads=PRODUCER_THREADS
                    )
                    # P uses the new running max; alpha = exp2(m_old-m_new)
                    # rescales O/l from previous tiles.
                    mo0 = sStat[stage, 1 - stat_w, rowA, 0]
                    mo1 = sStat[stage, 1 - stat_w, rowA + 8, 0]
                    m_new0 = cute.arch.fmax(m_run0, cute.arch.fmax(m0, mo0))
                    m_new1 = cute.arch.fmax(m_run1, cute.arch.fmax(m1, mo1))
                    alpha0 = exp2_f32(m_run0 - m_new0)
                    alpha1 = exp2_f32(m_run1 - m_new1)
                    m_run0 = m_new0
                    m_run1 = m_new1
                    l0 = cutlass.Float32(0.0)
                    l1 = cutlass.Float32(0.0)
                    for nt in cutlass.range_constexpr(2):
                        for i in cutlass.range_constexpr(2):
                            # Separate rowA/rowA+8 updates avoid values crossing
                            # DSL if-scopes, which are lowered as closures.
                            p0 = exp2_f32(rS[i, nt] - m_new0)
                            rS[i, nt] = p0
                            l0 += p0
                            p1 = exp2_f32(rS[i + 2, nt] - m_new1)
                            rS[i + 2, nt] = p1
                            l1 += p1
                    for off in cutlass.range_constexpr(2):
                        stride = 1 << off
                        o0 = cute.arch.shuffle_sync_bfly(
                            l0, offset=stride, mask=-1, mask_and_clamp=31
                        )
                        o1 = cute.arch.shuffle_sync_bfly(
                            l1, offset=stride, mask=-1, mask_and_clamp=31
                        )
                        l0 += o0
                        l1 += o1
                    # Pack c-fragment P into the MMA A-fragment (16-B store):
                    #   a0=(rowA, cols), a1=(rowA+8, cols),
                    #   a2/a3=the second n-tile's cols.
                    # Even c-fragment columns must occupy each word's low half
                    # to match the hardware A-fragment layout.
                    if cutlass.const_expr(is_fp8):
                        w_a0 = pack_f16x2(rS[0, 0], rS[1, 0])
                        w_a1 = pack_f16x2(rS[2, 0], rS[3, 0])
                        w_a2 = pack_f16x2(rS[0, 1], rS[1, 1])
                        w_a3 = pack_f16x2(rS[2, 1], rS[3, 1])
                    else:
                        w_a0 = pack_bf16x2(rS[0, 0], rS[1, 0])
                        w_a1 = pack_bf16x2(rS[2, 0], rS[3, 0])
                        w_a2 = pack_bf16x2(rS[0, 1], rS[1, 1])
                        w_a3 = pack_bf16x2(rS[2, 1], rS[3, 1])
                    p_frag = cute.make_rmem_tensor((4,), Uint32)
                    p_frag[0] = w_a0
                    p_frag[1] = w_a1
                    p_frag[2] = w_a2
                    p_frag[3] = w_a3
                    st_shared_u32x4(
                        sP.iterator + Int32((stage * 2 + stat_w) * 32 * 4 + lane * 4),
                        p_frag.load(),
                    )
                    l_run0 = l_run0 * alpha0 + l0
                    l_run1 = l_run1 * alpha1 + l1
                    # Publish alpha with the stage (one writer per row); the PV
                    # warps read it after the pipeline's consumer_wait.
                    if stat_w == 0 and lane % 4 == 0:
                        sAlpha[stage, rowA] = alpha0
                        sAlpha[stage, rowA + 8] = alpha1
                    pipe.producer_commit(prod_state)
                    prod_state.advance()
            # Exchange partial l through stage 0 (unread by consumers);
            # m is identical in both warps.
            if lane % 4 == 0:
                sStat[0, stat_w, rowA, 1] = l_run0
                sStat[0, stat_w, rowA + 8, 1] = l_run1
            cute.arch.barrier(barrier_id=BAR_PROD, number_of_threads=PRODUCER_THREADS)
            if stat_w == 0 and lane % 4 == 0:
                sMergeM[rowA] = m_run0
                sMergeM[rowA + 8] = m_run1
                sMergeL[rowA] = l_run0 + sStat[0, 1, rowA, 1]
                sMergeL[rowA + 8] = l_run1 + sStat[0, 1, rowA + 8, 1]

        # PV consumers W3-6.
        if warp_id >= W_PV0 and warp_id <= W_PV_END:
            cons_state = make_pipeline_state(PipelineUserType.Consumer, PIPE_STAGES)
            rowA = lane // 4
            col0 = (warp_id - W_PV0) * 128
            rA_u32 = cute.make_rmem_tensor((4,), Uint32)
            rB_u32 = cute.make_rmem_tensor((2,), Uint32)
            rA = cute.recast_tensor(rA_u32, mma_dtype)
            rB = cute.recast_tensor(rB_u32, mma_dtype)
            rO = cute.make_rmem_tensor((4, 16), Float32)
            rO.fill(0.0)
            for t in range(tiles):
                live = cutlass.Boolean(True)
                if cutlass.const_expr(skip_tail):
                    if Int32(t) >= Int32(skip_from):
                        live = Int32(sValidLen[1]) > Int32(t)
                if live:
                    pipe.consumer_wait(cons_state)
                    stage = cons_state.index
                    if warp_id >= W_PV0:
                        a0 = sAlpha[stage, rowA]
                        a1 = sAlpha[stage, rowA + 8]
                        for nt in cutlass.range_constexpr(16):
                            for i in cutlass.range_constexpr(4):
                                if i < 2:
                                    rO[i, nt] = rO[i, nt] * a0
                                else:
                                    rO[i, nt] = rO[i, nt] * a1
                        if cutlass.const_expr(is_fp8):
                            ksrc = sK16[t % PIPE_STAGES, None, None]
                        else:
                            ksrc = sK[t % ring_tiles, None, None]
                        # O += P @ V over the 32 KV of this tile; P comes from
                        # ONE 16-byte SMEM load per k-step (A-fragment layout).
                        for kstep in cutlass.range_constexpr(2):
                            rA_u32.store(
                                ld_shared_u32x4(
                                    sP.iterator
                                    + Int32((stage * 2 + kstep) * 32 * 4 + lane * 4)
                                )
                            )
                            for nt in cutlass.range_constexpr(16):
                                col = col0 + nt * 8
                                b_row = kstep * 16 + lane % 8 + (lane % 16) // 8 * 8
                                b_addr = ksrc.iterator + Int32(
                                    b_row * ELEMS_16BIT_PITCH + col
                                )
                                rB_u32.store(ldmatrix_b16_x2_trans(b_addr))
                                rO[None, nt] = mma_sync(rA, rB, rO[None, nt])
                        if cutlass.const_expr(is_fp8):
                            # sK16[t % 2] is free once this tile's V read is done.
                            cute.arch.mbarrier_arrive(sKFree + (t % PIPE_STAGES))
                        if cutlass.const_expr(not is_fp8 and recycle):
                            # BF16 reads the raw slot in place: PV is its last
                            # reader (QK's reads precede it through the pipeline).
                            # Only slots that will be re-armed are released.
                            if t + ring_tiles < tiles:
                                cute.arch.mbarrier_arrive(sRawFree + (t % ring_tiles))
                    pipe.consumer_release(cons_state)
                    cons_state.advance()
            if warp_id >= W_PV0:
                for nt in cutlass.range_constexpr(16):
                    col = col0 + nt * 8 + (lane % 4) * 2
                    sMerge[rowA, col] = rO[0, nt]
                    sMerge[rowA, col + 1] = rO[1, nt]
                    sMerge[rowA + 8, col] = rO[2, nt]
                    sMerge[rowA + 8, col + 1] = rO[3, nt]
        # FP8 W7-9 convert one tile ahead of PV into sK16[t % 2]. PV releases
        # tile t-2 through sKFree (initially free); BAR_CONV hands tile t to
        # producers between producer_acquire and QK MMA.
        if cutlass.const_expr(is_fp8):
            if warp_id >= W_CONV0 and warp_id <= W_CONV_END:
                gtid = tidx - W_CONV0 * 32  # 0..95
                for t in range(tiles):
                    live = cutlass.Boolean(True)
                    if cutlass.const_expr(skip_tail):
                        if Int32(t) >= Int32(skip_from):
                            live = Int32(sValidLen[1]) > Int32(t)
                    if live:
                        stage = t % PIPE_STAGES
                        if warp_id == W_CONV0 and lane == 0:
                            cute.arch.mbarrier_wait(
                                mbarFull + (t % ring_tiles), (t // ring_tiles) & 1
                            )
                        cute.arch.barrier(
                            barrier_id=BAR_RAW, number_of_threads=CONV_THREADS
                        )
                        if t >= PIPE_STAGES:
                            # Phase k releases tile stage+2k; tile t waits
                            # for t-2 with parity (t//2 - 1) & 1.
                            cute.arch.mbarrier_wait(
                                sKFree + stage, ((t // PIPE_STAGES) - 1) & 1
                            )
                        # 32 rows x 36 16B chunks over 96 lanes.
                        for c in range(gtid, TILE_KV * 36, CONV_THREADS):
                            row = c // 36
                            ck = c % 36
                            self._convert_chunk(
                                sKRaw_u32,
                                ((t % ring_tiles) * TILE_KV + row)
                                * (RAW_PITCH_FP8_ELEMS // 4)
                                + ck * 4,
                                sK16_u32,
                                (stage * TILE_KV + row) * (ELEMS_16BIT_PITCH // 2)
                                + ck * 8,
                            )
                        if cutlass.const_expr(recycle):
                            # This thread's raw reads of tile t are done; the
                            # slot is re-armed for tile t + ring_tiles.
                            if t + ring_tiles < tiles:
                                cute.arch.mbarrier_arrive(sRawFree + (t % ring_tiles))
                        cute.arch.barrier(barrier_id=BAR_CONV, number_of_threads=160)
                        cute.arch.barrier(barrier_id=BAR_CONV, number_of_threads=160)

        # DSMEM entry barrier publishes every CTA's PV stores. Each CTA
        # merges column slice bidx directly into gPartial (see _dsmem_merge).
        part_o_base = Int32((token * self.quarters + quarter) * PARTIAL_FLOATS + PART_O)
        self._dsmem_merge(
            sMerge,
            sCross,
            sMergeM,
            sMergeMB,
            sMergeL,
            sMergeLB,
            bidx,
            tidx,
            gPartial,
            part_o_base,
        )
        if cutlass.const_expr(gWvc is not None):
            assert self.off_wvcRowsA + self.off_wvcRowsB >= HEAD_PROJ, (
                "w_vc staging does not fit this ring",
                self.off_wvcRowsA,
                self.off_wvcRowsB,
            )
            # All threads passed the DSMEM entry barrier: raw ring and Q/K16
            # are dead. Stage w_vc around sLatent; signed phase tx-count lets
            # copies complete before the single arm lands.
            if split < local_heads:
                rows_a = self.off_wvcRowsA
                if tidx == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbarWvc, Int32(HEAD_PROJ * HEAD_LATENT * 2)
                    )
                if tidx < HEAD_PROJ:
                    src = gWvc.iterator + (split * Int32(HEAD_PROJ) + tidx) * Int32(
                        HEAD_LATENT
                    )
                    if tidx < rows_a:
                        cpasync_bulk_g2s(
                            src,
                            cute.make_ptr(
                                Uint8,
                                arena_i32
                                + Int32(self.off_sWvcA)
                                + tidx * Int32(WVC_PITCH),
                                cute.AddressSpace.smem,
                                assumed_align=16,
                            ),
                            mbarWvc,
                            HEAD_LATENT * 2,
                        )
                    else:
                        cpasync_bulk_g2s(
                            src,
                            cute.make_ptr(
                                Uint8,
                                arena_i32
                                + Int32(self.off_sWvcB - rows_a * WVC_PITCH)
                                + tidx * Int32(WVC_PITCH),
                                cute.AddressSpace.smem,
                                assumed_align=16,
                            ),
                            mbarWvc,
                            HEAD_LATENT * 2,
                        )

        # DSMEM already stored the unnormalized O slices; CTA 0 adds m/l,
        # which are identical across the cluster.
        if bidx == 0 and tidx < 16:
            gPartial[token, quarter, Int32(PART_M) + tidx] = sMergeMB[tidx]
            gPartial[token, quarter, Int32(PART_L) + tidx] = sMergeLB[tidx]
        # Each CTA orders its partial stores before thread 0's GPU release-add.
        # The token-wide counter wait also joins all cluster peers: each has
        # finished its DSMEM pulls before arriving, so the engine may reuse
        # this CTA's merge buffers after the wait.
        cute.arch.barrier(barrier_id=BAR_FINAL, number_of_threads=384)
        if tidx == 0:
            # Non-returning add is ordered before this thread's counter poll.
            red_add_release_gpu_u32(gCounter.iterator + token, Uint32(1))

        # Counter never resets: +splits per token/step, target (gen+1)*splits.
        # Thread 0 acquires all partials; the CTA barrier and GPU acquire
        # fence extend visibility to every reader (the CTA barrier alone
        # provides only cluster-local ordering).
        if tidx == 0:
            arrival_target = (Uint32(gen) + Uint32(1)) * Uint32(self.splits)
            arrived = load_acquire_gpu_u32(gCounter.iterator + token)
            while (arrived - arrival_target).to(Int32) < Int32(0):
                arrived = load_acquire_gpu_u32(gCounter.iterator + token)
        cute.arch.barrier(barrier_id=BAR_FINAL, number_of_threads=384)
        fence_acq_rel_gpu()

        # Split s finalizes head s: merge quarters, normalize, then project.
        if split < local_heads:
            out_scale_v = cutlass.Float32(self.out_scale)
            sLatent = cute.make_tensor(
                _smem_ptr(Float32, arena_i32, self.off_sLatent, assumed_align=16),
                cute.make_layout((HEAD_LATENT,)),
            )
            if tidx < 128:
                # One LDG.128 per quarter/thread covers 512 floats in 128
                # threads; issue before m/l loads to overlap their latency.
                col0 = tidx * 4
                o_off = Int32(PART_O) + split * Int32(HEAD_LATENT) + col0
                rPO = cute.make_rmem_tensor((self.quarters * 4,), Float32)
                for q in cutlass.range_constexpr(self.quarters):
                    pbase = (token * self.quarters + q) * Int32(PARTIAL_FLOATS) + o_off
                    pv = cute.make_tensor(
                        cute.make_ptr(
                            Float32,
                            (gPartial.iterator + pbase).toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout((4,)),
                    ).load()
                    for j in cutlass.range_constexpr(4):
                        rPO[q * 4 + j] = pv[j]
                # Broadcast m/l per quarter; each thread derives its alphas.
                rPM = cute.make_rmem_tensor((self.quarters,), Float32)
                rPL = cute.make_rmem_tensor((self.quarters,), Float32)
                m_new = cutlass.Float32(CLAMP_M)
                for q in cutlass.range_constexpr(self.quarters):
                    pm = gPartial[token, q, Int32(PART_M) + split]
                    rPM[q] = pm
                    rPL[q] = gPartial[token, q, Int32(PART_L) + split]
                    m_new = cute.arch.fmax(m_new, pm)
                rAlpha = cute.make_rmem_tensor((self.quarters,), Float32)
                total_l = cutlass.Float32(0.0)
                for q in cutlass.range_constexpr(self.quarters):
                    a_q = exp2_f32(rPM[q] - m_new)
                    rAlpha[q] = a_q
                    total_l = total_l + rPL[q] * a_q
                norm = cutlass.Float32(0.0)
                if total_l > 0.0:
                    norm = out_scale_v / total_l
                r_acc = cute.make_rmem_tensor((4,), Float32)
                for j in cutlass.range_constexpr(4):
                    r_acc[j] = cutlass.Float32(0.0)
                for q in cutlass.range_constexpr(self.quarters):
                    a_q = rAlpha[q]
                    for j in cutlass.range_constexpr(4):
                        r_acc[j] = r_acc[j] + rPO[q * 4 + j] * a_q
                for j in cutlass.range_constexpr(4):
                    r_acc[j] = r_acc[j] * norm
                cute.make_tensor(
                    cute.make_ptr(
                        Float32,
                        arena_i32 + Int32(self.off_sLatent) + col0 * 4,
                        cute.AddressSpace.smem,
                        assumed_align=16,
                    ),
                    cute.make_layout((4,)),
                ).store(r_acc.load())
                lat16 = cute.make_rmem_tensor((4,), BFloat16)
                for j in cutlass.range_constexpr(4):
                    lat16[j] = r_acc[j].to(BFloat16)
                cute.make_tensor(
                    cute.make_ptr(
                        BFloat16,
                        arena_i32 + Int32(self.off_sLatent16) + col0 * 2,
                        cute.AddressSpace.smem,
                        assumed_align=8,
                    ),
                    cute.make_layout((4,)),
                ).store(lat16.load())
            # Publish the normalized latent row to the projection threads.
            cute.arch.barrier(barrier_id=BAR_FINAL, number_of_threads=384)
            if cutlass.const_expr(gWvc is not None):
                assert gProj is not None, "gWvc requires gProj"
                # BF16 MMA, K=512: warp w owns n-tiles 2w/2w+1 (w_vc_t
                # rows 16w..16w+15). All A row addresses alias the latent
                # row; lanes 0-3 write row 0's eight columns per n-tile.
                if tidx < PROJ_THREADS:
                    cute.arch.mbarrier_wait(mbarWvc, 0)
                    pwarp = tidx // 32
                    plane = tidx % 32
                    m8 = (plane // 8) * 8
                    r8 = plane % 8
                    rows_a = self.off_wvcRowsA
                    a_base = arena_i32 + Int32(self.off_sLatent16) + m8 * 2
                    # B row of this lane for tile t: 16*pwarp + 8*t + r8;
                    # lanes 0-15 read k 0-7 / 8-15 of tile 0, 16-31 of
                    # tile 1 -> one x4 per k-step covers both tiles.
                    b_row = pwarp * Int32(16) + (plane // 16) * Int32(8) + r8
                    k_half = ((plane // 8) % 2) * 8
                    sel = (b_row + Int32(HEAD_PROJ - rows_a)) // Int32(HEAD_PROJ)
                    b_base = (
                        arena_i32
                        + Int32(self.off_sWvcA)
                        + b_row * Int32(WVC_PITCH)
                        + sel
                        * Int32(self.off_sWvcB - self.off_sWvcA - rows_a * WVC_PITCH)
                        + k_half * 2
                    )
                    # Keep K slabs rolled for instruction-cache locality;
                    # sum the two alternating fragments only after all slabs.
                    n_slabs = HEAD_LATENT // 32
                    r_acc = cute.make_rmem_tensor((16,), Float32)
                    for i in cutlass.range_constexpr(16):
                        r_acc[i] = Float32(0.0)
                    for ks in cutlass.range(n_slabs, unroll=1):
                        av, bv0, bv1 = _proj_ldmatrix(a_base, b_base, ks)
                        for st in cutlass.range_constexpr(2):
                            bv = bv0 if st == 0 else bv1
                            for t in cutlass.range_constexpr(2):
                                k = t * 2 + st
                                d = mma_sync_bf16_ssa(
                                    av[2 * st],
                                    av[2 * st],
                                    av[2 * st + 1],
                                    av[2 * st + 1],
                                    bv[2 * t],
                                    bv[2 * t + 1],
                                    r_acc[k * 4],
                                    r_acc[k * 4 + 1],
                                    r_acc[k * 4 + 2],
                                    r_acc[k * 4 + 3],
                                )
                                for j in cutlass.range_constexpr(4):
                                    r_acc[k * 4 + j] = d[j]
                    outs = [
                        [r_acc[t * 8 + j] + r_acc[t * 8 + 4 + j] for j in range(2)]
                        for t in range(2)
                    ]
                    if plane < 4:
                        for t in cutlass.range_constexpr(2):
                            col = pwarp * Int32(16) + Int32(t * 8) + plane * 2
                            for j in cutlass.range_constexpr(2):
                                gProj[
                                    token, split * Int32(HEAD_PROJ) + col + Int32(j)
                                ] = outs[t][j].to(BFloat16)
            # Join the CTA before publishing the output completion counter.
            cute.arch.barrier(barrier_id=BAR_FINAL, number_of_threads=384)
            if cutlass.const_expr(generation is None):
                if token == NUM_TOKENS - 1 and split == 0 and tidx == 0:
                    gGen[0] = (Uint32(gen) + Uint32(1)).to(Int32)
            if cutlass.const_expr(gDone is not None):
                # Each head publishes its output: NUM_TOKENS*local_heads
                # arrivals per generation.
                if tidx == 0:
                    fence_acq_rel_gpu()
                    atomic_add_acq_rel_gpu_u32(gDone.iterator, Uint32(1))

    @cute.jit
    def _untag_word(self, word: Uint32) -> Int32:
        """Low 24 bits of a Lamport-tagged word, sign-extended (a slot id,
        ``-1`` padding, or a length)."""
        value = Int32(word & Uint32(VALUE_MASK))
        if value >= Int32(MAX_TAGGED_SLOT):
            value = value - Int32(1 << TAG_SHIFT)
        return value

    @cute.jit
    def _canonical_index(
        self, r: Int32, split: Int32, a_base: Int32, b_base: Int32
    ) -> Int32:
        """Canonical selection index of ring row ``r`` of CTA ``split``.

        Contiguous: ``a_base = 16 + 64*split`` (local-window rows [0, 64)),
        ``b_base = f_len + 63*split - 64`` (selected rows [64, 127)); row
        127 is the CTA's initial token ``split``.  Strided: ``r * splits +
        split`` (the bases are unused; may pass TOPK in a ragged last tile).
        Callers mask the row at :meth:`_row_limit`.
        """
        if cutlass.const_expr(self.row_map == "strided"):
            idx = r * Int32(self.splits) + split
        else:
            idx = a_base + r
            if r >= Int32(PHASE_A_ROWS):
                idx = b_base + r
            if r == Int32(INITIAL_ROW):
                idx = split
        return idx

    @cute.jit
    def _row_limit(self, r: Int32, idx: Int32, f_len: Int32, valid_len: Int32) -> Int32:
        """Return the exclusive canonical-index bound: f_len for forced rows,
        valid_len for selected rows. Only phase B may read valid_len, which
        is published with selection; both maps keep phase A forced.
        """
        limit = valid_len
        if cutlass.const_expr(self.row_map == "strided"):
            if idx < Int32(FORCED_TOKENS):
                limit = f_len
        else:
            if r < Int32(PHASE_A_ROWS):
                limit = f_len
            if r == Int32(INITIAL_ROW):
                limit = f_len
        return limit

    @cute.jit
    def issue_gather(
        self,
        gKv: cute.Tensor,
        sSlots_ptr: cute.Pointer,
        sK_ptr: cute.Pointer,
        mbar_ptr: cute.Pointer,
        lane: Int32,
        raw_free_ptr=None,
        first_tile: cutlass.Constexpr = 0,
        end_tile: cutlass.Constexpr = None,
        n_active=None,
        gDefer=None,
    ):
        """Arm and issue tiles ``[first_tile, end_tile)`` (per-lane bulk copies).

        All 32 lanes of one warp call this.  Tile ``t`` lands in ring slot
        ``t % ring_tiles``: lane 0 arms ``mbar_ptr[slot]`` for the tile's
        ``TILE_KV * row_bytes`` and every lane copies one row (a copy may
        complete before the arm lands; the pending arrival keeps the phase
        open).  ``sK_ptr`` must be the raw gather ring this kernel lays out
        first in its SMEM sub-arena.

        Reuse blocks on last readers, so the calling warp must owe them no
        barrier. Before arming tile t, every lane waits on raw_free_ptr with
        parity ``(t // ring_tiles - 1) & 1`` for tile t-ring_tiles. Per slot:
        readers wait mbarFull(p), read, arrive sRawFree; issuer waits, rearms
        mbarFull, copies land, readers wait p^1. Neither barrier can advance
        two phases past a waiter, so one parity bit suffices (no ABA).
        Arrive/wait orders generic reads before async overwrites; only bulk
        copies write raw slots. A generic write followed by an async read
        would additionally require fence.proxy.async.

        Args:
            gKv: ``[total_slots, 576]`` physical-slot KV rows.
            sSlots_ptr: SMEM pointer to this CTA's ``slots_per_cta`` int32
                slot ids (-1 = masked, gathered from slot 0 and masked in
                the score pass).
            sK_ptr: SMEM pointer (Uint32-typed) to the raw gather ring;
                row ``r`` of tile ``t`` lands at byte offset
                ``((t % ring_tiles) * TILE_KV + r) * padded_row_bytes``
                (592 B for FP8, 1168 B for BF16).
            mbar_ptr: SMEM pointer to the ``ring_tiles`` slot mbarriers.
            lane: calling thread's lane id (0..31).
            raw_free_ptr: SMEM pointer to the ``ring_tiles`` slot-free
                mbarriers; required iff the range holds a reused slot.
            first_tile / end_tile: the tile range to issue (``end_tile``
                ``None`` = the CTA's tile count).
            n_active: runtime tile count; tiles from it on are not issued
                (every row masked; the readers skip them too).  ``None``
                issues the whole range.
            gDefer: optional ``int32[8]`` slots not to copy yet (their rows
                are still being written); the tile is armed in full and
                :meth:`issue_deferred` copies them later.

        Returns:
            This lane's deferred tiles as a bit mask (bit ``t``), or 0.
        """
        is_fp8 = _is_fp8(self.kv_dtype)
        ring_tiles = cutlass.const_expr(self.ring_tiles)
        if cutlass.const_expr(end_tile is None):
            end_tile = self.tiles_per_cta
        recycle = cutlass.const_expr(end_tile > ring_tiles)
        assert recycle == (raw_free_ptr is not None), "sRawFree iff a slot is reused"
        row_words = cutlass.const_expr(
            RAW_PITCH_FP8_ELEMS // 4 if is_fp8 else ELEMS_16BIT_PITCH // 2
        )
        row_bytes = cutlass.const_expr(ROW_ELEMS if is_fp8 else ROW_ELEMS * 2)
        sSlots = cute.make_tensor(sSlots_ptr, cute.make_layout((self.slots_per_cta,)))
        deferred = Int32(0)
        if cutlass.const_expr(gDefer is not None):
            r_defer = cute.make_rmem_tensor((8,), Int32)
            for d in cutlass.range_constexpr(8):
                r_defer[d] = Int32(gDefer[d])
        for t in range(first_tile, end_tile):
            live = cutlass.Boolean(True)
            if cutlass.const_expr(n_active is not None):
                live = n_active > Int32(t)
            if live:
                ring_slot = t % ring_tiles
                if cutlass.const_expr(recycle):
                    if t >= ring_tiles:
                        cute.arch.mbarrier_wait(
                            raw_free_ptr + ring_slot, ((t // ring_tiles) - 1) & 1
                        )
                if lane == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbar_ptr + ring_slot, Int32(TILE_KV * row_bytes)
                    )
                slot = Int32(sSlots[t * TILE_KV + lane])
                if slot < 0:
                    # Finite data avoids 0*NaN in PV. Masked rows still need
                    # copies to complete the full 32-row barrier tx-count.
                    slot = Int32(0)
                hold = cutlass.Boolean(False)
                if cutlass.const_expr(gDefer is not None):
                    for d in cutlass.range_constexpr(8):
                        if slot == r_defer[d]:
                            hold = cutlass.Boolean(True)
                if hold:
                    deferred = deferred | (Int32(1) << Int32(t))
                else:
                    src = gKv.iterator + Int64(slot) * ROW_ELEMS
                    dst = sK_ptr + (ring_slot * TILE_KV + lane) * row_words
                    cpasync_bulk_g2s(src, dst, mbar_ptr + ring_slot, row_bytes)
        return deferred

    @cute.jit
    def spin_gate(self, gate) -> None:
        """Acquire-spin until every ``(ticket, target)`` pair is met (signed
        difference, as the engine's own gates)."""
        for ticket, target in gate:
            seen = load_acquire_gpu_u32(ticket)
            while (seen - target).to(Int32) < Int32(0):
                seen = load_acquire_gpu_u32(ticket)

    @cute.jit
    def issue_deferred(
        self,
        gKv: cute.Tensor,
        sSlots_ptr: cute.Pointer,
        sK_ptr: cute.Pointer,
        mbar_ptr: cute.Pointer,
        lane: Int32,
        deferred: Int32,
    ) -> None:
        """Copy the rows :meth:`issue_gather` held back (``deferred`` bit
        ``t``: this lane's row of tile ``t``, a first-lap ring slot whose
        barrier is already armed for it)."""
        is_fp8 = _is_fp8(self.kv_dtype)
        row_words = cutlass.const_expr(
            RAW_PITCH_FP8_ELEMS // 4 if is_fp8 else ELEMS_16BIT_PITCH // 2
        )
        row_bytes = cutlass.const_expr(ROW_ELEMS if is_fp8 else ROW_ELEMS * 2)
        sSlots = cute.make_tensor(sSlots_ptr, cute.make_layout((self.slots_per_cta,)))
        for t in cutlass.range_constexpr(self.ring_tiles):
            if ((deferred >> Int32(t)) & Int32(1)) != Int32(0):
                slot = Int32(sSlots[t * TILE_KV + lane])
                src = gKv.iterator + Int64(slot) * ROW_ELEMS
                dst = sK_ptr + (t * TILE_KV + lane) * row_words
                cpasync_bulk_g2s(src, dst, mbar_ptr + t, row_bytes)

    @cute.jit
    def _convert_chunk(
        self,
        src_u32: cute.Tensor,
        src_off: Int32,
        dst_u32: cute.Tensor,
        dst_off: Int32,
    ) -> None:
        """Convert 16 FP8 bytes into eight packed f16 Uint32 words (32 B).

        ``src_off`` indexes ``src_u32`` in Uint32 units and must be 16-byte
        aligned; the eight output words land at ``dst_off .. dst_off + 7``
        (two ``STS.128``).
        """
        words = ld_shared_u32x4(src_u32.iterator + src_off)
        # Element order must be preserved: source bytes 4w..4w+3 become
        # f16 words 2w (bytes 4w, 4w+1) and 2w+1 (bytes 4w+2, 4w+3).  Q and
        # K share this routine, so any permutation cancels in the QK dot
        # product but would scramble the V columns of the PV product.
        r_lo = cute.make_rmem_tensor((4,), Uint32)
        r_hi = cute.make_rmem_tensor((4,), Uint32)
        for w in cutlass.range_constexpr(2):
            pair = fp8x4_to_fp16x4(words[w])
            r_lo[2 * w] = pair[0]
            r_lo[2 * w + 1] = pair[1]
            pair_hi = fp8x4_to_fp16x4(words[w + 2])
            r_hi[2 * w] = pair_hi[0]
            r_hi[2 * w + 1] = pair_hi[1]
        st_shared_u32x4(dst_u32.iterator + dst_off, r_lo.load())
        st_shared_u32x4(dst_u32.iterator + dst_off + 4, r_hi.load())

    @cute.jit
    def _dsmem_merge(
        self,
        sSrc: cute.Tensor,
        sAlphaScratch: cute.Tensor,
        sSrcM: cute.Tensor,
        sDstM: cute.Tensor,
        sSrcL: cute.Tensor,
        sDstL: cute.Tensor,
        bidx: Int32,
        tidx: Int32,
        gOut: cute.Tensor,
        g_base: Int32,
    ) -> None:
        """Single-round DSMEM merge of this CTA's column slice ``bidx``.

        One entry cluster barrier publishes sources. Each CTA pulls its
        ``HEAD_LATENT / ctas_per_cluster`` columns for rows [0, local_heads)
        from all peers, including itself through DSMEM. Rescale to a common
        max and sum in cluster-rank order; write O to
        ``gOut[g_base + row * HEAD_LATENT + col]`` and m/l to separate
        ``sDstM``/``sDstL`` buffers so peers can keep reading sources.
        ``sAlphaScratch`` is CTA-local fp32 ``(M_ROWS, ctas_per_cluster)``.
        """
        local_heads = cutlass.const_expr(self.local_heads)
        ctas = cutlass.const_expr(self.ctas_per_cluster)
        ncols = HEAD_LATENT // ctas
        chunks_per_row = cutlass.const_expr(ncols // 4)
        n_chunks = cutlass.const_expr(local_heads * chunks_per_row)
        chunks_per_thread = cutlass.const_expr(-(-n_chunks // 384))
        col0 = bidx * Int32(ncols)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        rM = cute.make_rmem_tensor((ctas,), Float32)
        rL = cute.make_rmem_tensor((ctas,), Float32)
        if tidx < 16:
            for q in cutlass.range_constexpr(ctas):
                rM[q] = ld_shared_cluster_f32(
                    mapa_shared_cluster_u32(sSrcM.iterator, Int32(q)) + tidx * 4
                )
                rL[q] = ld_shared_cluster_f32(
                    mapa_shared_cluster_u32(sSrcL.iterator, Int32(q)) + tidx * 4
                )
        r_peer = cute.make_rmem_tensor((ctas * 4 * chunks_per_thread,), Float32)
        # c = v*384+tidx maps consecutive threads to consecutive 16-B chunks
        # for conflict-free DSMEM/coalesced stores. Issue all pulls before use.
        for v in cutlass.range_constexpr(chunks_per_thread):
            c = Int32(v * 384) + tidx
            if c < Int32(n_chunks):
                row = c // Int32(chunks_per_row)
                col = col0 + (c % Int32(chunks_per_row)) * Int32(4)
                for q in cutlass.range_constexpr(ctas):
                    peer = ld_shared_cluster_f32x4(
                        mapa_shared_cluster_u32(sSrc.iterator, Int32(q))
                        + (row * Int32(MERGE_PITCH) + col) * 4
                    )
                    for j in cutlass.range_constexpr(4):
                        r_peer[(v * ctas + q) * 4 + j] = peer[j]
        if tidx < 16:
            m_new = rM[0]
            for q in cutlass.range_constexpr(1, ctas):
                m_new = cute.arch.fmax(m_new, rM[q])
            l_new = cutlass.Float32(0.0)
            for q in cutlass.range_constexpr(ctas):
                a_q = exp2_f32(rM[q] - m_new)
                sAlphaScratch[tidx, q] = a_q
                l_new = l_new + rL[q] * a_q
            sDstM[tidx] = m_new
            sDstL[tidx] = l_new
        cute.arch.barrier(barrier_id=BAR_MERGE, number_of_threads=384)
        r_out = cute.make_rmem_tensor((4,), Float32)
        for v in cutlass.range_constexpr(chunks_per_thread):
            c = Int32(v * 384) + tidx
            if c < Int32(n_chunks):
                row = c // Int32(chunks_per_row)
                col = col0 + (c % Int32(chunks_per_row)) * Int32(4)
                for j in cutlass.range_constexpr(4):
                    r_out[j] = cutlass.Float32(0.0)
                for q in cutlass.range_constexpr(ctas):
                    a_q = sAlphaScratch[row, q]
                    for j in cutlass.range_constexpr(4):
                        r_out[j] = r_out[j] + r_peer[(v * ctas + q) * 4 + j] * a_q
                cute.make_tensor(
                    cute.make_ptr(
                        Float32,
                        (
                            gOut.iterator + g_base + row * Int32(HEAD_LATENT) + col
                        ).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout((4,)),
                ).store(r_out.load())


__all__ = [
    "CLUSTERS",
    "CTAS_PER_CLUSTER",
    "FORCED_TOKENS",
    "INITIAL_TOKENS",
    "LOCAL_TOKENS",
    "PHASE_A_ROWS",
    "PHASE_A_TILES",
    "PHASE_B_ROWS",
    "INITIAL_ROW",
    "LongcatSparseMlaHeteroKernel",
    "NUM_TOKENS",
    "PARTIAL_FLOATS",
    "ARRIVALS_PER_TOKEN",
    "QUARTERS_PER_TOKEN",
    "RING_TILES",
    "SPLITS_PER_TOKEN",
    "TOPK",
    "split_geometry",
    "dynamic_smem_bytes",
]
