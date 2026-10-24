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

"""Attention branch 1 hosted by the production layer engine.

The TP8 worker uses eight heads, twelve CTAs per verify token, two-CTA
clusters and tcgen05 o_proj with direct transport publication. Its phases are::

    qkv_a -> q_b -> absorbed_query -> finalize -> sparse_mla
            -> o_proj -> allreduce_norm

MLP0 writes the QKV-A activation rows and releases its input-ready ticket.
The collective stores one NVFP4 packing for MLP1. KV dtype, cache extents,
softmax/dequant scales and the TP8 rank come from the serving configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._o_proj_collective import (
    LongcatOProjCollective,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._primitives import (
    make_l2_policy_evict_first,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._sparse_mla_hetero_kernel import (
    PARTIAL_FLOATS,
    TOPK,
    LongcatSparseMlaHeteroKernel,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._stream_gemv import (
    BAR_GEMV,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._stream_gemv import (
    mbar_offset as gemv_mbar_offset,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._stream_gemv import (
    prepare_stream_operand,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._stream_gemv import (
    smem_bytes as gemv_smem_bytes,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._stream_gemv import (
    stream_gemv_tile,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._wsgemv_tc import (
    LongcatWeightStreamGemv,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.engine._plan import (
    Phase,
    Prefetch,
    RoutineCall,
    RoutineDecl,
    TicketWait,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._absorbed_query import (
    LongcatAttention1AbsorbedQuery,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._attention1_finalize import (
    LongcatAttention1Finalize,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._collective_primitives import (
    fence_proxy_async_global,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._primitives import (
    atomic_add_acq_rel_gpu_u32,
    load_acquire_gpu_u32,
    red_add_relaxed_gpu_u32,
    red_add_release_gpu_u32,
)
from tokenspeed_kernel.thirdparty.cute_dsl.topk.top_k.single_pass_multi_cta_radix_topk import (
    fence_acq_rel_gpu,
)

TOKENS = 8
HIDDEN = 8192
QKV_WIDTH = 2112  # q_a 1536 | kv_a 512 | k_pe 64
Q_LORA = 1536
KV_LORA, NOPE, ROPE, QUERY_ROW = 512, 128, 64, 576
# QKV-A spans 88 tiles of 24 columns.
QKV_TILE_N = 24
Q_B_TILE_N = 48  # four tiles per head, the mapping every qb_ready reader uses
# Split Q-B tiles when the grid permits; the last sub-tile publishes
# qb_ready for the original 48-column tile.
Q_B_SPLIT = 3
QKV_TILES = QKV_WIDTH // QKV_TILE_N
Q_A_TILES = Q_LORA // QKV_TILE_N  # the 64 tiles q_b waits for
QKV_STAGES, Q_B_STAGES = 5, 3
QKV_NORM_EPS = 1e-5  # q_a / kv_a RMSNorm
THREADS = 192  # absorbed_query, finalize: warps 0-5
GEMV_THREADS = 288  # q_b: eight MMA warps and the DMA warp
# Warps 0-7 of the collective's consumers: 512 fragments a CTA in two polled
# rounds per thread. Warps 8-11 run the engine work (MLP1 weight prefetch).
COMM_THREADS = 256
NORM_BAR = 13  # the collective's CTA-wide barrier
SLOT_BYTES = 1048576  # stride of one transport slot in the symmetric buffer


@dataclass(frozen=True)
class WorkerShape:
    """What a worker program is specialized on (all compile-time).

    Args:
        heads: this rank's attention heads (8 on branch 1).
        kv_fp8: FP8 KV cache (and query); BF16 otherwise.
        splits: CTAs per verify token (12).
        kv_rows: rows of the paged KV cache view ``[kv_rows, 576]``.
        cos_rows: rows of the RoPE table ``[cos_rows, 64]``.
        cos_fp32: the RoPE table is fp32 (serving) rather than bf16.
        sm_scale / k_scale: sparse-MLA softmax scale and KV dequant scale.
        rope_interleaved: rotate pairs ``(2i, 2i+1)`` (LongCat) rather than
            ``(c, c + 32)``.
        rank: this rank in the TP8 transport, or None to build the program
            without the collective (single-GPU tests).
        indexer_rank: the rank that publishes nothing (branch 0's rank 0), or
            None when every rank produces (branch 1).
        norm_eps: epsilon of the post-attention RMSNorm.
        quant_outputs: NVFP4 packings of ``norm_out`` to produce (0 or 1).
        oproj_tc: o_proj on tcgen05 (``_wsgemv_tc``): 128-row tiles on 64
            CTAs with direct weight TMA, one K stage per head,
            each head's activation loaded once its ``head_ready`` fills.
        cluster: CTAs per physical cluster (2).
    """

    heads: int
    kv_fp8: bool
    splits: int = 12
    kv_rows: int = 4096
    cos_rows: int = 4096
    cos_fp32: bool = True
    sm_scale: float = 1.0
    k_scale: float = 1.0
    rope_interleaved: bool = True
    rank: int | None = None
    indexer_rank: int | None = None
    norm_eps: float = 1e-5
    quant_outputs: int = 0
    cluster: int = 2
    oproj_tc: bool = True

    @property
    def kv_dtype(self):
        return cutlass.Float8E4M3FN if self.kv_fp8 else BFloat16

    @property
    def cos_dtype(self):
        return Float32 if self.cos_fp32 else BFloat16

    @property
    def q_b_width(self) -> int:
        return self.heads * (NOPE + ROPE)

    @property
    def q_b_tiles(self) -> int:
        return self.q_b_width // Q_B_TILE_N

    @property
    def q_b_split(self) -> int:
        """Sub-tiles per q_b tile: ``Q_B_SPLIT`` if the grid has a CTA for
        each, else whole tiles."""
        return Q_B_SPLIT if self.q_b_tiles * Q_B_SPLIT <= self.ctas else 1

    @property
    def q_b_tile_n(self) -> int:
        """Rows per q_b streaming projection sub-tile."""
        return Q_B_TILE_N // self.q_b_split

    @property
    def abs_tiles(self) -> int:
        return self.heads * (KV_LORA // 64)

    @property
    def fin_tasks(self) -> int:
        # One query task per token (every head's rope part) and one KV task
        # per token: ``LongcatAttention1Finalize(query_per_token=True)``.
        return TOKENS + TOKENS

    @property
    def ctas(self) -> int:
        return TOKENS * self.splits

    @property
    def oproj_k(self) -> int:
        return self.heads * NOPE


def _gemv_adapter(
    *,
    a: str,
    b: str,
    c: str,
    k_dim: int,
    n_dim: int,
    x_stride: int,
    tile_n: int,
    stages: int,
    tile_base: int = 0,
    done: str,
    ready: str | None = None,
    also: str | None = None,
    row_sumsq: bool = False,
    rms: bool = False,
    split: int = 1,
    sub: str | None = None,
):
    """Streaming GEMV with split-tile publication and optional row RMS scale.

    Consecutive split sub-tiles arrive acq_rel on sub[tile]; the one completing
    split*gen publishes the tile's ready/done tickets. qkv_a accumulates q_a/kv_a
    row_sumsq; q_b uses raw q_a and gamma-folded weights, scaling by its row RMS.

    Activation TMA observes generic stores after the gate acquire and async-proxy
    fence. issue_before_gate lets the routine prefetch weights before that acquire.
    """
    extra = (also,) if also else ()

    @cute.jit
    def prepare(ctx):
        weight = cute.make_tensor(
            ctx.ptr(b, BFloat16),
            cute.make_layout((n_dim, k_dim), stride=(k_dim, 1)),
        )
        activation = cute.make_tensor(
            ctx.ptr(a, BFloat16),
            cute.make_layout((TOKENS, k_dim), stride=(x_stride, 1)),
        )
        return (
            *prepare_stream_operand(activation, TOKENS),
            *prepare_stream_operand(weight, tile_n),
        )

    @cute.jit
    def invoke(ctx):
        x_ready = None
        x_target = None
        if cutlass.const_expr(ctx.gated):
            # Issued before the gate (``Phase.issue_before_gate``): the
            # weight slices go out now, the routine's DMA lane waits for the
            # gate (and fences the async proxy) before any activation copy.
            x_ready, x_target = ctx.gate()
        else:
            fence_proxy_async_global()
        tile = ctx.tile + Int32(tile_base)
        sumsq = None
        if cutlass.const_expr(row_sumsq or rms):
            parity = ctx.gen & Uint32(1)
            sumsq = cute.make_tensor(
                ctx.ptr("row_sumsq", Float32) + Int32(parity) * Int32(16),
                cute.make_layout((16,)),
            )
        stream_gemv_tile(
            *ctx.prepared,
            cute.make_tensor(
                ctx.ptr(c, BFloat16),
                cute.make_layout((TOKENS, n_dim), stride=(n_dim, 1)),
            ),
            tile * Int32(tile_n),
            ctx.smem,
            cute.recast_ptr(ctx.smem + gemv_mbar_offset(tile_n, stages), dtype=Int64),
            ctx.tidx,
            k_dim,
            tile_n,
            stages=stages,
            threads=ctx.threads,
            row_sumsq=sumsq,
            sumsq_ranges=(
                ((0, Q_LORA, 0), (Q_LORA, Q_LORA + KV_LORA, 1)) if row_sumsq else None
            ),
            rms_scale=(Q_LORA, QKV_NORM_EPS, 0) if rms else None,
            weight_policy=make_l2_policy_evict_first(),
            x_ready=x_ready,
            x_target=x_target,
        )
        cute.arch.barrier(barrier_id=BAR_GEMV, number_of_threads=ctx.threads)
        words = [ctx.ticket(name) for name in (done,) + extra]
        slot = tile // Int32(split)
        row = None
        if cutlass.const_expr(ready is not None):
            row = ctx.ticket(ready) + slot
        sub_word = None
        if cutlass.const_expr(split > 1):
            sub_word = ctx.ticket(sub) + slot
        sub_target = Uint32(split) * ctx.gen - Uint32(1)
        tidx = ctx.tidx
        if tidx == 0:
            # One fence orders the tile's stores before every word; a release
            # per word would also wait out the previous word's round trip.
            fence_acq_rel_gpu()
            last = cutlass.Boolean(True)
            if cutlass.const_expr(split > 1):
                # The acq_rel add also acquires the sibling sub-tiles' stores,
                # and the fence below carries them into the publication.
                prev = atomic_add_acq_rel_gpu_u32(sub_word, Uint32(1))
                last = prev == sub_target
                if last:
                    fence_acq_rel_gpu()
            if last:
                if cutlass.const_expr(ready is not None):
                    red_add_relaxed_gpu_u32(row, Uint32(1))
                for word in words:
                    red_add_relaxed_gpu_u32(word, Uint32(1))

    return prepare, invoke


def _absorbed_adapter(shape: WorkerShape, routine: LongcatAttention1AbsorbedQuery):
    """Per-head absorbed query; waits on its own ``qb_ready`` tiles inside."""
    heads, qbw = shape.heads, shape.q_b_width

    @cute.jit
    def invoke(ctx):
        layouts = routine.prepare_device_routine()
        routine.absorbed_query_device_routine(
            cute.make_tensor(
                ctx.ptr("q_b_out", BFloat16),
                cute.make_layout((heads, TOKENS, NOPE), stride=(192, qbw, 1)),
            ),
            cute.make_tensor(
                ctx.ptr("w_kc", BFloat16),
                cute.make_ordered_layout((heads, KV_LORA, NOPE), order=(2, 1, 0)),
            ),
            cute.make_tensor(
                ctx.ptr("query", shape.kv_dtype),
                cute.make_layout(
                    (heads, TOKENS, KV_LORA), stride=(QUERY_ROW, heads * QUERY_ROW, 1)
                ),
            ),
            cute.make_tensor(ctx.ticket("GEN"), cute.make_layout((1,))),
            cute.make_tensor(
                ctx.ticket("qb_ready"), cute.make_layout((shape.q_b_tiles,))
            ),
            cute.make_tensor(
                ctx.ticket("abs_ready"), cute.make_layout((shape.abs_tiles,))
            ),
            cute.make_tensor(ctx.ticket("abs_done"), cute.make_layout((1,))),
            *layouts,
            ctx.smem,
            logical_worker_id=ctx.tile,
            routine_cta_rank=ctx.pair_rank,
            barrier_threads=ctx.threads,
        )

    return invoke


def _finalize_adapter(shape: WorkerShape, routine: LongcatAttention1Finalize):
    """RoPE the query tails, assemble and scatter the KV rows."""
    heads, qbw = shape.heads, shape.q_b_width

    @cute.jit
    def invoke(ctx):
        u32 = lambda name, n: cute.make_tensor(  # noqa: E731
            ctx.ticket(name), cute.make_layout((n,))
        )
        parity = Int32(ctx.gen & Uint32(1))
        stats = ctx.ptr("row_sumsq", Float32)
        routine.finalize_device_routine(
            cute.make_tensor(
                ctx.ptr("q_b_out", BFloat16),
                cute.make_layout((TOKENS, heads, 192), stride=(qbw, 192, 1)),
            ),
            cute.make_tensor(
                ctx.ptr("qkv_out", BFloat16),
                cute.make_layout((TOKENS, QKV_WIDTH), stride=(QKV_WIDTH, 1)),
            ),
            cute.make_tensor(ctx.ptr("positions", Int32), cute.make_layout((TOKENS,))),
            cute.make_tensor(
                ctx.ptr("cos_sin", shape.cos_dtype),
                cute.make_layout((shape.cos_rows, ROPE), stride=(ROPE, 1)),
            ),
            cute.make_tensor(ctx.ptr("cache_loc", Int32), cute.make_layout((TOKENS,))),
            cute.make_tensor(
                ctx.ptr("query", shape.kv_dtype),
                cute.make_layout(
                    (TOKENS, heads, QUERY_ROW), stride=(heads * QUERY_ROW, QUERY_ROW, 1)
                ),
            ),
            cute.make_tensor(
                ctx.ptr("kv_cache", shape.kv_dtype),
                cute.make_layout((shape.kv_rows, QUERY_ROW), stride=(QUERY_ROW, 1)),
            ),
            u32("GEN", 1),
            u32("qkv_done", 1),
            u32("GEN", 1),  # kv_norm_completed: unused with the epilogue norm
            u32("qb_ready", shape.q_b_tiles),
            u32("abs_ready", shape.abs_tiles),
            u32("prepare_done", 1),
            cute.make_tensor(stats + parity * Int32(16), cute.make_layout((16,))),
            cute.make_tensor(
                ctx.ptr("kv_gamma", BFloat16), cute.make_layout((KV_LORA,))
            ),
            logical_worker_id=ctx.tile,
            routine_cta_rank=ctx.pair_rank,
            barrier_threads=ctx.threads,
        )
        # Zero the idle row-statistics parity for the next generation.
        other = cute.make_tensor(
            stats + (Int32(1) - parity) * Int32(16), cute.make_layout((16,))
        )
        tile, rank, tidx = ctx.tile, ctx.pair_rank, ctx.tidx
        if tile == 0 and rank == 0 and tidx < 16:
            other[tidx] = Float32(0.0)

    return invoke


def _mla_adapter(shape: WorkerShape, routine: LongcatSparseMlaHeteroKernel):
    """Gather completed selection and run one (token,quarter) sparse-MLA cluster.

    CTA quarter*cluster+rank merges its head's partials, projects through w_vc
    into attn_proj and releases head_ready.
    """
    heads, quarters, cluster = shape.heads, routine.quarters, shape.cluster

    @cute.jit
    def invoke(ctx):
        tile = ctx.tile
        head = (tile % Int32(quarters)) * Int32(cluster) + ctx.pair_rank
        head_ready = ctx.ticket("head_ready")
        tidx = ctx.tidx
        routine.sparse_mla_device_routine(
            cute.make_tensor(
                ctx.ptr("query", shape.kv_dtype),
                cute.make_layout(
                    (TOKENS, heads, QUERY_ROW), stride=(heads * QUERY_ROW, QUERY_ROW, 1)
                ),
            ),
            cute.make_tensor(
                ctx.ptr("kv_cache", shape.kv_dtype),
                cute.make_layout((shape.kv_rows, QUERY_ROW), stride=(QUERY_ROW, 1)),
            ),
            cute.make_tensor(
                ctx.ptr("slots", Int32),
                cute.make_layout((TOKENS, TOPK), stride=(TOPK, 1)),
            ),
            cute.make_tensor(ctx.ptr("lens", Int32), cute.make_layout((TOKENS,))),
            cute.make_tensor(
                ctx.ptr("partial", Float32),
                cute.make_layout(
                    (TOKENS, quarters, PARTIAL_FLOATS),
                    stride=(quarters * PARTIAL_FLOATS, PARTIAL_FLOATS, 1),
                ),
            ),
            cute.make_tensor(ctx.ptr("counter", Int32), cute.make_layout((TOKENS,))),
            cute.make_tensor(ctx.ptr("counter", Int32), cute.make_layout((1,))),
            ctx.smem,
            tile * Int32(cluster) + ctx.pair_rank,
            tile // Int32(quarters),
            tile % Int32(quarters),
            ctx.pair_rank,
            ctx.tidx,
            gDone=None,  # head_ready is the only publish anything waits on
            generation=Int32(ctx.gen) - Int32(1),
            # Entered before the query and this step's KV rows exist: every
            # other selected row goes out at once, those wait for the gate.
            gate=ctx.gates() if ctx.gated else None,
            gDefer=(
                cute.make_tensor(
                    ctx.ptr("cache_loc", Int32), cute.make_layout((TOKENS,))
                )
                if ctx.gated
                else None
            ),
            gWvc=cute.make_tensor(
                ctx.ptr("w_vc_t", BFloat16),
                cute.make_layout((heads * NOPE, KV_LORA), stride=(KV_LORA, 1)),
            ),
            gProj=cute.make_tensor(
                ctx.ptr("attn_proj", BFloat16),
                cute.make_layout((TOKENS, heads * NOPE), stride=(heads * NOPE, 1)),
            ),
        )
        # The routine's closing CTA barrier ordered every attn_proj store of
        # this head, so one release-add covers the whole 128-column block.
        if tidx == 0 and head < Int32(heads):
            red_add_release_gpu_u32(head_ready + head, Uint32(1))

    return invoke


OPROJ_TC_STAGES = 5
# The tcgen05 o_proj's SharedStorage: 1 KiB of barriers, then five stages of
# a 128 x 128 bf16 weight tile and a 16 x 128 activation tile.
OPROJ_TC_SMEM = 1024 + OPROJ_TC_STAGES * (128 * 128 * 2 + 16 * 128 * 2)


def oproj_tc_kernel(shape: "WorkerShape") -> LongcatWeightStreamGemv:
    return LongcatWeightStreamGemv(
        n_dim=HIDDEN,
        k_dim=shape.oproj_k,
        k_splits=1,
        stage_k=NOPE,
        ab_stages=OPROJ_TC_STAGES,
        n_pad=16,
    )


def _oproj_tc_prepare(g: LongcatWeightStreamGemv, shape: "WorkerShape"):
    """The tcgen05 o_proj's MMA and operand TMA atoms (launch scope)."""
    k = shape.oproj_k

    @cute.jit
    def prepare(ctx):
        ordered = lambda name, rows: cute.make_tensor(  # noqa: E731
            ctx.ptr(name, BFloat16),
            cute.make_ordered_layout((rows, k, 1), order=(1, 0, 2)),
        )
        args = g.prepare(ordered("w_o", HIDDEN), ordered("attn_proj", TOKENS))
        if cutlass.const_expr(g.shared_storage.size_in_bytes() != OPROJ_TC_SMEM):
            raise ValueError(
                f"o_proj tc SMEM {g.shared_storage.size_in_bytes()} != {OPROJ_TC_SMEM}"
            )
        return args

    return prepare


def _oproj_tc_adapter(
    g: LongcatWeightStreamGemv,
    shape: "WorkerShape",
    collective: "LongcatOProjCollective | None" = None,
    publish_after: str | None = None,
):
    """Project one 128-row tile per CTA, with head h as K stage h.

    The collective epilogue multicasts directly, avoiding an o_local reread.
    MMAs may run before publish_after, but the slot read must wait for it.
    """

    @cute.jit
    def invoke(ctx):
        out = cute.make_tensor(
            ctx.ptr("o_local", BFloat16), cute.make_layout((TOKENS * HIDDEN,))
        )
        publish = None
        if cutlass.const_expr(collective is not None):
            bases = cute.make_tensor(ctx.ptr("bases", Int64), cute.make_layout((8,)))
            publish = (
                collective,
                bases[0],  # multicast VA
                ctx.ptr("rotation", Uint32),
                SLOT_BYTES,
                Int32(shape.rank),
                # The slot is read after an earlier collective's rotation.
                (ctx.ticket_target(publish_after),) if publish_after else (),
            )
        g.device_routine(
            *ctx.prepared,
            out,
            ctx.smem,
            ctx.tile,
            gate=cute.make_tensor(
                ctx.ticket("head_ready"), cute.make_layout((shape.heads,))
            ),
            gate_target=ctx.gen * Uint32(TOKENS),
            out_bf16=out,
            publish=publish,
        )

    return invoke


def norm_adapter(
    collective: LongcatOProjCollective,
    *,
    names: dict[str, str],
    cluster: int,
    quant_outputs: int,
    bump: str | None = None,
    bump_after: str | None = None,
    done: str | None = None,
    clearers: int = 0,
    rotated: str | None = None,
):
    """Reduce rank fragments, add residual, RMS-normalize and optionally pack.

    One cluster owns a token; each CTA handles 8192/cluster columns and uses its
    last six warps for communication. Shared by attention o_proj and MLP Down.

    bump increments once after reduction; bump_after waits for the named ticket
    so no Down worker is still reading the counter. Otherwise the entry gate
    orders the increment. done releases each CTA's output/packing readiness before
    rotation; it does not make the slot ready. The final rotation participant
    releases rotated; later in-grid producers must acquire it before reading
    the next slot. clearers re-arm retired data and join rotation participants.
    """
    if (rotated is None) != (done is None):
        raise ValueError("done hands on before the rotation; it needs rotated")
    consumers = TOKENS * cluster
    participants = consumers + clearers
    quant = quant_outputs

    @cute.jit
    def invoke(ctx):
        rows = cute.make_layout((TOKENS, HIDDEN), stride=(HIDDEN, 1))
        rotation = cute.make_tensor(
            ctx.ptr(names.get("rotation", "rotation"), Uint32), cute.make_layout((9,))
        )
        bases = cute.make_tensor(
            ctx.ptr(names.get("bases", "bases"), Int64), cute.make_layout((8,))
        )
        q_out = q_scale = q_global = None
        if cutlass.const_expr(quant >= 1):
            q_out = cute.make_tensor(
                ctx.ptr(names.get("quant_out", "quant_out"), Uint8),
                cute.make_layout((TOKENS, HIDDEN // 2), stride=(HIDDEN // 2, 1)),
            )
            q_scale = cute.make_tensor(
                ctx.ptr(names.get("quant_scale", "quant_scale"), cutlass.Float8E4M3FN),
                cute.make_layout((128 * (HIDDEN // 16),)),
            )
            q_global = cute.make_tensor(
                ctx.ptr(names.get("quant_global", "quant_global"), Float32),
                cute.make_layout((1,)),
            )
        slot = load_acquire_gpu_u32(rotation.iterator)
        collective.consume_reduce_norm(
            ctx.tidx - Int32(ctx.threads - COMM_THREADS),
            ctx.tidx,
            slot,
            bases[1],  # this rank's unicast alias of the slot
            ctx.tile * Int32(cluster) + ctx.pair_rank,
            Int32(consumers),
            ctx.tile,  # token
            ctx.pair_rank,  # which share of the row
            cute.make_tensor(
                ctx.ptr(names.get("residual", "residual"), BFloat16), rows
            ),
            cute.make_tensor(
                ctx.ptr(names.get("post_gamma", "post_gamma"), BFloat16),
                cute.make_layout((HIDDEN,)),
            ),
            cute.make_tensor(
                ctx.ptr(names.get("residual_out", "residual_out"), BFloat16), rows
            ),
            cute.make_tensor(
                ctx.ptr(names.get("norm_out", "norm_out"), BFloat16), rows
            ),
            cute.recast_ptr(ctx.smem, dtype=Float32),
            rotation,
            COMM_THREADS,
            Int32(ctx.threads),
            Int32(NORM_BAR),
            Int32(participants),
            q_out,
            q_scale,
            q_global,
            clear=clearers == 0,
            rotate=rotated is None,
        )
        if cutlass.const_expr(bump is not None):
            counter = cute.make_tensor(ctx.ptr(bump, Uint32), cute.make_layout((1,)))
            tile, rank, tidx = ctx.tile, ctx.pair_rank, ctx.tidx
            if cutlass.const_expr(bump_after is not None):
                bump_word, bump_target = ctx.ticket_target(bump_after)
            if tile == 0 and rank == 0 and tidx == 0:
                if cutlass.const_expr(bump_after is not None):
                    ready = load_acquire_gpu_u32(bump_word)
                    while (ready - bump_target).to(Int32) < Int32(0):
                        ready = load_acquire_gpu_u32(bump_word)
                counter[0] = counter[0] + Uint32(1)
        if cutlass.const_expr(done is not None):
            word = ctx.ticket(done)
            tidx = ctx.tidx
            # Every consumer thread's stores before the release; the
            # routine has returned, so its own barrier id is free.
            cute.arch.barrier(barrier_id=NORM_BAR, number_of_threads=ctx.threads)
            if tidx == 0:
                red_add_release_gpu_u32(word, Uint32(1))
        if cutlass.const_expr(rotated is not None):
            # The barrier above also ends this CTA's reads of the slot.
            rotated_word = ctx.ticket(rotated)
            rotated_here = collective.arrive_rotation(
                ctx.tidx, slot, rotation, Int32(participants)
            )
            if rotated_here:
                red_add_release_gpu_u32(rotated_word, Uint32(1))

    return invoke


def clear_adapter(
    collective: LongcatOProjCollective,
    *,
    clearers: int,
    participants: int,
    rotated: str | None = None,
):
    """Re-arm retired-slot sentinels across idle clearer CTAs.

    Each arrives in the consumers+clearers rotation and, if last, releases rotated.
    """

    @cute.jit
    def invoke(ctx):
        rotation = cute.make_tensor(ctx.ptr("rotation", Uint32), cute.make_layout((9,)))
        bases = cute.make_tensor(ctx.ptr("bases", Int64), cute.make_layout((8,)))
        # The acquire makes the last rotation's flag writes visible.
        slot = load_acquire_gpu_u32(rotation.iterator)
        collective.clear_retired_slot(
            ctx.tidx, bases[1], rotation, ctx.tile, Int32(clearers), Int32(ctx.threads)
        )
        cute.arch.barrier(barrier_id=NORM_BAR, number_of_threads=ctx.threads)
        rotated_here = collective.arrive_rotation(
            ctx.tidx, slot, rotation, Int32(participants)
        )
        if cutlass.const_expr(rotated is not None):
            rotated_word = ctx.ticket(rotated)
            if rotated_here:
                red_add_release_gpu_u32(rotated_word, Uint32(1))

    return invoke


def attention_phases(
    shape: WorkerShape, *, norm_prefetch: tuple[Prefetch, ...]
) -> tuple[Phase, ...]:
    """A1 phases between MLP0's ready ticket and MLP1's ready ticket."""
    input_ready = "mlp0_norm_done"
    norm_done, rotated, publish_after = "attn_norm_done", "attn_rotated", "mlp0_rotated"
    if (shape.heads, shape.splits, shape.cluster, shape.oproj_tc) != (8, 12, 2, True):
        raise ValueError(
            "the layer attention worker requires heads=8, splits=12, cluster=2, "
            "oproj_tc=True"
        )
    if shape.rank is None or shape.indexer_rank is not None or shape.quant_outputs != 1:
        raise ValueError(
            "the layer attention worker needs TP8 rank, no indexer rank, "
            "and one NVFP4 output"
        )
    heads, kv_dtype = shape.heads, shape.kv_dtype
    ctas = shape.ctas
    # The head's w_kc tile goes into SMEM before its Q-B tiles are waited on.
    absorbed = LongcatAttention1AbsorbedQuery(
        local_heads=heads,
        defer_pipeline_sync=True,
        kv_dtype=kv_dtype,
        prefetch_weight_before_activation=True,
    )
    finalize = LongcatAttention1Finalize(
        local_heads=heads,
        kv_dtype=kv_dtype,
        kv_norm_in_epilogue=True,
        eps=QKV_NORM_EPS,
        qkv_output_tiles=QKV_TILES,
        rope_interleaved=shape.rope_interleaved,
        # Query tasks write RoPE channels on CTAs left idle by absorbed
        # query. Sparse MLA waits for abs_done before consuming the query.
        wait_absorbed=False,
        query_per_token=True,
    )
    absorbed.prepare_device_routine()
    mla = LongcatSparseMlaHeteroKernel(
        local_heads=heads,
        kv_dtype=kv_dtype,
        sm_scale=shape.sm_scale,
        k_scale=shape.k_scale,
        splits=shape.splits,
        ctas_per_cluster=shape.cluster,
    )
    if mla.clusters * shape.cluster != ctas:
        raise ValueError(f"sparse MLA grid {mla.clusters} clusters != {ctas} CTAs")

    def gemv(name, *, tile_n, stages, threads=384, **kw):
        prepare, invoke = _gemv_adapter(tile_n=tile_n, stages=stages, **kw)
        return RoutineDecl(
            name,
            threads=threads,
            smem_bytes=gemv_smem_bytes(tile_n, stages),
            named_barriers=(BAR_GEMV,),
            host_prepare=prepare,
            invoke=invoke,
            gates_itself=True,
        )

    qkv_common = dict(
        a="x",
        b="w_qkv",
        c="qkv_out",
        k_dim=HIDDEN,
        n_dim=QKV_WIDTH,
        x_stride=HIDDEN,
        tile_n=QKV_TILE_N,
        stages=QKV_STAGES,
        row_sumsq=True,
    )
    q_a_cover = gemv("qkv_a_q_cover", done="comp_q", also="qkv_done", **qkv_common)
    rest = gemv("qkv_a_rest", done="qkv_done", tile_base=Q_A_TILES, **qkv_common)
    q_b_call = gemv(
        "q_b",
        threads=GEMV_THREADS,
        a="qkv_out",
        b="w_qb",
        c="q_b_out",
        k_dim=Q_LORA,
        n_dim=shape.q_b_width,
        x_stride=QKV_WIDTH,
        tile_n=shape.q_b_tile_n,
        stages=Q_B_STAGES,
        rms=True,
        done="qb_done",
        ready="qb_ready",
        split=shape.q_b_split,
        sub="qb_sub",
    )
    absorbed_call = RoutineDecl(
        "absorbed_query",
        threads=THREADS,
        smem_bytes=absorbed.shared_storage.size_in_bytes(),
        cluster_barriers=1,
        ctas_per_tile=2,
        named_barriers=(14, 15),
        invoke=_absorbed_adapter(shape, absorbed),
    )
    finalize_call = RoutineDecl(
        "finalize",
        threads=THREADS,
        smem_bytes=128,
        cluster_barriers=1,
        ctas_per_tile=2,
        named_barriers=(15,),
        invoke=_finalize_adapter(shape, finalize),
    )
    mla_call = RoutineDecl(
        "sparse_mla",
        threads=384,
        smem_bytes=mla.arena_bytes,
        streams=0,
        cluster_barriers=1,
        ctas_per_tile=shape.cluster,
        named_barriers=(4, 6, 7, 9, 10),
        invoke=_mla_adapter(shape, mla),
        gates_itself=True,
    )
    collective = LongcatOProjCollective(
        indexer_rank=None,
        cluster_size=shape.cluster,
        norm_eps=shape.norm_eps,
    )
    # The tcgen05 o_proj publishes from its epilogue.
    g = oproj_tc_kernel(shape)
    oproj_tc_call = RoutineDecl(
        "o_proj",
        threads=192,
        smem_bytes=OPROJ_TC_SMEM,
        smem_align=1024,
        tmem_cols=32,
        named_barriers=(1, 2, 3),
        host_prepare=_oproj_tc_prepare(g, shape),
        invoke=_oproj_tc_adapter(
            g,
            shape,
            collective,
            publish_after,
        ),
    )
    oproj_tiles = HIDDEN // 128
    if oproj_tiles > ctas:
        raise ValueError(f"the tcgen05 o_proj needs {oproj_tiles} CTAs")
    oproj_phases = [
        Phase(
            "o_proj",
            routines=(RoutineCall(oproj_tc_call, "oproj", tiles=oproj_tiles),),
        )
    ]
    collective_phases = []
    consumers = TOKENS * shape.cluster
    clearers = ctas - consumers
    norm_call = RoutineDecl(
        "allreduce_norm",
        threads=COMM_THREADS,
        smem_bytes=128,
        cluster_barriers=2,
        ctas_per_tile=shape.cluster,
        named_barriers=(NORM_BAR,),
        invoke=norm_adapter(
            collective,
            names={},
            cluster=shape.cluster,
            quant_outputs=shape.quant_outputs,
            done=norm_done,
            clearers=clearers,
            rotated=rotated,
        ),
    )
    clear_call = RoutineDecl(
        "allreduce_clear",
        threads=COMM_THREADS,
        smem_bytes=128,
        named_barriers=(NORM_BAR,),
        invoke=clear_adapter(
            collective,
            clearers=clearers,
            participants=consumers + clearers,
            rotated=rotated,
        ),
    )
    # With no idle CTAs the consumers clear, and one of them rotates.
    norm_produces = ((norm_done, consumers),) if norm_done else ()
    if rotated and not clearers:
        norm_produces += ((rotated, 1),)
    collective_phases.append(
        Phase(
            "allreduce_norm",
            prefetch=norm_prefetch,
            # Without the publish phase (it waited for ``publish_after``),
            # the CTAs that host no o_proj tile would reach the slot
            # clear before the earlier collective rotated.
            consumes=((TicketWait(publish_after),) if publish_after else ()),
            routines=(
                RoutineCall(
                    norm_call,
                    "norm",
                    tiles=TOKENS,
                    produces=norm_produces,
                ),
            )
            + (
                (
                    RoutineCall(
                        clear_call,
                        "norm",
                        tiles=clearers,
                        cta_offset=consumers,
                        # One of the consumers or the clearers rotates.
                        produces=((rotated, 1),) if rotated else (),
                    ),
                )
                if clearers
                else ()
            ),
        )
    )

    phases = (
        # QKV-A and Q-B stream their first weight stages into SMEM before
        # their inputs exist; only the activation copies wait for the gate.
        Phase(
            "qkv_a",
            consumes=(TicketWait(input_ready),),
            issue_before_gate=True,
            routines=(
                RoutineCall(
                    q_a_cover,
                    "qkv",
                    tiles=Q_A_TILES,
                    produces=(("comp_q", Q_A_TILES), ("qkv_done", Q_A_TILES)),
                ),
                RoutineCall(
                    rest,
                    "qkv",
                    tiles=QKV_TILES - Q_A_TILES,
                    cta_offset=Q_A_TILES,
                    produces=(("qkv_done", QKV_TILES - Q_A_TILES),),
                ),
            ),
        ),
        # Prefetch each head-CTA's absorbed-query weights during q_b.
        Phase(
            "q_b",
            consumes=(TicketWait("comp_q"),),
            issue_before_gate=True,
            prefetch=(
                Prefetch("w_qb", shape.q_b_width * Q_LORA * 2, before_gate=True),
                Prefetch("w_kc", heads * KV_LORA * NOPE * 2),
                Prefetch("w_vc_t", heads * NOPE * KV_LORA * 2),
            ),
            routines=(
                RoutineCall(
                    q_b_call,
                    "qb",
                    tiles=shape.q_b_tiles * shape.q_b_split,
                    # Published once per 48-column tile, by its last sub-tile.
                    produces=(
                        ("qb_done", shape.q_b_tiles),
                        ("qb_ready", shape.q_b_tiles),
                    ),
                ),
            ),
        ),
        Phase(
            "absorbed_query",
            routines=(
                RoutineCall(
                    absorbed_call,
                    "abs",
                    tiles=shape.abs_tiles // 2,
                    produces=(
                        ("abs_done", shape.abs_tiles),
                        ("abs_ready", shape.abs_tiles),
                    ),
                ),
            ),
        ),
        Phase(
            "finalize",
            routines=(
                RoutineCall(
                    finalize_call,
                    "fin",
                    tiles=shape.fin_tasks // 2,
                    # Past the absorbed query's CTAs when the grid has room:
                    # the tasks wait for the Q-B rope tiles only.
                    cta_offset=(
                        shape.abs_tiles
                        if shape.abs_tiles + shape.fin_tasks <= ctas
                        else 0
                    ),
                    produces=(("prepare_done", shape.fin_tasks),),
                ),
            ),
        ),
        Phase(
            "sparse_mla",
            consumes=(TicketWait("prepare_done"), TicketWait("abs_done")),
            # The selection is complete at entry: the routine gathers the KV
            # rows while the query is still being built (``gate`` above).
            issue_before_gate=True,
            routines=(
                RoutineCall(
                    mla_call,
                    "mla",
                    tiles=ctas // shape.cluster,
                    produces=(("head_ready", TOKENS * heads),),
                ),
            ),
        ),
        *oproj_phases,
        *collective_phases,
    )
    return phases


ATTENTION_OPERANDS = (
    "x",
    "w_qkv",
    "qkv_out",
    "w_qb",
    "q_b_out",
    "w_kc",
    "query",
    "row_sumsq",
    "kv_gamma",
    "positions",
    "cos_sin",
    "cache_loc",
    "kv_cache",
    "slots",
    "lens",
    "partial",
    "counter",
    "w_vc_t",
    "attn_proj",
    "w_o",
    "o_local",
    "bases",
    "rotation",
    "residual",
    "post_gamma",
    "residual_out",
    "norm_out",
    "quant_out",
    "quant_scale",
    "quant_global",
)


def make_worker_buffers(shape: WorkerShape, device) -> dict[str, torch.Tensor]:
    """Allocate zeroed scratch and intermediate buffers for the program.

    Caller supplies weights, step inputs, KV/selection/RoPE views, and optional
    collective bases/rotation/quant_global operands.
    """
    bf = dict(device=device, dtype=torch.bfloat16)
    kv = torch.float8_e4m3fn if shape.kv_fp8 else torch.bfloat16
    heads = shape.heads
    out = {
        "x": torch.zeros(TOKENS * HIDDEN, **bf),
        "qkv_out": torch.zeros(TOKENS, QKV_WIDTH, **bf),
        "q_b_out": torch.zeros(TOKENS, shape.q_b_width, **bf),
        "query": torch.zeros(TOKENS, heads, QUERY_ROW, device=device, dtype=kv),
        "row_sumsq": torch.zeros(32, device=device, dtype=torch.float32),
        "partial": torch.zeros(
            TOKENS,
            shape.splits // shape.cluster,
            PARTIAL_FLOATS,
            device=device,
            dtype=torch.float32,
        ),
        "counter": torch.zeros(TOKENS, device=device, dtype=torch.int32),
        "attn_proj": torch.zeros(TOKENS, shape.oproj_k, **bf),
        "o_local": torch.zeros(TOKENS, HIDDEN, **bf),
    }
    out["residual_out"] = torch.zeros(TOKENS, HIDDEN, **bf)
    out["norm_out"] = torch.zeros(TOKENS, HIDDEN, **bf)
    out["quant_out"] = torch.zeros(
        TOKENS, HIDDEN // 2, device=device, dtype=torch.uint8
    )
    out["quant_scale"] = torch.zeros(
        128 * (HIDDEN // 16), device=device, dtype=torch.float8_e4m3fn
    )
    return out


__all__ = [
    "WorkerShape",
    "make_worker_buffers",
    "attention_phases",
    "ATTENTION_OPERANDS",
]
