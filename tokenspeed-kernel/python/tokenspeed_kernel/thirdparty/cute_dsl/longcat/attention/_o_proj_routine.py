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

"""Fused o_proj: one CTA computes 64 columns of attn_proj @ W_o.T.

K=local_heads*128. Copy 64 BF16 weight rows before acquiring head readiness,
then copy eight activation rows and compute with mma.sync on twelve warps.
ROW_PAD separates ldmatrix row addresses into distinct 16-byte bank groups.
The caller supplies its CTA-wide barrier id, distinct from phase boundaries.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._primitives import (
    cpasync_bulk_g2s,
    mma_sync_bf16_ssa,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._sparse_mla_hetero_kernel import (
    ldmatrix_b16_x4,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._collective_primitives import (
    fence_proxy_async_global,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._primitives import (
    load_acquire_gpu_u32,
)
from tokenspeed_kernel.thirdparty.cute_dsl.topk.top_k.single_pass_multi_cta_radix_topk import (
    fence_acq_rel_gpu,
)

# Output columns one CTA owns: 8192 hidden / 128 CTAs.  Eight n-tiles of the
# m16n8k16 ``mma.sync``, one per warp of the first eight.
TILE_N = 64
# Bytes added to each staged weight / activation row so that the eight row
# addresses of one ``ldmatrix`` hit distinct 16-byte bank groups.
ROW_PAD = 16
# The cross-warp K-split hand-over: fp32 [2 halves][8 n-tiles][32 lanes][2].
REDUCE_BYTES = 16 * 64 * 4
# Two mbarriers, one per bulk-copy group; the caller owns where they sit.
MBAR_SLOTS = 2


def row_pitch(k_cols: int) -> int:
    """Padded byte pitch of one staged ``k_cols``-wide bf16 row."""
    return k_cols * 2 + ROW_PAD


def smem_bytes(k_cols: int, tokens: int) -> int:
    """SMEM the routine needs: the weight tile, the activation tile, the fold."""
    return (TILE_N + tokens) * row_pitch(k_cols) + REDUCE_BYTES


def mma_slabs(a_base, b_bases: list, ks_lo: int, ks_hi: int) -> list:
    """Accumulate 32-wide K slabs [ks_lo,ks_hi) for each B tile.

    Return four FP32 m16n8k16 C values per tile; rows 8-15 are zero padding.
    Preload the next slab's SSA fragments, accumulate alternating K steps into
    two fragments, then fold them in the existing order.
    """
    zero = Uint32(0)
    accs = [[Float32(0.0)] * 4 for _ in range(2 * len(b_bases))]
    av = ldmatrix_b16_x4(a_base + Int32(ks_lo * 64))
    bvs = [ldmatrix_b16_x4(b + Int32(ks_lo * 64)) for b in b_bases]
    for ks in range(ks_lo, ks_hi):
        if ks + 1 < ks_hi:
            av_next = ldmatrix_b16_x4(a_base + Int32((ks + 1) * 64))
            bvs_next = [ldmatrix_b16_x4(b + Int32((ks + 1) * 64)) for b in b_bases]
        for t, bv in enumerate(bvs):
            for s in range(2):
                k = t * 2 + s
                accs[k] = list(
                    mma_sync_bf16_ssa(
                        av[2 * s],
                        zero,
                        av[2 * s + 1],
                        zero,
                        bv[2 * s],
                        bv[2 * s + 1],
                        *accs[k],
                    )
                )
        if ks + 1 < ks_hi:
            av = av_next
            bvs = bvs_next
    return [
        [accs[t * 2][j] + accs[t * 2 + 1][j] for j in range(4)]
        for t in range(len(b_bases))
    ]


@cute.jit
def oproj_tile(
    g_attn_proj: cute.Tensor,
    g_o_proj: cute.Tensor,
    g_o_local: cute.Tensor,
    t_head_ready: cute.Tensor,
    gen: Uint32,
    cta: Int32,
    tidx: Int32,
    arena_addr: Int64,
    mbar: cute.Pointer,
    *,
    k_cols: cutlass.Constexpr,
    hidden: cutlass.Constexpr,
    tokens: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    bar_id: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    materialize_output: cutlass.Constexpr = True,
    publish_collective: cutlass.Constexpr = None,
    publish_base: Int64 = 0,
    publish_slot_base: Int64 = 0,
    publish_rank: cutlass.Constexpr = 1,
) -> None:
    """Compute and optionally multicast one CTA's 64-column BF16 output tile.

    Args:
        g_attn_proj: Row-major BF16[tokens,k_cols], 128 columns per ready head.
        g_o_proj: Row-major BF16[hidden,k_cols]; this CTA reads rows cta*64:+64.
        g_o_local: Row-major BF16[tokens,hidden], stored with materialize_output.
        t_head_ready: One monotonic word per head; target gen*tokens.
        arena_addr: 128-byte-aligned SMEM, sized by smem_bytes(k_cols,tokens).
        mbar: MBAR_SLOTS Int64 barriers initialized by this routine.
        bar_id, threads: CTA-wide named barrier id and participating thread count.
        materialize_output: Store this rank's partial to g_o_local.
        publish_collective: Optional collective; stage BF16-rounded output in
            retired weight SMEM and publish via publish_base/publish_slot_base/
            publish_rank to the current NVLS payload.
    """
    pitch = Int32(row_pitch(k_cols))
    row_bytes = k_cols * 2
    warp = tidx // 32
    lane = tidx % 32
    s_w = cute.make_ptr(Uint8, arena_addr, cute.AddressSpace.smem, assumed_align=128)
    s_a = cute.make_ptr(
        Uint8,
        arena_addr + Int64(TILE_N * row_pitch(k_cols)),
        cute.AddressSpace.smem,
        assumed_align=128,
    )
    mbar_w = mbar
    mbar_a = mbar + 1
    if tidx == 0:
        cute.arch.mbarrier_init(mbar_w, 1)
        cute.arch.mbarrier_init(mbar_a, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.mbarrier_arrive_and_expect_tx(mbar_w, Int32(TILE_N * row_bytes))
        cute.arch.mbarrier_arrive_and_expect_tx(mbar_a, Int32(tokens * row_bytes))
    cute.arch.barrier(barrier_id=bar_id, number_of_threads=threads)
    if tidx < TILE_N:
        cpasync_bulk_g2s(
            g_o_proj.iterator + (cta * TILE_N + tidx) * Int32(k_cols),
            s_w + tidx * pitch,
            mbar_w,
            row_bytes,
        )
    if warp == 0:
        # One lane per head counter polls the head tickets in parallel.
        if lane < Int32(heads):
            target = gen * Uint32(tokens)
            ready = load_acquire_gpu_u32(t_head_ready.iterator + lane)
            while (ready - target).to(Int32) < Int32(0):
                ready = load_acquire_gpu_u32(t_head_ready.iterator + lane)
        cute.arch.sync_warp()
    cute.arch.barrier(barrier_id=bar_id, number_of_threads=threads)
    if tidx < Int32(tokens):
        # attn_proj rows are generic stores from other SMs (acquired by the
        # gate above): order the async-proxy read behind them.
        fence_acq_rel_gpu()
        fence_proxy_async_global()
        cpasync_bulk_g2s(
            g_attn_proj.iterator + tidx * Int32(k_cols),
            s_a + tidx * pitch,
            mbar_a,
            row_bytes,
        )
    cute.arch.mbarrier_wait(mbar_w, 0)
    cute.arch.mbarrier_wait(mbar_a, 0)
    # Work split over all 12 warps, 8 n-tiles x ``slabs`` 32-wide k slabs:
    # warps 0-7 take n-tile ``w`` for slabs [0, split), warps 8-11 take
    # n-tiles ``2(w-8), 2(w-8)+1`` for slabs [split, slabs) (split = 2/3
    # balances the units) and hand their partials over through ``s_red``.
    # ldmatrix.x4 row addresses: lane l -> matrix l//8 (k offset (l//8)*8
    # within the slab), row l%8.
    slabs = k_cols // 32
    split = (2 * slabs) // 3
    s_red = cute.make_ptr(
        Float32,
        arena_addr + Int64((TILE_N + tokens) * row_pitch(k_cols)),
        cute.AddressSpace.smem,
        assumed_align=16,
    )
    m8 = (lane // 8) * 8
    r8 = lane % 8
    a_base = s_a + (r8 * pitch + m8 * 2)
    # Both roles park their fp32 (c0, c1) in ``s_red`` (rows 0-7: the warp
    # 8-11 halves, rows 8-15: the warp 0-7 halves) -- nothing assigned inside
    # a staged ``if`` is read after it.
    if warp >= 8:
        pair = (warp - 8) * 2
        b0 = s_w + ((pair * 8 + r8) * pitch + m8 * 2)
        b1 = s_w + (((pair + 1) * 8 + r8) * pitch + m8 * 2)
        accs_hi = mma_slabs(a_base, [b0, b1], split, slabs)
        for t in cutlass.range_constexpr(2):
            red_hi = cute.make_tensor(
                s_red + ((pair + t) * 64 + lane * 2), cute.make_layout((2,))
            )
            red_hi[0] = accs_hi[t][0]
            red_hi[1] = accs_hi[t][1]
    if warp < 8:
        b_base = s_w + ((warp * 8 + r8) * pitch + m8 * 2)
        acc_lo = mma_slabs(a_base, [b_base], 0, split)[0]
        red_lo = cute.make_tensor(
            s_red + ((8 + warp) * 64 + lane * 2), cute.make_layout((2,))
        )
        red_lo[0] = acc_lo[0]
        red_lo[1] = acc_lo[1]
    cute.arch.barrier(barrier_id=bar_id, number_of_threads=threads)
    # All MMA weight reads precede this barrier. Reuse the dead weight tile,
    # separately from s_red (whose FP32 partials the epilogue still reads).
    s_out = cute.make_tensor(
        cute.recast_ptr(s_w, dtype=BFloat16),
        cute.make_layout((tokens, TILE_N), stride=(TILE_N, 1)),
    )
    if warp < 8:
        r_hi = cute.make_tensor(s_red + (warp * 64 + lane * 2), cute.make_layout((2,)))
        r_lo = cute.make_tensor(
            s_red + ((8 + warp) * 64 + lane * 2), cute.make_layout((2,))
        )
        # c0, c1: row ``lane // 4``, columns ``(lane % 4) * 2 + {0, 1}``.
        out = cute.make_rmem_tensor((2,), BFloat16)
        out[0] = (r_lo[0] + r_hi[0]).to(BFloat16)
        out[1] = (r_lo[1] + r_hi[1]).to(BFloat16)
        if cutlass.const_expr(materialize_output):
            cute.make_tensor(
                cute.make_ptr(
                    BFloat16,
                    (
                        g_o_local.iterator
                        + (lane // 4) * Int32(hidden)
                        + cta * TILE_N
                        + warp * 8
                        + (lane % 4) * 2
                    ).toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                ),
                cute.make_layout((2,)),
            ).store(out.load())
        if cutlass.const_expr(publish_collective is not None):
            s_out[lane // 4, warp * 8 + (lane % 4) * 2] = out[0]
            s_out[lane // 4, warp * 8 + (lane % 4) * 2 + 1] = out[1]
    if cutlass.const_expr(publish_collective is not None):
        cute.arch.barrier(barrier_id=bar_id, number_of_threads=threads)
        publish_collective.publish_tile_fragments(
            s_out,
            publish_base,
            publish_slot_base,
            Int32(publish_rank),
            cta * TILE_N,
            tidx,
            Int32(threads),
            token_rows=tokens,
            cols=TILE_N,
        )
        # The caller's next CTA barrier (or this routine's next prologue)
        # must complete before the weight arena can be overwritten.


__all__ = [
    "MBAR_SLOTS",
    "REDUCE_BYTES",
    "ROW_PAD",
    "TILE_N",
    "mma_slabs",
    "oproj_tile",
    "row_pitch",
    "smem_bytes",
]
