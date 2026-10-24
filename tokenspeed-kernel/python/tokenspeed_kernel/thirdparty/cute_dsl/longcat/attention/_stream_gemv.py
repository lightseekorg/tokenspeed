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

"""Streaming verify-8 BF16 GEMV: C[8,tile_n] = X[8,K] @ W[tile_n,K]^T.

Each CTA streams K chunks through a full/empty mbarrier SMEM ring. Direct TMA
loads ordinary BF16 matrices into SW128 layout. Eight MMA warps use
mma.sync m16n8k16, padding the eight activation rows to sixteen; their FP32
partials fold through SMEM before BF16 output. tile_n is a multiple of 8 in
[8,64]. K_CHUNK and STAGES define the default slice width and ring depth.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32
from cutlass.cute.nvgpu import cpasync
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._primitives import (
    mma_sync_bf16_ssa,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._sparse_mla_hetero_kernel import (
    ldmatrix_b16_x4,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._collective_primitives import (
    fence_proxy_async_global,
    invalidate_mbarrier,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._primitives import (
    load_acquire_gpu_u32,
    load_relaxed_gpu_f32,
    red_add_gpu_f32,
)

NUM_TOKENS = 8
# Slice width and ring depth of the streaming pipeline.
K_CHUNK = 512  # elements per streamed slice
# Both operands use TMA SW128 with 64-element rows in K-stripe-major order.
STAGES = 5
MMA_WARPS = 8
THREADS = 384
# Eight MMA warps plus the last participating warp for DMA. Hosted calls
# may leave upper warps to the engine; standalone THREADS leaves W8-10 idle.
MIN_THREADS = (MMA_WARPS + 1) * 32
BAR_GEMV = 13  # named barrier for the final fold (all participating threads)


def stage_bytes(tile_n: int, w_bytes: int = 2, k_chunk: int = K_CHUNK) -> int:
    """Bytes per ring stage: tile_n BF16 weight rows and eight activation rows.

    w_bytes is the weight element size; k_chunk is the slice width in elements.
    """
    return tile_n * k_chunk * w_bytes + NUM_TOKENS * k_chunk * 2


def mbar_offset(
    tile_n: int, stages: int = STAGES, w_bytes: int = 2, k_chunk: int = K_CHUNK
) -> int:
    """Byte offset of the ``Int64[2 * stages]`` mbarriers inside the arena."""
    return (
        stages * stage_bytes(tile_n, w_bytes, k_chunk)
        + MMA_WARPS * (tile_n // 8) * 32 * 2 * 4
    )


def smem_bytes(
    tile_n: int, stages: int = STAGES, w_bytes: int = 2, k_chunk: int = K_CHUNK
) -> int:
    """Total dynamic SMEM: ring + fold scratch + mbarriers (128 B aligned)."""
    offset = mbar_offset(tile_n, stages, w_bytes, k_chunk)
    return ((offset + 2 * stages * 8 + 127) // 128) * 128


def _slab_addr(base_even, base_odd, ks: int, stripe_bytes: int = 128):
    """ldmatrix address: two 32-element slabs per 64-element stripe."""
    base = base_even if ks % 2 == 0 else base_odd
    return base + Int32((ks // 2) * stripe_bytes)


def _mma_slabs(
    a_bases, b_bases: list, ks_lo: int, ks_hi: int, accs: list, b_stripe_bytes: int
) -> list:
    """Accumulate ``A[16, k] @ B_t[8, k]^T`` over k slabs ``[ks_lo, ks_hi)``
    (32 k each) into ``accs`` (one ``[c0..c3]`` fp32 fragment per B tile).
    ``a_bases``/``b_bases[i]`` are ``(even, odd)`` lane base pointers."""
    zero = Uint32(0)
    av = ldmatrix_b16_x4(_slab_addr(*a_bases, ks_lo, NUM_TOKENS * 128))
    bvs = [ldmatrix_b16_x4(_slab_addr(*b, ks_lo, b_stripe_bytes)) for b in b_bases]
    for ks in range(ks_lo, ks_hi):
        if ks + 1 < ks_hi:
            av_next = ldmatrix_b16_x4(_slab_addr(*a_bases, ks + 1, NUM_TOKENS * 128))
            bvs_next = [
                ldmatrix_b16_x4(_slab_addr(*b, ks + 1, b_stripe_bytes)) for b in b_bases
            ]
        for t, bv in enumerate(bvs):
            for s in range(2):
                accs[t] = list(
                    mma_sync_bf16_ssa(
                        av[2 * s],
                        zero,
                        av[2 * s + 1],
                        zero,
                        bv[2 * s],
                        bv[2 * s + 1],
                        *accs[t],
                    )
                )
        if ks + 1 < ks_hi:
            av = av_next
            bvs = bvs_next
    return accs


def _stream_smem_layout(tile_rows: int, k_chunk: int):
    return cute.make_composed_layout(
        cute.make_swizzle(3, 4, 3),
        0,
        cute.make_layout(
            (64, tile_rows, k_chunk // 64), stride=(1, 64, tile_rows * 64)
        ),
    )


@cute.jit
def prepare_stream_operand(
    matrix: cute.Tensor,
    tile_rows: cutlass.Constexpr,
    k_chunk: cutlass.Constexpr = K_CHUNK,
):
    """Prepare direct TMA for row-strided BF16 matrix[N,K].

    Call from host launch scope. tile_rows divides N and is a multiple of 8 in
    [8,64]; k_chunk divides K and is a multiple of 512. Inner stride is 1,
    row stride >= K is a multiple of 8, and the base is 128-byte aligned.

    Returns:
        TMA atom and tensor shaped (64,N,K//64), addressed by
        (k_inner,row,k_outer), for stream_gemv_tile.
    """
    if cutlass.const_expr(len(matrix.shape) != 2):
        raise ValueError("stream operands need BF16 [N,K] and whole tiles")
    n, k = matrix.shape
    row_stride = matrix.stride[0]
    if cutlass.const_expr(
        matrix.element_type is not BFloat16
        or matrix.stride[1] != 1
        or row_stride <= 0
        or row_stride < k
        or row_stride % 8
        or tile_rows % 8
        or not 8 <= tile_rows <= 64
        or n % tile_rows
        or k_chunk % 512
        or k % k_chunk
    ):
        raise ValueError("stream operands need row-strided BF16 [N,K] and whole tiles")
    matrix_view = cute.make_tensor(
        matrix.iterator,
        cute.make_layout((64, n, k // 64), stride=(1, row_stride, 64)),
    )
    atom, tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        matrix_view,
        _stream_smem_layout(tile_rows, k_chunk),
        (64, tile_rows, k_chunk // 64),
    )
    return atom, tensor


@cute.jit
def _copy_stream_tile(
    atom: cute.CopyAtom,
    operand_tile: cute.Tensor,
    stage_addr: Int64,
    barrier: cute.Pointer,
    tile_rows: cutlass.Constexpr,
    k_chunk: cutlass.Constexpr,
    policy: Int64 | None,
):
    shared = cute.make_tensor(
        cute.make_ptr(BFloat16, stage_addr, cute.AddressSpace.smem, assumed_align=128),
        _stream_smem_layout(tile_rows, k_chunk),
    )
    shared_part, global_part = cpasync.tma_partition(
        atom,
        0,
        cute.make_layout(1),
        cute.group_modes(shared, 0),
        cute.group_modes(operand_tile, 0),
    )
    if cutlass.const_expr(policy is None):
        cute.copy(atom, global_part, shared_part, tma_bar_ptr=barrier)
    else:
        cute.copy(
            atom, global_part, shared_part, tma_bar_ptr=barrier, cache_policy=policy
        )


@cute.jit
def stream_gemv_tile(
    tma_atom_x: cute.CopyAtom,
    gX: cute.Tensor,
    tma_atom_w: cute.CopyAtom,
    gW: cute.Tensor,
    gC: cute.Tensor,
    row0: Int32,
    arena: cute.Pointer,
    mbar: cute.Pointer,
    tidx: Int32,
    k_dim: cutlass.Constexpr,
    tile_n: cutlass.Constexpr,
    stages: cutlass.Constexpr = STAGES,
    threads: cutlass.Constexpr = THREADS,
    row_sumsq: cute.Tensor | None = None,
    sumsq_ranges: cutlass.Constexpr = None,
    rms_scale: cutlass.Constexpr = None,
    weight_policy: Int64 | None = None,
    x_ready: cute.Pointer | None = None,
    x_target: Uint32 | None = None,
    gFoldQuery: cute.Tensor | None = None,
    k_chunk: cutlass.Constexpr = K_CHUNK,
):
    """Compute gC[:,row0:row0+tile_n] = gX @ gW[row0:row0+tile_n]^T.

    Args:
        tma_atom_x, gX: Prepared TMA operand for eight BF16 activation rows.
        tma_atom_w, gW: Prepared TMA operand for this weight tile.
        gC: BF16[8,N] output.
        row0: First output column, a multiple of tile_n.
        arena: 128-byte-aligned SMEM with smem_bytes(...) bytes.
        mbar: SMEM Int64[2*stages]: full barriers, then empty barriers.
        tidx: Calling thread in [0,threads); every participant must call.
        k_dim, tile_n: Compile-time matrix dimensions.
        stages: Ring depth; arena sizing must match.
        threads: Multiple of 32, at least MIN_THREADS. Eight warps compute,
            the last warp issues DMA; all participants join both BAR_GEMV
            rendezvous and the caller's closing barrier with the same width.
        row_sumsq: FP32[slots*8], indexed slot*8+token.
        sumsq_ranges: (column_begin,column_end,slot) tuples, 8-column aligned;
            atomically add squared BF16 outputs into row_sumsq.
        rms_scale: (dim,eps,slot); multiply outputs by
            rsqrt(row_sumsq[slot*8+token]/dim+eps) before BF16 rounding.
            Caller must acquire the ticket completing those row statistics.
        weight_policy: Optional weight-stream L2 cache policy.
        x_ready, x_target: Optional GPU-acquire ticket for activation TMA only.
            Prefetch the first stages weight chunks before the acquire; fence
            the async proxy after it to observe generic stores on other SMs.
        gFoldQuery: Optional [8,H,576] query. BF16-round NOPE512, convert to
            query dtype and store here; leave RoPE64 in gC[8,H*192,128:192]
            for subsequent rotation. Weights have 576 rows per head.
        k_chunk: Slice width in K elements; both prepared operands and SMEM
            sizing must use this width. Consumer K/fold order depends on it.
    """
    if cutlass.const_expr(threads % 32 or threads < MIN_THREADS):
        raise ValueError(f"threads must be a multiple of 32 >= {MIN_THREADS}")
    if cutlass.const_expr(tile_n % 8 or not 8 <= tile_n <= 64):
        # Groups and fold warps must cover every column: nonmultiples of 8
        # or more than eight groups would leave outputs unwritten.
        raise ValueError(f"tile_n {tile_n} must be a multiple of 8 in [8, 64]")
    if cutlass.const_expr(k_dim % k_chunk or k_chunk % 512):
        # A warp's slabs must start on an even slab: the two SW128 lane
        # addresses below assume it.
        raise ValueError(
            f"k_chunk {k_chunk} must be a multiple of 512 dividing {k_dim}"
        )
    groups = tile_n // 8
    chunks = k_dim // k_chunk
    row_bytes = cutlass.const_expr(k_chunk * 2)
    dma_warp = cutlass.const_expr(threads // 32 - 1)
    warp = tidx // 32
    lane = tidx % 32
    arena_addr = Int64(arena.toint())
    w_bytes = 2
    # Weight rows of a stage: k_chunk elements of w_bytes each.
    w_pitch = cutlass.const_expr(k_chunk * w_bytes)
    stage_tx = cutlass.const_expr(tile_n * w_pitch + NUM_TOKENS * row_bytes)
    s_bytes = stage_bytes(tile_n, w_bytes, k_chunk)
    full = mbar
    empty = mbar + stages
    # Order the preceding routine's shared stores before the DMA reuses it.
    cute.arch.fence_proxy("async.shared", space="cta")
    if tidx == 0:
        for s in cutlass.range_constexpr(stages):
            cute.arch.mbarrier_init(full + s, 1)
            cute.arch.mbarrier_init(empty + s, MMA_WARPS)
        cute.arch.mbarrier_init_fence()
    cute.arch.barrier(barrier_id=BAR_GEMV, number_of_threads=threads)

    # ---- DMA warp: keep stages slices in flight -------------------------
    if warp == dma_warp:
        tile_idx = row0 // tile_n
        weight_chunks = cute.local_tile(
            gW, (64, tile_n, k_chunk // 64), (0, tile_idx, None)
        )
        activation_chunks = cute.local_tile(
            gX, (64, NUM_TOKENS, k_chunk // 64), (0, 0, None)
        )
        first = 0
        if cutlass.const_expr(x_ready is not None):
            # Prefetch weights before activation readiness. Lane 0 arms each
            # barrier before the DMA warp collectively issues its TMA copy.
            first = min(stages, chunks)
            for c in cutlass.range_constexpr(first):
                if lane == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(full + c, Int32(stage_tx))
                cute.arch.sync_warp()
                _copy_stream_tile(
                    tma_atom_w,
                    weight_chunks[None, None, None, c],
                    arena_addr + Int64(c * s_bytes),
                    full + c,
                    tile_n,
                    k_chunk,
                    weight_policy,
                )
            if lane == 1:
                seen = load_acquire_gpu_u32(x_ready)
                while (seen - x_target).to(Int32) < Int32(0):
                    seen = load_acquire_gpu_u32(x_ready)
        cute.arch.sync_warp()
        # Every potential TMA issuer must observe the activation's generic
        # stores after the entry wait or the activation-ticket acquire.
        fence_proxy_async_global()
        for c in cutlass.range_constexpr(first):
            _copy_stream_tile(
                tma_atom_x,
                activation_chunks[None, None, None, c],
                arena_addr + Int64(c * s_bytes) + Int64(tile_n * w_pitch),
                full + c,
                NUM_TOKENS,
                k_chunk,
                None,
            )
        # Keep the producer loop compact; consumers retain their MMA order.
        for c in cutlass.range(first, chunks, 1, unroll=1):
            s = c % stages
            if c >= stages:
                cute.arch.mbarrier_wait(empty + s, ((c // stages) - 1) & 1)
            if lane == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(full + s, Int32(stage_tx))
            cute.arch.sync_warp()
            stage_addr = arena_addr + Int64(s * s_bytes)
            _copy_stream_tile(
                tma_atom_w,
                weight_chunks[None, None, None, c],
                stage_addr,
                full + s,
                tile_n,
                k_chunk,
                weight_policy,
            )
            _copy_stream_tile(
                tma_atom_x,
                activation_chunks[None, None, None, c],
                stage_addr + Int64(tile_n * w_pitch),
                full + s,
                NUM_TOKENS,
                k_chunk,
                None,
            )

    # ---- MMA warps: 2 slabs x all groups per chunk -----------------------
    if warp < MMA_WARPS:
        accs = [[Float32(0.0)] * 4 for _ in range(groups)]
        q = lane // 8  # 16-byte unit within the slab
        r8 = lane % 8
        slabs_per_warp = (k_chunk // 32) // MMA_WARPS  # even, so warp slabs start even
        for c in range(chunks):
            s = c % stages
            cute.arch.mbarrier_wait(full + s, (c // stages) & 1)
            stage_addr = arena_addr + Int64(s * s_bytes)
            # Warp w owns slabs [w*spw, (w+1)*spw) of the chunk: fold the
            # warp's k offset (a multiple of 128 B) into the bases so the
            # slab loop below is static.
            k_off = Int64(warp * slabs_per_warp * 64)
            a_stage_addr = stage_addr + Int64(tile_n * w_pitch)
            a_phase = (a_stage_addr // Int64(128) + Int64(r8)) % Int64(8)
            a_even = (Int64(q) ^ a_phase) * Int64(16)
            a_odd = (Int64(4 + q) ^ a_phase) * Int64(16)
            a_row = (
                a_stage_addr
                + (k_off // Int64(128)) * Int64(NUM_TOKENS * 128)
                + Int64(r8 * 128)
            )
            a_bases = (
                cute.make_ptr(
                    Uint8, a_row + a_even, cute.AddressSpace.smem, assumed_align=16
                ),
                cute.make_ptr(
                    Uint8, a_row + a_odd, cute.AddressSpace.smem, assumed_align=16
                ),
            )
            # TMA SW128 uses the absolute shared row phase even when the
            # arena is only 128B aligned: unit = q XOR ((row + phase) % 8).
            # Each K stripe spans all tile_n rows; tile_n is a multiple of 8.
            b_phase = (stage_addr // Int64(128) + Int64(r8)) % Int64(8)
            b_even = (Int64(q) ^ b_phase) * Int64(16)
            b_odd = (Int64(4 + q) ^ b_phase) * Int64(16)
            b_rows = [
                stage_addr
                + (k_off // Int64(128)) * Int64(tile_n * 128)
                + Int64((g * 8 + r8) * 128)
                for g in range(groups)
            ]
            b_bases = [
                (
                    cute.make_ptr(
                        Uint8,
                        b_row + b_even,
                        cute.AddressSpace.smem,
                        assumed_align=16,
                    ),
                    cute.make_ptr(
                        Uint8,
                        b_row + b_odd,
                        cute.AddressSpace.smem,
                        assumed_align=16,
                    ),
                )
                for b_row in b_rows
            ]
            accs = _mma_slabs(a_bases, b_bases, 0, slabs_per_warp, accs, tile_n * 128)
            cute.arch.sync_warp()
            if lane == 0:
                cute.arch.mbarrier_arrive(empty + s)
        # Park (c0, c1) -- rows 0-7 are the tokens, rows 8-15 padding.
        fold = cute.make_ptr(
            Float32,
            arena_addr + Int64(stages * s_bytes),
            cute.AddressSpace.smem,
            assumed_align=16,
        )
        for g in cutlass.range_constexpr(groups):
            slot = cute.make_tensor(
                fold + ((warp * groups + g) * 32 + lane) * 2, cute.make_layout((2,))
            )
            slot[0] = accs[g][0]
            slot[1] = accs[g][1]
    cute.arch.barrier(barrier_id=BAR_GEMV, number_of_threads=threads)

    if tidx == 0:
        for s in cutlass.range_constexpr(stages):
            invalidate_mbarrier(full + s)
            invalidate_mbarrier(empty + s)

    # ---- fold the 8 warps' partials, one warp per group -----------------
    if warp < groups:
        fold = cute.make_ptr(
            Float32,
            arena_addr + Int64(stages * s_bytes),
            cute.AddressSpace.smem,
            assumed_align=16,
        )
        c0 = Float32(0.0)
        c1 = Float32(0.0)
        for w in cutlass.range_constexpr(MMA_WARPS):
            slot = cute.make_tensor(
                fold + ((w * groups + warp) * 32 + lane) * 2, cute.make_layout((2,))
            )
            c0 = c0 + slot[0]
            c1 = c1 + slot[1]
        # c0, c1: token ``lane // 4``, columns ``(lane % 4) * 2 + {0, 1}``.
        token = lane // 4
        col = row0 + warp * 8 + (lane % 4) * 2
        if cutlass.const_expr(rms_scale is not None):
            rms_dim, rms_eps, rms_slot = rms_scale
            factor = cute.math.rsqrt(
                load_relaxed_gpu_f32(
                    row_sumsq.iterator + (rms_slot * NUM_TOKENS + token)
                )
                / Float32(rms_dim)
                + Float32(rms_eps),
                fastmath=True,
            )
            c0 = c0 * factor
            c1 = c1 * factor
        b0 = c0.to(BFloat16)
        b1 = c1.to(BFloat16)
        out = cute.make_rmem_tensor((2,), BFloat16)
        out[0] = b0
        out[1] = b1
        if cutlass.const_expr(gFoldQuery is not None):
            # Folded Q-B has 512 absorbed channels followed by 64 RoPE
            # channels per head. Finalize still reads RoPE at 128..191.
            head = col // Int32(576)
            channel = col % Int32(576)
            if channel < Int32(512):
                gFoldQuery[token, head, channel] = b0.to(gFoldQuery.element_type)
                gFoldQuery[token, head, channel + Int32(1)] = b1.to(
                    gFoldQuery.element_type
                )
            else:
                dst = head * Int32(192) + Int32(128) + channel - Int32(512)
                gC[token, dst] = b0
                gC[token, dst + Int32(1)] = b1
        else:
            n_cols = cute.size(gC.shape[1])
            cute.make_tensor(
                cute.make_ptr(
                    BFloat16,
                    (gC.iterator + token * n_cols + col).toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                ),
                cute.make_layout((2,)),
            ).store(out.load())
        if cutlass.const_expr(sumsq_ranges is not None):
            # Square the *stored* bf16 values and fold the four lanes of the
            # token group; the eight groups of a tile are disjoint columns.
            total = b0.to(Float32) * b0.to(Float32) + b1.to(Float32) * b1.to(Float32)
            for shift in cutlass.range_constexpr(2):
                total = total + cute.arch.shuffle_sync_bfly(total, 1 << shift)
            if lane % 4 == 0:
                for i in cutlass.range_constexpr(len(sumsq_ranges)):
                    begin, end, stat_slot = sumsq_ranges[i]
                    if col >= begin and col < end:
                        red_add_gpu_f32(
                            row_sumsq.iterator + (stat_slot * NUM_TOKENS + token), total
                        )


def make_folded_q_b_weight(
    q_b_folded: torch.Tensor, w_kc: torch.Tensor
) -> torch.Tensor:
    """Compose gamma-folded Q-B NOPE weights with w_kc once.

    Args:
        q_b_folded: BF16[H*192,K] Q-B weights with gamma folded in.
        w_kc: BF16[H,512,128] absorbed-query weights.

    Returns:
        Contiguous BF16[H*576,K]: 512 absorbed NOPE rows and 64 unchanged RoPE
        rows per head. Compose in FP32 with TF32 disabled, then round to BF16;
        this removes the original runtime Q-B intermediate BF16 rounding.
    """
    heads = w_kc.shape[0]
    if (
        w_kc.shape != (heads, 512, 128)
        or q_b_folded.ndim != 2
        or q_b_folded.shape[0] != heads * 192
        or q_b_folded.dtype != torch.bfloat16
        or w_kc.dtype != torch.bfloat16
        or q_b_folded.device != w_kc.device
    ):
        raise ValueError("folded Q-B needs BF16 [H*192,K] and [H,512,128]")
    qb = q_b_folded.reshape(heads, 192, -1)
    # Preserve the process preference after one-time weight preprocessing.
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        with torch.no_grad():
            nope = torch.bmm(w_kc.float(), qb[:, :128].float()).to(torch.bfloat16)
            weight = torch.cat((nope, qb[:, 128:]), dim=1).reshape(
                heads * 576, q_b_folded.shape[1]
            )
            return weight.contiguous()
    finally:
        torch.set_float32_matmul_precision(precision)


__all__ = [
    "prepare_stream_operand",
    "make_folded_q_b_weight",
    "stream_gemv_tile",
    "smem_bytes",
    "stage_bytes",
]
