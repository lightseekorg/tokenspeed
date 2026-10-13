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

"""Marlin W8A16 MXFP8 MoE apply for SM90 (Hopper).

Weight-only FP8 experts: MXFP8 weights (E4M3 with one E8M0 scale per 32 input
channels) are dequantized inside the Marlin grouped GEMM and multiplied with
bf16 activations. Hopper has no MXFP8 tensor-core path, and the E8M0 x E4M3
product is exact in bf16, so this matches a bf16 GEMM over the dequantized
weights up to accumulation order.

Both GEMMs reduce split-K partial tiles in fp32 in a fixed, lock-ordered
sequence and the top-k finalize is a plain sum outside the kernel. With
``plan["deterministic_finalize"]`` the route alignment is a stable sort too,
which makes the whole output reproducible run to run.

Marlin runs on Ampere-style mma.sync, which leaves Hopper tensor cores idle
once a batch is compute bound. Above ``_DENSE_MIN_ROUTES_PER_EXPERT`` routes
per expert (prefill), the local experts are instead dequantized from the
Marlin layout into one bf16 scratch per device and fed to the CUTLASS grouped
GEMM; the dequant is exact, so only the accumulation order changes.

At the other end, batches of at most ``_GEMV_MAX_ROUTES`` routes (decode)
leave Marlin too. With a route or two per local expert, Marlin spreads one
expert's K over most of the grid and sums the partial tiles in a serial lock
chain, so the expert is read at a fraction of HBM bandwidth. GEMV kernels read
the Marlin layout directly instead, split every route over output tiles and K
ranges, sum the partials in a fixed order and round where Marlin does.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.moe.marlin.mxfp4 import _swiglu_limit
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.thirdparty.cuda.marlin import gptq_marlin_repack
from tokenspeed_kernel.thirdparty.cuda.marlin_moe import (
    FLOAT8_E4M3FN_ID,
    marlin_make_workspace,
    marlin_permute_scales,
    moe_align_block_size,
    moe_wna16_marlin_gemm,
    mxfp4_marlin_process_scales,
)

if current_platform().is_nvidia:
    from tokenspeed_kernel.ops.moe.flashinfer.cutlass_unquant import (
        cutlass_unquant_fused_moe,
    )

MXFP8_BLOCK = 32
# GEMM1 writes 2 * ispp columns and GEMM2 reduces over ispp; Marlin tiles N in
# multiples of 128 and K in multiples of 64.
_ISPP_ALIGNMENT = 64
_HIDDEN_ALIGNMENT = 128
# Marlin's 8-bit repack stores each 16 (k) x 64 (n) tile as 1024 contiguous bytes.
_TILE_K = 16
_TILE_N = 64
# H20 crossover of Marlin vs dequant + CUTLASS bf16 at HY V4 expert shapes.
_DENSE_MIN_ROUTES_PER_EXPERT = 192
# Dequant program width along K: 8 Marlin k-tiles, 4 scale groups.
_DEQUANT_K_TILES = 8
# H20 crossover: two decode tokens at top-8. GEMV wins with one or two local
# routes; one token with three already loses to Marlin (serial per-route
# loop), and four tokens stay on Marlin.
_GEMV_MAX_ROUTES = 16
# Marlin k-tiles a GEMV program reads per step. Every K the GEMVs see (hidden,
# ispp) is a multiple of _ISPP_ALIGNMENT, so steps never straddle the end.
_GEMV_K_TILES = _ISPP_ALIGNMENT // _TILE_K
# 16 programs/SM is the occupancy that beat Marlin at T=1 L=1 and T=1 L=2
# on H20 (the default 8 only won L=1).
_GEMV_PROGRAMS_PER_SM = 16
_GEMV_NUM_WARPS = 4

# One bf16 copy of a layer's local experts per (device, geometry). MoE layers
# run one at a time on the compute stream, so every layer reuses it.
_DENSE_SCRATCH: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _dense_scratch(
    device: torch.device, num_experts: int, hidden: int, ispp: int
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (device, num_experts, hidden, ispp)
    if key not in _DENSE_SCRATCH:
        _DENSE_SCRATCH[key] = (
            torch.empty(
                (num_experts, 2 * ispp, hidden), dtype=torch.bfloat16, device=device
            ),
            torch.empty(
                (num_experts, hidden, ispp), dtype=torch.bfloat16, device=device
            ),
        )
    return _DENSE_SCRATCH[key]


@dataclass(frozen=True)
class _MarlinMxfp8State:
    expert_start: int
    hidden_size: int
    ispp: int
    cutlass_swiglu_limit: torch.Tensor | None
    dense_prefill: bool


@triton.jit
def _marlin_mxfp8_to_bf16_kernel(
    q_ptr,
    s_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    ROW_SHIFT: tl.constexpr,
    K_TILES: tl.constexpr,
):
    n_tile = tl.program_id(0)
    k_block = tl.program_id(1)
    expert = tl.program_id(2).to(tl.int64)
    GROUPS: tl.constexpr = K_TILES * 16 // 32

    # Byte b of tile word ((tc_col * 4 + t4) * 8 + warp * 2 + which) holds
    # n = warp * 16 + which * 8 + tc_col, k = (b & 1) * 8 + t4 * 2 + (b >> 1).
    k_tiles = k_block * K_TILES + tl.arange(0, K_TILES)
    q_offs = (
        expert * (N * K)
        + (k_tiles.to(tl.int64)[:, None] * (N // 64) + n_tile) * 1024
        + tl.arange(0, 1024)[None, :]
    )
    w = tl.load(q_ptr + q_offs).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    w = tl.reshape(w, (K_TILES, 8, 4, 4, 2, 2, 2))
    w = tl.permute(w, (3, 4, 1, 0, 6, 2, 5))
    w = tl.reshape(w, (64, GROUPS, 32))

    # Scale byte p of a 64-wide block holds n = (p >> 2 & 1) * 32 + (p & 1) * 16
    # + (p >> 1 & 1) * 8 + (p >> 3); E8M0 e -> 2^(e - 127).
    # Loading rejects 0x00 and 0xFF before any parameter is repacked.
    groups = k_block * GROUPS + tl.arange(0, GROUPS)
    s_offs = (
        expert * (K // 32 * N)
        + groups.to(tl.int64)[:, None] * N
        + n_tile * 64
        + tl.arange(0, 64)[None, :]
    )
    s = tl.load(s_ptr + s_offs).to(tl.int32)
    s = tl.reshape(s, (GROUPS, 8, 2, 2, 2))
    s = tl.permute(s, (2, 4, 3, 1, 0))
    s = tl.reshape(s, (64, GROUPS))
    scale = (s << 23).to(tl.float32, bitcast=True)

    out = tl.reshape(w * scale[:, :, None], (64, K_TILES * 16)).to(tl.bfloat16)
    # Wrap each row, not just the tile origin: half-swapping an odd number
    # of 64-row tiles can otherwise store past the final output row.
    rows = (n_tile * 64 + ROW_SHIFT + tl.arange(0, 64)) % N
    cols = k_block * K_TILES * 16 + tl.arange(0, K_TILES * 16)
    tl.store(
        out_ptr + expert * (N * K) + rows.to(tl.int64)[:, None] * K + cols[None, :],
        out,
    )


def dequantize_marlin_mxfp8(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    out: torch.Tensor,
    *,
    swap_halves: bool,
) -> torch.Tensor:
    """Dequantize Marlin-repacked MXFP8 experts back to bf16 ``[E, N, K]``.

    Args:
        qweight: :func:`marlin_mxfp8_moe_weights` int32 weights ``[E, K/16, 4N]``.
        scales: Its float8_e8m0fnu scales ``[E, K/32, N]``.
        out: bf16 ``[E, N, K]`` destination. All three tensors must be
            contiguous and on the same CUDA device, with positive dimensions.
        swap_halves: Write the second half of N first, turning a
            ``[gate; up]`` w13 into CUTLASS's ``[up; gate]`` order.

    Returns:
        The supplied ``out`` tensor, filled in place.

    Raises:
        TypeError: A tensor has the wrong storage dtype.
        ValueError: Shapes, alignment, strides or devices violate the layout.

    Validation reads tensor metadata only and is safe during graph capture.
    Scale encodings are validated once by the weight loader before repacking.
    """
    if qweight.dtype != torch.int32:
        raise TypeError("qweight must be int32 Marlin-packed storage")
    if scales.dtype != torch.float8_e8m0fnu:
        raise TypeError("scales must be float8_e8m0fnu Marlin-packed storage")
    if out.dtype != torch.bfloat16:
        raise TypeError("out must have bf16 dtype")
    if any(t.ndim != 3 for t in (qweight, scales, out)):
        raise ValueError("qweight, scales and out must be three-dimensional")
    if any(not t.is_contiguous() for t in (qweight, scales, out)):
        raise ValueError("qweight, scales and out must be contiguous")
    num_experts, size_n, size_k = out.shape
    if min(num_experts, size_n, size_k) <= 0:
        raise ValueError("dequant dimensions E, N and K must be positive")
    if tuple(qweight.shape) != (num_experts, size_k // _TILE_K, size_n * 4):
        raise ValueError(
            f"Marlin weights {tuple(qweight.shape)} do not match out {tuple(out.shape)}"
        )
    if tuple(scales.shape) != (num_experts, size_k // MXFP8_BLOCK, size_n):
        raise ValueError(
            f"Marlin scales {tuple(scales.shape)} do not match out {tuple(out.shape)}"
        )
    k_width = _DEQUANT_K_TILES * _TILE_K
    if size_n % _TILE_N != 0 or size_k % k_width != 0:
        raise ValueError(
            f"dequant needs N%{_TILE_N}==0 and K%{k_width}==0, got N={size_n}, K={size_k}"
        )
    if qweight.device != out.device or scales.device != out.device:
        raise ValueError("qweight, scales and out must share a device")
    if out.device.type != "cuda":
        raise ValueError("Marlin MXFP8 dequantization requires CUDA tensors")
    _marlin_mxfp8_to_bf16_kernel[(size_n // _TILE_N, size_k // k_width, num_experts)](
        qweight.view(torch.uint8),
        scales.view(torch.uint8),
        out,
        N=size_n,
        K=size_k,
        ROW_SHIFT=size_n // 2 if swap_halves else 0,
        K_TILES=_DEQUANT_K_TILES,
        num_warps=8,
    )
    return out


@triton.jit
def _local_silu_and_mul_kernel(
    x_ptr,
    out_ptr,
    ids_ptr,
    expert_start,
    num_local_experts,
    num_rows,
    ISPP: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    limit,
    HAS_LIMIT: tl.constexpr,
):
    cols = tl.arange(0, BLOCK)
    cmask = cols < ISPP
    for i in range(ROWS):
        row = tl.program_id(0) * ROWS + i
        if row < num_rows:
            local = tl.load(ids_ptr + row) - expert_start
            if (local >= 0) & (local < num_local_experts):
                base = x_ptr + row.to(tl.int64) * (2 * ISPP)
                gate = tl.load(base + cols, mask=cmask).to(tl.float32)
                up = tl.load(base + ISPP + cols, mask=cmask).to(tl.float32)
                if HAS_LIMIT:
                    gate = tl.minimum(gate, limit)
                    up = tl.clamp(up, -limit, limit)
                out = gate * tl.sigmoid(gate) * up
                tl.store(out_ptr + row.to(tl.int64) * ISPP + cols, out, mask=cmask)


@triton.jit
def _local_topk_sum_kernel(
    y_ptr,
    ids_ptr,
    expert_start,
    num_local_experts,
    out_ptr,
    HIDDEN: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    cmask = cols < HIDDEN
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k in tl.static_range(TOP_K):
        local = tl.load(ids_ptr + token * TOP_K + k) - expert_start
        if (local >= 0) & (local < num_local_experts):
            acc += tl.load(y_ptr + (token * TOP_K + k) * HIDDEN + cols, mask=cmask).to(
                tl.float32
            )
    tl.store(out_ptr + token * HIDDEN + cols, acc, mask=cmask)


@triton.jit
def _e4m3x4_to_f32(word):
    """The four E4M3 bytes of every int32 in ``word`` as FP32, low byte first.

    Two byte permutes pair the bytes for the hardware's two-wide E4M3 to FP16
    conversion; FP16 holds every E4M3 value exactly.
    """
    return tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b32 lo, hi, f0, f1;
        .reg .b16 l16, h16, pad, a0, a1, a2, a3;
        prmt.b32 lo, $4, $4, 0x4420;
        prmt.b32 hi, $4, $4, 0x4431;
        mov.b32 {l16, pad}, lo;
        mov.b32 {h16, pad}, hi;
        cvt.rn.f16x2.e4m3x2 f0, l16;
        cvt.rn.f16x2.e4m3x2 f1, h16;
        mov.b32 {a0, a2}, f0;
        mov.b32 {a1, a3}, f1;
        cvt.f32.f16 $0, a0;
        cvt.f32.f16 $1, a1;
        cvt.f32.f16 $2, a2;
        cvt.f32.f16 $3, a3;
        }
        """,
        constraints="=r,=r,=r,=r,r",
        args=[word],
        dtype=(tl.float32, tl.float32, tl.float32, tl.float32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _e4m3_tile_dot(q_ptrs, x_ptrs):
    """Each word's four weights times their activations, summed in FP32.

    Byte b of a Marlin word holds k = (b & 1) * 8 + t * 2 + (b >> 1) of its
    16-row k-tile; ``x_ptrs`` points at each word's k = 2t.
    """
    w0, w1, w2, w3 = _e4m3x4_to_f32(tl.load(q_ptrs))
    x0 = tl.load(x_ptrs).to(tl.float32)
    x1 = tl.load(x_ptrs + 8).to(tl.float32)
    x2 = tl.load(x_ptrs + 1).to(tl.float32)
    x3 = tl.load(x_ptrs + 9).to(tl.float32)
    return w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3


@triton.jit
def _marlin_mxfp8_gemv_kernel(
    x_ptr,
    x_stride,
    q_ptr,
    s_ptr,
    ids_ptr,
    partial_ptr,
    expert_start,
    num_local_experts,
    num_routes,
    N: tl.constexpr,
    K: tl.constexpr,
    ROUTES_PER_ROW: tl.constexpr,
    SPLIT_K_TILES: tl.constexpr,
    K_TILES: tl.constexpr,
):
    n_tile = tl.program_id(0)
    split = tl.program_id(1)
    # Word (c * 4 + t) * 8 + v of a Marlin tile holds in byte b the weight of
    # n = (v >> 1) * 16 + (v & 1) * 8 + c and k = (b & 1) * 8 + t * 2 + (b >> 1)
    # (see _marlin_mxfp8_to_bf16_kernel). A word's bytes share n, so tiles stay
    # (c, t, v). One E8M0 scale covers two k-tiles: each scale block's tensors
    # are small enough for Triton to load them in the weights' own layout
    # instead of moving the products through shared memory.
    c = tl.arange(0, 8)[:, None, None]
    t = tl.arange(0, 4)[None, :, None]
    v = tl.arange(0, 8)[None, None, :]
    words = (c * 4 + t) * 8 + v
    # The E8M0 byte of column n within its 64-wide scale block.
    s_in_tile = c * 8 + (v >> 2 & 1) * 4 + (v & 1) * 2 + (v >> 1 & 1)
    c_out = tl.arange(0, 8)[:, None]
    v_out = tl.arange(0, 8)[None, :]
    n_in_tile = (v_out >> 1) * 16 + (v_out & 1) * 8 + c_out
    tile_words = N // 64 * 256
    for route in range(num_routes):
        expert = tl.load(ids_ptr + route) - expert_start
        if (expert >= 0) & (expert < num_local_experts):
            q_base = q_ptr + expert.to(tl.int64) * (N * K // 4) + n_tile * 256
            s_base = s_ptr + expert.to(tl.int64) * (K // 32 * N) + n_tile * 64
            x_base = x_ptr + (route // ROUTES_PER_ROW).to(tl.int64) * x_stride
            acc = tl.zeros([8, 4, 8], dtype=tl.float32)
            for step in range(SPLIT_K_TILES // K_TILES):
                for i in tl.static_range(K_TILES // 2):
                    kb = (split * SPLIT_K_TILES + step * K_TILES) // 2 + i
                    q_ptrs = q_base + kb * 2 * tile_words + words
                    x_ptrs = x_base + kb * 32 + t * 2
                    pair = _e4m3_tile_dot(q_ptrs, x_ptrs) + _e4m3_tile_dot(
                        q_ptrs + tile_words, x_ptrs + 16
                    )
                    s = tl.load(s_base + kb * N + s_in_tile)
                    acc += pair * (s.to(tl.int32) << 23).to(tl.float32, bitcast=True)
            tl.store(
                partial_ptr
                + (split * num_routes + route).to(tl.int64) * N
                + n_tile * 64
                + n_in_tile,
                tl.sum(acc, axis=1),
            )


@triton.jit
def _gemv_silu_and_mul_kernel(
    partial_ptr,
    out_ptr,
    ids_ptr,
    expert_start,
    num_local_experts,
    num_routes,
    ISPP: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
    limit,
    HAS_LIMIT: tl.constexpr,
):
    route = tl.program_id(0)
    expert = tl.load(ids_ptr + route) - expert_start
    if (expert >= 0) & (expert < num_local_experts):
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        cmask = cols < ISPP
        gate = tl.zeros([BLOCK], dtype=tl.float32)
        up = tl.zeros([BLOCK], dtype=tl.float32)
        for split in tl.static_range(SPLITS):
            row = partial_ptr + (split * num_routes + route).to(tl.int64) * (2 * ISPP)
            gate += tl.load(row + cols, mask=cmask, other=0.0)
            up += tl.load(row + ISPP + cols, mask=cmask, other=0.0)
        # Marlin's GEMM1 writes bf16.
        gate = gate.to(tl.bfloat16).to(tl.float32)
        up = up.to(tl.bfloat16).to(tl.float32)
        if HAS_LIMIT:
            gate = tl.minimum(gate, limit)
            up = tl.clamp(up, -limit, limit)
        out = gate * tl.sigmoid(gate) * up
        tl.store(out_ptr + route.to(tl.int64) * ISPP + cols, out, mask=cmask)


@triton.jit
def _gemv_topk_sum_kernel(
    partial_ptr,
    weights_ptr,
    ids_ptr,
    expert_start,
    num_local_experts,
    out_ptr,
    num_routes,
    HIDDEN: tl.constexpr,
    TOP_K: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    cmask = cols < HIDDEN
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k in tl.static_range(TOP_K):
        route = token * TOP_K + k
        expert = tl.load(ids_ptr + route) - expert_start
        if (expert >= 0) & (expert < num_local_experts):
            y = tl.zeros([BLOCK], dtype=tl.float32)
            for split in tl.static_range(SPLITS):
                y += tl.load(
                    partial_ptr
                    + (split * num_routes + route).to(tl.int64) * HIDDEN
                    + cols,
                    mask=cmask,
                    other=0.0,
                )
            # Marlin's GEMM2 writes each weighted route in bf16.
            weighted = y * tl.load(weights_ptr + route)
            acc += weighted.to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + token.to(tl.int64) * HIDDEN + cols, acc, mask=cmask)


def _gemv_splits(size_n: int, size_k: int) -> int:
    """Pick how many K ranges a GEMV over ``size_n`` outputs splits into.

    Doubles the split while every range still holds whole steps and the grid
    stays within ``_GEMV_PROGRAMS_PER_SM`` programs per SM.
    """
    tiles = size_n // _TILE_N
    steps = size_k // (_TILE_K * _GEMV_K_TILES)
    budget = current_platform().sm_count * _GEMV_PROGRAMS_PER_SM
    splits = 1
    while steps % (splits * 2) == 0 and tiles * splits * 2 <= budget:
        splits *= 2
    return splits


def _marlin_mxfp8_gemv(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    expert_start: int,
    size_n: int,
    size_k: int,
    routes_per_row: int,
) -> torch.Tensor:
    """Multiply each route's activation row with its Marlin-layout expert.

    Args:
        x: bf16 activations ``[rows, size_k]``; route ``r`` reads row
            ``r // routes_per_row``.
        qweight: Marlin-repacked int32 experts ``[E, size_k/16, 4*size_n]``.
        scales: Their permuted float8_e8m0fnu scales ``[E, size_k/32, size_n]``.
        topk_ids: int32 global expert ids, one per route.
        expert_start: Global id of ``qweight``'s first expert; routes outside
            its experts are skipped.
        size_n/size_k: GEMV shape.
        routes_per_row: Routes sharing one row of ``x``.

    Returns:
        FP32 partial sums ``[splits, routes, size_n]`` over ``splits`` K
        ranges; rows of skipped routes are left unwritten.
    """
    num_routes = topk_ids.numel()
    splits = _gemv_splits(size_n, size_k)
    partial = torch.empty(
        (splits, num_routes, size_n), dtype=torch.float32, device=x.device
    )
    _marlin_mxfp8_gemv_kernel[(size_n // _TILE_N, splits)](
        x,
        x.stride(0),
        qweight,
        scales.view(torch.uint8),
        topk_ids,
        partial,
        expert_start,
        int(qweight.shape[0]),
        num_routes,
        N=size_n,
        K=size_k,
        ROUTES_PER_ROW=routes_per_row,
        SPLIT_K_TILES=size_k // _TILE_K // splits,
        K_TILES=_GEMV_K_TILES,
        num_warps=_GEMV_NUM_WARPS,
    )
    return partial


def _gemv_is_faster(num_routes: int, ep_size: int) -> bool:
    """True when GEMV should run. Host-only: this apply is captured.

    Without EP every route is local and GEMV loses at HY V4 top-8. With EP,
    T·k ≤ 16 is the H20 window. Counting local routes would need ``.item()``
    and break decode CUDA graphs; more than two local routes at EP32 is rare
    and only slower, never wrong.
    """
    return num_routes <= _GEMV_MAX_ROUTES and ep_size > 1


def _gemv_apply(
    x: torch.Tensor,
    w: torch.nn.Module,
    state: _MarlinMxfp8State,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    num_tokens, top_k = topk_ids.shape
    num_routes = num_tokens * top_k
    num_local_experts = int(w.w13_weight.shape[0])
    ids = topk_ids.view(-1)
    gate_up = _marlin_mxfp8_gemv(
        x,
        w.w13_weight,
        w.w13_weight_scale_inv,
        ids,
        expert_start=state.expert_start,
        size_n=2 * state.ispp,
        size_k=state.hidden_size,
        routes_per_row=top_k,
    )
    activated = torch.empty(num_routes, state.ispp, dtype=x.dtype, device=x.device)
    limit = _swiglu_limit(w)
    block = 256
    _gemv_silu_and_mul_kernel[(num_routes, triton.cdiv(state.ispp, block))](
        gate_up,
        activated,
        ids,
        state.expert_start,
        num_local_experts,
        num_routes,
        ISPP=state.ispp,
        SPLITS=gate_up.shape[0],
        BLOCK=block,
        limit=0.0 if limit is None else limit,
        HAS_LIMIT=limit is not None,
        num_warps=4,
    )
    down = _marlin_mxfp8_gemv(
        activated,
        w.w2_weight,
        w.w2_weight_scale_inv,
        ids,
        expert_start=state.expert_start,
        size_n=state.hidden_size,
        size_k=state.ispp,
        routes_per_row=1,
    )
    out = torch.empty(num_tokens, state.hidden_size, dtype=x.dtype, device=x.device)
    block = 1024
    _gemv_topk_sum_kernel[(num_tokens, triton.cdiv(state.hidden_size, block))](
        down,
        topk_weights.reshape(-1),
        ids,
        state.expert_start,
        num_local_experts,
        out,
        num_routes,
        HIDDEN=state.hidden_size,
        TOP_K=top_k,
        SPLITS=down.shape[0],
        BLOCK=block,
        num_warps=4,
    )
    return out


def _block_size_m(routes_per_expert: float) -> int:
    """Pick the token block granularity Marlin schedules over.

    Grow the block until each expert holds under ~0.9 blocks of routed tokens,
    keeping decode batches on tight blocks and prefill batches on wide ones.
    """
    block = 8
    for candidate in (8, 16, 32, 48, 64):
        block = candidate
        if routes_per_expert / candidate < 0.9:
            break
    return block


def _validate_mxfp8_scales(name: str, scale: torch.Tensor) -> None:
    """Reject encodings not faithfully supported by every execution branch.

    E8M0 0x00 is 2**-127, not zero. Marlin's BF16 scale decoder flushes it;
    0xFF denotes NaN. Check once while loading, before repacking any parameter,
    rather than silently changing the scale or synchronizing captured applies.
    """
    if bool(((scale == 0) | (scale == 255)).any()):
        raise ValueError(
            f"Marlin MXFP8 {name} contains unsupported E8M0 scale encoding "
            "0x00 (2**-127) or 0xFF (NaN); scales must be in [1, 254]."
        )


def marlin_mxfp8_moe_weights(plan: dict, w: torch.nn.Module) -> None:
    """Repack loader-format MXFP8 experts into the Marlin layout, once.

    Input (from the block-FP8 weight loader with ``(1, 32)`` blocks):
    ``w13_weight`` ``[E, 2*ispp, hidden]`` / ``w2_weight`` ``[E, hidden, ispp]``
    float8_e4m3fn and ``w13_weight_scale_inv`` / ``w2_weight_scale_inv`` raw
    E8M0 bytes as uint8 ``[E, N, K//32]``. Output: Marlin-repacked int32
    weights and permuted float8_e8m0fnu scales written back under the same
    names, and the expert geometry in ``w._moe_backend_state``, which must be
    None until then.
    """
    if isinstance(w._moe_backend_state, _MarlinMxfp8State):
        return

    activation = plan["activation"] or w.activation
    if activation not in {"silu", "swiglu"}:
        raise ValueError(f"Marlin MXFP8 MoE does not support activation {activation!r}")
    _swiglu_limit(w)

    w13 = w.w13_weight.data
    w2 = w.w2_weight.data
    w13_scale = w.w13_weight_scale_inv.data
    w2_scale = w.w2_weight_scale_inv.data
    if w13.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        raise TypeError(
            f"Marlin MXFP8 expects float8_e4m3fn experts, got {w13.dtype}/{w2.dtype}"
        )
    if w13_scale.dtype != torch.uint8 or w2_scale.dtype != torch.uint8:
        raise TypeError(
            "Marlin MXFP8 expects raw E8M0 scales as uint8, got "
            f"{w13_scale.dtype}/{w2_scale.dtype}"
        )
    num_experts, two_ispp, hidden = w13.shape
    ispp = w2.shape[2]
    if two_ispp != 2 * ispp or w2.shape[1] != hidden:
        raise ValueError(
            f"w13 {tuple(w13.shape)} and w2 {tuple(w2.shape)} disagree on the "
            "expert geometry"
        )
    if hidden % _HIDDEN_ALIGNMENT != 0 or ispp % _ISPP_ALIGNMENT != 0:
        raise ValueError(
            f"Marlin MXFP8 needs hidden%{_HIDDEN_ALIGNMENT}==0 and "
            f"ispp%{_ISPP_ALIGNMENT}==0, got hidden={hidden}, ispp={ispp}"
        )
    expected_scales = (
        (num_experts, two_ispp, hidden // MXFP8_BLOCK),
        (num_experts, hidden, ispp // MXFP8_BLOCK),
    )
    if (tuple(w13_scale.shape), tuple(w2_scale.shape)) != expected_scales:
        raise ValueError(
            f"MXFP8 scales {tuple(w13_scale.shape)}/{tuple(w2_scale.shape)} do "
            f"not match the 1x{MXFP8_BLOCK} blocks of the weights {expected_scales}"
        )

    _validate_mxfp8_scales("w13_weight_scale_inv", w13_scale)
    _validate_mxfp8_scales("w2_weight_scale_inv", w2_scale)

    device = w13.device
    perm = torch.empty(0, dtype=torch.int, device=device)

    def _repack(weight: torch.Tensor, size_n: int, size_k: int) -> torch.Tensor:
        # gptq_marlin_repack wants int32 [size_k/4, size_n]: four E4M3 bytes
        # per int32 along K, i.e. the transposed int32 view of [size_n, size_k].
        out = torch.empty(
            (num_experts, size_k // 16, size_n * 4), dtype=torch.int32, device=device
        )
        for e in range(num_experts):
            qw = weight[e].view(torch.int32).T.contiguous()
            out[e] = gptq_marlin_repack(
                qw, perm, size_k=size_k, size_n=size_n, num_bits=8
            )
        return out

    def _permute(scale: torch.Tensor, size_n: int, size_k: int) -> torch.Tensor:
        out = torch.empty(
            (num_experts, size_k // MXFP8_BLOCK, size_n),
            dtype=torch.uint8,
            device=device,
        )
        for e in range(num_experts):
            permuted = marlin_permute_scales(
                scale[e].T.contiguous(),
                size_k=size_k,
                size_n=size_n,
                group_size=MXFP8_BLOCK,
            )
            out[e] = mxfp4_marlin_process_scales(permuted).view(torch.uint8)
        return out.view(torch.float8_e8m0fnu)

    w13_marlin = _repack(w13, size_n=two_ispp, size_k=hidden)
    w13_scale_marlin = _permute(w13_scale, size_n=two_ispp, size_k=hidden)
    w.w13_weight = torch.nn.Parameter(w13_marlin, requires_grad=False)
    w.w13_weight_scale_inv = torch.nn.Parameter(w13_scale_marlin, requires_grad=False)
    del w13, w13_scale
    w2_marlin = _repack(w2, size_n=hidden, size_k=ispp)
    w2_scale_marlin = _permute(w2_scale, size_n=hidden, size_k=ispp)
    w.w2_weight = torch.nn.Parameter(w2_marlin, requires_grad=False)
    w.w2_weight_scale_inv = torch.nn.Parameter(w2_scale_marlin, requires_grad=False)

    limit = _swiglu_limit(w)
    # hidden is already a multiple of the dequant K width (_HIDDEN_ALIGNMENT).
    dense_prefill = ispp % (_DEQUANT_K_TILES * _TILE_K) == 0
    if dense_prefill:
        # Allocated with the weights so KV sizing sees it and no graph pool owns it.
        _dense_scratch(device, num_experts, hidden, ispp)
    w._moe_backend_state = _MarlinMxfp8State(
        # EP ranks own contiguous expert ranges. Routes to any other expert
        # land in the align kernel's invalid lane, which the GEMMs skip.
        expert_start=w.ep_rank * num_experts,
        hidden_size=hidden,
        ispp=ispp,
        cutlass_swiglu_limit=(
            None
            if limit is None
            else torch.full((num_experts,), limit, dtype=torch.float32, device=device)
        ),
        dense_prefill=dense_prefill,
    )


@register_kernel(
    "moe",
    "apply",
    name="marlin_mxfp8_precomputed_moe_apply",
    solution="marlin",
    weight_preprocessor=marlin_mxfp8_moe_weights,
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(9, 0),
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    traits={
        "weight_dtype": frozenset({"fp8"}),
        "fp8_scale_block_shape": frozenset({(1, MXFP8_BLOCK)}),
        "activation": frozenset({"silu", "swiglu"}),
        # silu_and_mul has no alpha/beta (see _swiglu_limit): standard form only.
        "swiglu_form": frozenset({"standard"}),
        # Large batches run FlashInfer CUTLASS, whose permutation needs
        # distinct expert ids per token.
        "expert_id_repeats": frozenset({False}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "deterministic_finalize": frozenset({False, True}),
        "supports_ep": frozenset({True}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({_ISPP_ALIGNMENT}),
        "hidden_alignment": frozenset({_HIDDEN_ALIGNMENT}),
        "internal_activation_dtype": frozenset({"input"}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.PORTABLE,
)
def marlin_mxfp8_precomputed_moe_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    router_logits: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    num_tokens_global: int | None = None,
    max_num_tokens_per_gpu: int | None = None,
    do_finalize: bool = True,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Apply a Marlin W8A16 MXFP8 MoE with precomputed routing.

    Args:
        plan: MoE plan; ``activation`` (``silu``/``swiglu``) computes
            ``silu(gate) * up`` clamped to ``w.swiglu_arg.limit`` when the
            checkpoint sets one (``gate`` from above, ``up`` on both sides).
            ``deterministic_finalize`` selects the stable route alignment on
            the Marlin path and the unfused top-k reduction on the dense one;
            the GEMV path is reproducible either way.
        x: bf16 hidden states ``[tokens, hidden]``.
        w: MoE layer processed by :func:`marlin_mxfp8_moe_weights`: the
            Marlin-repacked ``w13_weight``/``w2_weight`` (int32), permuted
            ``w13_weight_scale_inv``/``w2_weight_scale_inv`` (float8_e8m0fnu)
            and ``_moe_backend_state``, plus the layer's ``ep_size``/``ep_rank``
            and ``tp_size``/``tp_rank``.
        router_logits: Unused; routing is precomputed.
        topk_weights: Route weights ``[tokens, top_k]``.
        topk_ids: Global expert ids ``[tokens, top_k]``; a token's valid ids
            are distinct. Ids outside this rank's expert range, ``-1``
            included, contribute zero.
        num_tokens_global: Unused; distributed EP dispatch is not owned here.
        max_num_tokens_per_gpu: Unused capacity hint.
        do_finalize: Must be true (no deferred finalize).
        enable_pdl: Unused launch hint; the dense CUTLASS path keeps PDL off.

    Returns:
        This rank's expert contributions ``[tokens, hidden]`` in the dtype of
        ``x``, summed over the top-k routes in a fixed order.
    """
    del router_logits, num_tokens_global, max_num_tokens_per_gpu, enable_pdl
    if not do_finalize:
        raise ValueError("Marlin MXFP8 MoE cannot defer finalization")
    if topk_weights is None or topk_ids is None:
        raise ValueError("Marlin MXFP8 MoE requires precomputed top-k")
    if x.dtype != torch.bfloat16:
        raise TypeError(f"Marlin MXFP8 MoE requires bf16 activations, got {x.dtype}")

    activation = plan["activation"] or w.activation
    if activation not in {"silu", "swiglu"}:
        raise ValueError(f"Marlin MXFP8 MoE does not support activation {activation!r}")
    state = w._moe_backend_state
    if not isinstance(state, _MarlinMxfp8State):
        raise TypeError("invalid Marlin MXFP8 MoE state")
    hidden = state.hidden_size
    ispp = state.ispp
    num_local_experts = int(w.w13_weight.shape[0])

    if (
        x.ndim != 2
        or x.shape[1] != hidden
        or topk_ids.ndim != 2
        or topk_ids.shape[0] != x.shape[0]
        or topk_weights.shape != topk_ids.shape
    ):
        raise ValueError("MXFP8 activations and routing shapes disagree")
    if any(t.device != x.device for t in (topk_ids, topk_weights, w.w13_weight)):
        raise ValueError("MXFP8 activations, routing and weights must share a device")
    # All three implementations read dense rows. A dtype conversion alone
    # preserves strided FP32 route-weight views (including a flattened ::2).
    x = x.contiguous()
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    num_tokens, top_k = topk_ids.shape
    # Size blocks by the routes each expert receives across all ranks: in EP
    # only 1/ep_size of the routes land on this rank's experts.
    routes_per_expert = num_tokens * top_k / (num_local_experts * w.ep_size)
    if state.dense_prefill and routes_per_expert >= _DENSE_MIN_ROUTES_PER_EXPERT:
        return _dense_bf16_apply(plan, x, w, state, topk_weights, topk_ids)

    # Ids stay global: the align kernel and the two Triton kernels below test
    # them against this rank's range instead of a separately remapped copy.
    expert_start = state.expert_start
    if _gemv_is_faster(num_tokens * top_k, w.ep_size):
        return _gemv_apply(x, w, state, topk_weights, topk_ids)

    block_m = _block_size_m(routes_per_expert)
    sorted_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids,
        block_m,
        num_local_experts,
        expert_start=expert_start,
        deterministic=plan["deterministic_finalize"],
    )
    workspace = marlin_make_workspace(x.device)

    # GEMMs only touch local routes (1/ep_size of the rows), so the outputs are
    # left uninitialized and the activation and the finalize skip the rest
    # instead of streaming all T * top_k rows. The other routes, -1 included,
    # fill blocks of expert id -1 with or without EP; only the EP
    # specialization skips them instead of reading expert -1's weights.
    num_routes = num_tokens * top_k
    gemm1_n = 2 * ispp
    intermediate1 = moe_wna16_marlin_gemm(
        x,
        torch.empty(num_routes, gemm1_n, dtype=x.dtype, device=x.device),
        w.w13_weight,
        w.w13_weight_scale_inv,
        workspace,
        sorted_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        b_q_type_id=FLOAT8_E4M3FN_ID,
        moe_block_size=block_m,
        top_k=top_k,
        mul_topk_weights=False,
        is_ep=True,
        size_m=num_tokens,
        size_n=gemm1_n,
        size_k=hidden,
    )
    limit = _swiglu_limit(w)
    intermediate2 = torch.empty(num_routes, ispp, dtype=x.dtype, device=x.device)
    rows_per_program = 4
    _local_silu_and_mul_kernel[(triton.cdiv(num_routes, rows_per_program),)](
        intermediate1,
        intermediate2,
        topk_ids,
        expert_start,
        num_local_experts,
        num_routes,
        ISPP=ispp,
        BLOCK=triton.next_power_of_2(ispp),
        ROWS=rows_per_program,
        limit=0.0 if limit is None else limit,
        HAS_LIMIT=limit is not None,
        num_warps=8,
    )

    # GEMM2 folds the route weights in, so finalize is a plain sum over top_k.
    intermediate3 = moe_wna16_marlin_gemm(
        intermediate2,
        torch.empty(num_routes, hidden, dtype=x.dtype, device=x.device),
        w.w2_weight,
        w.w2_weight_scale_inv,
        workspace,
        sorted_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        b_q_type_id=FLOAT8_E4M3FN_ID,
        moe_block_size=block_m,
        top_k=1,
        mul_topk_weights=True,
        is_ep=True,
        size_m=num_routes,
        size_n=hidden,
        size_k=ispp,
    )
    out = torch.empty(num_tokens, hidden, dtype=x.dtype, device=x.device)
    block = 2048
    _local_topk_sum_kernel[(num_tokens, triton.cdiv(hidden, block))](
        intermediate3,
        topk_ids,
        expert_start,
        num_local_experts,
        out,
        HIDDEN=hidden,
        TOP_K=top_k,
        BLOCK=block,
        num_warps=8,
    )
    return out


def _dense_bf16_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    state: _MarlinMxfp8State,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    num_local_experts = int(w.w13_weight.shape[0])
    w13, w2 = _dense_scratch(x.device, num_local_experts, state.hidden_size, state.ispp)
    dequantize_marlin_mxfp8(w.w13_weight, w.w13_weight_scale_inv, w13, swap_halves=True)
    dequantize_marlin_mxfp8(w.w2_weight, w.w2_weight_scale_inv, w2, swap_halves=False)
    return cutlass_unquant_fused_moe(
        plan,
        x,
        w13,
        w2,
        topk_weights,
        topk_ids,
        swiglu_alpha=None,
        swiglu_beta=None,
        swiglu_limit=state.cutlass_swiglu_limit,
        ep_size=w.ep_size,
        ep_rank=w.ep_rank,
        tp_size=w.tp_size,
        tp_rank=w.tp_rank,
    )
