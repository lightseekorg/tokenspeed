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
"""Skinny FP8 GEMM on gfx950: few rows against a weight whose read dominates.

``C[M, N] = (A[M, K] * a_scale[M]) @ (B[N, K] * b_scale[N]).T`` with FP8 E4M3
codes, one FP32 scale per activation row and per weight output channel, FP32
accumulation and a BF16 result. Each program owns one 64-column weight slab of
one K slice and reads it from HBM once; all M rows (one row tile) multiply
against it with the unscaled FP8 MFMA (``v_mfma_f32_32x32x64_f8f6f4``).

Each K slice stores an FP32 partial, and a second launch sums the partials in
split order, then applies both scales. The split count is fixed per weight
shape and the MFMA order depends only on K, so a row's bits do not depend on
the other rows of the launch.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import compile_ahead, gl, gluon, tl
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import _get_partial_scratch

cdna4 = gl.amd.cdna4
async_copy = gl.amd.cdna4.async_copy

_FP8 = torch.float8_e4m3fn
# Columns of one weight slab, and of one reduce program.
SKINNY_FP8_BLOCK_N = 64
# K bytes of one tile of A and B: each MFMA takes 64 K values in order, so the
# tile does not change the result. Three tiles a ring (_NUM_STAGES).
_BLOCK_K = gl.constexpr(256)
_STAGES = gl.constexpr(3)
SKINNY_FP8_BLOCK_K = _BLOCK_K.value
_NUM_STAGES = _STAGES.value
# Row tiles: a launch runs the smallest one covering its M, so M stays a
# runtime argument and the kernel compiles once per tile.
SKINNY_FP8_ROW_TILES = (64, 128)
# Rows of every split's partial, whatever the row tile.
SKINNY_FP8_PARTIAL_M = SKINNY_FP8_ROW_TILES[-1]
# Rows one reduce program sums: four waves x 64 lanes x 4 FP32 values (one
# 16-byte load) cover 1024 values, 16 rows of the 64-column reduce tile.
SKINNY_FP8_REDUCE_M = 16
# The fewest rows the kernel takes, for every configured shape; the warm-up
# compiles with it.
SKINNY_FP8_MIN_M = 17
# Alignment in bytes of every tensor's base address: the direct-to-LDS loads
# need it, and it keeps the stores and scale loads vectorized. The kernels
# state it (multiple_of) instead of specializing their pointers on it.
_POINTER_ALIGN_BYTES = gl.constexpr(16)


def _launch_metadata(grid, kernel, args):
    m, n, k = args["M"], args["N"], args["K"]
    split_k = args["SPLIT_K"]
    return {
        "name": f"{kernel.name} [M={m}, N={n}, K={k}, SPLIT_K={split_k}]",
        "flops8": 2 * m * n * k,
        # Every column tile reads its A slice; partials are FP32.
        "bytes": n // args["BLOCK_N"] * m * k + n * k + m * n * split_k * 4,
    }


def _reduce_launch_metadata(grid, kernel, args):
    m, n, split_k = args["M"], args["N"], args["SPLIT_K"]
    return {
        "name": f"{kernel.name} [M={m}, N={n}, SPLIT_K={split_k}]",
        "flops32": m * n * (split_k + 1),
        "bytes": m * n * (split_k * 4 + 2) + 4 * (m + n),
    }


@gluon.constexpr_function
def _a_load_layout(block_m: int):
    # 16 consecutive K bytes per lane over the 256-byte K tile. After the K
    # bits, the lane bases list rows in the order of the row bases of the A
    # tile's padded LDS layout (compute_efficient_padded_shared_layout below),
    # so each wave's direct-to-LDS load writes one contiguous padded run; the
    # lowering refuses other orders as uncoalesced. The 128-row tile's fifth
    # to eighth waves take row 64.
    warps = [[1, 0], [2, 0]] + ([[64, 0]] if block_m == 128 else [])
    return gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [4, 0], [8, 0]],
        lane_bases=[[0, 16], [0, 32], [0, 64], [0, 128], [16, 0], [32, 0]],
        warp_bases=warps,
        block_bases=[],
        shape=[block_m, SKINNY_FP8_BLOCK_K],
    )


@gluon.constexpr_function
def _w_load_layout(num_warps: int):
    # Each lane loads 16 consecutive K bytes of one weight row.
    lanes_k = SKINNY_FP8_BLOCK_K // 16
    return gl.BlockedLayout([1, 16], [64 // lanes_k, lanes_k], [num_warps, 1], [1, 0])


@gluon.constexpr_function
def _w_shared_layout():
    # B tiles sit in LDS as [N, K], the weight's own order, and are read
    # transposed. The swizzle XORs each row's sixteen 16-byte vectors by a
    # phase that changes every row, so the transposed reads spread over the
    # banks while each row's 16-byte runs stay whole for the direct-to-LDS
    # loads. A padded [K, N] tile with padding after each wave's 1024-byte
    # load span (as for A) also lowers and gives the same bits (ops/README.md
    # compares the two); a per-row padding interval does not lower.
    runs = SKINNY_FP8_BLOCK_K // 16
    return gl.SwizzledSharedLayout(vec=16, per_phase=1, max_phase=runs, order=[1, 0])


# M follows the batch. The pointers are not specialized either: Triton keys a
# binary on each tensor's 16-byte alignment and, on AMD, on whether its storage
# fits in 2 GB, and the codes, the weight copy and the output may be views into
# any storage. The kernels advance every base in int64 and state the alignment.
@gluon.jit(
    launch_metadata=_launch_metadata,
    do_not_specialize=("M", "a_ptr", "b_ptr", "partial_ptr"),
)
def gluon_mm_fp8_skinny_gfx950(
    a_ptr,
    b_ptr,
    partial_ptr,
    M,
    N: gl.constexpr,
    K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    PARTIAL_M: gl.constexpr,
    SPLIT_K: gl.constexpr,
):
    """Stream B once per program; see the module docstring.

    Program ``pid``, in launch order, computes column tile
    ``pid // SPLIT_K`` over K slice ``pid % SPLIT_K`` and stores its unscaled
    FP32 partial at ``partial_ptr[split, row, column]``, a
    ``[SPLIT_K, PARTIAL_M, N]`` array, for
    ``gluon_mm_fp8_skinny_reduce_gfx950``. A and B tiles go straight to LDS
    two K tiles ahead of the MFMA that reads them, through a real K loop.
    """
    BLOCK_K: gl.constexpr = _BLOCK_K
    gl.static_assert(BLOCK_M == 64 or BLOCK_M == 128, "row tiles are 64 or 128")
    gl.static_assert(BLOCK_M <= PARTIAL_M, "partials hold a whole row tile")
    gl.static_assert(BLOCK_N == 64, "column tiles are 64 wide")
    gl.static_assert(N % BLOCK_N == 0, "N must be divisible by BLOCK_N")
    NUM_STAGES: gl.constexpr = _STAGES
    gl.static_assert(
        K % (SPLIT_K * BLOCK_K) == 0 and K // (SPLIT_K * BLOCK_K) >= NUM_STAGES,
        "each split needs at least three whole K tiles",
    )
    NUM_TILES: gl.constexpr = N // BLOCK_N
    K_SLICE: gl.constexpr = K // SPLIT_K
    K_TILES: gl.constexpr = K_SLICE // BLOCK_K

    pid = gl.program_id(axis=0)
    tile = pid // SPLIT_K
    split = pid % SPLIT_K

    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, 64],
        transposed=True,
        warps_per_cta=[gl.num_warps() // 2, 2],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=16)
    dot_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=16)
    load_a: gl.constexpr = _a_load_layout(BLOCK_M)
    # Zero scale operands: the compiler then emits the unscaled FP8 MFMA
    # (v_mfma_f32_32x32x64_f8f6f4) rather than the scaled one with unit
    # scales; the ISA test checks it. Gluon's plain mfma has no 32x32x64 FP8
    # shape.
    no_scale_a = gl.full(
        [BLOCK_M, BLOCK_K // 32],
        0,
        gl.uint8,
        cdna4.get_mfma_scale_layout(dot_a, [BLOCK_M, BLOCK_K // 32]),
    )
    no_scale_b = gl.full(
        [BLOCK_N, BLOCK_K // 32],
        0,
        gl.uint8,
        cdna4.get_mfma_scale_layout(dot_b, [BLOCK_N, BLOCK_K // 32]),
    )
    load_b: gl.constexpr = _w_load_layout(gl.num_warps())
    smem_a = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty,
        [NUM_STAGES, BLOCK_M, BLOCK_K],
        cdna4.compute_efficient_padded_shared_layout(
            dot_a, [BLOCK_M, BLOCK_K], a_ptr.dtype.element_ty, is_k_contig=True
        ),
    )
    smem_b = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty,
        [NUM_STAGES, BLOCK_N, BLOCK_K],
        _w_shared_layout(),
    )

    # Advance the bases once per program in int64; the tile offsets below are
    # int32, bounded by PARTIAL_M * K and BLOCK_N * K (asserted at import).
    a_base = gl.multiple_of(a_ptr + split * K_SLICE, _POINTER_ALIGN_BYTES)
    b_base = gl.multiple_of(
        b_ptr + (tile * BLOCK_N).to(gl.int64) * K + split * K_SLICE,
        _POINTER_ALIGN_BYTES,
    )
    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, load_a))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, load_a))
    offs_bn = gl.arange(0, BLOCK_N, gl.SliceLayout(1, load_b))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(0, load_b))
    a_offs = offs_am[:, None] * K + offs_ak[None, :]
    # Rows past M read as zero and are not stored.
    a_mask = offs_am[:, None] < M
    b_offs = offs_bn[:, None] * K + offs_bk[None, :]

    if BLOCK_M == 128:
        # 128 rows: issue tile t + 2 before reading tile t from LDS, and wait
        # for tile t + 1 at the end of the iteration, so two tiles stay in
        # flight through the MFMAs. 64 rows take the wait-first order below;
        # ops/README.md says why each tile takes its order.
        # Tiles 0 and 1 go out first, A then B (two commit groups a tile); each
        # iteration issues tile t + 2 into the buffer tile t - 1 left, multiplies
        # tile t, then waits for tile t + 1.
        for t in gl.static_range(NUM_STAGES - 1):
            async_copy.buffer_load_to_shared(
                smem_a.index(t), a_base, a_offs + t * BLOCK_K, mask=a_mask
            )
            async_copy.commit_group()
            async_copy.buffer_load_to_shared(
                smem_b.index(t), b_base, b_offs + t * BLOCK_K
            )
            async_copy.commit_group()
        # Tile 0 is complete once only tile 1's two groups are pending.
        async_copy.wait_group(2)
        acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma)
        k = (NUM_STAGES - 1) * BLOCK_K
        issue = NUM_STAGES - 1
        use = 0
        for _ in tl.range(0, K_TILES - (NUM_STAGES - 1)):
            async_copy.buffer_load_to_shared(
                smem_a.index(issue), a_base, a_offs + k, mask=a_mask
            )
            async_copy.commit_group()
            a = smem_a.index(use).load(dot_a)
            async_copy.buffer_load_to_shared(smem_b.index(issue), b_base, b_offs + k)
            async_copy.commit_group()
            b = smem_b.index(use).permute((1, 0)).load(dot_b)
            # No MX scales: the per-row and per-channel scales apply in the reduce.
            acc = cdna4.mfma_scaled(a, no_scale_a, "e4m3", b, no_scale_b, "e4m3", acc)
            async_copy.wait_group(2)
            k += BLOCK_K
            issue = (issue + 1) % NUM_STAGES
            use = (use + 1) % NUM_STAGES
        # The last two tiles: the first is complete, the second after a full wait.
        for t in gl.static_range(NUM_STAGES - 1):
            if t > 0:
                async_copy.wait_group(0)
            a = smem_a.index(use).load(dot_a)
            b = smem_b.index(use).permute((1, 0)).load(dot_b)
            acc = cdna4.mfma_scaled(a, no_scale_a, "e4m3", b, no_scale_b, "e4m3", acc)
            use = (use + 1) % NUM_STAGES
    else:
        # Tiles 0 and 1 go out first; each iteration issues tile t + 2 into the
        # buffer tile t - 1 left, then multiplies tile t.
        for t in gl.static_range(NUM_STAGES - 1):
            async_copy.buffer_load_to_shared(
                smem_b.index(t), b_base, b_offs + t * BLOCK_K
            )
            async_copy.commit_group()
            async_copy.buffer_load_to_shared(
                smem_a.index(t), a_base, a_offs + t * BLOCK_K, mask=a_mask
            )
            async_copy.commit_group()
        acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma)
        k = (NUM_STAGES - 1) * BLOCK_K
        issue = NUM_STAGES - 1
        use = 0
        for _ in tl.range(0, K_TILES - (NUM_STAGES - 1)):
            # Tile t is complete once only the next tile's two groups are pending.
            async_copy.wait_group(2)
            async_copy.buffer_load_to_shared(smem_b.index(issue), b_base, b_offs + k)
            async_copy.commit_group()
            b = smem_b.index(use).permute((1, 0)).load(dot_b)
            async_copy.buffer_load_to_shared(
                smem_a.index(issue), a_base, a_offs + k, mask=a_mask
            )
            async_copy.commit_group()
            a = smem_a.index(use).load(dot_a)
            # No MX scales: the per-row and per-channel scales apply in the reduce.
            acc = cdna4.mfma_scaled(a, no_scale_a, "e4m3", b, no_scale_b, "e4m3", acc)
            k += BLOCK_K
            issue = (issue + 1) % NUM_STAGES
            use = (use + 1) % NUM_STAGES
        for t in gl.static_range(NUM_STAGES - 1):
            async_copy.wait_group(2 * (NUM_STAGES - 2 - t))
            b = smem_b.index(use).permute((1, 0)).load(dot_b)
            a = smem_a.index(use).load(dot_a)
            acc = cdna4.mfma_scaled(a, no_scale_a, "e4m3", b, no_scale_b, "e4m3", acc)
            use = (use + 1) % NUM_STAGES

    offs_cm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, mfma))
    offs_cn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, mfma))
    # [SPLIT_K, PARTIAL_M, N] in int64 once; the tile offsets stay int32.
    partial_base = gl.multiple_of(
        partial_ptr + split.to(gl.int64) * (PARTIAL_M * N) + tile * BLOCK_N,
        _POINTER_ALIGN_BYTES,
    )
    cdna4.buffer_store(
        ptr=partial_base,
        offsets=offs_cm[:, None] * N + offs_cn[None, :],
        stored_value=acc,
        mask=offs_cm[:, None] < M,
    )


@gluon.jit(
    launch_metadata=_reduce_launch_metadata,
    do_not_specialize=("M", "partial_ptr", "a_scale_ptr", "b_scale_ptr", "c_ptr"),
)
def gluon_mm_fp8_skinny_reduce_gfx950(
    partial_ptr,
    a_scale_ptr,
    b_scale_ptr,
    c_ptr,
    M,
    N: gl.constexpr,
    PARTIAL_M: gl.constexpr,
    SPLIT_K: gl.constexpr,
    REDUCE_M: gl.constexpr,
    REDUCE_N: gl.constexpr,
):
    """Sum the ``[SPLIT_K, PARTIAL_M, N]`` partials in split order, scale the
    sum by ``a_scale[row] * b_scale[column]`` and store C in BF16; each program
    reduces ``REDUCE_M`` rows of ``REDUCE_N`` columns."""
    gl.static_assert(REDUCE_M == 16 and REDUCE_N == 64, "the reduce layout is 16 x 64")
    gl.static_assert(N % REDUCE_N == 0, "N must be divisible by REDUCE_N")
    NUM_TILES: gl.constexpr = N // REDUCE_N
    pid = gl.program_id(axis=0)
    row = (pid // NUM_TILES) * REDUCE_M
    column = (pid % NUM_TILES) * REDUCE_N
    # 16 x 64 values over four waves, 4 consecutive columns per lane.
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    offs_m = gl.arange(0, REDUCE_M, gl.SliceLayout(1, layout))
    offs_n = gl.arange(0, REDUCE_N, gl.SliceLayout(0, layout))
    row_mask = (row + offs_m) < M
    mask = row_mask[:, None]
    offsets = offs_m[:, None] * N + offs_n[None, :]
    base = gl.multiple_of(partial_ptr + row * N + column, _POINTER_ALIGN_BYTES)
    total = cdna4.buffer_load(ptr=base, offsets=offsets, mask=mask, other=0.0)
    for s in gl.static_range(1, SPLIT_K):
        total += cdna4.buffer_load(
            ptr=gl.multiple_of(base + s * (PARTIAL_M * N), _POINTER_ALIGN_BYTES),
            offsets=offsets,
            mask=mask,
            other=0.0,
        )
    # The scales apply after the split sum: one rounding order for every M.
    a_scale = cdna4.buffer_load(
        ptr=gl.multiple_of(a_scale_ptr + row, _POINTER_ALIGN_BYTES),
        offsets=offs_m,
        mask=row_mask,
        other=0.0,
    )
    b_scale = cdna4.buffer_load(
        ptr=gl.multiple_of(b_scale_ptr + column, _POINTER_ALIGN_BYTES), offsets=offs_n
    )
    total = total * a_scale[:, None] * b_scale[None, :]
    cdna4.buffer_store(
        ptr=gl.multiple_of(c_ptr + row.to(gl.int64) * N + column, _POINTER_ALIGN_BYTES),
        offsets=offs_m[:, None] * N + offs_n[None, :],
        stored_value=total.to(c_ptr.dtype.element_ty),
        mask=mask,
    )


# (SPLIT_K, last row count) by (N, K), from SKINNY_FP8_MIN_M rows: one split
# count per weight, whatever the row tile, so a row's bits do not depend on M.
# Measured on MI355X for these two weight shapes.
_SKINNY_FP8_CONFIGS: dict[tuple[int, int], tuple[int, int]] = {
    (4608, 8192): (2, 128),
    (1024, 8192): (8, 128),
}


# The kernels keep tile offsets in int32, and buffer loads and stores take
# them as 32-bit byte offsets from a 64-bit base, which advances per program
# (and per split, so SPLIT_K does not enter): the FP8 rows of A and the
# weight slab, a split's FP32 partial, and the reduce's BF16 rows of C, in
# bytes, for every configured shape.
for _n, _k in _SKINNY_FP8_CONFIGS:
    if (
        max(
            SKINNY_FP8_PARTIAL_M * _k,
            SKINNY_FP8_BLOCK_N * _k,
            SKINNY_FP8_PARTIAL_M * _n * 4,
            SKINNY_FP8_PARTIAL_M * _n * 2,
        )
        >= 2**31
    ):
        raise RuntimeError(f"skinny FP8 GEMM [{_n}, {_k}] overflows int32 tile offsets")

# The LDS ring: three K tiles of A (padded rows) and B for each row tile, with
# A's padding bounded by its worst case, 16 bytes a row.
_GFX950_LDS_BYTES = 160 * 1024
_A_ROW_PAD_MAX = 16
for _tile in SKINNY_FP8_ROW_TILES:
    _ring = _NUM_STAGES * (
        _tile * (SKINNY_FP8_BLOCK_K + _A_ROW_PAD_MAX)
        + SKINNY_FP8_BLOCK_N * SKINNY_FP8_BLOCK_K
    )
    if _ring > _GFX950_LDS_BYTES:
        raise RuntimeError(
            f"skinny FP8 GEMM row tile {_tile} overflows LDS ({_ring} B)"
        )


def supports_gluon_mm_fp8_skinny_gfx950(M: int, N: int, K: int) -> bool:
    """Whether the skinny FP8 GEMM has a measured config for this problem."""
    config = _SKINNY_FP8_CONFIGS.get((N, K))
    return config is not None and SKINNY_FP8_MIN_M <= M <= config[1]


def launch_gluon_mm_fp8_skinny_gfx950(
    A: torch.Tensor,
    A_scales: torch.Tensor,
    B: torch.Tensor,
    B_scales: torch.Tensor,
    *,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """Compute a few rows of the scaled FP8 ``A @ B.T`` by streaming B once.

    Args:
        A: Contiguous ``[M, K]`` FP8 E4M3 activation codes.
        A_scales: FP32 per-row scales, ``M`` contiguous values.
        B: Contiguous ``[N, K]`` FP8 E4M3 weight codes on ``A``'s device;
            ``(M, N, K)`` must pass ``supports_gluon_mm_fp8_skinny_gfx950``.
        B_scales: FP32 per-output-channel scales, ``N`` contiguous values.
        out: Contiguous ``[M, N]`` BF16 destination on ``A``'s device, or
            None to allocate one.

    Returns:
        ``[M, N]`` BF16 ``(A * A_scales) @ (B * B_scales).T``.
    """
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError(
            f"skinny FP8 GEMM expects [M, K] x [N, K], got {A.shape} {B.shape}"
        )
    M, K = A.shape
    N = B.shape[0]
    if A.dtype != _FP8 or B.dtype != _FP8:
        raise TypeError(
            f"skinny FP8 GEMM expects E4M3 codes, got {A.dtype=} {B.dtype=}"
        )
    if not A.is_cuda or B.device != A.device:
        raise ValueError(
            f"skinny FP8 GEMM expects inputs on one GPU, got {A.device} {B.device}"
        )
    if (
        A_scales.dtype != torch.float32
        or B_scales.dtype != torch.float32
        or A_scales.numel() != M
        or B_scales.numel() != N
        or not A_scales.is_contiguous()
        or not B_scales.is_contiguous()
    ):
        raise ValueError("skinny FP8 GEMM expects contiguous FP32 [M] and [N] scales")
    if not supports_gluon_mm_fp8_skinny_gfx950(M, N, K):
        raise ValueError(f"skinny FP8 GEMM has no config for M={M}, N={N}, K={K}")
    if not A.is_contiguous() or not B.is_contiguous():
        raise ValueError("skinny FP8 GEMM expects contiguous activations and weight")
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=A.device)
    elif (
        out.shape != (M, N)
        or not out.is_contiguous()
        or out.dtype != torch.bfloat16
        or out.device != A.device
    ):
        raise ValueError(
            f"skinny FP8 GEMM out must be contiguous [{M}, {N}] BF16 on {A.device}"
        )
    split_k, _ = _SKINNY_FP8_CONFIGS[(N, K)]
    num_tiles = N // SKINNY_FP8_BLOCK_N
    _check_aligned(A=A, A_scales=A_scales, B=B, B_scales=B_scales, out=out)
    partials = _get_partial_scratch(
        A.device, num_tiles, split_k, SKINNY_FP8_BLOCK_N, SKINNY_FP8_PARTIAL_M
    )
    row_tile = next(tile for tile in SKINNY_FP8_ROW_TILES if M <= tile)
    gluon_mm_fp8_skinny_gfx950[(num_tiles * split_k,)](
        A,
        B,
        partials,
        M,
        *_main_constexprs(N, K, row_tile),
        num_warps=_main_warps(row_tile),
    )
    gluon_mm_fp8_skinny_reduce_gfx950[(-(-M // SKINNY_FP8_REDUCE_M) * num_tiles,)](
        partials, A_scales, B_scales, out, M, *_reduce_constexprs(N, K), num_warps=4
    )
    return out


def _check_aligned(**tensors: torch.Tensor) -> None:
    # The kernels state this alignment instead of specializing on it.
    align = _POINTER_ALIGN_BYTES.value
    for name, tensor in tensors.items():
        if tensor.data_ptr() % align:
            raise ValueError(
                f"skinny FP8 GEMM needs a {align}-byte aligned {name}; got "
                f"{tensor.data_ptr():#x}"
            )


def _main_warps(row_tile: int) -> int:
    # The 128-row tile runs 8 waves (a 4 x 2 grid of 32 x 32 MFMA tiles), so
    # each wave keeps one output tile and twice as many waves issue loads.
    return 8 if row_tile == 128 else 4


def _main_constexprs(N: int, K: int, row_tile: int) -> tuple:
    split_k, _ = _SKINNY_FP8_CONFIGS[(N, K)]
    return (N, K, row_tile, SKINNY_FP8_BLOCK_N, SKINNY_FP8_PARTIAL_M, split_k)


def _reduce_constexprs(N: int, K: int) -> tuple:
    split_k = _SKINNY_FP8_CONFIGS[(N, K)][0]
    return (N, SKINNY_FP8_PARTIAL_M, split_k, SKINNY_FP8_REDUCE_M, SKINNY_FP8_BLOCK_N)


_warmed: set[tuple[int, int, torch.device]] = set()


def warm_gluon_mm_fp8_skinny_gfx950(N: int, K: int, device: torch.device) -> None:
    """Compile both row tiles' main kernel and the reduce for ``[N, K]`` ahead
    of serving, once per shape and device, so no row count of the window
    compiles in a CUDA graph capture or while serving.

    Args:
        N: Output channels of a configured weight shape.
        K: Reduction width of that shape.
        device: The GPU the kernels run on.
    """
    if (N, K) not in _SKINNY_FP8_CONFIGS:
        raise ValueError(f"skinny FP8 GEMM has no config for [{N}, {K}]")
    if (N, K, device) in _warmed:
        return
    codes = torch.empty((SKINNY_FP8_PARTIAL_M, K), dtype=_FP8, device=device)
    weight = torch.empty((N, K), dtype=_FP8, device=device)
    split_k = _SKINNY_FP8_CONFIGS[(N, K)][0]
    partials = torch.empty(
        (split_k, SKINNY_FP8_PARTIAL_M, N), dtype=torch.float32, device=device
    )
    scales = torch.empty(
        (max(SKINNY_FP8_PARTIAL_M, N),), dtype=torch.float32, device=device
    )
    out = torch.empty((SKINNY_FP8_PARTIAL_M, N), dtype=torch.bfloat16, device=device)
    for row_tile in SKINNY_FP8_ROW_TILES:
        compile_ahead(
            gluon_mm_fp8_skinny_gfx950,
            codes,
            weight,
            partials,
            SKINNY_FP8_MIN_M,
            *_main_constexprs(N, K, row_tile),
            num_warps=_main_warps(row_tile),
        )
    compile_ahead(
        gluon_mm_fp8_skinny_reduce_gfx950,
        partials,
        scales,
        scales,
        out,
        SKINNY_FP8_MIN_M,
        *_reduce_constexprs(N, K),
        num_warps=4,
    )
    _warmed.add((N, K, device))
