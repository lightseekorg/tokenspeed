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

"""Skinny dense16 GEMM on gfx950: few rows against a weight whose read dominates.

``C[M, N] = A[M, K] @ B[N, K].T`` in BF16, all three contiguous. Each program
owns one 64-column weight slab of one K slice and reads it from HBM exactly
once; all M rows (one row tile) multiply against it, so the weight traffic
does not grow with M.

Each K slice stores an FP32 partial, and a second launch sums the partials in
split order. The K split and the order of the MFMA accumulation depend only on
K and ``SPLIT_K``, never on M or its row tile, so a row's bits do not depend
on which rows share the launch.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from tokenspeed_kernel_amd._triton import compile_ahead, gl, gluon, tl
from tokenspeed_kernel_amd.ops.gfx950._xcd import _remap_xcd
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import _get_partial_scratch

cdna4 = gl.amd.cdna4
async_copy = gl.amd.cdna4.async_copy

# Lanes of a gfx950 wave.
_WAVE_LANES = 64
# K elements a lane moves in one 16-byte BF16 load.
_LOAD_K = 8
# Alignment in bytes of every tensor's base address: the direct-to-LDS loads
# of A and B have no lowering without it, and it keeps the stores vectorized.
# The kernels state it (multiple_of) since they do not specialize pointers.
_POINTER_ALIGN_BYTES = gl.constexpr(16)
# Elements one wave writes to LDS in one direct-to-LDS load of the weight
# tile: a contiguous span, which padding cannot cut.
_W_LOAD_SPAN = _WAVE_LANES * _LOAD_K
# Elements of padding after each span: one 16-byte run, the pad
# compute_efficient_padded_shared_layout gives the A tile.
_W_SPAN_PAD = _LOAD_K
# Rows of a 16-row group one lane loads for the A tile.
_A_ROWS_PER_LANE = 4
# Columns of one weight slab, and of one reduce program.
SKINNY_BLOCK_N = 64
# Row tiles: a launch runs the smallest one covering its M, so M stays a
# runtime argument and the kernel compiles once per tile.
SKINNY_ROW_TILES = (64, 128)
# Rows of every split's partial, whatever the row tile.
SKINNY_PARTIAL_M = SKINNY_ROW_TILES[-1]
# Waves of either kernel's program: the 2x2 MFMA wave grid of the row tile,
# and the same count for the reduce.
_NUM_WARPS = 4
# FP32 partials a lane moves in one 16-byte load.
_REDUCE_LOAD_N = 4
# Rows one reduce program sums: what its waves hold when every lane takes one
# 16-byte load per split, over SKINNY_BLOCK_N columns.
SKINNY_REDUCE_M = _NUM_WARPS * _WAVE_LANES * _REDUCE_LOAD_N // SKINNY_BLOCK_N


def _skinny_launch_metadata(grid, kernel, args):
    m, n, k = args["M"], args["N"], args["K"]
    split_k = args["SPLIT_K"]
    elem = args["a_ptr"].element_size()
    return {
        "name": f"{kernel.name} [M={m}, N={n}, K={k}, SPLIT_K={split_k}]",
        "flops16": 2 * m * n * k,
        # Every column tile reads its A slice.
        "bytes": (n // args["BLOCK_N"] * m * k + n * k) * elem + m * n * split_k * 4,
    }


def _skinny_reduce_launch_metadata(grid, kernel, args):
    m, n, split_k = args["M"], args["N"], args["SPLIT_K"]
    return {
        "name": f"{kernel.name} [M={m}, N={n}, SPLIT_K={split_k}]",
        "flops32": m * n * (split_k - 1),
        "bytes": m * n * (split_k * 4 + args["c_ptr"].element_size()),
    }


@gluon.constexpr_function
def _skinny_block_k(block_m: int) -> int:
    # Three K tiles of A and B must fit LDS; the K tile does not change the
    # result, each MFMA takes 16 K values in order whatever the tile.
    return 128 if block_m == 64 else 64


@gluon.constexpr_function
def _skinny_w_load_layout(block_k: int):
    # The tile is [K, N] with K contiguous, the weight's own order. Each lane
    # loads _LOAD_K consecutive K elements (16 bytes) of one weight row; a
    # wave's lanes cover the tile's K and as many rows as that leaves.
    lanes_k = block_k // _LOAD_K
    return gl.BlockedLayout(
        [_LOAD_K, 1], [lanes_k, _WAVE_LANES // lanes_k], [1, _NUM_WARPS], [0, 1]
    )


@gluon.constexpr_function
def _skinny_w_shared_layout(block_k: int):
    # B tiles sit in LDS in the same [K, N], K-contiguous order, which is also
    # the MFMA operand's, so a wave's direct-to-LDS load writes one
    # contiguous span of _W_LOAD_SPAN elements. The padding goes between
    # spans: a shorter interval (BLOCK_K, say) falls inside a span, and the
    # compiler then has no direct-to-LDS lowering for the load.
    return gl.PaddedSharedLayout.with_identity_for(
        [[_W_LOAD_SPAN, _W_SPAN_PAD]], [block_k, SKINNY_BLOCK_N], [0, 1]
    )


@gluon.constexpr_function
def _skinny_reduce_layout(reduce_m: int, reduce_n: int):
    # Each lane holds _REDUCE_LOAD_N consecutive columns of one row (16 bytes
    # of FP32 a load); a wave's lanes cover reduce_n columns of lanes_m rows.
    lanes_n = reduce_n // _REDUCE_LOAD_N
    lanes_m = _WAVE_LANES // lanes_n
    assert (
        lanes_m * lanes_n == _WAVE_LANES and reduce_m % lanes_m == 0
    ), "the reduce tile must fill whole waves"
    return gl.BlockedLayout(
        [1, _REDUCE_LOAD_N], [lanes_m, lanes_n], [reduce_m // lanes_m, 1], [1, 0]
    )


@gluon.constexpr_function
def _powers_of_two(first: int, end: int) -> list[int]:
    """``first``, ``2 * first``, ... below ``end``."""
    powers = []
    while first < end:
        powers.append(first)
        first *= 2
    return powers


@gluon.constexpr_function
def _skinny_a_load_layout(block_m: int, block_k: int):
    # A lane loads _LOAD_K consecutive K elements (16 bytes) of
    # _A_ROWS_PER_LANE rows, _NUM_WARPS apart: the waves interleave the rows
    # between. Lanes take the rest of K, then the rest of the rows. The row
    # lanes follow the offset-basis order of the padded LDS layout the tile
    # is stored in (compute_efficient_padded_shared_layout), which a
    # direct-to-LDS load requires: ascending for the 64-row tile, descending
    # for the 128-row one.
    lane_row = _NUM_WARPS * _A_ROWS_PER_LANE
    row_lanes = [[row, 0] for row in _powers_of_two(lane_row, block_m)]
    if block_m == SKINNY_ROW_TILES[-1]:
        # The helper orders the 128-row tile's row offsets high bit first.
        row_lanes.reverse()
    return gl.DistributedLinearLayout(
        reg_bases=[[0, k] for k in _powers_of_two(1, _LOAD_K)]
        + [[row, 0] for row in _powers_of_two(_NUM_WARPS, lane_row)],
        lane_bases=[[0, k] for k in _powers_of_two(_LOAD_K, block_k)] + row_lanes,
        warp_bases=[[row, 0] for row in _powers_of_two(1, _NUM_WARPS)],
        block_bases=[],
        shape=[block_m, block_k],
    )


@gluon.constexpr_function
def _skinny_dot_layout(operand_index: int, mfma_layout):
    # The MFMA operands hold _LOAD_K consecutive K elements a lane, the width
    # of the loads.
    return gl.DotOperandLayout(
        operand_index=operand_index, parent=mfma_layout, k_width=_LOAD_K
    )


# M follows the batch. The pointers are not specialized either: Triton keys a
# binary on each tensor's 16-byte alignment and, on AMD, on whether its storage
# fits in 2 GB, and the activation and the output may be views into any
# storage. The kernels advance every base in int64 and state the alignment.
@gluon.jit(
    launch_metadata=_skinny_launch_metadata,
    do_not_specialize=("M", "a_ptr", "b_ptr", "partial_ptr"),
)
def gluon_mm_a16w16_skinny_gfx950(
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
    NUM_XCDS: gl.constexpr,
):
    """Stream B once per program; see the module docstring.

    ``a_ptr`` is a contiguous ``[M, K]`` array and ``b_ptr`` a contiguous
    ``[N, K]`` one, both 16-byte aligned like ``partial_ptr``. Program ``pid``
    (after the XCD remap) computes column tile ``pid // SPLIT_K`` over K slice
    ``pid % SPLIT_K`` and stores its FP32 partial at
    ``partial_ptr[split, row, column]``, a ``[SPLIT_K, PARTIAL_M, N]`` array,
    for ``gluon_mm_a16w16_skinny_reduce_gfx950``.

    A and B tiles go straight to LDS two K tiles ahead of the MFMA that reads
    them. The K loop is a real loop: unrolling it keeps every tile's offsets
    live and costs the wave occupancy that hides the HBM latency.
    """
    BLOCK_K: gl.constexpr = _skinny_block_k(BLOCK_M)
    gl.static_assert(
        BLOCK_M == 64 or BLOCK_M == 128, "skinny GEMM row tiles are 64 or 128"
    )
    gl.static_assert(BLOCK_M <= PARTIAL_M, "partials hold a whole row tile")
    gl.static_assert(BLOCK_N == 64, "skinny GEMM column tiles are 64 wide")
    gl.static_assert(N % BLOCK_N == 0, "skinny GEMM needs N divisible by BLOCK_N")
    gl.static_assert(SPLIT_K > 1, "skinny GEMM splits K")
    # The tile offsets below stay int32, and the buffer loads and stores
    # address in bytes.
    A_BYTES: gl.constexpr = a_ptr.dtype.element_ty.primitive_bitwidth // 8
    B_BYTES: gl.constexpr = b_ptr.dtype.element_ty.primitive_bitwidth // 8
    PARTIAL_BYTES: gl.constexpr = partial_ptr.dtype.element_ty.primitive_bitwidth // 8
    gl.static_assert(
        (BLOCK_M + 1) * K * A_BYTES < 2**31
        and (BLOCK_N + 1) * K * B_BYTES < 2**31
        and (PARTIAL_M + 1) * N * PARTIAL_BYTES < 2**31,
        "skinny GEMM tile byte offsets must fit int32",
    )
    # K tiles in LDS at once: the tile at the MFMA and two in flight behind it.
    NUM_STAGES: gl.constexpr = 3
    gl.static_assert(
        K % (SPLIT_K * BLOCK_K) == 0 and K // (SPLIT_K * BLOCK_K) >= NUM_STAGES,
        "skinny GEMM needs at least three whole K tiles per split",
    )
    NUM_TILES: gl.constexpr = N // BLOCK_N
    K_SLICE: gl.constexpr = K // SPLIT_K
    K_TILES: gl.constexpr = K_SLICE // BLOCK_K

    pid = gl.program_id(axis=0)
    if NUM_XCDS > 1:
        pid = _remap_xcd(pid, NUM_TILES * SPLIT_K, NUM_XCDS)
    tile = pid // SPLIT_K
    split = pid % SPLIT_K

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[32, 32, 16], transposed=True, warps_per_cta=[2, 2]
    )
    dotOpLayoutA: gl.constexpr = _skinny_dot_layout(0, mfmaLayout)
    dotOpLayoutB: gl.constexpr = _skinny_dot_layout(1, mfmaLayout)
    loadLayoutA: gl.constexpr = _skinny_a_load_layout(BLOCK_M, BLOCK_K)
    loadLayoutB: gl.constexpr = _skinny_w_load_layout(BLOCK_K)
    smemA = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty,
        [NUM_STAGES, BLOCK_M, BLOCK_K],
        cdna4.compute_efficient_padded_shared_layout(
            dotOpLayoutA, [BLOCK_M, BLOCK_K], a_ptr.dtype.element_ty, is_k_contig=True
        ),
    )
    smemB = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty,
        [NUM_STAGES, BLOCK_K, BLOCK_N],
        _skinny_w_shared_layout(BLOCK_K),
    )

    # Advance the bases once per program in int64; the tile offsets below are
    # int32, bounded by (BLOCK_M + 1) * K and (BLOCK_N + 1) * K elements
    # (their bytes asserted above). A K slice and a weight row are whole numbers of 16-byte runs
    # (K is a multiple of SPLIT_K * BLOCK_K), so the bases keep the pointers'
    # alignment, which the launcher checks.
    a_base = gl.multiple_of(a_ptr + split * K_SLICE, _POINTER_ALIGN_BYTES)
    b_base = gl.multiple_of(
        b_ptr + (tile * BLOCK_N).to(gl.int64) * K + split * K_SLICE,
        _POINTER_ALIGN_BYTES,
    )
    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, loadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, loadLayoutA))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, loadLayoutB))
    offs_bn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, loadLayoutB))
    a_offs = offs_am[:, None] * K + offs_ak[None, :]
    # The row tile is taller than M: rows at and past M are not loaded. What
    # the MFMA computes for them is never stored (c_mask below).
    a_mask = offs_am[:, None] < M
    b_offs = offs_bk[:, None] + offs_bn[None, :] * K

    # Tiles 0 and 1 go out first; each loop iteration issues tile t + 2 into
    # the buffer tile t - 1 left, then multiplies tile t.
    for t in gl.static_range(NUM_STAGES - 1):
        async_copy.buffer_load_to_shared(smemB.index(t), b_base, b_offs + t * BLOCK_K)
        async_copy.commit_group()
        async_copy.buffer_load_to_shared(
            smemA.index(t), a_base, a_offs + t * BLOCK_K, mask=a_mask
        )
        async_copy.commit_group()
    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfmaLayout)
    k = (NUM_STAGES - 1) * BLOCK_K
    issue = NUM_STAGES - 1
    use = 0
    for _ in tl.range(0, K_TILES - (NUM_STAGES - 1)):
        # Tile t is complete once only the next tile's two groups are pending.
        async_copy.wait_group(2)
        async_copy.buffer_load_to_shared(smemB.index(issue), b_base, b_offs + k)
        async_copy.commit_group()
        b = smemB.index(use).load(dotOpLayoutB)
        async_copy.buffer_load_to_shared(
            smemA.index(issue), a_base, a_offs + k, mask=a_mask
        )
        async_copy.commit_group()
        a = smemA.index(use).load(dotOpLayoutA)
        acc = cdna4.mfma(a, b, acc)
        k += BLOCK_K
        issue = (issue + 1) % NUM_STAGES
        use = (use + 1) % NUM_STAGES
    for t in gl.static_range(NUM_STAGES - 1):
        async_copy.wait_group(2 * (NUM_STAGES - 2 - t))
        b = smemB.index(use).load(dotOpLayoutB)
        a = smemA.index(use).load(dotOpLayoutA)
        acc = cdna4.mfma(a, b, acc)
        use = (use + 1) % NUM_STAGES

    offs_cm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, mfmaLayout))
    offs_cn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, mfmaLayout))
    # Rows at and past M have no partial: the reduce never reads them, so the
    # mask only saves their stores.
    c_mask = offs_cm[:, None] < M
    # [SPLIT_K, PARTIAL_M, N] in int64 once; the tile offsets stay int32,
    # below (PARTIAL_M + 1) * N elements (their bytes asserted above). A tile starts on a 16-byte
    # boundary of the scratch: BLOCK_N FP32 columns are 256 bytes.
    partial_base = gl.multiple_of(
        partial_ptr + split.to(gl.int64) * (PARTIAL_M * N) + tile * BLOCK_N,
        _POINTER_ALIGN_BYTES,
    )
    cdna4.buffer_store(
        ptr=partial_base,
        offsets=offs_cm[:, None] * N + offs_cn[None, :],
        stored_value=acc,
        mask=c_mask,
    )


# As for the main kernel: neither M nor the pointers key a binary.
@gluon.jit(
    launch_metadata=_skinny_reduce_launch_metadata,
    do_not_specialize=("M", "partial_ptr", "c_ptr"),
)
def gluon_mm_a16w16_skinny_reduce_gfx950(
    partial_ptr,
    c_ptr,
    M,
    N: gl.constexpr,
    PARTIAL_M: gl.constexpr,
    SPLIT_K: gl.constexpr,
    REDUCE_M: gl.constexpr,
    REDUCE_N: gl.constexpr,
):
    """Sum the skinny GEMM's split-K partials, round once and store C.

    The ``[SPLIT_K, PARTIAL_M, N]`` partials add in split order
    (``((p0 + p1) + p2) + ...`` in FP32) into the contiguous ``[M, N]`` C;
    each program reduces ``REDUCE_M`` rows of ``REDUCE_N`` columns."""
    gl.static_assert(N % REDUCE_N == 0, "skinny reduce needs N divisible by REDUCE_N")
    # Every offset below is int32 except the bases: a program's own tile
    # offsets, and a split's distance from the first split's partial. The
    # buffer loads and stores address in bytes.
    PARTIAL_BYTES: gl.constexpr = partial_ptr.dtype.element_ty.primitive_bitwidth // 8
    C_BYTES: gl.constexpr = c_ptr.dtype.element_ty.primitive_bitwidth // 8
    gl.static_assert(
        (REDUCE_M + 1) * N * PARTIAL_BYTES < 2**31
        and (REDUCE_M + 1) * N * C_BYTES < 2**31
        and SPLIT_K * PARTIAL_M * N * PARTIAL_BYTES < 2**31,
        "skinny reduce byte offsets must fit int32",
    )
    NUM_TILES: gl.constexpr = N // REDUCE_N
    pid = gl.program_id(axis=0)
    row = (pid // NUM_TILES) * REDUCE_M
    column = (pid % NUM_TILES) * REDUCE_N
    layout: gl.constexpr = _skinny_reduce_layout(REDUCE_M, REDUCE_N)
    offs_m = gl.arange(0, REDUCE_M, gl.SliceLayout(1, layout))
    offs_n = gl.arange(0, REDUCE_N, gl.SliceLayout(0, layout))
    # Rows at and past M hold no partial and have no output row.
    mask = (row + offs_m)[:, None] < M
    offsets = offs_m[:, None] * N + offs_n[None, :]
    # A reduce tile starts on a 16-byte boundary of the scratch and of C:
    # REDUCE_N columns are 256 bytes of FP32 and 128 of BF16.
    tile = row.to(gl.int64) * N + column
    base = gl.multiple_of(partial_ptr + tile, _POINTER_ALIGN_BYTES)
    total = cdna4.buffer_load(ptr=base, offsets=offsets, mask=mask, other=0.0)
    for s in gl.static_range(1, SPLIT_K):
        total += cdna4.buffer_load(
            ptr=base + s * (PARTIAL_M * N), offsets=offsets, mask=mask, other=0.0
        )
    cdna4.buffer_store(
        ptr=gl.multiple_of(c_ptr + tile, _POINTER_ALIGN_BYTES),
        offsets=offsets,
        stored_value=total.to(c_ptr.dtype.element_ty),
        mask=mask,
    )


class _SkinnyConfig(NamedTuple):
    # K slices, each summed on its own; fixes the summation order.
    split_k: int
    # XCDs the programs are dealt over.
    num_xcds: int
    # First and last row count the kernel takes.
    first_m: int
    last_m: int


# By (N, K): shapes and row counts measured faster than the other gemm.mm
# solutions, torch.mm (hipBLASLt) with tuned picks included, in HIP-graph
# replay with cold weights. split_k fixes the summation order, so an entry
# must not change between row tiles.
_SKINNY_CONFIGS: dict[tuple[int, int], _SkinnyConfig] = {
    # An 8192-wide hidden state projected to a quarter of a 4096-wide latent.
    (1024, 8192): _SkinnyConfig(split_k=16, num_xcds=8, first_m=33, last_m=128),
}


def supports_gluon_mm_a16w16_skinny_gfx950(M: int, N: int, K: int) -> bool:
    """Whether the skinny GEMM has a measured config for this problem.

    Args:
        M: Activation rows.
        N: Weight rows (output columns).
        K: Inner dimension.

    Returns:
        True when ``(N, K)`` has a config whose row range holds ``M``.
    """
    config = _SKINNY_CONFIGS.get((N, K))
    return config is not None and config.first_m <= M <= config.last_m


def supports_gluon_mm_a16w16_skinny_weight_gfx950(N: int, K: int) -> bool:
    """Whether the skinny GEMM has measured rows for an ``[N, K]`` weight.

    Args:
        N: Weight rows (output columns).
        K: Inner dimension.

    Returns:
        True when ``(N, K)`` has a measured skinny GEMM configuration.
    """
    return (N, K) in _SKINNY_CONFIGS


# The launch and the ahead-of-time compile take their options from the same
# three helpers, so they name the same binaries.
def _main_options(N: int, K: int, row_tile: int) -> dict[str, int]:
    config = _SKINNY_CONFIGS[(N, K)]
    return dict(
        BLOCK_M=row_tile,
        BLOCK_N=SKINNY_BLOCK_N,
        PARTIAL_M=SKINNY_PARTIAL_M,
        SPLIT_K=config.split_k,
        NUM_XCDS=config.num_xcds,
        num_warps=_NUM_WARPS,
    )


def _reduce_options(N: int, K: int) -> dict[str, int]:
    return dict(
        PARTIAL_M=SKINNY_PARTIAL_M,
        SPLIT_K=_SKINNY_CONFIGS[(N, K)].split_k,
        REDUCE_M=SKINNY_REDUCE_M,
        REDUCE_N=SKINNY_BLOCK_N,
        num_warps=_NUM_WARPS,
    )


def _partials(device: torch.device, N: int, K: int) -> torch.Tensor:
    return _get_partial_scratch(
        device,
        N // SKINNY_BLOCK_N,
        _SKINNY_CONFIGS[(N, K)].split_k,
        SKINNY_BLOCK_N,
        SKINNY_PARTIAL_M,
    )


def _check_aligned(**tensors: torch.Tensor) -> None:
    # The kernels state this alignment instead of specializing on it.
    align = _POINTER_ALIGN_BYTES.value
    for name, tensor in tensors.items():
        if tensor.data_ptr() % align:
            raise ValueError(
                f"skinny GEMM needs a {align}-byte aligned {name}; got "
                f"{tensor.data_ptr():#x}"
            )


def compile_gluon_mm_a16w16_skinny_gfx950(B: torch.Tensor) -> None:
    """Compile, without launching, every skinny GEMM binary for a weight.

    That is one binary per row tile, and the reduce.

    The binaries depend on the weight's shape and on nothing a launch varies:
    not the row count, and not where the activation, the weight or the output
    live (the kernels do not specialize their pointers). After this no launch
    on an ``[N, K]`` weight compiles.

    Args:
        B: Contiguous ``[N, K]`` BF16 weight on a GPU;
            ``supports_gluon_mm_a16w16_skinny_weight_gfx950(N, K)`` must hold.
    """
    if B.ndim != 2 or B.dtype != torch.bfloat16 or not B.is_cuda:
        raise ValueError(
            f"skinny GEMM expects a BF16 [N, K] weight on a GPU, got {B.dtype} "
            f"{tuple(B.shape)} on {B.device}"
        )
    N, K = B.shape
    if not supports_gluon_mm_a16w16_skinny_weight_gfx950(N, K):
        raise ValueError(f"skinny GEMM has no config for N={N}, K={K}")
    if not B.is_contiguous():
        raise ValueError("skinny GEMM expects a contiguous weight")
    rows = _SKINNY_CONFIGS[(N, K)].first_m
    A = torch.empty((rows, K), dtype=B.dtype, device=B.device)
    out = torch.empty((rows, N), dtype=B.dtype, device=B.device)
    partials = _partials(B.device, N, K)
    for row_tile in SKINNY_ROW_TILES:
        compile_ahead(
            gluon_mm_a16w16_skinny_gfx950,
            A,
            B,
            partials,
            rows,
            N,
            K,
            **_main_options(N, K, row_tile),
        )
    compile_ahead(
        gluon_mm_a16w16_skinny_reduce_gfx950,
        partials,
        out,
        rows,
        N,
        **_reduce_options(N, K),
    )


def launch_gluon_mm_a16w16_skinny_gfx950(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """Compute a few rows of dense ``A @ B.T`` by streaming the weight once.

    Args:
        A: Contiguous ``[M, K]`` BF16 activation at a 16-byte aligned address.
        B: Contiguous ``[N, K]`` BF16 weight on ``A``'s device, 16-byte
            aligned; ``(M, N, K)`` must pass
            ``supports_gluon_mm_a16w16_skinny_gfx950``.
        out: Contiguous ``[M, N]`` BF16 destination on ``A``'s device, 16-byte
            aligned, or None to allocate one.

    Returns:
        ``[M, N]`` BF16 product.
    """
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError(
            f"skinny GEMM expects [M, K] x [N, K], got {A.shape} {B.shape}"
        )
    M, K = A.shape
    N = B.shape[0]
    if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
        raise TypeError(f"skinny GEMM expects BF16 inputs, got {A.dtype=} {B.dtype=}")
    if not A.is_cuda or B.device != A.device:
        raise ValueError(
            f"skinny GEMM expects inputs on one GPU, got {A.device} {B.device}"
        )
    if not supports_gluon_mm_a16w16_skinny_gfx950(M, N, K):
        raise ValueError(f"skinny GEMM has no config for M={M}, N={N}, K={K}")
    if not A.is_contiguous() or not B.is_contiguous():
        raise ValueError("skinny GEMM expects contiguous inputs")
    if out is None:
        out = torch.empty((M, N), dtype=A.dtype, device=A.device)
    elif (
        out.shape != (M, N)
        or not out.is_contiguous()
        or out.dtype != A.dtype
        or out.device != A.device
    ):
        raise ValueError(
            f"skinny GEMM out must be a contiguous [{M}, {N}] BF16 tensor on "
            f"{A.device}"
        )
    num_tiles = N // SKINNY_BLOCK_N
    split_k = _SKINNY_CONFIGS[(N, K)].split_k
    partials = _partials(A.device, N, K)
    _check_aligned(activation=A, weight=B, output=out, scratch=partials)
    row_tile = next(tile for tile in SKINNY_ROW_TILES if M <= tile)
    gluon_mm_a16w16_skinny_gfx950[(num_tiles * split_k,)](
        A, B, partials, M, N, K, **_main_options(N, K, row_tile)
    )
    gluon_mm_a16w16_skinny_reduce_gfx950[(-(-M // SKINNY_REDUCE_M) * num_tiles,)](
        partials, out, M, N, **_reduce_options(N, K)
    )
    return out
