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

"""Pipelined dense16 GEMM on gfx950: whole-K tiles, no split-K.

``C[M, N] = A[M, K] @ B[N, K].T`` in BF16, all three contiguous. Each program
computes one ``BLOCK_M x BLOCK_N`` output tile over all of K: A and B tiles
go straight to LDS (direct-to-LDS buffer loads) ``NUM_STAGES - 1`` K tiles
ahead of the MFMA that reads them, and the tile is stored once, so a launch is
a single kernel with no partials, counters or reduction. Programs of one
column tile are consecutive and, after the XCD remap, share an XCD, so the
row tiles of a weight tile meet it in that XCD's L2.

A row's result depends only on K and the config, never on the other rows of
the launch: every output accumulates its K tiles in order.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from tokenspeed_kernel_amd._triton import compile_ahead, gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950._xcd import _remap_xcd

cdna4 = gl.amd.cdna4
async_copy = gl.amd.cdna4.async_copy

# Alignment in bytes of every tensor's base address: the direct-to-LDS loads
# need it. The kernel states it (multiple_of) since it does not specialize
# pointers, and the launcher checks it.
_POINTER_ALIGN_BYTES = gl.constexpr(16)


def _pipelined_launch_metadata(grid, kernel, args):
    m, n, k = args["M"], args["N"], args["K"]
    elem = args["a_ptr"].element_size()
    column_tiles = triton.cdiv(n, args["BLOCK_N"])
    row_tiles = triton.cdiv(m, args["BLOCK_M"])
    return {
        "name": f"{kernel.name} [M={m}, N={n}, K={k}]",
        "flops16": 2 * m * n * k,
        # Every column tile reads A; every row tile reads its weight tile
        # (the later ones mostly from L2).
        "bytes": (column_tiles * m * k + row_tiles * n * k + m * n) * elem,
    }


@gluon.constexpr_function
def _direct_load_layout(shared_layout, num_warps: int):
    """Register layout whose waves each write one contiguous LDS span.

    The first three offset bases of the padded shared layout are the eight
    elements of a lane's 16-byte load, the next six its lane, so one wave's
    load fills one 512-element padding interval; the following bases pick the
    wave and then the further loads of a lane.
    """
    bases = [list(b) for b in shared_layout.offset_bases]
    warp_bits = num_warps.bit_length() - 1
    return gl.DistributedLinearLayout(
        reg_bases=bases[0:3] + bases[9 + warp_bits :],
        lane_bases=bases[3:9],
        warp_bases=bases[9 : 9 + warp_bits],
        block_bases=[],
        shape=shared_layout.shape,
    )


@gluon.jit(
    launch_metadata=_pipelined_launch_metadata,
    do_not_specialize=("M", "a_ptr", "b_ptr", "c_ptr"),
)
def gluon_mm_a16w16_pipelined_gfx950(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N: gl.constexpr,
    K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    NUM_XCDS: gl.constexpr,
):
    """One output tile over all of K; see the module docstring.

    ``a_ptr`` is a contiguous ``[M, K]`` array, ``b_ptr`` a contiguous
    ``[N, K]`` one and ``c_ptr`` a contiguous ``[M, N]`` one, all 16-byte
    aligned. Program ``pid`` (after the XCD remap) computes row tile
    ``pid % cdiv(M, BLOCK_M)`` of column tile ``pid // cdiv(M, BLOCK_M)``.
    """
    NUM_WARPS: gl.constexpr = WARPS_M * WARPS_N
    gl.static_assert(K % BLOCK_K == 0, "pipelined GEMM needs whole K tiles")
    K_TILES: gl.constexpr = K // BLOCK_K
    gl.static_assert(NUM_STAGES >= 2 and K_TILES >= NUM_STAGES)
    # Tile offsets are int32 element offsets from the int64 bases below;
    # buffer loads and stores address in bytes.
    ELEM_BYTES: gl.constexpr = a_ptr.dtype.element_ty.primitive_bitwidth // 8
    gl.static_assert(
        (BLOCK_M + 1) * K * ELEM_BYTES < 2**31
        and (BLOCK_N + 1) * K * ELEM_BYTES < 2**31
        and (BLOCK_M + 1) * N * ELEM_BYTES < 2**31,
        "pipelined GEMM tile byte offsets must fit int32",
    )

    pid = gl.program_id(axis=0)
    num_m = gl.cdiv(M, BLOCK_M)
    num_n: gl.constexpr = (N + BLOCK_N - 1) // BLOCK_N
    if NUM_XCDS > 1:
        pid = _remap_xcd(pid, num_m * num_n, NUM_XCDS)
    pid_m = pid % num_m
    pid_n = pid // num_m
    row0 = pid_m * BLOCK_M
    col0 = pid_n * BLOCK_N

    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[WARPS_M, WARPS_N],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=8)
    dot_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=8)
    shared_a: gl.constexpr = cdna4.compute_efficient_padded_shared_layout(
        dot_a, [BLOCK_M, BLOCK_K], a_ptr.dtype.element_ty, is_k_contig=True
    )
    shared_b: gl.constexpr = cdna4.compute_efficient_padded_shared_layout(
        dot_b, [BLOCK_K, BLOCK_N], b_ptr.dtype.element_ty, is_k_contig=True
    )
    load_a: gl.constexpr = _direct_load_layout(shared_a, NUM_WARPS)
    load_b: gl.constexpr = _direct_load_layout(shared_b, NUM_WARPS)
    smem_a = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [NUM_STAGES, BLOCK_M, BLOCK_K], shared_a
    )
    smem_b = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [NUM_STAGES, BLOCK_K, BLOCK_N], shared_b
    )

    # Rows past M and columns past N clamp onto the last row or column, so
    # every load stays in bounds; the store mask drops their outputs.
    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, load_a))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, load_a))
    offs_bn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, load_b))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, load_b))
    rows = gl.minimum(row0 + offs_am, M - 1) - row0
    cols = gl.minimum(col0 + offs_bn, N - 1) - col0
    # Advance the bases once per program in int64. Rows are whole numbers of
    # 16-byte runs (the launcher checks K), so they keep the alignment.
    a_base = gl.multiple_of(a_ptr + row0.to(gl.int64) * K, _POINTER_ALIGN_BYTES)
    b_base = gl.multiple_of(b_ptr + col0.to(gl.int64) * K, _POINTER_ALIGN_BYTES)
    a_offs = rows[:, None] * K + offs_ak[None, :]
    b_offs = cols[None, :] * K + offs_bk[:, None]

    for s in gl.static_range(NUM_STAGES - 1):
        async_copy.buffer_load_to_shared(smem_a.index(s), a_base, a_offs + s * BLOCK_K)
        async_copy.buffer_load_to_shared(smem_b.index(s), b_base, b_offs + s * BLOCK_K)
        async_copy.commit_group()

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma)
    for t in range(K_TILES):
        # Tile t has landed for every wave, and every wave is done reading
        # the buffer tile t - 1 used, which the next load refills.
        async_copy.wait_group(NUM_STAGES - 2)
        gl.barrier()
        # The last NUM_STAGES - 1 iterations reload the last tile (an L2 hit)
        # into a buffer no MFMA reads, which keeps the wait counts uniform.
        k_next = gl.minimum(t + NUM_STAGES - 1, K_TILES - 1) * BLOCK_K
        refill = (t + NUM_STAGES - 1) % NUM_STAGES
        async_copy.buffer_load_to_shared(smem_a.index(refill), a_base, a_offs + k_next)
        async_copy.buffer_load_to_shared(smem_b.index(refill), b_base, b_offs + k_next)
        async_copy.commit_group()
        a = async_copy.load_shared_relaxed(smem_a.index(t % NUM_STAGES), dot_a)
        b = async_copy.load_shared_relaxed(smem_b.index(t % NUM_STAGES), dot_b)
        acc = cdna4.mfma(a, b, acc)
    async_copy.wait_group(0)

    # Rows of 8 contiguous outputs a lane store 16 bytes at a time.
    lanes_n: gl.constexpr = BLOCK_N // 8 if BLOCK_N // 8 < 64 else 64
    store_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8], [64 // lanes_n, lanes_n], [NUM_WARPS, 1], [1, 0]
    )
    out = gl.convert_layout(acc.to(c_ptr.dtype.element_ty), store_layout)
    offs_cm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, store_layout))
    offs_cn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, store_layout))
    c_base = gl.multiple_of(c_ptr + row0.to(gl.int64) * N + col0, _POINTER_ALIGN_BYTES)
    mask = ((row0 + offs_cm)[:, None] < M) & ((col0 + offs_cn)[None, :] < N)
    cdna4.buffer_store(
        ptr=c_base,
        offsets=offs_cm[:, None] * N + offs_cn[None, :],
        stored_value=out,
        mask=mask,
    )


class _PipelinedConfig(NamedTuple):
    block_m: int
    block_n: int
    block_k: int
    warps_m: int
    warps_n: int
    num_stages: int
    # XCDs the programs are dealt over.
    num_xcds: int
    # First and last row count the kernel takes.
    first_m: int
    last_m: int


# By (N, K): shapes and row counts measured faster than the other gemm.mv
# and gemm.mm solutions (hipBLASLt included) inside a captured decode-like
# graph on MI350X, where every call meets a cold weight after an 84 MB
# hipBLASLt GEMM and reads an activation an elementwise kernel just wrote.
_PIPELINED_CONFIGS: dict[tuple[int, int], _PipelinedConfig] = {
    # Qwen3.8-Flash-Next MoE router gate (512 experts) past the decode GEMM's
    # 64 rows: 2560 / 256 = 10 K tiles over 32 column tiles. Faster than
    # hipBLASLt's split-K at 65-320 rows. At 8-64 rows it ties the decode
    # GEMM, which keeps them.
    (512, 2560): _PipelinedConfig(32, 16, 256, 2, 1, 3, 8, first_m=65, last_m=320),
    # Qwen3.8-Flash-Next LM head (248320-entry vocabulary) at 49-64 rows
    # (draft steps of 49-64 requests, verify of 13-16), where it beats
    # hipBLASLt; at 48 rows and below hipBLASLt is faster.
    (248320, 2560): _PipelinedConfig(64, 256, 64, 2, 4, 3, 8, first_m=49, last_m=64),
}


def supports_gluon_mm_a16w16_pipelined_gfx950(M: int, N: int, K: int) -> bool:
    """Whether the pipelined GEMM has a measured config for this problem.

    Args:
        M: Activation rows.
        N: Weight rows (output columns).
        K: Inner dimension.

    Returns:
        True when ``(N, K)`` has a config whose row range holds ``M``.
    """
    config = _PIPELINED_CONFIGS.get((N, K))
    return config is not None and config.first_m <= M <= config.last_m


def supports_gluon_mm_a16w16_pipelined_weight_gfx950(N: int, K: int) -> bool:
    """Whether the pipelined GEMM has measured rows for an ``[N, K]`` weight."""
    return (N, K) in _PIPELINED_CONFIGS


# The launch and the ahead-of-time compile take their options from this one
# helper, so they name the same binary.
def _options(N: int, K: int) -> dict[str, int]:
    config = _PIPELINED_CONFIGS[(N, K)]
    return dict(
        N=N,
        K=K,
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        WARPS_M=config.warps_m,
        WARPS_N=config.warps_n,
        NUM_STAGES=config.num_stages,
        NUM_XCDS=config.num_xcds,
        num_warps=config.warps_m * config.warps_n,
    )


def _check_aligned(**tensors: torch.Tensor) -> None:
    align = _POINTER_ALIGN_BYTES.value
    for name, tensor in tensors.items():
        if tensor.data_ptr() % align:
            raise ValueError(
                f"pipelined GEMM needs a {align}-byte aligned {name}; got "
                f"{tensor.data_ptr():#x}"
            )


def compile_gluon_mm_a16w16_pipelined_gfx950(B: torch.Tensor) -> None:
    """Compile, without launching, the pipelined GEMM binary for a weight.

    The binary depends on the weight's shape only: not the row count, and not
    where the activation, the weight or the output live (the kernel does not
    specialize its pointers). After this no launch on an ``[N, K]`` weight
    compiles.

    Args:
        B: Contiguous ``[N, K]`` BF16 weight on a GPU;
            ``supports_gluon_mm_a16w16_pipelined_weight_gfx950(N, K)`` must hold.
    """
    if B.ndim != 2 or B.dtype != torch.bfloat16 or not B.is_cuda:
        raise ValueError(
            f"pipelined GEMM expects a BF16 [N, K] weight on a GPU, got {B.dtype} "
            f"{tuple(B.shape)} on {B.device}"
        )
    N, K = B.shape
    if not supports_gluon_mm_a16w16_pipelined_weight_gfx950(N, K):
        raise ValueError(f"pipelined GEMM has no config for N={N}, K={K}")
    if not B.is_contiguous():
        raise ValueError("pipelined GEMM expects a contiguous weight")
    rows = _PIPELINED_CONFIGS[(N, K)].first_m
    A = torch.empty((rows, K), dtype=B.dtype, device=B.device)
    out = torch.empty((rows, N), dtype=B.dtype, device=B.device)
    compile_ahead(gluon_mm_a16w16_pipelined_gfx950, A, B, out, rows, **_options(N, K))


def launch_gluon_mm_a16w16_pipelined_gfx950(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """Compute dense ``A @ B.T`` with whole-K pipelined tiles in one launch.

    Args:
        A: Contiguous ``[M, K]`` BF16 activation at a 16-byte aligned address.
        B: Contiguous ``[N, K]`` BF16 weight on ``A``'s device, 16-byte
            aligned; ``(M, N, K)`` must pass
            ``supports_gluon_mm_a16w16_pipelined_gfx950``.
        out: Contiguous ``[M, N]`` BF16 destination on ``A``'s device, 16-byte
            aligned, or None to allocate one.

    Returns:
        ``[M, N]`` BF16 product.
    """
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError(
            f"pipelined GEMM expects [M, K] x [N, K], got {A.shape} {B.shape}"
        )
    M, K = A.shape
    N = B.shape[0]
    if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
        raise TypeError(
            f"pipelined GEMM expects BF16 inputs, got {A.dtype=} {B.dtype=}"
        )
    if not A.is_cuda or B.device != A.device:
        raise ValueError(
            f"pipelined GEMM expects inputs on one GPU, got {A.device} {B.device}"
        )
    if not supports_gluon_mm_a16w16_pipelined_gfx950(M, N, K):
        raise ValueError(f"pipelined GEMM has no config for M={M}, N={N}, K={K}")
    if not A.is_contiguous() or not B.is_contiguous():
        raise ValueError("pipelined GEMM expects contiguous inputs")
    if out is None:
        out = torch.empty((M, N), dtype=A.dtype, device=A.device)
    elif (
        out.shape != (M, N)
        or not out.is_contiguous()
        or out.dtype != A.dtype
        or out.device != A.device
    ):
        raise ValueError(
            f"pipelined GEMM out must be a contiguous [{M}, {N}] BF16 tensor on "
            f"{A.device}"
        )
    _check_aligned(activation=A, weight=B, output=out)
    config = _PIPELINED_CONFIGS[(N, K)]
    grid = triton.cdiv(M, config.block_m) * triton.cdiv(N, config.block_n)
    gluon_mm_a16w16_pipelined_gfx950[(grid,)](A, B, out, M, **_options(N, K))
    return out
