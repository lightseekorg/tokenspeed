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


"""Fixed-reduction BF16 projections with FP32 output on gfx950."""

from __future__ import annotations

from typing import Any

import torch
from tokenspeed_kernel_amd._extern import extern_compile_options
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950._xcd import _remap_xcd
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (
    _get_partial_scratch,
    _mfma_lds_gload_layout_a,
    _mfma_lds_gload_layout_b,
    _mfma_lds_shared_layout_a,
    _mfma_lds_shared_layout_b,
    _tile_to_pid,
    gluon_mm_a16w16_medium_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.splitk import check_splitk_counters

cdna4 = gl.amd.cdna4
async_copy = gl.amd.cdna4.async_copy


def supports_gluon_mm_bf16_fp32_gfx950(m: int, n: int, k: int) -> bool:
    if m < 0 or (n, k) not in {
        (8, 4608),
        (384, 4608),
        (3792, 4608),
        (3808, 4608),
        (4608, 1536),
    }:
        return False
    if max(m * k * 2, m * n * 4) > 2**32:
        return False
    # Split-K forms partial-tile element offsets with signed int32 arithmetic.
    tiles = triton.cdiv(m, 16) * triton.cdiv(n, 64)
    split_k = projection_split_k(n, k)
    return split_k == 1 or tiles * split_k * 16 * 64 <= 2**31


def _check_buffer_span(tensor: torch.Tensor) -> None:
    # Raw-buffer instructions use 32-bit byte offsets from the view's base.
    span = (
        sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
        + 1
    )
    if tensor.numel() and span * tensor.element_size() > 2**32:
        raise ValueError("gfx950 BF16 projection requires 32-bit buffer offsets")


# Fixed K partitions per projection shape. They shorten FP32 accumulation without
# a batch-dependent tree, and give decode's one 16-row tile enough CTAs: without
# a split, the router's 6 tiles (N=384) and wo's 72 tiles each walk all of K.
# The split does not depend on M, so a row's result does not depend on its
# batch. The splits were chosen from cold timings at 8, 128 and 2048 rows: they
# shorten decode launches several times. Larger wo splits gain little more at 8
# rows and cost more at 2048. A split must divide K into
# 64-wide blocks. A full attention layer's QKV projection (3808: a local
# layer's 3792 and its K and V tap rows) takes the local split, on the same
# 60 column tiles.
_PROJECTION_SPLIT_K = {
    (3792, 4608): 12,
    (3808, 4608): 12,
    (384, 4608): 18,
    (4608, 1536): 3,
}


def projection_split_k(n: int, k: int) -> int:
    """K slices of the projection of an ``[N, K]`` weight. The decode
    monokernels' BF16 projections follow them to give the same bits."""
    return _PROJECTION_SPLIT_K.get((n, k), 1)


# Rows past which a projection runs every K split in one program
# (gluon_mm_bf16_fp32_serial_gfx950) rather than a program a split, and past
# which it takes 128x128 tiles rather than 64x64. Every choice gives the same
# bits, so the rows only choose speed. The thresholds come from cold timings
# from 24 to 4096 rows: past them the serial program is several times faster
# than a program a split at prefill sizes and close to hipBLASLt. The router's
# 6 column tiles leave too few programs below 512 rows, and 128-wide tiles
# never pay off for it.
_SERIAL_PAST_ROWS = {
    (3792, 4608): 64,
    (3808, 4608): 64,
    (384, 4608): 512,
    (4608, 1536): 64,
}
_WIDE_TILE_PAST_ROWS = 768


def serial_projection_tile(m: int, n: int, k: int) -> tuple[int, int] | None:
    """The ``(BLOCK_M, BLOCK_N)`` tile of the serial-split kernel that
    ``launch_gluon_mm_bf16_fp32_gfx950`` runs for ``[m, k] @ [n, k].T``, or
    None for the program-a-split kernel (and the narrow one)."""
    past = _SERIAL_PAST_ROWS.get((n, k))
    if past is None or m <= past:
        return None
    if m > _WIDE_TILE_PAST_ROWS and n >= 1024:
        return 128, 128
    return 64, 64


def gluon_mm_bf16_fp32_num_counters_gfx950(m: int, n: int, k: int) -> int:
    """Return int32 arrival-count entries for A[m,k] @ B[n,k].T.

    No entries are required when the projection does not partition K over
    programs.
    """
    if projection_split_k(n, k) == 1 or serial_projection_tile(m, n, k) is not None:
        return 0
    return triton.cdiv(m, 16) * triton.cdiv(n, 64)


def _projection_metadata(
    grid: tuple[int, ...], kernel: Any, args: dict[str, Any]
) -> dict[str, str | int]:
    m, k = args["A"].shape
    n = args["B"].shape[0]
    return {
        "name": kernel.name,
        "flops32": 2 * m * n * k,
        "bytes": (m * k + n * k) * 2 + m * n * 4,
    }


@gluon.jit(launch_metadata=_projection_metadata)
def gluon_mm_bf16_fp32_narrow_gfx950(
    A,
    B,
    C,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    N: gl.constexpr,
    K: gl.constexpr,
    BLOCK_K: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 64], [4, 1], [1, 0])
    row = gl.program_id(0)
    columns = gl.program_id(1) * 4 + gl.arange(0, 4, gl.SliceLayout(1, layout))
    k = gl.arange(0, BLOCK_K, gl.SliceLayout(0, layout))
    a = gl.load(A + row * stride_am + k * stride_ak, k < K, other=0).to(gl.float32)
    b = gl.load(
        B + columns[:, None] * stride_bn + k[None, :] * stride_bk,
        (columns[:, None] < N) & (k[None, :] < K),
        other=0,
    ).to(gl.float32)
    value = gl.sum(a[None, :] * b, axis=1)
    gl.store(C + row * stride_cm + columns * stride_cn, value, columns < N)


@gluon.jit(launch_metadata=_projection_metadata)
def gluon_mm_bf16_fp32_serial_gfx950(
    A,
    B,
    C,
    M,
    N,
    stride_am,
    stride_bn,
    stride_cm,
    stride_cn,
    K: gl.constexpr,
    SPLIT_K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    NUM_XCDS: gl.constexpr,
):
    """The split-K projection's bits with every split in one program.

    Each program owns a ``BLOCK_M x BLOCK_N`` output tile and walks K once.
    Split ``s`` of the ``SPLIT_K`` slices is one MFMA chain from zero over
    its K range, in K order, with the instruction and operand layouts of
    ``gluon_mm_a16w16_medium_gfx950``; the program adds the splits' sums in
    split order, as that kernel's last-arriving program does. So every output
    has the bits of the split-K kernel without its FP32 partials in memory.
    """
    BLOCK_K: gl.constexpr = 64
    K_TILES: gl.constexpr = K // BLOCK_K
    SPLIT_TILES: gl.constexpr = K_TILES // SPLIT_K
    LOOKAHEAD: gl.constexpr = NUM_BUFFERS - 1
    gl.static_assert(K % (SPLIT_K * BLOCK_K) == 0, "K must split into 64-wide tiles")
    gl.static_assert(SPLIT_TILES >= 1 and NUM_BUFFERS >= 2)
    gl.static_assert(K_TILES >= LOOKAHEAD)
    NUM_WARPS: gl.constexpr = WARPS_M * WARPS_N

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    if NUM_XCDS > 1:
        pid = _remap_xcd(pid, num_pid_m * num_pid_n, NUM_XCDS)
    pid_m, pid_n = _tile_to_pid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    gLoadLayoutA: gl.constexpr = _mfma_lds_gload_layout_a(BLOCK_M, BLOCK_K, NUM_WARPS)
    gLoadLayoutB: gl.constexpr = _mfma_lds_gload_layout_b(BLOCK_N, BLOCK_K, NUM_WARPS)
    # The split-K kernel's MFMA instruction and operand layouts: they fix
    # each output's chain of products.
    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[WARPS_M, WARPS_N],
    )
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfmaLayout, k_width=8
    )
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfmaLayout, k_width=8
    )
    sharedLayoutA: gl.constexpr = _mfma_lds_shared_layout_a(
        dotOpLayoutA, BLOCK_M, BLOCK_K, A.dtype.element_ty
    )
    sharedLayoutB: gl.constexpr = _mfma_lds_shared_layout_b(
        dotOpLayoutB, BLOCK_N, BLOCK_K, B.dtype.element_ty
    )
    smemA = gl.allocate_shared_memory(
        A.dtype.element_ty, [NUM_BUFFERS, BLOCK_M, BLOCK_K], sharedLayoutA
    )
    smemB = gl.allocate_shared_memory(
        B.dtype.element_ty, [NUM_BUFFERS, BLOCK_K, BLOCK_N], sharedLayoutB
    )

    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))
    offs_bn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))
    # The host bounds every view to 32-bit offsets from its base
    # (supports_gluon_mm_bf16_fp32_gfx950, _check_buffer_span).
    a_offs = (
        pid_m * BLOCK_M * stride_am + offs_am[:, None] * stride_am + offs_ak[None, :]
    ).to(gl.int32)
    # Clamp weight rows past N to the last row; the store mask drops them.
    b_rows = gl.minimum(pid_n * BLOCK_N + offs_bn, N - 1)
    b_offs = (offs_bk[:, None] + b_rows[None, :] * stride_bn).to(gl.int32)
    a_mask = (pid_m * BLOCK_M + offs_am)[:, None] < M

    for first in gl.static_range(LOOKAHEAD):
        async_copy.buffer_load_to_shared(
            smemA.index(first),
            A,
            (a_offs + first * BLOCK_K).to(gl.int32),
            mask=a_mask,
        )
        async_copy.buffer_load_to_shared(
            smemB.index(first), B, (b_offs + first * BLOCK_K).to(gl.int32)
        )
        async_copy.commit_group()

    # -0.0 is the identity of FP32 addition: the first add leaves split 0's
    # sum, -0.0 included, as the split-K kernel starts from its partial.
    total = gl.full((BLOCK_M, BLOCK_N), -0.0, gl.float32, mfmaLayout)
    for split in range(SPLIT_K):
        acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfmaLayout)
        for j in gl.static_range(SPLIT_TILES):
            t = split * SPLIT_TILES + j
            # Tile t landed for this wave; the barrier makes every wave's part
            # visible and retires the reads of the buffer refilled next.
            async_copy.wait_group(LOOKAHEAD - 1)
            gl.barrier()
            # Past the last tile, reload it into a buffer no step reads again
            # (keeps one copy group a step).
            ahead = gl.minimum(t + LOOKAHEAD, K_TILES - 1) * BLOCK_K
            refill = (t + LOOKAHEAD) % NUM_BUFFERS
            async_copy.buffer_load_to_shared(
                smemA.index(refill), A, (a_offs + ahead).to(gl.int32), mask=a_mask
            )
            async_copy.buffer_load_to_shared(
                smemB.index(refill), B, (b_offs + ahead).to(gl.int32)
            )
            async_copy.commit_group()
            current = t % NUM_BUFFERS
            a = async_copy.load_shared_relaxed(smemA.index(current), dotOpLayoutA)
            b = async_copy.load_shared_relaxed(smemB.index(current), dotOpLayoutB)
            acc = cdna4.mfma(a, b, acc)
        total = total + acc
    async_copy.wait_group(0)

    offs_cm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, mfmaLayout))
    offs_cn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, mfmaLayout))
    cm = pid_m * BLOCK_M + offs_cm
    cn = pid_n * BLOCK_N + offs_cn
    gl.store(
        C + cm[:, None].to(gl.int64) * stride_cm + cn[None, :] * stride_cn,
        total,
        (cm[:, None] < M) & (cn[None, :] < N),
    )


def _projection_operands(
    A: torch.Tensor, B: torch.Tensor, out: torch.Tensor | None
) -> torch.Tensor:
    """Check the operands of a measured projection; return ``out`` or a new
    FP32 ``[M, N]``."""
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError("BF16 projection requires A[M,K] and B[N,K]")
    if A.dtype != torch.bfloat16 or B.dtype != A.dtype:
        raise ValueError("BF16 projection requires BF16 operands")
    if not A.is_cuda or A.device != B.device:
        raise ValueError("BF16 projection requires colocated GPU operands")
    m, k = A.shape
    n = B.shape[0]
    if not supports_gluon_mm_bf16_fp32_gfx950(m, n, k):
        raise ValueError(f"Unsupported FP32 projection shape {(m, n, k)}")
    if A.stride(1) != 1 or B.stride(1) != 1:
        raise ValueError("gfx950 BF16 projection requires unit inner strides")
    _check_buffer_span(A)
    _check_buffer_span(B)
    if out is None:
        out = torch.empty((m, n), dtype=torch.float32, device=A.device)
    if out.shape != (m, n) or out.dtype != torch.float32 or out.device != A.device:
        raise ValueError("Projection output must be colocated FP32 [M,N]")
    _check_buffer_span(out)
    return out


def _lds_aligned(tensor: torch.Tensor) -> torch.Tensor:
    # Direct-to-LDS loads need 32-bit alignment: copy odd-padded or offset views.
    if tensor.stride(0) % 2 or tensor.data_ptr() % 4:
        return tensor.clone(memory_format=torch.contiguous_format)
    return tensor


def launch_gluon_mm_bf16_fp32_gfx950(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor | None,
    *,
    counters: torch.Tensor,
) -> torch.Tensor:
    """Compute measured BF16 A[M,K] @ B[N,K].T with FP32 output.

    A and B are colocated GPU matrices with unit inner strides. The optional
    out tensor is FP32 [M,N] and may have row padding. Return out, or a new tensor.
    Each view must fit within the kernel's 32-bit buffer-offset range.
    Counters are zeroed int32 storage of at least the reported counter count;
    every launch resets them to zero for subsequent eager or graph execution.
    The kernel follows the rows (``serial_projection_tile``); every choice
    gives each output the same bits.
    """
    m, k = A.shape
    tile = serial_projection_tile(m, B.shape[0], k)
    if tile is not None:
        return launch_gluon_mm_bf16_fp32_serial_gfx950(A, B, out, tile=tile)
    return launch_gluon_mm_bf16_fp32_split_gfx950(A, B, out, counters=counters)


def launch_gluon_mm_bf16_fp32_split_gfx950(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor | None,
    *,
    counters: torch.Tensor,
) -> torch.Tensor:
    """``launch_gluon_mm_bf16_fp32_gfx950`` with a program a K split (the
    narrow kernel for 8 columns), whatever the rows; arguments as there."""
    out = _projection_operands(A, B, out)
    m, k = A.shape
    n = B.shape[0]
    if not m:
        return out
    if n == 8:
        gluon_mm_bf16_fp32_narrow_gfx950[(m, triton.cdiv(n, 4))](
            A,
            B,
            out,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            out.stride(0),
            out.stride(1),
            N=n,
            K=k,
            BLOCK_K=triton.next_power_of_2(k),
            num_warps=4,
            enable_fp_fusion=False,
        )
    else:
        A = _lds_aligned(A)
        B = _lds_aligned(B)
        split_k = projection_split_k(n, k)
        tiles_m, tiles_n = triton.cdiv(m, 16), triton.cdiv(n, 64)
        num_tiles = tiles_m * tiles_n
        if split_k > 1:
            check_splitk_counters(counters, num_tiles, A.device, "gfx950 FP32 GEMM")
            # Bucket capacity, not arithmetic, to bound eager scratch variants.
            capacity = triton.next_power_of_2(tiles_m) * tiles_n
            partial = _get_partial_scratch(A.device, capacity, split_k, 64, 16)
        else:
            partial = counters = out
        gluon_mm_a16w16_medium_gfx950[(num_tiles * split_k,)](
            A,
            B,
            out,
            out,
            out,
            partial,
            counters,
            m,
            n,
            k,
            A.stride(0),
            A.stride(1),
            B.stride(1),
            B.stride(0),
            out.stride(0),
            out.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_M=16,
            BLOCK_N=64,
            BLOCK_K=64,
            WARPS_M=1,
            WARPS_N=4,
            NUM_BUFFERS=2,
            GROUP_SIZE_M=2,
            ADD3=False,
            SPLIT_K=split_k,
            NUM_XCDS=1,
            PID_OFFSET=0,
            num_warps=4,
            **extern_compile_options(),
            llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),),
        )
    return out


# Per tile: waves (M x N), LDS buffers. Three buffers hide more latency on the
# 64x64 tiles; two leave the 128x128 tiles' LDS for occupancy.
_SERIAL_TILE_CONFIG = {(64, 64): (1, 4, 3), (128, 128): (2, 4, 2)}


def launch_gluon_mm_bf16_fp32_serial_gfx950(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor | None,
    *,
    tile: tuple[int, int],
) -> torch.Tensor:
    """``launch_gluon_mm_bf16_fp32_gfx950`` with every K split in one program
    on ``tile`` (``(BLOCK_M, BLOCK_N)``, a key of ``_SERIAL_TILE_CONFIG``),
    whatever the rows: the split-K kernel's bits. Other arguments as there,
    without counters."""
    out = _projection_operands(A, B, out)
    m, k = A.shape
    n = B.shape[0]
    if n == 8:
        raise ValueError("the 8-column projection takes the narrow kernel")
    warps_m, warps_n, buffers = _SERIAL_TILE_CONFIG[tile]
    if not m:
        return out
    A = _lds_aligned(A)
    B = _lds_aligned(B)
    block_m, block_n = tile
    gluon_mm_bf16_fp32_serial_gfx950[
        (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    ](
        A,
        B,
        out,
        m,
        n,
        A.stride(0),
        B.stride(0),
        out.stride(0),
        out.stride(1),
        K=k,
        SPLIT_K=projection_split_k(n, k),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        WARPS_M=warps_m,
        WARPS_N=warps_n,
        NUM_BUFFERS=buffers,
        GROUP_SIZE_M=8,
        NUM_XCDS=1,
        num_warps=warps_m * warps_n,
    )
    return out
