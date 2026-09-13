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

"""Dense16 WMMA GEMM epilogues for gfx1250."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton

_LARGEM_MIN_M = 512
_LARGEM_BLOCK_M_CROSSOVER = 12288
_LARGEM_SHAPES = {
    (512, 3072),
    (768, 7168),
    (1536, 7168),
    (1536, 2304),
    (3584, 7168),
    (4224, 7168),
    (7168, 1536),
    (7168, 2112),
    (7168, 3584),
    (7168, 6288),
    (7168, 8448),
}


def use_gluon_largem_gfx1250(m: int, k: int, n: int) -> bool:
    """Return whether CDNA5 large-M WMMA accepts this K3 projection shape."""

    return m >= _LARGEM_MIN_M and (k, n) in _LARGEM_SHAPES


@gluon.jit
def _wmma_tdm_dense_m16_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_om,
    stride_on,
    ACTUAL_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    K: gl.constexpr,
):
    """Fixed-M CDNA5 dense projection candidate."""
    M: gl.constexpr = 16
    pid_n = gl.program_id(0)

    gl.static_assert(
        BLOCK_N == 16 or BLOCK_N == 64,
        "candidate supports one or four WMMA output tiles",
    )
    gl.static_assert(0 < ACTUAL_M and ACTUAL_M <= M)
    gl.static_assert(BLOCK_K == 128, "candidate is tuned for 128-wide K tiles")
    gl.static_assert(K % BLOCK_K == 0, "K must tile exactly into BLOCK_K")
    gl.static_assert(NUM_BUFFERS == 3, "candidate uses a triple-buffer TDM pipeline")

    warp_bases: gl.constexpr = [] if BLOCK_N == 16 else [[0, 1], [0, 2]]
    wmma_layout: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=warp_bases,
        reg_bases=[],
        instr_shape=[16, 16, 32],
    )
    dot_layout_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=wmma_layout, k_width=8
    )
    dot_layout_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=wmma_layout, k_width=8
    )
    shared_layout_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [M, BLOCK_K], [1, 0]
    )
    shared_layout_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [BLOCK_N, BLOCK_K], [1, 0]
    )

    a_smem = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty,
        [NUM_BUFFERS, M, BLOCK_K],
        shared_layout_a,
    )
    b_smem = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty,
        [NUM_BUFFERS, BLOCK_N, BLOCK_K],
        shared_layout_b,
    )
    a_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=a_ptr,
        shape=(ACTUAL_M, K),
        strides=(stride_am, stride_ak),
        block_shape=(M, BLOCK_K),
        layout=shared_layout_a,
    )
    b_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * BLOCK_N * stride_bn,
        shape=(BLOCK_N, K),
        strides=(stride_bn, stride_bk),
        block_shape=(BLOCK_N, BLOCK_K),
        layout=shared_layout_b,
    )

    for tile in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.cdna5.tdm.async_load(a_desc, [0, tile * BLOCK_K], a_smem.index(tile))
        gl.amd.cdna5.tdm.async_load(b_desc, [0, tile * BLOCK_K], b_smem.index(tile))

    acc = gl.zeros((M, BLOCK_N), gl.float32, wmma_layout)
    num_k_tiles: gl.constexpr = K // BLOCK_K
    gl.amd.cdna5.tdm.async_wait(2 * (NUM_BUFFERS - 2))
    for tile in gl.static_range(num_k_tiles):
        with gl.amd.warp_pipeline_stage("tdm+lds", priority=1):
            a = a_smem.index(tile % NUM_BUFFERS).load(layout=dot_layout_a)
            b = (
                b_smem.index(tile % NUM_BUFFERS)
                .permute([1, 0])
                .load(layout=dot_layout_b)
            )
            gl.amd.cdna5.tdm.async_load(
                a_desc,
                [0, (tile + NUM_BUFFERS - 1) * BLOCK_K],
                a_smem.index((tile + NUM_BUFFERS - 1) % NUM_BUFFERS),
                pred=tile + NUM_BUFFERS - 1 < num_k_tiles,
            )
            gl.amd.cdna5.tdm.async_load(
                b_desc,
                [0, (tile + NUM_BUFFERS - 1) * BLOCK_K],
                b_smem.index((tile + NUM_BUFFERS - 1) % NUM_BUFFERS),
                pred=tile + NUM_BUFFERS - 1 < num_k_tiles,
            )
        gl.amd.cdna5.tdm.async_wait(2 * (NUM_BUFFERS - 2))
        with gl.amd.warp_pipeline_stage("wmma", priority=0):
            acc = gl.amd.cdna5.wmma(a, b, acc)
    gl.amd.cdna5.tdm.async_wait(0)

    offs_m = gl.arange(0, M, gl.SliceLayout(1, wmma_layout))
    offs_n = gl.arange(0, BLOCK_N, gl.SliceLayout(0, wmma_layout))
    tile_n = pid_n * BLOCK_N + offs_n
    output_offsets = (offs_m[:, None] * stride_om + tile_n[None, :] * stride_on).to(
        gl.int32
    )
    gl.amd.cdna5.buffer_store(
        acc.to(gl.bfloat16),
        out_ptr,
        output_offsets,
        mask=offs_m[:, None] < ACTUAL_M,
    )


def _launch_wmma_tdm_dense_tiles(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    block_n: int,
    num_warps: int,
) -> None:
    block_k, num_buffers = 128, 3
    for start in range(0, A.shape[0], 16):
        a_tile = A[start : start + 16]
        out_tile = out[start : start + 16]
        _wmma_tdm_dense_m16_kernel[(B.shape[0] // block_n,)](
            a_tile,
            B,
            out_tile,
            a_tile.stride(0),
            a_tile.stride(1),
            B.stride(0),
            B.stride(1),
            out_tile.stride(0),
            out_tile.stride(1),
            ACTUAL_M=a_tile.shape[0],
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            NUM_BUFFERS=num_buffers,
            K=A.shape[1],
            num_warps=num_warps,
            num_stages=1,
            waves_per_eu=1,
        )


def use_gluon_wmma_dense_gfx1250(m: int, k: int, n: int) -> bool:
    """Return whether CDNA5 dense16 WMMA accepts this K3 projection shape.

    M is tiled in 16-row chunks and each chunk re-reads all of B, so the
    advantage shrinks with every chunk added and rocBLAS wins past the
    ceiling. Preferring this over the M == 1 row-CTA GEMV is the registry's
    choice, not this predicate's.
    """

    return 1 <= m <= 32 and k % 128 == 0 and n % 16 == 0


def gluon_wmma_tdm_dense_gfx1250(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the CDNA5 dense projection candidate on any accepted shape."""
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D")
    m, k = A.shape
    n = B.shape[0]
    if B.shape[1] != k:
        raise ValueError(f"B must be [N, {k}], got {tuple(B.shape)}")
    for name, tensor in (("A", A), ("B", B)):
        if (
            tensor.dtype != torch.bfloat16
            or not tensor.is_cuda
            or not tensor.is_contiguous()
            or tensor.device != A.device
        ):
            raise ValueError(f"{name} must be contiguous GPU BF16 colocated with A")
    if not use_gluon_wmma_dense_gfx1250(m, k, n):
        raise ValueError(
            f"dense16 WMMA needs 1 <= M <= 32, K % 128 == 0 and N % 16 == 0, "
            f"got M={m}, K={k}, N={n}"
        )

    if out is None:
        out = A.new_empty((m, n))
    elif (
        tuple(out.shape) != (m, n)
        or out.dtype != torch.bfloat16
        or not out.is_cuda
        or not out.is_contiguous()
        or out.device != A.device
    ):
        raise ValueError(f"out must be contiguous GPU BF16 ({m}, {n}) colocated with A")

    # BLOCK_N selects the WMMA warp bases, so the warp count follows from it.
    block_n, num_warps = (64, 4) if n % 64 == 0 else (16, 1)
    _launch_wmma_tdm_dense_tiles(A, B, out, block_n=block_n, num_warps=num_warps)
    return out


def gluon_wmma_tdm_mla_qkv_gate_gfx1250(
    A: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    """Run the CDNA5 MLA QKV/gate projection candidate."""
    if A.shape[0] not in {1, 2, 4, 8, 16, 32}:
        raise ValueError("A must contain 1, 2, 4, 8, 16, or 32 rows")
    for name, tensor, shape in (
        ("A", A, (A.shape[0], 7168)),
        ("B", B, (3648, 7168)),
    ):
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != torch.bfloat16
            or not tensor.is_cuda
            or not tensor.is_contiguous()
            or tensor.device != A.device
        ):
            raise ValueError(
                f"{name} must be contiguous GPU BF16 {shape} colocated with A"
            )

    out = A.new_empty((A.shape[0], 3648))
    _launch_wmma_tdm_dense_tiles(
        A,
        B,
        out,
        block_n=64,
        num_warps=4,
    )
    return out


def gluon_wmma_tdm_kda_qkvfab_gfx1250(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the CDNA5 KDA QKVFAB projection candidate."""
    if A.shape[0] not in {1, 2, 4, 8, 16, 32}:
        raise ValueError("A must contain 1, 2, 4, 8, 16, or 32 rows")
    for name, tensor, shape in (
        ("A", A, (A.shape[0], 7168)),
        ("B", B, (6288, 7168)),
    ):
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != torch.bfloat16
            or not tensor.is_cuda
            or not tensor.is_contiguous()
            or tensor.device != A.device
        ):
            raise ValueError(
                f"{name} must be contiguous GPU BF16 {shape} colocated with A"
            )

    if out is None:
        out = A.new_empty((A.shape[0], 6288))
    elif (
        tuple(out.shape) != (A.shape[0], 6288)
        or out.dtype != torch.bfloat16
        or not out.is_cuda
        or not out.is_contiguous()
        or out.device != A.device
    ):
        raise ValueError(
            f"out must be contiguous GPU BF16 ({A.shape[0]}, 6288) " "colocated with A"
        )
    _launch_wmma_tdm_dense_tiles(
        A,
        B,
        out,
        block_n=16,
        num_warps=1,
    )
    return out


@gluon.jit
def _wmma_tdm_add3_m16_kernel(
    a_ptr,
    b_ptr,
    addend_a_ptr,
    addend_b_ptr,
    out_ptr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_addend_am,
    stride_addend_an,
    stride_addend_bm,
    stride_addend_bn,
    stride_om,
    stride_on,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
):
    """Fixed-shape CDNA5 TDM/LDS/WMMA projection-plus-add3 kernel."""
    M: gl.constexpr = 16
    K: gl.constexpr = 3584
    pid_n = gl.program_id(0)

    gl.static_assert(BLOCK_N == 64, "candidate uses one 16x64 WMMA output tile")
    gl.static_assert(BLOCK_K == 128, "candidate is tuned for 128-wide K tiles")
    gl.static_assert(NUM_BUFFERS == 3, "candidate uses a triple-buffer TDM pipeline")

    wmma_layout: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=[[0, 1], [0, 2]],
        reg_bases=[],
        instr_shape=[16, 16, 32],
    )
    dot_layout_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=wmma_layout, k_width=8
    )
    dot_layout_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=wmma_layout, k_width=8
    )
    shared_layout_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [M, BLOCK_K], [1, 0]
    )
    shared_layout_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [BLOCK_N, BLOCK_K], [1, 0]
    )

    a_smem = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty,
        [NUM_BUFFERS, M, BLOCK_K],
        shared_layout_a,
    )
    b_smem = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty,
        [NUM_BUFFERS, BLOCK_N, BLOCK_K],
        shared_layout_b,
    )
    a_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        block_shape=(M, BLOCK_K),
        layout=shared_layout_a,
    )
    b_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * BLOCK_N * stride_bn,
        shape=(BLOCK_N, K),
        strides=(stride_bn, stride_bk),
        block_shape=(BLOCK_N, BLOCK_K),
        layout=shared_layout_b,
    )

    # Two tiles are prefetched before the steady-state issue/wait/WMMA loop.
    for tile in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.cdna5.tdm.async_load(a_desc, [0, tile * BLOCK_K], a_smem.index(tile))
        gl.amd.cdna5.tdm.async_load(b_desc, [0, tile * BLOCK_K], b_smem.index(tile))

    acc = gl.zeros((M, BLOCK_N), gl.float32, wmma_layout)
    num_k_tiles: gl.constexpr = K // BLOCK_K
    gl.amd.cdna5.tdm.async_wait(2 * (NUM_BUFFERS - 2))
    for tile in gl.static_range(num_k_tiles):
        with gl.amd.warp_pipeline_stage("tdm+lds", priority=1):
            a = a_smem.index(tile % NUM_BUFFERS).load(layout=dot_layout_a)
            b = (
                b_smem.index(tile % NUM_BUFFERS)
                .permute([1, 0])
                .load(layout=dot_layout_b)
            )
            gl.amd.cdna5.tdm.async_load(
                a_desc,
                [0, (tile + NUM_BUFFERS - 1) * BLOCK_K],
                a_smem.index((tile + NUM_BUFFERS - 1) % NUM_BUFFERS),
                pred=tile + NUM_BUFFERS - 1 < num_k_tiles,
            )
            gl.amd.cdna5.tdm.async_load(
                b_desc,
                [0, (tile + NUM_BUFFERS - 1) * BLOCK_K],
                b_smem.index((tile + NUM_BUFFERS - 1) % NUM_BUFFERS),
                pred=tile + NUM_BUFFERS - 1 < num_k_tiles,
            )
        # Keep one complete A/B batch in flight while consuming the oldest.
        gl.amd.cdna5.tdm.async_wait(2 * (NUM_BUFFERS - 2))
        with gl.amd.warp_pipeline_stage("wmma", priority=0):
            acc = gl.amd.cdna5.wmma(a, b, acc)
    gl.amd.cdna5.tdm.async_wait(0)

    offs_m = gl.arange(0, M, gl.SliceLayout(1, wmma_layout))
    offs_n = gl.arange(0, BLOCK_N, gl.SliceLayout(0, wmma_layout))
    tile_n = pid_n * BLOCK_N + offs_n
    addend_a_offsets = (
        offs_m[:, None] * stride_addend_am + tile_n[None, :] * stride_addend_an
    ).to(gl.int32)
    addend_b_offsets = (
        offs_m[:, None] * stride_addend_bm + tile_n[None, :] * stride_addend_bn
    ).to(gl.int32)
    output_offsets = (offs_m[:, None] * stride_om + tile_n[None, :] * stride_on).to(
        gl.int32
    )

    # Preserve the materialized BF16 projection boundary used by torch.mm.
    projected = acc.to(gl.bfloat16)
    addend_a = gl.amd.cdna5.buffer_load(addend_a_ptr, addend_a_offsets)
    addend_b = gl.amd.cdna5.buffer_load(addend_b_ptr, addend_b_offsets)
    gl.amd.cdna5.buffer_store(
        (projected + addend_a + addend_b).to(gl.bfloat16),
        out_ptr,
        output_offsets,
    )


def gluon_wmma_tdm_add3_m16_gfx1250(
    A: torch.Tensor,
    B: torch.Tensor,
    addend_a: torch.Tensor,
    addend_b: torch.Tensor,
) -> torch.Tensor:
    """Run the Kimi-K3 M=16 CDNA5 TDM/LDS/WMMA projection-plus-add3 path."""
    for name, tensor, shape in (
        ("A", A, (16, 3584)),
        ("B", B, (7168, 3584)),
    ):
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != torch.bfloat16
            or not tensor.is_cuda
            or not tensor.is_contiguous()
            or tensor.device != A.device
        ):
            raise ValueError(
                f"{name} must be contiguous GPU BF16 {shape} colocated with A"
            )
    for name, tensor in (("addend_a", addend_a), ("addend_b", addend_b)):
        if (
            tuple(tensor.shape) != (16, 7168)
            or tensor.dtype != torch.bfloat16
            or not tensor.is_cuda
            or tensor.device != A.device
            or tensor.stride(1) != 1
        ):
            raise ValueError(
                f"{name} must be GPU BF16 (16, 7168) with unit inner stride "
                "colocated with A"
            )

    out = A.new_empty((16, 7168))
    block_n, block_k, num_buffers = 64, 128, 3
    _wmma_tdm_add3_m16_kernel[(7168 // block_n,)](
        A,
        B,
        addend_a,
        addend_b,
        out,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        addend_a.stride(0),
        addend_a.stride(1),
        addend_b.stride(0),
        addend_b.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        NUM_BUFFERS=num_buffers,
        num_warps=4,
        num_stages=1,
        waves_per_eu=1,
    )
    return out


@triton.jit
def _mm_a16w16_add3_m16_kernel(
    A,
    B,
    AddendA,
    AddendB,
    Out,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_aam: tl.constexpr,
    stride_aan: tl.constexpr,
    stride_abm: tl.constexpr,
    stride_abn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BF16_BOUNDARY: tl.constexpr,
):
    """Wave32 WMMA projection with a BF16-boundary add3 epilogue."""

    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, 16)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    output_mask = offs_n[None, :] < N
    if BF16_BOUNDARY:
        # Match torch.mm's materialized BF16 output before the add3 operation.
        projected = acc.to(tl.bfloat16)
    else:
        # Match gfx950's fused MFMA epilogue: add in FP32, cast at store.
        projected = acc
    add_a = tl.load(
        AddendA + offs_m[:, None] * stride_aam + offs_n[None, :] * stride_aan,
        mask=output_mask,
    )
    add_b = tl.load(
        AddendB + offs_m[:, None] * stride_abm + offs_n[None, :] * stride_abn,
        mask=output_mask,
    )
    tl.store(
        Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        projected + add_a + add_b,
        mask=output_mask,
    )


def triton_mm_a16w16_add3_m16_gfx1250(
    A: torch.Tensor,
    B: torch.Tensor,
    addend_a: torch.Tensor,
    addend_b: torch.Tensor,
    *,
    block_n: int = 32,
    block_k: int = 32,
    num_warps: int = 2,
    num_stages: int = 1,
    waves_per_eu: int = 1,
    bf16_boundary: bool = True,
) -> torch.Tensor:
    """Compute the Kimi K3 M16 projection-plus-add3 with wave32 WMMA."""

    if A.shape != (16, 3584) or B.shape != (7168, 3584):
        raise ValueError(
            "gfx1250 add3 requires A [16,3584] and B [7168,3584], "
            f"got {tuple(A.shape)} and {tuple(B.shape)}"
        )
    for name, tensor, expected_shape in (
        ("addend_a", addend_a, (16, 7168)),
        ("addend_b", addend_b, (16, 7168)),
    ):
        if tensor.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
    for name, tensor in (("B", B), ("addend_a", addend_a), ("addend_b", addend_b)):
        if tensor.device != A.device or tensor.dtype != A.dtype:
            raise ValueError(f"{name} must be colocated with A and have A's dtype")
    if A.dtype != torch.bfloat16 or not A.is_cuda or not A.is_contiguous():
        raise ValueError("A must be a contiguous GPU BF16 tensor")
    if not B.is_contiguous():
        raise ValueError("B must be contiguous")
    if addend_a.stride(1) != 1 or addend_b.stride(1) != 1:
        raise ValueError("addends must have unit inner stride")
    if block_n not in (16, 32, 64, 128) or block_k not in (
        16,
        32,
        64,
        128,
        256,
    ):
        raise ValueError("unsupported add3 output or K tile")
    if num_stages not in (1, 2, 3, 4):
        raise ValueError("num_stages must be between 1 and 4")

    out = torch.empty_like(addend_a)
    _mm_a16w16_add3_m16_kernel[(triton.cdiv(B.shape[0], block_n),)](
        A,
        B,
        addend_a,
        addend_b,
        out,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        addend_a.stride(0),
        addend_a.stride(1),
        addend_b.stride(0),
        addend_b.stride(1),
        out.stride(0),
        out.stride(1),
        N=B.shape[0],
        K=A.shape[1],
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BF16_BOUNDARY=bf16_boundary,
        num_warps=num_warps,
        num_stages=num_stages,
        waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim=16,
    )
    return out


@gluon.jit
def _largem_swizzle2d(pid, grid_m, grid_n, GROUP_M: gl.constexpr):
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = gl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    gl.assume(group_size >= 0)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    return pid_m, pid_n


@gluon.jit
def _wmma_tdm_dense_largem_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_om,
    stride_on,
    M,
    N,
    K,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GROUP_M: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
):
    """Dense BF16 CDNA5 TDM/WMMA GEMM for large aligned K3 prefixes."""
    gl.static_assert(BLOCK_K == 128, "large-M path is tuned for 128-wide K tiles")
    gl.static_assert(NUM_BUFFERS == 2, "large-M path uses a double-buffer TDM pipeline")

    pid = gl.program_id(0)
    grid_m = gl.cdiv(M, BLOCK_M)
    grid_n = gl.cdiv(N, BLOCK_N)
    pid_m, pid_n = _largem_swizzle2d(pid, grid_m, grid_n, GROUP_M)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    wmma_layout: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=[[0, 1], [1, 0]],
        reg_bases=[],
        instr_shape=[16, 16, 32],
    )
    dot_layout_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=wmma_layout, k_width=8
    )
    dot_layout_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=wmma_layout, k_width=8
    )
    shared_layout_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [BLOCK_M, BLOCK_K], [1, 0]
    )
    shared_layout_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [BLOCK_N, BLOCK_K], [1, 0]
    )

    a_smem = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty,
        [NUM_BUFFERS, BLOCK_M, BLOCK_K],
        shared_layout_a,
    )
    b_smem = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty,
        [NUM_BUFFERS, BLOCK_N, BLOCK_K],
        shared_layout_b,
    )
    a_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=a_ptr + off_m * stride_am,
        shape=(M - off_m, K),
        strides=(stride_am, stride_ak),
        block_shape=(BLOCK_M, BLOCK_K),
        layout=shared_layout_a,
    )
    b_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=b_ptr + off_n * stride_bn,
        shape=(N - off_n, K),
        strides=(stride_bn, stride_bk),
        block_shape=(BLOCK_N, BLOCK_K),
        layout=shared_layout_b,
    )

    gl.amd.cdna5.tdm.async_load(a_desc, [0, 0], a_smem.index(0))
    gl.amd.cdna5.tdm.async_load(b_desc, [0, 0], b_smem.index(0))

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, wmma_layout)
    num_k_tiles = gl.cdiv(K, BLOCK_K)
    gl.assume(num_k_tiles > 0)
    for i in range(0, num_k_tiles - 1):
        next_idx = i + 1
        gl.amd.cdna5.tdm.async_load(
            a_desc,
            [0, next_idx * BLOCK_K],
            a_smem.index(next_idx % NUM_BUFFERS),
        )
        gl.amd.cdna5.tdm.async_load(
            b_desc,
            [0, next_idx * BLOCK_K],
            b_smem.index(next_idx % NUM_BUFFERS),
        )
        gl.amd.cdna5.tdm.async_wait(2)
        a = a_smem.index(i % NUM_BUFFERS).load(layout=dot_layout_a)
        b = b_smem.index(i % NUM_BUFFERS).permute([1, 0]).load(layout=dot_layout_b)
        acc = gl.amd.cdna5.wmma(a, b, acc)

    gl.amd.cdna5.tdm.async_wait(0)
    last_idx = num_k_tiles - 1
    a = a_smem.index(last_idx % NUM_BUFFERS).load(layout=dot_layout_a)
    b = b_smem.index(last_idx % NUM_BUFFERS).permute([1, 0]).load(layout=dot_layout_b)
    acc = gl.amd.cdna5.wmma(a, b, acc)

    offs_m = off_m + gl.arange(0, BLOCK_M, gl.SliceLayout(1, wmma_layout))
    offs_n = off_n + gl.arange(0, BLOCK_N, gl.SliceLayout(0, wmma_layout))
    output_offsets = (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on).to(
        gl.int32
    )
    gl.amd.cdna5.buffer_store(
        acc.to(gl.bfloat16),
        out_ptr,
        output_offsets,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def gluon_mm_a16w16_largem_gfx1250(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run K3 BF16 projections with CDNA5 WMMA prefixes and vendor tails."""

    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError("gfx1250 large-M projection expects A [M,K], B [N,K]")
    m, k = map(int, A.shape)
    n = int(B.shape[0])
    if not use_gluon_largem_gfx1250(m, k, n):
        raise ValueError(
            f"gfx1250 large-M projection requires M >= {_LARGEM_MIN_M} "
            f"and a K3 shape, got M={m} N={n} K={k}"
        )
    if (
        A.dtype != torch.bfloat16
        or B.dtype != torch.bfloat16
        or not A.is_cuda
        or A.device != B.device
        or not A.is_contiguous()
        or not B.is_contiguous()
    ):
        raise ValueError(
            "gfx1250 large-M projection requires contiguous GPU BF16 inputs"
        )
    if k % 128 != 0:
        raise ValueError(
            f"gfx1250 large-M projection requires K divisible by 128, got K={k}"
        )

    if out is None:
        out = A.new_empty((m, n))
    elif (
        out.shape != (m, n)
        or out.dtype != A.dtype
        or out.device != A.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "gfx1250 large-M output must be contiguous BF16 with shape "
            f"{(m, n)} on {A.device}"
        )

    block_m = 128 if m < _LARGEM_BLOCK_M_CROSSOVER else 256
    tiled_n = n - n % 32
    block_n = next(
        candidate for candidate in (block_m, 128, 64, 32) if tiled_n % candidate == 0
    )
    tiled_m = m - m % block_m
    prefix_out = out if tiled_n == n else A.new_empty((m, tiled_n))
    if tiled_m > 0 and tiled_n > 0:
        a_prefix = A[:tiled_m]
        b_prefix = B[:tiled_n]
        out_prefix = prefix_out[:tiled_m]
        grid = (tiled_m // block_m) * (tiled_n // block_n)
        _wmma_tdm_dense_largem_kernel[(grid,)](
            a_prefix,
            b_prefix,
            out_prefix,
            a_prefix.stride(0),
            a_prefix.stride(1),
            b_prefix.stride(0),
            b_prefix.stride(1),
            out_prefix.stride(0),
            out_prefix.stride(1),
            tiled_m,
            tiled_n,
            k,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=128,
            GROUP_M=8,
            NUM_BUFFERS=2,
            num_warps=4,
            num_stages=1,
        )
    if tiled_m != m:
        torch.mm(A[tiled_m:], B[:tiled_n].T, out=prefix_out[tiled_m:])
    if tiled_n != n:
        out[:, :tiled_n].copy_(prefix_out)
        out[:, tiled_n:].copy_(torch.nn.functional.linear(A, B[tiled_n:]))
    return out


__all__ = [
    "use_gluon_largem_gfx1250",
    "use_gluon_wmma_dense_gfx1250",
    "gluon_mm_a16w16_largem_gfx1250",
    "gluon_wmma_tdm_dense_gfx1250",
    "gluon_wmma_tdm_kda_qkvfab_gfx1250",
    "gluon_wmma_tdm_mla_qkv_gate_gfx1250",
    "gluon_wmma_tdm_add3_m16_gfx1250",
    "triton_mm_a16w16_add3_m16_gfx1250",
]
