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
_LARGEM_WIDE_N = 3072
# From this many rows the 256 x 256 warp pipeline tile fills the device.
_LARGEM_WIDE_M = 4096
_LARGEM_BLOCK_K = 128
_LARGEM_WARP_PIPELINE_GROUP_M = 4
_LARGEM_WARP_PIPELINE_MIN_N = 1536
# A 2 x 2 CTA cluster multicasts each A and B panel to two CTAs, halving
# their L2 reads; it measured best across the K3 shapes, ragged ones included.
_LARGEM_WARP_PIPELINE_CLUSTER = (2, 2)
# MI450 has eight XCDs, each with its own L2; workgroups are dispatched
# round-robin over them.
_LARGEM_NUM_XCDS = 8
_DENSE_M64_MIN_M = 17
_DENSE_MAX_M = 64
_DENSE_M64_MAX_N = 8192
# This package runs on gfx1250, and that part has 256 CUs.
_NUM_CUS = 256
_REDUCE_BLOCK_N = 512
_MID_M_SPLIT_K = (5, 4, 2)
_LDS_BYTES = 320 * 1024
# Requested TDM buffers; _dense_num_buffers caps them by K and LDS.
_DENSE_BUFFERS = 8
_WARP_BASES_1 = ()
_WARP_BASES_2 = ((0, 1),)
_WARP_BASES_M2 = ((1, 0),)
_WARP_BASES_M4 = ((1, 0), (2, 0))
_WARP_BASES_M8 = ((1, 0), (2, 0), (4, 0))
_WARP_BASES_4 = ((0, 1), (1, 0))
_WARP_BASES_8 = ((0, 1), (1, 0), (2, 0))
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


def _wmma_tdm_dense_launch_metadata(grid, kernel, args):
    """Report dense WMMA work, partial or output stores, and add3 operand reads."""
    m = args["ACTUAL_M"]
    n = args["N"] * args["GROUPS"]
    k = args["K"]
    split_k = args["SPLIT_K"]
    output = args["partial_ptr"] if split_k > 1 else args["out_ptr"]
    addends = 2 if args["ADD3"] else 0
    return {
        "name": kernel.name,
        "flops16": 2 * m * n * k,
        "bytes": grid[0] * m * k * args["a_ptr"].element_size()
        + grid[2] * n * k * args["b_ptr"].element_size()
        + split_k * m * n * output.element_size()
        + addends * m * n * args["out_ptr"].element_size(),
    }


@gluon.jit(
    launch_metadata=_wmma_tdm_dense_launch_metadata,
    do_not_specialize=["ACTUAL_M", "split_stride"],
)
def _wmma_tdm_dense_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_om,
    stride_on,
    partial_ptr,
    split_stride,
    partial_row_stride,
    addend_a_ptr,
    addend_b_ptr,
    stride_addend_am,
    stride_addend_bm,
    stride_ag,
    stride_bg,
    stride_og,
    ACTUAL_M,
    N,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    K: gl.constexpr,
    SPLIT_K: gl.constexpr,
    WARP_BASES: gl.constexpr,
    EVEN_N: gl.constexpr,
    ADD3: gl.constexpr,
    GROUPS: gl.constexpr,
):
    """CDNA5 dense projection with an optional add3 epilogue.

    Program ``(n, split, m)`` owns one BLOCK_M-row, BLOCK_N-column output tile
    over one K partition, so B streams once per row tile. WARP_BASES places the
    warps on 16x16 WMMA tiles; the tiles they leave repeat in registers. Rows
    past ACTUAL_M and, unless EVEN_N, columns past N load as zero and are not
    stored. A single partition stores in ``out_ptr``'s dtype; several store
    FP32 partials for a separate reduction.

    ``GROUPS > 1`` runs that many independent GEMMs in one launch: axis 0 holds
    every group's column tiles, group g reads A, B and writes the output at
    element offsets g * stride_ag/bg/og, and its partials are columns
    [g * N, (g + 1) * N) of ``GROUPS * N``-wide partial rows.
    """
    pid_n = gl.program_id(0)
    pid_split = gl.program_id(1)
    off_m = gl.program_id(2) * BLOCK_M
    if GROUPS > 1:
        n_tiles = gl.cdiv(N, BLOCK_N)
        group = pid_n // n_tiles
        pid_n = pid_n % n_tiles
        a_ptr += group * stride_ag
        b_ptr += group * stride_bg
        out_ptr += group * stride_og
        partial_ptr += group * N

    gl.static_assert(
        BLOCK_M == 16 or BLOCK_M == 32 or BLOCK_M == 64 or BLOCK_M == 128,
        "row tile is 16, 32, 64 or 128 rows",
    )
    gl.static_assert(
        BLOCK_N == 16 or BLOCK_N == 32 or BLOCK_N == 64,
        "column tile is one, two or four WMMA output tiles",
    )
    gl.static_assert(
        BLOCK_K == 128 or BLOCK_K == 256, "candidate uses 128- or 256-wide K tiles"
    )
    gl.static_assert(K % BLOCK_K == 0, "K must tile exactly into BLOCK_K")
    gl.static_assert((K // BLOCK_K) % SPLIT_K == 0, "split-K must divide the K tiles")
    gl.static_assert(not ADD3 or SPLIT_K == 1, "add3 needs the complete K sum")

    wmma_layout: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=WARP_BASES,
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
        [[BLOCK_K, 8]], [BLOCK_M, BLOCK_K], [1, 0]
    )
    shared_layout_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_K, 8]], [BLOCK_N, BLOCK_K], [1, 0]
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
        shape=(ACTUAL_M - off_m, K),
        strides=(stride_am, stride_ak),
        block_shape=(BLOCK_M, BLOCK_K),
        layout=shared_layout_a,
    )
    if EVEN_N:
        b_rows = BLOCK_N
    else:
        b_rows = N - pid_n * BLOCK_N
    b_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * BLOCK_N * stride_bn,
        shape=(b_rows, K),
        strides=(stride_bn, stride_bk),
        block_shape=(BLOCK_N, BLOCK_K),
        layout=shared_layout_b,
    )

    num_k_tiles: gl.constexpr = K // BLOCK_K // SPLIT_K
    k_base = pid_split * num_k_tiles
    # Each K tile is one A and one B TDM load into buffer tile % NUM_BUFFERS;
    # NUM_BUFFERS - 1 tiles stay in flight ahead of the tile being read.
    for tile in gl.static_range(min(NUM_BUFFERS - 1, num_k_tiles)):
        offset = (k_base + tile) * BLOCK_K
        gl.amd.cdna5.tdm.async_load(a_desc, [0, offset], a_smem.index(tile))
        gl.amd.cdna5.tdm.async_load(b_desc, [0, offset], b_smem.index(tile))

    # The loop reads tile t + 1 from LDS while tile t's WMMAs run, and splits
    # each K tile into two halves with separate accumulators, so neither the
    # LDS latency nor one long dependent WMMA chain serializes a tile.
    half_k: gl.constexpr = BLOCK_K // 2
    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, wmma_layout)
    acc_hi = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, wmma_layout)
    gl.amd.cdna5.tdm.async_wait(2 * (min(NUM_BUFFERS - 1, num_k_tiles) - 1))
    a_lo = a_smem.index(0).slice(0, half_k, 1).load(layout=dot_layout_a)
    b_lo = b_smem.index(0).slice(0, half_k, 1).permute([1, 0]).load(layout=dot_layout_b)
    a_hi = a_smem.index(0).slice(half_k, half_k, 1).load(layout=dot_layout_a)
    b_hi = (
        b_smem.index(0)
        .slice(half_k, half_k, 1)
        .permute([1, 0])
        .load(layout=dot_layout_b)
    )
    for tile in gl.static_range(num_k_tiles):
        if tile + NUM_BUFFERS - 1 < num_k_tiles:
            # This buffer held tile - 1, which is already in registers.
            offset = (k_base + tile + NUM_BUFFERS - 1) * BLOCK_K
            buffer = (tile + NUM_BUFFERS - 1) % NUM_BUFFERS
            gl.amd.cdna5.tdm.async_load(a_desc, [0, offset], a_smem.index(buffer))
            gl.amd.cdna5.tdm.async_load(b_desc, [0, offset], b_smem.index(buffer))
        if tile + 1 < num_k_tiles:
            # Wait for tile + 1 only; later tiles' A/B pairs stay in flight.
            gl.amd.cdna5.tdm.async_wait(
                2 * (min(tile + NUM_BUFFERS - 1, num_k_tiles - 1) - tile - 1)
            )
            buffer = (tile + 1) % NUM_BUFFERS
            next_a_lo = (
                a_smem.index(buffer).slice(0, half_k, 1).load(layout=dot_layout_a)
            )
            next_b_lo = (
                b_smem.index(buffer)
                .slice(0, half_k, 1)
                .permute([1, 0])
                .load(layout=dot_layout_b)
            )
            next_a_hi = (
                a_smem.index(buffer).slice(half_k, half_k, 1).load(layout=dot_layout_a)
            )
            next_b_hi = (
                b_smem.index(buffer)
                .slice(half_k, half_k, 1)
                .permute([1, 0])
                .load(layout=dot_layout_b)
            )
        acc = gl.amd.cdna5.wmma(a_lo, b_lo, acc)
        acc_hi = gl.amd.cdna5.wmma(a_hi, b_hi, acc_hi)
        if tile + 1 < num_k_tiles:
            a_lo = next_a_lo
            b_lo = next_b_lo
            a_hi = next_a_hi
            b_hi = next_b_hi
    acc = acc + acc_hi
    gl.amd.cdna5.tdm.async_wait(0)

    tile_m = off_m + gl.arange(0, BLOCK_M, gl.SliceLayout(1, wmma_layout))
    offs_n = gl.arange(0, BLOCK_N, gl.SliceLayout(0, wmma_layout))
    tile_n = pid_n * BLOCK_N + offs_n
    mask = tile_m[:, None] < ACTUAL_M
    if not EVEN_N:
        mask = mask & (tile_n[None, :] < N)
    if SPLIT_K == 1:
        output_offsets = (tile_m[:, None] * stride_om + tile_n[None, :] * stride_on).to(
            gl.int32
        )
        result = acc.to(out_ptr.dtype.element_ty)
        if ADD3:
            # Preserve the materialized BF16 projection boundary used by torch.mm.
            addend_a = gl.amd.cdna5.buffer_load(
                addend_a_ptr,
                (tile_m[:, None] * stride_addend_am + tile_n[None, :]).to(gl.int32),
                mask=mask,
            )
            addend_b = gl.amd.cdna5.buffer_load(
                addend_b_ptr,
                (tile_m[:, None] * stride_addend_bm + tile_n[None, :]).to(gl.int32),
                mask=mask,
            )
            result = (result + addend_a + addend_b).to(gl.bfloat16)
        gl.amd.cdna5.buffer_store(result, out_ptr, output_offsets, mask=mask)
    else:
        partial_offsets = (
            pid_split * split_stride
            + tile_m[:, None] * partial_row_stride
            + tile_n[None, :]
        ).to(gl.int32)
        gl.amd.cdna5.buffer_store(acc, partial_ptr, partial_offsets, mask=mask)


def _gluon_wmma_dense_reduce_gfx1250_launch_metadata(grid, kernel, args):
    """Report the FP32 partial reduction and output store to Proton."""
    rows = grid[0]
    n = args["n"]
    split_k = args["SPLIT_K"]
    values = rows * n
    return {
        "name": kernel.name,
        "flops32": values * (split_k - 1),
        "bytes": values * split_k * args["partial_ptr"].element_size()
        + values * args["out_ptr"].element_size(),
    }


@gluon.jit(launch_metadata=_gluon_wmma_dense_reduce_gfx1250_launch_metadata)
def gluon_wmma_dense_reduce_gfx1250(
    partial_ptr,
    out_ptr,
    split_stride,
    row_stride,
    out_stride,
    n,
    SPLIT_K: gl.constexpr,
    BLOCK: gl.constexpr,
):
    """Sum split-K fp32 partials in split order and store one row tile in
    ``out_ptr``'s dtype."""
    row = gl.program_id(0)
    tile = gl.program_id(1)
    # 4 warps x 32 threads x 4 elements = 512 columns, one 16-byte buffer load.
    gl.static_assert(BLOCK == 512, "vectorized reduction tile is 512 columns")
    layout: gl.constexpr = gl.BlockedLayout([4], [32], [4], [0])
    offs = tile * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n
    acc = gl.zeros([BLOCK], gl.float32, layout)
    base = row * row_stride
    for split in range(0, SPLIT_K):
        acc += gl.amd.cdna5.buffer_load(
            partial_ptr + split * split_stride + base,
            offs.to(gl.int32),
            mask=mask,
            other=0.0,
        )
    gl.amd.cdna5.buffer_store(
        acc.to(out_ptr.dtype.element_ty),
        out_ptr + row * out_stride,
        offs.to(gl.int32),
        mask=mask,
    )


def _gluon_wmma_dense_reduce_fp8_gfx1250_launch_metadata(grid, kernel, args):
    """Report the FP32 partial reduction and the codes and scales it stores."""
    rows = grid[0]
    n = args["n"]
    split_k = args["SPLIT_K"]
    values = rows * n
    return {
        "name": kernel.name,
        "flops32": values * (split_k - 1),
        "bytes": values * split_k * args["partial_ptr"].element_size()
        + values
        + values // 32,
    }


@gluon.jit(launch_metadata=_gluon_wmma_dense_reduce_fp8_gfx1250_launch_metadata)
def gluon_wmma_dense_reduce_fp8_gfx1250(
    partial_ptr,
    codes_ptr,
    scales_ptr,
    split_stride,
    row_stride,
    n,
    SPLIT_K: gl.constexpr,
    BLOCK: gl.constexpr,
):
    """Sum split-K FP32 partials in split order, round to BF16, and store the
    row tile as group-32 E4M3 codes and UE8M0 scales.

    Rounded like ``gluon_quantize_fp8_group32_ue8m0_gfx1250`` reading the BF16
    result, so it matches the unfused reduction and quantization exactly. Each
    thread loads four contiguous partials; eight threads hold one group.
    """
    row = gl.program_id(0)
    tile = gl.program_id(1)
    gl.static_assert(BLOCK == 512, "vectorized reduction tile is 512 columns")
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    groups = tile * (BLOCK // 32) + gl.arange(
        0, BLOCK // 32, layout=gl.SliceLayout(1, layout)
    )
    lanes = gl.arange(0, 32, layout=gl.SliceLayout(0, layout))
    offs = groups[:, None] * 32 + lanes[None, :]
    mask = offs < n
    acc = gl.zeros([BLOCK // 32, 32], gl.float32, layout)
    base = row * row_stride
    for split in range(0, SPLIT_K):
        acc += gl.amd.cdna5.buffer_load(
            partial_ptr + split * split_stride + base,
            offs.to(gl.int32),
            mask=mask,
            other=0.0,
        )
    values = acc.to(gl.bfloat16).to(gl.float32)
    amax = gl.maximum(gl.max(gl.abs(values), axis=1), 1.0e-4)
    bits = (amax * (1.0 / 448.0)).to(gl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(gl.int32)
    scale = (exponent << 23).to(gl.float32, bitcast=True)
    codes = gl.div_rn(values, scale[:, None]).to(gl.float8e4nv)
    gl.amd.cdna5.buffer_store(codes, codes_ptr + row * n, offs.to(gl.int32), mask=mask)
    gl.amd.cdna5.buffer_store(
        exponent.to(gl.uint8),
        scales_ptr + row * (n // 32),
        groups.to(gl.int32),
        mask=groups < n // 32,
    )


def _dense_split_k(
    n: int,
    block_n: int,
    k_tiles: int,
    num_buffers: int,
) -> int:
    """Return the largest split that divides the K tiles and fits on the CUs.

    Each split keeps at least eight K tiles, and at least one full buffer pipeline.
    """

    ctas = triton.cdiv(n, block_n)
    min_tiles = max(num_buffers, 8)
    for split in (8, 4, 2):
        if (
            k_tiles % split == 0
            and k_tiles // split >= min_tiles
            and ctas * split <= _NUM_CUS
        ):
            return split
    return 1


def _dense_num_buffers(
    requested: int,
    k_tiles: int,
    ctas: int,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
) -> int:
    """Return the TDM buffer count: the request, capped by K and by LDS.

    Decode streams B once, so the buffers ahead of the WMMAs set the bytes in
    flight. A grid past 256 CUs keeps two programs resident on a CU, which
    then split its 320 KB; deeper buffers beat a third resident program. The
    padded layout adds 8 elements per K row.
    """
    stage_bytes = (block_m + block_n) * (block_k + 8) * 2
    per_cta = _LDS_BYTES // min(triton.cdiv(ctas, _NUM_CUS), 2)
    return max(2, min(requested, k_tiles, per_cta // stage_bytes))


def _launch_wmma_tdm_dense_tiles(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    block_n: int,
    block_k: int,
    split_k: int | None,
    num_buffers: int,
    block_m: int,
    warp_bases: tuple[tuple[int, int], ...],
    addends: tuple[torch.Tensor, torch.Tensor] | None = None,
    fp8_out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """Launch one dense WMMA grid over every ``block_m``-row tile of ``A``.

    ``warp_bases`` places the warps on 16x16 WMMA tiles. ``addends`` fuses
    ``bf16(A @ B.T) + addends[0] + addends[1]`` and needs one K split.
    The result is stored in ``out``'s dtype.

    A ``[M, G, K]`` ``A`` with ``[G, N, K]`` ``B`` and contiguous
    ``[M, G, N]`` ``out`` runs the G group products in the same grid.
    ``fp8_out`` (codes ``[M, G * N]``, scales ``[M, G * N / 32]``) replaces
    ``out`` with the group-32 FP8 quantization of the BF16 result, written by
    the split-K reduction, so it needs more than one K split.
    """
    grouped = A.ndim == 3
    groups = A.shape[1] if grouped else 1
    m = A.shape[0]
    k_tiles = A.shape[-1] // block_k
    n = B.shape[-2]
    if split_k is None:
        split_k = (
            1
            if addends is not None
            else _dense_split_k(groups * n, block_n, k_tiles, num_buffers)
        )
    if k_tiles % split_k != 0:
        raise ValueError(f"split_k={split_k} must divide K/{block_k}={k_tiles}")
    if split_k > 1 and k_tiles // split_k < 2:
        raise ValueError(
            "each split needs at least two K tiles for a full TDM pipeline"
        )
    if split_k > 1 and addends is not None:
        raise ValueError("the add3 epilogue needs split_k=1")
    if fp8_out is not None and split_k == 1:
        raise ValueError("the FP8 output is quantized by the split-K reduction")
    num_buffers = _dense_num_buffers(
        num_buffers,
        k_tiles // split_k,
        groups * triton.cdiv(n, block_n) * split_k * triton.cdiv(m, block_m),
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
    )
    partial = (
        torch.empty((split_k, m, groups * n), device=A.device, dtype=torch.float32)
        if split_k > 1
        else out
    )
    if out is None:
        # Every program stores a partial; the FP8 reduction writes the result.
        out = partial
    if grouped:
        group_strides = (A.stride(1), B.stride(0), out.stride(-2))
        A, B = A[:, 0], B[0]
        tile_out = out if out is partial else out[:, 0]
    else:
        group_strides = (0, 0, 0)
        tile_out = out
    if addends is None:
        # None compiles the unused add3 operands out of the kernel arguments.
        addend_args = (None, None, None, None)
    else:
        addend_args = (*addends, addends[0].stride(0), addends[1].stride(0))
    _wmma_tdm_dense_kernel[
        (groups * triton.cdiv(n, block_n), split_k, triton.cdiv(m, block_m))
    ](
        A,
        B,
        tile_out,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        tile_out.stride(0),
        tile_out.stride(-1),
        partial,
        partial.stride(0),
        partial.stride(1),
        *addend_args,
        *group_strides,
        ACTUAL_M=m,
        N=n,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        NUM_BUFFERS=num_buffers,
        K=A.shape[1],
        SPLIT_K=split_k,
        WARP_BASES=warp_bases,
        EVEN_N=n % block_n == 0,
        ADD3=addends is not None,
        GROUPS=groups,
        num_warps=1 << len(warp_bases),
        num_stages=1,
        waves_per_eu=1,
    )
    if fp8_out is not None:
        codes, scales = fp8_out
        gluon_wmma_dense_reduce_fp8_gfx1250[
            (m, triton.cdiv(groups * n, _REDUCE_BLOCK_N))
        ](
            partial,
            codes,
            scales,
            partial.stride(0),
            partial.stride(1),
            groups * n,
            SPLIT_K=split_k,
            BLOCK=_REDUCE_BLOCK_N,
            num_warps=4,
            num_stages=1,
        )
    elif split_k > 1:
        _reduce_dense_partials(partial, out.view(m, groups * n) if grouped else out)


def _reduce_dense_partials(partial: torch.Tensor, out: torch.Tensor) -> None:
    """Sum ``[split_k, M, N]`` FP32 partials into ``out``'s dtype."""
    split_k, m, n = partial.shape
    gluon_wmma_dense_reduce_gfx1250[(m, triton.cdiv(n, _REDUCE_BLOCK_N))](
        partial,
        out,
        partial.stride(0),
        partial.stride(1),
        out.stride(0),
        n,
        SPLIT_K=split_k,
        BLOCK=_REDUCE_BLOCK_N,
        num_warps=4,
        num_stages=1,
    )


def _use_dense_m64_tile(m: int) -> bool:
    return _DENSE_M64_MIN_M <= m <= _DENSE_MAX_M


def use_gluon_wmma_dense_gfx1250(m: int, k: int, n: int) -> bool:
    """Return whether CDNA5 dense WMMA accepts this K3 projection shape.

    Up to 16 rows run as one 16-row tile. Past that, two 16-row tiles would
    each re-read all of B, so 17 to 64 rows run as one 64-row tile that
    streams B once; past that tile rocBLAS keeps the shape. Through 8192
    columns the 64-row tile gives each of the 256 CUs one 32-column block;
    wider outputs need a second round of blocks, so rocBLAS keeps those too.
    Preferring this over the M == 1 row-CTA GEMV is the registry's choice,
    not this predicate's.
    """
    if k % 128 != 0 or n % 16 != 0:
        return False
    if _use_dense_m64_tile(m):
        return n <= _DENSE_M64_MAX_N
    return 1 <= m < _DENSE_M64_MIN_M


def _check_projection_rows(m: int) -> None:
    if not 1 <= m <= _DENSE_MAX_M:
        raise ValueError(f"A must contain 1 to {_DENSE_MAX_M} rows, got {m}")


def gluon_wmma_tdm_dense_gfx1250(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    split_k: int | None,
) -> torch.Tensor:
    """Compute a small-M dense BF16 projection on CDNA5.

    Args:
        A: Contiguous BF16 activations shaped ``[M, K]``.
        B: Contiguous BF16 weights shaped ``[N, K]`` on A's device.
        out: Optional BF16 destination with contiguous columns.
        split_k: One of 1, 2, 4, or 8 partitions, or ``None`` for CU-based selection.

    Returns:
        The BF16 ``[M, N]`` projection, using ``out`` when supplied.
    """
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
    if split_k not in (None, 1, 2, 4, 8):
        raise ValueError("split_k must be one of 1, 2, 4, or 8")
    if not use_gluon_wmma_dense_gfx1250(m, k, n):
        raise ValueError(
            f"dense WMMA needs 1 <= M <= {_DENSE_MAX_M}, K % 128 == 0, "
            f"N % 16 == 0 and, from M={_DENSE_M64_MIN_M}, N <= "
            f"{_DENSE_M64_MAX_N}; got M={m}, K={k}, N={n}"
        )

    if out is None:
        out = A.new_empty((m, n))
    elif (
        tuple(out.shape) != (m, n)
        or out.dtype != torch.bfloat16
        or not out.is_cuda
        or out.device != A.device
        or out.stride(-1) != 1
        or out.stride(0) < n
    ):
        raise ValueError(
            f"out must be GPU BF16 ({m}, {n}) with unit row stride colocated with A"
        )

    block_k = 256 if k % 256 == 0 else 128
    if _use_dense_m64_tile(m):
        _launch_wmma_tdm_dense_tiles(
            A,
            B,
            out,
            block_m=64,
            block_n=32,
            block_k=block_k,
            warp_bases=_WARP_BASES_4,
            split_k=split_k,
            num_buffers=_DENSE_BUFFERS,
        )
        return out
    # One warp per 16-column WMMA tile along N.
    block_n = 32 if n % 32 == 0 and n // 32 >= 192 else 16
    _launch_wmma_tdm_dense_tiles(
        A,
        B,
        out,
        block_m=16,
        block_n=block_n,
        block_k=block_k,
        warp_bases=_WARP_BASES_2 if block_n == 32 else _WARP_BASES_1,
        split_k=split_k,
        num_buffers=_DENSE_BUFFERS,
    )
    return out


def gluon_wmma_tdm_mla_qkv_gate_gfx1250(
    A: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    """Run the CDNA5 MLA QKV/gate projection candidate."""
    _check_projection_rows(A.shape[0])
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
    if _use_dense_m64_tile(A.shape[0]):
        _launch_wmma_tdm_dense_tiles(
            A,
            B,
            out,
            block_m=64,
            block_n=32,
            block_k=256,
            warp_bases=_WARP_BASES_4,
            split_k=1,
            num_buffers=_DENSE_BUFFERS,
        )
        return out
    _launch_wmma_tdm_dense_tiles(
        A,
        B,
        out,
        block_m=16,
        block_n=16,
        block_k=256,
        warp_bases=_WARP_BASES_1,
        split_k=None,
        num_buffers=_DENSE_BUFFERS,
    )
    return out


def gluon_wmma_tdm_kda_qkvfab_gfx1250(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the CDNA5 KDA QKVFAB projection candidate."""
    _check_projection_rows(A.shape[0])
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
    if _use_dense_m64_tile(A.shape[0]):
        # 6288 columns leave a 16-column tail in the last 32-column tile.
        _launch_wmma_tdm_dense_tiles(
            A,
            B,
            out,
            block_m=64,
            block_n=32,
            block_k=256,
            warp_bases=_WARP_BASES_4,
            split_k=1,
            num_buffers=_DENSE_BUFFERS,
        )
        return out
    _launch_wmma_tdm_dense_tiles(
        A,
        B,
        out,
        block_m=16,
        block_n=16,
        block_k=256,
        warp_bases=_WARP_BASES_1,
        split_k=None,
        num_buffers=_DENSE_BUFFERS,
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

    gl.static_assert(
        BLOCK_N == 16 or BLOCK_N == 32,
        "candidate supports one or two WMMA output tiles",
    )
    gl.static_assert(
        BLOCK_K == 128 or BLOCK_K == 256, "candidate uses 128- or 256-wide K tiles"
    )
    gl.static_assert(
        K % BLOCK_K == 0 and K // BLOCK_K >= NUM_BUFFERS - 1 and NUM_BUFFERS >= 2,
        "the TDM prologue must fit inside K",
    )

    # One warp per 16-column WMMA tile, laid out along N.
    warp_bases: gl.constexpr = [] if BLOCK_N == 16 else [[0, 1]]
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

    # NUM_BUFFERS - 1 tiles are prefetched before the issue/wait/WMMA loop.
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
        # Keep up to NUM_BUFFERS - 2 A/B batches in flight while consuming the oldest.
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
    block_n, block_k, num_buffers = 32, 256, 6
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
        num_warps=block_n // 16,
        num_stages=1,
        waves_per_eu=1,
    )
    return out


def gluon_wmma_tdm_add3_gfx1250(
    A: torch.Tensor,
    B: torch.Tensor,
    addend_a: torch.Tensor,
    addend_b: torch.Tensor,
) -> torch.Tensor:
    """Run the Kimi-K3 CDNA5 projection-plus-add3 path on up to 64 rows.

    Up to 16 rows run as one 16-row tile and 17 to 64 rows as one 64-row tile.
    The sum keeps the M16 kernel's rounding: the projection is materialized in
    BF16 before each BF16 addition.
    """
    m = A.shape[0] if A.ndim == 2 else 0
    if not 1 <= m <= _DENSE_MAX_M:
        raise ValueError(f"A must hold 1 to {_DENSE_MAX_M} rows, got {tuple(A.shape)}")
    for name, tensor, shape in (
        ("A", A, (m, 3584)),
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
            tuple(tensor.shape) != (m, 7168)
            or tensor.dtype != torch.bfloat16
            or not tensor.is_cuda
            or tensor.device != A.device
            or tensor.stride(1) != 1
        ):
            raise ValueError(
                f"{name} must be GPU BF16 ({m}, 7168) with unit inner stride "
                "colocated with A"
            )

    out = A.new_empty((m, 7168))
    if _use_dense_m64_tile(m):
        block_m, warp_bases = 64, _WARP_BASES_4
    else:
        block_m, warp_bases = 16, _WARP_BASES_2
    _launch_wmma_tdm_dense_tiles(
        A,
        B,
        out,
        block_m=block_m,
        block_n=32,
        block_k=256,
        warp_bases=warp_bases,
        split_k=1,
        num_buffers=_DENSE_BUFFERS,
        addends=(addend_a, addend_b),
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
    WARP_BASES: gl.constexpr,
):
    """Dense BF16 CDNA5 TDM/WMMA GEMM for large K3 projections.

    M and N need not be multiples of the block shape. The TDM descriptors are
    bounded by the M and N passed in, so a trailing partial tile loads only
    the rows that exist, and the output store is masked against the same
    bounds, so whatever fills the rest of the tile cannot reach memory.
    """
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
        warp_bases=WARP_BASES,
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
        [[BLOCK_K, 8]], [BLOCK_M, BLOCK_K], [1, 0]
    )
    shared_layout_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_K, 8]], [BLOCK_N, BLOCK_K], [1, 0]
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
        acc.to(out_ptr.dtype.element_ty),
        out_ptr,
        output_offsets,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _wmma_tdm_dense_largem_warp_pipeline_launch_metadata(grid, kernel, args):
    """Report useful FLOPs and the issued A/B tile reads plus the BF16 store."""
    m, n, k = args["M"], args["N"], args["K"]
    block_m, block_n = 256 * args["CTA_M"], 256 * args["CTA_N"]
    elem = args["a_ptr"].element_size()
    grid_m = triton.cdiv(m, block_m)
    grid_n = triton.cdiv(n, block_n)
    return {
        "name": f"{kernel.name} [M={m}, N={n}, K={k}]",
        "flops16": 2 * m * n * k,
        # Every cluster streams its A row panel and B column panel once; the
        # multicast hands each to all of the cluster's CTAs.
        "bytes": (grid_n * m + grid_m * n) * k * elem + m * n * elem,
    }


@gluon.jit
def _largem_warp_pipeline_xcd_tile(
    pid, grid_m, grid_n, NUM_XCDS: gl.constexpr, GROUP_M: gl.constexpr
):
    # Workgroups are dispatched round-robin over the XCDs. Give each XCD a
    # contiguous run of tile ids, then group GROUP_M row tiles per column
    # sweep, so tiles that share A rows or B columns run on the same L2.
    num_tiles = grid_m * grid_n
    xcd = pid % NUM_XCDS
    tile = (
        xcd * (num_tiles // NUM_XCDS)
        + gl.minimum(xcd, num_tiles % NUM_XCDS)
        + pid // NUM_XCDS
    )
    width = GROUP_M * grid_n
    first_m = (tile // width) * GROUP_M
    group_size = gl.minimum(grid_m - first_m, GROUP_M)
    pid_m = first_m + (tile % width) % group_size
    pid_n = (tile % width) // group_size
    return pid_m, pid_n


@gluon.jit
def _largem_warp_pipeline_consume(
    a_smem,
    b_smem,
    slot,
    refill_slot,
    a_desc,
    b_desc,
    acc,
    DOT_A: gl.constexpr,
    DOT_B: gl.constexpr,
    BLOCK_K: gl.constexpr,
    REFILL: gl.constexpr,
    CLUSTER: gl.constexpr,
):
    # One K tile in two half-K steps. The two waves sharing a SIMD alternate
    # between stage0 (TDM issue + LDS reads) and stage1 (WMMA), so one wave's
    # LDS reads overlap the other wave's matrix work.
    HALF_K: gl.constexpr = BLOCK_K // 2
    gl.amd.cdna5.tdm.async_wait(0)
    with gl.amd.warp_pipeline_stage("stage0"):
        if REFILL:
            # Waves 0-3 (one per SIMD) issue the refill of the other slot.
            gl.amd.cdna5.tdm.async_load(
                a_desc, [0, 0], a_smem.index(refill_slot), warp_used_hint=0b1111
            )
            gl.amd.cdna5.tdm.async_load(
                b_desc, [0, 0], b_smem.index(refill_slot), warp_used_hint=0b1111
            )
        a = a_smem.index(slot).slice(0, HALF_K, 1).load(layout=DOT_A)
        b = b_smem.index(slot).slice(0, HALF_K, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("stage1"):
        acc = gl.amd.cdna5.wmma(a, b, acc)
    with gl.amd.warp_pipeline_stage("stage0"):
        a = a_smem.index(slot).slice(HALF_K, HALF_K, 1).load(layout=DOT_A)
        b = (
            b_smem.index(slot)
            .slice(HALF_K, HALF_K, 1)
            .permute([1, 0])
            .load(layout=DOT_B)
        )
        if REFILL:
            a_desc = gl.amd.cdna5.tdm.update_tensor_descriptor(
                a_desc, add_offsets=[0, BLOCK_K]
            )
            b_desc = gl.amd.cdna5.tdm.update_tensor_descriptor(
                b_desc, add_offsets=[0, BLOCK_K]
            )
        if CLUSTER:
            # Keep the cluster's CTAs within one K tile of each other so
            # each multicast refill finds every destination slot free.
            gl.amd.cdna5.cluster.arrive()
    with gl.amd.warp_pipeline_stage("stage1"):
        acc = gl.amd.cdna5.wmma(a, b, acc)
        if CLUSTER:
            gl.amd.cdna5.cluster.wait()
    return a_desc, b_desc, acc


@gluon.constexpr_function
def _largem_warp_pipeline_layouts(cta_m, cta_n, block_k, num_warps):
    """LDS and WMMA layouts for a ``cta_m x cta_n`` cluster of 256 x 256 tiles.

    The layouts span the whole cluster tile, and their CGA part places each
    CTA's 256 x 256 slice, so one TDM load multicasts an A row panel to the
    CTAs of a cluster row and a B column panel to those of a cluster column.
    """
    tile = 256
    block_m, block_n = cta_m * tile, cta_n * tile
    # CTA-id bits walk M first, then N; the same bases as
    # make_cga_layout([cta_m, cta_n], [cta_m, cta_n], [0, 1]).
    cga = [[1 << i, 0] for i in range(cta_m.bit_length() - 1)] + [
        [0, 1 << i] for i in range(cta_n.bit_length() - 1)
    ]
    padded_a = gl.PaddedSharedLayout.with_identity_for(
        [[block_k, 8]], [block_m, block_k], [1, 0]
    )
    padded_b = gl.PaddedSharedLayout.with_identity_for(
        [[block_k, 8]], [block_n, block_k], [1, 0]
    )
    # Splitting the A rows and B columns across the two LDS partitions lets
    # paired waves read different partitions in the same instruction.
    _, _, local = gl.amd.cdna5.make_partitioned_dot_layouts(
        block_m,
        block_n,
        padded_a,
        padded_b,
        num_warps,
        [16, 16, 32],
        a_transposed=False,
        b_transposed=True,
        slice_m=tile,
        slice_n=tile,
        transposed=True,
    )
    wmma_layout = gl.amd.AMDWMMALayout(
        3, local.transposed, local.warp_bases, local.reg_bases, local.instr_shape, cga
    )
    cga_a = gl.DotOperandLayout(0, wmma_layout, 8).cga_layout
    # B sits in LDS as [N, K], so its CGA bases swap the operand's (K, N).
    cga_b = [[x[1], x[0]] for x in gl.DotOperandLayout(1, wmma_layout, 8).cga_layout]
    shared_a, shared_b, _ = gl.amd.cdna5.make_partitioned_dot_layouts(
        tile,
        tile,
        gl.PaddedSharedLayout.with_identity_for(
            [[block_k, 8]], [block_m, block_k], [1, 0], cga_a
        ),
        gl.PaddedSharedLayout.with_identity_for(
            [[block_k, 8]], [block_n, block_k], [1, 0], cga_b
        ),
        num_warps,
        [16, 16, 32],
        a_transposed=False,
        b_transposed=True,
        slice_m=tile,
        slice_n=tile,
        transposed=True,
    )
    shared_out = gl.PaddedSharedLayout.with_identity_for(
        [[tile, 8]], [block_m, block_n], [1, 0], cga
    )
    return shared_a, shared_b, shared_out, wmma_layout


# The live row count varies with the prefill chunk, so it must not split the
# compile cache on its divisibility.
@gluon.jit(
    launch_metadata=_wmma_tdm_dense_largem_warp_pipeline_launch_metadata,
    do_not_specialize=["M"],
)
def _wmma_tdm_dense_largem_warp_pipeline_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    stride_am,
    stride_bn,
    stride_om,
    M,
    N,
    K,
    CTA_M: gl.constexpr,
    CTA_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GROUP_M: gl.constexpr,
    NUM_XCDS: gl.constexpr,
):
    """Eight-wave warp pipeline BF16 CDNA5 GEMM for wide large-M projections.

    Computes ``out = A @ B.T`` for unit-inner-stride A ``[M, K]``, B ``[N, K]``
    and out ``[M, N]``; K must be a multiple of BLOCK_K. Each program is a
    ``CTA_M x CTA_N`` cluster of CTAs, one 256 x 256 output tile each, that
    share A and B panels through TDM multicast. Ragged M and N edges are
    handled by bounding the TDM descriptors at the real M and N, which
    zero-fills the loads past the edge and clips the TDM store; CTAs wholly
    past the edge still run the loop, so every cluster barrier completes.
    """
    gl.static_assert(gl.num_warps() == 8, "the warp pipeline needs two waves per SIMD")
    gl.static_assert(gl.num_ctas() == CTA_M * CTA_N, "one CTA per cluster tile")
    BLOCK_M: gl.constexpr = CTA_M * 256
    BLOCK_N: gl.constexpr = CTA_N * 256
    CLUSTER: gl.constexpr = CTA_M * CTA_N > 1
    # Built here rather than passed in: layout objects are not valid launch
    # arguments for the JIT hooks (compile monitor) that serialize them.
    layouts: gl.constexpr = _largem_warp_pipeline_layouts(CTA_M, CTA_N, BLOCK_K, 8)
    SHARED_A: gl.constexpr = layouts[0]
    SHARED_B: gl.constexpr = layouts[1]
    SHARED_OUT: gl.constexpr = layouts[2]
    WMMA_LAYOUT: gl.constexpr = layouts[3]
    pid_m, pid_n = _largem_warp_pipeline_xcd_tile(
        gl.program_id(0),
        gl.cdiv(M, BLOCK_M),
        gl.cdiv(N, BLOCK_N),
        NUM_XCDS,
        GROUP_M,
    )
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 8)

    a_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=a_ptr + off_m * stride_am,
        shape=(M - off_m, K),
        strides=(stride_am, 1),
        block_shape=(BLOCK_M, BLOCK_K),
        layout=SHARED_A,
    )
    b_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=b_ptr + off_n * stride_bn,
        shape=(N - off_n, K),
        strides=(stride_bn, 1),
        block_shape=(BLOCK_N, BLOCK_K),
        layout=SHARED_B,
    )
    a_smem = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [2, BLOCK_M, BLOCK_K], SHARED_A
    )
    b_smem = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [2, BLOCK_N, BLOCK_K], SHARED_B
    )

    gl.amd.cdna5.tdm.async_load(a_desc, [0, 0], a_smem.index(0), warp_used_hint=0b1111)
    gl.amd.cdna5.tdm.async_load(b_desc, [0, 0], b_smem.index(0), warp_used_hint=0b1111)
    a_desc = gl.amd.cdna5.tdm.update_tensor_descriptor(a_desc, add_offsets=[0, BLOCK_K])
    b_desc = gl.amd.cdna5.tdm.update_tensor_descriptor(b_desc, add_offsets=[0, BLOCK_K])
    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, WMMA_LAYOUT)
    num_k_tiles = gl.cdiv(K, BLOCK_K)
    gl.assume(num_k_tiles > 0)
    # Double buffer: consuming K tile i refills the other slot with tile i + 1.
    for i in range(0, num_k_tiles - 1):
        a_desc, b_desc, acc = _largem_warp_pipeline_consume(
            a_smem,
            b_smem,
            i % 2,
            (i + 1) % 2,
            a_desc,
            b_desc,
            acc,
            dot_a,
            dot_b,
            BLOCK_K,
            True,
            CLUSTER,
        )
    a_desc, b_desc, acc = _largem_warp_pipeline_consume(
        a_smem,
        b_smem,
        (num_k_tiles - 1) % 2,
        0,
        a_desc,
        b_desc,
        acc,
        dot_a,
        dot_b,
        BLOCK_K,
        False,
        CLUSTER,
    )

    # The operand buffers are dead here, so the output staging tile reuses
    # their LDS; one TDM store then writes the clipped tile.
    out_smem = gl.allocate_shared_memory(
        out_ptr.dtype.element_ty, [BLOCK_M, BLOCK_N], SHARED_OUT
    )
    out_smem.store(acc.to(out_ptr.dtype.element_ty))
    out_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=out_ptr + off_m * stride_om + off_n,
        shape=(M - off_m, N - off_n),
        strides=(stride_om, 1),
        block_shape=(BLOCK_M, BLOCK_N),
        layout=SHARED_OUT,
    )
    gl.amd.cdna5.tdm.async_store(out_desc, [0, 0], out_smem)
    gl.amd.cdna5.tdm.async_wait(0)


def gluon_mm_a16w16_largem_gfx1250(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run K3 BF16 projections on one CDNA5 WMMA kernel launch.

    From ``_LARGEM_WIDE_M`` rows this is the eight-wave warp pipeline kernel with
    256 x 256 tiles; fewer rows use the four-warp 128 x 128 kernel. Either
    covers the whole output, including ragged M and N edges, rather than
    having the caller peel them off into vendor calls.
    """

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
    _launch_largem(A, B, out)
    return out


def _launch_largem(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    """Store ``A @ B.T`` in ``out``'s dtype with the large-M kernels."""
    m, k = A.shape
    n = B.shape[0]
    # The warp pipeline kernel writes its tile with one TDM store, so it needs a
    # contiguous output; narrower outputs leave its 256-wide grid underfilled.
    if m >= _LARGEM_WIDE_M and n >= _LARGEM_WARP_PIPELINE_MIN_N and out.is_contiguous():
        cta_m, cta_n = _LARGEM_WARP_PIPELINE_CLUSTER
        grid = triton.cdiv(m, 256 * cta_m) * triton.cdiv(n, 256 * cta_n)
        _wmma_tdm_dense_largem_warp_pipeline_kernel[(grid,)](
            A,
            B,
            out,
            A.stride(0),
            B.stride(0),
            out.stride(0),
            m,
            n,
            k,
            CTA_M=cta_m,
            CTA_N=cta_n,
            BLOCK_K=_LARGEM_BLOCK_K,
            GROUP_M=_LARGEM_WARP_PIPELINE_GROUP_M,
            NUM_XCDS=_LARGEM_NUM_XCDS,
            num_warps=8,
            waves_per_eu=2,
            num_ctas=cta_m * cta_n,
        )
        return

    if m >= _LARGEM_WIDE_M and n >= _LARGEM_WIDE_N:
        block_m, block_n = 256, 256
        warp_bases, num_warps = _WARP_BASES_8, 8
        # A 256-wide tile already spans enough columns that grouping rows
        # for L2 reuse costs more than it returns.
        group_m = 1
    else:
        block_m = 128 if m < _LARGEM_BLOCK_M_CROSSOVER else 256
        # Only the block shape has to divide evenly here. Ragged M and N edges
        # are handled inside the kernel: its TDM descriptors are bounded by the
        # real M and N, so the trailing partial tiles load only the rows that
        # exist, and its stores are masked against the same bounds.
        block_n = block_m if n % block_m == 0 else 128
        warp_bases, num_warps = _WARP_BASES_4, 4
        group_m = 8
    grid = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
    _wmma_tdm_dense_largem_kernel[(grid,)](
        A,
        B,
        out,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        out.stride(0),
        out.stride(1),
        m,
        n,
        k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=128,
        GROUP_M=group_m,
        NUM_BUFFERS=2,
        WARP_BASES=warp_bases,
        num_warps=num_warps,
        num_stages=1,
    )


def _select_split_k(ctas: int, k_tiles: int, candidates: tuple[int, ...]) -> int:
    """Return the first candidate that divides the K tiles, leaves at least
    two per split, and keeps the grid within one wave on the CUs."""
    for split in candidates:
        if k_tiles % split == 0 and k_tiles // split >= 2 and ctas * split <= _NUM_CUS:
            return split
    return 1


def _mm_a16w16_tiles(
    m: int, n: int, k: int
) -> tuple[int, int, int, int, tuple[tuple[int, int], ...], int] | None:
    """Return ``(block_m, block_n, block_k, num_buffers, warp_bases, split_k)``
    for the dense kernel, or None once 128x64 tiles alone fill the CUs."""
    if m <= 32:
        block_k = 256 if k % 256 == 0 else 128
        if m <= 16:
            return 16, 16, block_k, 6, _WARP_BASES_1, 1
        return 32, 16, block_k, 6, _WARP_BASES_M2, 1
    if m <= 128:
        block_m = 64 if m <= 64 else 128
        block_k = 256 if block_m == 64 and k % 256 == 0 else 128
        m_tiles = triton.cdiv(m, block_m)
        k_tiles = k // block_k
        split_k = _select_split_k(m_tiles * triton.cdiv(n, 16), k_tiles, _MID_M_SPLIT_K)
        if split_k == _MID_M_SPLIT_K[0]:
            warp_bases = _WARP_BASES_M4 if block_m == 64 else _WARP_BASES_M8
            return block_m, 16, block_k, 4, warp_bases, split_k
        split_k = _select_split_k(m_tiles * triton.cdiv(n, 32), k_tiles, _MID_M_SPLIT_K)
        return block_m, 32, block_k, 4, _WARP_BASES_4, split_k
    ctas = triton.cdiv(m, 128) * triton.cdiv(n, 64)
    if ctas > _NUM_CUS:
        return None
    k_tiles = k // 128
    split_k = 2 if k_tiles % 2 == 0 and k_tiles >= 16 and 2 * ctas <= _NUM_CUS else 1
    return 128, 64, 128, 4, _WARP_BASES_8, split_k


def use_gluon_mm_a16w16_gfx1250(m: int, k: int, n: int) -> bool:
    """Return whether the general CDNA5 BF16 GEMM accepts this shape."""
    return m >= 1 and n % 16 == 0 and k % 128 == 0


def gluon_mm_a16w16_gfx1250(
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: torch.dtype,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``A @ B.T`` from BF16 operands with FP32 accumulation on CDNA5.

    Batches the dense kernel can tile within one wave of the CUs run there,
    split along K while the tiles leave CUs idle; larger ones run on the
    large-M kernel.

    Args:
        A: BF16 activations shaped ``[M, K]`` with unit column stride.
        B: Contiguous BF16 weights shaped ``[N, K]`` on A's device.
        out_dtype: FP32 or BF16 result dtype.
        out: Optional ``[M, N]`` destination of ``out_dtype`` with unit column
            stride.

    Returns:
        The ``[M, N]`` projection, using ``out`` when supplied.
    """
    if out_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"out_dtype must be FP32 or BF16, got {out_dtype}")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D")
    m, k = A.shape
    n = B.shape[0]
    if B.shape[1] != k:
        raise ValueError(f"B must be [N, {k}], got {tuple(B.shape)}")
    if (
        A.dtype != torch.bfloat16
        or B.dtype != torch.bfloat16
        or not A.is_cuda
        or B.device != A.device
        or A.stride(1) != 1
        or not B.is_contiguous()
    ):
        raise ValueError(
            "A must be GPU BF16 with unit column stride and B contiguous GPU "
            "BF16 colocated with A"
        )
    if not use_gluon_mm_a16w16_gfx1250(m, k, n):
        raise ValueError(
            f"BF16 GEMM needs M >= 1, N % 16 == 0 and K % 128 == 0, "
            f"got M={m}, N={n}, K={k}"
        )
    if out is None:
        out = torch.empty((m, n), device=A.device, dtype=out_dtype)
    elif (
        tuple(out.shape) != (m, n)
        or out.dtype != out_dtype
        or out.device != A.device
        or out.stride(1) != 1
    ):
        raise ValueError(
            f"out must be GPU {out_dtype} ({m}, {n}) with unit column stride"
        )

    tiles = _mm_a16w16_tiles(m, n, k)
    if tiles is None:
        _launch_largem(A, B, out)
        return out
    block_m, block_n, block_k, num_buffers, warp_bases, split_k = tiles
    _launch_wmma_tdm_dense_tiles(
        A,
        B,
        out,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_buffers=num_buffers,
        warp_bases=warp_bases,
        split_k=split_k,
    )
    return out


def _mm_a16w16_fp8_tiles(
    m: int, n: int, k: int
) -> tuple[int, int, int, int, tuple[tuple[int, int], ...], int] | None:
    """Return the dense tiles for a quantized ``[M, N]`` result, or None.

    The split-K reduction writes the codes, so up to 32 rows trade the
    unsplit tiles for 32-column tiles split K in two.
    """
    if m <= 32:
        block_k = 256 if k % 256 == 0 else 128
        if (k // block_k) % 2 or k // block_k < 4:
            return None
        if m <= 16:
            return 16, 32, block_k, 6, _WARP_BASES_2, 2
        return 32, 32, block_k, 6, _WARP_BASES_4, 2
    tiles = _mm_a16w16_tiles(m, n, k)
    return tiles if tiles is not None and tiles[5] > 1 else None


def supports_gluon_grouped_mm_a16w16_fp8_gfx1250(
    m: int, g: int, n: int, k: int
) -> bool:
    """Return whether the grouped GEMM can quantize its output here: the
    dense tiles split K, so the reduction writes the codes."""
    if not use_gluon_mm_a16w16_gfx1250(m, k, n) or n % 32:
        return False
    return _mm_a16w16_fp8_tiles(m, g * n, k) is not None


def gluon_grouped_mm_a16w16_gfx1250(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    out: torch.Tensor | None,
    quantize: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute ``out[:, g] = x[:, g] @ weight[g].T`` for every group in one grid.

    Batches the dense tiles cover run every group in one launch (plus the
    split-K reduction); larger ones run ``gluon_mm_a16w16_gfx1250`` per group.

    Args:
        x: BF16 ``[M, G, K]`` with unit inner stride; token and group strides
            are free (e.g. a slice of padded attention heads).
        weight: Contiguous BF16 ``[G, N, K]`` on x's device.
        out: Optional contiguous BF16 ``[M, G, N]`` destination; None with
            ``quantize``.
        quantize: Return instead the group-32 E4M3 codes ``[M, G * N]`` and
            uint8 UE8M0 scales ``[M, G * N / 32]`` of the BF16 result, which
            needs ``supports_gluon_grouped_mm_a16w16_fp8_gfx1250``.

    Returns:
        BF16 ``[M, G, N]``, or (codes, scales) with ``quantize``.
    """
    if x.ndim != 3 or weight.ndim != 3:
        raise ValueError("grouped GEMM expects [M, G, K] and [G, N, K]")
    m, g, k = x.shape
    n = weight.shape[1]
    if weight.shape[0] != g or weight.shape[2] != k:
        raise ValueError("grouped GEMM group/K mismatch")
    if (
        x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_cuda
        or weight.device != x.device
        or x.stride(2) != 1
        or not weight.is_contiguous()
    ):
        raise ValueError(
            "x must be GPU BF16 with unit inner stride and weight contiguous "
            "GPU BF16 colocated with x"
        )
    if not use_gluon_mm_a16w16_gfx1250(m, k, n):
        raise ValueError(
            f"grouped GEMM needs M >= 1, N % 16 == 0 and K % 128 == 0, "
            f"got M={m}, N={n}, K={k}"
        )
    if quantize:
        if out is not None:
            raise ValueError("the quantized grouped GEMM allocates its output")
        if not supports_gluon_grouped_mm_a16w16_fp8_gfx1250(m, g, n, k):
            raise ValueError(f"no split-K grouped tiles for M={m} G={g} N={n} K={k}")
        tiles = _mm_a16w16_fp8_tiles(m, g * n, k)
        fp8_out = (
            torch.empty((m, g * n), dtype=torch.float8_e4m3fn, device=x.device),
            torch.empty((m, g * n // 32), dtype=torch.uint8, device=x.device),
        )
    else:
        tiles = _mm_a16w16_tiles(m, g * n, k)
        fp8_out = None
        if out is None:
            out = torch.empty((m, g, n), dtype=x.dtype, device=x.device)
        elif (
            tuple(out.shape) != (m, g, n)
            or out.dtype != x.dtype
            or out.device != x.device
            or not out.is_contiguous()
        ):
            raise ValueError(f"out must be contiguous GPU BF16 ({m}, {g}, {n})")
        if tiles is None:
            for group in range(g):
                gluon_mm_a16w16_gfx1250(
                    x[:, group], weight[group], x.dtype, out=out[:, group]
                )
            return out
    block_m, block_n, block_k, num_buffers, warp_bases, split_k = tiles
    _launch_wmma_tdm_dense_tiles(
        x,
        weight,
        out,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_buffers=num_buffers,
        warp_bases=warp_bases,
        split_k=split_k,
        fp8_out=fp8_out,
    )
    return out if fp8_out is None else fp8_out


__all__ = [
    "use_gluon_largem_gfx1250",
    "use_gluon_mm_a16w16_gfx1250",
    "supports_gluon_grouped_mm_a16w16_fp8_gfx1250",
    "gluon_grouped_mm_a16w16_gfx1250",
    "use_gluon_wmma_dense_gfx1250",
    "gluon_mm_a16w16_gfx1250",
    "gluon_mm_a16w16_largem_gfx1250",
    "gluon_wmma_tdm_dense_gfx1250",
    "gluon_wmma_tdm_kda_qkvfab_gfx1250",
    "gluon_wmma_tdm_mla_qkv_gate_gfx1250",
    "gluon_wmma_tdm_add3_m16_gfx1250",
    "triton_mm_a16w16_add3_m16_gfx1250",
]
