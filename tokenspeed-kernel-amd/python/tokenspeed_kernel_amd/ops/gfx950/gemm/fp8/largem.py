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

"""Large-row block-scaled FP8 GEMM for GLM-5.3 on gfx950.

The checkpoint remains canonical ``[N, K]``.  Model-load preparation creates a
non-persistent K128-major copy while retaining the logical two-dimensional
shape. The kernel uses N64 physical tiles. For each K128 quantization
block it combines two native K64 FP8 dot instructions into one FP32
partial, then applies that block's activation and weight scales before adding
the result to the FP32 accumulator.

The path is opt-in and limited to the 8144/8192-row prefill contract.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, gluon_builtin, tl, triton

GLUON_BLOCK_FP8_WEIGHT_LAYOUT = "gluon_gfx950_k128_n64"
GLM53_BLOCK_FP8_PRIMARY_ROWS = frozenset({8144, 8192})
GLM53_BLOCK_FP8_PROJECTION_SHAPES = frozenset(
    {
        (1024, 4096),
        (4096, 512),
        (6144, 4096),
        (4096, 3072),
        (2048, 4096),
        (4096, 1536),
        (4096, 4096),
    }
)
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 128
NUM_WARPS = 4
WARPS_M = 2
WARPS_N = 2
GROUP_SIZE_M = 8
_FP8_DTYPES = (torch.float8_e4m3fn,)


def _block_fp8_launch_metadata(grid, kernel, args):
    """Report logical FP8 work and memory traffic to Proton."""
    m, n, k = args["M"], args["N"], args["K"]
    scale_bytes = 4 * (m + n // 128) * (k // 128)
    return {
        "name": kernel.name,
        "flops8": 2 * m * n * k,
        "bytes": m * k + n * k + scale_bytes + m * n * args["c_ptr"].element_size(),
    }


@gluon_builtin
def _mfma_unscaled_fp8(a, b, acc, *, _semantic):
    """Emit gfx950's native unscaled FP8 MFMA without synthetic scales."""
    fp8_format = "e4m3" if a.dtype == gl.float8e4nv else "e5m2"
    output = _semantic.dot_scaled(
        a,
        None,
        fp8_format,
        b,
        None,
        fp8_format,
        acc,
        fast_math=False,
        lhs_k_pack=True,
        rhs_k_pack=True,
        out_dtype=gl.float32,
    )
    return gl.tensor(output.handle, acc.type)


@gluon.constexpr_function
def _padded_shared_layout(operand_layout, shape, dtype, is_k_contig):
    """Retain CDNA4's bank-conflict padding with an async-copy-safe identity."""
    efficient = gl.amd.cdna4.compute_efficient_padded_shared_layout(
        operand_layout,
        shape,
        dtype,
        is_k_contig=is_k_contig,
    )
    assert efficient is not None
    padding = list(efficient.interval_padding_pairs)
    assert len(padding) == 1
    return gl.PaddedSharedLayout.with_identity_for(
        [[int(padding[0][0]), int(padding[0][1])]],
        shape,
        [1, 0],
    )


@gluon.jit(launch_metadata=_block_fp8_launch_metadata)
def gluon_mm_fp8_blockscale_largem_gfx950(
    a_ptr,
    packed_b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    c_ptr,
    M,
    N,
    K: gl.constexpr,
    stride_am,
    stride_ak,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
):
    """Double-buffered K128 FP8 MFMA with FP32 block-scale accumulation."""
    tile_id = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    tiles_per_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // tiles_per_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((tile_id % tiles_per_group) % group_m)
    pid_n = (tile_id % tiles_per_group) // group_m

    gl.static_assert(BLOCK_M == 64, "candidate requires a 64-row tile")
    gl.static_assert(BLOCK_N == 64, "candidate requires a 64-column tile")
    gl.static_assert(BLOCK_K == 128, "one MFMA partial must equal one scale block")
    gl.static_assert(
        WARPS_M == 2 and WARPS_N == 2,
        "candidate requires a 2x2 wave arrangement",
    )
    k_tiles: gl.constexpr = K // BLOCK_K
    gl.static_assert(K % BLOCK_K == 0, "K must contain complete K128 scale blocks")
    gl.static_assert(k_tiles >= 4, "candidate requires at least four K128 blocks")
    gl.static_assert(k_tiles % 2 == 0, "double-buffered loop requires paired K tiles")

    # Each of four waves moves one 32-byte slice per K128 tile.  Explicit
    # linear layouts keep those DMA copies affine and make the packed B tile's
    # N64 dimension the contiguous 16-byte vector.
    load_a_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [32, 0]],
        lane_bases=[
            [0, 16],
            [0, 32],
            [0, 64],
            [1, 0],
            [2, 0],
            [4, 0],
        ],
        warp_bases=[[8, 0], [16, 0]],
        block_bases=[],
        shape=[BLOCK_M, BLOCK_K],
    )
    load_b_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [64, 0]],
        lane_bases=[
            [0, 16],
            [0, 32],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
        ],
        warp_bases=[[16, 0], [32, 0]],
        block_bases=[],
        shape=[BLOCK_K, 64],
    )
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, 64],
        transposed=True,
        warps_per_cta=[WARPS_M, WARPS_N],
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=mfma_layout,
        k_width=16,
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1,
        parent=mfma_layout,
        k_width=16,
    )
    shared_a_layout: gl.constexpr = _padded_shared_layout(
        dot_a_layout,
        [BLOCK_M, BLOCK_K],
        a_ptr.dtype.element_ty,
        True,
    )
    shared_a = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty,
        [2, BLOCK_M, BLOCK_K],
        shared_a_layout,
    )
    shared_b_layout: gl.constexpr = _padded_shared_layout(
        dot_b_layout,
        [BLOCK_K, BLOCK_N],
        packed_b_ptr.dtype.element_ty,
        False,
    )
    shared_b = gl.allocate_shared_memory(
        packed_b_ptr.dtype.element_ty,
        [2, BLOCK_K, BLOCK_N],
        shared_b_layout,
    )
    load_m = gl.arange(0, BLOCK_M, gl.SliceLayout(1, load_a_layout))
    load_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, load_a_layout))
    load_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, load_b_layout))
    load_n = gl.arange(0, 64, gl.SliceLayout(0, load_b_layout))
    global_load_m = pid_m * BLOCK_M + load_m
    a_mask = global_load_m[:, None] < M
    a_offsets_0 = (
        global_load_m[:, None] * stride_am + load_ak[None, :] * stride_ak
    ).to(gl.int32)
    a_offsets_1 = (a_offsets_0 + BLOCK_K * stride_ak).to(gl.int32)

    # Packed order is [K/128, N/64, 128, 64].  Keeping the tensor's logical
    # [N,K] shape makes checkpoint and model metadata independent of this copy.
    packed_tile_base = pid_n * BLOCK_K * BLOCK_N
    b_offsets_0 = (packed_tile_base + load_bk[:, None] * BLOCK_N + load_n[None, :]).to(
        gl.int32
    )
    b_offsets_1 = (b_offsets_0 + BLOCK_K * N).to(gl.int32)
    a_pair_step = 2 * BLOCK_K * stride_ak
    b_pair_step = 2 * BLOCK_K * N

    output_m = gl.arange(0, BLOCK_M, gl.SliceLayout(1, mfma_layout))
    output_n = gl.arange(0, 64, gl.SliceLayout(0, mfma_layout))
    global_output_m = pid_m * BLOCK_M + output_m
    global_output_n = pid_n * BLOCK_N + output_n
    output_mask = (global_output_m[:, None] < M) & (global_output_n[None, :] < N)
    async_copy: gl.constexpr = gl.amd.cdna4.async_copy
    shared_a.index(0).store(gl.load(a_ptr + a_offsets_0, mask=a_mask, other=0))
    async_copy.buffer_load_to_shared(shared_b.index(0), packed_b_ptr, b_offsets_0)
    async_copy.commit_group()
    shared_a.index(1).store(gl.load(a_ptr + a_offsets_1, mask=a_mask, other=0))
    async_copy.buffer_load_to_shared(shared_b.index(1), packed_b_ptr, b_offsets_1)
    async_copy.commit_group()
    a_offsets_0 = (a_offsets_0 + a_pair_step).to(gl.int32)
    a_offsets_1 = (a_offsets_1 + a_pair_step).to(gl.int32)
    b_offsets_0 = (b_offsets_0 + b_pair_step).to(gl.int32)
    b_offsets_1 = (b_offsets_1 + b_pair_step).to(gl.int32)

    gl.barrier()
    async_copy.wait_group(0)
    activation = shared_a.index(0).load(dot_a_layout).to(gl.float8e4nv, bitcast=True)
    weight = async_copy.load_shared_relaxed(shared_b.index(0), dot_b_layout).to(
        gl.float8e4nv, bitcast=True
    )
    accumulator = gl.zeros((BLOCK_M, 64), gl.float32, mfma_layout)
    main_loop_pairs: gl.constexpr = (k_tiles - 2) // 2
    for pair in tl.range(0, main_loop_pairs):
        even_k_tile = pair * 2
        partial = gl.zeros((BLOCK_M, 64), gl.float32, mfma_layout)
        partial = _mfma_unscaled_fp8(activation, weight, partial)
        activation_scale = gl.amd.cdna4.buffer_load(
            ptr=a_scale_ptr,
            offsets=(global_output_m * stride_asm + even_k_tile * stride_ask).to(
                gl.int32
            ),
            mask=global_output_m < M,
            other=0.0,
        ).to(gl.float32)
        weight_scale = gl.amd.cdna4.buffer_load(
            ptr=b_scale_ptr,
            offsets=(
                (global_output_n // 128) * stride_bsn + even_k_tile * stride_bsk
            ).to(gl.int32),
        ).to(gl.float32)
        accumulator = gl.fma(
            weight_scale[None, :],
            partial * activation_scale[:, None],
            accumulator,
        )
        async_copy.wait_group(0)
        activation = (
            shared_a.index(1).load(dot_a_layout).to(gl.float8e4nv, bitcast=True)
        )
        weight = async_copy.load_shared_relaxed(shared_b.index(1), dot_b_layout).to(
            gl.float8e4nv, bitcast=True
        )
        gl.barrier()
        shared_a.index(0).store(gl.load(a_ptr + a_offsets_0, mask=a_mask, other=0))
        async_copy.buffer_load_to_shared(shared_b.index(0), packed_b_ptr, b_offsets_0)
        async_copy.commit_group()

        odd_k_tile = even_k_tile + 1
        partial = gl.zeros((BLOCK_M, 64), gl.float32, mfma_layout)
        partial = _mfma_unscaled_fp8(activation, weight, partial)
        activation_scale = gl.amd.cdna4.buffer_load(
            ptr=a_scale_ptr,
            offsets=(global_output_m * stride_asm + odd_k_tile * stride_ask).to(
                gl.int32
            ),
            mask=global_output_m < M,
            other=0.0,
        ).to(gl.float32)
        weight_scale = gl.amd.cdna4.buffer_load(
            ptr=b_scale_ptr,
            offsets=(
                (global_output_n // 128) * stride_bsn + odd_k_tile * stride_bsk
            ).to(gl.int32),
        ).to(gl.float32)
        accumulator = gl.fma(
            weight_scale[None, :],
            partial * activation_scale[:, None],
            accumulator,
        )
        async_copy.wait_group(0)
        activation = (
            shared_a.index(0).load(dot_a_layout).to(gl.float8e4nv, bitcast=True)
        )
        weight = async_copy.load_shared_relaxed(shared_b.index(0), dot_b_layout).to(
            gl.float8e4nv, bitcast=True
        )
        gl.barrier()
        shared_a.index(1).store(gl.load(a_ptr + a_offsets_1, mask=a_mask, other=0))
        async_copy.buffer_load_to_shared(shared_b.index(1), packed_b_ptr, b_offsets_1)
        async_copy.commit_group()
        a_offsets_0 = (a_offsets_0 + a_pair_step).to(gl.int32)
        a_offsets_1 = (a_offsets_1 + a_pair_step).to(gl.int32)
        b_offsets_0 = (b_offsets_0 + b_pair_step).to(gl.int32)
        b_offsets_1 = (b_offsets_1 + b_pair_step).to(gl.int32)

    penultimate_k_tile: gl.constexpr = main_loop_pairs * 2
    partial = gl.zeros((BLOCK_M, 64), gl.float32, mfma_layout)
    partial = _mfma_unscaled_fp8(activation, weight, partial)
    activation_scale = gl.amd.cdna4.buffer_load(
        ptr=a_scale_ptr,
        offsets=(global_output_m * stride_asm + penultimate_k_tile * stride_ask).to(
            gl.int32
        ),
        mask=global_output_m < M,
        other=0.0,
    ).to(gl.float32)
    weight_scale = gl.amd.cdna4.buffer_load(
        ptr=b_scale_ptr,
        offsets=(
            (global_output_n // 128) * stride_bsn + penultimate_k_tile * stride_bsk
        ).to(gl.int32),
    ).to(gl.float32)
    accumulator = gl.fma(
        weight_scale[None, :],
        partial * activation_scale[:, None],
        accumulator,
    )
    async_copy.wait_group(0)
    activation = shared_a.index(1).load(dot_a_layout).to(gl.float8e4nv, bitcast=True)
    weight = async_copy.load_shared_relaxed(shared_b.index(1), dot_b_layout).to(
        gl.float8e4nv, bitcast=True
    )
    final_k_tile: gl.constexpr = penultimate_k_tile + 1
    partial = gl.zeros((BLOCK_M, 64), gl.float32, mfma_layout)
    partial = _mfma_unscaled_fp8(activation, weight, partial)
    activation_scale = gl.amd.cdna4.buffer_load(
        ptr=a_scale_ptr,
        offsets=(global_output_m * stride_asm + final_k_tile * stride_ask).to(gl.int32),
        mask=global_output_m < M,
        other=0.0,
    ).to(gl.float32)
    weight_scale = gl.amd.cdna4.buffer_load(
        ptr=b_scale_ptr,
        offsets=((global_output_n // 128) * stride_bsn + final_k_tile * stride_bsk).to(
            gl.int32
        ),
    ).to(gl.float32)
    accumulator = gl.fma(
        weight_scale[None, :],
        partial * activation_scale[:, None],
        accumulator,
    )
    c_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c_offsets = output_m[:, None] * stride_cm + output_n[None, :] * stride_cn
    gl.amd.cdna4.buffer_store(
        ptr=c_base,
        offsets=c_offsets.to(gl.int32),
        stored_value=accumulator.to(c_ptr.dtype.element_ty),
        mask=output_mask,
    )


def supports_gluon_fp8_blockscale_largem(m: int, n: int, k: int) -> bool:
    """Return whether the experimental kernel owns the audited exact contract."""
    return (
        m in GLM53_BLOCK_FP8_PRIMARY_ROWS
        and (n, k) in GLM53_BLOCK_FP8_PROJECTION_SHAPES
    )


def pack_gluon_fp8_blockscale_weight(
    weight: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Create a private K128-major candidate weight copy.

    The returned tensor deliberately retains the checkpoint's logical ``[N,K]``
    shape.  The caller must pass the matching private layout marker when the
    packed tensor is consumed.
    """
    if weight.ndim != 2:
        raise ValueError(f"block-FP8 weight must have shape [N,K], got {weight.shape}")
    if weight.dtype not in _FP8_DTYPES:
        raise TypeError(f"block-FP8 weight must use float8_e4m3fn, got {weight.dtype}")
    n, k = weight.shape
    if n % BLOCK_N != 0 or k % BLOCK_K != 0:
        raise ValueError(
            f"packed block-FP8 weight requires N%{BLOCK_N}=0 and K%{BLOCK_K}=0; "
            f"got N={n}, K={k}"
        )
    if out is None:
        out = torch.empty_like(weight, memory_format=torch.contiguous_format)
    if out.shape != weight.shape or out.dtype != weight.dtype:
        raise ValueError(
            "packed output must match the canonical weight shape and dtype"
        )
    if out.device != weight.device or not out.is_contiguous():
        raise ValueError("packed output must be contiguous on the weight device")
    if weight.device.type != "meta" and out.data_ptr() == weight.data_ptr():
        raise ValueError("packed output must not alias the canonical checkpoint weight")

    canonical = weight.contiguous()
    out.view(k // BLOCK_K, n // BLOCK_N, BLOCK_K, BLOCK_N).copy_(
        canonical.view(n // BLOCK_N, BLOCK_N, k // BLOCK_K, BLOCK_K).permute(2, 0, 3, 1)
    )
    return out


def launch_gluon_mm_fp8_blockscale_largem_gfx950(
    activation: torch.Tensor,
    packed_weight: torch.Tensor,
    activation_scales: torch.Tensor,
    weight_scales: torch.Tensor,
    out_dtype: torch.dtype,
    *,
    block_size: list[int],
    weight_layout: str,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute the exact experimental GLM-5.3 large-row projection contract."""
    if weight_layout != GLUON_BLOCK_FP8_WEIGHT_LAYOUT:
        raise ValueError("block-FP8 weight requires the N64 packed layout")

    m, k = activation.shape
    n, weight_k = packed_weight.shape
    if block_size != [128, 128] or weight_k != k:
        raise ValueError("block-FP8 operands require matching K128 scale blocks")
    if not supports_gluon_fp8_blockscale_largem(m, n, k):
        raise ValueError(f"unsupported block-FP8 shape: M={m}, N={n}, K={k}")
    if (
        activation.dtype != torch.float8_e4m3fn
        or packed_weight.dtype != torch.float8_e4m3fn
        or activation_scales.dtype != torch.float32
        or weight_scales.dtype != torch.float32
        or out_dtype != torch.bfloat16
    ):
        raise TypeError(
            "block-FP8 projection requires E4M3, FP32 scales, and BF16 output"
        )
    if not packed_weight.is_contiguous():
        raise ValueError("block-FP8 packed weight must be contiguous")
    k_tiles = k // BLOCK_K
    if activation_scales.shape != (m, k_tiles) or weight_scales.shape != (
        n // 128,
        k_tiles,
    ):
        raise ValueError("block-FP8 scales do not match operand shapes")

    output = out
    if output is None:
        output = torch.empty((m, n), device=activation.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)
    gluon_mm_fp8_blockscale_largem_gfx950[grid](
        activation.view(torch.uint8),
        packed_weight.view(torch.uint8),
        activation_scales,
        weight_scales,
        output,
        m,
        n,
        k,
        activation.stride(0),
        activation.stride(1),
        activation_scales.stride(0),
        activation_scales.stride(1),
        weight_scales.stride(0),
        weight_scales.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        WARPS_M=WARPS_M,
        WARPS_N=WARPS_N,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=NUM_WARPS,
        num_stages=1,
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),),
    )
    return output


__all__ = [
    "GLM53_BLOCK_FP8_PRIMARY_ROWS",
    "GLM53_BLOCK_FP8_PROJECTION_SHAPES",
    "GLUON_BLOCK_FP8_WEIGHT_LAYOUT",
    "launch_gluon_mm_fp8_blockscale_largem_gfx950",
    "pack_gluon_fp8_blockscale_weight",
    "supports_gluon_fp8_blockscale_largem",
]
