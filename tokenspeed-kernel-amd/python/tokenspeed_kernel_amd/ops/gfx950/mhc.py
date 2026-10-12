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

"""GFX950 mHC projection, reduction, and Sinkhorn kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton

cdna4 = gl.amd.cdna4
async_copy = gl.amd.cdna4.async_copy

_PREFILL_BLOCK_M = 64
_PREFILL_BLOCK_N = 32
_PREFILL_BLOCK_K = 128
_PREFILL_NUM_WARPS = 4
_PREFILL_OUTPUTS = 24
# Gluon kernels cannot close over ordinary Python globals. Keep explicit
# compile-time mirrors while retaining plain integers for host-side validation
# and launch arithmetic.
_PREFILL_BLOCK_M_GL = gl.constexpr(_PREFILL_BLOCK_M)
_PREFILL_BLOCK_N_GL = gl.constexpr(_PREFILL_BLOCK_N)
_PREFILL_BLOCK_K_GL = gl.constexpr(_PREFILL_BLOCK_K)
_PREFILL_NUM_WARPS_GL = gl.constexpr(_PREFILL_NUM_WARPS)
_PREFILL_OUTPUTS_GL = gl.constexpr(_PREFILL_OUTPUTS)

__all__ = [
    "gluon_mhc_pre_reduce_apply_gfx950",
    "launch_gluon_mhc_apply_pre_gfx950",
    "launch_gluon_mhc_mixes_gfx950",
    "launch_gluon_mhc_post_gfx950",
    "launch_gluon_mhc_prefill_gfx950",
    "launch_gluon_mhc_prefill_project_gfx950",
]


def _mhc_prefill_launch_metadata(grid, kernel, args):
    """Expose logical projection work to Proton."""
    tokens = args["num_tokens"]
    hidden = args["HIDDEN_SIZE"]
    splits = grid[0]
    return {
        "name": kernel.name,
        "flops32": 2 * tokens * 24 * 4 * hidden,
        "bytes": tokens * 4 * hidden * 2
        + 24 * 4 * hidden * 4
        + splits * tokens * (24 + 1) * 4,
    }


@gluon.jit(launch_metadata=_mhc_prefill_launch_metadata)
def gluon_mhc_prefill_project_gfx950(
    residual,
    fn,
    out_mul,
    out_sqrsum,
    num_tokens,
    HIDDEN_SIZE: gl.constexpr,
    SPLIT_K: gl.constexpr,
    K_TILES: gl.constexpr,
):
    """Four-wave FP32-weight projection matching the GLM mHC contract."""
    split_id = gl.program_id(0)
    token_block = gl.program_id(1)
    hc_hidden_size: gl.constexpr = 4 * HIDDEN_SIZE

    # Each wave owns sixteen rows. Residual values stay in VGPRs, matching the
    # reference kernel's one-load-per-row dataflow, while all four waves share
    # one asynchronously prefetched FP32 weight tile through LDS.
    a_load_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8], [16, 4], [_PREFILL_NUM_WARPS_GL, 1], [1, 0]
    )
    b_load_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [64, 1], [1, _PREFILL_NUM_WARPS_GL], [0, 1]
    )
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 4],
        transposed=True,
        warps_per_cta=[_PREFILL_NUM_WARPS_GL, 1],
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=1
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=1
    )
    # FP32 weights use the same four-float bank swizzle as the audited
    # SGLang/AITER implementation: K is contiguous and each N row advances
    # the XOR phase by one 128-bit vector.
    weight_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(4, 1, 16, order=[0, 1])
    weight_shared = gl.allocate_shared_memory(
        gl.float32,
        [2, _PREFILL_BLOCK_K_GL, _PREFILL_BLOCK_N_GL],
        weight_shared_layout,
    )

    m_layout: gl.constexpr = gl.SliceLayout(1, a_load_layout)
    a_k_layout: gl.constexpr = gl.SliceLayout(0, a_load_layout)
    b_k_layout: gl.constexpr = gl.SliceLayout(1, b_load_layout)
    n_layout: gl.constexpr = gl.SliceLayout(0, b_load_layout)
    rows = token_block * _PREFILL_BLOCK_M_GL + gl.arange(
        0, _PREFILL_BLOCK_M_GL, layout=m_layout
    )
    a_k = gl.arange(0, _PREFILL_BLOCK_K_GL, layout=a_k_layout)
    b_k = gl.arange(0, _PREFILL_BLOCK_K_GL, layout=b_k_layout)
    cols = gl.arange(0, _PREFILL_BLOCK_N_GL, layout=n_layout)
    row_mask = rows[:, None] < num_tokens
    split_start = split_id * SPLIT_K

    a_base_offsets = (rows[:, None] * hc_hidden_size + split_start + a_k[None, :]).to(
        gl.int32
    )
    b_base_offsets = (cols[None, :] * hc_hidden_size + split_start + b_k[:, None]).to(
        gl.int32
    )
    b_mask = cols[None, :] < _PREFILL_OUTPUTS_GL

    async_copy.buffer_load_to_shared(
        weight_shared.index(0),
        fn,
        b_base_offsets,
        mask=b_mask,
        other=0.0,
    )
    async_copy.commit_group()
    async_copy.buffer_load_to_shared(
        weight_shared.index(1),
        fn,
        (b_base_offsets + _PREFILL_BLOCK_K_GL).to(gl.int32),
        mask=b_mask,
        other=0.0,
    )
    async_copy.commit_group()

    projection = gl.zeros(
        (_PREFILL_BLOCK_M_GL, _PREFILL_BLOCK_N_GL), gl.float32, mfma_layout
    )
    square_sum = gl.zeros((_PREFILL_BLOCK_M_GL,), gl.float32, m_layout)

    # K_TILES is even for every supported hidden-size/split combination. The
    # loop consumes two tiles while refilling their two fixed LDS buffers, so
    # compilation stays compact even for the 128-tile production projection.
    for pair in tl.range(0, K_TILES // 2 - 1):
        async_copy.wait_group(1)
        weight_0 = weight_shared.index(0).load(dot_b_layout)
        even_k = pair * 2 * _PREFILL_BLOCK_K_GL
        residual_0_fp32 = cdna4.buffer_load(
            residual,
            (a_base_offsets + even_k).to(gl.int32),
            mask=row_mask,
            other=0.0,
        ).to(gl.float32)
        square_sum += gl.sum(residual_0_fp32 * residual_0_fp32, axis=1)
        projection = cdna4.mfma(
            gl.convert_layout(residual_0_fp32, dot_a_layout),
            weight_0,
            projection,
        )

        next_even_k = (pair * 2 + 2) * _PREFILL_BLOCK_K_GL
        async_copy.buffer_load_to_shared(
            weight_shared.index(0),
            fn,
            (b_base_offsets + next_even_k).to(gl.int32),
            mask=b_mask,
            other=0.0,
        )
        async_copy.commit_group()

        # Waiting for one group leaves the just-issued even refill in flight
        # while making the older odd tile safe to consume.
        async_copy.wait_group(1)
        weight_1 = weight_shared.index(1).load(dot_b_layout)
        residual_1_fp32 = cdna4.buffer_load(
            residual,
            (a_base_offsets + (pair * 2 + 1) * _PREFILL_BLOCK_K_GL).to(gl.int32),
            mask=row_mask,
            other=0.0,
        ).to(gl.float32)
        square_sum += gl.sum(residual_1_fp32 * residual_1_fp32, axis=1)
        projection = cdna4.mfma(
            gl.convert_layout(residual_1_fp32, dot_a_layout),
            weight_1,
            projection,
        )

        next_odd_k = (pair * 2 + 3) * _PREFILL_BLOCK_K_GL
        async_copy.buffer_load_to_shared(
            weight_shared.index(1),
            fn,
            (b_base_offsets + next_odd_k).to(gl.int32),
            mask=b_mask,
            other=0.0,
        )
        async_copy.commit_group()

    final_even_k: gl.constexpr = (K_TILES - 2) * _PREFILL_BLOCK_K_GL
    async_copy.wait_group(1)
    weight_0 = weight_shared.index(0).load(dot_b_layout)
    residual_0_fp32 = cdna4.buffer_load(
        residual,
        (a_base_offsets + final_even_k).to(gl.int32),
        mask=row_mask,
        other=0.0,
    ).to(gl.float32)
    square_sum += gl.sum(residual_0_fp32 * residual_0_fp32, axis=1)
    projection = cdna4.mfma(
        gl.convert_layout(residual_0_fp32, dot_a_layout),
        weight_0,
        projection,
    )
    async_copy.wait_group(0)
    weight_1 = weight_shared.index(1).load(dot_b_layout)
    residual_1 = cdna4.buffer_load(
        residual,
        (a_base_offsets + (K_TILES - 1) * _PREFILL_BLOCK_K_GL).to(gl.int32),
        mask=row_mask,
        other=0.0,
    ).to(gl.float32)
    square_sum += gl.sum(residual_1 * residual_1, axis=1)
    projection = cdna4.mfma(
        gl.convert_layout(residual_1, dot_a_layout),
        weight_1,
        projection,
    )

    output_rows = token_block * _PREFILL_BLOCK_M_GL + gl.arange(
        0, _PREFILL_BLOCK_M_GL, layout=gl.SliceLayout(1, mfma_layout)
    )
    output_cols = gl.arange(
        0, _PREFILL_BLOCK_N_GL, layout=gl.SliceLayout(0, mfma_layout)
    )
    output_offsets = (
        split_id * num_tokens * _PREFILL_OUTPUTS_GL
        + output_rows[:, None] * _PREFILL_OUTPUTS_GL
        + output_cols[None, :]
    ).to(gl.int32)
    cdna4.buffer_store(
        projection,
        out_mul,
        output_offsets,
        mask=(output_rows[:, None] < num_tokens)
        & (output_cols[None, :] < _PREFILL_OUTPUTS_GL),
    )
    cdna4.buffer_store(
        square_sum,
        out_sqrsum,
        (split_id * num_tokens + rows).to(gl.int32),
        mask=rows < num_tokens,
    )


def launch_gluon_mhc_prefill_project_gfx950(
    residual: torch.Tensor,
    fn: torch.Tensor,
    out_mul: torch.Tensor,
    out_sqrsum: torch.Tensor,
    *,
    n_splits: int,
    block_m: int,
    block_k: int,
) -> None:
    """Project four BF16 streams with FP32 weights and fixed GFX950 tiles.

    The portable prefill hook supplies ``block_m`` and ``block_k``; this kernel
    uses its own tile size instead.
    """
    num_tokens, _, hidden_size = residual.shape
    if hidden_size not in (4096, 7168) or n_splits not in (1, 2, 4, 8):
        raise ValueError("Unsupported GFX950 mHC prefill shape or split count")
    if fn.shape != (24, 4 * hidden_size):
        raise ValueError("GFX950 mHC projection weight shape mismatch")
    if not all(
        tensor.is_contiguous() for tensor in (residual, fn, out_mul, out_sqrsum)
    ):
        raise ValueError("GFX950 mHC projection tensors must be contiguous")
    split_k = 4 * hidden_size // n_splits
    if num_tokens * 4 * hidden_size >= 2**31:
        raise ValueError("GFX950 mHC prefill projection exceeds 32-bit buffer offsets")
    gluon_mhc_prefill_project_gfx950[
        (
            n_splits,
            (num_tokens + _PREFILL_BLOCK_M - 1) // _PREFILL_BLOCK_M,
        )
    ](
        residual,
        fn,
        out_mul,
        out_sqrsum,
        num_tokens,
        HIDDEN_SIZE=hidden_size,
        SPLIT_K=split_k,
        K_TILES=split_k // _PREFILL_BLOCK_K,
        num_warps=_PREFILL_NUM_WARPS,
        num_stages=1,
        waves_per_eu=1,
    )


def _mhc_prefill_mix_metadata(grid, kernel, args):
    return {
        "name": kernel.name,
        "bytes": args["num_tokens"] * (args["n_splits"] * 25 + 24) * 4,
    }


@gluon.jit(launch_metadata=_mhc_prefill_mix_metadata)
def gluon_mhc_prefill_mix_gfx950(
    projection,
    square_sum,
    hc_scale,
    hc_base,
    pre_mix,
    post_mix,
    comb_mix,
    num_tokens,
    n_splits,
    HIDDEN_SIZE: gl.constexpr,
    RMS_EPS: gl.constexpr,
    HC_EPS: gl.constexpr,
    SINKHORN_ITERS: gl.constexpr,
):
    token = gl.program_id(0)
    vector_layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    pre_post_offsets = gl.arange(0, 8, layout=vector_layout)
    pre_post_values = gl.zeros([8], gl.float32, layout=vector_layout)
    matrix_layout: gl.constexpr = gl.BlockedLayout([1, 1], [16, 4], [1, 1], [1, 0])
    rows = gl.arange(0, 4, layout=gl.SliceLayout(1, matrix_layout))
    cols = gl.arange(0, 4, layout=gl.SliceLayout(0, matrix_layout))
    comb_offsets = rows[:, None] * 4 + cols[None, :]
    comb_values = gl.zeros([4, 4], gl.float32, layout=matrix_layout)
    rms_sum = 0.0
    # Keep batch-dependent split counts out of the compilation key and retain
    # the sequential partial-sum order used by the portable implementation.
    for split in range(n_splits):
        split_base = split * num_tokens * 24 + token * 24
        pre_post_values += gl.load(projection + split_base + pre_post_offsets)
        comb_values += gl.load(projection + split_base + 8 + comb_offsets)
        rms_sum += gl.load(square_sum + split * num_tokens + token)

    inverse_rms = gl.rsqrt(rms_sum / (4 * HIDDEN_SIZE) + RMS_EPS)
    pre_post_scale = gl.where(
        pre_post_offsets < 4, gl.load(hc_scale), gl.load(hc_scale + 1)
    )
    pre_post_values = 1.0 / (
        1.0
        + gl.exp(
            -(
                pre_post_values * inverse_rms * pre_post_scale
                + gl.load(hc_base + pre_post_offsets)
            )
        )
    )
    gl.store(
        pre_mix + token * 4 + pre_post_offsets,
        pre_post_values + HC_EPS,
        mask=pre_post_offsets < 4,
    )
    gl.store(
        post_mix + token * 4 + pre_post_offsets - 4,
        pre_post_values * 2.0,
        mask=pre_post_offsets >= 4,
    )
    comb = comb_values * inverse_rms * gl.load(hc_scale + 2) + gl.load(
        hc_base + 8 + comb_offsets
    )
    row_max = gl.max(comb, axis=1)
    comb = gl.exp(comb - row_max[:, None])
    row_sum = gl.sum(comb, axis=1)
    comb = comb / row_sum[:, None] + HC_EPS
    col_sum = gl.sum(comb, axis=0)
    comb = comb / (col_sum[None, :] + HC_EPS)
    for _ in gl.static_range(1, SINKHORN_ITERS):
        row_sum = gl.sum(comb, axis=1)
        comb = comb / (row_sum[:, None] + HC_EPS)
        col_sum = gl.sum(comb, axis=0)
        comb = comb / (col_sum[None, :] + HC_EPS)
    gl.store(comb_mix + token * 16 + comb_offsets, comb)


def launch_gluon_mhc_prefill_mix_gfx950(
    projection: torch.Tensor,
    square_sum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    pre_mix: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    *,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    n_splits: int,
    num_tokens: int,
) -> None:
    """Reduce FP32 projection partials and write all four-stream mix tensors."""
    gluon_mhc_prefill_mix_gfx950[(num_tokens,)](
        projection,
        square_sum,
        hc_scale,
        hc_base,
        pre_mix,
        post_mix,
        comb_mix,
        num_tokens,
        n_splits,
        HIDDEN_SIZE=hidden_size,
        RMS_EPS=rms_eps,
        HC_EPS=hc_eps,
        SINKHORN_ITERS=sinkhorn_iters,
        num_warps=1,
    )


def _mhc_prefill_apply_metadata(grid, kernel, args):
    tokens = args["residual"].numel() // (4 * args["HIDDEN_SIZE"])
    norm_bytes = args["norm_weight"].element_size() if args["NORM"] else 0
    return {
        "name": kernel.name,
        "bytes": tokens * args["HIDDEN_SIZE"] * (10 + norm_bytes),
        "flops32": tokens * (args["HIDDEN_SIZE"] * (12 if args["NORM"] else 8)),
    }


@gluon.jit(launch_metadata=_mhc_prefill_apply_metadata)
def gluon_mhc_prefill_apply_gfx950(
    pre_mix,
    residual,
    layer_input,
    norm_weight,
    norm_weight_stride,
    HIDDEN_SIZE: gl.constexpr,
    BLOCK_H: gl.constexpr,
    NORM: gl.constexpr,
    NORM_EPS: gl.constexpr,
):
    token = gl.program_id(0)
    block = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout([4], [64], [4], [0])
    hidden = block * BLOCK_H + gl.arange(0, BLOCK_H, layout=layout)
    mask = hidden < HIDDEN_SIZE
    accumulator = gl.zeros([BLOCK_H], gl.float32, layout=layout)
    for stream in gl.static_range(4):
        coefficient = gl.load(pre_mix + token * 4 + stream)
        value = gl.load(
            residual + (token * 4 + stream) * HIDDEN_SIZE + hidden,
            mask=mask,
            other=0.0,
        ).to(gl.float32)
        accumulator += coefficient * value
    # The public operation rounds the unnormalized layer input to BF16 before
    # optional RMS normalization; preserve that rounding when fusing the stage.
    rounded = accumulator.to(gl.bfloat16)
    if NORM:
        values = rounded.to(gl.float32)
        sum_square = gl.sum(gl.where(mask, values * values, 0.0), axis=0)
        inverse_rms = gl.rsqrt(sum_square / HIDDEN_SIZE + NORM_EPS)
        weight = gl.load(
            norm_weight + hidden * norm_weight_stride, mask=mask, other=0.0
        ).to(gl.float32)
        rounded = (values * inverse_rms * weight).to(gl.bfloat16)
    gl.store(layer_input + token * HIDDEN_SIZE + hidden, rounded, mask=mask)


def launch_gluon_mhc_prefill_apply_gfx950(
    pre_mix: torch.Tensor,
    residual: torch.Tensor,
    layer_input: torch.Tensor,
    *,
    hidden_size: int,
    norm_weight: torch.Tensor | None,
    norm_eps: float | None,
) -> None:
    """Mix four BF16 streams, with optional RMS normalization after rounding."""
    block_h = 1 << (hidden_size - 1).bit_length() if norm_weight is not None else 1024
    gluon_mhc_prefill_apply_gfx950[
        (residual.shape[0], (hidden_size + block_h - 1) // block_h)
    ](
        pre_mix,
        residual,
        layer_input,
        norm_weight if norm_weight is not None else layer_input,
        norm_weight.stride(0) if norm_weight is not None else 1,
        HIDDEN_SIZE=hidden_size,
        BLOCK_H=block_h,
        NORM=norm_weight is not None,
        NORM_EPS=norm_eps if norm_eps is not None else 0.0,
        num_warps=4,
    )


def launch_gluon_mhc_prefill_gfx950(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    *,
    norm_weight: torch.Tensor | None,
    norm_eps: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute complete four-stream BF16 mHC prefill on gfx950.

    The projection, split reduction, Sinkhorn mapping, output mixing and
    optional output normalization are all Gluon kernels. Outputs preserve the
    residual's leading dimensions and have BF16/FP32/FP32 storage respectively.
    """
    outer_shape = residual.shape[:-2]
    hidden_size = residual.shape[-1]
    residual_flat = residual.view(-1, 4, hidden_size)
    num_tokens = residual_flat.shape[0]
    if num_tokens == 0:
        return (
            residual.new_empty(*outer_shape, hidden_size),
            torch.empty(
                *outer_shape, 4, 1, dtype=torch.float32, device=residual.device
            ),
            torch.empty(
                *outer_shape, 4, 4, dtype=torch.float32, device=residual.device
            ),
        )
    # These bounded split choices retain the projection's tuned geometry.
    n_splits = (
        8
        if num_tokens <= 2048
        else 4 if num_tokens <= 4096 else 2 if num_tokens <= 8192 else 1
    )
    projection = torch.empty(
        n_splits, num_tokens, 24, dtype=torch.float32, device=residual.device
    )
    square_sum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )
    pre_mix = torch.empty(num_tokens, 4, dtype=torch.float32, device=residual.device)
    post_mix = torch.empty_like(pre_mix)
    comb_mix = torch.empty(num_tokens, 16, dtype=torch.float32, device=residual.device)
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )
    launch_gluon_mhc_prefill_project_gfx950(
        residual_flat,
        fn,
        projection,
        square_sum,
        n_splits=n_splits,
        block_m=_PREFILL_BLOCK_M,
        block_k=_PREFILL_BLOCK_K,
    )
    launch_gluon_mhc_prefill_mix_gfx950(
        projection,
        square_sum,
        hc_scale,
        hc_base,
        pre_mix,
        post_mix,
        comb_mix,
        hidden_size=hidden_size,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        n_splits=n_splits,
        num_tokens=num_tokens,
    )
    launch_gluon_mhc_prefill_apply_gfx950(
        pre_mix,
        residual_flat,
        layer_input,
        hidden_size=hidden_size,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )
    return (
        layer_input.view(*outer_shape, hidden_size),
        post_mix.view(*outer_shape, 4, 1),
        comb_mix.view(*outer_shape, 4, 4),
    )


@gluon.jit
def gluon_mhc_pre_gfx950(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    layer_input,
    post_mix,
    comb_mix,
    num_tokens,
    HIDDEN_SIZE: gl.constexpr,
    RMS_EPS: gl.constexpr,
    HC_EPS: gl.constexpr,
    SINKHORN_ITERS: gl.constexpr,
    N_SPLITS: gl.constexpr,
    BLOCK_H: gl.constexpr,
):
    token = gl.program_id(0)
    hidden_block = gl.program_id(1)
    split_layout: gl.constexpr = gl.BlockedLayout([2], [64], [1], [0])
    splits = gl.arange(0, 128, layout=split_layout)
    split_mask = splits < N_SPLITS

    square_parts = gl.load(
        gemm_out_sqrsum + splits * num_tokens + token,
        mask=split_mask,
        other=0.0,
    ).to(gl.float32)
    square_sum = gl.sum(square_parts, axis=0)
    inverse_rms = gl.rsqrt(square_sum / (4 * HIDDEN_SIZE) + RMS_EPS)
    pre_scale = gl.load(hc_scale).to(gl.float32)

    hidden_layout: gl.constexpr = gl.BlockedLayout([8], [64], [1], [0])
    dims = hidden_block * BLOCK_H + gl.arange(0, BLOCK_H, layout=hidden_layout)
    hidden_acc = gl.zeros([BLOCK_H], gl.float32, layout=hidden_layout)
    for head in gl.static_range(4):
        projection_parts = gl.load(
            gemm_out_mul + splits * num_tokens * 24 + token * 24 + head,
            mask=split_mask,
            other=0.0,
        ).to(gl.float32)
        projection = gl.sum(projection_parts, axis=0)
        pre = (
            1.0
            / (
                1.0
                + gl.exp(
                    -(
                        projection * inverse_rms * pre_scale
                        + gl.load(hc_base + head).to(gl.float32)
                    )
                )
            )
            + HC_EPS
        )
        values = gl.load(
            residual + token * 4 * HIDDEN_SIZE + head * HIDDEN_SIZE + dims
        ).to(gl.float32)
        hidden_acc += pre * values
    gl.store(layer_input + token * HIDDEN_SIZE + dims, hidden_acc.to(gl.bfloat16))

    if hidden_block == 0:
        vector_layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
        lane = gl.arange(0, 64, layout=vector_layout)
        post_projection = gl.zeros([64], gl.float32, layout=vector_layout)
        for split in gl.static_range(N_SPLITS):
            post_projection += gl.load(
                gemm_out_mul + split * num_tokens * 24 + token * 24 + lane,
                mask=(lane >= 4) & (lane < 8),
                other=0.0,
            ).to(gl.float32)
        post_scale = gl.load(hc_scale + 1).to(gl.float32)
        post_values = 2.0 / (
            1.0
            + gl.exp(
                -(
                    post_projection * inverse_rms * post_scale
                    + gl.load(
                        hc_base + lane, mask=(lane >= 4) & (lane < 8), other=0.0
                    ).to(gl.float32)
                )
            )
        )
        gl.store(
            post_mix + token * 4 + lane - 4,
            post_values,
            mask=(lane >= 4) & (lane < 8),
        )

        matrix_layout: gl.constexpr = gl.BlockedLayout([1, 1], [4, 16], [1, 1], [1, 0])
        rows = gl.arange(0, 4, layout=gl.SliceLayout(1, matrix_layout))
        cols = gl.arange(0, 16, layout=gl.SliceLayout(0, matrix_layout))
        active = cols[None, :] < 4
        comb_offsets = rows[:, None] * 4 + cols[None, :]
        comb_projection = gl.zeros([4, 16], gl.float32, layout=matrix_layout)
        for split in gl.static_range(N_SPLITS):
            comb_projection += gl.load(
                gemm_out_mul + split * num_tokens * 24 + token * 24 + 8 + comb_offsets,
                mask=active,
                other=0.0,
            ).to(gl.float32)
        comb_scale = gl.load(hc_scale + 2).to(gl.float32)
        comb = gl.where(
            active,
            comb_projection * inverse_rms * comb_scale
            + gl.load(hc_base + 8 + comb_offsets, mask=active, other=0.0).to(
                gl.float32
            ),
            0.0,
        )
        row_max = gl.max(gl.where(active, comb, -float("inf")), axis=1)
        comb = gl.where(active, gl.exp(comb - row_max[:, None]), comb)
        row_sum = gl.sum(gl.where(active, comb, 0.0), axis=1)
        comb = gl.where(active, comb / row_sum[:, None] + HC_EPS, comb)
        col_sum = gl.sum(gl.where(active, comb, 0.0), axis=0)
        comb = gl.where(active, comb / (col_sum[None, :] + HC_EPS), comb)
        for _ in gl.static_range(1, SINKHORN_ITERS):
            row_sum = gl.sum(gl.where(active, comb, 0.0), axis=1)
            comb = gl.where(active, comb / (row_sum[:, None] + HC_EPS), comb)
            col_sum = gl.sum(gl.where(active, comb, 0.0), axis=0)
            comb = gl.where(active, comb / (col_sum[None, :] + HC_EPS), comb)
        gl.store(comb_mix + token * 16 + comb_offsets, comb, mask=active)


def _mhc_pre_reduce_apply_block_h(
    num_tokens: int, hidden_size: int, n_splits: int
) -> int:
    if (num_tokens, hidden_size, n_splits) == (64, 4096, 64):
        return 1024
    return 512


def gluon_mhc_pre_reduce_apply_gfx950(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    residual: torch.Tensor,
    layer_input: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    n_splits: int,
    num_tokens: int,
) -> None:
    """Reduce mHC partials, apply the pre mix, and emit post coefficients."""
    if gemm_out_mul.shape != (n_splits, num_tokens, 24):
        raise ValueError("gemm_out_mul must have shape [n_splits, tokens, 24]")
    if gemm_out_sqrsum.shape != (n_splits, num_tokens):
        raise ValueError("gemm_out_sqrsum must have shape [n_splits, tokens]")
    if residual.shape != (num_tokens, 4, hidden_size):
        raise ValueError("residual must have shape [tokens, 4, hidden_size]")
    if layer_input.shape != (num_tokens, hidden_size):
        raise ValueError("layer_input must have shape [tokens, hidden_size]")
    expected_splits = hidden_size // 64
    if hidden_size not in (4096, 7168) or n_splits != expected_splits:
        raise ValueError(
            "GFX950 mHC specialization requires hidden_size=4096 or 7168 "
            "with the matching split-K decomposition"
        )
    if sinkhorn_iters != 20:
        raise ValueError("GFX950 mHC specialization requires 20 Sinkhorn iterations")
    if not 1 <= num_tokens <= 64:
        raise ValueError("GFX950 mHC specialization requires 1-64 tokens")

    block_h = _mhc_pre_reduce_apply_block_h(num_tokens, hidden_size, n_splits)
    gluon_mhc_pre_gfx950[(num_tokens, hidden_size // block_h)](
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        layer_input,
        post_mix,
        comb_mix,
        num_tokens,
        HIDDEN_SIZE=hidden_size,
        RMS_EPS=rms_eps,
        HC_EPS=hc_eps,
        SINKHORN_ITERS=sinkhorn_iters,
        N_SPLITS=n_splits,
        BLOCK_H=block_h,
        num_warps=1,
    )


# ===-----------------------------------------------------------------------===#
# DeepSeek V4.1 hc=4 streaming helpers: post-mapping and the stream collapse
# with optional fused RMSNorm. Both are pure HBM streams over the residual.
# ===-----------------------------------------------------------------------===#

_POST_BLOCK_H = 512


def _mhc_post_metadata(grid, kernel, args):
    tokens = args["num_tokens"]
    hidden = args["HIDDEN_SIZE"]
    return {
        "name": kernel.name,
        # Read the layer output and four streams, write four streams.
        "bytes": tokens * hidden * (2 + 8 + 8) + tokens * 20 * 4,
        "flops32": tokens * hidden * 4 * 9,
    }


@gluon.jit(launch_metadata=_mhc_post_metadata, do_not_specialize=("num_tokens",))
def gluon_mhc_post_gfx950(
    hidden_states,
    residual,
    post,
    comb,
    out,
    num_tokens,
    HIDDEN_SIZE: gl.constexpr,
    BLOCK_H: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    # One token per CTA keeps the 20 coefficients scalar (SMEM) loads, which
    # do not queue behind the vector stores; each lane moves eight contiguous
    # BF16 values (16-byte accesses).
    token = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [NUM_WARPS], [0])
    hidden = gl.program_id(1) * BLOCK_H + gl.arange(0, BLOCK_H, layout=layout)
    mask = hidden < HIDDEN_SIZE
    p0 = gl.load(post + token * 4)
    p1 = gl.load(post + token * 4 + 1)
    p2 = gl.load(post + token * 4 + 2)
    p3 = gl.load(post + token * 4 + 3)
    c0 = gl.load(comb + token * 16)
    c1 = gl.load(comb + token * 16 + 1)
    c2 = gl.load(comb + token * 16 + 2)
    c3 = gl.load(comb + token * 16 + 3)
    c4 = gl.load(comb + token * 16 + 4)
    c5 = gl.load(comb + token * 16 + 5)
    c6 = gl.load(comb + token * 16 + 6)
    c7 = gl.load(comb + token * 16 + 7)
    c8 = gl.load(comb + token * 16 + 8)
    c9 = gl.load(comb + token * 16 + 9)
    c10 = gl.load(comb + token * 16 + 10)
    c11 = gl.load(comb + token * 16 + 11)
    c12 = gl.load(comb + token * 16 + 12)
    c13 = gl.load(comb + token * 16 + 13)
    c14 = gl.load(comb + token * 16 + 14)
    c15 = gl.load(comb + token * 16 + 15)
    x = cdna4.buffer_load(hidden_states, token * HIDDEN_SIZE + hidden, mask=mask).to(
        gl.float32
    )
    base = token * (4 * HIDDEN_SIZE) + hidden
    r0 = cdna4.buffer_load(residual, base, mask=mask).to(gl.float32)
    r1 = cdna4.buffer_load(residual, base + HIDDEN_SIZE, mask=mask).to(gl.float32)
    r2 = cdna4.buffer_load(residual, base + 2 * HIDDEN_SIZE, mask=mask).to(gl.float32)
    r3 = cdna4.buffer_load(residual, base + 3 * HIDDEN_SIZE, mask=mask).to(gl.float32)
    # Same accumulation order as the portable kernel: post * x first, then
    # input streams 0..3; comb is indexed [input stream, output stream].
    out0 = p0 * x
    out0 += c0 * r0
    out0 += c4 * r1
    out0 += c8 * r2
    out0 += c12 * r3
    cdna4.buffer_store(out0.to(gl.bfloat16), out, base, mask=mask)
    out1 = p1 * x
    out1 += c1 * r0
    out1 += c5 * r1
    out1 += c9 * r2
    out1 += c13 * r3
    cdna4.buffer_store(out1.to(gl.bfloat16), out, base + HIDDEN_SIZE, mask=mask)
    out2 = p2 * x
    out2 += c2 * r0
    out2 += c6 * r1
    out2 += c10 * r2
    out2 += c14 * r3
    cdna4.buffer_store(out2.to(gl.bfloat16), out, base + 2 * HIDDEN_SIZE, mask=mask)
    out3 = p3 * x
    out3 += c3 * r0
    out3 += c7 * r1
    out3 += c11 * r2
    out3 += c15 * r3
    cdna4.buffer_store(out3.to(gl.bfloat16), out, base + 3 * HIDDEN_SIZE, mask=mask)


def launch_gluon_mhc_post_gfx950(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Apply the hc=4 mHC post-mapping and residual update.

    Args:
        hidden_states: BF16 layer output ``[..., hidden_size]``.
        residual: BF16 residual streams ``[..., 4, hidden_size]``.
        post: FP32 post coefficients ``[..., 4, 1]``.
        comb: FP32 combination ``[..., 4, 4]`` indexed [input, output].

    Returns:
        BF16 ``post * hidden_states + comb^T residual`` shaped like ``residual``.
    """
    hidden_size = residual.shape[-1]
    if residual.shape[-2] != 4:
        raise ValueError("GFX950 mHC post requires four residual streams")
    tensors = (hidden_states, residual, post, comb)
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("GFX950 mHC post tensors must be contiguous")
    if residual.numel() >= 2**31:
        raise ValueError("GFX950 mHC post exceeds 32-bit buffer offsets")
    out = torch.empty_like(residual)
    num_tokens = residual.numel() // (4 * hidden_size)
    if num_tokens:
        gluon_mhc_post_gfx950[(num_tokens, triton.cdiv(hidden_size, _POST_BLOCK_H))](
            hidden_states,
            residual,
            post,
            comb,
            out,
            num_tokens,
            HIDDEN_SIZE=hidden_size,
            BLOCK_H=_POST_BLOCK_H,
            NUM_WARPS=_POST_BLOCK_H // 512,
            num_warps=_POST_BLOCK_H // 512,
        )
    return out


def _mhc_apply_pre_metadata(grid, kernel, args):
    tokens = args["num_tokens"]
    hidden = args["HIDDEN_SIZE"]
    return {
        "name": kernel.name,
        "bytes": tokens * hidden * (8 + 2)
        + tokens * 16
        + (hidden * 2 if args["NORM"] else 0),
        "flops32": tokens * hidden * (12 if args["NORM"] else 8),
    }


@gluon.jit(launch_metadata=_mhc_apply_pre_metadata, do_not_specialize=("num_tokens",))
def gluon_mhc_apply_pre_gfx950(
    pre_mix,
    residual,
    norm_weight,
    out,
    num_tokens,
    HIDDEN_SIZE: gl.constexpr,
    BLOCK_H: gl.constexpr,
    NORM: gl.constexpr,
    EPS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    # One token per CTA so the optional RMSNorm reduces in one pass; every
    # lane moves eight contiguous BF16 values (16-byte accesses).
    token = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [NUM_WARPS], [0])
    hidden = gl.arange(0, BLOCK_H, layout=layout)
    mask = hidden < HIDDEN_SIZE
    base = token * (4 * HIDDEN_SIZE) + hidden
    acc = gl.zeros([BLOCK_H], gl.float32, layout=layout)
    for stream in gl.static_range(4):
        coefficient = gl.load(pre_mix + token * 4 + stream)
        acc += coefficient * cdna4.buffer_load(
            residual, base + stream * HIDDEN_SIZE, mask=mask
        ).to(gl.float32)
    # The weighted stream sum is rounded to BF16 before RMSNorm reads it,
    # as in the unfused path.
    values = acc.to(gl.bfloat16)
    if NORM:
        normed = values.to(gl.float32)
        variance = gl.sum(normed * normed, axis=0) / HIDDEN_SIZE
        scale = gl.rsqrt(variance + EPS)
        weight = cdna4.buffer_load(norm_weight, hidden, mask=mask).to(gl.float32)
        values = (normed * scale * weight).to(gl.bfloat16)
    cdna4.buffer_store(values, out, token * HIDDEN_SIZE + hidden, mask=mask)


def launch_gluon_mhc_apply_pre_gfx950(
    residual: torch.Tensor,
    pre_mix: torch.Tensor,
    *,
    norm_weight: torch.Tensor | None,
    norm_eps: float | None,
) -> torch.Tensor:
    """Mix four BF16 streams with ``pre_mix``, optionally followed by RMSNorm.

    Args:
        residual: BF16 residual streams ``[..., 4, hidden_size]``.
        pre_mix: FP32 coefficients ``[..., 4]``.
        norm_weight: Optional BF16 or FP32 RMSNorm weight ``[hidden_size]``.
        norm_eps: RMSNorm epsilon, given together with ``norm_weight``.

    Returns:
        BF16 layer input ``[..., hidden_size]``.
    """
    if (norm_weight is None) != (norm_eps is None):
        raise ValueError("norm_weight and norm_eps must be provided together")
    hidden_size = residual.shape[-1]
    if residual.shape[-2] != 4 or pre_mix.shape[-1] != 4:
        raise ValueError("GFX950 mHC apply-pre requires four residual streams")
    if norm_weight is not None and norm_weight.shape != (hidden_size,):
        raise ValueError("GFX950 mHC apply-pre norm weight shape mismatch")
    tensors = (pre_mix, residual) + (() if norm_weight is None else (norm_weight,))
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("GFX950 mHC apply-pre tensors must be contiguous")
    if residual.numel() >= 2**31:
        raise ValueError("GFX950 mHC apply-pre exceeds 32-bit buffer offsets")
    out = residual.new_empty((*residual.shape[:-2], hidden_size))
    num_tokens = residual.numel() // (4 * hidden_size)
    if num_tokens:
        norm = norm_weight is not None
        gluon_mhc_apply_pre_gfx950[(num_tokens,)](
            pre_mix,
            residual,
            norm_weight if norm else out,
            out,
            num_tokens,
            HIDDEN_SIZE=hidden_size,
            BLOCK_H=triton.next_power_of_2(hidden_size),
            NORM=norm,
            EPS=norm_eps if norm else 0.0,
            NUM_WARPS=4,
            num_warps=4,
            # Match the portable kernels' rounding: the plain collapse rounds
            # each product (no FMA), the normalized one contracts to FMA.
            enable_fp_fusion=norm,
        )
    return out


# ===-----------------------------------------------------------------------===#
# DeepSeek V4.1 hc=4 mixes: one pass over the residual producing split-K
# projection partials and squared sums, then a per-token reduction that forms
# the pre/post/combination coefficients.
# ===-----------------------------------------------------------------------===#

# (BLOCK_M, NUM_WARPS) per batch bucket: small batches waste fewer MFMA rows
# with 64-row blocks; large prefill uses 256-row blocks of eight waves (32 rows
# each), which share every split weight tile across more rows.
_MIXES_SMALL = (64, 4)
_MIXES_LARGE = (256, 8)
_MIXES_BLOCK_K = 64
_MIXES_REDUCE_WARPS = 4


def _mhc_mixes_project_metadata(grid, kernel, args):
    tokens = args["num_tokens"]
    k = args["K"]
    splits = grid[0]
    return {
        "name": kernel.name,
        # Three BF16 MFMAs (hi/mid/lo weight terms) over 32 padded outputs.
        "flops16": 3 * 2 * tokens * 32 * k,
        "bytes": tokens * k * 2 + grid[1] * 24 * k * 4 + splits * tokens * 25 * 4,
    }


@gluon.jit
def _mhc_mixes_k_permutation(k, BLOCK_K: gl.constexpr):
    # The reduction is order-free, so the logical MFMA K index is permuted to
    # give every lane one contiguous run of BLOCK_K // 4 elements: lane group
    # g of a 16x16x32 operand reads columns [g * BLOCK_K // 4, ...).
    return ((k // 8) % 4) * (BLOCK_K // 4) + (k // 32) * 8 + k % 8


@gluon.jit
def _mhc_mixes_split_weight(w):
    # Split FP32 weights into three BF16 terms whose sum is exact: hi and mid
    # are 8-bit truncations, and the remainder has at most 8 significant bits.
    hi = (w.to(gl.uint32, bitcast=True) & 0xFFFF0000).to(gl.float32, bitcast=True)
    rest = w - hi
    mid = (rest.to(gl.uint32, bitcast=True) & 0xFFFF0000).to(gl.float32, bitcast=True)
    lo = rest - mid
    return hi.to(gl.bfloat16), mid.to(gl.bfloat16), lo.to(gl.bfloat16)


@gluon.jit(
    launch_metadata=_mhc_mixes_project_metadata,
    do_not_specialize=("num_tokens", "tiles_per_split"),
)
def gluon_mhc_mixes_project_gfx950(
    residual,
    fn,
    out_mul,
    out_sqrsum,
    num_tokens,
    tiles_per_split,
    K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    split = gl.program_id(0)
    row_block = gl.program_id(1)
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[NUM_WARPS, 1],
    )
    a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=8
    )
    b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=8
    )
    # The CTA loads and splits each FP32 weight tile once, contiguous K
    # values per lane, and shares the BF16 terms with all waves through LDS.
    W_PER_LANE: gl.constexpr = BLOCK_K * 32 // (64 * NUM_WARPS)
    w_layout: gl.constexpr = gl.BlockedLayout(
        [W_PER_LANE, 1],
        [BLOCK_K // W_PER_LANE, 64 * W_PER_LANE // BLOCK_K],
        [1, NUM_WARPS],
        [0, 1],
    )
    w_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 8, order=[0, 1])
    w_shared = gl.allocate_shared_memory(gl.bfloat16, [3, BLOCK_K, 32], w_shared_layout)

    rows = row_block * BLOCK_M + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, a_layout)
    )
    a_k = _mhc_mixes_k_permutation(
        gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, a_layout)), BLOCK_K
    )
    w_k = _mhc_mixes_k_permutation(
        gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, w_layout)), BLOCK_K
    )
    w_cols = gl.arange(0, 32, layout=gl.SliceLayout(0, w_layout))
    a_mask = rows[:, None] < num_tokens
    w_mask = w_cols[None, :] < 24
    k_start = split * tiles_per_split * BLOCK_K
    a_offsets = rows[:, None] * K + k_start + a_k[None, :]
    w_offsets = w_cols[None, :] * K + k_start + w_k[:, None]

    acc = gl.zeros([BLOCK_M, 32], gl.float32, mfma_layout)
    square_sum = gl.zeros([BLOCK_M], gl.float32, gl.SliceLayout(1, a_layout))
    # Masked buffer loads return zero, so padded rows and outputs need no
    # select instructions.
    x = cdna4.buffer_load(residual, a_offsets, mask=a_mask)
    w = cdna4.buffer_load(fn, w_offsets, mask=w_mask)
    for tile in range(tiles_per_split):
        # Prefetch the next tile before consuming this one; the final
        # iteration's prefetch is masked off.
        more = tile + 1 < tiles_per_split
        next_offset = (tile + 1) * BLOCK_K
        x_next = cdna4.buffer_load(
            residual, a_offsets + next_offset, mask=a_mask & more
        )
        w_next = cdna4.buffer_load(fn, w_offsets + next_offset, mask=w_mask & more)
        hi, mid, lo = _mhc_mixes_split_weight(w)
        w_shared.index(0).store(lo)
        w_shared.index(1).store(mid)
        w_shared.index(2).store(hi)
        x_fp32 = x.to(gl.float32)
        square_sum += gl.sum(x_fp32 * x_fp32, axis=1)
        for term in gl.static_range(3):
            acc = cdna4.mfma(x, w_shared.index(term).load(b_layout), acc)
        x = x_next
        w = w_next

    out_rows = row_block * BLOCK_M + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, mfma_layout)
    )
    out_cols = gl.arange(0, 32, layout=gl.SliceLayout(0, mfma_layout))
    cdna4.buffer_store(
        acc,
        out_mul,
        (split * num_tokens + out_rows[:, None]) * 24 + out_cols[None, :],
        mask=(out_rows[:, None] < num_tokens) & (out_cols[None, :] < 24),
    )
    cdna4.buffer_store(
        square_sum,
        out_sqrsum,
        split * num_tokens + rows,
        mask=rows < num_tokens,
    )


def _mhc_mixes_reduce_metadata(grid, kernel, args):
    tokens = args["num_tokens"]
    return {
        "name": kernel.name,
        "bytes": tokens * (args["n_splits"] * 25 + 24) * 4,
    }


@gluon.jit
def _reciprocal(x):
    return gl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1", "=v,v", [x], dtype=gl.float32, is_pure=True, pack=1
    )


@gluon.jit(
    launch_metadata=_mhc_mixes_reduce_metadata,
    do_not_specialize=("num_tokens", "n_splits"),
)
def gluon_mhc_mixes_reduce_gfx950(
    projection,
    square_sum,
    hc_scale,
    hc_base,
    pre_mix,
    post_mix,
    comb_mix,
    num_tokens,
    n_splits,
    HIDDEN_SIZE: gl.constexpr,
    RMS_EPS: gl.constexpr,
    HC_EPS: gl.constexpr,
    SINKHORN_ITERS: gl.constexpr,
    SPLIT_REGS: gl.constexpr,
    SPLIT_LANES: gl.constexpr,
    SPLIT_WARPS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    # Sixteen lanes own one 4x4 element each of a token's coefficients; split
    # partials spread over registers, lane groups and waves. After the split
    # sum, the Sinkhorn row and column sums are two-step lane reductions.
    # Prefill keeps splits in registers (no redundant Sinkhorn work); decode
    # has many splits per token and spreads them over lanes and waves.
    TOKEN_LANES: gl.constexpr = 4 // SPLIT_LANES
    TOKEN_WARPS: gl.constexpr = NUM_WARPS // SPLIT_WARPS
    TOKENS: gl.constexpr = TOKEN_LANES * TOKEN_WARPS
    SPLIT_BLOCK: gl.constexpr = SPLIT_REGS * SPLIT_LANES * SPLIT_WARPS
    layout: gl.constexpr = gl.BlockedLayout(
        [SPLIT_REGS, 1, 1, 1],
        [SPLIT_LANES, TOKEN_LANES, 4, 4],
        [SPLIT_WARPS, TOKEN_WARPS, 1, 1],
        [3, 2, 1, 0],
    )
    matrix_layout: gl.constexpr = gl.SliceLayout(0, layout)
    splits = gl.arange(
        0,
        SPLIT_BLOCK,
        layout=gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, layout))),
    )
    tokens = gl.program_id(0) * TOKENS + gl.arange(
        0,
        TOKENS,
        layout=gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, layout))),
    )
    rows = gl.arange(
        0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, layout)))
    )
    cols = gl.arange(
        0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    )
    matrix = rows[None, None, :, None] * 4 + cols[None, None, None, :]
    token_live = (tokens < num_tokens)[None, :, None, None]

    # ``head`` covers projection columns 0..15 (pre, post and two comb rows,
    # only the first two rows are used); ``comb`` covers columns 8..23.
    head = gl.zeros([SPLIT_BLOCK, TOKENS, 4, 4], gl.float32, layout)
    comb = gl.zeros([SPLIT_BLOCK, TOKENS, 4, 4], gl.float32, layout)
    squares = gl.zeros([SPLIT_BLOCK, TOKENS, 4, 4], gl.float32, layout)
    for split_start in range(0, n_splits, SPLIT_BLOCK):
        split = (split_start + splits)[:, None, None, None]
        live = (split < n_splits) & token_live
        row = split * num_tokens + tokens[None, :, None, None]
        head += gl.load(projection + row * 24 + matrix, mask=live, other=0.0)
        comb += gl.load(projection + row * 24 + 8 + matrix, mask=live, other=0.0)
        squares += gl.load(square_sum + row + matrix * 0, mask=live, other=0.0)
    head = gl.sum(head, axis=0)
    comb = gl.sum(comb, axis=0)
    inverse_rms = gl.rsqrt(gl.sum(squares, axis=0) / (4 * HIDDEN_SIZE) + RMS_EPS)

    out_rows = gl.arange(
        0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(2, matrix_layout))
    )[None, :, None]
    out_cols = gl.arange(
        0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, matrix_layout))
    )[None, None, :]
    out_tokens = (
        gl.program_id(0) * TOKENS
        + gl.arange(
            0, TOKENS, layout=gl.SliceLayout(1, gl.SliceLayout(2, matrix_layout))
        )[:, None, None]
    )
    out_matrix = out_rows * 4 + out_cols
    out_live = out_tokens < num_tokens
    head_scale = gl.where(out_rows == 0, gl.load(hc_scale), gl.load(hc_scale + 1))
    head_base = gl.load(hc_base + out_matrix, mask=out_rows < 2, other=0.0)
    head = 1.0 / (1.0 + gl.exp(-(head * inverse_rms * head_scale + head_base)))
    gl.store(
        pre_mix + out_tokens * 4 + out_matrix,
        head + HC_EPS,
        mask=out_live & (out_rows == 0),
    )
    gl.store(
        post_mix + out_tokens * 4 + out_matrix - 4,
        head * 2.0,
        mask=out_live & (out_rows == 1),
    )

    comb = comb * inverse_rms * gl.load(hc_scale + 2) + gl.load(
        hc_base + 8 + out_matrix
    )
    # The Sinkhorn chain is latency-bound on one element per lane; dividing
    # through the hardware reciprocal (~1 ulp) instead of the IEEE division
    # sequence shortens each of its 40 steps and stays far inside the mixes'
    # tolerance.
    row_max = gl.max(comb, axis=2)
    comb = gl.exp(comb - row_max[:, :, None])
    row_sum = gl.sum(comb, axis=2)
    comb = comb * _reciprocal(row_sum)[:, :, None] + HC_EPS
    col_sum = gl.sum(comb, axis=1)
    comb = comb * _reciprocal(col_sum + HC_EPS)[:, None, :]
    for _ in gl.static_range(1, SINKHORN_ITERS):
        row_sum = gl.sum(comb, axis=2)
        comb = comb * _reciprocal(row_sum + HC_EPS)[:, :, None]
        col_sum = gl.sum(comb, axis=1)
        comb = comb * _reciprocal(col_sum + HC_EPS)[:, None, :]
    gl.store(comb_mix + out_tokens * 16 + out_matrix, comb, mask=out_live)


def _mhc_mixes_config(num_tokens: int, k_tiles: int) -> tuple[int, int, int]:
    """Return (block_m, num_warps, n_splits) for a batch.

    Block shapes are two compile-time buckets; the split count only reaches
    runtime arguments. Splits divide the K tiles and give at least
    ``min_ctas`` CTAs. Decode tiles are latency-bound, so small batches spread
    over more CTAs than one wave of the 256 CUs.
    """
    if num_tokens <= 1024:
        block_m, num_warps = _MIXES_SMALL
        min_ctas = 160 if num_tokens <= 256 else 512
    else:
        block_m, num_warps = _MIXES_LARGE
        min_ctas = 256
    row_blocks = triton.cdiv(num_tokens, block_m)
    divisors = [d for d in range(1, k_tiles + 1) if k_tiles % d == 0]
    splits = next((d for d in divisors if row_blocks * d >= min_ctas), k_tiles)
    return block_m, num_warps, splits


def launch_gluon_mhc_mixes_gfx950(
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute hc=4 mHC pre/post/combination coefficients on gfx950.

    Args:
        residual: Contiguous BF16 residual streams ``[T, 4, H]``.
        weight: Contiguous FP32 mixing projection ``[24, 4 * H]``.
        scale: FP32 pre/post/combination scales ``[3]``.
        base: FP32 mixing biases ``[24]``.
        rms_eps: Epsilon of the residual RMS normalization.
        hc_eps: Epsilon of the pre-mix and Sinkhorn normalization.
        sinkhorn_iters: Positive number of Sinkhorn iterations.

    Returns:
        FP32 pre ``[T, 4]``, post ``[T, 4]`` and combination ``[T, 4, 4]``
        indexed [input stream, output stream].
    """
    tokens, hc_mult, hidden_size = residual.shape
    k = 4 * hidden_size
    if hc_mult != 4 or k % _MIXES_BLOCK_K:
        raise ValueError("GFX950 mHC mixes requires [T, 4, H] with 4 * H % 64 == 0")
    if residual.numel() >= 2**31:
        raise ValueError("GFX950 mHC mixes exceeds 32-bit buffer offsets")
    device = residual.device
    pre = torch.empty((tokens, 4), device=device, dtype=torch.float32)
    post = torch.empty_like(pre)
    comb = torch.empty((tokens, 4, 4), device=device, dtype=torch.float32)
    if tokens == 0:
        return pre, post, comb
    k_tiles = k // _MIXES_BLOCK_K
    block_m, num_warps, splits = _mhc_mixes_config(tokens, k_tiles)
    projection = torch.empty((splits, tokens, 24), device=device, dtype=torch.float32)
    square_sum = torch.empty((splits, tokens), device=device, dtype=torch.float32)
    gluon_mhc_mixes_project_gfx950[(splits, triton.cdiv(tokens, block_m))](
        residual,
        weight,
        projection,
        square_sum,
        tokens,
        k_tiles // splits,
        K=k,
        BLOCK_M=block_m,
        BLOCK_K=_MIXES_BLOCK_K,
        NUM_WARPS=num_warps,
        num_warps=num_warps,
    )
    # Splits sit in registers for prefill, and also spread over lanes (and
    # waves when there are few tokens) as the split count grows.
    if splits <= 8:
        split_regs, split_lanes, split_warps = 8, 1, 1
    elif tokens > 256:
        split_regs, split_lanes, split_warps = 8, 4, 1
    else:
        split_regs, split_lanes, split_warps = 8, 4, 4
    tokens_per_cta = (4 // split_lanes) * (_MIXES_REDUCE_WARPS // split_warps)
    gluon_mhc_mixes_reduce_gfx950[(triton.cdiv(tokens, tokens_per_cta),)](
        projection,
        square_sum,
        scale,
        base,
        pre,
        post,
        comb,
        tokens,
        splits,
        HIDDEN_SIZE=hidden_size,
        RMS_EPS=rms_eps,
        HC_EPS=hc_eps,
        SINKHORN_ITERS=sinkhorn_iters,
        SPLIT_REGS=split_regs,
        SPLIT_LANES=split_lanes,
        SPLIT_WARPS=split_warps,
        NUM_WARPS=_MIXES_REDUCE_WARPS,
        num_warps=_MIXES_REDUCE_WARPS,
    )
    return pre, post, comb
