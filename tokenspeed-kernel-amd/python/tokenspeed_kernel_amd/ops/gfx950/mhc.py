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
from tokenspeed_kernel_amd._triton import gl, gluon, tl

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
    matrix_layout: gl.constexpr = gl.BlockedLayout([1, 1], [4, 16], [1, 1], [1, 0])
    rows = gl.arange(0, 4, layout=gl.SliceLayout(1, matrix_layout))
    cols = gl.arange(0, 16, layout=gl.SliceLayout(0, matrix_layout))
    active = cols[None, :] < 4
    comb_offsets = rows[:, None] * 4 + cols[None, :]
    comb_values = gl.zeros([4, 16], gl.float32, layout=matrix_layout)
    rms_sum = 0.0
    # Keep batch-dependent split counts out of the compilation key and retain
    # the sequential partial-sum order used by the portable implementation.
    for split in range(n_splits):
        split_base = split * num_tokens * 24 + token * 24
        pre_post_values += gl.load(projection + split_base + pre_post_offsets)
        comb_values += gl.load(
            projection + split_base + 8 + comb_offsets,
            mask=active,
            other=0.0,
        )
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
        hc_base + 8 + comb_offsets, mask=active, other=0.0
    )
    row_max = gl.max(gl.where(active, comb, -float("inf")), axis=1)
    comb = gl.where(active, gl.exp(comb - row_max[:, None]), 0.0)
    row_sum = gl.sum(comb, axis=1)
    comb = gl.where(active, comb / row_sum[:, None] + HC_EPS, 0.0)
    col_sum = gl.sum(comb, axis=0)
    comb = gl.where(active, comb / (col_sum[None, :] + HC_EPS), 0.0)
    for _ in gl.static_range(1, SINKHORN_ITERS):
        row_sum = gl.sum(comb, axis=1)
        comb = gl.where(active, comb / (row_sum[:, None] + HC_EPS), 0.0)
        col_sum = gl.sum(comb, axis=0)
        comb = gl.where(active, comb / (col_sum[None, :] + HC_EPS), 0.0)
    gl.store(comb_mix + token * 16 + comb_offsets, comb, mask=active)


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
