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

"""Linear projection and dual-AttnRes partials for gfx950."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._extern import extern_compile_options
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16 import gemv
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (
    gluon_mm_a16w16_medium_gfx950,
)

_INPUT_SIZE = gl.constexpr(7168)
_BLOCK_N_SIZE = 16
# Four-token tile width: the narrowest whose CTAs fit one wave of 256 CUs, one per CU.
_M4_BLOCK_N_SIZES = {3648: 16, 6288: 32}
_BLOCK_N = gl.constexpr(_BLOCK_N_SIZE)
_BLOCK_K = gl.constexpr(512)
_NUM_WARPS = gl.constexpr(8)
_LANES = gl.constexpr(64)
_PARTIAL_BLOCK = gl.constexpr(8192)
# Tokens the GEMV-projection launch takes: the decode GEMV's MFMA tile.
_GEMV_TOKENS = range(5, gemv.MAX_M + 1)
# The K3 MLA q_a/kv_a/gate and KDA qkvfab projections.
_GEMV_OUTPUT_SIZES = frozenset({3648, 6288})
# Half the hidden size, padded to a power of two: 8 waves x 64 lanes x 8 elements.
_HALF = gl.constexpr(4096)
_HALF_WAVES = gl.constexpr(8)
_HALF_ELEMENTS = gl.constexpr(8)


@gluon.jit
def _half_offsets(start):
    """``start`` plus a ``[8, 64, 8]`` (wave, lane, element) half-row index.

    Four waves hold the eight Triton waves, wave ``w`` holding ``w`` and
    ``w + 4``, so in-thread, lane and cross-wave sums pair the same values
    as Triton's 8-wave butterfly.
    """
    gl.static_assert(gl.num_warps() * 2 == _HALF_WAVES)
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, _HALF_ELEMENTS], [1, _LANES, 1], [gl.num_warps(), 1, 1], [2, 1, 0]
    )
    waves = gl.arange(0, _HALF_WAVES, gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    lanes = gl.arange(0, _LANES, gl.SliceLayout(0, gl.SliceLayout(2, layout)))
    elements = gl.arange(
        0, _HALF_ELEMENTS, gl.SliceLayout(0, gl.SliceLayout(1, layout))
    )
    return (
        start
        + waves[:, None, None] * (_LANES * _HALF_ELEMENTS)
        + lanes[None, :, None] * _HALF_ELEMENTS
        + elements[None, None, :]
    )


@gluon.jit
def _half_sum(values):
    return gl.sum(gl.sum(gl.sum(values, axis=2), axis=1), axis=0)


@gluon.jit
def _dual_partial_triton_order(
    token,
    blocks_ptr,
    score_weight_a_ptr,
    score_weight_b_ptr,
    max_a_ptr,
    sum_a_ptr,
    accumulator_a_ptr,
    max_b_ptr,
    sum_b_ptr,
    accumulator_b_ptr,
    num_blocks,
    block_stride,
    token_stride,
    eps,
):
    """One token's two partials, bitwise with ``attnres_partial_dual``."""
    col0 = _half_offsets(0)
    col1 = _half_offsets(_HALF)
    mask0 = col0 < _INPUT_SIZE
    mask1 = col1 < _INPUT_SIZE
    wa0 = gl.load(score_weight_a_ptr + col0, mask=mask0, other=0.0).to(gl.float32)
    wa1 = gl.load(score_weight_a_ptr + col1, mask=mask1, other=0.0).to(gl.float32)
    wb0 = gl.load(score_weight_b_ptr + col0, mask=mask0, other=0.0).to(gl.float32)
    wb1 = gl.load(score_weight_b_ptr + col1, mask=mask1, other=0.0).to(gl.float32)
    acc_a0 = gl.zeros_like(wa0)
    acc_a1 = gl.zeros_like(wa0)
    acc_b0 = gl.zeros_like(wa0)
    acc_b1 = gl.zeros_like(wa0)
    m_a = gl.full((), -float("inf"), gl.float32)
    s_a = gl.full((), 0.0, gl.float32)
    m_b = gl.full((), -float("inf"), gl.float32)
    s_b = gl.full((), 0.0, gl.float32)
    b = 0
    while b < num_blocks:
        base = blocks_ptr + b * block_stride + token * token_stride
        v0 = gl.load(base + col0, mask=mask0, other=0.0).to(gl.float32)
        v1 = gl.load(base + col1, mask=mask1, other=0.0).to(gl.float32)
        sq = _half_sum(v0 * v0) + _half_sum(v1 * v1)
        rsig = gl.rsqrt(sq / _INPUT_SIZE + eps)
        dot_a = _half_sum(v0 * wa0) + _half_sum(v1 * wa1)
        logit_a = dot_a * rsig
        m_an = gl.maximum(m_a, logit_a)
        corr_a = gl.exp(m_a - m_an)
        wgt_a = gl.exp(logit_a - m_an)
        acc_a0 = acc_a0 * corr_a + wgt_a * v0
        acc_a1 = acc_a1 * corr_a + wgt_a * v1
        s_a = s_a * corr_a + wgt_a
        m_a = m_an
        dot_b = _half_sum(v0 * wb0) + _half_sum(v1 * wb1)
        logit_b = dot_b * rsig
        m_bn = gl.maximum(m_b, logit_b)
        corr_b = gl.exp(m_b - m_bn)
        wgt_b = gl.exp(logit_b - m_bn)
        acc_b0 = acc_b0 * corr_b + wgt_b * v0
        acc_b1 = acc_b1 * corr_b + wgt_b * v1
        s_b = s_b * corr_b + wgt_b
        m_b = m_bn
        b += 1
    row = token * _INPUT_SIZE
    gl.store(accumulator_a_ptr + row + col0, acc_a0, mask=mask0)
    gl.store(accumulator_a_ptr + row + col1, acc_a1, mask=mask1)
    gl.store(max_a_ptr + token, m_a)
    gl.store(sum_a_ptr + token, s_a)
    gl.store(accumulator_b_ptr + row + col0, acc_b0, mask=mask0)
    gl.store(accumulator_b_ptr + row + col1, acc_b1, mask=mask1)
    gl.store(max_b_ptr + token, m_b)
    gl.store(sum_b_ptr + token, s_b)


@gluon.jit(do_not_specialize=("num_tokens",))
def gluon_linear_attnres_partials_gemv_gfx950(
    hidden_ptr,
    weight_ptr,
    output_ptr,
    blocks_ptr,
    score_weight_a_ptr,
    score_weight_b_ptr,
    max_a_ptr,
    sum_a_ptr,
    accumulator_a_ptr,
    max_b_ptr,
    sum_b_ptr,
    accumulator_b_ptr,
    num_blocks,
    block_stride,
    eps,
    num_tokens,
    rows,
    output_size: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
):
    """One dual-AttnRes CTA per token, then the decode GEMV's MFMA tiles.

    Both match their standalone kernels bit for bit: the tiles are
    ``gluon_gemv_a16w16_gfx950``'s, and the partials follow
    ``attnres_partial_dual``'s summation order.
    """
    pid = gl.program_id(0)
    if pid < num_tokens:
        _dual_partial_triton_order(
            pid,
            blocks_ptr,
            score_weight_a_ptr,
            score_weight_b_ptr,
            max_a_ptr,
            sum_a_ptr,
            accumulator_a_ptr,
            max_b_ptr,
            sum_b_ptr,
            accumulator_b_ptr,
            num_blocks,
            block_stride,
            _INPUT_SIZE,
            eps,
        )
    else:
        gemv._mfma_tile(
            pid - num_tokens,
            hidden_ptr,
            weight_ptr,
            s_ptr=weight_ptr,
            y_ptr=output_ptr,
            M=num_tokens,
            N=output_size,
            rows=rows,
            stride_xm=_INPUT_SIZE,
            stride_ym=output_size,
            K=_INPUT_SIZE,
            BLOCK_M=16,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            W_FP8=False,
        )


@gluon.jit
def gluon_linear_attnres_partials_gfx950(
    hidden_ptr,
    weight_ptr,
    output_ptr,
    blocks_ptr,
    score_weight_a_ptr,
    score_weight_b_ptr,
    max_a_ptr,
    sum_a_ptr,
    accumulator_a_ptr,
    max_b_ptr,
    sum_b_ptr,
    accumulator_b_ptr,
    num_blocks,
    block_stride,
    eps,
    num_tokens: gl.constexpr,
    attnres_program_offset: gl.constexpr,
    output_size: gl.constexpr,
):
    """Run shared-weight projection CTAs and one dual-AttnRes CTA per token."""
    pid = gl.program_id(0)
    is_attnres = (pid >= attnres_program_offset) & (
        pid < attnres_program_offset + num_tokens
    )
    projection_pid = pid if pid < attnres_program_offset else pid - num_tokens
    if not is_attnres:
        layout: gl.constexpr = gl.BlockedLayout(
            [1, _BLOCK_K // _LANES],
            [1, _LANES],
            [gl.num_warps(), 1],
            [1, 0],
        )
        output_layout: gl.constexpr = gl.SliceLayout(1, layout)
        input_layout: gl.constexpr = gl.SliceLayout(0, layout)
        output_offsets = projection_pid * _BLOCK_N + gl.arange(
            0, _BLOCK_N, layout=output_layout
        )
        accumulator_0 = gl.zeros([_BLOCK_N], gl.float32, output_layout)
        if num_tokens >= 2:
            accumulator_1 = gl.zeros([_BLOCK_N], gl.float32, output_layout)
        if num_tokens == 4:
            accumulator_2 = gl.zeros([_BLOCK_N], gl.float32, output_layout)
            accumulator_3 = gl.zeros([_BLOCK_N], gl.float32, output_layout)
        for input_start in range(0, _INPUT_SIZE, _BLOCK_K):
            input_offsets = input_start + gl.arange(0, _BLOCK_K, layout=input_layout)
            hidden_0 = gl.amd.cdna4.buffer_load(
                hidden_ptr,
                input_offsets.to(gl.int32),
            ).to(gl.float32)
            weight = gl.amd.cdna4.buffer_load(
                weight_ptr,
                (
                    output_offsets[:, None].to(gl.int64) * _INPUT_SIZE
                    + input_offsets[None, :].to(gl.int64)
                ).to(gl.int32),
                cache=".cg",
            ).to(gl.float32)
            hidden_0 = gl.convert_layout(hidden_0[None, :], layout)
            accumulator_0 += gl.sum(weight * hidden_0, axis=1)
            if num_tokens >= 2:
                hidden_1 = gl.amd.cdna4.buffer_load(
                    hidden_ptr,
                    (_INPUT_SIZE + input_offsets).to(gl.int32),
                ).to(gl.float32)
                hidden_1 = gl.convert_layout(hidden_1[None, :], layout)
                accumulator_1 += gl.sum(weight * hidden_1, axis=1)
            if num_tokens == 4:
                hidden_2 = gl.amd.cdna4.buffer_load(
                    hidden_ptr,
                    (2 * _INPUT_SIZE + input_offsets).to(gl.int32),
                ).to(gl.float32)
                hidden_3 = gl.amd.cdna4.buffer_load(
                    hidden_ptr,
                    (3 * _INPUT_SIZE + input_offsets).to(gl.int32),
                ).to(gl.float32)
                hidden_2 = gl.convert_layout(hidden_2[None, :], layout)
                hidden_3 = gl.convert_layout(hidden_3[None, :], layout)
                accumulator_2 += gl.sum(weight * hidden_2, axis=1)
                accumulator_3 += gl.sum(weight * hidden_3, axis=1)
        gl.store(output_ptr + output_offsets, accumulator_0)
        if num_tokens >= 2:
            gl.store(
                output_ptr + output_size + output_offsets,
                accumulator_1,
            )
        if num_tokens == 4:
            gl.store(
                output_ptr + 2 * output_size + output_offsets,
                accumulator_2,
            )
            gl.store(
                output_ptr + 3 * output_size + output_offsets,
                accumulator_3,
            )
        return

    token = pid - attnres_program_offset
    partial_layout: gl.constexpr = gl.BlockedLayout(
        [_PARTIAL_BLOCK // (_LANES * gl.num_warps())],
        [_LANES],
        [gl.num_warps()],
        [0],
    )
    offsets = gl.arange(0, _PARTIAL_BLOCK, layout=partial_layout)
    mask = offsets < _INPUT_SIZE
    score_weight_a = gl.amd.cdna4.buffer_load(
        score_weight_a_ptr,
        offsets.to(gl.int32),
        mask=mask,
        other=0.0,
    ).to(gl.float32)
    score_weight_b = gl.amd.cdna4.buffer_load(
        score_weight_b_ptr,
        offsets.to(gl.int32),
        mask=mask,
        other=0.0,
    ).to(gl.float32)
    accumulator_a = gl.zeros([_PARTIAL_BLOCK], gl.float32, partial_layout)
    accumulator_b = gl.zeros([_PARTIAL_BLOCK], gl.float32, partial_layout)
    max_a = gl.full((), -float("inf"), gl.float32)
    sum_a = gl.full((), 0.0, gl.float32)
    max_b = gl.full((), -float("inf"), gl.float32)
    sum_b = gl.full((), 0.0, gl.float32)
    block = 0
    while block < num_blocks:
        block_offset = block.to(gl.int64) * block_stride + token * _INPUT_SIZE
        value = gl.amd.cdna4.buffer_load(
            blocks_ptr,
            (block_offset + offsets).to(gl.int32),
            mask=mask,
            other=0.0,
        ).to(gl.float32)
        inverse_rms = gl.rsqrt(gl.sum(value * value, axis=0) / _INPUT_SIZE + eps)

        logit_a = gl.sum(value * score_weight_a, axis=0) * inverse_rms
        next_max_a = gl.maximum(max_a, logit_a)
        correction_a = gl.exp(max_a - next_max_a)
        weight_a = gl.exp(logit_a - next_max_a)
        accumulator_a = accumulator_a * correction_a + value * weight_a
        sum_a = sum_a * correction_a + weight_a
        max_a = next_max_a

        logit_b = gl.sum(value * score_weight_b, axis=0) * inverse_rms
        next_max_b = gl.maximum(max_b, logit_b)
        correction_b = gl.exp(max_b - next_max_b)
        weight_b = gl.exp(logit_b - next_max_b)
        accumulator_b = accumulator_b * correction_b + value * weight_b
        sum_b = sum_b * correction_b + weight_b
        max_b = next_max_b
        block += 1

    gl.amd.cdna4.buffer_store(
        accumulator_a,
        accumulator_a_ptr,
        (token * _INPUT_SIZE + offsets).to(gl.int32),
        mask=mask,
    )
    gl.amd.cdna4.buffer_store(
        accumulator_b,
        accumulator_b_ptr,
        (token * _INPUT_SIZE + offsets).to(gl.int32),
        mask=mask,
    )
    gl.store(max_a_ptr + token, max_a)
    gl.store(sum_a_ptr + token, sum_a)
    gl.store(max_b_ptr + token, max_b)
    gl.store(sum_b_ptr + token, sum_b)


@gluon.jit
def gluon_linear_attnres_partials_m4_gfx950(
    hidden_ptr,
    weight_ptr,
    output_ptr,
    blocks_ptr,
    score_weight_a_ptr,
    score_weight_b_ptr,
    max_a_ptr,
    sum_a_ptr,
    accumulator_a_ptr,
    max_b_ptr,
    sum_b_ptr,
    accumulator_b_ptr,
    num_blocks,
    block_stride,
    eps,
    output_size: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    """Run an M=4 MFMA projection alongside dual-AttnRes CTAs."""
    pid = gl.program_id(0)
    if pid >= 4:
        gluon_mm_a16w16_medium_gfx950(
            hidden_ptr,
            weight_ptr,
            output_ptr,
            output_ptr,
            output_ptr,
            output_ptr,
            output_ptr,
            4,
            output_size,
            _INPUT_SIZE,
            _INPUT_SIZE,
            1,
            1,
            _INPUT_SIZE,
            output_size,
            1,
            output_size,
            1,
            output_size,
            1,
            BLOCK_M=16,
            BLOCK_N=BLOCK_N,
            BLOCK_K=512,
            WARPS_M=1,
            WARPS_N=4,
            NUM_BUFFERS=3,
            GROUP_SIZE_M=1,
            ADD3=False,
            SPLIT_K=1,
            NUM_XCDS=1,
            PID_OFFSET=4,
        )
    else:
        gluon_linear_attnres_partials_gfx950(
            hidden_ptr,
            weight_ptr,
            output_ptr,
            blocks_ptr,
            score_weight_a_ptr,
            score_weight_b_ptr,
            max_a_ptr,
            sum_a_ptr,
            accumulator_a_ptr,
            max_b_ptr,
            sum_b_ptr,
            accumulator_b_ptr,
            num_blocks,
            block_stride,
            eps,
            4,
            0,
            output_size,
        )


def launch_gluon_linear_attnres_partials_gfx950(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    blocks: torch.Tensor,
    score_weight_a: torch.Tensor,
    score_weight_b: torch.Tensor,
    scratch_a: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    scratch_b: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    eps: float,
    out: torch.Tensor,
) -> torch.Tensor:
    """Run a decode projection and two AttnRes block reductions."""
    num_tokens = hidden_states.shape[0]
    output_size = weight.shape[0]
    gemv_projection = num_tokens in _GEMV_TOKENS
    if gemv_projection and output_size not in _GEMV_OUTPUT_SIZES:
        raise ValueError(
            f"{num_tokens} tokens need an output size in {sorted(_GEMV_OUTPUT_SIZES)}"
        )
    if num_tokens not in (1, 2, 4) and not gemv_projection:
        raise ValueError("tokens must be 1, 2, 4 or 5-16")
    if output_size % _BLOCK_N_SIZE != 0:
        raise ValueError(f"output size must be divisible by {_BLOCK_N_SIZE}")
    expected = (
        (hidden_states, (num_tokens, 7168), "hidden states"),
        (weight, (output_size, 7168), "weight"),
        (score_weight_a, (7168,), "first score weight"),
        (score_weight_b, (7168,), "second score weight"),
        (out, (num_tokens, output_size), "output"),
    )
    for tensor, shape, name in expected:
        if tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be contiguous BF16 {shape}")
        if (
            not tensor.is_cuda
            or not tensor.is_contiguous()
            or tensor.device != hidden_states.device
        ):
            raise ValueError(f"{name} must be contiguous and colocated")
    if (
        blocks.ndim != 3
        or not 1 <= blocks.shape[0] <= 11
        or tuple(blocks.shape[1:]) != (num_tokens, 7168)
        or blocks.dtype != torch.bfloat16
        or not blocks.is_cuda
        or not blocks.is_contiguous()
        or blocks.device != hidden_states.device
    ):
        raise ValueError("residual blocks must be contiguous BF16 [KB,tokens,7168]")
    expected_scratch = (
        (num_tokens,),
        (num_tokens,),
        (num_tokens, 7168),
    )
    for scratch in (scratch_a, scratch_b):
        if len(scratch) != 3:
            raise ValueError("scratch must be an (max, sum, accumulator) tuple")
        for tensor, shape in zip(scratch, expected_scratch, strict=True):
            if (
                tuple(tensor.shape) != shape
                or tensor.dtype != torch.float32
                or not tensor.is_cuda
                or not tensor.is_contiguous()
                or tensor.device != hidden_states.device
            ):
                raise ValueError(
                    f"scratch must contain contiguous FP32 {expected_scratch}"
                )
    if eps <= 0.0:
        raise ValueError("AttnRes epsilon must be positive")

    if gemv_projection:
        cus = torch.cuda.get_device_properties(
            hidden_states.device
        ).multi_processor_count
        _, _, block_n, block_k, _, num_warps = gemv._config(
            num_tokens, output_size, 7168, cus
        )
        # Wider tiles leave a CU per AttnRes CTA; row count never changes the sum order.
        rows = triton.cdiv(output_size, cus - num_tokens)
        if triton.next_power_of_2(rows) != block_n:
            raise ValueError(f"no single-wave GEMV tiling for {output_size} columns")
        gluon_linear_attnres_partials_gemv_gfx950[
            (num_tokens + triton.cdiv(output_size, rows),)
        ](
            hidden_states,
            weight,
            out,
            blocks,
            score_weight_a,
            score_weight_b,
            *scratch_a,
            *scratch_b,
            blocks.shape[0],
            blocks.stride(0),
            float(eps),
            num_tokens,
            rows,
            output_size,
            block_n,
            block_k,
            num_warps=num_warps,
        )
        return out

    m4_block_n = _M4_BLOCK_N_SIZES.get(output_size) if num_tokens == 4 else None
    if m4_block_n is not None:
        m4_programs = triton.cdiv(output_size, m4_block_n) + 4
        gluon_linear_attnres_partials_m4_gfx950[(m4_programs,)](
            hidden_states,
            weight,
            out,
            blocks,
            score_weight_a,
            score_weight_b,
            *scratch_a,
            *scratch_b,
            blocks.shape[0],
            blocks.stride(0),
            float(eps),
            output_size,
            m4_block_n,
            num_warps=4,
            num_stages=1,
            waves_per_eu=1,
            llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),),
            **extern_compile_options(),
        )
        return out

    projection_programs = output_size // _BLOCK_N_SIZE
    attnres_program_offset = (
        192 if num_tokens == 2 and output_size == 6288 else projection_programs
    )
    gluon_linear_attnres_partials_gfx950[(projection_programs + num_tokens,)](
        hidden_states,
        weight,
        out,
        blocks,
        score_weight_a,
        score_weight_b,
        *scratch_a,
        *scratch_b,
        blocks.shape[0],
        blocks.stride(0),
        float(eps),
        num_tokens,
        attnres_program_offset,
        output_size,
        num_warps=_NUM_WARPS.value,
        num_stages=1,
        waves_per_eu=1,
    )
    return out


__all__ = ["launch_gluon_linear_attnres_partials_gfx950"]
