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
from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import _mfma_lds_mediumm_kernel

_INPUT_SIZE = gl.constexpr(7168)
_BLOCK_N_SIZE = 16
_M4_WAVES_PER_EU = 1
_BLOCK_N = gl.constexpr(_BLOCK_N_SIZE)
_BLOCK_K = gl.constexpr(512)
_NUM_WARPS = gl.constexpr(8)
_LANES = gl.constexpr(64)
_PARTIAL_BLOCK = gl.constexpr(8192)


@gluon.jit
def _linear_attnres_partials_kernel(
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
    projection_programs: gl.constexpr = output_size // _BLOCK_N
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
def _linear_attnres_partials_m4_mfma_kernel(
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
):
    """Run an M=4 MFMA projection alongside dual-AttnRes CTAs."""
    pid = gl.program_id(0)
    if pid >= 4:
        _mfma_lds_mediumm_kernel(
            hidden_ptr,
            weight_ptr,
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
            BLOCK_N=_BLOCK_N,
            BLOCK_K=512,
            WARPS_M=1,
            WARPS_N=4,
            NUM_BUFFERS=3,
            GROUP_SIZE_M=1,
            ADD3=False,
            PID_OFFSET=4,
        )
    else:
        _linear_attnres_partials_kernel(
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


def gluon_linear_attnres_partials_gfx950(
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
    if num_tokens not in (1, 2, 4):
        raise ValueError("tokens must be one of 1, 2, or 4")
    output_size = weight.shape[0]
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

    if num_tokens == 4 and output_size == 6288:
        _linear_attnres_partials_m4_mfma_kernel[(output_size // _BLOCK_N_SIZE + 4,)](
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
            num_warps=4,
            num_stages=1,
            waves_per_eu=_M4_WAVES_PER_EU,
            llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),),
        )
        return out

    projection_programs = output_size // _BLOCK_N_SIZE
    attnres_program_offset = (
        192 if num_tokens == 2 and output_size == 6288 else projection_programs
    )
    _linear_attnres_partials_kernel[(projection_programs + num_tokens,)](
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
        num_warps=_NUM_WARPS,
        num_stages=1,
        waves_per_eu=1,
    )
    return out


__all__ = ["gluon_linear_attnres_partials_gfx950"]
