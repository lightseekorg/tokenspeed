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

"""Absorbed MLA query projection and optional RoPE assembly for gfx950."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

_LANES = gl.constexpr(64)
_BLOCK_N = 32
_NUM_WARPS = 4


@gluon.jit
def _mla_absorb_query_kernel(
    query_nope_ptr,
    weight_ptr,
    query_rope_ptr,
    output_ptr,
    query_nope_head_stride,
    query_rope_head_stride,
    output_head_stride,
    NOPE: gl.constexpr,
    ROPE: gl.constexpr,
    LATENT: gl.constexpr,
    HAS_ROPE: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    pid = gl.program_id(0)
    num_pid_n: gl.constexpr = LATENT // BLOCK_N
    head = pid // num_pid_n
    pid_n = pid % num_pid_n
    layout: gl.constexpr = gl.BlockedLayout(
        [(BLOCK_N + NUM_WARPS - 1) // NUM_WARPS, NOPE // _LANES],
        [1, _LANES],
        [NUM_WARPS, 1],
        [1, 0],
    )
    n_layout: gl.constexpr = gl.SliceLayout(1, layout)
    k_layout: gl.constexpr = gl.SliceLayout(0, layout)
    offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=n_layout)
    offs_k = gl.arange(0, NOPE, layout=k_layout)
    query = gl.amd.cdna4.buffer_load(
        query_nope_ptr,
        (head * query_nope_head_stride + offs_k).to(gl.int32),
    ).to(gl.float32)
    weight = gl.amd.cdna4.buffer_load(
        weight_ptr,
        (
            head * NOPE * LATENT
            + offs_k[None, :].to(gl.int64) * LATENT
            + offs_n[:, None].to(gl.int64)
        ).to(gl.int32),
    )
    query = gl.convert_layout(query[None, :], layout)
    projected = gl.sum(weight.to(gl.float32) * query, axis=1)
    gl.store(
        output_ptr + head * output_head_stride + offs_n,
        projected.to(output_ptr.dtype.element_ty),
    )
    if HAS_ROPE and pid_n < gl.cdiv(ROPE, BLOCK_N):
        gl.store(
            output_ptr + head * output_head_stride + LATENT + offs_n,
            gl.load(
                query_rope_ptr + head * query_rope_head_stride + offs_n,
                mask=offs_n < ROPE,
            ),
            mask=offs_n < ROPE,
        )


def gluon_mla_absorb_query_gfx950(
    query_nope: torch.Tensor,
    weight: torch.Tensor,
    *,
    query_rope: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project one token of per-head BF16 MLA query latents."""

    heads, nope, latent = weight.shape
    if tuple(query_nope.shape) != (1, heads, nope):
        raise ValueError("MLA absorbed query requires query_nope [1, heads, nope]")
    rope = 0 if query_rope is None else int(query_rope.shape[-1])
    if query_rope is not None and tuple(query_rope.shape[:2]) != (1, heads):
        raise ValueError("MLA absorbed query requires query_rope [1, heads, rope]")
    expected_output = (1, heads, latent + rope)
    tensors = (query_nope, weight)
    if query_rope is not None:
        tensors += (query_rope,)
    if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
        raise TypeError("MLA absorbed query requires BF16 inputs")
    if any(
        not tensor.is_cuda or tensor.device != query_nope.device for tensor in tensors
    ):
        raise ValueError("MLA absorbed query requires colocated GPU tensors")
    if query_nope.stride(-1) != 1:
        raise ValueError("MLA absorbed query requires unit-stride query_nope")
    if query_rope is not None and query_rope.stride(-1) != 1:
        raise ValueError("MLA absorbed query requires unit-stride query_rope")
    if not weight.is_contiguous():
        raise ValueError("MLA absorbed query requires contiguous weight")
    if out is None:
        out = query_nope.new_empty(expected_output)
    elif (
        tuple(out.shape) != expected_output
        or out.dtype != torch.bfloat16
        or out.device != query_nope.device
        or out.stride(-1) != 1
    ):
        raise ValueError("MLA absorbed query out must be unit-stride BF16")

    rope_tensor = query_nope if query_rope is None else query_rope
    _mla_absorb_query_kernel[(heads * latent // _BLOCK_N,)](
        query_nope,
        weight,
        rope_tensor,
        out,
        query_nope.stride(1),
        rope_tensor.stride(1),
        out.stride(1),
        NOPE=nope,
        ROPE=rope,
        LATENT=latent,
        HAS_ROPE=query_rope is not None,
        BLOCK_N=_BLOCK_N,
        NUM_WARPS=_NUM_WARPS,
        num_warps=_NUM_WARPS,
        num_stages=1,
        waves_per_eu=0,
    )
    return out


__all__ = ["gluon_mla_absorb_query_gfx950"]
