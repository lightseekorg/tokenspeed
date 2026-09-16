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

"""Absorbed MLA value projection and optional output gate for gfx1250."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

_LANES = gl.constexpr(32)
_BLOCK_N = 32
_NUM_WARPS = 4


@gluon.jit
def _mla_project_value_kernel(
    attention_ptr,
    weight_ptr,
    gate_ptr,
    output_ptr,
    HEADS: gl.constexpr,
    LATENT: gl.constexpr,
    VALUE: gl.constexpr,
    GATE_STRIDE_B: gl.constexpr,
    GATE_STRIDE_N: gl.constexpr,
    HAS_GATE: gl.constexpr,
    BATCHED: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    pid = gl.program_id(0)
    num_pid_n: gl.constexpr = VALUE // BLOCK_N
    batch_head = pid // num_pid_n
    if BATCHED:
        batch = batch_head // HEADS
        head = batch_head % HEADS
    else:
        batch = 0
        head = batch_head
    pid_n = pid % num_pid_n
    layout: gl.constexpr = gl.BlockedLayout(
        [(BLOCK_N + NUM_WARPS - 1) // NUM_WARPS, LATENT // _LANES],
        [1, _LANES],
        [NUM_WARPS, 1],
        [1, 0],
    )
    n_layout: gl.constexpr = gl.SliceLayout(1, layout)
    k_layout: gl.constexpr = gl.SliceLayout(0, layout)
    offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=n_layout)
    offs_k = gl.arange(0, LATENT, layout=k_layout)
    attention = gl.amd.cdna5.buffer_load(
        attention_ptr,
        (batch_head * LATENT + offs_k).to(gl.int32),
    ).to(gl.float32)
    weight = gl.amd.cdna5.buffer_load(
        weight_ptr,
        (
            head * LATENT * VALUE
            + offs_k[None, :].to(gl.int64) * VALUE
            + offs_n[:, None].to(gl.int64)
        ).to(gl.int32),
    )
    attention = gl.convert_layout(attention[None, :], layout)
    projected = gl.sum(weight.to(gl.float32) * attention, axis=1)
    projected = projected.to(gl.bfloat16).to(gl.float32)
    if HAS_GATE:
        gate = gl.load(
            gate_ptr + batch * GATE_STRIDE_B + (head * VALUE + offs_n) * GATE_STRIDE_N
        ).to(gl.float32)
        projected *= 1.0 / (1.0 + gl.exp(-gate))
    gl.store(
        output_ptr + batch_head * VALUE + offs_n,
        projected.to(output_ptr.dtype.element_ty),
    )


def gluon_mla_project_value_gfx1250(
    attention: torch.Tensor,
    weight: torch.Tensor,
    *,
    gate: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project BF16 latent rows per head and optionally apply a sigmoid gate.

    Args:
        attention: Contiguous BF16 latent values shaped ``[batch, heads, latent]``.
        weight: Contiguous BF16 projection weights shaped
            ``[heads, latent, value]``.
        gate: Optional BF16 sigmoid gate shaped ``[batch, heads * value]``
            with a contiguous inner dimension.
        out: Optional contiguous BF16 output shaped ``[batch, heads * value]``.

    Returns:
        The projected values in ``out`` when supplied, otherwise a new tensor.
    """

    heads, latent, value = weight.shape
    batch = attention.shape[0]
    expected_attention = (batch, heads, latent)
    expected_output = (batch, heads * value)
    for tensor, shape, name in (
        (attention, expected_attention, "attention"),
        (weight, (heads, latent, value), "weight"),
    ):
        if tuple(tensor.shape) != shape:
            raise ValueError(f"MLA value projection requires {name} {shape}")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"MLA value projection requires BF16 {name}")
        if (
            not tensor.is_cuda
            or not tensor.is_contiguous()
            or tensor.device != attention.device
        ):
            raise ValueError(
                f"MLA value projection requires contiguous colocated {name}"
            )
    if gate is not None:
        if gate.shape != expected_output or gate.dtype != torch.bfloat16:
            raise ValueError(
                f"MLA value projection requires BF16 gate {expected_output}"
            )
        if not gate.is_cuda or gate.device != attention.device or gate.stride(1) != 1:
            raise ValueError(
                "MLA value projection requires a colocated gate with "
                "contiguous inner dimension"
            )
    if out is None:
        out = attention.new_empty(expected_output)
    elif (
        tuple(out.shape) != expected_output
        or out.dtype != torch.bfloat16
        or out.device != attention.device
        or not out.is_contiguous()
    ):
        raise ValueError("MLA value projection out must be contiguous BF16")

    gate_tensor = attention if gate is None else gate
    _mla_project_value_kernel[(batch * heads * value // _BLOCK_N,)](
        attention,
        weight,
        gate_tensor,
        out,
        HEADS=heads,
        LATENT=latent,
        VALUE=value,
        GATE_STRIDE_B=0 if gate is None else gate.stride(0),
        GATE_STRIDE_N=0 if gate is None else gate.stride(1),
        HAS_GATE=gate is not None,
        BATCHED=batch > 1,
        BLOCK_N=_BLOCK_N,
        NUM_WARPS=_NUM_WARPS,
        num_warps=_NUM_WARPS,
        num_stages=1,
        waves_per_eu=0,
    )
    return out


__all__ = ["gluon_mla_project_value_gfx1250"]
