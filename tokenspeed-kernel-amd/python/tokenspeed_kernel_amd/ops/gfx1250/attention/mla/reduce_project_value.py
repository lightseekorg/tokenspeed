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

"""Split-MLA reduction with value projection and optional gate for gfx1250."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

_LANES = gl.constexpr(32)
_BLOCK_N = 8
_NUM_WARPS = 8


@gluon.jit
def _mla_reduce_project_value_kernel(
    split_output_ptr,
    split_max_ptr,
    split_expsum_ptr,
    seq_len_ptr,
    weight_ptr,
    gate_ptr,
    output_ptr,
    LATENT: gl.constexpr,
    VALUE: gl.constexpr,
    HAS_GATE: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    NUM_KV_SPLITS: gl.constexpr,
    TILE_SIZE: gl.constexpr,
):
    """Reduce log2-space split partials into the BF16 value/gate epilogue."""

    pid = gl.program_id(0)
    blocks_per_head: gl.constexpr = VALUE // BLOCK_N
    head = pid // blocks_per_head
    pid_n = pid % blocks_per_head
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

    seq_len = gl.load(seq_len_ptr)
    tiles_per_split = gl.cdiv(seq_len, NUM_KV_SPLITS * TILE_SIZE)
    active_splits = gl.cdiv(seq_len, tiles_per_split * TILE_SIZE)
    e_sum = 0.0
    e_max = -float("inf")
    acc = gl.zeros([LATENT], gl.float32, k_layout)
    for split in range(0, NUM_KV_SPLITS):
        valid = split < active_splits
        split_offset = head * NUM_KV_SPLITS + split
        partial = gl.load(
            split_output_ptr + split_offset * LATENT + offs_k,
            mask=valid,
            other=0.0,
        ).to(gl.float32)
        split_max = gl.load(
            split_max_ptr + split_offset,
            mask=valid,
            other=-float("inf"),
        ).to(gl.float32)
        split_expsum = gl.load(
            split_expsum_ptr + split_offset,
            mask=valid,
            other=0.0,
        ).to(gl.float32)
        next_max = gl.maximum(split_max, e_max)
        old_scale = gl.exp2(e_max - next_max)
        split_scale = gl.exp2(split_max - next_max)
        acc = acc * old_scale + partial * split_scale
        e_sum = e_sum * old_scale + split_expsum * split_scale
        e_max = next_max

    # Preserve both materialized BF16 boundaries: the latent reducer output
    # and the following value projection.
    attention = gl.where(e_sum == 0.0, 0.0, acc / e_sum).to(gl.bfloat16)
    weight = gl.amd.cdna5.buffer_load(
        weight_ptr,
        (
            head * LATENT * VALUE
            + offs_k[None, :].to(gl.int64) * VALUE
            + offs_n[:, None].to(gl.int64)
        ).to(gl.int32),
    ).to(gl.float32)
    attention = gl.convert_layout(attention[None, :], layout)
    projected = gl.sum(weight * attention.to(gl.float32), axis=1)
    projected = projected.to(gl.bfloat16).to(gl.float32)
    if HAS_GATE:
        gate = gl.load(gate_ptr + head * VALUE + offs_n).to(gl.float32)
        projected *= 1.0 / (1.0 + gl.exp(-gate))
    gl.store(
        output_ptr + head * VALUE + offs_n,
        projected.to(output_ptr.dtype.element_ty),
    )


def gluon_mla_reduce_project_value_gfx1250(
    split_output: torch.Tensor,
    split_max: torch.Tensor,
    split_expsum: torch.Tensor,
    cache_seqlens: torch.Tensor,
    weight: torch.Tensor,
    *,
    gate: torch.Tensor | None = None,
    page_size: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Reduce split attention and emit projected, optionally gated BF16 values."""

    num_splits = split_output.shape[2]
    heads, latent, value = weight.shape
    expected = (
        (
            split_output,
            (1, heads, num_splits, latent),
            torch.float32,
            "split output",
        ),
        (split_max, (1, heads, num_splits), torch.float32, "split maxima"),
        (
            split_expsum,
            (1, heads, num_splits),
            torch.float32,
            "split exponent sums",
        ),
        (cache_seqlens, (1,), torch.int32, "sequence lengths"),
        (weight, (heads, latent, value), torch.bfloat16, "value weight"),
    )
    if gate is not None:
        expected += ((gate, (1, heads * value), torch.bfloat16, "gate"),)
    if page_size != 64:
        raise ValueError("MLA projected-value reducer requires page size 64")
    for tensor, shape, dtype, name in expected:
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or not tensor.is_cuda
            or not tensor.is_contiguous()
            or tensor.device != split_output.device
        ):
            raise ValueError(
                f"MLA projected-value reducer requires contiguous colocated {name} "
                f"{shape} {dtype}"
            )
    if (
        tuple(out.shape) != (1, heads * value)
        or out.dtype != torch.bfloat16
        or not out.is_cuda
        or not out.is_contiguous()
        or out.device != split_output.device
    ):
        raise ValueError(
            "MLA projected-value reducer requires contiguous colocated out "
            f"{(1, heads * value)} {torch.bfloat16}"
        )

    gate_tensor = split_output if gate is None else gate
    _mla_reduce_project_value_kernel[(heads * value // _BLOCK_N,)](
        split_output,
        split_max,
        split_expsum,
        cache_seqlens,
        weight,
        gate_tensor,
        out,
        LATENT=latent,
        VALUE=value,
        HAS_GATE=gate is not None,
        BLOCK_N=_BLOCK_N,
        NUM_WARPS=_NUM_WARPS,
        NUM_KV_SPLITS=num_splits,
        TILE_SIZE=page_size,
        num_warps=_NUM_WARPS,
        num_stages=1,
        waves_per_eu=0,
    )
    return out


__all__ = ["gluon_mla_reduce_project_value_gfx1250"]
