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
_BLOCK_N = 32


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
    HEADS: gl.constexpr,
    GATE_STRIDE_B: gl.constexpr,
    GATE_STRIDE_N: gl.constexpr,
    HAS_GATE: gl.constexpr,
    BATCHED: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    NUM_KV_SPLITS: gl.constexpr,
    TILE_SIZE: gl.constexpr,
):
    """Reduce log2-space split partials into the BF16 value/gate epilogue."""

    pid = gl.program_id(0)
    blocks_per_head: gl.constexpr = VALUE // BLOCK_N
    batch_head = pid // blocks_per_head
    if BATCHED:
        batch = batch_head // HEADS
        head = batch_head % HEADS
    else:
        batch = 0
        head = batch_head
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

    # Match the standalone decode reducer's split reduction layout and order.
    # This preserves its materialized BF16 latent boundary before projection.
    tpw_k: gl.constexpr = gl.constexpr(min(_LANES, LATENT))
    wpc_k: gl.constexpr = gl.constexpr(min(NUM_WARPS, LATENT // min(_LANES, LATENT)))
    spt_k: gl.constexpr = gl.constexpr(
        LATENT // (min(_LANES, LATENT) * min(NUM_WARPS, LATENT // min(_LANES, LATENT)))
    )
    reduce_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[NUM_KV_SPLITS, spt_k],
        threads_per_warp=[1, tpw_k],
        warps_per_cta=[1, wpc_k],
        order=[1, 0],
    )
    split_layout: gl.constexpr = gl.SliceLayout(1, reduce_layout)
    reduce_k_layout: gl.constexpr = gl.SliceLayout(0, reduce_layout)
    offs_split = gl.arange(0, NUM_KV_SPLITS, layout=split_layout)
    reduce_offs_k = gl.arange(0, LATENT, layout=reduce_k_layout)

    seq_len = gl.load(seq_len_ptr + batch)
    tiles_per_split = gl.cdiv(seq_len, NUM_KV_SPLITS * TILE_SIZE)
    active_splits = gl.cdiv(seq_len, tiles_per_split * TILE_SIZE)
    split_mask = offs_split < gl.full(
        [NUM_KV_SPLITS],
        active_splits,
        dtype=gl.int32,
        layout=split_layout,
    )
    split_offset = batch_head * NUM_KV_SPLITS + offs_split
    split_max = gl.load(
        split_max_ptr + split_offset,
        mask=split_mask,
        other=-float("inf"),
    ).to(gl.float32)
    overall_max = gl.max(split_max)
    split_expsum = gl.load(
        split_expsum_ptr + split_offset,
        mask=split_mask,
        other=0.0,
    ).to(gl.float32)
    split_scale = gl.exp2(split_max - overall_max)
    overall_expsum = gl.sum(split_expsum * split_scale)
    partial = gl.load(
        split_output_ptr + split_offset[:, None] * LATENT + reduce_offs_k[None, :],
        mask=split_mask[:, None],
        other=0.0,
    ).to(gl.float32)
    acc_sum = gl.sum(partial * split_scale[:, None], axis=0)

    # Preserve both materialized BF16 boundaries: the latent reducer output
    # and the following value projection.
    attention = gl.where(
        overall_expsum == 0.0,
        0.0,
        acc_sum / overall_expsum,
    ).to(gl.bfloat16)
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
    output_offset = batch_head * VALUE + offs_n
    if HAS_GATE:
        gate = gl.load(
            gate_ptr + batch * GATE_STRIDE_B + (head * VALUE + offs_n) * GATE_STRIDE_N
        ).to(gl.float32)
        projected *= 1.0 / (1.0 + gl.exp(-gate))
    gl.store(
        output_ptr + output_offset,
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

    batch = split_output.shape[0]
    num_splits = split_output.shape[2]
    heads, latent, value = weight.shape
    expected = (
        (
            split_output,
            (batch, heads, num_splits, latent),
            torch.float32,
            "split output",
        ),
        (split_max, (batch, heads, num_splits), torch.float32, "split maxima"),
        (
            split_expsum,
            (batch, heads, num_splits),
            torch.float32,
            "split exponent sums",
        ),
        (cache_seqlens, (batch,), torch.int32, "sequence lengths"),
        (weight, (heads, latent, value), torch.bfloat16, "value weight"),
    )
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
    gate_shape = (batch, heads * value)
    if gate is not None and (
        tuple(gate.shape) != gate_shape
        or gate.dtype != torch.bfloat16
        or not gate.is_cuda
        or gate.device != split_output.device
        or gate.stride(1) != 1
    ):
        raise ValueError(
            "MLA projected-value reducer requires a colocated BF16 gate "
            f"{gate_shape} with contiguous inner dimension"
        )
    if (
        tuple(out.shape) != (batch, heads * value)
        or out.dtype != torch.bfloat16
        or not out.is_cuda
        or not out.is_contiguous()
        or out.device != split_output.device
    ):
        raise ValueError(
            "MLA projected-value reducer requires contiguous colocated out "
            f"{(batch, heads * value)} {torch.bfloat16}"
        )

    gate_tensor = split_output if gate is None else gate
    num_warps = 8 if batch == 1 else 4
    _mla_reduce_project_value_kernel[(batch * heads * value // _BLOCK_N,)](
        split_output,
        split_max,
        split_expsum,
        cache_seqlens,
        weight,
        gate_tensor,
        out,
        LATENT=latent,
        VALUE=value,
        HEADS=heads,
        GATE_STRIDE_B=0 if gate is None else gate.stride(0),
        GATE_STRIDE_N=0 if gate is None else gate.stride(1),
        HAS_GATE=gate is not None,
        BATCHED=batch > 1,
        BLOCK_N=_BLOCK_N,
        NUM_WARPS=num_warps,
        NUM_KV_SPLITS=num_splits,
        TILE_SIZE=page_size,
        num_warps=num_warps,
        num_stages=1,
        waves_per_eu=0,
    )
    return out


__all__ = ["gluon_mla_reduce_project_value_gfx1250"]
