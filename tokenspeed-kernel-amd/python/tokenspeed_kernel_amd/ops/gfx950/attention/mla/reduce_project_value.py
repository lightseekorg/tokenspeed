# Copyright (c) 2026 LightSeek Foundation

"""Split-MLA reduction with value projection and optional gate for gfx950."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

_LANES = gl.constexpr(64)


@gluon.jit
def _mla_reduce_project_value_kernel(
    logits_ptr,
    mid_lse_ptr,
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
    PAGE_SIZE: gl.constexpr,
):
    """Reduce split attention directly into the BF16 value/gate epilogue."""

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

    seq_len = gl.load(seq_len_ptr + batch)
    num_pages = gl.cdiv(seq_len, PAGE_SIZE)
    pages_per_split = gl.cdiv(num_pages, NUM_KV_SPLITS)
    e_sum = 0.0
    e_max = -float("inf")
    acc = gl.zeros([LATENT], gl.float32, k_layout)
    for split in range(0, NUM_KV_SPLITS):
        valid = split * pages_per_split < num_pages
        partial = gl.load(
            logits_ptr + (batch_head * NUM_KV_SPLITS + split) * LATENT + offs_k,
            mask=valid,
            other=0.0,
        ).to(gl.float32)
        split_lse = gl.load(
            mid_lse_ptr + batch_head * NUM_KV_SPLITS + split,
            mask=valid,
            other=-float("inf"),
        ).to(gl.float32)
        next_max = gl.maximum(split_lse, e_max)
        old_scale = gl.exp(e_max - next_max)
        split_scale = gl.exp(split_lse - next_max)
        acc = acc * old_scale + partial * split_scale
        e_sum = e_sum * old_scale + split_scale
        e_max = next_max

    # Preserve both materialized BF16 boundaries: the split reducer's latent
    # output and the following value projection.
    attention = (acc / e_sum).to(gl.bfloat16)
    weight = gl.amd.cdna4.buffer_load(
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


def gluon_mla_reduce_project_value_gfx950(
    logits: torch.Tensor,
    mid_lse: torch.Tensor,
    cache_seqlens: torch.Tensor,
    weight: torch.Tensor,
    *,
    gate: torch.Tensor | None = None,
    page_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce split MLA partials and directly emit projected values."""

    batch = logits.shape[0]
    num_splits = logits.shape[2]
    heads, latent, value = weight.shape
    expected = (
        (logits, (batch, heads, num_splits, latent), torch.bfloat16, "logits"),
        (mid_lse, (batch, heads, num_splits), torch.float32, "mid LSE"),
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
            or tensor.device != logits.device
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
        or gate.device != logits.device
        or gate.stride(1) != 1
    ):
        raise ValueError(
            "MLA projected-value reducer requires a colocated BF16 gate "
            f"{gate_shape} with contiguous inner dimension"
        )
    output_shape = (batch, heads * value)
    if out is None:
        out = logits.new_empty(output_shape)
    elif (
        tuple(out.shape) != output_shape
        or out.dtype != torch.bfloat16
        or not out.is_cuda
        or not out.is_contiguous()
        or out.device != logits.device
    ):
        raise ValueError(
            "MLA projected-value reducer requires contiguous colocated out "
            f"{output_shape} {torch.bfloat16}"
        )
    block_n = 8
    num_warps = 8
    gate_tensor = logits if gate is None else gate
    _mla_reduce_project_value_kernel[(batch * heads * value // block_n,)](
        logits,
        mid_lse,
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
        BLOCK_N=block_n,
        NUM_WARPS=num_warps,
        NUM_KV_SPLITS=num_splits,
        PAGE_SIZE=page_size,
        num_warps=num_warps,
        num_stages=1,
        waves_per_eu=0,
    )
    return out


__all__ = ["gluon_mla_reduce_project_value_gfx950"]
