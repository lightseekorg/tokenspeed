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

"""Triton activation helper kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.gemm.fp8_utils import (
    create_per_token_group_quant_fp8_output_scale,
)

__all__ = [
    "fused_gate_sigmoid_mul_add",
    "fused_swiglu_fp8_ue8m0",
    "sigmoid_mul",
    "silu_and_mul",
    "swiglu_oai",
]


@triton.jit
def _fused_gate_sigmoid_mul_add_kernel(
    hidden_states_ptr,
    gate_weight_ptr,
    shared_output_ptr,
    final_ptr,
    hidden_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token_id = tl.program_id(0).to(tl.int64)
    row_offset = token_id * hidden_dim

    # Phase 1: gate = dot(hidden_states[token_id], gate_weight)
    # BLOCK >= hidden_dim so this loop is single-iteration (unrolled away).
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k_offset in range(0, hidden_dim, BLOCK):
        cols = k_offset + tl.arange(0, BLOCK)
        mask = cols < hidden_dim
        h = tl.load(hidden_states_ptr + row_offset + cols, mask=mask, other=0.0)
        w = tl.load(gate_weight_ptr + cols, mask=mask, other=0.0)
        acc += h.to(tl.float32) * w.to(tl.float32)
    gate_val = tl.sigmoid(tl.sum(acc, axis=0))

    # Phase 2: final[token_id] += gate_val * shared_output[token_id]
    for n_offset in range(0, hidden_dim, BLOCK):
        cols = n_offset + tl.arange(0, BLOCK)
        mask = cols < hidden_dim
        s = tl.load(shared_output_ptr + row_offset + cols, mask=mask)
        f = tl.load(final_ptr + row_offset + cols, mask=mask)
        out = f.to(tl.float32) + gate_val * s.to(tl.float32)
        tl.store(final_ptr + row_offset + cols, out.to(f.dtype), mask=mask)


def fused_gate_sigmoid_mul_add(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_output: torch.Tensor,
    final_hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Fused ``final_hidden_states += sigmoid(hidden_states @ gate_weight) * shared_output``.

    Computes the gate dot-product (reduction over hidden_dim), applies sigmoid,
    multiplies by ``shared_output``, and adds to ``final_hidden_states`` in-place.

    Args:
        hidden_states: ``[num_tokens, hidden_dim]`` contiguous input.
        gate_weight: ``[hidden_dim]`` contiguous 1-D weight vector.
        shared_output: ``[num_tokens, hidden_dim]`` contiguous shared expert output.
        final_hidden_states: ``[num_tokens, hidden_dim]`` contiguous MoE output,
            modified in-place.

    Returns:
        ``final_hidden_states`` (same storage, mutated in-place).
    """
    if hidden_states.ndim != 2:
        raise ValueError(f"hidden_states must be 2D, got {hidden_states.ndim}D")
    if not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be contiguous")
    if gate_weight.ndim != 1:
        raise ValueError(f"gate_weight must be 1D, got {gate_weight.ndim}D")
    if not gate_weight.is_contiguous():
        raise ValueError("gate_weight must be contiguous")
    if not shared_output.is_contiguous():
        raise ValueError("shared_output must be contiguous")
    if not final_hidden_states.is_contiguous():
        raise ValueError("final_hidden_states must be contiguous")

    num_tokens, hidden_dim = hidden_states.shape
    if gate_weight.shape[0] != hidden_dim:
        raise ValueError(
            f"gate_weight dim mismatch: expected {hidden_dim}, got {gate_weight.shape[0]}"
        )
    if shared_output.shape != (num_tokens, hidden_dim):
        raise ValueError(
            f"shared_output shape mismatch: expected {(num_tokens, hidden_dim)}, "
            f"got {shared_output.shape}"
        )
    if final_hidden_states.shape != (num_tokens, hidden_dim):
        raise ValueError(
            f"final_hidden_states shape mismatch: expected {(num_tokens, hidden_dim)}, "
            f"got {final_hidden_states.shape}"
        )

    if num_tokens == 0:
        return final_hidden_states

    BLOCK = triton.next_power_of_2(hidden_dim)
    num_warps = 4 if BLOCK <= 2048 else (8 if BLOCK <= 4096 else 16)
    grid = (num_tokens,)
    _fused_gate_sigmoid_mul_add_kernel[grid](
        hidden_states,
        gate_weight,
        shared_output,
        final_hidden_states,
        hidden_dim=hidden_dim,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return final_hidden_states


@triton.jit
def _sigmoid_mul_kernel(
    x_ptr,
    gate_ptr,
    n_elements,
    hidden_dim: tl.constexpr,
    head_dim: tl.constexpr,
    gate_row_stride: tl.constexpr,
    gate_head_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    row = offsets // hidden_dim
    col = offsets % hidden_dim
    head = col // head_dim
    d = col % head_dim
    gate_addrs = gate_ptr + row * gate_row_stride + head * gate_head_stride + d

    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    g = tl.load(gate_addrs, mask=mask).to(tl.float32)
    out = x * tl.sigmoid(g)
    tl.store(x_ptr + offsets, out, mask=mask)


def sigmoid_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """In-place ``x *= sigmoid(gate)``.

    ``x`` must be contiguous 2D ``[num_tokens, hidden_dim]`` and is mutated.
    ``gate`` may be either

    - 2D contiguous ``[num_tokens, hidden_dim]``, or
    - 3D ``[num_tokens, num_heads, head_dim]`` with ``stride(-1) == 1`` —
      the strided view that ``torch.chunk(q_gate, 2, dim=-1)`` produces from
      a packed ``[num_tokens, num_heads, 2 * head_dim]`` tensor.

    The strided form lets callers skip the ``.reshape(-1)`` copy after the
    chunk; both layouts share the same kernel via the explicit gate strides.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got {x.ndim}D")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if gate.stride(-1) != 1:
        raise ValueError(f"gate must have stride(-1) == 1, got {gate.stride()}")
    if x.dtype != gate.dtype:
        raise ValueError(f"dtype mismatch: x={x.dtype} gate={gate.dtype}")

    num_tokens, hidden_dim = x.shape

    if gate.ndim == 2:
        if gate.shape != x.shape:
            raise ValueError(f"shape mismatch: x={x.shape} gate={gate.shape}")
        head_dim = hidden_dim
        gate_row_stride = gate.stride(0)
        gate_head_stride = hidden_dim
    elif gate.ndim == 3:
        gate_tokens, num_heads, head_dim = gate.shape
        if gate_tokens != num_tokens:
            raise ValueError(f"num_tokens mismatch: x={num_tokens} gate={gate_tokens}")
        if num_heads * head_dim != hidden_dim:
            raise ValueError(
                f"hidden_dim mismatch: x={hidden_dim} gate={num_heads}*{head_dim}"
            )
        gate_row_stride = gate.stride(0)
        gate_head_stride = gate.stride(1)
    else:
        raise ValueError(f"gate must be 2D or 3D, got {gate.ndim}D")

    n = x.numel()
    if n == 0:
        return x

    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _sigmoid_mul_kernel[grid](
        x,
        gate,
        n,
        hidden_dim=hidden_dim,
        head_dim=head_dim,
        gate_row_stride=gate_row_stride,
        gate_head_stride=gate_head_stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return x


@triton.jit
def _silu_and_mul_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    hidden_dim: tl.constexpr,
    input_stride_row: tl.constexpr,
    out_stride_row: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    row = offsets // hidden_dim
    col = offsets % hidden_dim
    gate_addrs = x_ptr + row * input_stride_row + col
    up_addrs = gate_addrs + hidden_dim

    gate = tl.load(gate_addrs, mask=mask).to(tl.float32)
    up = tl.load(up_addrs, mask=mask).to(tl.float32)
    out = gate * tl.sigmoid(gate) * up
    tl.store(out_ptr + row * out_stride_row + col, out, mask=mask)


def silu_and_mul(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Fused ``SiLU(x[..., :D]) * x[..., D:]``.

    ``x`` is interpreted as ``[..., 2 * D]`` with gate values in the first half
    and up values in the second half. The output has shape ``[..., D]``.
    """
    del enable_pdl
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"last dimension must be even, got {x.shape[-1]}")
    if x.stride(-1) != 1:
        x = x.contiguous()

    hidden_dim = x.shape[-1] // 2
    output_shape = (*x.shape[:-1], hidden_dim)
    if out is None:
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    elif tuple(out.shape) != output_shape:
        raise ValueError(f"out shape must be {output_shape}, got {tuple(out.shape)}")

    if out.stride(-1) != 1:
        raise ValueError("out must have stride(-1) == 1")

    flat_x = x.reshape(-1, x.shape[-1])
    flat_out = out.reshape(-1, hidden_dim)
    n = flat_out.numel()
    if n == 0:
        return out

    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _silu_and_mul_kernel[grid](
        flat_x,
        flat_out,
        n,
        hidden_dim=hidden_dim,
        input_stride_row=flat_x.stride(0),
        out_stride_row=flat_out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


@triton.jit
def _swiglu_oai_kernel(
    gate_up_ptr,
    out_ptr,
    n_elements,
    hidden_dim: tl.constexpr,
    input_stride_row: tl.constexpr,
    out_stride_row: tl.constexpr,
    alpha: tl.constexpr,
    limit: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    row = offset // hidden_dim
    col = offset % hidden_dim
    gate_ptr = gate_up_ptr + row * input_stride_row + col
    up_ptr = gate_ptr + hidden_dim

    gate = tl.load(gate_ptr, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr, mask=mask, other=0.0).to(tl.float32)
    gate = tl.minimum(gate, limit)
    up = tl.clamp(up, -limit, limit)
    out = gate * tl.sigmoid(alpha * gate) * (up + 1.0)
    tl.store(out_ptr + row * out_stride_row + col, out, mask=mask)


def swiglu_oai(
    gate_up: torch.Tensor,
    *,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> torch.Tensor:
    """Fused ``gate * sigmoid(alpha * gate) * (up + 1)``.

    ``gate_up`` is interpreted as ``[..., 2 * D]`` with gate values in the first
    half and up values in the second half. The gate is upper-clamped to ``limit``
    and up is clamped to ``[-limit, limit]``. The output has shape ``[..., D]``.
    """
    if gate_up.shape[-1] % 2 != 0:
        raise ValueError(f"last dimension must be even, got {gate_up.shape[-1]}")
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    if gate_up.stride(-1) != 1:
        gate_up = gate_up.contiguous()

    hidden_dim = gate_up.shape[-1] // 2
    out = torch.empty(
        (*gate_up.shape[:-1], hidden_dim),
        dtype=gate_up.dtype,
        device=gate_up.device,
    )
    flat_input = gate_up.reshape(-1, gate_up.shape[-1])
    flat_out = out.reshape(-1, hidden_dim)
    n_elements = flat_out.numel()
    if n_elements == 0:
        return out

    block_size = 1024
    _swiglu_oai_kernel[((n_elements + block_size - 1) // block_size,)](
        flat_input,
        flat_out,
        n_elements,
        hidden_dim=hidden_dim,
        input_stride_row=flat_input.stride(0),
        out_stride_row=flat_out.stride(0),
        alpha=alpha,
        limit=limit,
        BLOCK_SIZE=block_size,
    )
    return out


# ---------------------------------------------------------------------------
# Fused SwiGLU + FP8 UE8M0 quantization
# ---------------------------------------------------------------------------


@triton.jit
def _fused_swiglu_fp8_ue8m0_kernel(
    gate_up_ptr,
    out_ptr,
    scale_ptr,
    M,
    N: tl.constexpr,
    gate_up_stride_row,
    out_stride_row,
    scale_col_stride,
    swiglu_limit,
    swiglu_alpha,
    swiglu_beta,
    eps,
    bit8_min,
    bit8_max,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    groups_per_row = N // GROUP_SIZE
    row = pid // groups_per_row
    group_col = pid % groups_per_row

    gate_offset = (
        row.to(tl.int64) * gate_up_stride_row + group_col.to(tl.int64) * GROUP_SIZE
    )
    up_offset = gate_offset + N
    out_offset = row.to(tl.int64) * out_stride_row + group_col.to(tl.int64) * GROUP_SIZE

    cols = tl.arange(0, GROUP_SIZE)

    gate = tl.load(gate_up_ptr + gate_offset + cols).to(tl.float32)
    up = tl.load(gate_up_ptr + up_offset + cols).to(tl.float32)

    if swiglu_limit > 0.0:
        gate = tl.minimum(gate, swiglu_limit)
        up = tl.clamp(up, -swiglu_limit, swiglu_limit)

    silu_gate = gate * tl.sigmoid(swiglu_alpha * gate)
    y = silu_gate * (up + swiglu_beta)

    _absmax = tl.max(tl.abs(y))
    scale_raw = tl.maximum(_absmax / bit8_max, eps)
    exponent = tl.ceil(tl.log2(scale_raw))
    y_s = tl.exp2(exponent)
    y_q = tl.clamp(y / y_s, bit8_min, bit8_max).to(out_ptr.dtype.element_ty)

    tl.store(out_ptr + out_offset + cols, y_q)

    scale_pack_col = group_col // 4
    scale_pack_pos = group_col % 4
    scale_ptr_offset = scale_pack_col.to(tl.int64) * scale_col_stride + row.to(tl.int64)
    exponent_biased = tl.clamp(exponent + 127.0, 0.0, 255.0).to(tl.uint32)
    packed_scale = exponent_biased << (scale_pack_pos * 8)
    tl.atomic_or(scale_ptr + scale_ptr_offset, packed_scale, sem="relaxed")


def fused_swiglu_fp8_ue8m0(
    gate_up: torch.Tensor,
    swiglu_limit: float = 0.0,
    swiglu_alpha: float = 1.0,
    swiglu_beta: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused SwiGLU activation + FP8 UE8M0 block-scale quantization.

    Reads a ``[M, 2*N]`` gate_up tensor (gate in the first half, up in the
    second half), applies ``clamp + SiLU(gate) * up``, and quantizes the
    result to FP8 E4M3 with UE8M0 packed block scales in one kernel pass.

    Args:
        gate_up: ``[M, 2*N]`` tensor (BF16 or FP8; cast to float32 internally).
        swiglu_limit: Clamp bound. 0 or negative disables clamping.
        swiglu_alpha: Sigmoid multiplier applied to the gate.
        swiglu_beta: Value added to the up projection before multiplication.

    Returns:
        ``(fp8_out, scale)``: ``fp8_out`` is ``[M, N]`` float8_e4m3fn,
        ``scale`` is UE8M0 packed int32 column-major TMA-aligned.
    """
    assert gate_up.ndim == 2, f"Expected 2D input, got {gate_up.ndim}D"
    M, two_N = gate_up.shape
    assert two_N % 2 == 0
    N = two_N // 2
    assert N % 128 == 0, f"N={N} must be multiple of 128 for UE8M0 group_size=128"

    GROUP_SIZE = 128
    dtype = torch.float8_e4m3fn
    info = torch.finfo(dtype)

    out = torch.empty((M, N), device=gate_up.device, dtype=dtype)
    scale = create_per_token_group_quant_fp8_output_scale(
        x_shape=(M, N),
        device=gate_up.device,
        group_size=GROUP_SIZE,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )

    num_groups = M * (N // GROUP_SIZE)
    _fused_swiglu_fp8_ue8m0_kernel[(num_groups,)](
        gate_up,
        out,
        scale,
        M,
        N,
        gate_up.stride(0),
        out.stride(0),
        scale.stride(-1),
        swiglu_limit if swiglu_limit is not None and swiglu_limit > 0 else 0.0,
        swiglu_alpha,
        swiglu_beta,
        1e-10,
        bit8_min=info.min,
        bit8_max=info.max,
        GROUP_SIZE=GROUP_SIZE,
        num_warps=min(max(GROUP_SIZE // 256, 1), 8),
        num_stages=1,
    )

    return out, scale
