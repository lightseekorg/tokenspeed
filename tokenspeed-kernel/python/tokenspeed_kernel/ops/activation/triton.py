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
from tokenspeed_kernel._triton import libdevice, tl, triton

__all__ = [
    "add3",
    "fused_gate_sigmoid_mul_add",
    "fused_swiglu_fp8_ue8m0",
    "fused_swiglu_fp8_ue8m0_masked_packed",
    "sigmoid_mul",
    "silu_and_mul",
    "situ_and_mul",
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


@triton.jit
def _situ_and_mul_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    beta,
    linear_beta,
    hidden_dim: tl.constexpr,
    input_stride_row: tl.constexpr,
    out_stride_row: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    row = offsets // hidden_dim
    col = offsets % hidden_dim
    gate_addrs = x_ptr + row * input_stride_row + col
    up_addrs = gate_addrs + hidden_dim

    gate = tl.load(gate_addrs, mask=mask).to(tl.float32)
    up = tl.load(up_addrs, mask=mask).to(tl.float32)
    gate = beta * libdevice.tanh(gate / beta) * tl.sigmoid(gate)
    if HAS_LINEAR_BETA:
        up = linear_beta * libdevice.tanh(up / linear_beta)
    tl.store(out_ptr + row * out_stride_row + col, gate * up, mask=mask)


def situ_and_mul(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    beta: float = 1.0,
    linear_beta: float | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Apply SiTU to a concatenated ``[gate, up]`` tensor.

    SiTU computes ``beta * tanh(gate / beta) * sigmoid(gate) * up``.
    When ``linear_beta`` is provided, it first soft-clips the up branch as
    ``linear_beta * tanh(up / linear_beta)``. The nonlinear math is evaluated
    in FP32 and the result is stored in the input/output dtype.

    Args:
        x: Tensor shaped ``[..., 2 * D]`` with a contiguous final dimension.
        out: Optional output tensor shaped ``[..., D]``.
        beta: Positive gate soft-clipping scale.
        linear_beta: Optional positive up-branch soft-clipping scale.
        enable_pdl: Reserved for API compatibility; ignored on this kernel.

    Returns:
        Tensor shaped ``[..., D]`` in the same dtype and device as ``x``.
    """
    del enable_pdl
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"last dimension must be even, got {x.shape[-1]}")
    if beta <= 0.0:
        raise ValueError(f"beta must be positive, got {beta}")
    if linear_beta is not None and linear_beta <= 0.0:
        raise ValueError(f"linear_beta must be positive, got {linear_beta}")
    if x.stride(-1) != 1:
        x = x.contiguous()

    hidden_dim = x.shape[-1] // 2
    output_shape = (*x.shape[:-1], hidden_dim)
    if out is None:
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    elif tuple(out.shape) != output_shape:
        raise ValueError(f"out shape must be {output_shape}, got {tuple(out.shape)}")
    if out.dtype != x.dtype or out.device != x.device:
        raise ValueError("out must have the same dtype and device as x")
    if out.stride(-1) != 1:
        raise ValueError("out must have stride(-1) == 1")

    flat_x = x.reshape(-1, x.shape[-1])
    # A tensor can have stride(-1) == 1 while its outer dimensions are
    # non-contiguous. In that case reshape() allocates storage, so launching
    # directly into the reshaped tensor would leave the caller's ``out``
    # unchanged. Compute into a contiguous temporary and copy it back.
    kernel_out = (
        out
        if out.is_contiguous()
        else torch.empty_like(out, memory_format=torch.contiguous_format)
    )
    flat_out = kernel_out.reshape(-1, hidden_dim)
    n = flat_out.numel()
    if n == 0:
        return out

    block_size = 1024
    grid = (triton.cdiv(n, block_size),)
    _situ_and_mul_kernel[grid](
        flat_x,
        flat_out,
        n,
        float(beta),
        1.0 if linear_beta is None else float(linear_beta),
        hidden_dim=hidden_dim,
        input_stride_row=flat_x.stride(0),
        out_stride_row=flat_out.stride(0),
        HAS_LINEAR_BETA=linear_beta is not None,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    if kernel_out is not out:
        out.copy_(kernel_out)
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
    PACK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    # Each program covers PACK=4 adjacent scale groups of one row: the four
    # UE8M0 exponents that share one packed int32 are produced together, so
    # the scale write is a plain store instead of four atomic_or's racing on
    # the same word.
    pid = tl.program_id(0)
    groups_per_row: tl.constexpr = N // GROUP_SIZE
    packs_per_row: tl.constexpr = (groups_per_row + PACK - 1) // PACK
    row = pid // packs_per_row
    pack_col = pid % packs_per_row

    BLOCK: tl.constexpr = GROUP_SIZE * PACK
    col0 = pack_col * BLOCK
    cols = col0 + tl.arange(0, BLOCK)
    col_mask = cols < N

    if ENABLE_PDL:
        # ``gate_up`` is produced by the first DeepGEMM. Do not read it until
        # that PDL producer has made its stores visible.
        tl.extra.cuda.gdc_wait()

    gate_base = row.to(tl.int64) * gate_up_stride_row
    gate = tl.load(gate_up_ptr + gate_base + cols, mask=col_mask, other=0.0).to(
        tl.float32
    )
    up = tl.load(gate_up_ptr + gate_base + N + cols, mask=col_mask, other=0.0).to(
        tl.float32
    )

    if swiglu_limit > 0.0:
        gate = tl.minimum(gate, swiglu_limit)
        up = tl.clamp(up, -swiglu_limit, swiglu_limit)

    silu_gate = gate * tl.sigmoid(swiglu_alpha * gate)
    y = tl.reshape(silu_gate * (up + swiglu_beta), (PACK, GROUP_SIZE))

    _absmax = tl.max(tl.abs(y), axis=1)
    scale_raw = tl.maximum(_absmax / bit8_max, eps)
    exponent = tl.ceil(tl.log2(scale_raw))
    y_s = tl.exp2(exponent)
    y_q = tl.clamp(y / y_s[:, None], bit8_min, bit8_max).to(out_ptr.dtype.element_ty)

    out_base = row.to(tl.int64) * out_stride_row
    tl.store(out_ptr + out_base + cols, tl.reshape(y_q, (BLOCK,)), mask=col_mask)

    group_ids = pack_col * PACK + tl.arange(0, PACK)
    group_mask = group_ids < groups_per_row
    exponent_biased = tl.where(
        group_mask, tl.clamp(exponent + 127.0, 0.0, 255.0), 0.0
    ).to(tl.uint32)
    packed_scale = tl.sum(exponent_biased << (tl.arange(0, PACK) * 8))
    scale_ptr_offset = pack_col.to(tl.int64) * scale_col_stride + row.to(tl.int64)
    tl.store(scale_ptr + scale_ptr_offset, packed_scale)

    if ENABLE_PDL:
        # Both activation values and packed scales are ready for DeepGEMM 2.
        tl.extra.cuda.gdc_launch_dependents()


def fused_swiglu_fp8_ue8m0(
    gate_up: torch.Tensor,
    swiglu_limit: float = 0.0,
    swiglu_alpha: float = 1.0,
    swiglu_beta: float = 0.0,
    *,
    enable_pdl: bool = False,
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
        enable_pdl: Join an SM90+ Programmatic Dependent Launch chain. The
            kernel waits for the preceding producer before reading ``gate_up``
            and releases the following dependent after writing both outputs.

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
    # Every packed word, including tail bytes, is overwritten by the fused
    # kernel. Allocate directly so a zero-fill kernel does not break the
    # DeepGEMM 1 -> SwiGLU PDL chain.
    groups_per_row = N // GROUP_SIZE
    aligned_m = (M + 3) // 4 * 4
    packed_groups = (groups_per_row + 3) // 4
    scale_base = torch.empty(
        (packed_groups, aligned_m), device=gate_up.device, dtype=torch.int32
    )
    scale = scale_base.transpose(0, 1)[:M, :]

    PACK = 4
    packs_per_row = (groups_per_row + PACK - 1) // PACK
    num_programs = M * packs_per_row
    pdl_kwargs = {"launch_pdl": True} if enable_pdl else {}
    _fused_swiglu_fp8_ue8m0_kernel[(num_programs,)](
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
        PACK=PACK,
        ENABLE_PDL=enable_pdl,
        num_warps=2,
        num_stages=1,
        **pdl_kwargs,
    )

    return out, scale


@triton.jit
def _fused_swiglu_fp8_ue8m0_masked_packed_row(
    gate_up_ptr,
    out_ptr,
    scale_ptr,
    expert,
    row,
    pack_col,
    N: tl.constexpr,
    gate_up_stride_e,
    gate_up_stride_m,
    out_stride_e,
    out_stride_m,
    scale_stride_e,
    scale_stride_m,
    scale_stride_p,
    eps,
    bit8_min,
    bit8_max,
    GROUP_SIZE: tl.constexpr,
    PACK: tl.constexpr,
):
    """Process one expert row and one four-scale pack."""
    groups_per_row: tl.constexpr = N // GROUP_SIZE
    BLOCK: tl.constexpr = GROUP_SIZE * PACK
    col0 = pack_col * BLOCK
    cols = col0 + tl.arange(0, BLOCK)
    col_mask = cols < N
    gate_base = (
        expert.to(tl.int64) * gate_up_stride_e + row.to(tl.int64) * gate_up_stride_m
    )
    gate = tl.load(gate_up_ptr + gate_base + cols, mask=col_mask, other=0.0).to(
        tl.float32
    )
    up = tl.load(gate_up_ptr + gate_base + N + cols, mask=col_mask, other=0.0).to(
        tl.float32
    )

    y = tl.reshape(gate * tl.sigmoid(gate) * up, (PACK, GROUP_SIZE))
    _absmax = tl.max(tl.abs(y), axis=1)
    scale_raw = tl.maximum(_absmax / bit8_max, eps)
    exponent = tl.ceil(tl.log2(scale_raw))
    y_s = tl.exp2(exponent)
    y_q = tl.clamp(y / y_s[:, None], bit8_min, bit8_max).to(out_ptr.dtype.element_ty)

    out_base = expert.to(tl.int64) * out_stride_e + row.to(tl.int64) * out_stride_m
    tl.store(
        out_ptr + out_base + cols,
        tl.reshape(y_q, (BLOCK,)),
        mask=col_mask,
    )

    group_ids = pack_col * PACK + tl.arange(0, PACK)
    group_mask = group_ids < groups_per_row
    exponent_biased = tl.where(
        group_mask, tl.clamp(exponent + 127.0, 0.0, 255.0), 0.0
    ).to(tl.uint32)
    packed_scale = tl.sum(exponent_biased << (tl.arange(0, PACK) * 8))
    scale_offset = (
        expert.to(tl.int64) * scale_stride_e
        + row.to(tl.int64) * scale_stride_m
        + pack_col.to(tl.int64) * scale_stride_p
    )
    tl.store(scale_ptr + scale_offset, packed_scale)


@triton.jit
def _fused_swiglu_fp8_ue8m0_masked_packed_kernel(
    gate_up_ptr,
    out_ptr,
    scale_ptr,
    masked_m_ptr,
    N: tl.constexpr,
    gate_up_stride_e,
    gate_up_stride_m,
    out_stride_e,
    out_stride_m,
    scale_stride_e,
    scale_stride_m,
    scale_stride_p,
    eps,
    bit8_min,
    bit8_max,
    GROUP_SIZE: tl.constexpr,
    PACK: tl.constexpr,
    ROW_SPLITS: tl.constexpr,
    USE_ROW_LOOP: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    """Masked SwiGLU quantization writing DeepGEMM's packed scale layout."""
    pid = tl.program_id(0)
    expert = tl.program_id(1)
    groups_per_row: tl.constexpr = N // GROUP_SIZE
    packs_per_row: tl.constexpr = (groups_per_row + PACK - 1) // PACK

    if ENABLE_PDL:
        # The first masked DeepGEMM owns ``gate_up``. The row-count tensor is
        # already available, but waiting here keeps every source load behind
        # the producer's completion point.
        tl.extra.cuda.gdc_wait()

    valid_rows = tl.load(masked_m_ptr + expert)

    if USE_ROW_LOOP:
        # Sparse decode: launch a bounded number of row splits per expert and
        # let each CTA walk only valid rows. This avoids scheduling one CTA for
        # every row in DeepEP's heavily over-provisioned capacity buffer.
        pack_col = pid // ROW_SPLITS
        row = pid % ROW_SPLITS
        while row < valid_rows:
            _fused_swiglu_fp8_ue8m0_masked_packed_row(
                gate_up_ptr,
                out_ptr,
                scale_ptr,
                expert,
                row,
                pack_col,
                N,
                gate_up_stride_e,
                gate_up_stride_m,
                out_stride_e,
                out_stride_m,
                scale_stride_e,
                scale_stride_m,
                scale_stride_p,
                eps,
                bit8_min,
                bit8_max,
                GROUP_SIZE,
                PACK,
            )
            row += ROW_SPLITS
    else:
        # Dense/full-capacity fallback: retain one CTA per row and pack so full
        # loads keep the original parallelism and do not serialize row work.
        row = pid // packs_per_row
        pack_col = pid % packs_per_row
        if row >= valid_rows:
            # Block termination implicitly signals PDL completion, so invalid
            # capacity rows retain the fast early-exit path.
            return
        _fused_swiglu_fp8_ue8m0_masked_packed_row(
            gate_up_ptr,
            out_ptr,
            scale_ptr,
            expert,
            row,
            pack_col,
            N,
            gate_up_stride_e,
            gate_up_stride_m,
            out_stride_e,
            out_stride_m,
            scale_stride_e,
            scale_stride_m,
            scale_stride_p,
            eps,
            bit8_min,
            bit8_max,
            GROUP_SIZE,
            PACK,
        )

    if ENABLE_PDL:
        # Release the second masked DeepGEMM only after every active row has
        # stored both its FP8 values and its packed UE8M0 scales.
        tl.extra.cuda.gdc_launch_dependents()


def _masked_swiglu_row_splits(capacity: int, expected_m: int | None) -> int:
    """Choose bounded row parallelism without reading device-side masks."""
    if expected_m is None:
        return capacity
    expected_m = int(expected_m)
    if expected_m <= 0:
        raise ValueError(f"expected_m must be positive, got {expected_m}")

    # Two-way over-provisioning absorbs ordinary router imbalance. A minimum of
    # 16 splits protects a hot expert while still shrinking Qwen3.5 EP4's
    # sparse grid from 65,536 CTAs to 1,024 CTAs.
    target = max(16, min(expected_m, capacity) * 2)
    power_of_two = 1 << (target - 1).bit_length()
    return min(capacity, power_of_two)


def fused_swiglu_fp8_ue8m0_masked_packed(
    gate_up: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int | None = None,
    *,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused masked SwiGLU and FP8 quantization with packed UE8M0 scales.

    This is the decode-oriented counterpart of
    :func:`fused_swiglu_fp8_ue8m0`. It writes the packed int32, MN-major,
    TMA-aligned scale tensor consumed directly by DeepGEMM, avoiding a scale
    clear followed by separate transpose and pack kernels. Rows beyond each
    expert's ``masked_m`` are left uninitialized because masked grouped GEMM
    never reads them.

    Args:
        gate_up: ``[experts, capacity, 2*N]`` BF16 gate/up activations.
        masked_m: ``[experts]`` int32 valid-row counts.
        expected_m: Host-side estimate of valid rows per expert. Sparse launches
            use it only to choose row parallelism; ``masked_m`` remains the
            correctness bound. ``None`` retains one CTA per capacity row.
        enable_pdl: Join an SM90+ Programmatic Dependent Launch chain. The
            kernel waits for the preceding masked GEMM before reading
            ``gate_up`` and releases the following GEMM after its stores.

    Returns:
        A pair ``(out, scales)``. ``out`` has shape
        ``[experts, capacity, N]`` and FP8 E4M3 dtype. ``scales`` is an int32
        view of shape ``[experts, capacity, ceil((N / 128) / 4)]`` with
        MN-major, four-exponents-per-word UE8M0 packing.
    """
    assert gate_up.ndim == 3, f"Expected 3D input, got {gate_up.ndim}D"
    num_experts, capacity, two_N = gate_up.shape
    assert two_N % 2 == 0
    N = two_N // 2
    GROUP_SIZE = 128
    PACK = 4
    assert N % GROUP_SIZE == 0, f"N={N} must be a multiple of {GROUP_SIZE}"
    assert masked_m.shape == (num_experts,)
    assert masked_m.dtype == torch.int32

    out = torch.empty(
        (num_experts, capacity, N),
        dtype=torch.float8_e4m3fn,
        device=gate_up.device,
    )
    groups_per_row = N // GROUP_SIZE
    packs_per_row = (groups_per_row + PACK - 1) // PACK
    aligned_capacity = (capacity + 3) // 4 * 4
    scale_base = torch.empty(
        (num_experts, packs_per_row, aligned_capacity),
        dtype=torch.int32,
        device=gate_up.device,
    )
    scales = scale_base.transpose(1, 2)[:, :capacity, :]

    row_splits = _masked_swiglu_row_splits(capacity, expected_m)
    use_row_loop = row_splits < capacity
    grid_rows = row_splits if use_row_loop else capacity
    info = torch.finfo(out.dtype)
    pdl_kwargs = {"launch_pdl": True} if enable_pdl else {}
    _fused_swiglu_fp8_ue8m0_masked_packed_kernel[
        (grid_rows * packs_per_row, num_experts)
    ](
        gate_up,
        out,
        scales,
        masked_m,
        N,
        gate_up.stride(0),
        gate_up.stride(1),
        out.stride(0),
        out.stride(1),
        scales.stride(0),
        scales.stride(1),
        scales.stride(2),
        1e-10,
        bit8_min=info.min,
        bit8_max=info.max,
        GROUP_SIZE=GROUP_SIZE,
        PACK=PACK,
        ROW_SPLITS=row_splits,
        USE_ROW_LOOP=use_row_loop,
        ENABLE_PDL=enable_pdl,
        num_warps=2,
        num_stages=1,
        **pdl_kwargs,
    )
    return out, scales


@triton.jit
def _rmsnorm_gated_kernel(
    x_ptr,
    gate_ptr,
    weight_ptr,
    out_ptr,
    eps,
    gate_stride,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, head_dim)
    mask_h = offs_h < num_heads
    idx = token * num_heads * head_dim + offs_h[:, None] * head_dim + offs_d[None, :]
    x = tl.load(x_ptr + idx, mask=mask_h[:, None], other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=1) / head_dim
    rsig = tl.math.rsqrt(var + eps)
    w = tl.load(weight_ptr + offs_d).to(tl.float32)
    gate_idx = token * gate_stride + offs_h[:, None] * head_dim + offs_d[None, :]
    g = tl.load(gate_ptr + gate_idx, mask=mask_h[:, None], other=0.0).to(tl.float32)
    y = x * rsig[:, None] * w[None, :] * tl.sigmoid(g)
    tl.store(out_ptr + idx, y.to(out_ptr.dtype.element_ty), mask=mask_h[:, None])


def rmsnorm_gated_sigmoid(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Per-head RMSNorm fused with a sigmoid output gate: ``rmsnorm(x)*w*sigmoid(g)``.

    Args:
        x: ``[num_tokens, num_heads*head_dim]`` bf16 input (contiguous).
        gate: same shape as ``x`` with unit inner stride; raw gate logits.
        weight: ``[head_dim]`` RMSNorm weight.
        eps: RMSNorm epsilon.
        num_heads / head_dim: per-head norm geometry.

    Returns:
        ``[num_tokens, num_heads*head_dim]`` tensor of ``x``'s dtype.
    """
    assert x.is_contiguous() and gate.shape == x.shape and gate.stride(-1) == 1
    out = torch.empty_like(x)
    _rmsnorm_gated_kernel[(x.shape[0],)](
        x,
        gate,
        weight,
        out,
        eps,
        gate.stride(0),
        num_heads=num_heads,
        head_dim=head_dim,
        BLOCK_H=triton.next_power_of_2(num_heads),
        num_warps=4,
    )
    return out


@triton.jit
def _add3_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    n_cols,
    stride_a,
    stride_b,
    stride_c,
    stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < n_cols
    a = tl.load(a_ptr + row * stride_a + col, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + row * stride_b + col, mask=mask).to(tl.float32)
    c = tl.load(c_ptr + row * stride_c + col, mask=mask).to(tl.float32)
    tl.store(
        out_ptr + row * stride_o + col,
        (a + b + c).to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def add3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Elementwise ``a + b + c`` in one kernel (fp32 accumulate, a's dtype out).

    Args:
        a/b/c: same-shape contiguous CUDA tensors.

    Returns:
        New tensor of ``a``'s dtype.
    """
    assert a.shape == b.shape == c.shape and a.dim() == 2
    # Row-strided views welcome (e.g. column slices); inner dim must be dense.
    assert a.stride(1) == b.stride(1) == c.stride(1) == 1
    if a.numel() == 0 or not a.is_cuda:
        return a + b + c
    rows, cols = a.shape
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    BLOCK = 1024
    _add3_kernel[(rows, (cols + BLOCK - 1) // BLOCK)](
        a,
        b,
        c,
        out,
        cols,
        a.stride(0),
        b.stride(0),
        c.stride(0),
        out.stride(0),
        BLOCK=BLOCK,
    )
    return out


@triton.jit
def _attnres_partial_kernel(
    blocks_ptr,  # [KB, T, H]
    wp_ptr,  # [H] precomputed rms_w * res_w
    m_ptr,  # [T]
    s_ptr,  # [T]
    acc_ptr,  # [T, H] fp32
    n_blocks,
    n_cols: tl.constexpr,
    stride_bk,
    stride_bt,
    eps,
    BLOCK: tl.constexpr,
):
    """Online-softmax partial over the static block candidates (aux stream).

    Score weights and the running accumulator stay in registers; each block
    row is read from global memory exactly once.
    """
    t = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    n_iters: tl.constexpr = (n_cols + BLOCK - 1) // BLOCK
    tl.static_assert(n_iters == 2)

    col0 = offs
    col1 = BLOCK + offs
    mask0 = col0 < n_cols
    mask1 = col1 < n_cols
    wp0 = tl.load(wp_ptr + col0, mask=mask0, other=0.0).to(tl.float32)
    wp1 = tl.load(wp_ptr + col1, mask=mask1, other=0.0).to(tl.float32)

    acc0 = tl.zeros([BLOCK], tl.float32)
    acc1 = tl.zeros([BLOCK], tl.float32)
    m_run = -float("inf")
    s_run = 0.0
    b = 0
    while b < n_blocks:
        base = blocks_ptr + b * stride_bk + t * stride_bt
        v0 = tl.load(base + col0, mask=mask0, other=0.0).to(tl.float32)
        v1 = tl.load(base + col1, mask=mask1, other=0.0).to(tl.float32)
        sq = tl.sum(v0 * v0) + tl.sum(v1 * v1)
        dot = tl.sum(v0 * wp0) + tl.sum(v1 * wp1)
        rsig = tl.math.rsqrt(sq / n_cols + eps)
        logit = dot * rsig
        m_new = tl.maximum(m_run, logit)
        corr = tl.exp(m_run - m_new)
        wgt = tl.exp(logit - m_new)
        acc0 = acc0 * corr + wgt * v0
        acc1 = acc1 * corr + wgt * v1
        s_run = s_run * corr + wgt
        m_run = m_new
        b += 1
    tl.store(acc_ptr + t * n_cols + col0, acc0, mask=mask0)
    tl.store(acc_ptr + t * n_cols + col1, acc1, mask=mask1)
    tl.store(m_ptr + t, m_run)
    tl.store(s_ptr + t, s_run)


@triton.jit
def _attnres_partial_dual_kernel(
    blocks_ptr,  # [KB, T, H]
    wp_a_ptr,  # [H] precomputed side-A rms_w * res_w
    wp_b_ptr,  # [H] side-B product
    m_a_ptr,
    s_a_ptr,
    acc_a_ptr,
    m_b_ptr,
    s_b_ptr,
    acc_b_ptr,
    n_blocks,
    n_cols: tl.constexpr,
    stride_bk,
    stride_bt,
    eps,
    BLOCK: tl.constexpr,
):
    """Two online-softmax partials over the same block sweep (aux stream).

    Side A is this layer's mlp-side mix, side B the next layer's attn-side
    mix. Both consume the identical block-snapshot set, so one kernel pays
    the global reads and the single-CTA latency once for both.
    """
    t = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    n_iters: tl.constexpr = (n_cols + BLOCK - 1) // BLOCK
    tl.static_assert(n_iters == 2)

    col0 = offs
    col1 = BLOCK + offs
    mask0 = col0 < n_cols
    mask1 = col1 < n_cols
    wa0 = tl.load(wp_a_ptr + col0, mask=mask0, other=0.0).to(tl.float32)
    wa1 = tl.load(wp_a_ptr + col1, mask=mask1, other=0.0).to(tl.float32)
    wb0 = tl.load(wp_b_ptr + col0, mask=mask0, other=0.0).to(tl.float32)
    wb1 = tl.load(wp_b_ptr + col1, mask=mask1, other=0.0).to(tl.float32)

    acc_a0 = tl.zeros([BLOCK], tl.float32)
    acc_a1 = tl.zeros([BLOCK], tl.float32)
    acc_b0 = tl.zeros([BLOCK], tl.float32)
    acc_b1 = tl.zeros([BLOCK], tl.float32)
    m_a = -float("inf")
    s_a = 0.0
    m_b = -float("inf")
    s_b = 0.0
    b = 0
    while b < n_blocks:
        base = blocks_ptr + b * stride_bk + t * stride_bt
        v0 = tl.load(base + col0, mask=mask0, other=0.0).to(tl.float32)
        v1 = tl.load(base + col1, mask=mask1, other=0.0).to(tl.float32)
        sq = tl.sum(v0 * v0) + tl.sum(v1 * v1)
        rsig = tl.math.rsqrt(sq / n_cols + eps)
        dot_a = tl.sum(v0 * wa0) + tl.sum(v1 * wa1)
        logit_a = dot_a * rsig
        m_an = tl.maximum(m_a, logit_a)
        corr_a = tl.exp(m_a - m_an)
        wgt_a = tl.exp(logit_a - m_an)
        acc_a0 = acc_a0 * corr_a + wgt_a * v0
        acc_a1 = acc_a1 * corr_a + wgt_a * v1
        s_a = s_a * corr_a + wgt_a
        m_a = m_an
        dot_b = tl.sum(v0 * wb0) + tl.sum(v1 * wb1)
        logit_b = dot_b * rsig
        m_bn = tl.maximum(m_b, logit_b)
        corr_b = tl.exp(m_b - m_bn)
        wgt_b = tl.exp(logit_b - m_bn)
        acc_b0 = acc_b0 * corr_b + wgt_b * v0
        acc_b1 = acc_b1 * corr_b + wgt_b * v1
        s_b = s_b * corr_b + wgt_b
        m_b = m_bn
        b += 1
    tl.store(acc_a_ptr + t * n_cols + col0, acc_a0, mask=mask0)
    tl.store(acc_a_ptr + t * n_cols + col1, acc_a1, mask=mask1)
    tl.store(m_a_ptr + t, m_a)
    tl.store(s_a_ptr + t, s_a)
    tl.store(acc_b_ptr + t * n_cols + col0, acc_b0, mask=mask0)
    tl.store(acc_b_ptr + t * n_cols + col1, acc_b1, mask=mask1)
    tl.store(m_b_ptr + t, m_b)
    tl.store(s_b_ptr + t, s_b)


@triton.jit
def _attnres_combine_kernel(
    prefix_ptr,  # [T, H]
    wp_ptr,  # [H] precomputed rms_w * res_w
    outw_ptr,  # out-norm weight or dummy
    m_ptr,
    s_ptr,
    acc_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    stride_p,
    stride_o,
    eps,
    HAS_OUTNORM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fold the prefix candidate into the block partial; optional out-norm.

    All operands are read once and stay register-resident.
    """
    t = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    n_iters: tl.constexpr = (n_cols + BLOCK - 1) // BLOCK
    tl.static_assert(n_iters == 2)

    col0 = offs
    col1 = BLOCK + offs
    mask0 = col0 < n_cols
    mask1 = col1 < n_cols

    v0 = tl.load(prefix_ptr + t * stride_p + col0, mask=mask0, other=0.0).to(tl.float32)
    v1 = tl.load(prefix_ptr + t * stride_p + col1, mask=mask1, other=0.0).to(tl.float32)
    wp0 = tl.load(wp_ptr + col0, mask=mask0, other=0.0).to(tl.float32)
    wp1 = tl.load(wp_ptr + col1, mask=mask1, other=0.0).to(tl.float32)
    sq = tl.sum(v0 * v0) + tl.sum(v1 * v1)
    dot = tl.sum(v0 * wp0) + tl.sum(v1 * wp1)
    rsig = tl.math.rsqrt(sq / n_cols + eps)
    logit_p = dot * rsig
    m_b = tl.load(m_ptr + t)
    s_b = tl.load(s_ptr + t)
    m = tl.maximum(m_b, logit_p)
    corr = tl.exp(m_b - m)
    w_p = tl.exp(logit_p - m)
    inv_s = 1.0 / (s_b * corr + w_p)

    a0 = tl.load(acc_ptr + t * n_cols + col0, mask=mask0, other=0.0)
    a1 = tl.load(acc_ptr + t * n_cols + col1, mask=mask1, other=0.0)
    mix0 = ((a0 * corr + w_p * v0) * inv_s).to(tl.bfloat16).to(tl.float32)
    mix1 = ((a1 * corr + w_p * v1) * inv_s).to(tl.bfloat16).to(tl.float32)

    if HAS_OUTNORM:
        mix_sq = tl.sum(mix0 * mix0) + tl.sum(mix1 * mix1)
        rsig_mix = tl.math.rsqrt(mix_sq / n_cols + eps)
        ow0 = tl.load(outw_ptr + col0, mask=mask0, other=0.0).to(tl.float32)
        ow1 = tl.load(outw_ptr + col1, mask=mask1, other=0.0).to(tl.float32)
        tl.store(
            out_ptr + t * stride_o + col0,
            (mix0 * rsig_mix * ow0).to(out_ptr.dtype.element_ty),
            mask=mask0,
        )
        tl.store(
            out_ptr + t * stride_o + col1,
            (mix1 * rsig_mix * ow1).to(out_ptr.dtype.element_ty),
            mask=mask1,
        )
    else:
        tl.store(
            out_ptr + t * stride_o + col0,
            mix0.to(out_ptr.dtype.element_ty),
            mask=mask0,
        )
        tl.store(
            out_ptr + t * stride_o + col1,
            mix1.to(out_ptr.dtype.element_ty),
            mask=mask1,
        )


def attnres_partial(blocks, wp, eps, scratch):
    """Blocks-side online-softmax partial. scratch = (m [T], s [T], acc [T,H] fp32)."""
    KB, T, H = blocks.shape
    m, s_, acc = scratch
    _attnres_partial_kernel[(T,)](
        blocks,
        wp,
        m,
        s_,
        acc,
        KB,
        n_cols=H,
        stride_bk=blocks.stride(0),
        stride_bt=blocks.stride(1),
        eps=eps,
        BLOCK=4096,
        num_warps=8,
    )


def attnres_partial_dual(blocks, wp_a, wp_b, eps, scratch_a, scratch_b):
    """Both mix partials (mlp-side A, next-layer attn-side B) in one sweep.

    Args:
        blocks: ``[KB, T, H]`` block snapshots shared by both sides.
        wp_a/wp_b: precomputed ``rms_w * res_w`` products per side (``[H]``).
        eps: shared RMS epsilon.
        scratch_a/scratch_b: (m [T], s [T], acc [T, H] fp32) per side.
    """
    KB, T, H = blocks.shape
    m_a, s_a, acc_a = scratch_a
    m_b, s_b, acc_b = scratch_b
    _attnres_partial_dual_kernel[(T,)](
        blocks,
        wp_a,
        wp_b,
        m_a,
        s_a,
        acc_a,
        m_b,
        s_b,
        acc_b,
        KB,
        n_cols=H,
        stride_bk=blocks.stride(0),
        stride_bt=blocks.stride(1),
        eps=eps,
        BLOCK=4096,
        num_warps=8,
    )


def attnres_combine(prefix, wp, out_norm_w, eps, scratch, out):
    """Merge the prefix candidate into the partial; optional fused out-norm.

    Args:
        prefix: ``[T, H]`` residual stream.
        scratch: (m, s, acc) from :func:`attnres_partial`.
        out: ``[T, H]`` mixed (and out-normed) hidden destination.

    Returns:
        ``out``.
    """
    T, H = prefix.shape
    m, s_, acc = scratch
    _attnres_combine_kernel[(T,)](
        prefix,
        wp,
        out_norm_w if out_norm_w is not None else wp,
        m,
        s_,
        acc,
        out,
        n_cols=H,
        stride_p=prefix.stride(0),
        stride_o=out.stride(0),
        eps=eps,
        HAS_OUTNORM=out_norm_w is not None,
        BLOCK=4096,
        num_warps=8,
    )
    return out
