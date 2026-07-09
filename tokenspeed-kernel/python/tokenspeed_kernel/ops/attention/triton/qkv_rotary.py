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

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _packed_qkv_complex_rotary_kernel(
    QKV,
    FREQS,
    Q_OUT,
    K_OUT,
    V_OUT,
    total_tokens: tl.constexpr,
    packed_stride: tl.constexpr,
    q_offset: tl.constexpr,
    k_offset: tl.constexpr,
    v_offset: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    freqs_stride_t: tl.constexpr,
    freqs_stride_pair: tl.constexpr,
    freqs_stride_ri: tl.constexpr,
    head_dim: tl.constexpr,
    copy_v: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    block_t = tl.program_id(0)
    head = tl.program_id(1)

    offs_t = block_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_p = tl.arange(0, BLOCK_P)
    half = head_dim // 2
    mask = (offs_t[:, None] < total_tokens) & (offs_p[None, :] < half)

    even_d = offs_p * 2
    odd_d = even_d + 1
    q_base = QKV + offs_t[:, None] * packed_stride + q_offset + head * head_dim
    k_base = QKV + offs_t[:, None] * packed_stride + k_offset + head * head_dim

    q_even = tl.load(q_base + even_d[None, :], mask=mask, other=0.0).to(tl.float32)
    q_odd = tl.load(q_base + odd_d[None, :], mask=mask, other=0.0).to(tl.float32)
    k_even = tl.load(k_base + even_d[None, :], mask=mask, other=0.0).to(tl.float32)
    k_odd = tl.load(k_base + odd_d[None, :], mask=mask, other=0.0).to(tl.float32)

    freq_base = (
        FREQS + offs_t[:, None] * freqs_stride_t + offs_p[None, :] * freqs_stride_pair
    )
    real = tl.load(freq_base, mask=mask, other=0.0).to(tl.float32)
    imag = tl.load(freq_base + freqs_stride_ri, mask=mask, other=0.0).to(tl.float32)

    q_even_out = q_even * real - q_odd * imag
    q_odd_out = q_odd * real + q_even * imag
    k_even_out = k_even * real - k_odd * imag
    k_odd_out = k_odd * real + k_even * imag

    out_base = offs_t[:, None] * out_stride_t + head * out_stride_h
    tl.store(Q_OUT + out_base + even_d[None, :], q_even_out, mask=mask)
    tl.store(Q_OUT + out_base + odd_d[None, :], q_odd_out, mask=mask)
    tl.store(K_OUT + out_base + even_d[None, :], k_even_out, mask=mask)
    tl.store(K_OUT + out_base + odd_d[None, :], k_odd_out, mask=mask)

    if copy_v:
        v_base = QKV + offs_t[:, None] * packed_stride + v_offset + head * head_dim
        v_even = tl.load(v_base + even_d[None, :], mask=mask, other=0.0)
        v_odd = tl.load(v_base + odd_d[None, :], mask=mask, other=0.0)
        tl.store(V_OUT + out_base + even_d[None, :], v_even, mask=mask)
        tl.store(V_OUT + out_base + odd_d[None, :], v_odd, mask=mask)


def packed_qkv_complex_rotary(
    qkv: torch.Tensor,
    q_size: int,
    kv_size: int,
    num_heads: int,
    head_dim: int,
    freqs_cis: torch.Tensor,
    *,
    copy_v: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply Kimi-style complex RoPE while splitting packed QKV.

    Kimi's reference path views adjacent head-dimension pairs as complex
    numbers and multiplies them by ``freqs_cis``. This helper performs the same
    pairwise rotation from packed QKV, materializing contiguous Q/K outputs for
    FA4 while leaving V as a packed-QKV view by default.

    Args:
        qkv: Packed QKV tensor shaped ``[batch, seq, q + k + v]`` or
            ``[tokens, q + k + v]`` with contiguous last dimension.
        q_size: Packed Q width in elements for this rank.
        kv_size: Packed K/V width in elements for this rank.
        num_heads: Number of Q/K/V heads on this rank.
        head_dim: Per-head dimension. Must be even.
        freqs_cis: Complex tensor shaped ``[tokens, head_dim / 2]``.
        copy_v: If true, materialize contiguous V. Otherwise return a strided
            view into the packed input.

    Returns:
        ``(q, k, v)`` tensors shaped ``[tokens, heads, head_dim]``. Q and K are
        contiguous complex-RoPE outputs. V is contiguous only when
        ``copy_v=True``.
    """
    qkv_flat = qkv.reshape(-1, qkv.shape[-1])
    total_tokens = qkv_flat.shape[0]
    packed_stride = qkv_flat.stride(0)
    assert head_dim % 2 == 0
    assert q_size == num_heads * head_dim
    assert kv_size == num_heads * head_dim
    assert freqs_cis.shape == (total_tokens, head_dim // 2)
    assert freqs_cis.is_complex()

    q_out = torch.empty(
        (total_tokens, num_heads, head_dim), device=qkv.device, dtype=qkv.dtype
    )
    k_out = torch.empty_like(q_out)
    if copy_v:
        v_out = torch.empty_like(q_out)
    else:
        v_view = qkv_flat[:, q_size + kv_size : q_size + kv_size + kv_size]
        v_out = v_view.reshape(total_tokens, num_heads, head_dim)

    freqs_real = torch.view_as_real(freqs_cis)
    block_t = 16
    block_p = triton.next_power_of_2(head_dim // 2)
    grid = (triton.cdiv(total_tokens, block_t), num_heads)
    _packed_qkv_complex_rotary_kernel[grid](
        qkv_flat,
        freqs_real,
        q_out,
        k_out,
        v_out,
        total_tokens=total_tokens,
        packed_stride=packed_stride,
        q_offset=0,
        k_offset=q_size,
        v_offset=q_size + kv_size,
        out_stride_t=q_out.stride(0),
        out_stride_h=q_out.stride(1),
        freqs_stride_t=freqs_real.stride(0),
        freqs_stride_pair=freqs_real.stride(1),
        freqs_stride_ri=freqs_real.stride(2),
        head_dim=head_dim,
        copy_v=copy_v,
        BLOCK_T=block_t,
        BLOCK_P=block_p,
    )
    return q_out, k_out, v_out


@triton.jit
def _packed_qkv_neox_rotary_kernel(
    QKV,
    COS,
    SIN,
    Q_OUT,
    K_OUT,
    V_OUT,
    total_tokens: tl.constexpr,
    packed_stride: tl.constexpr,
    q_offset: tl.constexpr,
    k_offset: tl.constexpr,
    v_offset: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    cos_stride_t: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    copy_v: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    block_t = tl.program_id(0)
    head = tl.program_id(1)

    offs_t = block_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)
    mask = (offs_t[:, None] < total_tokens) & (offs_d[None, :] < head_dim)

    half = head_dim // 2
    rot_d = tl.where(offs_d < half, offs_d + half, offs_d - half)
    rot_sign = tl.where(offs_d < half, -1.0, 1.0)
    rot_mask = (offs_t[:, None] < total_tokens) & (rot_d[None, :] < head_dim)

    q_ptr = (
        QKV
        + offs_t[:, None] * packed_stride
        + q_offset
        + head * head_dim
        + offs_d[None, :]
    )
    k_ptr = (
        QKV
        + offs_t[:, None] * packed_stride
        + k_offset
        + head * head_dim
        + offs_d[None, :]
    )
    q_rot_ptr = (
        QKV
        + offs_t[:, None] * packed_stride
        + q_offset
        + head * head_dim
        + rot_d[None, :]
    )
    k_rot_ptr = (
        QKV
        + offs_t[:, None] * packed_stride
        + k_offset
        + head * head_dim
        + rot_d[None, :]
    )

    cos_ptr = COS + offs_t[:, None] * cos_stride_t + offs_d[None, :]
    sin_ptr = SIN + offs_t[:, None] * cos_stride_t + offs_d[None, :]

    q = tl.load(q_ptr, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(k_ptr, mask=mask, other=0.0).to(tl.float32)
    q_rot = (
        tl.load(q_rot_ptr, mask=rot_mask, other=0.0).to(tl.float32) * rot_sign[None, :]
    )
    k_rot = (
        tl.load(k_rot_ptr, mask=rot_mask, other=0.0).to(tl.float32) * rot_sign[None, :]
    )
    cos = tl.load(cos_ptr, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr, mask=mask, other=0.0).to(tl.float32)

    q_out = q * cos + q_rot * sin
    k_out = k * cos + k_rot * sin

    out_ptr = offs_t[:, None] * out_stride_t + head * out_stride_h + offs_d[None, :]
    tl.store(Q_OUT + out_ptr, q_out, mask=mask)
    tl.store(K_OUT + out_ptr, k_out, mask=mask)

    if copy_v:
        v_ptr = (
            QKV
            + offs_t[:, None] * packed_stride
            + v_offset
            + head * head_dim
            + offs_d[None, :]
        )
        v = tl.load(v_ptr, mask=mask, other=0.0)
        tl.store(V_OUT + out_ptr, v, mask=mask)


def packed_qkv_neox_rotary(
    qkv: torch.Tensor,
    q_size: int,
    kv_size: int,
    num_heads: int,
    head_dim: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    copy_v: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply NeoX rotary while splitting packed QKV for vision attention.

    The helper is backend-agnostic: it prepares Q/K/V tensors from a packed QKV
    projection. The current runtime caller uses it for the Qwen vision FA4 path.

    Args:
        qkv: Packed QKV tensor shaped ``[batch, seq, q + k + v]`` or
            ``[tokens, q + k + v]`` with contiguous last dimension.
        q_size: Packed Q width in elements for this rank.
        kv_size: Packed K/V width in elements for this rank.
        num_heads: Number of Q/K/V heads on this rank.
        head_dim: Per-head dimension.
        cos: Full-width rotary cos tensor shaped ``[tokens, head_dim]``.
        sin: Full-width rotary sin tensor shaped ``[tokens, head_dim]``.
        copy_v: If true, materialize contiguous V. Otherwise return a strided
            view into the packed input.

    Returns:
        ``(q, k, v)`` tensors shaped ``[tokens, heads, head_dim]``. Q and K are
        contiguous rotary outputs. V is contiguous only when ``copy_v=True``.
    """
    qkv_flat = qkv.reshape(-1, qkv.shape[-1])
    total_tokens = qkv_flat.shape[0]
    packed_stride = qkv_flat.stride(0)
    assert q_size == num_heads * head_dim
    assert kv_size == num_heads * head_dim
    assert cos.shape == (total_tokens, head_dim)
    assert sin.shape == (total_tokens, head_dim)

    q_out = torch.empty(
        (total_tokens, num_heads, head_dim), device=qkv.device, dtype=qkv.dtype
    )
    k_out = torch.empty_like(q_out)
    if copy_v:
        v_out = torch.empty_like(q_out)
    else:
        v_view = qkv_flat[:, q_size + kv_size : q_size + kv_size + kv_size]
        v_out = v_view.reshape(total_tokens, num_heads, head_dim)

    block_t = 16
    block_d = triton.next_power_of_2(head_dim)
    grid = (triton.cdiv(total_tokens, block_t), num_heads)
    _packed_qkv_neox_rotary_kernel[grid](
        qkv_flat,
        cos,
        sin,
        q_out,
        k_out,
        v_out,
        total_tokens=total_tokens,
        packed_stride=packed_stride,
        q_offset=0,
        k_offset=q_size,
        v_offset=q_size + kv_size,
        out_stride_t=q_out.stride(0),
        out_stride_h=q_out.stride(1),
        cos_stride_t=cos.stride(0),
        num_heads=num_heads,
        head_dim=head_dim,
        copy_v=copy_v,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
    )
    return q_out, k_out, v_out


__all__ = ["packed_qkv_complex_rotary", "packed_qkv_neox_rotary"]
