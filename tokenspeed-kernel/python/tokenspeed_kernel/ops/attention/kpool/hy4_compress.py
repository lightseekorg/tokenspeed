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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Hy4-flash compressed indexer 4:1 gated pooling, M-RoPE, and index-K store ops."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.kvcache.index_k_scale import (
    encode_index_k_scale,
    index_k_scale_ue8m0,
    index_k_scale_word,
    round_index_k_scale,
)


def apply_interleaved_rope_last_dim(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply interleaved (even/odd pair) RoPE in FP32 to ``x``.

    Matches ``HYV4FlashRotaryEmbedding`` + ``apply_rotary_pos_emb``:
    ``rotate_half`` maps ``(x0, x1, x2, x3, ...) -> (-x1, x0, -x3, x2, ...)``.
    ``cos`` and ``sin`` have shape ``[..., rope_dim // 2]`` (one entry per pair)
    or ``[..., rope_dim]`` (duplicated halves ``[c, c]``).
    """
    orig_dtype = x.dtype
    rope_dim = x.shape[-1]
    half = rope_dim // 2
    if cos.shape[-1] == rope_dim:
        cos_half = cos[..., :half]
        sin_half = sin[..., :half]
    elif cos.shape[-1] == half:
        cos_half = cos
        sin_half = sin
    else:
        raise ValueError(
            f"cos/sin last dim ({cos.shape[-1]}) incompatible with rope_dim={rope_dim}"
        )
    xf = x.float()
    x_even = xf[..., 0::2]
    x_odd = xf[..., 1::2]
    cf = cos_half.float()
    sf = sin_half.float()
    out_even = x_even * cf - x_odd * sf
    out_odd = x_odd * cf + x_even * sf
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2).to(orig_dtype)


def compute_hy4_mrope_cos_sin(
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    mrope_section: tuple[int, int, int] | list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute FP32 interleaved 3D M-RoPE ``(cos, sin)`` of shape ``[T, half]``.

    Args:
        positions: ``[T]`` (1D text) or ``[3, T]`` (3D T/H/W M-RoPE).
        inv_freq: ``[half]`` inverse frequencies on the target device.
        mrope_section: Three counts ``(s0, s1, s2)`` summing to ``half``.

    Returns:
        ``(cos, sin)`` in FP32 with shape ``[T, half]``.
    """
    if positions.ndim == 1:
        pos_3d = positions.unsqueeze(0).expand(3, -1)
    elif positions.ndim == 2 and positions.shape[0] == 3:
        pos_3d = positions
    else:
        raise ValueError(
            f"positions must have shape [T] or [3, T], got {tuple(positions.shape)}"
        )
    freqs = pos_3d.float().unsqueeze(-1) * inv_freq.to(
        device=positions.device, dtype=torch.float32
    ).view(1, 1, -1)
    cos_3d = freqs.cos()
    sin_3d = freqs.sin()

    cos = cos_3d[0].clone()
    sin = sin_3d[0].clone()
    for dim_idx, offset in ((1, 1), (2, 2)):
        length = int(mrope_section[dim_idx]) * 3
        sl = slice(offset, length, 3)
        cos[..., sl] = cos_3d[dim_idx, ..., sl]
        sin[..., sl] = sin_3d[dim_idx, ..., sl]
    return cos, sin


@triton.jit(
    do_not_specialize=[
        "num_rows",
        "index_num_pages",
        "stride_iv_page",
        "stride_is_page",
    ]
)
def _hy4_index_k_quantize_scatter_kernel(
    keys_ptr,
    write_slots_ptr,
    index_values_ptr,
    index_scales_ptr,
    num_rows,
    index_num_pages,
    stride_kr,
    stride_iv_page,
    stride_iv_row,
    stride_is_page,
    stride_is_row,
    INDEX_ROWS_PER_PAGE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    UE8M0: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if row >= num_rows:
        return
    offs = tl.arange(0, BLOCK_D)
    mask = offs < HEAD_DIM
    x = tl.load(keys_ptr + row * stride_kr + offs, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-4)
    scale, word = index_k_scale_word(absmax / 448.0, UE8M0)
    quantized = tl.minimum(tl.maximum(x / scale, -448.0), 448.0)

    slot = tl.load(write_slots_ptr + row).to(tl.int64)
    page = slot // INDEX_ROWS_PER_PAGE
    index_row = slot % INDEX_ROWS_PER_PAGE
    valid = (slot >= 0) & (page < index_num_pages)
    value_base = page * stride_iv_page + index_row * stride_iv_row
    scale_base = page * stride_is_page + index_row * stride_is_row
    tl.store(
        index_values_ptr + value_base + offs,
        quantized.to(index_values_ptr.dtype.element_ty),
        mask=mask & valid,
    )
    tl.store(index_scales_ptr + scale_base, word, mask=valid)


def hy4_pool_group_keys(
    grouped_token_keys: torch.Tensor,
    key_value_weight: torch.Tensor,
    key_gate_weight: torch.Tensor,
    intra_group_ape: torch.Tensor,
    key_norm_weight: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Pool ``[G, K, D]`` normalized token keys into ``[G, D]`` group keys.

    Exact match to ``HYV4FlashCompressedIndexer.pool_groups``:
    values and gates are projected in FP32, ``intra_group_ape`` is added to the
    gates, softmax is computed along the intra-group axis in FP32, and the
    weighted sum is cast back to ``grouped_token_keys.dtype`` before RMSNorm
    (``key_norm``).
    """
    orig_dtype = grouped_token_keys.dtype
    group_f = grouped_token_keys.float()
    values = F.linear(group_f, key_value_weight.float())
    gates = F.linear(
        group_f, key_gate_weight.float()
    ) + intra_group_ape.float().unsqueeze(0)
    weights = F.softmax(gates, dim=1)
    pooled = (values * weights).sum(dim=1).to(orig_dtype)
    pooled_f = pooled.float()
    normed = (
        pooled_f * torch.rsqrt(pooled_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    ).to(orig_dtype)
    return key_norm_weight.to(orig_dtype) * normed


def hy4_quantize_and_store_index_k(
    group_keys: torch.Tensor,
    write_slots: torch.Tensor,
    index_values: torch.Tensor,
    index_scales: torch.Tensor,
) -> None:
    """Quantize ``[G, 128]`` Hy4-flash pooled group keys to FP8+scale and scatter."""
    num_rows, head_dim = group_keys.shape
    if num_rows == 0:
        return
    if head_dim != 128:
        raise ValueError(
            f"hy4_quantize_and_store_index_k expects head_dim=128, got {head_dim}"
        )
    if not group_keys.is_cuda:
        rows_per_page = index_values.shape[1]
        num_pages = index_values.shape[0]
        xf = group_keys.float()
        absmax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
        scale = round_index_k_scale(absmax / 448.0)
        q = (xf / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        encoded = encode_index_k_scale(scale)
        slots = write_slots.to(torch.int64)
        valid = slots >= 0
        pages = torch.div(slots, rows_per_page, rounding_mode="floor")
        row_ids = torch.remainder(slots, rows_per_page)
        valid = valid & (pages < num_pages)
        if bool(valid.any()):
            vp = pages[valid]
            vr = row_ids[valid]
            index_values.view(torch.uint8)[vp, vr] = q.view(torch.uint8)[valid]
            index_scales.view(torch.int32)[vp, vr, 0] = encoded.view(torch.int32)[
                valid, 0
            ]
        return

    keys_contig = group_keys.contiguous()
    slots_contig = write_slots.contiguous()
    # `index_k_scale_word` emits raw int32 scale bit patterns; view `index_scales`
    # as int32 so Triton stores raw bits instead of int-to-float casting on SM90.
    scales_i32 = index_scales.view(torch.int32)
    _hy4_index_k_quantize_scatter_kernel[(num_rows,)](
        keys_contig,
        slots_contig,
        index_values,
        scales_i32,
        num_rows,
        index_values.shape[0],
        keys_contig.stride(0),
        index_values.stride(0),
        index_values.stride(1),
        scales_i32.stride(0),
        scales_i32.stride(1),
        INDEX_ROWS_PER_PAGE=index_values.shape[1],
        HEAD_DIM=head_dim,
        UE8M0=index_k_scale_ue8m0(),
        BLOCK_D=128,
        num_warps=4,
    )


__all__ = [
    "apply_interleaved_rope_last_dim",
    "compute_hy4_mrope_cos_sin",
    "hy4_pool_group_keys",
    "hy4_quantize_and_store_index_k",
]
