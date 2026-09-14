# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
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

"""DSV4 indexer query RoPE, Hadamard transform, and MXFP4 quantization."""

from __future__ import annotations

import functools
import logging

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.dsv4._triton.common import (
    DEEPSEEK_V4_INDEXER_DIM,
    DEEPSEEK_V4_MXFP4_BLOCK_SIZE,
    DEEPSEEK_V4_ROPE_DIM,
    _dsv4_mxfp4_e2m1_nibble,
)

logger = logging.getLogger("tokenspeed_kernel.ops.attention.dsv4.triton")


@triton.jit
def _dsv4_fused_indexer_q_rope_hadamard_mxfp4_kernel(
    positions_ptr,
    index_q_ptr,
    index_q_stride0,
    index_q_stride1,
    cos_sin_cache_ptr,
    cos_sin_cache_stride,
    q_packed_ptr,
    q_packed_stride0,
    q_packed_stride1,
    q_scale_ptr,
    q_scale_stride0,
    q_scale_stride1,
    weights_ptr,
    weights_stride,
    weights_softmax_scale,
    weights_head_scale,
    weights_out_ptr,
    weights_out_stride,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    HALF_BLOCK: tl.constexpr,
    HADAMARD_SCALE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    quant_block_idx = tl.program_id(2)

    pos = tl.load(positions_ptr + token_idx)
    dim = tl.arange(0, TRITON_BLOCK_SIZE)
    q_base = index_q_ptr + token_idx * index_q_stride0 + head_idx * index_q_stride1
    q = tl.load(q_base + dim, mask=dim < HEAD_DIM, other=0.0).to(tl.float32)

    NOPE_DIM: tl.constexpr = HEAD_DIM - ROPE_DIM
    HALF_ROPE: tl.constexpr = ROPE_DIM // 2
    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_DIM // 2

    pair_2d = tl.reshape(q, (NUM_PAIRS, 2))
    even, odd = tl.split(pair_2d)
    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair = pair_idx - NOPE_PAIRS
    is_rope = rope_pair >= 0
    cs_idx = tl.maximum(rope_pair, 0)
    cs_base = cos_sin_cache_ptr + pos * cos_sin_cache_stride
    cos_v = tl.load(cs_base + cs_idx, mask=is_rope, other=1.0).to(tl.float32)
    sin_v = tl.load(cs_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0).to(
        tl.float32
    )
    rotated_even = even * cos_v - odd * sin_v
    rotated_odd = odd * cos_v + even * sin_v
    rotated = tl.interleave(rotated_even, rotated_odd)
    rotated = rotated.to(tl.bfloat16).to(tl.float32)

    in_idx = tl.arange(0, TRITON_BLOCK_SIZE)
    out_idx = quant_block_idx * QUANT_BLOCK + tl.arange(0, QUANT_BLOCK)
    bits = (in_idx[:, None] & out_idx[None, :]).to(tl.int32)
    parity = bits ^ (bits >> 4)
    parity = parity ^ (parity >> 2)
    parity = parity ^ (parity >> 1)
    parity = parity & 1
    signs = tl.where(parity == 0, 1.0, -1.0)
    hadamard = tl.sum(rotated[:, None] * signs, axis=0) * HADAMARD_SCALE
    hadamard = hadamard.to(tl.bfloat16).to(tl.float32)

    hadamard_2d = tl.reshape(hadamard, (HALF_BLOCK, 2))
    x_lo, x_hi = tl.split(hadamard_2d)
    amax = tl.maximum(tl.max(tl.abs(x_lo)), tl.max(tl.abs(x_hi)))
    amax = tl.maximum(amax, 1.0e-4)
    exponent = tl.ceil(tl.log2(amax / 6.0))
    exponent = tl.minimum(tl.maximum(exponent, -127.0), 127.0)
    inv_scale = tl.exp2(-exponent)
    lo = _dsv4_mxfp4_e2m1_nibble(x_lo * inv_scale)
    hi = _dsv4_mxfp4_e2m1_nibble(x_hi * inv_scale)
    packed = lo | (hi << 4)
    scale = (exponent + 127.0).to(tl.uint8)

    packed_base = (
        q_packed_ptr
        + token_idx * q_packed_stride0
        + head_idx * q_packed_stride1
        + quant_block_idx * HALF_BLOCK
    )
    scale_base = (
        q_scale_ptr
        + token_idx * q_scale_stride0
        + head_idx * q_scale_stride1
        + quant_block_idx
    )
    tl.store(packed_base + tl.arange(0, HALF_BLOCK), packed)
    tl.store(scale_base, scale)

    weights = tl.load(weights_ptr + token_idx * weights_stride + head_idx).to(
        tl.float32
    )
    weights = weights * weights_softmax_scale * weights_head_scale
    tl.store(
        weights_out_ptr + token_idx * weights_out_stride + head_idx,
        weights,
        mask=quant_block_idx == 0,
    )


@triton.jit
def _dsv4_fused_indexer_q_rope_hadamard_mxfp4_serial_four_block_kernel(
    positions_ptr,
    index_q_ptr,
    index_q_stride0,
    index_q_stride1,
    cos_sin_cache_ptr,
    cos_sin_cache_stride,
    q_packed_ptr,
    q_packed_stride0,
    q_packed_stride1,
    q_scale_ptr,
    q_scale_stride0,
    q_scale_stride1,
    weights_ptr,
    weights_stride,
    weights_softmax_scale,
    weights_head_scale,
    weights_out_ptr,
    weights_out_stride,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    HALF_BLOCK: tl.constexpr,
    NUM_QUANT_BLOCKS: tl.constexpr,
    HADAMARD_SCALE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    """Reuse Q/RoPE setup while producing the four fixed MXFP4 blocks."""

    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    pos = tl.load(positions_ptr + token_idx)
    dim = tl.arange(0, TRITON_BLOCK_SIZE)
    q_base = index_q_ptr + token_idx * index_q_stride0 + head_idx * index_q_stride1
    q = tl.load(q_base + dim, mask=dim < HEAD_DIM, other=0.0).to(tl.float32)

    NOPE_DIM: tl.constexpr = HEAD_DIM - ROPE_DIM
    HALF_ROPE: tl.constexpr = ROPE_DIM // 2
    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_DIM // 2

    pair_2d = tl.reshape(q, (NUM_PAIRS, 2))
    even, odd = tl.split(pair_2d)
    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair = pair_idx - NOPE_PAIRS
    is_rope = rope_pair >= 0
    cs_idx = tl.maximum(rope_pair, 0)
    cs_base = cos_sin_cache_ptr + pos * cos_sin_cache_stride
    cos_v = tl.load(cs_base + cs_idx, mask=is_rope, other=1.0).to(tl.float32)
    sin_v = tl.load(cs_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0).to(
        tl.float32
    )
    rotated_even = even * cos_v - odd * sin_v
    rotated_odd = odd * cos_v + even * sin_v
    rotated = tl.interleave(rotated_even, rotated_odd)
    rotated = rotated.to(tl.bfloat16).to(tl.float32)

    in_idx = tl.arange(0, TRITON_BLOCK_SIZE)
    for quant_block_idx in tl.static_range(0, NUM_QUANT_BLOCKS):
        out_idx = quant_block_idx * QUANT_BLOCK + tl.arange(0, QUANT_BLOCK)
        bits = (in_idx[:, None] & out_idx[None, :]).to(tl.int32)
        parity = bits ^ (bits >> 4)
        parity = parity ^ (parity >> 2)
        parity = parity ^ (parity >> 1)
        parity = parity & 1
        signs = tl.where(parity == 0, 1.0, -1.0)
        hadamard = tl.sum(rotated[:, None] * signs, axis=0) * HADAMARD_SCALE
        hadamard = hadamard.to(tl.bfloat16).to(tl.float32)

        hadamard_2d = tl.reshape(hadamard, (HALF_BLOCK, 2))
        x_lo, x_hi = tl.split(hadamard_2d)
        amax = tl.maximum(tl.max(tl.abs(x_lo)), tl.max(tl.abs(x_hi)))
        amax = tl.maximum(amax, 1.0e-4)
        exponent = tl.ceil(tl.log2(amax / 6.0))
        exponent = tl.minimum(tl.maximum(exponent, -127.0), 127.0)
        inv_scale = tl.exp2(-exponent)
        lo = _dsv4_mxfp4_e2m1_nibble(x_lo * inv_scale)
        hi = _dsv4_mxfp4_e2m1_nibble(x_hi * inv_scale)
        packed = lo | (hi << 4)
        scale = (exponent + 127.0).to(tl.uint8)

        packed_base = (
            q_packed_ptr
            + token_idx * q_packed_stride0
            + head_idx * q_packed_stride1
            + quant_block_idx * HALF_BLOCK
        )
        scale_base = (
            q_scale_ptr
            + token_idx * q_scale_stride0
            + head_idx * q_scale_stride1
            + quant_block_idx
        )
        tl.store(packed_base + tl.arange(0, HALF_BLOCK), packed)
        tl.store(scale_base, scale)

    weights = tl.load(weights_ptr + token_idx * weights_stride + head_idx).to(
        tl.float32
    )
    weights = weights * weights_softmax_scale * weights_head_scale
    tl.store(weights_out_ptr + token_idx * weights_out_stride + head_idx, weights)


def _dsv4_use_serial_four_block_indexer_q(
    *,
    num_tokens: int,
    head_dim: int,
    rope_dim: int,
    quant_block: int,
    is_cuda: bool,
    is_hip: bool,
    capability: tuple[int, int] | None,
    shapes_valid: bool,
    dtypes_valid: bool,
    devices_valid: bool,
    inner_strides: tuple[int, int, int, int],
) -> bool:
    """Return whether the exact 8192-token SM100 specialization is eligible."""

    return (
        num_tokens == 8192
        and head_dim == DEEPSEEK_V4_INDEXER_DIM
        and rope_dim == DEEPSEEK_V4_ROPE_DIM
        and quant_block == DEEPSEEK_V4_MXFP4_BLOCK_SIZE
        and is_cuda
        and not is_hip
        and capability == (10, 0)
        and shapes_valid
        and dtypes_valid
        and devices_valid
        and inner_strides == (1, 1, 1, 1)
    )


@functools.cache
def _dsv4_indexer_q_cuda_capability(
    device: torch.device | int | None,
) -> tuple[int, int] | None:
    try:
        if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
            return None
        return torch.cuda.get_device_capability(device)
    except (AssertionError, RuntimeError, TypeError, ValueError):
        return None


@functools.cache
def _log_serial_four_block_indexer_q_selection(capability: tuple[int, int]) -> None:
    logger.info(
        "DeepSeek V4 Indexer-Q launch selection: serial_four_block=True "
        "tokens=8192 capability=%s",
        capability,
    )


def _dsv4_serial_four_block_indexer_q_supported(
    index_q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
) -> bool:
    if index_q.dim() != 3:
        return False
    num_tokens, num_heads, head_dim = index_q.shape
    capability = (
        _dsv4_indexer_q_cuda_capability(index_q.device) if num_tokens == 8192 else None
    )
    supported = _dsv4_use_serial_four_block_indexer_q(
        num_tokens=num_tokens,
        head_dim=head_dim,
        rope_dim=DEEPSEEK_V4_ROPE_DIM,
        quant_block=DEEPSEEK_V4_MXFP4_BLOCK_SIZE,
        is_cuda=index_q.is_cuda,
        is_hip=getattr(torch.version, "hip", None) is not None,
        capability=capability,
        shapes_valid=(
            positions.shape == (num_tokens,)
            and cos_sin_cache.dim() == 2
            and cos_sin_cache.shape[1] == DEEPSEEK_V4_ROPE_DIM
            and weights.shape == (num_tokens, num_heads)
        ),
        dtypes_valid=(
            index_q.dtype == torch.bfloat16
            and positions.dtype == torch.int64
            and cos_sin_cache.dtype == torch.float32
            and weights.dtype in (torch.bfloat16, torch.float32)
        ),
        devices_valid=(
            positions.device == index_q.device
            and cos_sin_cache.device == index_q.device
            and weights.device == index_q.device
        ),
        inner_strides=(
            index_q.stride(-1),
            positions.stride(-1) if positions.dim() == 1 else 0,
            cos_sin_cache.stride(-1) if cos_sin_cache.dim() == 2 else 0,
            weights.stride(-1) if weights.dim() == 2 else 0,
        ),
    )
    if supported:
        assert capability is not None
        _log_serial_four_block_indexer_q_selection(capability)
    return supported


def dsv4_fused_indexer_q_rope_hadamard_mxfp4(
    *,
    index_q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
    prefer_serial_four_block: bool,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    num_tokens, num_heads, head_dim = index_q.shape
    q_packed = torch.empty(
        (num_tokens, num_heads, head_dim // 2),
        dtype=torch.uint8,
        device=index_q.device,
    )
    q_scale_bytes = torch.empty(
        (num_tokens, num_heads, head_dim // DEEPSEEK_V4_MXFP4_BLOCK_SIZE),
        dtype=torch.uint8,
        device=index_q.device,
    )
    weights_out = torch.empty_like(weights, dtype=torch.float32)
    if num_tokens == 0:
        return (q_packed, q_scale_bytes.view(torch.int32).squeeze(-1)), weights_out

    use_serial_four_block = prefer_serial_four_block and (
        _dsv4_serial_four_block_indexer_q_supported(
            index_q,
            positions,
            cos_sin_cache,
            weights,
        )
    )
    kernel = (
        _dsv4_fused_indexer_q_rope_hadamard_mxfp4_serial_four_block_kernel
        if use_serial_four_block
        else _dsv4_fused_indexer_q_rope_hadamard_mxfp4_kernel
    )
    grid = (
        (num_tokens, num_heads)
        if use_serial_four_block
        else (num_tokens, num_heads, head_dim // DEEPSEEK_V4_MXFP4_BLOCK_SIZE)
    )
    launch_kwargs = {
        "HEAD_DIM": head_dim,
        "ROPE_DIM": DEEPSEEK_V4_ROPE_DIM,
        "QUANT_BLOCK": DEEPSEEK_V4_MXFP4_BLOCK_SIZE,
        "HALF_BLOCK": DEEPSEEK_V4_MXFP4_BLOCK_SIZE // 2,
        "HADAMARD_SCALE": head_dim**-0.5,
        "TRITON_BLOCK_SIZE": triton.next_power_of_2(head_dim),
        "num_warps": 4,
    }
    if use_serial_four_block:
        launch_kwargs["NUM_QUANT_BLOCKS"] = head_dim // DEEPSEEK_V4_MXFP4_BLOCK_SIZE
    kernel[grid](
        positions,
        index_q,
        index_q.stride(0),
        index_q.stride(1),
        cos_sin_cache,
        cos_sin_cache.stride(0),
        q_packed,
        q_packed.stride(0),
        q_packed.stride(1),
        q_scale_bytes,
        q_scale_bytes.stride(0),
        q_scale_bytes.stride(1),
        weights,
        weights.stride(0),
        softmax_scale,
        head_scale,
        weights_out,
        weights_out.stride(0),
        **launch_kwargs,
    )
    return (
        q_packed,
        q_scale_bytes.view(torch.int32).squeeze(-1).contiguous(),
    ), weights_out
