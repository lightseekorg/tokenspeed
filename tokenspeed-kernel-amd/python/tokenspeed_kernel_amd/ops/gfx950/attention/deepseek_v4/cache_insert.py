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

"""DeepSeek V4 CSA indexer-cache insertion for AMD GFX950."""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl

__all__ = ["gluon_deepseek_v4_fused_csa_indexer_fp8_cache_insert_gfx950"]


@gluon.jit
def _deepseek_v4_fused_csa_indexer_fp8_cache_insert_kernel(
    state_cache,
    token_to_req_indices,
    positions,
    compressor_slot_mapping,
    block_table,
    block_table_base_offsets,
    rms_norm_weight,
    cos_sin_cache,
    k_cache_fp8,
    k_cache_scale,
    kv_slot_mapping,
    state_cache_stride0: tl.int64,
    state_cache_stride1: tl.int64,
    block_table_stride0: tl.int64,
    cos_sin_stride0: tl.int64,
    cache_stride_bytes: tl.int64,
    state_block_size: tl.int32,
    block_table_rows: tl.int32,
    block_table_width: tl.int32,
    cos_sin_rows: tl.int64,
    cache_rows: tl.int64,
    rms_norm_eps: tl.float32,
    HAS_BLOCK_TABLE_BASE: gl.constexpr,
):
    token_idx = gl.program_id(0)

    state_slot = gl.load(compressor_slot_mapping + token_idx).to(gl.int64)
    if state_slot < 0:
        return

    position = gl.load(positions + token_idx).to(gl.int64)
    if position < 0 or (position + 1) % 4 != 0:
        return

    kv_slot = gl.load(kv_slot_mapping + token_idx).to(gl.int64)
    if kv_slot < 0 or kv_slot >= cache_rows * 64:
        return

    req_idx = gl.load(token_to_req_indices + token_idx).to(gl.int64)
    if req_idx < 0 or req_idx >= block_table_rows:
        return

    compressed_position = (position // 4) * 4
    if compressed_position >= cos_sin_rows:
        return

    if HAS_BLOCK_TABLE_BASE:
        base_logical_page = gl.load(block_table_base_offsets + req_idx).to(gl.int64)
    else:
        base_logical_page = gl.full((), 0, gl.int64)

    compression_layout: gl.constexpr = gl.BlockedLayout(
        [1, 2],
        [1, 64],
        [2, 1],
        [1, 0],
    )
    window_layout: gl.constexpr = gl.SliceLayout(1, compression_layout)
    dim_layout: gl.constexpr = gl.SliceLayout(0, compression_layout)
    window_offsets = gl.arange(0, 8, layout=window_layout)
    dims = gl.arange(0, 128, layout=dim_layout)

    window_positions = position - 7 + window_offsets
    valid_positions = window_positions >= 0
    table_indices = window_positions // state_block_size - base_logical_page
    valid_positions &= (table_indices >= 0) & (table_indices < block_table_width)
    safe_table_indices = gl.maximum(table_indices, 0)
    block_numbers = gl.load(
        block_table + req_idx * block_table_stride0 + safe_table_indices.to(gl.int64),
        mask=valid_positions,
        other=-1,
    ).to(gl.int64)
    valid_positions &= block_numbers >= 0

    safe_blocks = gl.maximum(block_numbers, 0)
    safe_positions = gl.maximum(window_positions, 0) % state_block_size
    head_offsets = gl.where(window_offsets >= 4, 128, 0)
    row_offsets = (
        safe_blocks[:, None] * state_cache_stride0
        + safe_positions[:, None].to(gl.int64) * state_cache_stride1
        + head_offsets[:, None].to(gl.int64)
        + dims[None, :].to(gl.int64)
    )
    valid_rows = valid_positions[:, None]
    scores = gl.load(
        state_cache + row_offsets + 256,
        mask=valid_rows,
        other=-1.0e30,
    ).to(gl.float32)
    score_max = gl.max(scores, axis=0)
    probabilities = gl.exp(scores - score_max[None, :])
    probability_sum = gl.sum(probabilities, axis=0)
    values = gl.load(
        state_cache + row_offsets,
        mask=valid_rows,
        other=0.0,
    ).to(gl.float32)
    compressed = gl.sum(values * probabilities, axis=0) / probability_sum

    variance = gl.sum(compressed * compressed, axis=0) / 128.0
    rms_weight = gl.load(rms_norm_weight + dims).to(gl.float32)
    normed = compressed * gl.rsqrt(variance + rms_norm_eps) * rms_weight

    pairs = normed.reshape([64, 2])
    even, odd = gl.split(pairs)
    pair_indices = gl.arange(0, 64, layout=even.type.layout)
    rope_pairs = pair_indices - 32
    is_rope = rope_pairs >= 0
    safe_rope_pairs = gl.maximum(rope_pairs, 0)
    cos_sin_base = compressed_position * cos_sin_stride0
    cos_values = gl.load(
        cos_sin_cache + cos_sin_base + safe_rope_pairs,
        mask=is_rope,
        other=1.0,
    ).to(gl.float32)
    sin_values = gl.load(
        cos_sin_cache + cos_sin_base + 32 + safe_rope_pairs,
        mask=is_rope,
        other=0.0,
    ).to(gl.float32)
    rotated_even = even * cos_values - odd * sin_values
    rotated_odd = even * sin_values + odd * cos_values
    rotated = gl.join(rotated_even, rotated_odd).reshape([128])
    rotated = rotated.to(gl.bfloat16).to(gl.float32)

    # Give every output lane the complete input row. This is the conservative
    # Gluon equivalent of the established Triton Walsh-Hadamard sign reduction.
    hadamard_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1],
        [1, 64],
        [1, 2],
        [1, 0],
    )
    hadamard_input_layout: gl.constexpr = gl.SliceLayout(1, hadamard_layout)
    hadamard_output_layout: gl.constexpr = gl.SliceLayout(0, hadamard_layout)
    hadamard_input = gl.convert_layout(rotated, hadamard_input_layout)
    input_indices = gl.arange(0, 128, layout=hadamard_input_layout)
    output_indices = gl.arange(0, 128, layout=hadamard_output_layout)
    parity = (input_indices[:, None] & output_indices[None, :]).to(gl.int32)
    parity ^= parity >> 4
    parity ^= parity >> 2
    parity ^= parity >> 1
    signs = gl.where((parity & 1) == 0, 1.0, -1.0)
    hadamard = gl.sum(hadamard_input[:, None] * signs, axis=0) * (128.0**-0.5)
    hadamard = hadamard.to(gl.bfloat16).to(gl.float32)

    scale_input = gl.maximum(gl.max(gl.abs(hadamard), axis=0) / 448.0, 1.0e-10)
    scale = gl.exp2(gl.ceil(gl.log2(scale_input)))
    quantized = gl.minimum(gl.maximum(hadamard / scale, -448.0), 448.0).to(
        gl.float8e4nv
    )

    cache_page = kv_slot // 64
    cache_position = kv_slot % 64
    value_base = cache_page * cache_stride_bytes + cache_position * 128
    gl.store(k_cache_fp8 + value_base + output_indices, quantized)

    scale_stride = cache_stride_bytes // 4
    scale_base = cache_page * scale_stride + 64 * 32 + cache_position
    gl.store(
        k_cache_scale + scale_base + output_indices, scale, mask=output_indices == 0
    )


def _check_tensor(name: str, value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


def _check_metadata_tensor(
    name: str,
    tensor: torch.Tensor,
    device: torch.device,
    dtypes: tuple[torch.dtype, ...],
) -> None:
    if tensor.dtype not in dtypes:
        expected = ", ".join(str(dtype) for dtype in dtypes)
        raise TypeError(f"{name} must have dtype in ({expected}), got {tensor.dtype}")
    if tensor.dim() != 1:
        raise ValueError(f"{name} must be 1-D, got {tuple(tensor.shape)}")
    if tensor.device != device or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous and on {device}")


def gluon_deepseek_v4_fused_csa_indexer_fp8_cache_insert_gfx950(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    compressor_slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    compressor_block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    kv_cache_2d: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
    compress_ratio: int,
    block_table_base_offsets: torch.Tensor | None = None,
) -> None:
    """Compress CSA indexer state and insert page-planar FP8 rows on GFX950.

    Args:
        state_cache: FP32 paged compressor state shaped
            `[pages, compressor_block_size, 512]`. Each row contains 256 KV
            values followed by 256 score values.
        token_to_req_indices: Contiguous int32 request index for each input row.
        positions: Contiguous int32 or int64 absolute positions.
        compressor_slot_mapping: Contiguous int32 or int64 state slots. A
            negative slot suppresses the row before any compression work.
        block_table: Contiguous int32 logical-to-physical state-page table.
        compressor_block_size: Number of state rows per page.
        rms_norm_weight: Contiguous FP32 or BF16 RMSNorm weight of width 128.
        rms_norm_eps: Positive finite RMSNorm epsilon.
        cos_sin_cache: FP32 or BF16 fused GPT-J cos/sin rows shaped `[N, 64]`.
        kv_cache_2d: Uint8 page-planar output. Every page contains 64 by 128
            E4M3 value bytes followed by 64 FP32 scales.
        kv_slot_mapping: Contiguous int32 or int64 compressed-cache slots. A
            negative slot suppresses the row.
        kv_cache_block_size: Output page size. GFX950 supports 64.
        compress_ratio: CSA compression ratio. This kernel supports 4.
        block_table_base_offsets: Optional contiguous int32 or int64 logical
            page base for each request.

    Returns:
        None. Rows are written only when `(position + 1) % 4 == 0` and both
        state and output slots are valid.
    """

    state_cache = _check_tensor("state_cache", state_cache)
    token_to_req_indices = _check_tensor("token_to_req_indices", token_to_req_indices)
    positions = _check_tensor("positions", positions)
    compressor_slot_mapping = _check_tensor(
        "compressor_slot_mapping", compressor_slot_mapping
    )
    block_table = _check_tensor("block_table", block_table)
    rms_norm_weight = _check_tensor("rms_norm_weight", rms_norm_weight)
    cos_sin_cache = _check_tensor("cos_sin_cache", cos_sin_cache)
    kv_cache_2d = _check_tensor("kv_cache_2d", kv_cache_2d)
    kv_slot_mapping = _check_tensor("kv_slot_mapping", kv_slot_mapping)
    if block_table_base_offsets is not None:
        block_table_base_offsets = _check_tensor(
            "block_table_base_offsets", block_table_base_offsets
        )

    if state_cache.dtype != torch.float32:
        raise TypeError(f"state_cache must be FP32, got {state_cache.dtype}")
    if state_cache.dim() != 3 or state_cache.shape[2] != 512:
        raise ValueError(
            "state_cache must have shape [pages, compressor_block_size, 512], "
            f"got {tuple(state_cache.shape)}"
        )
    if not state_cache.is_cuda:
        raise ValueError("state_cache must be on an AMD GPU")
    if state_cache.stride(2) != 1 or state_cache.stride(1) < 512:
        raise ValueError(
            "state_cache must have unit inner stride and non-overlapping rows"
        )
    device = state_cache.device

    if isinstance(compressor_block_size, bool) or not isinstance(
        compressor_block_size, int
    ):
        raise TypeError("compressor_block_size must be an int")
    if compressor_block_size <= 0 or compressor_block_size != state_cache.shape[1]:
        raise ValueError(
            "compressor_block_size must be positive and match state_cache.shape[1]"
        )
    if kv_cache_block_size != 64:
        raise ValueError(
            f"GFX950 FP8 indexer insertion requires kv_cache_block_size=64, got "
            f"{kv_cache_block_size}"
        )
    if compress_ratio != 4:
        raise ValueError(
            f"GFX950 CSA indexer insertion requires compress_ratio=4, got "
            f"{compress_ratio}"
        )

    _check_metadata_tensor(
        "token_to_req_indices", token_to_req_indices, device, (torch.int32,)
    )
    integer_dtypes = (torch.int32, torch.int64)
    _check_metadata_tensor("positions", positions, device, integer_dtypes)
    _check_metadata_tensor(
        "compressor_slot_mapping",
        compressor_slot_mapping,
        device,
        integer_dtypes,
    )
    _check_metadata_tensor("kv_slot_mapping", kv_slot_mapping, device, integer_dtypes)

    num_actual = min(
        compressor_slot_mapping.numel(), positions.numel(), kv_slot_mapping.numel()
    )
    if token_to_req_indices.numel() < num_actual:
        raise ValueError("token_to_req_indices is shorter than the insertion inputs")

    if block_table.dtype != torch.int32:
        raise TypeError(f"block_table must be int32, got {block_table.dtype}")
    if block_table.dim() != 2 or block_table.shape[0] == 0 or block_table.shape[1] == 0:
        raise ValueError(
            f"block_table must be non-empty and 2-D, got {tuple(block_table.shape)}"
        )
    if (
        block_table.device != device
        or block_table.stride(1) != 1
        or block_table.stride(0) < block_table.shape[1]
    ):
        raise ValueError("block_table must have contiguous rows on the state device")
    if block_table_base_offsets is not None:
        _check_metadata_tensor(
            "block_table_base_offsets",
            block_table_base_offsets,
            device,
            integer_dtypes,
        )
        if block_table_base_offsets.numel() < block_table.shape[0]:
            raise ValueError(
                "block_table_base_offsets must provide one value per block-table row"
            )

    if rms_norm_weight.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(
            f"rms_norm_weight must be FP32 or BF16, got {rms_norm_weight.dtype}"
        )
    if (
        rms_norm_weight.shape != (128,)
        or rms_norm_weight.device != device
        or not rms_norm_weight.is_contiguous()
    ):
        raise ValueError("rms_norm_weight must be contiguous [128] on the state device")
    try:
        eps = float(rms_norm_eps)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError("rms_norm_eps must be a finite positive real scalar") from error
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("rms_norm_eps must be finite and positive")

    if cos_sin_cache.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(
            f"cos_sin_cache must be FP32 or BF16, got {cos_sin_cache.dtype}"
        )
    if (
        cos_sin_cache.dim() != 2
        or cos_sin_cache.shape[0] == 0
        or cos_sin_cache.shape[1] != 64
    ):
        raise ValueError(
            f"cos_sin_cache must have shape [nonzero, 64], got "
            f"{tuple(cos_sin_cache.shape)}"
        )
    if cos_sin_cache.device != device or cos_sin_cache.stride(1) != 1:
        raise ValueError(
            "cos_sin_cache must have unit inner stride on the state device"
        )

    if kv_cache_2d.dtype != torch.uint8:
        raise TypeError(f"kv_cache_2d must be uint8, got {kv_cache_2d.dtype}")
    if (
        kv_cache_2d.dim() != 2
        or kv_cache_2d.shape[0] == 0
        or kv_cache_2d.shape[1] < 64 * (128 + 4)
    ):
        raise ValueError(
            "kv_cache_2d must have shape [nonzero pages, >= 8448], got "
            f"{tuple(kv_cache_2d.shape)}"
        )
    if (
        kv_cache_2d.device != device
        or kv_cache_2d.stride(1) != 1
        or kv_cache_2d.stride(0) < kv_cache_2d.shape[1]
        or kv_cache_2d.stride(0) % 4 != 0
        or kv_cache_2d.shape[1] % 4 != 0
        or kv_cache_2d.storage_offset() % 4 != 0
    ):
        raise ValueError(
            "kv_cache_2d must be an aligned, unit-inner-stride page view on the "
            "state device"
        )

    if num_actual == 0:
        return
    if torch.version.hip is None:
        raise RuntimeError("GFX950 Gluon cache insertion requires a ROCm build")

    cache_fp8 = kv_cache_2d.view(torch.float8_e4m3fn)
    cache_scale = kv_cache_2d.view(torch.float32)
    base_offsets_arg = (
        block_table_base_offsets
        if block_table_base_offsets is not None
        else token_to_req_indices
    )
    _deepseek_v4_fused_csa_indexer_fp8_cache_insert_kernel[(num_actual,)](
        state_cache,
        token_to_req_indices[:num_actual],
        positions[:num_actual],
        compressor_slot_mapping[:num_actual],
        block_table,
        base_offsets_arg,
        rms_norm_weight,
        cos_sin_cache,
        cache_fp8,
        cache_scale,
        kv_slot_mapping[:num_actual],
        state_cache.stride(0),
        state_cache.stride(1),
        block_table.stride(0),
        cos_sin_cache.stride(0),
        kv_cache_2d.stride(0),
        compressor_block_size,
        block_table.shape[0],
        block_table.shape[1],
        cos_sin_cache.shape[0],
        kv_cache_2d.shape[0],
        eps,
        HAS_BLOCK_TABLE_BASE=block_table_base_offsets is not None,
        num_warps=2,
        num_stages=1,
        waves_per_eu=0,
    )
