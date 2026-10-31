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

import torch
from tokenspeed_kernel._triton import tl, triton

_DEEPSEEK_V4_MEGAMOE_FP8_BLOCK_SIZE = 128


def mega_moe_shared_sf_row(token_id: int, block_m: int) -> int:
    """Map tokens into native shared-scale pages for positive ``block_m``.

    Each BLOCK_M page is padded to 128 rows, then its 4-by-32 scale blocks are
    transposed for UTCCP. This also accepts tensor indices for CPU layout checks.
    """
    token_in_page = token_id % block_m
    return (
        token_id // block_m * ((block_m + 127) // 128 * 128)
        + token_in_page // 128 * 128
        + token_in_page % 32 * 4
        + token_in_page % 128 // 32
    )


_mega_moe_shared_sf_row_kernel = triton.jit(mega_moe_shared_sf_row)


@triton.jit
def _stage_mxfp4_mega_moe_inputs_kernel(
    hidden_states,
    x_fp8,
    x_sf,
    topk_ids,
    topk_weights,
    topk_idx_out,
    topk_weights_out,
    shared_l1_acts_sf,
    shared_sf_stride_k,
    hidden_stride_m: tl.constexpr,
    hidden_stride_k: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_stride_k: tl.constexpr,
    x_sf_stride_m: tl.constexpr,
    x_sf_stride_k: tl.constexpr,
    topk_ids_stride_m: tl.constexpr,
    topk_ids_stride_k: tl.constexpr,
    topk_weights_stride_m: tl.constexpr,
    topk_weights_stride_k: tl.constexpr,
    topk_idx_stride_m: tl.constexpr,
    topk_idx_stride_k: tl.constexpr,
    topk_weights_out_stride_m: tl.constexpr,
    topk_weights_out_stride_k: tl.constexpr,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    WRITE_SHARED_SF: tl.constexpr,
    SHARED_BLOCK_M: tl.constexpr,
) -> None:
    token_id = tl.program_id(0).to(tl.int64)
    k_block_id = tl.program_id(1)

    k_offsets = k_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < hidden_size
    hidden = tl.load(
        hidden_states + token_id * hidden_stride_m + k_offsets * hidden_stride_k,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)

    num_groups: tl.constexpr = BLOCK_K // GROUP_K
    hidden_groups = tl.reshape(tl.abs(hidden), [num_groups, GROUP_K])
    amax = tl.max(hidden_groups, axis=1)
    amax = tl.maximum(amax, 1.0e-4)

    scale = amax / 448.0
    scale_bits = scale.to(tl.uint32, bitcast=True)
    scale_exp = ((scale_bits >> 23) & 0xFF) + ((scale_bits & 0x7FFFFF) != 0).to(
        tl.uint32
    )
    scale_exp = tl.minimum(tl.maximum(scale_exp, 1), 254)
    rounded_scale = (scale_exp << 23).to(tl.float32, bitcast=True)

    hidden_groups = tl.reshape(hidden, [num_groups, GROUP_K])
    scaled = hidden_groups * (1.0 / rounded_scale)[:, None]
    scaled = tl.reshape(scaled, [BLOCK_K])
    fp8 = scaled.to(tl.float8e4nv)
    tl.store(
        x_fp8 + token_id * x_stride_m + k_offsets * x_stride_k,
        fp8,
        mask=k_mask,
    )

    scale_offsets = tl.arange(0, num_groups)
    packed_scale = tl.sum(scale_exp << (scale_offsets * 8), axis=0).to(tl.int32)
    tl.store(
        x_sf + token_id * x_sf_stride_m + k_block_id * x_sf_stride_k,
        packed_scale,
    )
    if WRITE_SHARED_SF:
        shared_row = _mega_moe_shared_sf_row_kernel(token_id, SHARED_BLOCK_M)
        # Shared activations alias x_fp8; only their scale layout differs.
        tl.store(
            shared_l1_acts_sf
            + shared_row
            + k_block_id.to(tl.int64) * shared_sf_stride_k,
            packed_scale,
        )

    if k_block_id == 0:
        topk_offsets = tl.arange(0, BLOCK_TOPK)
        topk_mask = topk_offsets < top_k

        ids = tl.load(
            topk_ids + token_id * topk_ids_stride_m + topk_offsets * topk_ids_stride_k,
            mask=topk_mask,
            other=0,
        ).to(tl.int64)
        tl.store(
            topk_idx_out
            + token_id * topk_idx_stride_m
            + topk_offsets * topk_idx_stride_k,
            ids,
            mask=topk_mask,
        )

        weights = tl.load(
            topk_weights
            + token_id * topk_weights_stride_m
            + topk_offsets * topk_weights_stride_k,
            mask=topk_mask,
            other=0.0,
        )
        tl.store(
            topk_weights_out
            + token_id * topk_weights_out_stride_m
            + topk_offsets * topk_weights_out_stride_k,
            weights,
            mask=topk_mask,
        )


def stage_mxfp4_mega_moe_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    *,
    shared_l1_acts_sf: torch.Tensor | None,
    shared_block_m: int,
) -> None:
    """Quantize hidden states and copy routes into native MegaMoE input views.

    ``x_fp8`` and ``x_sf`` receive FP8 activations and packed group-32 UE8M0
    scales; ``topk_idx_out`` and ``topk_weights_out`` receive their input routes.
    With shared fusion, pass the native MN-major ``shared_l1_acts_sf`` view and
    the native tile query's ``shared_block_m`` for this call. The same scale
    bytes are stored in both layouts, with no second activation quantization.
    Without shared fusion, pass ``None`` and zero. No kernel launches for zero
    rows; the caller must still participate in the native collective.
    """
    num_tokens, hidden_size = hidden_states.shape
    if num_tokens == 0:
        return
    if hidden_size % _DEEPSEEK_V4_MEGAMOE_FP8_BLOCK_SIZE != 0:
        raise ValueError(
            "DeepSeek V4 MegaMoE input staging requires hidden_size to be "
            f"a multiple of {_DEEPSEEK_V4_MEGAMOE_FP8_BLOCK_SIZE}."
        )
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "DeepSeek V4 MegaMoE input staging requires topk_weights and "
            "topk_ids to have the same shape."
        )
    if shared_l1_acts_sf is None:
        if shared_block_m != 0:
            raise ValueError("shared_block_m must be zero without shared scale output.")
        shared_sf_stride_k = 0
    else:
        if shared_block_m <= 0:
            raise ValueError("Shared scale output requires a positive shared_block_m.")
        shared_sf_rows = (
            triton.cdiv(num_tokens, shared_block_m)
            * triton.cdiv(shared_block_m, 128)
            * 128
        )
        if (
            shared_l1_acts_sf.dtype != torch.int32
            or shared_l1_acts_sf.ndim != 2
            or shared_l1_acts_sf.shape[0] < shared_sf_rows
            or shared_l1_acts_sf.shape[1] != hidden_size // 128
            or shared_l1_acts_sf.stride(0) != 1
            or shared_l1_acts_sf.stride(1) < shared_l1_acts_sf.shape[0]
            or shared_l1_acts_sf.device != hidden_states.device
        ):
            raise ValueError(
                "Shared scale output must be an int32 MN-major tensor on the "
                "input device with enough padded BLOCK_M pages and hidden_size / 128 columns."
            )
        shared_sf_stride_k = shared_l1_acts_sf.stride(1)

    block_k = _DEEPSEEK_V4_MEGAMOE_FP8_BLOCK_SIZE
    grid = (num_tokens, triton.cdiv(hidden_size, block_k))
    block_topk = triton.next_power_of_2(topk_ids.shape[1])
    _stage_mxfp4_mega_moe_inputs_kernel[grid](
        hidden_states,
        x_fp8,
        x_sf,
        topk_ids,
        topk_weights,
        topk_idx_out,
        topk_weights_out,
        shared_l1_acts_sf,
        shared_sf_stride_k,
        hidden_states.stride(0),
        hidden_states.stride(1),
        x_fp8.stride(0),
        x_fp8.stride(1),
        x_sf.stride(0),
        x_sf.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        topk_idx_out.stride(0),
        topk_idx_out.stride(1),
        topk_weights_out.stride(0),
        topk_weights_out.stride(1),
        hidden_size,
        topk_ids.shape[1],
        BLOCK_K=block_k,
        GROUP_K=32,
        BLOCK_TOPK=block_topk,
        WRITE_SHARED_SF=shared_l1_acts_sf is not None,
        SHARED_BLOCK_M=shared_block_m,
        num_warps=4,
    )
