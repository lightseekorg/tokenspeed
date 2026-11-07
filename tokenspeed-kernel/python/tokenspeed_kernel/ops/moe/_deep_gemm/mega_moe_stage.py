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
from tokenspeed_kernel.platform import pdl_enabled

_DEEPSEEK_V4_MEGAMOE_FP8_BLOCK_SIZE = 128


@triton.jit(do_not_specialize=["num_tokens", "shared_block_m"])
def _stage_mxfp4_mega_moe_inputs_kernel(
    hidden_states,
    x_fp8,
    x_sf,
    shared_x_sf,
    topk_ids,
    topk_weights,
    topk_idx_out,
    topk_weights_out,
    live_requests,
    num_tokens,
    shared_block_m,
    hidden_stride_m: tl.constexpr,
    hidden_stride_k: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_stride_k: tl.constexpr,
    x_sf_stride_m: tl.constexpr,
    x_sf_stride_k: tl.constexpr,
    shared_x_sf_stride_m: tl.constexpr,
    shared_x_sf_stride_k: tl.constexpr,
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
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    HAS_SHARED: tl.constexpr,
    HAS_LIVE: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
) -> None:
    # The outputs live in the MegaMoE symmetric buffer, which the previous
    # MegaMoE kernel may still read, so even the stores wait for the upstream.
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    token_ids = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    token_mask = token_ids < num_tokens
    k_block_id = tl.program_id(1)

    k_offsets = k_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = token_mask[:, None] & (k_offsets[None, :] < hidden_size)
    hidden = tl.load(
        hidden_states
        + token_ids[:, None] * hidden_stride_m
        + k_offsets[None, :] * hidden_stride_k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    num_groups: tl.constexpr = BLOCK_K // GROUP_K
    hidden_groups = tl.reshape(tl.abs(hidden), [BLOCK_M, num_groups, GROUP_K])
    amax = tl.max(hidden_groups, axis=2)
    amax = tl.maximum(amax, 1.0e-4)

    scale = amax / 448.0
    scale_bits = scale.to(tl.uint32, bitcast=True)
    scale_exp = ((scale_bits >> 23) & 0xFF) + ((scale_bits & 0x7FFFFF) != 0).to(
        tl.uint32
    )
    scale_exp = tl.minimum(tl.maximum(scale_exp, 1), 254)
    rounded_scale = (scale_exp << 23).to(tl.float32, bitcast=True)

    hidden_groups = tl.reshape(hidden, [BLOCK_M, num_groups, GROUP_K])
    scaled = hidden_groups * (1.0 / rounded_scale)[:, :, None]
    scaled = tl.reshape(scaled, [BLOCK_M, BLOCK_K])
    tl.store(
        x_fp8 + token_ids[:, None] * x_stride_m + k_offsets[None, :] * x_stride_k,
        scaled.to(tl.float8e4nv),
        mask=mask,
    )

    scale_offsets = tl.arange(0, num_groups)
    packed_scale = tl.sum(scale_exp << (scale_offsets[None, :] * 8), axis=1).to(
        tl.int32
    )
    tl.store(
        x_sf + token_ids * x_sf_stride_m + k_block_id * x_sf_stride_k,
        packed_scale,
        mask=token_mask,
    )
    if HAS_SHARED:
        # The fused shared expert reads the same scales MN-major, rows grouped
        # by the MegaMoE BLOCK_M and transposed within each 128 for UTCCP.
        m_block = token_ids // shared_block_m
        m_in_block = token_ids % shared_block_m
        shared_rows = m_block * ((shared_block_m + 127) // 128 * 128) + (
            m_in_block // 128 * 128 + m_in_block % 32 * 4 + m_in_block % 128 // 32
        )
        tl.store(
            shared_x_sf
            + shared_rows * shared_x_sf_stride_m
            + k_block_id * shared_x_sf_stride_k,
            packed_scale,
            mask=token_mask,
        )

    if k_block_id == 0:
        topk_offsets = tl.arange(0, BLOCK_TOPK)
        topk_mask = token_mask[:, None] & (topk_offsets[None, :] < top_k)

        ids = tl.load(
            topk_ids
            + token_ids[:, None] * topk_ids_stride_m
            + topk_offsets[None, :] * topk_ids_stride_k,
            mask=topk_mask,
            other=0,
        ).to(tl.int64)
        if HAS_LIVE:
            # A padded graph replay stages the rank's whole batch, request-major
            # with the filler requests last: route no row past the live
            # requests' share, as MegaMoE skips negative ids in dispatch and
            # combine. The product of two runtime counts is taken in int64.
            live_rows = (
                num_tokens
                * tl.load(live_requests).to(tl.int64)
                // tl.load(live_requests + 1)
            )
            filler = token_ids >= live_rows
            ids = tl.where(filler[:, None], -1, ids)
        tl.store(
            topk_idx_out
            + token_ids[:, None] * topk_idx_stride_m
            + topk_offsets[None, :] * topk_idx_stride_k,
            ids,
            mask=topk_mask,
        )

        weights = tl.load(
            topk_weights
            + token_ids[:, None] * topk_weights_stride_m
            + topk_offsets[None, :] * topk_weights_stride_k,
            mask=topk_mask,
            other=0.0,
        )
        if HAS_LIVE:
            weights = tl.where(filler[:, None], 0.0, weights)
        tl.store(
            topk_weights_out
            + token_ids[:, None] * topk_weights_out_stride_m
            + topk_offsets[None, :] * topk_weights_out_stride_k,
            weights,
            mask=topk_mask,
        )


def _stage_block_m(num_tokens: int) -> int:
    """Tokens per program: one for small batches, more as the batch grows."""
    return 32 if num_tokens >= 1024 else 8 if num_tokens >= 64 else 1


def stage_mxfp4_mega_moe_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    shared_x_sf: torch.Tensor | None,
    shared_block_m: int | None,
    live_requests: torch.Tensor | None,
) -> None:
    """Write MegaMoE's FP8 inputs and routing into its symmetric buffer.

    ``shared_x_sf`` and ``shared_block_m``, both given or both None, add the
    scale view a fused shared expert reads; ``shared_block_m`` is the BLOCK_M
    MegaMoE runs this token count with. ``live_requests``, an int32 device
    pair (live, padded) read at launch, routes only the first
    ``num_tokens * live // padded`` rows: a CUDA graph replayed at a padded
    batch size stages its filler rows with expert id -1 and weight 0. That
    needs ``hidden_states`` to be the rank's whole padded batch, request-major
    with the same rows per request and the filler requests last, not a TP
    shard of it (see ``mega_moe_live_requests``). Codes and scales are staged
    for every row. None routes every row.
    """
    num_tokens, hidden_size = hidden_states.shape
    if (shared_x_sf is None) != (shared_block_m is None):
        raise ValueError("shared_x_sf and shared_block_m go together")
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
    if live_requests is not None and (
        live_requests.shape != (2,) or live_requests.dtype != torch.int32
    ):
        raise ValueError("live_requests must be an int32 (live, padded) pair")

    block_k = _DEEPSEEK_V4_MEGAMOE_FP8_BLOCK_SIZE
    block_m = _stage_block_m(num_tokens)
    grid = (triton.cdiv(num_tokens, block_m), triton.cdiv(hidden_size, block_k))
    block_topk = triton.next_power_of_2(topk_ids.shape[1])
    enable_pdl = pdl_enabled()
    _stage_mxfp4_mega_moe_inputs_kernel[grid](
        hidden_states,
        x_fp8,
        x_sf,
        shared_x_sf,
        topk_ids,
        topk_weights,
        topk_idx_out,
        topk_weights_out,
        live_requests,
        num_tokens,
        shared_block_m or 1,
        hidden_states.stride(0),
        hidden_states.stride(1),
        x_fp8.stride(0),
        x_fp8.stride(1),
        x_sf.stride(0),
        x_sf.stride(1),
        shared_x_sf.stride(0) if shared_x_sf is not None else 0,
        shared_x_sf.stride(1) if shared_x_sf is not None else 0,
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
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        GROUP_K=32,
        BLOCK_TOPK=block_topk,
        HAS_SHARED=shared_x_sf is not None,
        HAS_LIVE=live_requests is not None,
        ENABLE_PDL=enable_pdl,
        num_warps=4,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
