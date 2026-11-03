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

"""Model-facing fused-MoE entry points and the dispatch policy that picks
warp decode, medium decode, direct/precomputed MFMA decode, package
prefill, or the generic pipelined path by batch size and weight layout.

Distinct from mxfp4/moe.py, which is the staged decode entry taking
precomputed top-k."""

from __future__ import annotations

from typing import Optional

import torch
from tokenspeed_kernel_amd._triton import triton
from tokenspeed_kernel_amd.ops.gfx950.moe._common import (
    FnSpecs,
    FusedActivation,
    RaggedTensorMetadata,
    swiglu_fn,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused._common import (
    _extract_gluon_raw_s,
    _extract_gluon_raw_w,
    _extract_gluon_raw_w_unshuffled,
    _make_dummy,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused._layouts import (
    _moe_partial_reduce_shared,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused.gemm_api import (
    gluon_mxfp_ragged_matmul,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused.quantize import (
    _dynamic_fp8_quantize,
    _quantize_mxfp4_activation,
    fp8_quantize,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused.routing import (
    _ROUTING_METHOD_RENORMALIZE,
    SMALLM_MAX_M,
    _biased_grouped_topk_reference,
    _grouped_topk_reference,
    _has_incomplete_grouped_routing,
    _normalize_route_weights,
    _route_from_topk,
    _softmax_topk_reference,
    _stable_topk_smaller_index,
    _uses_grouped_routing,
    default_biased_grouped_route,
    default_biased_route,
    default_grouped_route,
    default_packed_topk_route,
    default_route,
    default_scaled_route,
    gluon_biased_grouped_fused_route,
    gluon_biased_grouped_route_supported,
    gluon_fused_route,
    gluon_precomputed_topk_flat_m1_route,
    gluon_precomputed_topk_fused_route,
    gluon_precomputed_topk_route_supported,
    gluon_route_supported,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused.tuning import (
    _select_grouped_moe_config,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused.warp_decode import (
    _gluon_mxfp4_fp8_warp_decode_moe,
    _warp_decode_precomputed_situ_sorted_stage1_kernel,
    _warp_decode_precomputed_situ_stage1_kernel,
    _warp_decode_sorted_stage2_fp8_mxfp4_kernel,
    _warp_decode_stage2_fp8_mxfp4_kernel,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.prefill_stage2 import (
    gluon_mxfp4_moe_stage2_reduce_kernel,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.scale_layout import (
    MXFP4_BLOCK,
    empty_swizzled_cdna4_mxfp4_scale,
    is_swizzled_cdna4_mxfp4_scale,
)

_DEFAULT_SWIGLU_ALPHA = 1.702


_DEFAULT_SWIGLU_LIMIT = 7.0


_DEFAULT_SWIGLU_BETA = 1.0


_DEFAULT_SWIGLU_ACT = FusedActivation(
    FnSpecs("swiglu", swiglu_fn, ("alpha", "limit", "beta"), reduction_n=2),
    (_DEFAULT_SWIGLU_ALPHA, _DEFAULT_SWIGLU_LIMIT, _DEFAULT_SWIGLU_BETA),
)


def _swiglu_activation(alpha: float, limit: float, beta: float) -> FusedActivation:
    if (
        float(alpha) == _DEFAULT_SWIGLU_ALPHA
        and float(limit) == _DEFAULT_SWIGLU_LIMIT
        and float(beta) == _DEFAULT_SWIGLU_BETA
    ):
        return _DEFAULT_SWIGLU_ACT
    return FusedActivation(
        FnSpecs("swiglu", swiglu_fn, ("alpha", "limit", "beta"), reduction_n=2),
        (float(alpha), float(limit), float(beta)),
    )


def _mxfp4_input_metadata(
    hidden_states: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> tuple[int, int, torch.device, torch.dtype]:
    if isinstance(hidden_states, tuple):
        packed, _ = hidden_states
        if packed.ndim != 2 or packed.dtype != torch.uint8:
            raise ValueError("packed MXFP4 input must be rank-2 uint8")
        return packed.shape[0], packed.shape[1] * 2, packed.device, torch.bfloat16
    return (*hidden_states.shape, hidden_states.device, hidden_states.dtype)


def _validate_deferred_out(
    output: torch.Tensor,
    *,
    tokens: int,
    topk: int,
    width: int,
    device: torch.device,
    inputs: tuple[torch.Tensor | None, ...],
) -> None:
    """Check a caller-owned ``[tokens * topk, width]`` BF16 route destination."""
    if (
        output.shape != (tokens * topk, width)
        or output.dtype != torch.bfloat16
        or output.device != device
        or not output.is_contiguous()
    ):
        raise ValueError("deferred output must be contiguous colocated BF16 routes")
    if not output.numel():
        return
    start = output.data_ptr()
    end = start + output.numel() * output.element_size()
    for tensor in inputs:
        if tensor is None or not tensor.numel():
            continue
        # Cheap common case: the whole input allocation is disjoint. Eager
        # small-M prefill is host bound, so avoid the span walk below.
        storage = tensor.untyped_storage()
        base = storage.data_ptr()
        if base + storage.nbytes() <= start or end <= base:
            continue
        if tensor.device != device:
            continue
        # Disjoint views of one symmetric heap are valid. Include the full
        # addressed span of strided inputs, including gaps between rows.
        offsets = [
            (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
        ]
        first = (
            tensor.data_ptr()
            + sum(min(0, offset) for offset in offsets) * tensor.element_size()
        )
        last = (
            tensor.data_ptr()
            + (1 + sum(max(0, offset) for offset in offsets)) * tensor.element_size()
        )
        if first < end and start < last:
            raise ValueError("deferred output must not overlap an input")


def _resolve_mxfp4_activation(
    hidden_states: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    x_quantized: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return token-ordered E2M1 data and CDNA4 E8M0 scales."""
    if isinstance(hidden_states, tuple):
        if x_quantized is not None:
            raise ValueError("MXFP4 input cannot supply two quantized pairs")
        x_quantized = hidden_states
    elif x_quantized is None:
        return _quantize_mxfp4_activation(hidden_states)

    rows, width, device, _ = _mxfp4_input_metadata(hidden_states)
    packed, scales = x_quantized
    if packed.shape != (rows, width // 2):
        raise ValueError("packed MXFP4 input does not match hidden states")
    if packed.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError("MXFP4 input and scales must use uint8 storage")
    if (
        not packed.is_contiguous()
        or packed.device != device
        or scales.device != device
        or scales.ndim != 2
        or not is_swizzled_cdna4_mxfp4_scale(scales)
        or scales.shape[0] < triton.cdiv(width // MXFP4_BLOCK, 8) * 8 * 32
        or scales.shape[1] < triton.cdiv(rows, 32)
    ):
        raise ValueError(
            "MXFP4 input must use contiguous data and complete CDNA4-swizzled "
            "scales on the hidden-state device"
        )
    return packed, scales


# Tuned decode dispatch defaults (rocprofv3 real-GPU tuned). Decode owns the
# small-M regime below the package-prefill gate; the kernel is chosen purely by
# batch size and whether the caller supplied precomputed top-k:
#   M <= _DIRECT_DECODE_MAX_M                       -> direct top-k MXFP4 MFMA decode
#   _PRECOMPUTED_MFMA_MIN_M <= M <= _DECODE_MAX_M   -> precomputed-activation MFMA decode
#   (no precomputed top-k) M <= _DECODE_MAX_M       -> route-owned direct decode
# These replace the former GLUON_MXFP4_* environment overrides.
_DECODE_MAX_M = 8


_DIRECT_DECODE_MAX_M = 2


_DEFERRED_DIRECT_DECODE_MAX_M = 16


_PRECOMPUTED_MFMA_MIN_M = 4


# Widest activation the precomputed-SiTU entry point serves with the
# warp-decode kernels; anything wider goes to package prefill.
#
# Package prefill tiles routed rows into BLOCK_M=128 groups, which pays off once
# a batch fills those tiles. Speculative decode never does: with EAGLE3 the
# decode width is concurrency x num_draft_tokens, so a Kimi-K3 batch of 8 is 32
# rows spread over hundreds of experts and the tiles are almost entirely
# padding. Measured end to end on 8x gfx950 at 50K/500, moving this bound from
# 16 to 64 lowers TPOT by 17% at concurrency 8 and 4% at concurrency 16, while
# concurrency 4 -- below the bound either way -- is unchanged, and a
# concurrency that does not cross the bound does not move.
_SITU_WARP_DECODE_MAX_M = 64


_ROUTE_OWNED_MIN_M = 1


_DIRECT_STAGE2_BLOCK_N = 16
_DIRECT_STAGE2_BLOCK_K = 128


# Key: hidden width, intermediate width, global experts, top-k. Finalized
# decode up to this many rows takes the direct path with the slot-parallel
# stage 2, whose short K keeps the per-token top-k chain latency bound.
_SLOT_PARALLEL_DIRECT_DECODE_MAX_M = {(5120, 576, 384, 6): 32}


def _select_direct_stage2_tiles(
    num_tokens: int,
    *,
    ep_deferred: bool,
) -> tuple[int, int]:
    """Return the rocprof-tuned gfx950 ``(BLOCK_N, BLOCK_K)`` pair."""
    if not ep_deferred:
        return _DIRECT_STAGE2_BLOCK_N, _DIRECT_STAGE2_BLOCK_K
    # Wide output tiles amortize deferred expert stores at the largest captured
    # batch; deeper K tiles shorten the four-tile dependency chain at M <= 8.
    block_n = 32 if num_tokens >= 16 else _DIRECT_STAGE2_BLOCK_N
    block_k = 256 if num_tokens <= 8 else _DIRECT_STAGE2_BLOCK_K
    return block_n, block_k


_SITU_INTERMEDIATE_SCALES: dict[tuple[torch.device, float], torch.Tensor] = {}
_A8W4_STAGE1_NUM_BUFFERS = 2
_A8W4_STAGE2_NUM_WARPS = 1
_A8W4_SORTED_MIN_CONCURRENCY = 8
_A8W4_SORTED_MIN_DRAFT_TOKENS = 4
_A8W4_SORTED_STAGE1_MIN_M = _A8W4_SORTED_MIN_CONCURRENCY * _A8W4_SORTED_MIN_DRAFT_TOKENS
_A8W4_SORTED_XCD_SWIZZLE = 1
_A8W4_SORTED_STAGE2_N_TILES = 4


def _select_a8w4_stage2_block_n(num_tokens: int) -> int:
    """Keep the combined-top-k stage-2 grid near a fixed machine-wave count."""
    if num_tokens <= 8:
        return 16
    if num_tokens <= 16:
        return 32
    if num_tokens <= 32:
        return 64
    return 128


def _use_a8w4_combined_topk(num_tokens: int, fuse_shared_down: bool) -> bool:
    """Select widths where in-CTA top-k reduction beats split reduction."""
    return not fuse_shared_down and 8 <= num_tokens <= 16


def _situ_intermediate_scale(device: torch.device, max_abs: float) -> torch.Tensor:
    key = (device, max_abs)
    scale = _SITU_INTERMEDIATE_SCALES.get(key)
    if scale is None:
        scale = torch.tensor([max_abs / 448.0], dtype=torch.float32, device=device)
        _SITU_INTERMEDIATE_SCALES[key] = scale
    return scale


def gluon_mxfp4_fp8_precomputed_situ(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_weight,
    w2_weight,
    *,
    w13_mx_scale: torch.Tensor,
    w2_mx_scale: torch.Tensor,
    situ_beta: float,
    situ_linear_beta: float,
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
    shared_input: torch.Tensor | None = None,
    shared_weight: torch.Tensor | None = None,
    shared_out: torch.Tensor | None = None,
    expert_start: int = 0,
    global_num_experts: int | None = None,
    prefill_activation_format: str = "e2m1",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    """Run SiTU decode (route-direct, combined top-k, or expert-sorted tiles)
    or block-ragged SiTU prefill.

    ``expert_start`` and ``global_num_experts`` describe a contiguous local EP
    shard. Global top-k IDs outside that shard contribute zero to this rank.
    """
    if (
        hidden_states.ndim != 2
        or hidden_states.shape[0] <= 0
        or hidden_states.dtype != torch.bfloat16
        or out_dtype != torch.bfloat16
        or topk_ids.ndim != 2
        or topk_weights.shape != topk_ids.shape
        or situ_beta <= 0.0
        or situ_linear_beta <= 0.0
    ):
        return None

    M, D = hidden_states.shape
    TOPK = int(topk_ids.shape[1])
    fuse_shared_down = shared_input is not None or shared_weight is not None
    if fuse_shared_down:
        if shared_input is None or shared_weight is None:
            raise ValueError("shared input and weight must be provided together")
        if (
            tuple(shared_input.shape) != (M, 768)
            or tuple(shared_weight.shape) != (7168, 768)
            or shared_input.dtype != torch.bfloat16
            or shared_weight.dtype != torch.bfloat16
            or not shared_input.is_cuda
            or not shared_weight.is_cuda
            or not shared_input.is_contiguous()
            or not shared_weight.is_contiguous()
            or shared_input.device != hidden_states.device
            or shared_weight.device != hidden_states.device
        ):
            raise ValueError(
                "shared down fusion requires contiguous colocated BF16 "
                "[M, 768] input and [7168, 768] weight"
            )
    if expert_start < 0:
        raise ValueError("expert_start must be non-negative")
    if M > _SITU_WARP_DECODE_MAX_M:
        if fuse_shared_down:
            return None
        result = _maybe_gluon_package_mxfp4_prefill(
            hidden_states,
            hidden_states.new_empty((M, 0)),
            w13_weight,
            w2_weight,
            w13_mx_scale=w13_mx_scale,
            w2_mx_scale=w2_mx_scale,
            top_k=TOPK,
            correction_bias=None,
            n_group=0,
            topk_group=0,
            routed_scaling_factor=1.0,
            normalize_topk_weights=True,
            routing_method_type=0,
            precomputed_topk_weights=topk_weights,
            precomputed_topk_ids=topk_ids,
            out_dtype=out_dtype,
            swiglu_alpha=float(situ_beta),
            swiglu_limit=0.0,
            swiglu_beta=0.0,
            do_finalize=True,
            x_quantized=None,
            situ_linear_beta=float(situ_linear_beta),
            expert_start=expert_start,
            global_num_experts=global_num_experts,
            out=out,
            activation_format=prefill_activation_format,
        )
        return result

    w13_raw = _extract_gluon_raw_w(w13_weight)
    w2_raw = _extract_gluon_raw_w(w2_weight)
    w13_scale = _extract_gluon_raw_s(w13_mx_scale)
    w2_scale = _extract_gluon_raw_s(w2_mx_scale)
    if not all(
        isinstance(t, torch.Tensor) for t in (w13_raw, w2_raw, w13_scale, w2_scale)
    ):
        return None
    if not (
        w13_raw.ndim == 3
        and w2_raw.ndim == 3
        and w13_raw.dtype == torch.uint8
        and w2_raw.dtype == torch.uint8
        and bool(getattr(w13_raw, "is_shuffled_for_gluon_dot", False))
        and bool(getattr(w2_raw, "is_shuffled_for_gluon_dot", False))
    ):
        return None

    num_local_experts = int(w13_raw.shape[0])
    if global_num_experts is None:
        global_num_experts = num_local_experts
    if (
        global_num_experts < num_local_experts
        or expert_start + num_local_experts > global_num_experts
    ):
        raise ValueError("local expert range exceeds global expert count")

    two_i = int(w13_raw.shape[2])
    if two_i % 2 or int(getattr(w13_raw, "original_k_pk", 0)) * 2 != D:
        return None
    i_dim = two_i // 2
    N = int(getattr(w2_raw, "original_n", int(w2_raw.shape[2])))
    if int(getattr(w2_raw, "original_k_pk", 0)) * 2 != i_dim or N != D:
        return None

    # Tiny-M kernels index routing tensors as flat [M * TOPK] arrays. Preserve
    # that contract when the caller supplies an otherwise valid strided view.
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    x_fp8, x_scale = _dynamic_fp8_quantize(hidden_states)
    inter_scale = _situ_intermediate_scale(
        hidden_states.device, float(situ_beta * situ_linear_beta)
    )
    use_sorted_route_tiles = (
        M >= _A8W4_SORTED_STAGE1_MIN_M
        and i_dim == 384
        # Sorted W2 tiles load whole BLOCK_N column tiles without a column mask.
        and N % (128 * _A8W4_SORTED_STAGE2_N_TILES) == 0
        and not fuse_shared_down
        and expert_start == 0
        and global_num_experts == num_local_experts
    )
    if not use_sorted_route_tiles:
        inter = torch.empty(
            (M * TOPK, i_dim),
            dtype=torch.float8_e4m3fn,
            device=hidden_states.device,
        )
    if out is None:
        out = torch.empty((M, N), dtype=out_dtype, device=hidden_states.device)
    elif (
        tuple(out.shape) != (M, N)
        or out.dtype != out_dtype
        or out.device != hidden_states.device
        or out.stride(1) != 1
        or out.stride(0) < N
    ):
        raise ValueError("SiTU output must be a row-major colocated [M, N] tensor")
    if fuse_shared_down:
        if shared_out is None:
            shared_out = torch.empty(
                (M, 7168), dtype=torch.bfloat16, device=hidden_states.device
            )
        elif (
            tuple(shared_out.shape) != (M, 7168)
            or shared_out.dtype != torch.bfloat16
            or shared_out.device != hidden_states.device
            or not shared_out.is_contiguous()
        ):
            raise ValueError(
                "shared output must be contiguous colocated BF16 [M, 7168]"
            )
    elif shared_out is not None:
        raise ValueError("shared output requires shared input and weight")
    dummy_bias = _make_dummy(hidden_states.device, torch.float32, 1)

    block_n = 128
    block_k = 256
    num_warps = 4
    k_iters = (D + block_k - 1) // block_k
    even_k = D % block_k == 0
    num_buffers = min(_A8W4_STAGE1_NUM_BUFFERS, k_iters + (1 if even_k else 0))
    if use_sorted_route_tiles:
        from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.moe_sorting import (
            gluon_moe_sorting,
        )

        (
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            _,
        ) = gluon_moe_sorting(
            topk_ids,
            topk_weights,
            num_local_experts,
            N,
            out_dtype,
            16,
            use_minimum_route_chunks=True,
            expert_start=expert_start,
            out=out,
            runtime_rows=False,
        )
        inter = torch.empty(
            (int(sorted_ids.shape[0]), i_dim),
            dtype=torch.float8_e4m3fn,
            device=hidden_states.device,
        )
        grid = (int(sorted_expert_ids.shape[0]) * triton.cdiv(two_i, block_n),)
        _warp_decode_precomputed_situ_sorted_stage1_kernel[grid](
            x_fp8.view(torch.uint8),
            w13_raw,
            w13_scale,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            inter,
            M,
            D,
            i_dim,
            x_fp8.stride(0),
            x_fp8.stride(1),
            w13_raw.stride(0),
            w13_raw.stride(-2),
            w13_raw.stride(-1),
            w13_scale.stride(0),
            w13_scale.stride(-2),
            w13_scale.stride(-1),
            inter.stride(0),
            inter.stride(1),
            x_scale,
            inter_scale,
            dummy_bias,
            TOPK=TOPK,
            BLOCK_K=block_k,
            BLOCK_N=block_n,
            BLOCK_M=16,
            NUM_BUFFERS=num_buffers,
            NUM_WARPS=num_warps,
            W_PRESHUFFLED=True,
            EVEN_K=even_k,
            HAS_BIAS=False,
            SITU_BETA=float(situ_beta),
            SITU_LINEAR_BETA=float(situ_linear_beta),
            EXPERT_START=expert_start,
            NUM_LOCAL_EXPERTS=num_local_experts,
            XCD_SWIZZLE=_A8W4_SORTED_XCD_SWIZZLE,
            num_warps=num_warps,
        )
    else:
        grid = (M * triton.cdiv(two_i, block_n) * TOPK,)
        _warp_decode_precomputed_situ_stage1_kernel[grid](
            x_fp8.view(torch.uint8),
            w13_raw,
            w13_scale,
            topk_ids,
            inter,
            M,
            D,
            i_dim,
            x_fp8.stride(0),
            x_fp8.stride(1),
            topk_ids.stride(0),
            topk_ids.stride(1),
            w13_raw.stride(0),
            w13_raw.stride(-2),
            w13_raw.stride(-1),
            w13_scale.stride(0),
            w13_scale.stride(-2),
            w13_scale.stride(-1),
            inter.stride(0),
            inter.stride(1),
            x_scale,
            inter_scale,
            dummy_bias,
            TOPK=TOPK,
            BLOCK_K=block_k,
            BLOCK_N=block_n,
            BLOCK_M=16,
            NUM_BUFFERS=num_buffers,
            NUM_WARPS=num_warps,
            W_PRESHUFFLED=True,
            EVEN_K=even_k,
            HAS_BIAS=False,
            SITU_BETA=float(situ_beta),
            SITU_LINEAR_BETA=float(situ_linear_beta),
            EXPERT_START=expert_start,
            NUM_LOCAL_EXPERTS=num_local_experts,
            num_warps=num_warps,
        )

    if use_sorted_route_tiles:
        partial = torch.empty(
            (M * TOPK, N), dtype=out_dtype, device=hidden_states.device
        )
        sorted_stage2_num_warps = _A8W4_SORTED_STAGE2_N_TILES
        sorted_stage2_block_n = 128 * sorted_stage2_num_warps
        sorted_stage2_programs = int(sorted_expert_ids.shape[0]) * triton.cdiv(
            N, sorted_stage2_block_n
        )
        _warp_decode_sorted_stage2_fp8_mxfp4_kernel[(sorted_stage2_programs,)](
            inter,
            w2_raw,
            w2_scale,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            partial,
            M,
            N,
            int(w2_raw.shape[2]),
            i_dim,
            inter.stride(0),
            inter.stride(1),
            w2_raw.stride(0),
            w2_raw.stride(-2),
            w2_raw.stride(-1),
            w2_scale.stride(0),
            w2_scale.stride(-2),
            w2_scale.stride(-1),
            partial.stride(0),
            partial.stride(1),
            inter_scale,
            I_PACKED=i_dim // 2,
            TOPK=TOPK,
            BLOCK_M=16,
            BLOCK_K=128,
            BLOCK_N=sorted_stage2_block_n,
            NUM_WARPS_N=sorted_stage2_num_warps,
            W_PRESHUFFLED=True,
            EXPERT_START=expert_start,
            NUM_LOCAL_EXPERTS=num_local_experts,
            XCD_SWIZZLE=_A8W4_SORTED_XCD_SWIZZLE,
            num_warps=sorted_stage2_num_warps,
        )
        reduce_block_n = 256
        reduce_programs = M * triton.cdiv(N, reduce_block_n)
        gluon_mxfp4_moe_stage2_reduce_kernel[(reduce_programs,)](
            partial,
            out,
            M,
            N,
            TOPK * partial.stride(0),
            partial.stride(0),
            partial.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_M=1,
            BLOCK_N=reduce_block_n,
            TOP_K=TOPK,
            # The sort drops routes that name no local expert, leaving their
            # partial rows unwritten.
            MASK_INVALID_ROUTES=True,
            route_ids_ptr=topk_ids,
            stride_rt=topk_ids.stride(0),
            stride_rs=topk_ids.stride(1),
            expert_start=expert_start,
            num_experts=num_local_experts,
            num_warps=1,
        )
        return out

    # Accumulate all routed experts in each output CTA when the reduction does
    # not also own the shared-expert projection. Scaling BLOCK_N with M keeps
    # enough one-wave CTAs resident while avoiding the partial tensor and a
    # separate top-k reduction.
    combine_topk = _use_a8w4_combined_topk(M, fuse_shared_down)
    if combine_topk:
        stage2_out = out
        stage2_stride_om = out.stride(0)
    else:
        stage2_out = torch.empty(
            (M * TOPK, N), dtype=out_dtype, device=hidden_states.device
        )
        stage2_stride_om = stage2_out.stride(0)
    stage2_block_n = _select_a8w4_stage2_block_n(M) if combine_topk else 64
    stage2_num_warps = _A8W4_STAGE2_NUM_WARPS
    routed_stage2_programs = M * triton.cdiv(N, stage2_block_n)
    if not combine_topk:
        routed_stage2_programs *= TOPK
    shared_block_n = 4
    num_shared_pid_n = triton.cdiv(7168, shared_block_n)
    _warp_decode_stage2_fp8_mxfp4_kernel[(routed_stage2_programs,)](
        inter,
        w2_raw,
        w2_scale,
        topk_ids,
        topk_weights,
        stage2_out,
        M,
        N,
        int(w2_raw.shape[2]),
        i_dim,
        inter.stride(0),
        inter.stride(1),
        w2_raw.stride(0),
        w2_raw.stride(-2),
        w2_raw.stride(-1),
        w2_scale.stride(0),
        w2_scale.stride(-2),
        w2_scale.stride(-1),
        stage2_stride_om,
        stage2_out.stride(1),
        0,
        inter_scale,
        dummy_bias,
        I_PACKED=i_dim // 2,
        TOPK=TOPK,
        BLOCK_K=128,
        BLOCK_N=stage2_block_n,
        M_DUP=1,
        W_PRESHUFFLED=True,
        HAS_BIAS=False,
        SPLIT_K=1,
        ROUND_TOPK_PARTIALS=combine_topk,
        SPLIT_TOPK=not combine_topk,
        EXPERT_START=expert_start,
        NUM_LOCAL_EXPERTS=num_local_experts,
        num_warps=stage2_num_warps,
    )
    if combine_topk:
        return out

    partial = stage2_out
    reduce_block_n = 256
    reduce_programs = M * triton.cdiv(N, reduce_block_n)
    reduce_grid = reduce_programs + (M * num_shared_pid_n if fuse_shared_down else 0)
    reduce = (
        _moe_partial_reduce_shared
        if fuse_shared_down
        else gluon_mxfp4_moe_stage2_reduce_kernel
    )
    reduce[(reduce_grid,)](
        partial,
        out,
        *((shared_input, shared_weight, shared_out) if fuse_shared_down else ()),
        M,
        N,
        *(
            (partial.stride(0), TOPK * partial.stride(0))
            if fuse_shared_down
            else (TOPK * partial.stride(0), partial.stride(0))
        ),
        partial.stride(1),
        out.stride(0),
        out.stride(1),
        *(
            (
                shared_input.stride(0),
                shared_input.stride(1),
                shared_out.stride(0),
                shared_out.stride(1),
            )
            if fuse_shared_down
            else ()
        ),
        BLOCK_N=reduce_block_n,
        **(
            {
                "SPLIT_K": TOPK,
                "NUM_REDUCE_PROGRAMS": reduce_programs,
                "NUM_SHARED_PID_N": num_shared_pid_n,
                "SHARED_BLOCK_N": shared_block_n,
            }
            if fuse_shared_down
            else {"BLOCK_M": 1, "TOP_K": TOPK, "MASK_INVALID_ROUTES": False}
        ),
        num_warps=1,
    )
    if fuse_shared_down:
        return out, shared_out
    return out


def gluon_mxfp_fused_moe(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    *,
    w13_mx_scale: torch.Tensor,
    w2_mx_scale: torch.Tensor,
    w13_act_scale: torch.Tensor,
    w2_act_scale: torch.Tensor,
    top_k: int,
    w13_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    enable_warp_decode: bool = True,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    swiglu_beta: float = 1.0,
) -> torch.Tensor:
    """Route + dispatch GEMM + SwiGLU + combine GEMM, all fused for the
    gluon mxfp4 / fp8-activation path.

    Inputs:
        hidden_states: ``(n_tokens, hidden)`` activation in bf16/fp16.
        router_logits: ``(n_tokens, num_experts)`` raw router logits.
        w13_weight, w2_weight: gluon-swizzled MXFP4 expert weights
            (``RaggedTensorMetadata``-compatible wrapped tensors).
        w13_bias, w2_bias: optional float32 expert biases.
        w13_mx_scale, w2_mx_scale: gluon-swizzled MXFP4 expert weight
            scales for the two GEMMs.
        w13_act_scale, w2_act_scale: per-tensor FP8 activation scales
            for the two GEMMs.
        out_dtype: output dtype for the final combine output.
        top_k: routing top_k.
        swiglu_alpha / swiglu_limit: SwiGLU activation parameters.

        enable_warp_decode: Whether to try the gfx950 small-M warp-decode path.
    """
    x_fp8 = fp8_quantize(hidden_states, w13_act_scale)

    n_tokens = router_logits.shape[0]
    use_medium_decode = int(n_tokens) in (8, 16)

    # Warp-decode small-M MoE is only the fastest path for M<=4. M=8/16
    # intentionally falls through to the medium-decode direct path below.
    # It self-guards (returns None) for any shape it does not cover; the
    # tokenspeed-kernel registration wrapper owns the environment/platform gate.
    if enable_warp_decode:
        out = _gluon_mxfp4_fp8_warp_decode_moe(
            x_fp8,
            router_logits,
            w13_weight,
            w2_weight,
            w13_bias=w13_bias,
            w2_bias=w2_bias,
            w13_mx_scale=w13_mx_scale,
            w2_mx_scale=w2_mx_scale,
            w13_act_scale=w13_act_scale,
            w2_act_scale=w2_act_scale,
            out_dtype=out_dtype,
            top_k=top_k,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            swiglu_beta=swiglu_beta,
        )
        if out is not None:
            return out

    # Decode-small GPT-OSS routing is launch-overhead dominated. Prefer the
    # single-kernel Gluon route when both M<=16 and G=M*top_k stays within
    # the rank-tile bound; fall back for larger/unsupported route shapes.
    if n_tokens <= SMALLM_MAX_M and gluon_route_supported(
        router_logits, top_k, router_logits.dtype
    ):
        ragged_metadata, gather_indx, scatter_indx, gate_scal = gluon_fused_route(
            router_logits,
            top_k,
            dtype=router_logits.dtype,
        )
    else:
        ragged_metadata, gather_indx, scatter_indx, gate_scal = default_route(
            router_logits,
            top_k,
            dtype=router_logits.dtype,
        )

    act = _swiglu_activation(swiglu_alpha, swiglu_limit, swiglu_beta)

    gemm1_input = x_fp8

    intermediate_cache = gluon_mxfp_ragged_matmul(
        gemm1_input,
        w13_weight,
        w13_bias,
        w_mx_scale=w13_mx_scale,
        x_global_scale=w13_act_scale,
        out_dtype=out_dtype,
        a_ragged_metadata=ragged_metadata,
        gather_indx=gather_indx,
        fused_activation=act,
        out_quant_scale=w2_act_scale,
        prefer_unshuffled_w=use_medium_decode,
    )

    gemm2_input = intermediate_cache

    return gluon_mxfp_ragged_matmul(
        gemm2_input,
        w2_weight,
        w2_bias,
        w_mx_scale=w2_mx_scale,
        x_global_scale=w2_act_scale,
        out_dtype=out_dtype,
        a_ragged_metadata=ragged_metadata,
        scatter_indx=scatter_indx,
        gammas=gate_scal,
        n_tokens=n_tokens,
        n_expts_act=top_k,
        prefer_unshuffled_w=use_medium_decode,
    )


def _maybe_precomputed_mxfp4_direct_mfma_decode(
    hidden_states: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    *,
    w13_mx_scale: torch.Tensor,
    w2_mx_scale: torch.Tensor,
    top_k: int,
    w13_bias: Optional[torch.Tensor],
    w2_bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    max_m: int,
    route_tile_max_m: int,
    precomputed_topk_weights: torch.Tensor | None,
    precomputed_topk_ids: torch.Tensor | None,
    swiglu_alpha: float,
    swiglu_limit: float,
    swiglu_beta: float,
    expert_start: int,
    global_num_experts: int | None,
    do_finalize: bool,
    x_quantized: tuple[torch.Tensor, torch.Tensor] | None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, None, torch.Tensor] | None:
    """Select direct or route-tiled MFMA with token/slot intermediates.

    ``max_m`` bounds direct execution; ``route_tile_max_m`` bounds the measured
    EP4/EP8 route-tiled schedule. Both reuse gdot128-shuffled weights/scales
    and preserve the per-route gate/up, SwiGLU, quantization and W2 order.
    """
    n_tokens, hidden_dim, device, input_dtype = _mxfp4_input_metadata(hidden_states)
    direct_max_m = _DIRECT_DECODE_MAX_M if do_finalize else max_m
    probe_max_m = max(min(max_m, direct_max_m), route_tile_max_m)
    if do_finalize:
        probe_max_m = max(probe_max_m, max(_SLOT_PARALLEL_DIRECT_DECODE_MAX_M.values()))
    if (
        precomputed_topk_weights is None
        or precomputed_topk_ids is None
        or n_tokens <= 0
        or n_tokens > probe_max_m
        or input_dtype != torch.bfloat16
        or out_dtype != torch.bfloat16
        or w13_bias is not None
        or w2_bias is not None
        or precomputed_topk_ids.ndim != 2
        or precomputed_topk_weights.shape != precomputed_topk_ids.shape
        or int(precomputed_topk_ids.shape[0]) != n_tokens
        or int(precomputed_topk_ids.shape[1]) != top_k
        or (
            out is not None
            and (
                out.shape
                != (
                    (n_tokens, hidden_dim)
                    if do_finalize
                    else (n_tokens * top_k, hidden_dim)
                )
                or out.dtype != out_dtype
                or out.device != device
                or not out.is_contiguous()
            )
        )
    ):
        return None

    w13_runtime = _extract_gluon_raw_w(w13_weight)
    w2_runtime = _extract_gluon_raw_w(w2_weight)
    w13_scale = _extract_gluon_raw_s(w13_mx_scale)
    w2_scale = _extract_gluon_raw_s(w2_mx_scale)
    if (
        not isinstance(w13_runtime, torch.Tensor)
        or not isinstance(w2_runtime, torch.Tensor)
        or not isinstance(w13_scale, torch.Tensor)
        or not isinstance(w2_scale, torch.Tensor)
        or w13_runtime.dtype != torch.uint8
        or w2_runtime.dtype != torch.uint8
        or w13_scale.dtype != torch.uint8
        or w2_scale.dtype != torch.uint8
        or w13_runtime.ndim != 3
        or w2_runtime.ndim != 3
    ):
        return None
    if (
        not bool(getattr(w13_runtime, "is_shuffled_for_gluon_dot", False))
        or not bool(getattr(w2_runtime, "is_shuffled_for_gluon_dot", False))
        or int(getattr(w13_runtime, "gluon_dot_block_k_pk", 0)) != 128
        or int(getattr(w13_runtime, "gluon_dot_block_n", 0)) != 128
        or int(getattr(w2_runtime, "gluon_dot_block_k_pk", 0)) != 128
        or int(getattr(w2_runtime, "gluon_dot_block_n", 0)) != 128
        or w13_scale.stride(-2) != 1
        or w2_scale.stride(-2) != 1
    ):
        return None

    if hidden_dim % MXFP4_BLOCK != 0:
        return None
    w13_k_pk = int(getattr(w13_runtime, "original_k_pk", int(w13_runtime.shape[1])))
    if w13_k_pk * 2 != hidden_dim:
        return None
    inter_dim = int(w13_runtime.shape[2]) // 2
    if int(w13_runtime.shape[2]) != 2 * inter_dim or inter_dim % MXFP4_BLOCK != 0:
        return None
    w2_k_pk = int(getattr(w2_runtime, "original_k_pk", int(w2_runtime.shape[1])))
    out_dim = int(getattr(w2_runtime, "original_n", int(w2_runtime.shape[2])))
    if w2_k_pk * 2 != inter_dim or out_dim != hidden_dim:
        return None

    local_experts = int(w13_runtime.shape[0])
    is_ep_deferred = not do_finalize and (
        global_num_experts == 4 * local_experts
        or (
            global_num_experts == 256
            and local_experts == 32
            and hidden_dim == inter_dim == out_dim == 4096
            and top_k == 8
        )
    )
    use_route_tiles = (
        is_ep_deferred
        and int(w13_runtime.shape[0]) in (32, 64)
        and global_num_experts == 256
        and hidden_dim == inter_dim == out_dim == 4096
        and int(w2_runtime.shape[2]) == 4096
        and top_k == 8
        and 4 <= n_tokens <= min(route_tile_max_m, 48)
        and swiglu_alpha == 1.0
        and swiglu_limit == 0.0
        and swiglu_beta == 0.0
        and precomputed_topk_ids.is_contiguous()
    )
    slot_parallel = do_finalize and n_tokens <= _SLOT_PARALLEL_DIRECT_DECODE_MAX_M.get(
        (hidden_dim, inter_dim, global_num_experts, top_k), 0
    )
    if (
        n_tokens > min(max_m, direct_max_m)
        and not use_route_tiles
        and not slot_parallel
    ):
        return None

    topk_ids = (
        precomputed_topk_ids
        if precomputed_topk_ids.dtype == torch.int32
        else precomputed_topk_ids.to(torch.int32)
    )
    topk_weights = (
        precomputed_topk_weights
        if precomputed_topk_weights.dtype == torch.float32
        else precomputed_topk_weights.to(torch.float32)
    )

    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.decode_stage1 import (
        gluon_mxfp4_moe_stage1_route_tiles_gfx950,
        invoke_stage1_mxfp4_mfma_decode_gluon,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.decode_stage2 import (
        gluon_mxfp4_moe_stage2_route_tiles_gfx950,
        invoke_stage2_mxfp4_mfma_decode_gluon,
    )

    q_hidden, q_hidden_scale = _resolve_mxfp4_activation(
        hidden_states,
        x_quantized,
    )
    q_inter = torch.empty(
        (n_tokens * top_k, inter_dim // 2),
        dtype=torch.uint8,
        device=device,
    )
    q_inter_scale = empty_swizzled_cdna4_mxfp4_scale(
        n_tokens * top_k,
        inter_dim // 32,
        device=device,
    )
    if use_route_tiles:
        from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.moe_sorting import (
            gluon_mxfp4_moe_route_descriptors_gfx950,
            gluon_mxfp4_moe_route_pack_gfx950,
            gluon_mxfp4_moe_route_tiles_gfx950,
        )

        if out is None:
            out = torch.empty(
                (n_tokens * top_k, out_dim), dtype=out_dtype, device=device
            )
        expanded_idx = torch.empty(
            (n_tokens * top_k,), dtype=torch.int32, device=device
        )
        # Descriptors group one expert's routes; results retain token/slot order.
        num_local_experts = int(w13_runtime.shape[0])
        block_m = 16
        group_count = torch.empty(1, dtype=torch.int32, device=device)
        if n_tokens <= 16:
            sorted_routes = torch.empty(
                block_m * top_k, dtype=torch.int32, device=device
            )
            descriptors = torch.empty(
                (num_local_experts, 3), dtype=torch.int32, device=device
            )
            gluon_mxfp4_moe_route_descriptors_gfx950[(1,)](
                topk_ids,
                sorted_routes,
                descriptors,
                group_count,
                expanded_idx,
                n_tokens,
                EXPERT_START=expert_start,
                NUM_EXPERTS=num_local_experts,
                TOPK=top_k,
                MAX_ROWS=block_m,
                num_warps=4,
            )
        else:
            max_rows = 64
            sorted_routes = torch.empty(
                num_local_experts * max_rows, dtype=torch.int32, device=device
            )
            descriptors = torch.empty(
                (num_local_experts * (max_rows // block_m), 3),
                dtype=torch.int32,
                device=device,
            )
            expert_counts = torch.empty(
                num_local_experts, dtype=torch.int32, device=device
            )
            gluon_mxfp4_moe_route_pack_gfx950[(num_local_experts,)](
                topk_ids,
                sorted_routes,
                expert_counts,
                expanded_idx,
                n_tokens,
                EXPERT_START=expert_start,
                NUM_EXPERTS=num_local_experts,
                TOPK=top_k,
                MAX_ROWS=max_rows,
                num_warps=4,
            )
            gluon_mxfp4_moe_route_tiles_gfx950[(1,)](
                expert_counts,
                descriptors,
                group_count,
                NUM_EXPERTS=num_local_experts,
                BLOCK_M=block_m,
                MAX_ROWS=max_rows,
                num_warps=4,
            )
        gluon_mxfp4_moe_stage1_route_tiles_gfx950[(2048,)](
            q_hidden,
            q_hidden_scale,
            w13_runtime,
            w13_scale,
            sorted_routes,
            descriptors,
            group_count,
            q_inter,
            q_inter_scale,
            HIDDEN=hidden_dim,
            INTERMEDIATE=inter_dim,
            TOPK=top_k,
            X_ROW=q_hidden.stride(0),
            XS0=q_hidden_scale.stride(0),
            XS1=q_hidden_scale.stride(1),
            WE=w13_runtime.stride(0),
            SE=w13_scale.stride(0),
            SS0=w13_scale.stride(1),
            SS1=w13_scale.stride(2),
            OS0=q_inter_scale.stride(0),
            OS1=q_inter_scale.stride(1),
            BLOCK_N=64,
            BLOCK_K=128,
            BLOCK_M=block_m,
            num_warps=4,
        )
        gluon_mxfp4_moe_stage2_route_tiles_gfx950[(2048,)](
            q_inter,
            q_inter_scale,
            w2_runtime,
            w2_scale,
            sorted_routes,
            descriptors,
            group_count,
            out,
            INTERMEDIATE=inter_dim,
            OUTPUT=out_dim,
            XS0=q_inter_scale.stride(0),
            XS1=q_inter_scale.stride(1),
            WE=w2_runtime.stride(0),
            SE=w2_scale.stride(0),
            SS0=w2_scale.stride(1),
            SS1=w2_scale.stride(2),
            BLOCK_N=16,
            BLOCK_K=256 if n_tokens <= 8 else 128,
            BLOCK_M=block_m,
            num_warps=1,
        )
        return out, None, expanded_idx

    stage1_block_n = 64
    invoke_stage1_mxfp4_mfma_decode_gluon(
        q_hidden,
        q_hidden_scale,
        w13_runtime,
        w13_scale,
        topk_ids,
        q_inter,
        top_k,
        expert_start,
        q_inter_scale,
        BLOCK_N=stage1_block_n,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
        swiglu_beta=swiglu_beta,
    )
    if out is None:
        output_shape = (
            (n_tokens, out_dim) if do_finalize else (n_tokens * top_k, out_dim)
        )
        out = torch.empty(output_shape, dtype=out_dtype, device=device)
    stage2_block_n, stage2_block_k = _select_direct_stage2_tiles(
        n_tokens,
        ep_deferred=is_ep_deferred,
    )
    if slot_parallel:
        stage2_block_n = 32 if n_tokens >= 12 else 16
    stage2_pipeline_k = not is_ep_deferred
    expanded_idx = (
        None
        if do_finalize
        else torch.empty((n_tokens * top_k,), dtype=torch.int32, device=device)
    )
    invoke_stage2_mxfp4_mfma_decode_gluon(
        q_inter,
        q_inter_scale,
        w2_runtime,
        w2_scale,
        topk_ids,
        topk_weights,
        out,
        top_k,
        expert_start,
        do_finalize,
        expanded_idx,
        BLOCK_N=stage2_block_n,
        BLOCK_K=stage2_block_k,
        PIPELINE_K=stage2_pipeline_k,
        SLOT_PARALLEL=slot_parallel,
    )
    if not do_finalize:
        assert expanded_idx is not None
        return out, None, expanded_idx
    return out


def _maybe_precomputed_mxfp4_mfma_decode(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    *,
    w13_mx_scale: torch.Tensor,
    w2_mx_scale: torch.Tensor,
    top_k: int,
    w13_bias: Optional[torch.Tensor],
    w2_bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    max_m: int,
    precomputed_topk_weights: torch.Tensor | None,
    precomputed_topk_ids: torch.Tensor | None,
    swiglu_alpha: float,
    swiglu_limit: float,
    swiglu_beta: float,
    min_m: int = _PRECOMPUTED_MFMA_MIN_M,
) -> torch.Tensor | None:
    """Precomputed top-k dynamic MXFP4-activation decode path.

    This covers the M=4/8 regime where the BF16-activation scalar decode path
    is dominated by global-load waits, while the reference MXFP4xMXFP4 MFMA path is
    faster once routing/top-k has already been computed by the caller.
    """
    n_tokens = int(hidden_states.shape[0])
    if (
        precomputed_topk_weights is None
        or precomputed_topk_ids is None
        or n_tokens < min_m
        or n_tokens > max_m
        or hidden_states.dtype != torch.bfloat16
        or out_dtype != torch.bfloat16
        or w13_bias is not None
        or w2_bias is not None
        or top_k != int(precomputed_topk_ids.shape[1])
    ):
        return None

    topk_ids = (
        precomputed_topk_ids
        if precomputed_topk_ids.dtype == torch.int32
        else precomputed_topk_ids.to(torch.int32)
    )
    topk_weights = (
        precomputed_topk_weights
        if precomputed_topk_weights.dtype == torch.float32
        else precomputed_topk_weights.to(torch.float32)
    )

    w13_runtime = _extract_gluon_raw_w(w13_weight)
    w2_runtime = _extract_gluon_raw_w(w2_weight)
    w13_scale = _extract_gluon_raw_s(w13_mx_scale)
    w2_scale = _extract_gluon_raw_s(w2_mx_scale)
    if (
        not isinstance(w13_runtime, torch.Tensor)
        or not isinstance(w2_runtime, torch.Tensor)
        or not isinstance(w13_scale, torch.Tensor)
        or not isinstance(w2_scale, torch.Tensor)
        or w13_runtime.dtype != torch.uint8
        or w2_runtime.dtype != torch.uint8
        or w13_scale.dtype != torch.uint8
        or w2_scale.dtype != torch.uint8
        or w13_runtime.ndim != 3
        or w2_runtime.ndim != 3
    ):
        return None
    if (
        not bool(getattr(w13_runtime, "is_shuffled_for_gluon_dot", False))
        or not bool(getattr(w2_runtime, "is_shuffled_for_gluon_dot", False))
        or int(getattr(w13_runtime, "gluon_dot_block_k_pk", 0)) != 128
        or int(getattr(w13_runtime, "gluon_dot_block_n", 0)) != 128
        or int(getattr(w2_runtime, "gluon_dot_block_k_pk", 0)) != 128
        or int(getattr(w2_runtime, "gluon_dot_block_n", 0)) != 128
        or w13_scale.stride(-2) != 1
        or w2_scale.stride(-2) != 1
    ):
        return None

    hidden_dim = int(hidden_states.shape[1])
    w13_k_pk = int(getattr(w13_runtime, "original_k_pk", int(w13_runtime.shape[1])))
    if w13_k_pk * 2 != hidden_dim:
        return None
    inter_dim = int(w13_runtime.shape[2]) // 2
    w2_k_pk = int(getattr(w2_runtime, "original_k_pk", int(w2_runtime.shape[1])))
    out_dim = int(getattr(w2_runtime, "original_n", int(w2_runtime.shape[2])))
    if w2_k_pk * 2 != inter_dim or out_dim != hidden_dim:
        return None
    if not gluon_precomputed_topk_route_supported(
        topk_weights,
        topk_ids,
        num_experts=int(w13_runtime.shape[0]),
        dtype=router_logits.dtype,
    ):
        return None

    use_flat_m1_route = n_tokens == 1
    if use_flat_m1_route:
        ragged_metadata, gather_indx, scatter_indx, gate_scal = (
            gluon_precomputed_topk_flat_m1_route(
                topk_weights,
                topk_ids,
                num_experts=int(w13_runtime.shape[0]),
                dtype=router_logits.dtype,
            )
        )
        x_scale_ragged_padded = False
    else:
        ragged_metadata, gather_indx, scatter_indx, gate_scal = (
            gluon_precomputed_topk_fused_route(
                topk_weights,
                topk_ids,
                num_experts=int(w13_runtime.shape[0]),
                dtype=router_logits.dtype,
            )
        )
        x_scale_ragged_padded = True
    tiny_m_matmul_kwargs = (
        {
            "block_m": 64,
            "block_n": 128,
            "block_k": 256,
            "use_slice_n": False,
        }
        if n_tokens <= 2
        else {}
    )
    gemm1_input, gemm1_scale = _quantize_mxfp4_activation(
        hidden_states,
        gather_indx=gather_indx,
        ragged_metadata=ragged_metadata if x_scale_ragged_padded else None,
    )
    act = _swiglu_activation(swiglu_alpha, swiglu_limit, swiglu_beta)
    intermediate_cache = gluon_mxfp_ragged_matmul(
        gemm1_input,
        w13_runtime,
        None,
        w_mx_scale=w13_scale,
        x_mx_scale=gemm1_scale,
        x_format="e2m1",
        out_dtype=out_dtype,
        a_ragged_metadata=ragged_metadata,
        fused_activation=act,
        x_scale_ragged_padded=x_scale_ragged_padded,
        **tiny_m_matmul_kwargs,
    )
    gemm2_input, gemm2_scale = _quantize_mxfp4_activation(
        intermediate_cache,
        ragged_metadata=ragged_metadata if x_scale_ragged_padded else None,
    )
    return gluon_mxfp_ragged_matmul(
        gemm2_input,
        w2_runtime,
        None,
        w_mx_scale=w2_scale,
        x_mx_scale=gemm2_scale,
        x_format="e2m1",
        out_dtype=out_dtype,
        a_ragged_metadata=ragged_metadata,
        scatter_indx=scatter_indx,
        gammas=gate_scal,
        n_tokens=n_tokens,
        n_expts_act=top_k,
        x_scale_ragged_padded=x_scale_ragged_padded,
        **tiny_m_matmul_kwargs,
    )


def _maybe_route_owned_mxfp4_mfma_decode(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    *,
    w13_mx_scale: torch.Tensor,
    w2_mx_scale: torch.Tensor,
    top_k: int,
    correction_bias: torch.Tensor | None,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    routing_method_type: int,
    w13_bias: Optional[torch.Tensor],
    w2_bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    max_m: int,
    swiglu_alpha: float,
    swiglu_limit: float,
    swiglu_beta: float,
    allow_generic_fallback: bool = True,
) -> torch.Tensor | None:
    """Route-owned MXFP4xMXFP4 decode for tiny batches without precomputed top-k.

    Computes top-k in Gluon (softmax or sigmoid-bias) directly from the router
    logits, then prefers the direct top-k MXFP4xMXFP4 decode path. When
    ``allow_generic_fallback`` is set it falls back to the generic ragged MFMA;
    otherwise it returns ``None`` so the caller's own generic path takes over.
    """
    n_tokens = int(router_logits.shape[0])
    if n_tokens < _ROUTE_OWNED_MIN_M or n_tokens > max_m:
        return None
    if not gluon_route_supported(router_logits, top_k, router_logits.dtype):
        return None

    method = int(routing_method_type)
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.routing import (
        invoke_sigmoid_bias_topk_route_gluon,
        invoke_softmax_topk_route_gluon,
    )

    if method == _ROUTING_METHOD_RENORMALIZE:
        return None
    if correction_bias is not None and n_group == 1 and topk_group == 1:
        topk_ids, topk_weights = invoke_sigmoid_bias_topk_route_gluon(
            router_logits,
            correction_bias,
            top_k,
            routed_scaling_factor=routed_scaling_factor,
            normalize_topk_weights=normalize_topk_weights,
        )
    elif method == 0:
        # Global softmax, optionally with a choice bias. Grouped softmax only
        # degenerates to this when there is exactly one group.
        if n_group not in (0, 1) or topk_group not in (0, 1):
            return None
        if correction_bias is not None and (n_group != 0 or topk_group != 0):
            return None
        # Scaling-semantics guard. For the grouped-one-group config
        # (n_group == topk_group == 1) the generic path routes through
        # ``default_grouped_route`` -> ``_grouped_topk_reference``, which does
        # NOT apply ``routed_scaling_factor`` when weights are left unnormalized
        # (scale_when_unnormalized=False). ``invoke_softmax_topk_route_gluon``
        # always multiplies by ROUTED_SCALING_FACTOR, so those two disagree by
        # exactly ``routed_scaling_factor`` for unnormalized weights. Defer that
        # case to the generic path so results are unchanged. (The non-grouped
        # n_group==topk_group==0 case uses default_scaled_route ->
        # _softmax_topk_reference with scale_when_unnormalized=True, which
        # matches the gluon kernel, so it is safe here.)
        uses_grouped = _uses_grouped_routing(n_group, topk_group)
        if (
            uses_grouped
            and not normalize_topk_weights
            and float(routed_scaling_factor) != 1.0
        ):
            return None
        topk_ids, topk_weights = invoke_softmax_topk_route_gluon(
            router_logits,
            top_k,
            correction_bias=correction_bias,
            routed_scaling_factor=routed_scaling_factor,
            normalize_topk_weights=normalize_topk_weights,
        )
    else:
        return None

    out = _maybe_precomputed_mxfp4_direct_mfma_decode(
        hidden_states,
        w13_weight,
        w2_weight,
        w13_mx_scale=w13_mx_scale,
        w2_mx_scale=w2_mx_scale,
        top_k=top_k,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        out_dtype=out_dtype,
        max_m=max_m,
        route_tile_max_m=0,
        precomputed_topk_weights=topk_weights,
        precomputed_topk_ids=topk_ids,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
        swiglu_beta=swiglu_beta,
        do_finalize=True,
        expert_start=0,
        global_num_experts=None,
        x_quantized=None,
    )
    if out is not None:
        return out
    if not allow_generic_fallback:
        return None

    return _maybe_precomputed_mxfp4_mfma_decode(
        hidden_states,
        router_logits,
        w13_weight,
        w2_weight,
        w13_mx_scale=w13_mx_scale,
        w2_mx_scale=w2_mx_scale,
        top_k=top_k,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        out_dtype=out_dtype,
        max_m=max_m,
        precomputed_topk_weights=topk_weights,
        precomputed_topk_ids=topk_ids,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
        swiglu_beta=swiglu_beta,
        do_finalize=True,
        expert_start=0,
        global_num_experts=None,
    )


# Tuned default: rocprofv3 kernel-trace (real GPU time) shows package prefill
# beats the reference ragged path at every M >= 9 (1.03x-1.24x, bit-exact) on
# the Kimi shape, and the decode kernels own M <= 8. So package prefill is
# selected automatically for M >= this threshold (no env toggle).
_PACKAGE_PREFILL_MIN_M = 9


def _maybe_gluon_package_mxfp4_prefill(
    hidden_states: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    router_logits: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    *,
    w13_mx_scale: torch.Tensor,
    w2_mx_scale: torch.Tensor,
    top_k: int,
    correction_bias: torch.Tensor | None,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    routing_method_type: int,
    precomputed_topk_weights: torch.Tensor | None,
    precomputed_topk_ids: torch.Tensor | None,
    out_dtype: torch.dtype,
    swiglu_alpha: float,
    swiglu_limit: float,
    swiglu_beta: float,
    do_finalize: bool,
    x_quantized: tuple[torch.Tensor, torch.Tensor] | None,
    situ_linear_beta: float | None = None,
    expert_start: int = 0,
    global_num_experts: int | None = None,
    out: torch.Tensor | None = None,
    deferred_out: torch.Tensor | None = None,
    force_reduce: bool | None = None,
    activation_format: str = "e2m1",
) -> torch.Tensor | tuple[torch.Tensor, None, torch.Tensor] | None:
    """Dispatch into the dedicated gfx950 block-ragged prefill package.

    The package supports E2M1 or E4M3 block-scaled activations with MXFP4
    weights. The block-aligned sort and both stage GEMMs are launched directly.
    ``expert_start`` lets an EP rank consume global route IDs while sorting only
    its contiguous local expert range.

    Selection is automatic: this returns ``None`` (so the caller falls back to
    the reference path) unless the batch is large enough
    (``M >= _PACKAGE_PREFILL_MIN_M``) and the weights were gdot128-preshuffled
    (the preprocessor attaches the zero-copy gdot128-storage aliases).
    """
    n_tokens, hidden_dim, device, _ = _mxfp4_input_metadata(hidden_states)
    if n_tokens < _PACKAGE_PREFILL_MIN_M:
        return None
    if out_dtype != torch.bfloat16:
        return None
    if activation_format not in ("e2m1", "e4m3"):
        raise ValueError(
            "package prefill activation_format must be e2m1 or e4m3, "
            f"got {activation_format}"
        )
    if (
        x_quantized is not None or isinstance(hidden_states, tuple)
    ) and activation_format != "e2m1":
        raise ValueError("prequantized package input must use the E2M1 format")
    package_w13 = getattr(w13_weight, "gluon_package_prefill_weight", None)
    package_w13_scale = getattr(w13_weight, "gluon_package_prefill_scale", None)
    package_w2 = getattr(w2_weight, "gluon_package_prefill_weight", None)
    package_w2_scale = getattr(w2_weight, "gluon_package_prefill_scale", None)
    if not all(
        isinstance(t, torch.Tensor)
        for t in (package_w13, package_w13_scale, package_w2, package_w2_scale)
    ):
        return None

    topk_weights = precomputed_topk_weights
    topk_ids = precomputed_topk_ids
    if topk_weights is None or topk_ids is None:
        method = int(routing_method_type)
        if method == _ROUTING_METHOD_RENORMALIZE:
            topk_logits, topk_ids = _stable_topk_smaller_index(
                router_logits, k=top_k, dim=-1, sorted=True
            )
            topk_weights = topk_logits.exp()
            topk_weights = _normalize_route_weights(
                topk_weights,
                normalize_topk_weights=normalize_topk_weights,
                routed_scaling_factor=1.0,
                scale_when_unnormalized=False,
            )
            topk_weights = topk_weights.to(torch.float32)
            topk_ids = topk_ids.to(torch.int32)
        elif _uses_grouped_routing(n_group, topk_group):
            if correction_bias is None:
                topk_weights, topk_ids = _grouped_topk_reference(
                    router_logits,
                    top_k,
                    n_group=n_group,
                    topk_group=topk_group,
                    routed_scaling_factor=routed_scaling_factor,
                    normalize_topk_weights=normalize_topk_weights,
                )
            elif n_group == 1 and topk_group == 1:
                from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.routing import (
                    gluon_topk_route_supported,
                    invoke_sigmoid_bias_topk_route_gluon,
                )

                if gluon_topk_route_supported(router_logits, top_k):
                    topk_ids, topk_weights = invoke_sigmoid_bias_topk_route_gluon(
                        router_logits,
                        correction_bias,
                        top_k,
                        routed_scaling_factor=routed_scaling_factor,
                        normalize_topk_weights=normalize_topk_weights,
                    )
                else:
                    topk_weights, topk_ids = _biased_grouped_topk_reference(
                        router_logits,
                        correction_bias,
                        top_k,
                        n_group=n_group,
                        topk_group=topk_group,
                        routed_scaling_factor=routed_scaling_factor,
                        normalize_topk_weights=normalize_topk_weights,
                    )
            else:
                topk_weights, topk_ids = _biased_grouped_topk_reference(
                    router_logits,
                    correction_bias,
                    top_k,
                    n_group=n_group,
                    topk_group=topk_group,
                    routed_scaling_factor=routed_scaling_factor,
                    normalize_topk_weights=normalize_topk_weights,
                )
        elif _has_incomplete_grouped_routing(n_group, topk_group):
            return None
        else:
            topk_weights, topk_ids = _softmax_topk_reference(
                router_logits,
                top_k,
                correction_bias=correction_bias,
                routed_scaling_factor=routed_scaling_factor,
                normalize_topk_weights=normalize_topk_weights,
            )

    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    n_experts = int(package_w13.shape[0])
    if global_num_experts is None:
        global_num_experts = n_experts
    if (
        global_num_experts < n_experts
        or expert_start < 0
        or expert_start + n_experts > global_num_experts
    ):
        raise ValueError("local expert range exceeds global expert count")
    if force_reduce is None:
        # EP ranks own only a fraction of each token's routes. For TP E2M1,
        # paired-column atomics avoid the partials/reduce cost through 2048
        # rows. Larger outputs still favor scratch plus FP32 reduction.
        is_ep_shard = global_num_experts != n_experts or expert_start != 0
        atomic_max_m = 2048 if activation_format == "e2m1" else 64
        force_reduce = False if is_ep_shard else n_tokens > atomic_max_m
    inter_dim = int(package_w13.shape[1]) // 2
    if (
        int(package_w13.shape[1]) != 2 * inter_dim
        or int(package_w13.shape[2]) * 2 != hidden_dim
        or int(package_w2.shape[1]) != hidden_dim
        or int(package_w2.shape[2]) * 2 < inter_dim
    ):
        return None

    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.moe_sorting import (
        gluon_moe_sorting,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.prefill_stage1 import (
        invoke_gluon_mxfp4_moe_stage1,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.prefill_stage2 import (
        invoke_gluon_mxfp4_moe_stage2_1x2,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.quantize_gluon import (
        quantize_mxfp8_sorted_routes,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.scale import (
        gather_package_cdna4_scale,
    )

    config = _select_grouped_moe_config(
        n_tokens,
        hidden_dim=hidden_dim,
        inter_dim=inter_dim,
        stage2_k=int(package_w2.shape[2]) * 2,
        num_local_experts=n_experts,
        num_experts=global_num_experts,
        top_k=top_k,
        activation_format=activation_format,
        do_finalize=do_finalize,
        precomputed_routes=(
            precomputed_topk_ids is not None and precomputed_topk_weights is not None
        ),
        swiglu_args=(swiglu_alpha, swiglu_limit, swiglu_beta),
        situ_linear_beta=situ_linear_beta,
    )
    sort_block_m = config.block_m
    # A single route chunk fuses the short-prefill histogram and prefix scan.
    # Larger prefills retain chunk ordering for the stage-1 activation gather.
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, out = (
        gluon_moe_sorting(
            topk_ids,
            topk_weights,
            n_experts,
            hidden_dim,
            out_dtype,
            sort_block_m,
            use_minimum_route_chunks=config.use_minimum_route_chunks,
            runtime_rows=config.runtime_rows,
            expert_start=expert_start,
            out=out,
        )
    )

    # Stage 1 consumes route-sorted activations. E4M3 quantization gathers and
    # quantizes directly into that order; the existing E2M1 path preserves its
    # token-order quantize plus scale-gather contract.
    input_sorted = activation_format == "e4m3"
    if input_sorted:
        q_hidden, stage1_scale = quantize_mxfp8_sorted_routes(
            hidden_states,
            sorted_ids,
            num_valid_ids,
        )
    else:
        q_hidden, q_hidden_scale = _resolve_mxfp4_activation(
            hidden_states,
            x_quantized,
        )
        stage1_scale = gather_package_cdna4_scale(
            q_hidden_scale,
            sorted_ids,
            source_rows=n_tokens,
            cols=hidden_dim,
            top_k=top_k,
            flatten_topk=False,
        )
    stage2_k = int(package_w2.shape[2]) * 2
    stage2_k_scale = stage2_k // MXFP4_BLOCK
    stage2_scale_rows = triton.cdiv(int(sorted_ids.shape[0]), 32) * 32
    # Stage 1 writes the activated intermediate directly in the selected
    # sorted block-scaled format, including the zero K-padding required by W2.
    intermediate_format = activation_format
    inter_div = 1 if intermediate_format == "e4m3" else 2
    inter_dtype = torch.float8_e4m3fn if intermediate_format == "e4m3" else torch.uint8
    q_inter = torch.empty(
        (sorted_ids.shape[0], stage2_k // inter_div),
        dtype=inter_dtype,
        device=device,
    )
    stage2_scale = torch.empty(
        (stage2_scale_rows, stage2_k_scale),
        dtype=torch.uint8,
        device=device,
    )
    invoke_gluon_mxfp4_moe_stage1(
        q_hidden,
        package_w13.view(torch.uint8),
        None,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        q_inter,
        top_k,
        w1_scale=package_w13_scale.view(torch.uint8),
        a1_scale=stage1_scale,
        block_m=sort_block_m,
        num_warps=config.stage1_num_warps,
        b_preshuffled=True,
        b_gdot128=True,
        swiglu_alpha=float(swiglu_alpha),
        swiglu_limit=float(swiglu_limit),
        swiglu_beta=float(swiglu_beta),
        situ_linear_beta=situ_linear_beta,
        output_sorted=True,
        output_quantized=True,
        out_scale=stage2_scale,
        output_k=stage2_k,
        a_format=activation_format,
        output_format=intermediate_format,
        input_sorted=input_sorted,
        source_tokens=n_tokens,
    )

    # Both stages consume the same route blocking so the lower-padding choice
    # does not require a second sort or any metadata conversion.
    stage2_block_m = sort_block_m
    s2_sorted_ids = sorted_ids
    s2_sorted_weights = sorted_weights
    s2_sorted_expert_ids = sorted_expert_ids
    s2_num_valid_ids = num_valid_ids

    # Keep the physical A/W/scale buffers; skip only whole padded BK128
    # tiles. Short intermediates still need the two-tile stage2 prologue.
    stage2_logical_k = max(256, triton.cdiv(inter_dim, 128) * 128)
    stage2_out = invoke_gluon_mxfp4_moe_stage2_1x2(
        q_inter[:, : stage2_logical_k // inter_div],
        None,
        package_w2.view(torch.uint8),
        s2_sorted_ids,
        s2_sorted_expert_ids,
        s2_num_valid_ids,
        out,
        top_k,
        do_finalize,
        # Deferred E2M1 consumers use expanded_idx to mask nonlocal routes.
        initialize_unowned_routes=do_finalize or activation_format != "e2m1",
        deferred_out=deferred_out,
        w2_scale=package_w2_scale.view(torch.uint8),
        a2_scale=stage2_scale,
        sorted_weights=s2_sorted_weights,
        b_preshuffled=True,
        b_gdot128=True,
        block_m=stage2_block_m,
        sort_block_m=stage2_block_m,
        force_reduce=force_reduce,
        input_sorted=True,
        a_format=intermediate_format,
    )
    if not do_finalize:
        from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.routing import (
            gluon_local_route_map,
        )

        expanded_idx = gluon_local_route_map(topk_ids, expert_start, n_experts)
        return stage2_out, None, expanded_idx
    return out


def gluon_mxfp_dynamic_mxfp4_fused_moe(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    *,
    w13_mx_scale: torch.Tensor,
    w2_mx_scale: torch.Tensor,
    top_k: int,
    correction_bias: torch.Tensor | None,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    routing_method_type: int = 0,
    w13_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    swiglu_beta: float = 1.0,
    precomputed_topk_weights: torch.Tensor | None = None,
    precomputed_topk_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Route + dispatch + combine for dynamic MXFP4 activations on gfx950.

    The route path follows DeepSeek/Kimi grouped-biased top-k when the model
    supplies a correction bias; the small-M decode case fuses top-k and ragged
    metadata construction in one Gluon kernel.
    """
    route_dtype = router_logits.dtype

    has_precomputed_topk = (
        precomputed_topk_weights is not None and precomputed_topk_ids is not None
    )
    n_tokens = router_logits.shape[0]
    # Package prefill is selected automatically by batch size and weight layout
    # (see _maybe_gluon_package_mxfp4_prefill); it returns None when not
    # applicable and the dispatch falls through to decode / the reference path.
    # It uses the in-house gluon_moe_sorting and runs on the caller's stream, so
    # no cross-stream fence or default-stream ownership is required.
    package_prefill_out = _maybe_gluon_package_mxfp4_prefill(
        hidden_states,
        router_logits,
        w13_weight,
        w2_weight,
        w13_mx_scale=w13_mx_scale,
        w2_mx_scale=w2_mx_scale,
        top_k=top_k,
        correction_bias=correction_bias,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=routed_scaling_factor,
        normalize_topk_weights=normalize_topk_weights,
        routing_method_type=routing_method_type,
        precomputed_topk_weights=precomputed_topk_weights,
        precomputed_topk_ids=precomputed_topk_ids,
        out_dtype=out_dtype,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
        swiglu_beta=swiglu_beta,
        do_finalize=True,
        x_quantized=None,
        expert_start=0,
        global_num_experts=None,
    )
    if package_prefill_out is not None:
        return package_prefill_out

    # Small-M decode fast paths (M < _PACKAGE_PREFILL_MIN_M). The kernel is
    # selected purely by batch size and whether the caller supplied precomputed
    # top-k; each helper returns None when the weights/shapes are unsupported
    # and the dispatch falls through to the generic reference path below.
    if has_precomputed_topk:
        if int(n_tokens) <= _DECODE_MAX_M:
            decode_out = _maybe_precomputed_mxfp4_direct_mfma_decode(
                hidden_states,
                w13_weight,
                w2_weight,
                w13_mx_scale=w13_mx_scale,
                w2_mx_scale=w2_mx_scale,
                top_k=top_k,
                w13_bias=w13_bias,
                w2_bias=w2_bias,
                out_dtype=out_dtype,
                max_m=_DECODE_MAX_M,
                route_tile_max_m=0,
                precomputed_topk_weights=precomputed_topk_weights,
                precomputed_topk_ids=precomputed_topk_ids,
                swiglu_alpha=swiglu_alpha,
                swiglu_limit=swiglu_limit,
                swiglu_beta=swiglu_beta,
                expert_start=0,
                global_num_experts=None,
                do_finalize=True,
                x_quantized=None,
            )
            if decode_out is not None:
                return decode_out

            if int(n_tokens) >= _PRECOMPUTED_MFMA_MIN_M:
                decode_out = _maybe_precomputed_mxfp4_mfma_decode(
                    hidden_states,
                    router_logits,
                    w13_weight,
                    w2_weight,
                    w13_mx_scale=w13_mx_scale,
                    w2_mx_scale=w2_mx_scale,
                    top_k=top_k,
                    w13_bias=w13_bias,
                    w2_bias=w2_bias,
                    out_dtype=out_dtype,
                    max_m=_DECODE_MAX_M,
                    precomputed_topk_weights=precomputed_topk_weights,
                    precomputed_topk_ids=precomputed_topk_ids,
                    swiglu_alpha=swiglu_alpha,
                    swiglu_limit=swiglu_limit,
                    swiglu_beta=swiglu_beta,
                )
                if decode_out is not None:
                    return decode_out

    else:
        if int(n_tokens) <= _DECODE_MAX_M:
            decode_out = _maybe_route_owned_mxfp4_mfma_decode(
                hidden_states,
                router_logits,
                w13_weight,
                w2_weight,
                w13_mx_scale=w13_mx_scale,
                w2_mx_scale=w2_mx_scale,
                top_k=top_k,
                correction_bias=correction_bias,
                n_group=n_group,
                topk_group=topk_group,
                routed_scaling_factor=routed_scaling_factor,
                normalize_topk_weights=normalize_topk_weights,
                routing_method_type=routing_method_type,
                w13_bias=w13_bias,
                w2_bias=w2_bias,
                out_dtype=out_dtype,
                max_m=_DECODE_MAX_M,
                swiglu_alpha=swiglu_alpha,
                swiglu_limit=swiglu_limit,
                swiglu_beta=swiglu_beta,
                allow_generic_fallback=True,
            )
            if decode_out is not None:
                return decode_out

    if has_precomputed_topk:
        ragged_metadata, gather_indx, scatter_indx, gate_scal = _route_from_topk(
            precomputed_topk_weights.to(torch.float32),
            precomputed_topk_ids.to(torch.int32),
            num_experts=router_logits.shape[1],
            dtype=route_dtype,
        )
    else:
        ragged_metadata, gather_indx, scatter_indx, gate_scal = _dynamic_mxfp4_route(
            router_logits,
            top_k,
            correction_bias=correction_bias,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            normalize_topk_weights=normalize_topk_weights,
            routing_method_type=routing_method_type,
            dtype=route_dtype,
        )
    return _gluon_mxfp_dynamic_mxfp4_fused_moe_from_route(
        hidden_states,
        w13_weight,
        w2_weight,
        w13_mx_scale=w13_mx_scale,
        w2_mx_scale=w2_mx_scale,
        ragged_metadata=ragged_metadata,
        gather_indx=gather_indx,
        scatter_indx=scatter_indx,
        gate_scal=gate_scal,
        top_k=top_k,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        out_dtype=out_dtype,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
        swiglu_beta=swiglu_beta,
    )


def gluon_mxfp_precomputed_mxfp4_fused_moe(
    hidden_states: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    *,
    w13_mx_scale: torch.Tensor,
    w2_mx_scale: torch.Tensor,
    expert_start: int,
    global_num_experts: int,
    do_finalize: bool,
    x_quantized: tuple[torch.Tensor, torch.Tensor] | None,
    w13_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    swiglu_beta: float = 1.0,
    out: torch.Tensor | None = None,
    deferred_out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, None, torch.Tensor]:
    """Dispatch precomputed routes through dynamic A4W4 gfx950 experts.

    Expert IDs are global and local weights form the contiguous range starting
    at ``expert_start``. Only deferred rows selected by the returned local map
    are defined; consumers must not read routes marked ``-1``.
    Deferred mode returns one canonical row per route so
    the shared model epilogue can normalize, weight, and reduce across ranks.
    ``out`` is the finalized ``[tokens, hidden]`` destination; with
    ``do_finalize=False``, ``deferred_out`` may be a contiguous BF16
    ``[tokens * top_k, hidden]`` route destination that overlaps no input.
    """
    if do_finalize and deferred_out is not None:
        raise ValueError("deferred_out requires do_finalize=False")
    if not do_finalize and out is not None:
        raise ValueError("deferred routes take deferred_out, not out")
    rows, hidden_dim, device, _ = _mxfp4_input_metadata(hidden_states)
    if isinstance(hidden_states, tuple):
        _resolve_mxfp4_activation(hidden_states, x_quantized)
    if topk_ids.ndim != 2:
        raise ValueError(f"topk_ids must be rank-2, got {tuple(topk_ids.shape)}")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "topk_weights and topk_ids must have the same shape, got "
            f"{tuple(topk_weights.shape)} and {tuple(topk_ids.shape)}"
        )

    w13_raw = _extract_gluon_raw_w_unshuffled(w13_weight)
    if not isinstance(w13_raw, torch.Tensor) or w13_raw.ndim != 3:
        raise ValueError("w13_weight must expose a rank-3 expert weight tensor")
    num_experts = int(w13_raw.shape[0])
    if (
        expert_start < 0
        or global_num_experts < num_experts
        or expert_start + num_experts > global_num_experts
    ):
        raise ValueError("local expert range exceeds global expert count")
    n_tokens, top_k = topk_ids.shape
    if n_tokens != rows:
        raise ValueError("input and top-k must have the same token count")
    if deferred_out is not None:
        if out_dtype != torch.bfloat16:
            raise ValueError("deferred output requires BF16 output dtype")
        _validate_deferred_out(
            deferred_out,
            tokens=n_tokens,
            topk=top_k,
            width=hidden_dim,
            device=device,
            inputs=(
                *(
                    hidden_states
                    if isinstance(hidden_states, tuple)
                    else (hidden_states,)
                ),
                *(x_quantized if x_quantized is not None else ()),
                topk_weights,
                topk_ids,
                w13_raw,
                _extract_gluon_raw_w(w13_weight),
                _extract_gluon_raw_w_unshuffled(w2_weight),
                _extract_gluon_raw_w(w2_weight),
                _extract_gluon_raw_s(w13_mx_scale),
                _extract_gluon_raw_s(w2_mx_scale),
                w13_bias,
                w2_bias,
            ),
        )
    if n_tokens == 0:
        destination = out if do_finalize else deferred_out
        empty = (
            destination
            if destination is not None
            else torch.empty((0, hidden_dim), dtype=out_dtype, device=device)
        )
        if do_finalize:
            return empty
        return (
            empty,
            None,
            torch.empty((0,), dtype=torch.int32, device=device),
        )

    direct_max_m = _DECODE_MAX_M if do_finalize else _DEFERRED_DIRECT_DECODE_MAX_M
    direct_out = _maybe_precomputed_mxfp4_direct_mfma_decode(
        hidden_states,
        w13_weight,
        w2_weight,
        w13_mx_scale=w13_mx_scale,
        w2_mx_scale=w2_mx_scale,
        top_k=int(top_k),
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        out_dtype=out_dtype,
        max_m=direct_max_m,
        route_tile_max_m=0 if do_finalize else 48,
        precomputed_topk_weights=topk_weights,
        precomputed_topk_ids=topk_ids,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
        swiglu_beta=swiglu_beta,
        expert_start=expert_start,
        global_num_experts=global_num_experts,
        do_finalize=do_finalize,
        x_quantized=x_quantized,
        out=out if do_finalize else deferred_out,
    )
    if direct_out is not None:
        return direct_out

    package_out = _maybe_gluon_package_mxfp4_prefill(
        hidden_states,
        topk_weights,
        w13_weight,
        w2_weight,
        w13_mx_scale=w13_mx_scale,
        w2_mx_scale=w2_mx_scale,
        top_k=int(top_k),
        correction_bias=None,
        n_group=0,
        topk_group=0,
        routed_scaling_factor=1.0,
        normalize_topk_weights=False,
        routing_method_type=0,
        precomputed_topk_weights=topk_weights,
        precomputed_topk_ids=topk_ids,
        out_dtype=out_dtype,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
        swiglu_beta=swiglu_beta,
        do_finalize=do_finalize,
        x_quantized=x_quantized,
        expert_start=expert_start,
        global_num_experts=global_num_experts,
        out=out,
        deferred_out=deferred_out,
    )
    if package_out is not None:
        return package_out
    if (
        isinstance(hidden_states, tuple)
        or expert_start != 0
        or global_num_experts != num_experts
        or not do_finalize
    ):
        raise ValueError(
            "gfx950 dynamic A4W4 EP requires the direct decode or package prefill path"
        )
    ragged_metadata, gather_indx, scatter_indx, gate_scal = _route_from_topk(
        topk_weights,
        topk_ids,
        num_experts,
        dtype=topk_weights.dtype,
    )

    return _gluon_mxfp_dynamic_mxfp4_fused_moe_from_route(
        hidden_states,
        w13_weight,
        w2_weight,
        w13_mx_scale=w13_mx_scale,
        w2_mx_scale=w2_mx_scale,
        ragged_metadata=ragged_metadata,
        gather_indx=gather_indx,
        scatter_indx=scatter_indx,
        gate_scal=gate_scal,
        top_k=int(topk_ids.shape[1]),
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        out_dtype=out_dtype,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
        swiglu_beta=swiglu_beta,
        out=out,
    )


def _gluon_mxfp_dynamic_mxfp4_fused_moe_from_route(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    *,
    w13_mx_scale: torch.Tensor,
    w2_mx_scale: torch.Tensor,
    ragged_metadata: RaggedTensorMetadata,
    gather_indx: torch.Tensor,
    scatter_indx: torch.Tensor,
    gate_scal: torch.Tensor,
    top_k: int,
    w13_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    swiglu_beta: float = 1.0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    n_tokens = hidden_states.shape[0]

    act = FusedActivation(
        FnSpecs("swiglu", swiglu_fn, ("alpha", "limit", "beta"), reduction_n=2),
        (swiglu_alpha, swiglu_limit, swiglu_beta),
    )

    gemm1_input, gemm1_scale = _quantize_mxfp4_activation(
        hidden_states,
        gather_indx=gather_indx,
        ragged_metadata=ragged_metadata,
    )
    intermediate_cache, gemm2_scale = gluon_mxfp_ragged_matmul(
        gemm1_input,
        w13_weight,
        w13_bias,
        w_mx_scale=w13_mx_scale,
        x_mx_scale=gemm1_scale,
        x_format="e2m1",
        out_dtype=out_dtype,
        a_ragged_metadata=ragged_metadata,
        fused_activation=act,
        out_quant_format="mxfp4",
        x_scale_ragged_padded=True,
    )
    return gluon_mxfp_ragged_matmul(
        intermediate_cache,
        w2_weight,
        w2_bias,
        w_mx_scale=w2_mx_scale,
        x_mx_scale=gemm2_scale,
        x_format="e2m1",
        out_dtype=out_dtype,
        a_ragged_metadata=ragged_metadata,
        scatter_indx=scatter_indx,
        gammas=gate_scal,
        n_tokens=n_tokens,
        n_expts_act=top_k,
        x_scale_ragged_padded=True,
        out=out,
    )


def _dynamic_mxfp4_route(
    router_logits: torch.Tensor,
    top_k: int,
    *,
    correction_bias: torch.Tensor | None,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    routing_method_type: int,
    dtype: torch.dtype,
) -> tuple[RaggedTensorMetadata, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_tokens = router_logits.shape[0]

    if int(routing_method_type) == _ROUTING_METHOD_RENORMALIZE:
        return default_packed_topk_route(
            router_logits,
            top_k,
            normalize_topk_weights=normalize_topk_weights,
            dtype=dtype,
        )

    if _uses_grouped_routing(n_group, topk_group):
        if correction_bias is None:
            return default_grouped_route(
                router_logits,
                top_k,
                n_group=n_group,
                topk_group=topk_group,
                routed_scaling_factor=routed_scaling_factor,
                normalize_topk_weights=normalize_topk_weights,
                dtype=dtype,
            )
        if n_tokens <= SMALLM_MAX_M and gluon_biased_grouped_route_supported(
            router_logits,
            correction_bias,
            top_k,
            n_group=n_group,
            topk_group=topk_group,
            dtype=dtype,
        ):
            return gluon_biased_grouped_fused_route(
                router_logits,
                correction_bias,
                top_k,
                n_group=n_group,
                topk_group=topk_group,
                routed_scaling_factor=routed_scaling_factor,
                normalize_topk_weights=normalize_topk_weights,
                dtype=dtype,
            )
        return default_biased_grouped_route(
            router_logits,
            correction_bias,
            top_k,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            normalize_topk_weights=normalize_topk_weights,
            dtype=dtype,
        )

    if _has_incomplete_grouped_routing(n_group, topk_group):
        raise ValueError(
            "grouped routing requires both n_group and topk_group; "
            f"got n_group={n_group}, topk_group={topk_group}"
        )

    if correction_bias is not None:
        return default_biased_route(
            router_logits,
            correction_bias,
            top_k,
            routed_scaling_factor=routed_scaling_factor,
            normalize_topk_weights=normalize_topk_weights,
            dtype=dtype,
        )

    # Dynamic MXFP4 follows runtime TopK semantics: select from the full-row
    # softmax. With normalize_topk_weights=False, gate weights must remain
    # full-row probabilities instead of selected-logit softmax probabilities.
    return default_scaled_route(
        router_logits,
        top_k,
        routed_scaling_factor=routed_scaling_factor,
        normalize_topk_weights=normalize_topk_weights,
        dtype=dtype,
    )
