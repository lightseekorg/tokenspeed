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

import logging
from typing import Optional, Set

import torch
from tokenspeed_kernel.ops.moe.expert_location_dispatch import (  # noqa: F401
    ExpertLocationDispatchInfo,
    topk_ids_logical_to_physical,
    transform_select_experts_inputs,
)
from tokenspeed_kernel.selection import (
    SelectionOracle,
    register_oracle,
    select_kernel,
)
from tokenspeed_kernel.signature import (
    ScaleFormat,
    dense_tensor_format,
    format_signature,
    tensor_format,
)

logger = logging.getLogger(__name__)


class _MoEOracle(SelectionOracle):
    """Adjust MoE kernel scores based on runtime traits like num_tokens."""

    def adjust(self, spec, platform, traits):
        if traits and spec.mode == "combine":
            num_tokens = traits.get("num_tokens")
            if num_tokens is not None:
                is_small_batch = num_tokens <= 32
                is_triton = spec.solution == "triton"
                # According to micro benchmark results, torch.compile can get better performance for small token.
                if is_small_batch and not is_triton:
                    return 15  # prefer torch_compile for small M
                if not is_small_batch and is_triton:
                    return 15  # prefer triton for large M
                return 5
        return 10


register_oracle("moe", _MoEOracle())

__all__ = [
    "ExpertLocationDispatchInfo",
    "topk_ids_logical_to_physical",
    "transform_select_experts_inputs",
    "moe_route",
    "moe_dispatch",
    "moe_experts",
    "moe_combine",
    "moe_fused",
]

# ---------------------------------------------------------------------------
# Interface feature constants — used by both registration and API calls
# to ensure only signature-compatible kernels are selected.
# ---------------------------------------------------------------------------

# moe/fused interface features
FUSED_SELF_ROUTING = "self_routing"  # kernel does routing internally (trtllm)
FUSED_PRE_ROUTED = (
    "pre_routed"  # caller provides topk_weights/topk_ids (marlin, cutlass, reference)
)

# Weight format values used by moe_fused(weight_format=...).
WEIGHT_BF16 = "bf16"  # dense bfloat16 weights
WEIGHT_FP8 = "fp8"  # FP8 block-scaled weights
WEIGHT_MXFP4 = "mxfp4"  # MXFP4 block-scaled weights
WEIGHT_NVFP4 = "nvfp4"  # NVFP4 block-scaled weights (CuteDSL)

# moe/route trait values — used via traits={"output_type": ...}
ROUTE_OUTPUT_TOPK = "topk"  # returns (topk_weights, topk_ids)
ROUTE_OUTPUT_RAGGED_METADATA = (
    "ragged_metadata"
    # returns (ragged_metadata, gather_indx, scatter_indx, gate_scal)
)

# moe/experts interface features
EXPERTS_DISPATCH_SORTED = (
    "dispatch_sorted"  # expects sorted_token_ids from dispatch stage
)
EXPERTS_RAGGED_METADATA = (
    "ragged_metadata"  # expects RaggedTensorMetadata + gather/scatter indices
)
EXPERTS_DISPATCH_GEMM = (
    "dispatch_gemm"  # gather/dispatch tokens then GEMM (uses gather_indx)
)
EXPERTS_GEMM_COMBINE = (
    "gemm_combine"  # GEMM then scatter/combine results (uses scatter_indx)
)


_FP8_SCALE = ScaleFormat(
    storage_dtype=torch.float32,
    granularity="block",
    block_shape=(128, 128),
)
_FP8_PER_TENSOR_SCALE = ScaleFormat(
    storage_dtype=torch.float32,
    granularity="tensor",
)
_NVFP4_SCALE = ScaleFormat(
    storage_dtype=torch.float32,
    granularity="block",
    block_shape=(16,),
)
_MXFP4_SCALE = ScaleFormat(
    storage_dtype=torch.uint8,
    granularity="block",
    block_shape=(32,),
)


def _single_dense_tensor_format_signature(role: str, storage_dtype: torch.dtype):
    return format_signature(**{role: dense_tensor_format(storage_dtype)})


def _moe_dispatch_format_signature(storage_dtype: torch.dtype, traits: Optional[dict]):
    comm_strategy = (traits or {}).get("comm_strategy")
    if storage_dtype == torch.int32 or comm_strategy == "local":
        return _single_dense_tensor_format_signature("indices", storage_dtype)
    return _single_dense_tensor_format_signature("x", storage_dtype)


def _moe_fused_format_signature(
    storage_dtype: torch.dtype,
    weight_format: str,
    *,
    fp8_scale_granularity: str = "block",
):
    if weight_format == WEIGHT_FP8:
        weight = tensor_format("scaled-fp8", torch.float8_e4m3fn, scale=_FP8_SCALE)
    elif weight_format == WEIGHT_NVFP4:
        weight = tensor_format("nvfp4", torch.uint8, scale=_NVFP4_SCALE)
    elif weight_format == WEIGHT_MXFP4:
        weight = tensor_format("mxfp4", torch.uint8, scale=_MXFP4_SCALE)
    elif weight_format == WEIGHT_BF16:
        weight = dense_tensor_format(torch.bfloat16)
    else:
        raise ValueError(f"Unsupported MoE fused weight_format={weight_format!r}")

    if fp8_scale_granularity == "tensor":
        fp8_scale_format = _FP8_PER_TENSOR_SCALE
    elif fp8_scale_granularity == "block":
        fp8_scale_format = _FP8_SCALE
    else:
        raise ValueError(f"Unsupported fp8_scale_granularity={fp8_scale_granularity!r}")

    if storage_dtype == torch.uint8 and weight_format == WEIGHT_NVFP4:
        x = tensor_format("nvfp4", torch.uint8, scale=_NVFP4_SCALE)
    elif storage_dtype == torch.uint8 and weight_format == WEIGHT_MXFP4:
        x = tensor_format("mxfp4", torch.uint8, scale=_MXFP4_SCALE)
    elif storage_dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        x = tensor_format("scaled-fp8", storage_dtype, scale=fp8_scale_format)
    else:
        x = dense_tensor_format(storage_dtype)

    return format_signature(x=x, weight=weight)


def moe_route(
    *args,
    dtype: torch.dtype = torch.bfloat16,
    features: Optional[Set[str]] = None,
    traits: Optional[dict] = None,
    expected_kernel_name: Optional[str] = None,
    **kwargs,
):
    """Top-k expert routing.

    Routing traits (pass via ``traits``):

    * ``{"output_type": "topk"}``: returns (topk_weights, topk_ids).
    * ``{"output_type": "ragged_metadata"}``: returns
      (ragged_metadata, gather_indx, scatter_indx, gate_scal).
    * ``{"biased": True/False}``: whether correction_bias is applied.
    * ``{"grouped": True/False}``: whether grouped expert selection is used.
    """
    signature = _single_dense_tensor_format_signature("logits", dtype)
    kernel = select_kernel(
        "moe",
        "route",
        signature,
        features=frozenset(features) if features else None,
        traits=traits or {},
        expected_kernel_name=expected_kernel_name,
    )

    return kernel(*args, **kwargs)


def moe_dispatch(
    *args,
    dtype: torch.dtype = torch.int32,
    traits: Optional[dict] = None,
    expected_kernel_name: Optional[str] = None,
    **kwargs,
):
    """Dispatch tokens to experts (local permutation).

    Returns ``(sorted_token_ids, expert_ids, num_tokens_post_padded)``.
    """
    selection_traits = traits or {"comm_strategy": "local"}
    signature = _moe_dispatch_format_signature(dtype, selection_traits)
    kernel = select_kernel(
        "moe",
        "dispatch",
        signature,
        traits=selection_traits,
        expected_kernel_name=expected_kernel_name,
    )

    return kernel(*args, **kwargs)


def moe_experts(
    *args,
    dtype: torch.dtype = torch.bfloat16,
    features: Optional[Set[str]] = None,
    weight_format: Optional[str] = None,
    fp8_scale_granularity: str = "block",
    traits: Optional[dict] = None,
    expected_kernel_name: Optional[str] = None,
    **kwargs,
):
    """Expert FFN computation via MoE GEMM kernel.

    Interface features (pass via ``features``):

    * ``{"dispatch_sorted"}``: triton backend.
      Expects sorted_token_ids, expert_ids, num_tokens_post_padded from
      the dispatch stage.
    * ``{"ragged_metadata"}``: triton_kernels backend.
      Expects ``a_ragged_metadata`` (a ``RaggedTensorMetadata``) plus
      ``gather_indx`` / ``scatter_indx``.
    * ``{"dispatch_gemm"}``: gather/dispatch tokens then GEMM (uses gather_indx).
    * ``{"gemm_combine"}``: GEMM then scatter/combine results (uses scatter_indx).
    """
    if weight_format is None:
        signature = _single_dense_tensor_format_signature("x", dtype)
    else:
        signature = _moe_fused_format_signature(
            dtype, weight_format, fp8_scale_granularity=fp8_scale_granularity
        )
    kernel = select_kernel(
        "moe",
        "experts",
        signature,
        features=frozenset(features) if features else None,
        traits=traits or {},
        expected_kernel_name=expected_kernel_name,
    )

    return kernel(*args, **kwargs)


def moe_combine(
    *args,
    dtype: torch.dtype = torch.bfloat16,
    traits: Optional[dict] = None,
    expected_kernel_name: Optional[str] = None,
    **kwargs,
):
    """Combine expert outputs with weighted reduction."""
    signature = _single_dense_tensor_format_signature("x", dtype)
    kernel = select_kernel(
        "moe",
        "combine",
        signature,
        traits=traits or {},
        expected_kernel_name=expected_kernel_name,
    )

    return kernel(*args, **kwargs)


def moe_fused(
    *args,
    dtype: torch.dtype = torch.bfloat16,
    features: Optional[Set[str]] = None,
    weight_format: str = WEIGHT_BF16,
    fp8_scale_granularity: str = "block",
    traits: Optional[dict] = None,
    expected_kernel_name: Optional[str] = None,
    **kwargs,
) -> torch.Tensor:
    """End-to-end fused MoE: route + permute + experts + combine.

    Interface features (pass via ``features``):

    * ``{"self_routing"}``: kernel does routing internally (trtllm).
    * ``{"pre_routed"}``: routing already done by caller (cutlass, reference).

    Args:
        weight_format: Weight tensor encoding used for the expert weights.
            Supported values are:

            * ``"bf16"``: dense bfloat16 weights with no scale tensor.
            * ``"fp8"``: FP8 E4M3 weights with float32 block scales.
              Fused MoE kernels currently register this as a fixed
              ``block_shape=(128, 128)`` format.
            * ``"mxfp4"``: packed MXFP4 weights stored as uint8 with uint8
              block scales over 32-value blocks.
            * ``"nvfp4"``: packed NVFP4 weights stored as uint8 with float32
              block scales over 16-value blocks.

            The activation/input format is selected by ``dtype``. When
            ``dtype=torch.uint8``, ``weight_format`` disambiguates whether the
            input is interpreted as MXFP4 or NVFP4.
    """
    signature = _moe_fused_format_signature(
        dtype, weight_format, fp8_scale_granularity=fp8_scale_granularity
    )
    selection_traits = dict(traits or {})
    kernel = select_kernel(
        "moe",
        "fused",
        signature,
        features=frozenset(features) if features else None,
        traits=selection_traits,
        expected_kernel_name=expected_kernel_name,
    )

    return kernel(*args, **kwargs)
