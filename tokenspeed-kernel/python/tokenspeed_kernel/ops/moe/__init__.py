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

# Backend registration (side-effect imports)
import tokenspeed_kernel.numerics.reference.moe  # noqa: F401
import tokenspeed_kernel.ops.moe.cuda  # noqa: F401
import tokenspeed_kernel.ops.moe.deepep  # noqa: F401
import tokenspeed_kernel.ops.moe.flashinfer  # noqa: F401
import tokenspeed_kernel.ops.moe.reference  # noqa: F401
import tokenspeed_kernel.ops.moe.triton  # noqa: F401
import tokenspeed_kernel.ops.moe.triton_kernels  # noqa: F401
import tokenspeed_kernel.ops.moe.trtllm  # noqa: F401
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

# Weight format trait values — used via traits={"weight_dtype": ...}
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
    kernel = select_kernel(
        "moe",
        "route",
        dtype,
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
    kernel = select_kernel(
        "moe",
        "dispatch",
        dtype,
        traits=traits or {"comm_strategy": "local"},
        expected_kernel_name=expected_kernel_name,
    )

    return kernel(*args, **kwargs)


def moe_experts(
    *args,
    dtype: torch.dtype = torch.bfloat16,
    features: Optional[Set[str]] = None,
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
    kernel = select_kernel(
        "moe",
        "experts",
        dtype,
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
    kernel = select_kernel(
        "moe",
        "combine",
        dtype,
        traits=traits or {},
        expected_kernel_name=expected_kernel_name,
    )

    return kernel(*args, **kwargs)


def moe_fused(
    *args,
    dtype: torch.dtype = torch.bfloat16,
    features: Optional[Set[str]] = None,
    traits: Optional[dict] = None,
    expected_kernel_name: Optional[str] = None,
    **kwargs,
) -> torch.Tensor:
    """End-to-end fused MoE: route + permute + experts + combine.

    Interface features (pass via ``features``):

    * ``{"self_routing"}``: kernel does routing internally (trtllm).
    * ``{"pre_routed"}``: routing already done by caller (cutlass, reference).

    Weight format traits (pass via ``traits``):

    * ``{"weight_dtype": "bf16"}``: dense bfloat16 weights.
    * ``{"weight_dtype": "fp8"}``: FP8 block-scaled weights.
    * ``{"weight_dtype": "mxfp4"}``: MXFP4 block-scaled weights.
    """
    kernel = select_kernel(
        "moe",
        "fused",
        dtype,
        features=frozenset(features) if features else None,
        traits=traits or {},
        expected_kernel_name=expected_kernel_name,
    )

    return kernel(*args, **kwargs)
