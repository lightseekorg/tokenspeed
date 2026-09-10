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

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import torch
from tokenspeed_kernel.platform import current_platform, pdl_enabled
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.registry import KernelRegistry, Priority
from tokenspeed_kernel.selection import (
    NoKernelFoundError,
    select_kernel,
    spec_matches_traits,
)
from tokenspeed_kernel.signature import (
    MXFP8_BLOCK_SCALE,
    dense_tensor_format,
    format_signature,
    tensor_format,
)

AttentionResult = torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]


# One UE8M0 scale per 32 consecutive head_dim elements (MXFP8).
MXFP8_ATTENTION_BLOCK_SCALE = MXFP8_BLOCK_SCALE


def _attention_format_signature(**roles: torch.Tensor):
    return format_signature(
        **{role: dense_tensor_format(tensor.dtype) for role, tensor in roles.items()}
    )


def _mxfp8_attention_format_signature(**roles: torch.Tensor):
    return format_signature(
        **{
            role: tensor_format(
                "mxfp8", tensor.dtype, scale=MXFP8_ATTENTION_BLOCK_SCALE
            )
            for role, tensor in roles.items()
        }
    )


def _blockscaled_signature_and_scales(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    q_scale: torch.Tensor | None,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
):
    """Pick dense vs MXFP8 signature and build the scale kwargs splat.

    q_scale selects the block-scaled path; k_scale/v_scale must accompany it.
    Returns (signature, scale_kwargs) for the paged-KV-cache entry points.
    """
    if q_scale is not None:
        assert (
            k_scale is not None and v_scale is not None
        ), "MXFP8 attention requires q_scale, k_scale, and v_scale together"
        signature = _mxfp8_attention_format_signature(
            q=q, k_cache=k_cache, v_cache=v_cache
        )
    else:
        signature = _attention_format_signature(q=q, k_cache=k_cache, v_cache=v_cache)
    return signature, dict(q_scale=q_scale, k_scale=k_scale, v_scale=v_scale)


LSE_LN = math.log2(math.e)


# ===-----------------------------------------------------------------------===#
# Attention Utilities
# ===-----------------------------------------------------------------------===#


def attn_merge_state(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
    *,
    lse_scale_log2: float = LSE_LN,
    inplace: bool = False,
    override: str | None = None,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge two partial attention states.

    Args:
        out_a: First partial output with shape [total_q, num_heads, head_dim].
        lse_a: First partial log-sum-exp with shape [total_q, num_heads].
        out_b: Second partial output with shape [total_q, num_heads, head_dim].
        lse_b: Second partial log-sum-exp with shape [total_q, num_heads].
        lse_scale_log2: Multiplier that converts input LSE to log2 domain.
        inplace: Whether to write the merged state back into ``out_a``/``lse_a``.
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.

    This is shared by MHA and MLA because the merge only depends on partial
    attention outputs and LSE values, not on how the K/V states were produced.
    """
    traits = {
        "head_dim": out_a.shape[-1],
    }
    signature = _attention_format_signature(out_a=out_a, out_b=out_b)
    kernel = select_kernel(
        "attention",
        "attn_merge_state",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )

    shape_params = {
        "total_q": out_a.shape[0],
        "num_heads": out_a.shape[1],
        "head_dim": out_a.shape[2],
    }
    ShapeCapture.get().record(
        "attention",
        "attn_merge_state",
        kernel.name,
        out_a.dtype,
        shape_params,
    )

    with kernel_scope(
        "attention",
        "attn_merge_state",
        out_a.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            out_a=out_a,
            lse_a=lse_a,
            out_b=out_b,
            lse_b=lse_b,
            lse_scale_log2=lse_scale_log2,
            inplace=inplace,
            enable_pdl=pdl_enabled(),
        )


# Backend registration (side-effect imports)
import tokenspeed_kernel.ops.attention.merge_state_cuda  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.merge_state_triton  # noqa: E402,F401

__all__ = ["attn_merge_state"]
