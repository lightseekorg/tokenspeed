# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


@dataclass
class FusedSetKVBufferArg:
    value: torch.Tensor
    k_buffer: torch.Tensor
    v_buffer: torch.Tensor
    k_scale: Optional[float]
    v_scale: Optional[float]
    cache_loc: torch.Tensor


def apply_rope(
    # embedding inputs
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    # embedding options
    is_neox: bool = True,
    offsets: torch.Tensor | None = None,
    rotary_dim: int | None = None,
    fused_set_kv_buffer_arg: FusedSetKVBufferArg | None = None,
    output_q_rope: torch.Tensor | None = None,
    output_k_rope: torch.Tensor | None = None,
    enable_pdl: bool = False,
    # dispatch options
    solution: str | None = None,
    override: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embedding through the registered embedding.rope kernel.

    Args:
        positions: Token positions. Flattened to [num_tokens] before dispatch.
        query: Query tensor with shape [num_tokens, num_q_heads * head_size].
        key: Key tensor with shape [num_tokens, num_kv_heads * head_size].
        head_size: Per-head hidden dimension.
        cos_sin_cache: Packed RoPE cache with shape [max_position, rotary_dim]
            as concat(cos, sin) along the last dimension.
        is_neox: Whether to use Neox-style half-split rotation. False uses
            GPT-J interleaved-pair rotation.
        offsets: Optional per-token offsets added to positions. Not supported
            by embedding.rope yet.
        rotary_dim: Number of rotated channels per head. Defaults to
            cos_sin_cache.shape[-1]. Must be even and no larger than head_size.
        fused_set_kv_buffer_arg: Optional fused KV-cache write arguments. Both
            CUDA and Triton implementations currently require k_scale and
            v_scale to be None.
        output_q_rope: Optional output buffer for the rotated query. If omitted,
            query is updated in place.
        output_k_rope: Optional output buffer for the rotated key. If omitted,
            key is updated in place.
        enable_pdl: Passed through to kernels that support PDL.
        solution: Optional registered solution to select.
        override: Optional exact kernel-name or solution override.

    Returns:
        (rotated_query, rotated_key). These are output_q_rope / output_k_rope
        when provided, otherwise the input query / key.
    """
    if rotary_dim is None:
        rotary_dim = cos_sin_cache.shape[-1]
    assert offsets is None, "embedding.rope does not support offsets"
    assert rotary_dim % 2 == 0, "embedding.rope requires even rotary_dim"
    assert rotary_dim <= head_size, "embedding.rope requires rotary_dim <= head_size"
    assert (
        cos_sin_cache.shape[-1] == rotary_dim
    ), "embedding.rope requires cos_sin_cache last dim to equal rotary_dim"

    positions = positions.flatten()
    num_tokens = positions.shape[0]
    if num_tokens == 0:
        return (
            output_q_rope if output_q_rope is not None else query,
            output_k_rope if output_k_rope is not None else key,
        )
    num_q_heads = query.shape[-1] // head_size
    num_kv_heads = key.shape[-1] // head_size

    traits = {
        "head_size": head_size,
        "partial_rotary": rotary_dim != head_size,
        "is_neox": is_neox,
        "has_fused_kv": fused_set_kv_buffer_arg is not None,
        "has_q_out": output_q_rope is not None,
        "has_k_out": output_k_rope is not None,
    }
    signature = format_signature(
        query=dense_tensor_format(query.dtype),
        key=dense_tensor_format(key.dtype),
    )
    kernel = select_kernel(
        "embedding",
        "rope",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )

    shape_params = {
        "num_tokens": num_tokens,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_size": head_size,
        "rotary_dim": rotary_dim,
        "has_fused_kv": fused_set_kv_buffer_arg is not None,
        "has_q_out": output_q_rope is not None,
        "has_k_out": output_k_rope is not None,
    }
    ShapeCapture.get().record(
        "embedding",
        "rope",
        kernel.name,
        query.dtype,
        shape_params,
    )

    with kernel_scope(
        "embedding",
        "rope",
        query.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        kernel(
            positions=positions,
            query=query,
            key=key,
            head_size=head_size,
            cos_sin_cache=cos_sin_cache,
            is_neox=is_neox,
            rotary_dim=rotary_dim,
            fused_set_kv_buffer_arg=fused_set_kv_buffer_arg,
            output_q_rope=output_q_rope,
            output_k_rope=output_k_rope,
            enable_pdl=enable_pdl,
        )

    return (
        output_q_rope if output_q_rope is not None else query,
        output_k_rope if output_k_rope is not None else key,
    )


__all__ = ["FusedSetKVBufferArg", "apply_rope"]
