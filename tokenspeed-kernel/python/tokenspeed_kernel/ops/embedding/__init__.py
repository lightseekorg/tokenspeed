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

import torch
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


@dataclass
class FusedSetKVBufferArg:
    value: torch.Tensor
    k_buffer: torch.Tensor
    v_buffer: torch.Tensor
    k_scale: float | None
    v_scale: float | None
    cache_loc: torch.Tensor


@dataclass
class FusedMLASetKVBufferArg:
    k_nope: torch.Tensor
    kv_buffer: torch.Tensor
    cache_loc: torch.Tensor


def apply_rope(
    # embedding inputs
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    # embedding options
    is_neox: bool = True,
    fused_set_kv_buffer_arg: FusedSetKVBufferArg | None = None,
    fused_mla_set_kv_buffer_arg: FusedMLASetKVBufferArg | None = None,
    q_rope_out: torch.Tensor | None = None,
    k_rope_out: torch.Tensor | None = None,
    # dispatch options
    enable_pdl: bool = False,
    solution: str | None = None,
    override: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embedding through the registered embedding.rope kernel.

    Args:
        positions: Token positions. Flattened to [num_tokens] before dispatch.
        q: Query tensor with shape [num_tokens, num_q_heads * head_size].
        k: Key tensor with shape [num_tokens, num_kv_heads * head_size].
        head_size: Per-head hidden dimension.
        cos_sin_cache: Packed RoPE cache with shape [max_position, rotary_dim]
            as concat(cos, sin) along the last dimension.
        is_neox: Whether to use Neox-style half-split rotation. False uses
            GPT-J interleaved-pair rotation.
        fused_set_kv_buffer_arg: Optional fused KV-cache write arguments. Both
            CUDA and Triton implementations currently require k_scale and
            v_scale to be None.
        fused_mla_set_kv_buffer_arg: Optional fused MLA KV-cache write
            arguments. When provided, rotated K is written to the MLA cache
            instead of k/k_rope_out because MLA stores [k_nope | k_rope] in a
            single cache row with different component widths.
        q_rope_out: Optional output buffer for the rotated query. If omitted,
            q is updated in place.
        k_rope_out: Optional output buffer for the rotated key. If omitted,
            k is updated in place.
        enable_pdl: Passed through to kernels that support PDL.
        solution: Optional registered solution to select.
        override: Optional exact kernel-name or solution override.

    Returns:
        (rotated_q, rotated_k). These are q_rope_out / k_rope_out when provided,
        otherwise the input q / k.
    """
    rotary_dim = cos_sin_cache.shape[-1]
    assert rotary_dim % 2 == 0, "embedding.rope requires even rotary_dim"
    assert rotary_dim <= head_size, "embedding.rope requires rotary_dim <= head_size"
    if fused_set_kv_buffer_arg is not None and fused_mla_set_kv_buffer_arg is not None:
        raise ValueError("standard and MLA fused KV writes are mutually exclusive")
    if fused_mla_set_kv_buffer_arg is not None and k_rope_out is not None:
        raise ValueError("MLA fused KV write stores rotated K directly in the cache")

    positions = positions.flatten()
    num_tokens = positions.shape[0]
    if num_tokens == 0:
        return (
            q_rope_out if q_rope_out is not None else q,
            k_rope_out if k_rope_out is not None else k,
        )
    num_q_heads = q.numel() // (num_tokens * head_size)
    num_kv_heads = k.numel() // (num_tokens * head_size)

    traits = {
        "head_size": head_size,
        "partial_rotary": rotary_dim != head_size,
        "is_neox": is_neox,
        "has_fused_kv": fused_set_kv_buffer_arg is not None,
        "has_fused_mla_kv": fused_mla_set_kv_buffer_arg is not None,
        "has_q_out": q_rope_out is not None,
        "has_k_out": k_rope_out is not None,
    }
    signature = format_signature(
        q=dense_tensor_format(q.dtype),
        k=dense_tensor_format(k.dtype),
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
        "has_fused_mla_kv": fused_mla_set_kv_buffer_arg is not None,
        "has_q_out": q_rope_out is not None,
        "has_k_out": k_rope_out is not None,
    }
    ShapeCapture.get().record(
        "embedding",
        "rope",
        kernel.name,
        q.dtype,
        shape_params,
    )

    with kernel_scope(
        "embedding",
        "rope",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        kernel(
            positions=positions,
            q=q,
            k=k,
            head_size=head_size,
            cos_sin_cache=cos_sin_cache,
            is_neox=is_neox,
            fused_set_kv_buffer_arg=fused_set_kv_buffer_arg,
            fused_mla_set_kv_buffer_arg=fused_mla_set_kv_buffer_arg,
            q_rope_out=q_rope_out,
            k_rope_out=k_rope_out,
            enable_pdl=enable_pdl,
        )

    return (
        q_rope_out if q_rope_out is not None else q,
        k_rope_out if k_rope_out is not None else k,
    )


def apply_rope_mla(
    # embedding inputs
    positions: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    q_nope: torch.Tensor,
    k_nope: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    # embedding options
    is_neox: bool = True,
    quantize_dtype: torch.dtype = torch.float8_e4m3fn,
    quant_scale_q: float | torch.Tensor = 1.0,
    quant_scale_kv: float | torch.Tensor = 1.0,
    q_rope_out: torch.Tensor | None = None,
    k_rope_out: torch.Tensor | None = None,
    q_nope_out: torch.Tensor | None = None,
    k_nope_out: torch.Tensor | None = None,
    enable_pdl: bool = False,
    # dispatch options
    solution: str | None = None,
    override: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MLA RoPE and quantize query/key parts to FP8.

    Args:
        positions: Token positions. Flattened to [tokens] before dispatch.
        q_rope: Query RoPE slice with shape [tokens, q_heads, rope_dim].
        k_rope: Key RoPE slice with shape [tokens, kv_heads, rope_dim].
        q_nope: Query non-RoPE slice with shape [tokens, q_heads, nope_dim].
        k_nope: Key non-RoPE slice with shape [tokens, kv_heads, nope_dim].
        cos_sin_cache: Packed RoPE cache as concat(cos, sin) on the last dim.
        is_neox: Whether to use Neox-style half-split rotation.
        quantize_dtype: Output FP8 dtype. Currently only e4m3fn is supported.
        quant_scale_q: Quantization scale multiplied into query tensors before
            the FP8 cast.
        quant_scale_kv: Quantization scale multiplied into key tensors before
            the FP8 cast.
        q_rope_out: Optional FP8 output buffer for rotated q_rope. If omitted
            together with q_nope_out, a combined query output is allocated and
            this slice is derived from it.
        k_rope_out: Optional FP8 output buffer for rotated k_rope. If omitted
            together with k_nope_out, a combined key output is allocated and this
            slice is derived from it.
        q_nope_out: Optional FP8 output buffer for q_nope. If omitted together
            with q_rope_out, a combined query output is allocated and this slice
            is derived from it.
        k_nope_out: Optional FP8 output buffer for k_nope. If omitted together
            with k_rope_out, a combined key output is allocated and this slice is
            derived from it.
        enable_pdl: Passed through to kernels that support PDL.
        solution: Optional registered solution to select.
        override: Optional exact kernel-name or solution override.

    Returns:
        (query_fp8, key_fp8), where the last dimension is concat(nope, rope).
    """
    positions = positions.flatten()
    query_fp8 = None
    key_fp8 = None

    if q_rope_out is None and q_nope_out is None:
        query_fp8 = torch.empty(
            q_nope.shape[:-1] + (q_nope.shape[-1] + q_rope.shape[-1],),
            dtype=quantize_dtype,
            device=q_nope.device,
        )
        q_nope_out = query_fp8[..., : q_nope.shape[-1]]
        q_rope_out = query_fp8[..., q_nope.shape[-1] :]
    else:
        q_rope_out = (
            torch.empty(q_rope.shape, dtype=quantize_dtype, device=q_rope.device)
            if q_rope_out is None
            else q_rope_out
        )
        q_nope_out = (
            torch.empty(q_nope.shape, dtype=quantize_dtype, device=q_nope.device)
            if q_nope_out is None
            else q_nope_out
        )

    if k_rope_out is None and k_nope_out is None:
        key_fp8 = torch.empty(
            k_nope.shape[:-1] + (k_nope.shape[-1] + k_rope.shape[-1],),
            dtype=quantize_dtype,
            device=k_nope.device,
        )
        k_nope_out = key_fp8[..., : k_nope.shape[-1]]
        k_rope_out = key_fp8[..., k_nope.shape[-1] :]
    else:
        k_rope_out = (
            torch.empty(k_rope.shape, dtype=quantize_dtype, device=k_rope.device)
            if k_rope_out is None
            else k_rope_out
        )
        k_nope_out = (
            torch.empty(k_nope.shape, dtype=quantize_dtype, device=k_nope.device)
            if k_nope_out is None
            else k_nope_out
        )

    num_tokens = q_rope.shape[0]
    if num_tokens == 0:
        return (
            (
                query_fp8
                if query_fp8 is not None
                else torch.cat((q_nope_out, q_rope_out), dim=-1)
            ),
            (
                key_fp8
                if key_fp8 is not None
                else torch.cat((k_nope_out, k_rope_out), dim=-1)
            ),
        )

    traits = {
        "is_neox": bool(is_neox),
        "quantize_dtype": quantize_dtype,
        "has_scale_q_tensor": isinstance(quant_scale_q, torch.Tensor),
        "has_scale_kv_tensor": isinstance(quant_scale_kv, torch.Tensor),
    }
    signature = format_signature(
        q_rope=dense_tensor_format(q_rope.dtype),
        k_rope=dense_tensor_format(k_rope.dtype),
        q_nope=dense_tensor_format(q_nope.dtype),
        k_nope=dense_tensor_format(k_nope.dtype),
    )
    kernel = select_kernel(
        "embedding",
        "rope_mla",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )

    shape_params = {
        "num_tokens": num_tokens,
        "q_heads": q_rope.shape[1],
        "kv_heads": k_rope.shape[1],
        "q_nope_dim": q_nope.shape[-1],
        "k_nope_dim": k_nope.shape[-1],
        "rope_dim": q_rope.shape[-1],
        "is_neox": bool(is_neox),
    }
    ShapeCapture.get().record(
        "embedding",
        "rope_mla",
        kernel.name,
        q_rope.dtype,
        shape_params,
    )
    with kernel_scope(
        "embedding",
        "rope_mla",
        q_rope.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        kernel(
            positions=positions,
            q_rope=q_rope,
            k_rope=k_rope,
            q_nope=q_nope,
            k_nope=k_nope,
            cos_sin_cache=cos_sin_cache,
            q_rope_out=q_rope_out,
            k_rope_out=k_rope_out,
            q_nope_out=q_nope_out,
            k_nope_out=k_nope_out,
            is_neox=is_neox,
            quant_scale_q=quant_scale_q,
            quant_scale_kv=quant_scale_kv,
            enable_pdl=enable_pdl,
        )

    query_fp8 = (
        query_fp8
        if query_fp8 is not None
        else torch.cat((q_nope_out, q_rope_out), dim=-1)
    )
    key_fp8 = (
        key_fp8 if key_fp8 is not None else torch.cat((k_nope_out, k_rope_out), dim=-1)
    )
    return query_fp8, key_fp8


__all__ = [
    "FusedMLASetKVBufferArg",
    "FusedSetKVBufferArg",
    "apply_rope",
    "apply_rope_mla",
]


# Backend registration (side-effect imports).
import tokenspeed_kernel.ops.embedding.cuda  # noqa: E402,F401
import tokenspeed_kernel.ops.embedding.flashinfer  # noqa: E402,F401
import tokenspeed_kernel.ops.embedding.triton  # noqa: E402,F401
