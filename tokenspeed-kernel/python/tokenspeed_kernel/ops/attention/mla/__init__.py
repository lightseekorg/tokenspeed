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
# MLA Kernels
# ===-----------------------------------------------------------------------===#


def mla_project_value_prefers_contiguous_weight(
    *,
    dtype: torch.dtype,
    heads: int,
    latent_dim: int,
    value_dim: int,
    gated: bool = False,
    batch_size: int = 1,
) -> bool:
    """Whether the selected kernel wants a contiguous weight."""
    signature = format_signature(
        attention=dense_tensor_format(dtype),
        weight=dense_tensor_format(dtype),
        out=dense_tensor_format(dtype),
    )
    traits = {
        "batch_size": batch_size,
        "num_heads": heads,
        "latent_dim": latent_dim,
        "value_dim": value_dim,
        "gate_kind": "sigmoid" if gated else "none",
        "inputs_contiguous": True,
    }
    try:
        select_kernel("attention", "mla_project_value", signature, traits=traits)
    except NoKernelFoundError:
        return False
    return True


def mla_project_value(
    attention: torch.Tensor,
    weight: torch.Tensor,
    *,
    gate: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> torch.Tensor:
    """Project per-head MLA values and optionally apply a sigmoid gate.

    The headwise projection accumulates in FP32 and is materialized in the
    input dtype before the optional gate, preserving the unfused numerical
    boundary.

    Args:
        attention: Absorbed values shaped ``[batch, heads, latent_dim]``.
        weight: Per-head projection shaped ``[heads, latent_dim, value_dim]``.
        gate: Optional raw sigmoid gate shaped ``[batch, heads * value_dim]``.
        out: Optional output with the same shape as ``gate`` when provided, or
            ``[batch, heads * value_dim]`` otherwise.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.

    Returns:
        Projected values shaped ``[batch, heads * value_dim]``.
    """
    if attention.ndim != 3 or attention.shape[0] < 1:
        raise ValueError("attention must have shape [batch, heads, latent_dim]")
    if weight.ndim != 3 or weight.shape[:2] != attention.shape[1:]:
        raise ValueError("weight must have shape [heads, latent_dim, value_dim]")
    if attention.dtype != weight.dtype or attention.device != weight.device:
        raise ValueError("attention and weight must match dtype and device")

    batch, heads, latent_dim = attention.shape
    value_dim = weight.shape[2]
    expected_output = (batch, heads * value_dim)
    if gate is not None and (
        tuple(gate.shape) != expected_output
        or gate.dtype != attention.dtype
        or gate.device != attention.device
    ):
        raise ValueError(f"gate must match attention and have shape {expected_output}")
    if out is None:
        out = attention.new_empty(expected_output)
    elif (
        tuple(out.shape) != expected_output
        or out.dtype != attention.dtype
        or out.device != attention.device
        or not out.is_contiguous()
    ):
        raise ValueError(f"out must be contiguous and have shape {expected_output}")

    signature = _attention_format_signature(
        attention=attention,
        weight=weight,
        out=out,
    )
    traits = {
        "batch_size": batch,
        "num_heads": heads,
        "latent_dim": latent_dim,
        "value_dim": value_dim,
        "gate_kind": "none" if gate is None else "sigmoid",
        "inputs_contiguous": (
            attention.is_contiguous()
            and weight.is_contiguous()
            and (gate is None or gate.stride(-1) == 1)
            and out.is_contiguous()
        ),
    }
    try:
        kernel = select_kernel(
            "attention",
            "mla_project_value",
            signature,
            traits=traits,
            solution=solution,
            override=override,
        )
    except NoKernelFoundError:
        if override is not None or solution is not None:
            raise
        kernel = None

    if kernel is None and not traits["inputs_contiguous"]:
        try:
            candidate = select_kernel(
                "attention",
                "mla_project_value",
                signature,
                traits={**traits, "inputs_contiguous": True},
            )
        except NoKernelFoundError:
            candidate = None
        if candidate is not None:
            attention = attention.contiguous()
            weight = weight.contiguous()
            gate = None if gate is None else gate.contiguous()
            kernel = candidate

    if kernel is not None:
        shape_params = {
            "batch_size": batch,
            "num_heads": heads,
            "latent_dim": latent_dim,
            "value_dim": value_dim,
            "gate_kind": traits["gate_kind"],
        }
        ShapeCapture.get().record(
            "attention",
            "mla_project_value",
            kernel.name,
            attention.dtype,
            shape_params,
        )
        with kernel_scope(
            "attention",
            "mla_project_value",
            attention.dtype,
            kernel_name=kernel.name,
            **shape_params,
        ):
            return kernel(attention=attention, weight=weight, gate=gate, out=out)

    output_view = out.view(batch, heads, value_dim)
    if current_platform().is_nvidia:
        torch.bmm(
            attention.transpose(0, 1),
            weight,
            out=output_view.transpose(0, 1),
        )
    else:
        projected = torch.bmm(attention.transpose(0, 1).contiguous(), weight)
        output_view.copy_(projected.transpose(0, 1))
    if gate is not None:
        if out.is_cuda:
            from tokenspeed_kernel.ops.activation.triton import sigmoid_mul

            sigmoid_mul(out, gate)
        else:
            out.copy_(out.float() * torch.sigmoid(gate.float()))
    return out


def mla_normalize_project_query(
    query: torch.Tensor,
    kv: torch.Tensor,
    query_norm_weight: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    projection_weight: torch.Tensor,
    *,
    eps: float,
    prepare_absorbed_query: bool = False,
    qk_nope_head_dim: int | None = None,
    qk_rope_head_dim: int | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Normalize MLA query/KV latents and project the normalized query.

    The query normalization is materialized in its input dtype before the
    projection. ``kv`` is normalized in place so callers can retain a view
    into a larger latent-cache tensor.

    Args:
        query: Query latent shaped ``[tokens, query_width]``.
        kv: KV latent shaped ``[tokens, kv_width]``; modified in place.
        query_norm_weight: Query RMSNorm weight shaped ``[query_width]``.
        kv_norm_weight: KV RMSNorm weight shaped ``[kv_width]``.
        projection_weight: Query projection weight shaped
            ``[output_width, query_width]``.
        eps: Positive RMSNorm epsilon.
        prepare_absorbed_query: Whether to prepare the per-head query layout
            consumed by MLA absorb decode when a compatible kernel is available.
        qk_nope_head_dim: Per-head NoPE width required when preparing an
            absorbed query.
        qk_rope_head_dim: Per-head RoPE width required when preparing an
            absorbed query.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.

    Returns:
        A ``(query, absorbed_query)`` pair. ``query`` normally has shape
        ``[tokens, output_width]`` and ``absorbed_query`` is ``None``. When an
        absorbed destination is prepared, ``query`` instead has shape
        ``[tokens, heads, qk_nope_head_dim]`` and ``absorbed_query`` has shape
        ``[tokens, heads, kv_width + qk_rope_head_dim]``; only its RoPE tail is
        populated because the absorb BMM owns the latent prefix.
    """
    if query.ndim != 2 or query.shape[0] < 1:
        raise ValueError("query must have shape [tokens, query_width]")
    tokens, query_width = query.shape
    if kv.ndim != 2 or kv.shape[0] != tokens:
        raise ValueError("kv must have shape [tokens, kv_width]")
    kv_width = kv.shape[1]
    output_width = projection_weight.shape[0] if projection_weight.ndim == 2 else 0
    expected = (
        (query_norm_weight, (query_width,), "query_norm_weight"),
        (kv_norm_weight, (kv_width,), "kv_norm_weight"),
        (projection_weight, (output_width, query_width), "projection_weight"),
    )
    for tensor, shape, name in expected:
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
    for tensor, name in (
        (kv, "kv"),
        (query_norm_weight, "query_norm_weight"),
        (kv_norm_weight, "kv_norm_weight"),
        (projection_weight, "projection_weight"),
    ):
        if tensor.dtype != query.dtype or tensor.device != query.device:
            raise ValueError(f"{name} must match query dtype and device")
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name} must have unit inner stride")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    num_heads = None
    if prepare_absorbed_query:
        if qk_nope_head_dim is None or qk_nope_head_dim <= 0:
            raise ValueError(
                "qk_nope_head_dim must be positive when preparing an absorbed query"
            )
        if qk_rope_head_dim is None or qk_rope_head_dim <= 0:
            raise ValueError(
                "qk_rope_head_dim must be positive when preparing an absorbed query"
            )
        head_width = qk_nope_head_dim + qk_rope_head_dim
        if output_width % head_width != 0:
            raise ValueError(
                f"projection width {output_width} is not divisible by head width {head_width}"
            )
        num_heads = output_width // head_width

    def select_for_layout(
        *, prefix_width: int, tail_width: int, required: bool = False
    ):
        tensor_roles = {
            "query": dense_tensor_format(query.dtype),
            "kv": dense_tensor_format(kv.dtype),
            "projection_weight": dense_tensor_format(projection_weight.dtype),
            "out": dense_tensor_format(query.dtype),
        }
        split_output = tail_width > 0
        if split_output:
            tensor_roles["tail_out"] = dense_tensor_format(query.dtype)
        signature = format_signature(**tensor_roles)
        traits = {
            "num_tokens": tokens,
            "query_width": query_width,
            "kv_width": kv_width,
            "output_width": output_width,
            "output_prefix_width": prefix_width,
            "output_tail_width": tail_width,
            "split_output": split_output,
            "inputs_contiguous": all(
                tensor.is_contiguous()
                for tensor in (
                    query,
                    kv,
                    query_norm_weight,
                    kv_norm_weight,
                    projection_weight,
                )
            ),
            "outputs_inner_contiguous": True,
        }
        try:
            return select_kernel(
                "attention",
                "mla_normalize_project_query",
                signature,
                traits=traits,
                solution=solution,
                override=override,
            )
        except NoKernelFoundError:
            if required:
                raise
            return None

    kernel = None
    split_selected = False
    if num_heads is not None:
        assert qk_nope_head_dim is not None and qk_rope_head_dim is not None
        kernel = select_for_layout(
            prefix_width=qk_nope_head_dim,
            tail_width=qk_rope_head_dim,
        )
        split_selected = kernel is not None
    if kernel is None:
        kernel = select_for_layout(
            prefix_width=output_width,
            tail_width=0,
            required=override is not None or solution is not None,
        )

    absorbed_query = None
    tail_out = None
    if split_selected:
        assert num_heads is not None
        assert qk_nope_head_dim is not None and qk_rope_head_dim is not None
        out = query.new_empty(tokens, num_heads, qk_nope_head_dim)
        absorbed_query = query.new_empty(tokens, num_heads, kv_width + qk_rope_head_dim)
        tail_out = absorbed_query[..., kv_width:]
    else:
        out = query.new_empty((tokens, output_width))

    if kernel is not None:
        shape_params = {
            "num_tokens": tokens,
            "query_width": query_width,
            "kv_width": kv_width,
            "output_width": output_width,
        }
        ShapeCapture.get().record(
            "attention",
            "mla_normalize_project_query",
            kernel.name,
            query.dtype,
            shape_params,
        )
        with kernel_scope(
            "attention",
            "mla_normalize_project_query",
            query.dtype,
            kernel_name=kernel.name,
            **shape_params,
        ):
            output = kernel(
                query=query,
                kv=kv,
                query_norm_weight=query_norm_weight,
                kv_norm_weight=kv_norm_weight,
                projection_weight=projection_weight,
                eps=eps,
                out=out,
                tail_out=tail_out,
            )
            return output, absorbed_query

    projection_out = out
    if query.is_cuda and query.dtype == torch.bfloat16:
        if current_platform().is_amd:
            from tokenspeed_kernel.ops.layernorm.triton import (
                rmsnorm_fused_parallel,
            )
        else:
            from tokenspeed_kernel.ops.layernorm.cuda import rmsnorm_fused_parallel

        query_norm = torch.empty_like(query)
        rmsnorm_fused_parallel(
            input1=query,
            weight1=query_norm_weight,
            output1=query_norm,
            input2=kv,
            weight2=kv_norm_weight,
            output2=kv,
            eps=eps,
        )
        if current_platform().is_cdna5 and tokens > 1:
            from tokenspeed_kernel.ops.gemm.kimi3 import _try_gluon_largem_gfx1250

            if (
                _try_gluon_largem_gfx1250(
                    query_norm,
                    projection_weight,
                    out=projection_out,
                )
                is None
            ):
                from tokenspeed_kernel.ops.gemm import mm

                mm(query_norm, projection_weight, out=projection_out)
        else:
            from tokenspeed_kernel.ops.gemm.triton_gemv import decode_gemv

            decode_gemv(query_norm, projection_weight, out=projection_out)
    else:
        query_fp32 = query.float()
        query_norm = query_fp32 * torch.rsqrt(
            query_fp32.square().mean(dim=-1, keepdim=True) + eps
        )
        query_norm = (query_norm * query_norm_weight.float()).to(query.dtype)
        kv_fp32 = kv.float()
        kv_norm = kv_fp32 * torch.rsqrt(
            kv_fp32.square().mean(dim=-1, keepdim=True) + eps
        )
        kv.copy_((kv_norm * kv_norm_weight.float()).to(kv.dtype))
        torch.mm(query_norm, projection_weight.t(), out=projection_out)
    return out, None


def mla_prefill(
    # attention inputs
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    softmax_scale: float,
    # attention options
    seq_lens_kv: torch.Tensor | None = None,
    is_causal: bool = True,
    logit_cap: float = 0.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
    # dispatch options
    override: str | None = None,
    solution: str | None = None,
) -> AttentionResult:
    """MLA prefill/cross-attention from explicit, non-cached Q/K/V tensors.

    This API is for the non-absorbed MLA path. Callers materialize full
    per-head K/V before calling this function, so the kernel contract is close
    to MHA ragged attention. It is used for both prompt/new-token causal
    prefill and prefix-cache replay chunks after the compressed MLA cache has
    been read and expanded by the model.

    Args:
        q: Query tensor with shape [total_q, num_q_heads, qk_head_dim], where
            qk_head_dim = qk_nope_head_dim + qk_rope_head_dim.
        k: Key tensor with shape [total_kv, num_kv_heads, qk_head_dim]. For
            DeepSeek MLA prefill today, num_kv_heads is normally num_q_heads
            after expanding the shared RoPE key part across heads.
        v: Value tensor with shape [total_kv, num_kv_heads, v_head_dim].
        cu_seqlens_q: Query cumulative sequence lengths with shape [batch + 1].
        cu_seqlens_kv: KV cumulative sequence lengths with shape [batch + 1].
            This is independent from cu_seqlens_q so prefix-cache chunks can use
            q_lens != kv_lens.
        max_seqlen_q: Maximum query length in the batch.
        max_seqlen_kv: Maximum KV length in the batch.
        softmax_scale: Scale applied to QK logits before softmax.
        seq_lens_kv: Optional per-request KV lengths with shape [batch]. Some
            backends need this in addition to cu_seqlens_kv.
        is_causal: Whether to apply a causal mask between Q and KV. Prefix-cache
            replay chunks should pass False because all prefix tokens precede all
            extend tokens.
        logit_cap: Optional soft cap applied to attention logits.
        return_lse: Whether to also return natural-log log-sum-exp values with
            shape [total_q, num_q_heads]. Required when partial attention states
            will be merged.
        out: Optional output tensor with shape [total_q, num_q_heads, v_head_dim].
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Returns:
        Attention output with shape [total_q, num_q_heads, v_head_dim], or
        (output, lse) when return_lse is True.
    """
    batch_size = cu_seqlens_q.shape[0] - 1
    traits = {
        "qk_head_dim": q.shape[-1],
        "v_head_dim": v.shape[-1],
        "is_causal": is_causal,
        "support_logit_cap": logit_cap != 0.0,
        "return_lse": return_lse,
    }
    signature = _attention_format_signature(q=q, k=k, v=v)
    kernel = select_kernel(
        "attention",
        "mla_prefill",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )

    shape_params = {
        "batch_size": batch_size,
        "total_q": q.shape[0],
        "total_kv": k.shape[0],
        "num_q_heads": q.shape[1],
        "num_kv_heads": k.shape[1],
        "qk_head_dim": q.shape[-1],
        "v_head_dim": v.shape[-1],
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_kv": max_seqlen_kv,
    }
    ShapeCapture.get().record(
        "attention",
        "mla_prefill",
        kernel.name,
        q.dtype,
        shape_params,
    )

    with kernel_scope(
        "attention",
        "mla_prefill",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            softmax_scale=softmax_scale,
            seq_lens_kv=seq_lens_kv,
            is_causal=is_causal,
            logit_cap=logit_cap,
            return_lse=return_lse,
            out=out,
        )


def mla_use_absorbed_extend(
    *,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    num_q_heads: int,
    page_size: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    max_seqlen_q: int | None = None,
    solution: str | None = None,
) -> bool:
    """Return whether a registered kernel supports absorbed MLA extend.

    Args:
        q_dtype: Absorbed query dtype.
        kv_dtype: Compressed KV-cache dtype.
        num_q_heads: Number of local query heads.
        page_size: Number of cache tokens per page.
        qk_nope_head_dim: Original non-RoPE query/key dimension.
        kv_lora_rank: Compressed MLA latent rank.
        qk_rope_head_dim: RoPE query/key dimension.
        max_seqlen_q: Optional maximum query length used to filter kernels whose
            registered shape domain is narrower than their operator API.
        solution: Optional kernel solution to restrict the query.

    Returns:
        Whether the current platform has a matching causal absorbed-extend
        implementation. Kernel registrations remain the source of truth for
        hardware, dtype, and shape support.
    """
    signature = format_signature(
        q=dense_tensor_format(q_dtype),
        kv_cache=dense_tensor_format(kv_dtype),
    )
    traits = {
        "num_q_heads": num_q_heads,
        "page_size": page_size,
        "qk_nope_head_dim": qk_nope_head_dim,
        "kv_lora_rank": kv_lora_rank,
        "qk_rope_head_dim": qk_rope_head_dim,
        "is_causal": True,
        "support_logit_cap": False,
        "return_lse": False,
    }
    if max_seqlen_q is not None:
        traits["max_seqlen_q"] = max_seqlen_q
    candidates = KernelRegistry.get().get_for_operator(
        "attention",
        "mla_extend_with_kvcache",
        platform=current_platform(),
        format_signature=signature,
        solution=solution,
    )
    return any(spec_matches_traits(spec, traits) for spec in candidates)


def mla_extend_with_kvcache(
    # attention inputs
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    # MLA dimensions
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    # attention options
    is_causal: bool = True,
    logit_cap: float = 0.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
    # dispatch options
    override: str | None = None,
    solution: str | None = None,
) -> AttentionResult:
    """MLA multi-token attention over a compressed paged KV cache.

    The model supplies packed absorbed queries and prewrites the current tokens
    to the compressed cache. A zero-length prefix represents initial prefill;
    a nonzero prefix represents cached extend.

    Args:
        q: Packed absorbed query shaped ``[total_q, num_q_heads,
            kv_lora_rank + qk_rope_head_dim]``.
        kv_cache: Compressed paged cache shaped ``[num_pages, page_size, 1,
            kv_lora_rank + qk_rope_head_dim]``.
        page_table: Page table shaped ``[batch, max_pages_per_seq]``.
        cache_seqlens: Total visible KV lengths, including the current query
            tokens, shaped ``[batch]``.
        cu_seqlens_q: Packed query boundaries shaped ``[batch + 1]``.
        cu_seqlens_kv: Packed total-KV boundaries shaped ``[batch + 1]``.
        max_seqlen_q: Maximum query length in the batch.
        max_seqlen_k: Maximum visible KV length in the batch.
        qk_nope_head_dim: Original non-RoPE query/key dimension.
        kv_lora_rank: Compressed MLA latent rank and output head dimension.
        qk_rope_head_dim: RoPE query/key dimension.
        softmax_scale: Scale applied to QK logits.
        is_causal: Whether each query chunk is a causal suffix of its cache.
        logit_cap: Optional soft cap applied to logits.
        return_lse: Whether to return natural-log log-sum-exp values.
        out: Optional output shaped ``[total_q, num_q_heads, kv_lora_rank]``.
        override: Optional exact kernel name.
        solution: Optional kernel solution backend.

    Returns:
        Latent attention output, or ``(output, lse)`` when supported and
        ``return_lse`` is true. The caller applies the MLA value projection.
    """
    batch_size = cache_seqlens.shape[0]
    traits = {
        "page_size": kv_cache.shape[1],
        "num_q_heads": q.shape[1],
        "max_seqlen_q": max_seqlen_q,
        "qk_nope_head_dim": qk_nope_head_dim,
        "kv_lora_rank": kv_lora_rank,
        "qk_rope_head_dim": qk_rope_head_dim,
        "is_causal": is_causal,
        "support_logit_cap": logit_cap != 0.0,
        "return_lse": return_lse,
    }
    signature = _attention_format_signature(q=q, kv_cache=kv_cache)
    kernel = select_kernel(
        "attention",
        "mla_extend_with_kvcache",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )

    shape_params = {
        "batch_size": batch_size,
        "total_q": q.shape[0],
        "num_q_heads": q.shape[1],
        "num_pages": kv_cache.shape[0],
        "page_size": kv_cache.shape[1],
        "max_pages_per_seq": page_table.shape[1],
        "qk_nope_head_dim": qk_nope_head_dim,
        "kv_lora_rank": kv_lora_rank,
        "qk_rope_head_dim": qk_rope_head_dim,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
    }
    ShapeCapture.get().record(
        "attention",
        "mla_extend_with_kvcache",
        kernel.name,
        q.dtype,
        shape_params,
    )

    with kernel_scope(
        "attention",
        "mla_extend_with_kvcache",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            q=q,
            kv_cache=kv_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            softmax_scale=softmax_scale,
            is_causal=is_causal,
            logit_cap=logit_cap,
            return_lse=return_lse,
            out=out,
        )


def supports_mla_decode_query_blocks(
    *,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    page_size: int,
    num_q_heads: int,
    q_len: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    sliding_window: bool,
    solution: str | None = None,
) -> bool:
    """Whether an MLA decode kernel takes a proposal block on the query axis.

    A block drafter can lay its proposal out two ways: one flattened row per
    block position, each carrying the block-end cache length, or the block on
    the query axis with one page table row per request. Both spell the same
    mask, but a kernel serves one or the other, so the caller has to know
    which before it builds the metadata.

    ``True`` means the selected kernel declared both this ``q_len`` and a
    proposal block of that width. A kernel that merely omits a trait matches by
    omission, which is not proof, so omission answers ``False``.

    Args:
        q_dtype: Query dtype.
        kv_dtype: KV cache dtype.
        page_size: Tokens per KV cache page.
        num_q_heads: Query heads this rank owns.
        q_len: Query rows per request, i.e. the proposal block width.
        kv_lora_rank: MLA latent width.
        qk_rope_head_dim: RoPE width.
        sliding_window: Whether the layer bounds its history, since a kernel
            may serve one of the two masks and not the other.
        solution: The solution the call will pin, so the answer is about the
            kernel that call will actually reach.

    Returns:
        Whether the block may be handed over on the query axis.
    """
    try:
        kernel = select_kernel(
            "attention",
            "mla_decode_with_kvcache",
            format_signature(
                q=dense_tensor_format(q_dtype),
                kv_cache=dense_tensor_format(kv_dtype),
            ),
            traits={
                "sliding_window": sliding_window,
                "page_size": page_size,
                "q_len": q_len,
                "num_q_heads": num_q_heads,
                "kv_lora_rank": kv_lora_rank,
                "qk_rope_head_dim": qk_rope_head_dim,
                "support_logit_cap": False,
                "return_lse": False,
                "block_on_query_axis": True,
                "noncausal_block_size": q_len,
            },
            solution=solution,
        )
    except NoKernelFoundError:
        return False
    spec = KernelRegistry.get().get_by_name(kernel.name)
    if spec is None:
        return False
    return q_len in spec.traits.get("q_len", ()) and q_len in spec.traits.get(
        "noncausal_block_size", ()
    )


def mla_decode_with_kvcache(
    # attention inputs
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    # MLA dimensions
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    # attention options
    logit_cap: float = 0.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
    window_left: int = -1,
    noncausal_block_size: int = 1,
    # dispatch options
    override: str | None = None,
    solution: str | None = None,
    # optional projected-value epilogue
    value_weight: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
) -> AttentionResult:
    """MLA absorbed decode over compressed paged MLA KV cache.

    This API is for the absorbed MLA decode path. The model has already
    transformed the non-RoPE query part into latent space using the key half of
    kv_b_proj, so Q and the compressed cache share the same q/k dimension:
    kv_lora_rank + qk_rope_head_dim. The kernel returns the attention-weighted
    latent value. When ``value_weight`` is provided, a supporting kernel may
    instead apply the value projection and optional output gate while reducing
    the attention partials. Otherwise the API composes the latent decode and
    value projection.

    Args:
        q: Absorbed query with shape
            [batch, q_len, num_q_heads, kv_lora_rank + qk_rope_head_dim]. For
            plain decode q_len is 1; speculative/draft paths may pass q_len > 1.
        kv_cache: Paged compressed MLA cache with shape
            [num_pages, page_size, 1, kv_lora_rank + qk_rope_head_dim]. The first
            kv_lora_rank elements are latent KV; the final qk_rope_head_dim
            elements are the RoPE key part.
        page_table: Page table with shape [batch, max_pages_per_seq].
        cache_seqlens: Visible KV lengths in the cache, shape [batch]. These
            lengths include current decode tokens when they were prewritten.
        max_seqlen_k: Maximum visible KV length.
        qk_nope_head_dim: Original non-RoPE q/k head dim. Some backends need
            this for kernel specialization even though q stores the absorbed
            latent dimension.
        kv_lora_rank: MLA latent rank R. The output head dim is R.
        qk_rope_head_dim: RoPE q/k head dim.
        softmax_scale: Scale applied to QK logits before softmax.
        logit_cap: Optional soft cap applied to attention logits.
        window_left: Number of historical positions visible to the first query
            in a proposal block, or -1 for full attention. A non-causal row
            also sees the whole proposal block. DFlash2 passes the model's
            ``sliding_window - 1`` value here.
        noncausal_block_size: Proposal rows per request. Use one for ordinary
            causal decode. A block reaches a kernel in one of two layouts, and
            which one this is follows from the shapes: flattened when ``q_len``
            is 1 and the batch carries ``noncausal_block_size`` rows per
            request, each with the block-end ``cache_seqlens``; on the query
            axis when ``q_len`` equals it and the batch, page table and
            ``cache_seqlens`` carry one entry per request. Ask
            :func:`supports_mla_decode_query_blocks` before building the
            second, since not every kernel reads it.
        return_lse: Whether to also return log-sum-exp values.
        out: Optional output tensor with shape [batch, q_len, num_q_heads,
            kv_lora_rank]. When ``value_weight`` is provided, this is required
            and has shape [batch, num_q_heads * value_head_dim].
        value_weight: Optional per-head value projection with shape
            [num_q_heads, kv_lora_rank, value_head_dim].
        gate: Optional raw sigmoid gate with shape
            [batch, num_q_heads * value_head_dim]. Requires ``value_weight``.
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Returns:
        Latent attention output with shape [batch, q_len, num_q_heads,
        kv_lora_rank], or (output, lse) when return_lse is True. When
        ``value_weight`` is provided, returns ``out`` containing the projected
        and optionally gated value.
    """
    if gate is not None and value_weight is None:
        raise ValueError("gate requires value_weight")
    if window_left < -1:
        raise ValueError(f"window_left must be -1 or non-negative, got {window_left}")
    if noncausal_block_size <= 0:
        raise ValueError(
            f"noncausal_block_size must be positive, got {noncausal_block_size}"
        )
    if 0 <= window_left < noncausal_block_size - 1:
        raise ValueError(
            "window_left must cover the complete non-causal block; got "
            f"window_left={window_left}, block_size={noncausal_block_size}"
        )

    projected_value = value_weight is not None
    if projected_value:
        if q.ndim != 4 or q.shape[1] != 1:
            raise ValueError(
                "projected MLA decode requires q shape [batch,1,heads,dim]"
            )
        if value_weight.ndim != 3 or value_weight.shape[:2] != (
            q.shape[2],
            kv_lora_rank,
        ):
            raise ValueError(
                "value_weight must have shape [heads,kv_lora_rank,value_head_dim]"
            )
        if out is None:
            raise ValueError("projected MLA decode requires out")
        expected_output = (q.shape[0], q.shape[2] * value_weight.shape[2])
        if out.shape != expected_output:
            raise ValueError(f"out must have shape {expected_output}")
        if (
            out.dtype != value_weight.dtype
            or out.device != q.device
            or not out.is_contiguous()
        ):
            raise ValueError(
                "out must match value_weight dtype and be contiguous and colocated with q"
            )
        if gate is not None and gate.shape != expected_output:
            raise ValueError(f"gate must have shape {expected_output}")
        if return_lse:
            raise ValueError("projected MLA decode does not support return_lse")

    # No windowed kernel fuses the value projection, so compose the windowed
    # latent decode with the standalone projection. Kernel choice is left to
    # the ``sliding_window`` trait below rather than pinned here: more than one
    # implementation applies the mask now.
    if window_left >= 0 and projected_value:
        attention = mla_decode_with_kvcache(
            q=q,
            kv_cache=kv_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            softmax_scale=softmax_scale,
            logit_cap=logit_cap,
            window_left=window_left,
            noncausal_block_size=noncausal_block_size,
            override=override,
            solution=solution,
        )
        return mla_project_value(
            attention.reshape(q.shape[0], q.shape[2], kv_lora_rank),
            value_weight,
            gate=gate,
            out=out,
        )

    traits = {
        "batch_size": q.shape[0],
        "page_size": kv_cache.shape[1],
        "q_len": q.shape[1],
        "num_q_heads": q.shape[2],
        "batch_size_div_64": q.shape[0] % 64 == 0,
        "qk_nope_head_dim": qk_nope_head_dim,
        "kv_lora_rank": kv_lora_rank,
        "qk_rope_head_dim": qk_rope_head_dim,
        "support_logit_cap": logit_cap != 0.0,
        "return_lse": return_lse,
        "sliding_window": window_left >= 0,
        # A proposal block reaches a kernel one of two ways: flattened to one
        # row per position on the batch axis, or whole on the query axis. A
        # kernel reads one or the other, never both.
        "block_on_query_axis": q.shape[1] == noncausal_block_size,
        # Greater than one only for a block drafter's non-causal proposal, so
        # a kernel can declare itself for that case without also volunteering
        # for ordinary decode or target verify.
        "noncausal_block_size": noncausal_block_size,
    }
    if projected_value:
        traits.update(
            {
                "value_head_dim": value_weight.shape[2],
                "gate_kind": "none" if gate is None else "sigmoid",
            }
        )
        signature = _attention_format_signature(
            q=q,
            kv_cache=kv_cache,
            value_weight=value_weight,
            out=out,
        )
    else:
        signature = _attention_format_signature(q=q, kv_cache=kv_cache)
    dispatch_mode = (
        "mla_decode_projected_value" if projected_value else "mla_decode_with_kvcache"
    )
    try:
        kernel = select_kernel(
            "attention",
            dispatch_mode,
            signature,
            traits=traits,
            solution=solution,
            override=override,
        )
    except NoKernelFoundError:
        if projected_value:
            if override is not None or solution is not None:
                raise
            attention = mla_decode_with_kvcache(
                q=q,
                kv_cache=kv_cache,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                max_seqlen_k=max_seqlen_k,
                qk_nope_head_dim=qk_nope_head_dim,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                softmax_scale=softmax_scale,
                logit_cap=logit_cap,
            )
            return mla_project_value(
                attention.reshape(q.shape[0], q.shape[2], kv_lora_rank),
                value_weight,
                gate=gate,
                out=out,
            )
        if q.dtype == kv_cache.dtype:
            raise
        q = q.to(kv_cache.dtype)
        signature = _attention_format_signature(q=q, kv_cache=kv_cache)
        kernel = select_kernel(
            "attention",
            "mla_decode_with_kvcache",
            signature,
            traits=traits,
            solution=solution,
            override=override,
        )

    shape_params = {
        "batch_size": q.shape[0],
        "q_len": q.shape[1],
        "num_q_heads": q.shape[2],
        "num_pages": kv_cache.shape[0],
        "page_size": kv_cache.shape[1],
        "max_pages_per_seq": page_table.shape[1],
        "qk_nope_head_dim": qk_nope_head_dim,
        "kv_lora_rank": kv_lora_rank,
        "qk_rope_head_dim": qk_rope_head_dim,
        "max_seqlen_k": max_seqlen_k,
        "window_left": window_left,
        "noncausal_block_size": noncausal_block_size,
    }
    if projected_value:
        shape_params["value_head_dim"] = value_weight.shape[2]
    ShapeCapture.get().record(
        "attention",
        dispatch_mode,
        kernel.name,
        q.dtype,
        shape_params,
    )

    with kernel_scope(
        "attention",
        dispatch_mode,
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        if projected_value:
            return kernel(
                q=q,
                kv_cache=kv_cache,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                max_seqlen_k=max_seqlen_k,
                qk_nope_head_dim=qk_nope_head_dim,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                softmax_scale=softmax_scale,
                value_weight=value_weight,
                gate=gate,
                out=out,
                logit_cap=logit_cap,
            )
        kernel_kwargs = dict(
            q=q,
            kv_cache=kv_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            softmax_scale=softmax_scale,
            logit_cap=logit_cap,
            return_lse=return_lse,
            out=out,
        )
        # Forward the mask arguments only where they carry information, so a
        # kernel registered for plain decode is never handed a keyword it does
        # not take. A block of one with no window is plain decode.
        if window_left >= 0 or noncausal_block_size != 1:
            kernel_kwargs.update(
                window_left=window_left,
                noncausal_block_size=noncausal_block_size,
            )
        return kernel(**kernel_kwargs)


# Backend registration (side-effect imports)
# isort: off
import tokenspeed_kernel.ops.attention.mla.cuda  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.mla.tokenspeed_mla  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.mla.triton  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.mla.gluon  # noqa: E402,F401

# isort: on


__all__ = [
    "mla_project_value_prefers_contiguous_weight",
    "mla_project_value",
    "mla_normalize_project_query",
    "mla_prefill",
    "mla_use_absorbed_extend",
    "mla_extend_with_kvcache",
    "supports_mla_decode_query_blocks",
    "mla_decode_with_kvcache",
]
