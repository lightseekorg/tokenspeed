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
from tokenspeed_kernel.platform import current_platform
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
# DSv4 Kernels
# ===-----------------------------------------------------------------------===#


def dsv4_indexer_cache_format(use_fp4: bool | None = None) -> str:
    """Resolve the DeepSeek V4 indexer cache format for this kernel platform.

    Args:
        use_fp4: Explicit format request. ``True`` selects MXFP4, ``False``
            selects scaled FP8, and ``None`` selects the platform default.

    Returns:
        ``"mxfp4"`` or ``"fp8_scaled"``.
    """

    if use_fp4 is not None:
        return "mxfp4" if use_fp4 else "fp8_scaled"
    platform = current_platform()
    return (
        "mxfp4"
        if platform.is_nvidia and platform.arch_version.major >= 10
        else "fp8_scaled"
    )


def dsv4_padded_heads(num_local_heads: int) -> int:
    """Return the local head extent required by DeepSeek V4 kernels.

    Args:
        num_local_heads: Number of attention heads assigned to this rank.

    Returns:
        A kernel-compatible local head extent. GFX950 accepts the native
        16-head Pro TP8 and 32-head Pro TP4 shapes; other platform behavior
        retains the 64/128-head padding policy.
    """

    if current_platform().is_cdna4 and num_local_heads in (16, 32):
        return num_local_heads
    if num_local_heads <= 64:
        return 64
    if num_local_heads <= 128:
        return 128
    raise ValueError(
        f"DeepSeek V4 attention supports at most 128 local heads, got {num_local_heads}"
    )


def dsv4_reset_attention_state() -> None:
    """Reset backend-owned value-dependent state before a DSV4 forward."""
    from tokenspeed_kernel.ops.attention.dsv4.cuda import reset_dsv4_tile_metadata

    reset_dsv4_tile_metadata()


def dsv4_swa_cache_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rms_norm_eps: float,
    page_size: int,
    q_out: torch.Tensor | None = None,
    override: str | None = None,
    solution: str | None = None,
    *,
    validate_positions: bool,
) -> None:
    """Normalize/rotate Q and rotate/quantize/insert DeepSeek V4 SWA K/V.

    Args:
        q: Query latents shaped ``[tokens, heads, 512]``. Updated in place when
            ``q_out`` is not provided.
        kv: Shared K/V latents shaped ``[tokens, 512]``.
        swa_kv_cache: Uint8 page-planar V4 FP8 SWA cache.
        slot_mapping: Destination cache slot for each inserted token. Slots
            outside the cache capacity suppress insertion.
        positions: Absolute positions for all query/KV tokens.
        cos_sin_cache: FP32 GPT-J-style fused cosine/sine cache of width 64.
        rms_norm_eps: Positive epsilon used to normalize Q.
        page_size: Number of cache entries in each page.
        q_out: Optional contiguous destination for normalized and rotated Q.
            When provided, ``q`` is left unchanged.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.
        validate_positions: Check that every position indexes
            ``cos_sin_cache``. Runtime integrations may disable this only after
            validating the same positions once for an equal cache capacity in
            the current forward.

    Returns:
        None. Q and the selected cache rows are written in place.
    """
    if q.ndim != 3 or q.shape[-1] != 512:
        raise ValueError(
            f"q must have shape [tokens, heads, 512], got {tuple(q.shape)}"
        )
    if kv.shape != (q.shape[0], 512):
        raise ValueError(f"kv must have shape [tokens, 512], got {tuple(kv.shape)}")
    if q.dtype not in (torch.float16, torch.bfloat16) or kv.dtype != q.dtype:
        raise TypeError("q and kv must have matching float16 or bfloat16 dtypes")
    if not q.is_contiguous() or not kv.is_contiguous():
        raise ValueError("q and kv must be contiguous")
    if positions.ndim != 1 or positions.numel() != q.shape[0]:
        raise ValueError("positions must have one entry per query token")
    if slot_mapping.ndim != 1 or slot_mapping.numel() > q.shape[0]:
        raise ValueError("slot_mapping must be one-dimensional and no longer than q")
    if positions.dtype not in (torch.int32, torch.int64):
        raise TypeError("positions must have dtype int32 or int64")
    if slot_mapping.dtype not in (torch.int32, torch.int64):
        raise TypeError("slot_mapping must have dtype int32 or int64")
    if not positions.is_contiguous() or not slot_mapping.is_contiguous():
        raise ValueError("positions and slot_mapping must be contiguous")
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if rms_norm_eps <= 0.0:
        raise ValueError(f"rms_norm_eps must be positive, got {rms_norm_eps}")
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[-1] != 64:
        raise ValueError("cos_sin_cache must have shape [max_position, 64]")
    if cos_sin_cache.dtype != torch.float32 or not cos_sin_cache.is_contiguous():
        raise TypeError("cos_sin_cache must be contiguous float32")
    row_bytes = 448 + 2 * 64 + 448 // 64 + 1
    if (
        swa_kv_cache.dtype != torch.uint8
        or swa_kv_cache.ndim != 2
        or swa_kv_cache.shape[1] < page_size * row_bytes
        or swa_kv_cache.stride(1) != 1
    ):
        raise ValueError(
            "swa_kv_cache must be a 2D uint8 page-planar cache with "
            f"at least {page_size * row_bytes} bytes per page"
        )
    tensors = (kv, swa_kv_cache, slot_mapping, positions, cos_sin_cache)
    if any(tensor.device != q.device for tensor in tensors):
        raise ValueError("all DeepSeek V4 SWA cache tensors must share a device")
    if validate_positions:
        positions_valid = (
            (positions >= 0) & (positions < cos_sin_cache.shape[0])
        ).all()
        position_error = "positions entries must index cos_sin_cache"
        if positions.device.type == "cpu":
            if not bool(positions_valid.item()):
                raise ValueError(position_error)
        else:
            torch._assert_async(positions_valid, position_error)
    if q_out is not None and (
        q_out.shape != q.shape
        or q_out.dtype != q.dtype
        or q_out.device != q.device
        or not q_out.is_contiguous()
    ):
        raise ValueError(
            "q_out must be contiguous and match q shape, dtype, and device"
        )

    signature = format_signature(
        q=dense_tensor_format(q.dtype),
        kv=dense_tensor_format(kv.dtype),
        swa_kv_cache=dense_tensor_format(swa_kv_cache.dtype),
    )
    traits = {
        "head_dim": int(q.shape[-1]),
        "rope_dim": int(cos_sin_cache.shape[-1]),
        "quant_block_size": 64,
        "cache_layout": "fp8_swa_page_planar",
        "has_q_out": q_out is not None,
    }
    kernel = select_kernel(
        "attention",
        "dsv4_swa_cache_insert",
        signature,
        traits=traits,
        override=override,
        solution=solution,
    )
    shape_params = {
        "tokens": int(q.shape[0]),
        "insert_tokens": min(int(kv.shape[0]), int(slot_mapping.numel())),
        "num_heads": int(q.shape[1]),
        "head_dim": int(q.shape[2]),
        "rope_dim": int(cos_sin_cache.shape[-1]),
        "page_size": int(page_size),
        "num_pages": int(swa_kv_cache.shape[0]),
        "has_q_out": q_out is not None,
    }
    ShapeCapture.get().record(
        "attention",
        "dsv4_swa_cache_insert",
        kernel.name,
        q.dtype,
        shape_params,
    )
    with kernel_scope(
        "attention",
        "dsv4_swa_cache_insert",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        kernel(
            q=q,
            kv=kv,
            swa_kv_cache=swa_kv_cache,
            slot_mapping=slot_mapping,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
            rms_norm_eps=rms_norm_eps,
            page_size=page_size,
            q_out=q_out,
        )


def dsv4_csa_indexer_fp8_cache_insert(
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
    compress_ratio: int = 4,
    block_table_base_offsets: torch.Tensor | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> None:
    """Compress and insert DeepSeek V4 FP8 CSA indexer-cache rows.

    Args:
        state_cache: FP32 paged compressor values and scores.
        token_to_req_indices: Request index for each input token.
        positions: Absolute token positions.
        compressor_slot_mapping: Compressor-state slots for input tokens.
        block_table: Logical-to-physical compressor-state page table.
        compressor_block_size: Number of compressor-state rows per page.
        rms_norm_weight: Width-128 RMSNorm weight.
        rms_norm_eps: RMSNorm epsilon.
        cos_sin_cache: Width-64 fused cosine and sine cache.
        kv_cache_2d: Uint8 page-planar FP8 indexer cache.
        kv_slot_mapping: Destination indexer-cache slots.
        kv_cache_block_size: Number of destination rows per page.
        compress_ratio: CSA compression ratio, currently four.
        block_table_base_offsets: Optional logical page base per request.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.

    Returns:
        None. Valid rows are written in place.
    """

    signature = _attention_format_signature(
        state_cache=state_cache,
        kv_cache=kv_cache_2d,
    )
    traits = {
        "index_head_dim": int(rms_norm_weight.numel()),
        "compress_ratio": int(compress_ratio),
        "page_size": int(kv_cache_block_size),
        "cache_format": "fp8_scaled_page_planar",
    }
    kernel = select_kernel(
        "attention",
        "dsv4_csa_indexer_fp8_cache_insert",
        signature,
        traits=traits,
        override=override,
        solution=solution,
    )
    shape_params = {
        "tokens": min(
            int(positions.numel()),
            int(compressor_slot_mapping.numel()),
            int(kv_slot_mapping.numel()),
        ),
        **traits,
    }
    ShapeCapture.get().record(
        "attention",
        "dsv4_csa_indexer_fp8_cache_insert",
        kernel.name,
        state_cache.dtype,
        shape_params,
    )
    with kernel_scope(
        "attention",
        "dsv4_csa_indexer_fp8_cache_insert",
        state_cache.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        kernel(
            state_cache=state_cache,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            compressor_slot_mapping=compressor_slot_mapping,
            block_table=block_table,
            compressor_block_size=compressor_block_size,
            rms_norm_weight=rms_norm_weight,
            rms_norm_eps=rms_norm_eps,
            cos_sin_cache=cos_sin_cache,
            kv_cache_2d=kv_cache_2d,
            kv_slot_mapping=kv_slot_mapping,
            kv_cache_block_size=kv_cache_block_size,
            compress_ratio=compress_ratio,
            block_table_base_offsets=block_table_base_offsets,
        )


def dsv4_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> torch.Tensor:
    """Run DeepSeek V4 selected attention over a dense K/V workspace.

    Args:
        q: BF16 queries shaped ``[tokens, heads, 512]``.
        kv: BF16 selected K/V workspace with rows of width 512.
        indices: Selected workspace row indices shaped ``[tokens, width]``.
            Negative entries are ignored. Nonnegative entries in each active
            prefix must be smaller than the number of rows in ``kv``.
        lens: Valid selected width for each query token.
        attn_sink: One attention sink logit per query head.
        softmax_scale: Scale applied to query-key dot products.
        out: Optional output shaped like ``q``.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.

    Returns:
        BF16 attention output shaped like ``q``.
    """
    if q.ndim != 3 or q.shape[0] < 1 or q.shape[-1] != 512:
        raise ValueError(
            f"q must have shape [tokens, heads, 512], got {tuple(q.shape)}"
        )
    if kv.ndim < 2 or kv.shape[-1] != 512:
        raise ValueError(f"kv must contain rows of width 512, got {tuple(kv.shape)}")
    tokens = int(q.shape[0])
    if indices.ndim != 2 or indices.shape[0] != tokens:
        raise ValueError("indices must have one row per query token")
    if lens.ndim != 1 or lens.numel() != tokens:
        raise ValueError("lens must have one entry per query token")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("indices must have dtype int32 or int64")
    if lens.dtype not in (torch.int32, torch.int64):
        raise TypeError("lens must have dtype int32 or int64")
    if attn_sink.numel() < q.shape[1]:
        raise ValueError("attn_sink must provide one value per query head")
    if any(tensor.device != q.device for tensor in (kv, indices, lens, attn_sink)):
        raise ValueError("all selected-attention tensors must share a device")
    if out is not None and (
        out.shape != q.shape or out.dtype != q.dtype or out.device != q.device
    ):
        raise ValueError("out must match q shape, dtype, and device")

    signature = _attention_format_signature(q=q, kv=kv)
    traits = {
        "head_dim": int(q.shape[-1]),
        "cache_layout": "dense_workspace",
        "support_sink": True,
        "selected_width": int(indices.shape[-1]),
        "metadata_dtypes": frozenset({indices.dtype, lens.dtype}),
    }
    kernel = select_kernel(
        "attention",
        "dsv4_prefill",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )
    shape_params = {
        "tokens": int(q.shape[0]),
        "num_heads": int(q.shape[1]),
        "head_dim": int(q.shape[2]),
        "selected_width": int(indices.shape[-1]),
        "kv_rows": int(kv.numel() // q.shape[-1]),
    }
    ShapeCapture.get().record(
        "attention",
        "dsv4_prefill",
        kernel.name,
        q.dtype,
        shape_params,
    )
    with kernel_scope(
        "attention",
        "dsv4_prefill",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            q=q,
            kv=kv,
            indices=indices,
            lens=lens,
            attn_sink=attn_sink,
            softmax_scale=softmax_scale,
            out=out,
        )


def dsv4_decode(
    q: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    swa_slots: torch.Tensor,
    swa_lens: torch.Tensor,
    swa_page_size: int,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    extra_kv_cache: torch.Tensor | None = None,
    extra_slots: torch.Tensor | None = None,
    extra_lens: torch.Tensor | None = None,
    extra_page_size: int | None = None,
    out: torch.Tensor | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> torch.Tensor:
    """Run DeepSeek V4 selected attention over page-planar FP8 caches.

    SWA and optional extra compressed rows form independent selected segments.
    Invalid negative slots and entries beyond each segment's per-token length
    do not contribute to attention.

    Args:
        q: BF16 queries shaped ``[tokens, heads, 512]``.
        swa_kv_cache: Uint8 page-planar SWA cache shaped
            ``[pages, page_size * row_bytes]``.
        swa_slots: Selected global SWA slots with one row per query token.
            Negative entries are ignored. Nonnegative entries in each active
            prefix must be smaller than ``pages * swa_page_size``.
        swa_lens: Valid SWA selection length for each query token.
        swa_page_size: Number of SWA rows in each cache page.
        attn_sink: One attention sink logit per query head.
        softmax_scale: Scale applied to query-key dot products.
        extra_kv_cache: Optional uint8 page-planar compressed cache.
        extra_slots: Selected global slots in ``extra_kv_cache``. Nonnegative
            entries in each active prefix must be smaller than
            ``extra_pages * extra_page_size``.
        extra_lens: Valid extra selection length for each query token.
        extra_page_size: Number of rows in each extra cache page.
        out: Optional output shaped like ``q``.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.

    Returns:
        BF16 attention output shaped like ``q``.
    """
    if q.dim() != 3 or q.shape[0] < 1 or q.shape[-1] != 512:
        raise ValueError(
            f"q must have shape [tokens, heads, 512], got {tuple(q.shape)}"
        )
    if swa_kv_cache.dim() != 2 or swa_kv_cache.dtype != torch.uint8:
        raise ValueError("swa_kv_cache must be a 2D uint8 page-planar cache")
    if swa_page_size <= 0 or swa_kv_cache.shape[1] % swa_page_size:
        raise ValueError("swa_kv_cache width must be divisible by swa_page_size")
    tokens = int(q.shape[0])
    if swa_slots.dim() < 2 or int(swa_slots.shape[0]) != tokens:
        raise ValueError("swa_slots must have one row per query token")
    if swa_lens.numel() != tokens:
        raise ValueError("swa_lens must have one entry per query token")
    if swa_slots.dtype not in (torch.int32, torch.int64):
        raise TypeError("swa_slots must have dtype int32 or int64")
    if swa_lens.dtype not in (torch.int32, torch.int64):
        raise TypeError("swa_lens must have dtype int32 or int64")
    if attn_sink.numel() < q.shape[1]:
        raise ValueError("attn_sink must provide one value per query head")
    if any(
        tensor.device != q.device
        for tensor in (swa_kv_cache, swa_slots, swa_lens, attn_sink)
    ):
        raise ValueError("all paged selected-attention tensors must share a device")

    extra_values = (extra_kv_cache, extra_slots, extra_lens, extra_page_size)
    has_extra_segment = any(value is not None for value in extra_values)
    if has_extra_segment and not all(value is not None for value in extra_values):
        raise ValueError(
            "extra_kv_cache, extra_slots, extra_lens, and extra_page_size "
            "must be provided together"
        )
    if extra_kv_cache is not None:
        assert extra_slots is not None
        assert extra_lens is not None
        assert extra_page_size is not None
        if extra_kv_cache.dim() != 2 or extra_kv_cache.dtype != torch.uint8:
            raise ValueError("extra_kv_cache must be a 2D uint8 page-planar cache")
        if extra_page_size <= 0 or extra_kv_cache.shape[1] % extra_page_size:
            raise ValueError(
                "extra_kv_cache width must be divisible by extra_page_size"
            )
        if extra_slots.dim() < 2 or int(extra_slots.shape[0]) != tokens:
            raise ValueError("extra_slots must have one row per query token")
        if extra_lens.numel() != tokens:
            raise ValueError("extra_lens must have one entry per query token")
        if extra_slots.dtype not in (torch.int32, torch.int64):
            raise TypeError("extra_slots must have dtype int32 or int64")
        if extra_lens.dtype not in (torch.int32, torch.int64):
            raise TypeError("extra_lens must have dtype int32 or int64")
        if any(
            tensor.device != q.device
            for tensor in (extra_kv_cache, extra_slots, extra_lens)
        ):
            raise ValueError("all extra selected-attention tensors must share a device")
    if out is not None and (
        out.shape != q.shape or out.dtype != q.dtype or out.device != q.device
    ):
        raise ValueError("out must match q shape, dtype, and device")

    swa_width = int(swa_slots.numel() // tokens)
    extra_width = int(extra_slots.numel() // tokens) if extra_slots is not None else 0
    signature = _attention_format_signature(q=q, swa_kv_cache=swa_kv_cache)
    traits = {
        "tokens": tokens,
        "head_dim": int(q.shape[-1]),
        "num_heads": int(q.shape[1]),
        "cache_layout": "fp8_swa_page_planar",
        "topk_layout": "global_slots",
        "support_sink": True,
        "has_extra": has_extra_segment,
        "has_extra_segment": has_extra_segment,
        "swa_selected_width": swa_width,
        "extra_selected_width": extra_width,
        "swa_page_size": int(swa_page_size),
        "extra_page_size": int(extra_page_size or 0),
        "metadata_dtypes": frozenset(
            {
                swa_slots.dtype,
                swa_lens.dtype,
                *(
                    (extra_slots.dtype, extra_lens.dtype)
                    if extra_slots is not None and extra_lens is not None
                    else ()
                ),
            }
        ),
    }
    kernel = select_kernel(
        "attention",
        "dsv4_decode",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )
    shape_params = {
        "tokens": tokens,
        "num_heads": int(q.shape[1]),
        "head_dim": int(q.shape[2]),
        "swa_selected_width": swa_width,
        "extra_selected_width": extra_width,
        "swa_page_size": int(swa_page_size),
        "extra_page_size": int(extra_page_size or 0),
        "has_extra_segment": has_extra_segment,
    }
    ShapeCapture.get().record(
        "attention",
        "dsv4_decode",
        kernel.name,
        q.dtype,
        shape_params,
    )
    with kernel_scope(
        "attention",
        "dsv4_decode",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            q=q,
            swa_kv_cache=swa_kv_cache,
            swa_slots=swa_slots,
            swa_lens=swa_lens,
            swa_page_size=swa_page_size,
            attn_sink=attn_sink,
            softmax_scale=softmax_scale,
            extra_kv_cache=extra_kv_cache,
            extra_slots=extra_slots,
            extra_lens=extra_lens,
            extra_page_size=extra_page_size,
            out=out,
        )


def dsv4_prefill_topk(
    index_q: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    cu_seqlen_k_start: torch.Tensor,
    cu_seqlen_k_end: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    page_size: int,
    topk: int,
    max_seqlen_k: int,
    index_k_format: str = "mxfp4",
    block_table_base_offsets: torch.Tensor | None = None,
    gathered_k: tuple[torch.Tensor, torch.Tensor] | None = None,
    gather_workspace: tuple[torch.Tensor, torch.Tensor] | None = None,
    out: torch.Tensor | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """Compute DSV4 prefill sparse-indexer top-k over packed cache rows.

    Args:
        index_q: Prepared ``(values, scales)`` query pair. MXFP4 values are
            packed uint8; scaled-FP8 values use ``float8_e4m3fn``.
        weights: Contiguous FP32 per-token index-head weights.
        index_k_cache: Page-backed uint8 index-K cache.
        block_table: Logical-to-physical page table for the gathered requests.
        cu_seq_lens: Cumulative gathered key-row lengths for those requests.
            With a compact table these lengths cover retained rows only.
        cu_seqlen_k_start: Inclusive gathered-key start for every query row.
        cu_seqlen_k_end: Exclusive gathered-key end for every query row.
        seq_lens: Candidate count for every query row.
        page_size: Number of index-K rows in each cache page.
        topk: Number of local candidate offsets to select.
        max_seqlen_k: Maximum candidate count represented by the logits.
        index_k_format: ``"mxfp4"`` or ``"fp8_scaled"``.
        block_table_base_offsets: Optional logical base page for each row of a
            compact ``block_table``. Returned indices are absolute logical
            offsets when provided.
        gathered_k: Optional previously gathered ``(values, scales)`` pair to
            reuse instead of gathering index_k_cache again.
        gather_workspace: Optional caller-owned MXFP4 value/scale buffers. The
            returned gathered_k aliases these buffers.
        out: Optional caller-owned int32 output with shape ``[tokens, topk]``
            (or a larger first dimension).
        override: Optional exact registered kernel name.
        solution: Optional registered solution to force through selection.

    Returns:
        A pair ``(indices, gathered_k)``. Indices are local offsets within each
        query row's packed candidate range when ``block_table_base_offsets`` is
        absent, or absolute logical offsets when it is present. Invalid entries
        are set to -1.
    """
    q_values, _ = index_q
    if index_k_format not in ("mxfp4", "fp8_scaled"):
        raise ValueError(
            "index_k_format must be 'mxfp4' or 'fp8_scaled', got " f"{index_k_format!r}"
        )
    if q_values.ndim < 3:
        raise ValueError(
            "index_q values must have at least 3 dimensions, got "
            f"{tuple(q_values.shape)}"
        )
    logical_head_dim = (
        q_values.shape[-1] * 2 if index_k_format == "mxfp4" else q_values.shape[-1]
    )
    traits = {
        "index_heads": int(q_values.shape[-2]),
        "head_dim": int(logical_head_dim),
        "topk": int(topk),
        "page_size": int(page_size),
        "index_k_format": index_k_format,
    }
    if weights.dtype != torch.float32:
        raise TypeError(f"weights must be float32, got {weights.dtype}")
    signature = _attention_format_signature(
        q=q_values, weights=weights, index_k_cache=index_k_cache
    )
    kernel = select_kernel(
        "attention",
        "dsv4_prefill_topk",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )
    shape_params = {
        "tokens": int(q_values.shape[0]),
        "index_heads": int(q_values.shape[-2]),
        "head_dim": int(traits["head_dim"]),
        "page_size": int(page_size),
        "topk": int(topk),
        "max_seqlen_k": int(max_seqlen_k),
        "index_k_format": index_k_format,
    }
    ShapeCapture.get().record(
        "attention",
        "dsv4_prefill_topk",
        kernel.name,
        q_values.dtype,
        shape_params,
    )
    with kernel_scope(
        "attention",
        "dsv4_prefill_topk",
        q_values.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        kernel_kwargs = dict(
            index_q=index_q,
            weights=weights,
            index_k_cache=index_k_cache,
            block_table=block_table,
            cu_seq_lens=cu_seq_lens,
            cu_seqlen_k_start=cu_seqlen_k_start,
            cu_seqlen_k_end=cu_seqlen_k_end,
            seq_lens=seq_lens,
            page_size=page_size,
            topk=topk,
            max_seqlen_k=max_seqlen_k,
            index_k_format=index_k_format,
            gathered_k=gathered_k,
            gather_workspace=gather_workspace,
            out=out,
        )
        spec = KernelRegistry.get().get_by_name(kernel.name)
        if spec is not None and spec.solution == "gluon":
            kernel_kwargs["block_table_base_offsets"] = block_table_base_offsets
        return kernel(**kernel_kwargs)


def dsv4_decode_topk(
    index_q: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    page_size: int,
    topk: int,
    max_context_len: int,
    plan: object,
    index_k_format: str = "mxfp4",
    block_table_base_offsets: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    persistent_topk_workspace: torch.Tensor | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> torch.Tensor:
    """Compute DSV4 decode sparse-indexer top-k over a paged index-K cache.

    Args:
        index_q: Prepared ``(values, scales)`` query pair. MXFP4 values are
            packed uint8; scaled-FP8 values use ``float8_e4m3fn``.
        weights: Contiguous FP32 per-token index-head weights.
        index_k_cache: Page-backed uint8 index-K cache.
        context_lens: Int32 context lengths shaped ``[tokens, 1]``.
        block_table: Int32 page table with one row per query token.
        page_size: Number of index-K rows in each cache page.
        topk: Number of local candidate offsets to select.
        max_context_len: Maximum context represented by block_table.
        plan: Opaque schedule returned by :func:`dsv4_plan`.
        index_k_format: ``"mxfp4"`` or ``"fp8_scaled"``.
        block_table_base_offsets: Optional logical base page for each decode
            row. Returned indices are absolute logical offsets when provided.
        out: Optional caller-owned int32 output with shape ``[tokens, topk]``
            (or a larger first dimension).
        persistent_topk_workspace: Optional caller-owned uint8 workspace of at
            least 1 MiB for the persistent local top-k implementation.
        override: Optional exact registered kernel name.
        solution: Optional registered solution to force through selection.

    Returns:
        Int32 local offsets into each token's logical index-K context when
        ``block_table_base_offsets`` is absent, or absolute logical offsets when
        it is present. Invalid entries are -1; the return aliases out when out
        is provided.
    """
    q_values, _ = index_q
    if index_k_format not in ("mxfp4", "fp8_scaled"):
        raise ValueError(
            "index_k_format must be 'mxfp4' or 'fp8_scaled', got " f"{index_k_format!r}"
        )
    if q_values.ndim < 3:
        raise ValueError(
            "index_q values must have at least 3 dimensions, got "
            f"{tuple(q_values.shape)}"
        )
    logical_head_dim = (
        q_values.shape[-1] * 2 if index_k_format == "mxfp4" else q_values.shape[-1]
    )
    traits = {
        "index_heads": int(q_values.shape[-2]),
        "head_dim": int(logical_head_dim),
        "topk": int(topk),
        "page_size": int(page_size),
        "index_k_format": index_k_format,
    }
    if weights.dtype != torch.float32:
        raise TypeError(f"weights must be float32, got {weights.dtype}")
    signature = _attention_format_signature(
        q=q_values, weights=weights, index_k_cache=index_k_cache
    )
    kernel = select_kernel(
        "attention",
        "dsv4_decode_topk",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )
    shape_params = {
        "tokens": int(q_values.shape[0]),
        "index_heads": int(q_values.shape[-2]),
        "head_dim": int(traits["head_dim"]),
        "page_size": int(page_size),
        "topk": int(topk),
        "max_context_len": int(max_context_len),
        "index_k_format": index_k_format,
    }
    ShapeCapture.get().record(
        "attention",
        "dsv4_decode_topk",
        kernel.name,
        q_values.dtype,
        shape_params,
    )
    with kernel_scope(
        "attention",
        "dsv4_decode_topk",
        q_values.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        kernel_kwargs = dict(
            index_q=index_q,
            weights=weights,
            index_k_cache=index_k_cache,
            context_lens=context_lens,
            block_table=block_table,
            page_size=page_size,
            topk=topk,
            max_context_len=max_context_len,
            plan=plan,
            index_k_format=index_k_format,
            out=out,
            persistent_topk_workspace=persistent_topk_workspace,
        )
        spec = KernelRegistry.get().get_by_name(kernel.name)
        if spec is not None and spec.solution == "gluon":
            kernel_kwargs["block_table_base_offsets"] = block_table_base_offsets
        return kernel(**kernel_kwargs)


def dsv4_plan(
    *,
    page_size: int,
    seq_lens_2d: torch.Tensor,
    out: object | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> object | None:
    """Build or refresh an opaque DeepSeek V4 decode-indexer plan.

    Args:
        page_size: Indexer KV-cache page size.
        seq_lens_2d: Per-token context lengths shaped ``[tokens, 1]``.
        out: Optional previously allocated plan object to refresh in place.
        override: Optional exact kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Returns:
        Opaque backend-owned plan object, or None when the selected backend does
        not require an explicit plan.
    """
    if seq_lens_2d.dtype != torch.int32:
        seq_lens_2d = seq_lens_2d.to(torch.int32)
    traits = {"page_size": int(page_size)}
    try:
        kernel = select_kernel(
            "attention",
            "dsv4_plan",
            format_signature(),
            traits=traits,
            solution=solution,
            override=override,
        )
    except NoKernelFoundError:
        return None

    shape_params = {
        "batch_size": int(seq_lens_2d.shape[0]),
        "tokens": int(seq_lens_2d.numel()),
        "page_size": int(page_size),
    }
    ShapeCapture.get().record(
        "attention", "dsv4_plan", kernel.name, seq_lens_2d.dtype, shape_params
    )
    with kernel_scope(
        "attention",
        "dsv4_plan",
        seq_lens_2d.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            seq_lens_2d=seq_lens_2d,
            page_size=page_size,
            out=out,
        )


def dsv4_warmup(
    *,
    hidden_size: int,
    num_attention_heads: int,
    head_dim: int,
    hc_mult: int,
    kv_lora_rank: int,
    index_n_heads: int,
    index_head_dim: int,
    indexer_cache_block_size: int,
    max_decode_tokens: int,
    mxfp4_block_size: int,
    tp_size: int,
    max_tokens: int,
    device: torch.device,
    solution: str | None = None,
) -> None:
    """Warm selected DeepSeek V4 kernels for serving shapes.

    Runtime provides only model geometry and serving bounds. Vendor and
    architecture selection, optional-library behavior, and synchronization are
    owned by the selected kernel implementation.
    """
    try:
        kernel = select_kernel(
            "attention",
            "dsv4_warmup",
            format_signature(),
            traits={},
            solution=solution,
        )
    except NoKernelFoundError:
        return
    kernel(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        head_dim=head_dim,
        hc_mult=hc_mult,
        kv_lora_rank=kv_lora_rank,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        indexer_cache_block_size=indexer_cache_block_size,
        max_decode_tokens=max_decode_tokens,
        mxfp4_block_size=mxfp4_block_size,
        tp_size=tp_size,
        max_tokens=max_tokens,
        device=device,
    )


# Backend registration (side-effect imports)
# isort: off
import tokenspeed_kernel.ops.attention.dsv4.cuda  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.dsv4.triton  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.dsv4.deep_gemm  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.dsv4.gluon  # noqa: E402,F401

# isort: on

__all__ = [
    "dsv4_indexer_cache_format",
    "dsv4_padded_heads",
    "dsv4_reset_attention_state",
    "dsv4_swa_cache_insert",
    "dsv4_csa_indexer_fp8_cache_insert",
    "dsv4_prefill",
    "dsv4_decode",
    "dsv4_prefill_topk",
    "dsv4_decode_topk",
    "dsv4_plan",
    "dsv4_warmup",
]
