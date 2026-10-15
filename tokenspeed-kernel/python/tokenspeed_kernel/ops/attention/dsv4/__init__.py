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

import torch
from tokenspeed_kernel.platform import PlatformInfo, current_platform
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import (
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
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[-1] != 64:
        raise ValueError("cos_sin_cache must have shape [max_position, 64]")
    if cos_sin_cache.dtype != torch.float32 or not cos_sin_cache.is_contiguous():
        raise TypeError("cos_sin_cache must be contiguous float32")
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
        "quant_block_size": 64,
        "rope_dim": int(cos_sin_cache.shape[-1]),
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
        "page_size": int(kv_cache_block_size),
        "compress_ratio": int(compress_ratio),
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
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    override: str | None = None,
    solution: str | None = None,
    lse: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run DeepSeek V4 selected attention over a dense K/V workspace.

    Args:
        q: BF16 queries shaped ``[tokens, heads, 512]``.
        kv: BF16 selected K/V workspace with rows of width 512.
        indices: Selected workspace row indices shaped ``[tokens, width]``.
            Negative entries are ignored. Nonnegative entries in each active
            prefix must be smaller than the number of rows in ``kv``.
        lens: Valid selected width for each query token.
        attn_sink: One sink logit per head, or None for a no-sink partial.
        softmax_scale: Scale applied to query-key dot products.
        out: Optional output shaped like ``q``.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.
        lse: Optional FP32 [tokens, heads] output for natural-log LSE.
            Partial output requires attn_sink=None; the sink is applied at merge.

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
    if attn_sink is not None and attn_sink.numel() < q.shape[1]:
        raise ValueError("attn_sink must provide one value per query head")
    if any(
        tensor is not None and tensor.device != q.device
        for tensor in (kv, indices, lens, attn_sink, lse)
    ):
        raise ValueError("all selected-attention tensors must share a device")
    if out is not None and (
        out.shape != q.shape or out.dtype != q.dtype or out.device != q.device
    ):
        raise ValueError("out must match q shape, dtype, and device")

    if lse is not None:
        if attn_sink is not None:
            raise ValueError("partial attention LSE requires no sink")
        if (
            lse.shape != q.shape[:2]
            or lse.dtype != torch.float32
            or not lse.is_contiguous()
        ):
            raise ValueError("lse must be contiguous FP32 [tokens, heads]")

    signature = _attention_format_signature(q=q, kv=kv)
    traits = {
        "num_q_heads": int(q.shape[1]),
        "head_dim": int(q.shape[-1]),
        "selected_width": int(indices.shape[-1]),
        "cache_layout": "dense_workspace",
        "metadata_dtypes": frozenset({indices.dtype, lens.dtype}),
        "sinks": attn_sink is not None,
    }
    if lse is not None:
        traits["return_lse"] = True
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
            **({"lse": lse} if lse is not None else {}),
        )


_DSV4_PARTIAL_DECODE_TRAITS = {"return_lse": True, "sinks": False}


def dsv4_decode_supports_partials(platform: PlatformInfo) -> bool:
    """Whether ``platform`` has a decode kernel that can emit a no-sink LSE.

    Decode context parallelism attends to each rank's cache shard separately
    and merges the partials through their LSE, so it needs a ``dsv4_decode``
    kernel explicitly registered with ``sinks`` including False and
    ``return_lse`` including True.

    Args:
        platform: Hardware the kernel must be registered for.

    Returns:
        True when at least one registered ``dsv4_decode`` kernel satisfies the
        platform and both traits.
    """
    specs = KernelRegistry.get().get_for_operator(
        "attention", "dsv4_decode", platform=platform
    )
    return any(
        spec_matches_traits(spec, _DSV4_PARTIAL_DECODE_TRAITS, require_all_traits=True)
        for spec in specs
    )


def dsv4_decode(
    q: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    swa_slots: torch.Tensor,
    swa_lens: torch.Tensor,
    swa_page_size: int,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    extra_kv_cache: torch.Tensor | None = None,
    extra_slots: torch.Tensor | None = None,
    extra_lens: torch.Tensor | None = None,
    extra_page_size: int | None = None,
    out: torch.Tensor | None = None,
    override: str | None = None,
    solution: str | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
        attn_sink: One attention sink logit per query head, or None to omit
            the sink when computing a DCP partial.
        softmax_scale: Scale applied to query-key dot products.
        extra_kv_cache: Optional uint8 page-planar compressed cache.
        extra_slots: Selected global slots in ``extra_kv_cache``. Nonnegative
            entries in each active prefix must be smaller than
            ``extra_pages * extra_page_size``.
        extra_lens: Extra scan length for each query token, including any -1
            holes inside that prefix. Owner filtering must not shorten it when
            indices retain their original order.
        extra_page_size: Number of rows in each extra cache page.
        out: Optional output buffer shaped like ``q``. When provided, the
            returned output is this same tensor, including with return_lse=True.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.
        return_lse: Return a partial and its natural-log LSE. The LSE never
            includes a sink, so ``attn_sink`` must be None; callers combining
            partials apply the sink once after merging them.
    Returns:
        BF16 attention output shaped like ``q``. With return_lse=True, also
        return natural-log FP32 LSE shaped [tokens, heads, 1], with one query
        per token.
    """
    if return_lse and attn_sink is not None:
        raise ValueError("dsv4_decode returns a no-sink LSE; pass attn_sink=None")
    if q.dim() != 3 or q.shape[0] < 1 or q.shape[-1] != 512:
        raise ValueError(
            f"q must have shape [tokens, heads, 512], got {tuple(q.shape)}"
        )
    tokens = int(q.shape[0])
    if swa_slots.dim() < 2 or int(swa_slots.shape[0]) != tokens:
        raise ValueError("swa_slots must have one row per query token")
    if swa_lens.numel() != tokens:
        raise ValueError("swa_lens must have one entry per query token")

    extra_values = (extra_kv_cache, extra_slots, extra_lens, extra_page_size)
    has_extra_segment = any(value is not None for value in extra_values)
    if has_extra_segment and not all(value is not None for value in extra_values):
        raise ValueError(
            "extra_kv_cache, extra_slots, extra_lens, and extra_page_size "
            "must be provided together"
        )
    if has_extra_segment:
        if extra_slots.dim() < 2 or int(extra_slots.shape[0]) != tokens:
            raise ValueError("extra_slots must have one row per query token")
        if extra_lens.numel() != tokens:
            raise ValueError("extra_lens must have one entry per query token")
    if out is not None and (out.shape != q.shape or out.dtype != q.dtype):
        raise ValueError("out must match q shape and dtype")

    swa_width = int(swa_slots.numel() // tokens)
    extra_width = int(extra_slots.numel() // tokens) if extra_slots is not None else 0
    signature = _attention_format_signature(q=q, swa_kv_cache=swa_kv_cache)
    traits = {
        "num_tokens": tokens,
        "num_q_heads": int(q.shape[1]),
        "head_dim": int(q.shape[-1]),
        "swa_page_size": int(swa_page_size),
        "extra_page_size": int(extra_page_size or 0),
        "swa_selected_width": swa_width,
        "extra_selected_width": extra_width,
        "cache_layout": "fp8_swa_page_planar",
        "has_extra_segment": has_extra_segment,
        "metadata_dtypes": frozenset(
            {
                swa_slots.dtype,
                swa_lens.dtype,
                *((extra_slots.dtype, extra_lens.dtype) if has_extra_segment else ()),
            }
        ),
        "return_lse": return_lse,
        "sinks": attn_sink is not None,
        "topk_layout": "global_slots",
    }
    kernel = select_kernel(
        "attention",
        "dsv4_decode",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )
    if return_lse:
        spec = KernelRegistry.get().get_by_name(kernel.name)
        if spec is None or True not in spec.traits.get("return_lse", ()):
            raise RuntimeError(
                f"kernel {kernel.name!r} does not declare no-sink DSV4 LSE support"
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
    if return_lse:
        shape_params["return_lse"] = True
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
            **({"return_lse": True} if return_lse else {}),
        )


# Backend registration (side-effect imports)
# isort: off
import tokenspeed_kernel.ops.attention.dsv4.cuda  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.dsv4.triton  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.dsv4.gluon  # noqa: E402,F401

# isort: on

__all__ = [
    "dsv4_indexer_cache_format",
    "dsv4_swa_cache_insert",
    "dsv4_csa_indexer_fp8_cache_insert",
    "dsv4_prefill",
    "dsv4_decode",
    "dsv4_decode_supports_partials",
]
