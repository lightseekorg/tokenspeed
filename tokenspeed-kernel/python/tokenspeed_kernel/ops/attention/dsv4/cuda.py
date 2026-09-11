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

"""CUDA DeepSeek V4 attention kernels."""

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from tokenspeed_kernel.thirdparty.flash_mla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
    get_mla_metadata,
)

platform = current_platform()

try:
    from tokenspeed_kernel.thirdparty.cuda.dsv4_attention import (
        fused_qnorm_rope_kv_insert as _fused_qnorm_rope_kv_insert,
    )
    from tokenspeed_kernel.thirdparty.cuda.dsv4_attention import (
        has_fused_qnorm_rope_kv_insert,
        has_indexer_mxfp4_paged_gather,
        has_indexer_topk_prefill,
        has_persistent_topk,
        indexer_mxfp4_paged_gather,
        indexer_topk_prefill,
        persistent_topk,
    )
except ImportError:

    def has_fused_qnorm_rope_kv_insert() -> bool:
        return False

    def has_indexer_topk_prefill() -> bool:
        return False

    def has_indexer_mxfp4_paged_gather() -> bool:
        return False

    def has_persistent_topk() -> bool:
        return False

    _fused_qnorm_rope_kv_insert = error_fn
    indexer_mxfp4_paged_gather = error_fn
    indexer_topk_prefill = error_fn
    persistent_topk = error_fn


if platform.is_nvidia and has_fused_qnorm_rope_kv_insert():

    @register_kernel(
        "attention",
        "dsv4_swa_cache_insert",
        name="cuda_dsv4_swa_cache_insert",
        solution="cuda",
        capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
        signatures=frozenset(
            format_signature(
                q=dense_tensor_format(dtype),
                kv=dense_tensor_format(dtype),
                swa_kv_cache=dense_tensor_format(torch.uint8),
            )
            for dtype in (torch.float16, torch.bfloat16)
        ),
        traits={
            "head_dim": frozenset({512}),
            "rope_dim": frozenset({64}),
            "quant_block_size": frozenset({64}),
            "cache_layout": frozenset({"fp8_swa_page_planar"}),
            "has_q_out": frozenset({True, False}),
        },
        priority=Priority.SPECIALIZED,
        tags={"nvidia", "cache_insert", "latency"},
    )
    def cuda_dsv4_swa_cache_insert(
        q: torch.Tensor,
        kv: torch.Tensor,
        swa_kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        rms_norm_eps: float,
        page_size: int,
        q_out: torch.Tensor | None = None,
    ) -> None:
        q_destination = q
        if q_out is not None:
            q_out.copy_(q)
            q_destination = q_out
        _fused_qnorm_rope_kv_insert(
            q_destination,
            kv,
            swa_kv_cache,
            slot_mapping,
            positions,
            cos_sin_cache,
            rms_norm_eps,
            page_size,
        )


fused_qnorm_rope_kv_insert = _fused_qnorm_rope_kv_insert

_dsv4_tile_meta_cache: dict[tuple, object] = {}


def reset_dsv4_tile_metadata() -> None:
    """Discard value-dependent DSV4 FlashMLA schedules before a new forward."""
    _dsv4_tile_meta_cache.clear()


def _get_dsv4_tile_meta(
    q: torch.Tensor,
    selected_width: int,
    page_size: int,
    extra_page_size: int | None,
    extra_selected_width: int,
) -> object:
    phase = "graph" if torch.cuda.is_current_stream_capturing() else "eager"
    key = (
        phase,
        q.device,
        q.dtype,
        tuple(q.shape),
        int(selected_width),
        int(page_size),
        int(extra_page_size or 0),
        int(extra_selected_width),
    )
    meta = _dsv4_tile_meta_cache.get(key)
    if meta is not None and getattr(meta, "have_initialized", False):
        config = meta.config
        if (
            config.page_block_size != int(page_size)
            or config.extra_page_block_size
            != (None if extra_page_size is None else int(extra_page_size))
            or config.extra_topk
            != (None if extra_selected_width == 0 else int(extra_selected_width))
        ):
            meta = None
    if meta is None:
        meta = get_mla_metadata()[0]
        _dsv4_tile_meta_cache[key] = meta
    return meta


def _fp8_page_planar_cache_view(
    cache: torch.Tensor,
    page_size: int,
    row_bytes: int,
) -> torch.Tensor:
    required_width = int(page_size) * int(row_bytes)
    if cache.ndim != 2 or cache.shape[1] < required_width:
        raise ValueError(
            "DSV4 FP8 cache page is smaller than its logical row layout: "
            f"shape={tuple(cache.shape)}, required_width={required_width}"
        )
    return torch.as_strided(
        cache,
        (cache.shape[0], int(page_size), 1, row_bytes),
        (cache.stride(0), row_bytes, row_bytes, 1),
    )


def _dsv4_fp8_row_bytes(head_dim: int, rope_dim: int = 64) -> int:
    nope_dim = int(head_dim) - int(rope_dim)
    if nope_dim <= 0 or nope_dim % 64:
        raise ValueError(
            f"DSV4 FP8 cache requires a positive 64-aligned NoPE dim, got {nope_dim}"
        )
    return nope_dim + 2 * int(rope_dim) + nope_dim // 64 + 1


if (
    platform.is_nvidia
    and platform.is_hopper_plus
    and flash_mla_with_kvcache is not error_fn
):

    @register_kernel(
        "attention",
        "dsv4_decode",
        name="flashmla_dsv4_decode",
        solution="flashmla",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    swa_kv_cache=dense_tensor_format(torch.uint8),
                )
            }
        ),
        traits={
            "head_dim": frozenset({512}),
            "cache_layout": frozenset({"fp8_swa_page_planar"}),
            "topk_layout": frozenset({"global_slots"}),
            "support_sink": frozenset({True}),
            "has_extra_segment": frozenset({False, True}),
            "metadata_dtypes": frozenset({torch.int32}),
        },
        priority=Priority.PERFORMANT,
        tags={"nvidia", "paged_cache", "selected_attention"},
    )
    def flashmla_dsv4_decode(
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
    ) -> torch.Tensor:
        q_kernel = q.unsqueeze(1)
        swa_indices = swa_slots.reshape(q.shape[0], 1, -1)
        row_bytes = _dsv4_fp8_row_bytes(q.shape[-1])
        extra_cache = None
        extra_indices = None
        if extra_kv_cache is not None:
            assert extra_slots is not None
            assert extra_page_size is not None
            extra_cache = _fp8_page_planar_cache_view(
                extra_kv_cache,
                extra_page_size,
                row_bytes,
            )
            extra_indices = extra_slots.reshape(q.shape[0], 1, -1)
        result, _ = flash_mla_with_kvcache(
            q=q_kernel,
            k_cache=_fp8_page_planar_cache_view(
                swa_kv_cache,
                swa_page_size,
                row_bytes,
            ),
            block_table=None,
            cache_seqlens=None,
            head_dim_v=q.shape[-1],
            tile_scheduler_metadata=_get_dsv4_tile_meta(
                q_kernel,
                swa_indices.shape[-1],
                swa_page_size,
                extra_page_size,
                0 if extra_slots is None else extra_slots.shape[-1],
            ),
            softmax_scale=float(softmax_scale),
            is_fp8_kvcache=True,
            indices=swa_indices,
            attn_sink=attn_sink,
            extra_k_cache=extra_cache,
            extra_indices_in_kvcache=extra_indices,
            topk_length=swa_lens,
            extra_topk_length=extra_lens,
        )
        if result.dim() == 4:
            result = result.squeeze(1)
        if out is not None:
            out.copy_(result)
            return out
        return result


if (
    platform.is_nvidia
    and platform.is_hopper_plus
    and flash_mla_sparse_fwd is not error_fn
):

    @register_kernel(
        "attention",
        "dsv4_prefill",
        name="flashmla_dsv4_prefill",
        solution="flashmla",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    kv=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        traits={
            "head_dim": frozenset({512}),
            "cache_layout": frozenset({"dense_workspace"}),
            "support_sink": frozenset({True}),
            "metadata_dtypes": frozenset({torch.int32}),
        },
        priority=Priority.PERFORMANT,
    )
    def flashmla_dsv4_prefill(
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        lens: torch.Tensor,
        attn_sink: torch.Tensor,
        softmax_scale: float,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        result, _, _ = flash_mla_sparse_fwd(
            q=q,
            kv=kv.reshape(-1, 1, q.shape[-1]),
            indices=indices.unsqueeze(1),
            sm_scale=float(softmax_scale),
            attn_sink=attn_sink,
            topk_length=lens,
        )
        if out is not None:
            out.copy_(result)
            return out
        return result


__all__ = [
    "fused_qnorm_rope_kv_insert",
    "has_fused_qnorm_rope_kv_insert",
    "has_indexer_mxfp4_paged_gather",
    "has_indexer_topk_prefill",
    "has_persistent_topk",
    "indexer_mxfp4_paged_gather",
    "indexer_topk_prefill",
    "persistent_topk",
    "reset_dsv4_tile_metadata",
]
