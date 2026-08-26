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

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

platform = current_platform()

flash_mla_with_kvcache = error_fn
flash_mla_sparse_fwd = error_fn
get_mla_metadata = error_fn

if platform.is_nvidia and platform.is_hopper_plus:
    try:
        from flash_mla import (
            flash_mla_sparse_fwd,
            flash_mla_with_kvcache,
            get_mla_metadata,
        )
    except ImportError:
        pass


_decode_sched_meta_cache: dict[tuple, object] = {}
_dsv4_tile_meta_cache: dict[tuple, object] = {}
_query_workspace_cache: dict[tuple, torch.Tensor] = {}


def reset_dsv4_tile_metadata() -> None:
    """Discard value-dependent DSV4 FlashMLA schedules before a new forward."""
    _dsv4_tile_meta_cache.clear()


def _flashmla_sparse_prefill_head_multiple() -> int:
    return 128 if platform.is_nvidia and platform.is_blackwell_plus else 64


def _flashmla_sparse_prefill_padded_heads(num_heads: int) -> int:
    head_multiple = _flashmla_sparse_prefill_head_multiple()
    return ((int(num_heads) + head_multiple - 1) // head_multiple) * head_multiple


def _flashmla_sparse_decode_padded_heads(num_heads: int) -> int:
    num_heads = int(num_heads)
    if num_heads <= 64:
        return 64
    if num_heads <= 128:
        return 128
    return num_heads


def _get_query_workspace(
    *,
    q: torch.Tensor,
    shape: tuple[int, ...],
    cache_prefix: str,
) -> torch.Tensor:
    key = (cache_prefix, q.device, q.dtype, shape)
    workspace = _query_workspace_cache.get(key)
    if workspace is None:
        workspace = torch.empty(shape, dtype=q.dtype, device=q.device)
        workspace.zero_()
        _query_workspace_cache[key] = workspace
    return workspace


def _pad_prefill_query(q: torch.Tensor) -> tuple[torch.Tensor, int]:
    q = q.reshape(-1, q.shape[-2], q.shape[-1]).contiguous()
    actual_heads = q.shape[1]
    padded_heads = _flashmla_sparse_prefill_padded_heads(actual_heads)
    if padded_heads == actual_heads:
        return q, actual_heads
    q_padded = _get_query_workspace(
        q=q,
        shape=(q.shape[0], padded_heads, q.shape[2]),
        cache_prefix="prefill",
    )
    q_padded[:, :actual_heads, :].copy_(q)
    q_padded[:, actual_heads:, :].zero_()
    return q_padded, actual_heads


def _pad_decode_query(q: torch.Tensor, q_len_per_req: int) -> tuple[torch.Tensor, int]:
    if q.dim() == 3:
        q = q.reshape(-1, int(q_len_per_req), q.shape[1], q.shape[2])
    elif q.dim() != 4:
        raise ValueError(
            "FlashMLA sparse decode q must be [tokens, heads, dim] or "
            f"[batch, q_len, heads, dim], got {tuple(q.shape)}"
        )
    q = q.contiguous()
    actual_heads = q.shape[2]
    padded_heads = _flashmla_sparse_decode_padded_heads(actual_heads)
    if padded_heads == actual_heads:
        return q, actual_heads
    q_padded = _get_query_workspace(
        q=q,
        shape=(q.shape[0], q.shape[1], padded_heads, q.shape[3]),
        cache_prefix="decode",
    )
    q_padded[:, :, :actual_heads, :].copy_(q)
    q_padded[:, :, actual_heads:, :].zero_()
    return q_padded, actual_heads


def _flatten_regular_kv_cache(kv_cache: torch.Tensor, page_size: int) -> torch.Tensor:
    if kv_cache.dim() == 2:
        return kv_cache.view(-1, 1, kv_cache.shape[-1])
    if kv_cache.dim() == 3:
        return kv_cache.reshape(-1, kv_cache.shape[-2], kv_cache.shape[-1])
    if kv_cache.dim() == 4:
        if kv_cache.shape[1] != int(page_size):
            raise ValueError(
                f"paged kv_cache page size mismatch: got {kv_cache.shape[1]}, "
                f"expected {page_size}"
            )
        return kv_cache.reshape(-1, kv_cache.shape[-2], kv_cache.shape[-1])
    raise ValueError(f"unsupported kv_cache shape {tuple(kv_cache.shape)}")


def _paged_sparse_kv_cache(
    sparse_kv_cache: torch.Tensor, page_size: int
) -> torch.Tensor:
    if sparse_kv_cache.dim() == 2:
        return sparse_kv_cache.view(-1, int(page_size), 1, sparse_kv_cache.shape[-1])
    if sparse_kv_cache.dim() == 4:
        return sparse_kv_cache
    raise ValueError(
        f"unsupported sparse_kv_cache shape {tuple(sparse_kv_cache.shape)}"
    )


def _get_decode_sched_meta(
    *,
    q: torch.Tensor,
    num_reqs: int,
    q_len_per_req: int,
    actual_heads: int,
    topk: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> object:
    key = (
        q.device,
        q.dtype,
        int(num_reqs),
        int(q_len_per_req),
        int(actual_heads),
        int(topk),
        int(kv_lora_rank),
        int(qk_rope_head_dim),
    )
    meta = _decode_sched_meta_cache.get(key)
    if meta is None:
        meta = get_mla_metadata()[0]
        _decode_sched_meta_cache[key] = meta
    return meta


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
        "dsa_decode",
        name="flashmla_dsa_decode",
        solution="flashmla",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=frozenset({format_signature(q=dense_tensor_format(torch.bfloat16))}),
        traits={
            "page_size": frozenset({64}),
            "q_len_per_req": frozenset({1, 2, 3, 4, 5, 6}),
            "qk_nope_head_dim": frozenset({128, 192}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({64}),
            "topk": frozenset({512, 1024, 2048}),
            "kv_cache_available": frozenset({False, True}),
            "sparse_kv_cache_available": frozenset({True}),
            "topk_layout": frozenset({"global_slots"}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
        priority=Priority.PERFORMANT,
    )
    def flashmla_dsa_decode(
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        sparse_kv_cache: torch.Tensor | None,
        topk_slots: torch.Tensor,
        topk_lens: torch.Tensor | None,
        max_seqlen_k: int,
        qk_nope_head_dim: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        softmax_scale: float,
        page_size: int,
        q_len_per_req: int = 1,
        kv_seq_lens: torch.Tensor | None = None,
        logit_cap: float = 0.0,
        k_scale: float = 1.0,
        return_lse: bool = False,
        out: torch.Tensor | None = None,
        enable_pdl: bool = False,
    ) -> torch.Tensor:
        del kv_seq_lens
        if sparse_kv_cache is None:
            raise RuntimeError("FlashMLA sparse decode requires sparse_kv_cache")
        if return_lse:
            raise RuntimeError("FlashMLA sparse decode does not support return_lse")
        if logit_cap != 0.0:
            raise RuntimeError("FlashMLA sparse decode does not support logit_cap")
        q_padded, actual_heads = _pad_decode_query(q, q_len_per_req)
        num_reqs = q_padded.shape[0]
        kv_paged = _paged_sparse_kv_cache(sparse_kv_cache, page_size)
        result, _ = flash_mla_with_kvcache(
            q=q_padded,
            k_cache=kv_paged,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=int(kv_lora_rank),
            tile_scheduler_metadata=_get_decode_sched_meta(
                q=q_padded,
                num_reqs=num_reqs,
                q_len_per_req=q_padded.shape[1],
                actual_heads=actual_heads,
                topk=topk_slots.shape[-1],
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
            ),
            softmax_scale=float(softmax_scale) * float(k_scale),
            is_fp8_kvcache=True,
            indices=topk_slots.view(num_reqs, q_padded.shape[1], -1),
        )
        if result.dim() == 4:
            result = result[:, :, :actual_heads, :].reshape(
                -1, actual_heads, result.shape[-1]
            )
        else:
            result = result[:, :actual_heads, :]
        if out is not None:
            out.reshape_as(result).copy_(result)
            return out
        return result


if (
    platform.is_nvidia
    and platform.is_hopper_plus
    and flash_mla_with_kvcache is not error_fn
):

    @register_kernel(
        "attention",
        "dsv4_paged_selected_attention",
        name="flashmla_dsv4_paged_selected_attention",
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
    def flashmla_dsv4_paged_selected_attention(
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
        "dsa_prefill",
        name="flashmla_dsa_prefill",
        solution="flashmla",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=frozenset({format_signature(q=dense_tensor_format(torch.bfloat16))}),
        traits={
            "page_size": frozenset({64}),
            "q_len_per_req": frozenset({1}),
            "qk_nope_head_dim": frozenset({128, 192}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({64}),
            "topk": frozenset({512, 1024, 2048}),
            "kv_cache_available": frozenset({True}),
            "sparse_kv_cache_available": frozenset({False, True}),
            "topk_layout": frozenset({"global_slots"}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
        priority=Priority.PERFORMANT,
    )
    def flashmla_dsa_prefill(
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        sparse_kv_cache: torch.Tensor | None,
        topk_slots: torch.Tensor,
        topk_lens: torch.Tensor | None,
        max_seqlen_k: int,
        qk_nope_head_dim: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        softmax_scale: float,
        page_size: int,
        q_len_per_req: int = 1,
        kv_seq_lens: torch.Tensor | None = None,
        logit_cap: float = 0.0,
        k_scale: float = 1.0,
        return_lse: bool = False,
        out: torch.Tensor | None = None,
        enable_pdl: bool = False,
    ) -> torch.Tensor:
        if kv_cache is None:
            raise RuntimeError("FlashMLA sparse prefill requires kv_cache")
        if return_lse:
            raise RuntimeError("FlashMLA sparse prefill does not support return_lse")
        if logit_cap != 0.0:
            raise RuntimeError("FlashMLA sparse prefill does not support logit_cap")
        q_kernel, actual_heads = _pad_prefill_query(q)
        kv = _flatten_regular_kv_cache(kv_cache, page_size)
        result, _, _ = flash_mla_sparse_fwd(
            q=q_kernel,
            kv=kv,
            indices=topk_slots.unsqueeze(1),
            sm_scale=float(softmax_scale) * float(k_scale),
            d_v=int(kv_lora_rank),
        )
        result = result[:, :actual_heads, :]
        if out is not None:
            out.reshape_as(result).copy_(result)
            return out
        return result


if (
    platform.is_nvidia
    and platform.is_hopper_plus
    and flash_mla_sparse_fwd is not error_fn
):

    @register_kernel(
        "attention",
        "dsv4_selected_attention",
        name="flashmla_dsv4_selected_attention",
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
    def flashmla_dsv4_selected_attention(
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


# ------------------------------------------------------------------------------
# Direct export
# ------------------------------------------------------------------------------

__all__ = [
    "flash_mla_sparse_fwd",
    "flash_mla_with_kvcache",
    "get_mla_metadata",
]
