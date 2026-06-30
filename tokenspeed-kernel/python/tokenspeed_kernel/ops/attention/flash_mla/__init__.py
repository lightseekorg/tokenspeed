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
_query_workspace_cache: dict[tuple, torch.Tensor] = {}


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
        logit_cap: float = 0.0,
        k_scale: float = 1.0,
        return_lse: bool = False,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        logit_cap: float = 0.0,
        k_scale: float = 1.0,
        return_lse: bool = False,
        out: torch.Tensor | None = None,
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


# ------------------------------------------------------------------------------
# Direct export
# ------------------------------------------------------------------------------

__all__ = [
    "flash_mla_sparse_fwd",
    "flash_mla_with_kvcache",
    "get_mla_metadata",
]
