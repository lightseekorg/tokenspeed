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

"""Optional Blackwell Prims-TS FP8 D576 sparse attention registrations."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

sparse_attention = None
if current_platform().is_nvidia:
    try:
        from tokenspeed_kernel.thirdparty.flashinfer.dsa_prims import sparse_attention
    except ImportError:
        # FlashInfer releases without PR #7984 lack the sparse MLA wrapper.
        pass


def has_prims_sparse_mla() -> bool:
    """Whether the installed FlashInfer provides Prims-TS sparse MLA."""
    return sparse_attention is not None


def _attention(
    *,
    mode: str,
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
    q_len_per_req: int,
    kv_seq_lens: torch.Tensor | None,
    logit_cap: float,
    k_scale: float,
    return_lse: bool,
    out: torch.Tensor | None,
    enable_pdl: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if kv_cache is None or sparse_kv_cache is not None:
        raise ValueError(
            "Prims-TS DSA requires a native KV cache without a packed cache"
        )
    if logit_cap != 0.0:
        raise ValueError("Prims-TS DSA does not support logit_cap")
    if not enable_pdl:
        # Upstream owns PDL with no off switch; reject rather than ignore the request.
        raise ValueError(
            "Prims-TS DSA requires PDL enabled; upstream owns its scheduling"
        )
    if (kv_lora_rank, qk_rope_head_dim) != (512, 64):
        raise ValueError("Prims-TS DSA requires D576 (latent 512, QK-only tail 64)")
    if qk_nope_head_dim not in (128, 192) or page_size != 64:
        raise ValueError("unsupported Prims-TS DSA query or page geometry")
    if q.ndim not in (3, 4) or q.shape[-1] != 576 or not 1 <= q.shape[-2] <= 128:
        raise ValueError("Prims-TS DSA requires 1..128 query heads of dimension 576")
    if q.dtype != torch.float8_e4m3fn or kv_cache.dtype != q.dtype:
        raise TypeError("Prims-TS DSA requires native E4M3 queries and KV cache")
    if kv_cache.device != q.device or not kv_cache.is_contiguous():
        raise ValueError("KV cache must be contiguous on the query device")
    cache_shape = tuple(kv_cache.shape)
    if not (
        (kv_cache.ndim == 2 and cache_shape[-1] == 576)
        or (kv_cache.ndim == 3 and cache_shape[1:] == (1, 576))
        or (kv_cache.ndim == 4 and cache_shape[1:] in ((1, 64, 576), (64, 1, 576)))
    ):
        raise ValueError(f"unsupported native KV cache shape {cache_shape}")
    query = q.reshape(-1, q.shape[-2], 576).contiguous()
    rows, heads, _ = query.shape
    if q_len_per_req < 1 or rows % q_len_per_req:
        raise ValueError("query width must divide the query row count")
    if q.ndim == 4 and q.shape[1] != q_len_per_req:
        raise ValueError("query width must match the 4-D query shape")
    if topk_slots.ndim != 2 or topk_slots.shape[0] != rows:
        raise ValueError("topk_slots must have one row per query token")
    if topk_slots.shape[1] not in (512, 1024, 2048):
        raise ValueError("Prims-TS DSA supports top-k 512, 1024, or 2048")
    if topk_slots.device != q.device or topk_slots.dtype != torch.int32:
        raise ValueError("topk_slots must use INT32 on the query device")
    if not topk_slots.is_contiguous():
        raise ValueError("topk_slots must be contiguous")
    if topk_lens is not None and (
        topk_lens.shape != (rows,)
        or topk_lens.device != q.device
        or topk_lens.dtype != torch.int32
        or not topk_lens.is_contiguous()
    ):
        raise ValueError(
            "topk_lens must be contiguous INT32 [tokens] on the query device"
        )
    if out is not None and (
        out.device != q.device
        or out.dtype != torch.bfloat16
        or out.numel() != rows * heads * 512
        or not out.is_contiguous()
    ):
        raise ValueError(
            "out must be contiguous BF16 on the query device with the output size"
        )
    # The indexer enforces causality; the selected slots alone define this core's reads.
    del max_seqlen_k, kv_seq_lens
    return sparse_attention(
        query,
        kv_cache.view(-1, 576),
        topk_slots,
        topk_lens,
        mode=mode,
        query_width=q_len_per_req,
        softmax_scale=float(softmax_scale) * float(k_scale),
        return_lse=return_lse,
        out=out,
    )


def flashinfer_prims_dsa_decode(**kwargs):
    """Run FP8 D576 sparse decode or speculative verification."""
    return _attention(mode="decode", **kwargs)


def flashinfer_prims_dsa_prefill(**kwargs):
    """Run FP8 D576 sparse prefill with the same native cache contract."""
    return _attention(mode="prefill", **kwargs)


if has_prims_sparse_mla():
    for _mode, _function in (
        ("dsa_decode", flashinfer_prims_dsa_decode),
        ("dsa_prefill", flashinfer_prims_dsa_prefill),
    ):
        register_kernel(
            "attention",
            _mode,
            name=_function.__name__,
            solution="flashinfer_prims",
            capability=CapabilityRequirement(
                min_arch_version=ArchVersion(10, 0),
                max_arch_version=ArchVersion(10, 3),
                vendors=frozenset({"nvidia"}),
            ),
            signatures=format_signatures("q", "dense", {torch.float8_e4m3fn}),
            # Keep TRTLLM's established default until end-to-end validation.
            priority=Priority.PERFORMANT,
            traits={
                "q_len": (
                    frozenset(range(1, 9)) if _mode == "dsa_decode" else frozenset({1})
                ),
                "qk_nope_head_dim": frozenset({128, 192}),
                "kv_lora_rank": frozenset({512}),
                "qk_rope_head_dim": frozenset({64}),
                "page_size": frozenset({64}),
                "topk": frozenset({512, 1024, 2048}),
                "has_kv_cache": frozenset({True}),
                "has_sparse_kv_cache": frozenset({False}),
                "logit_cap": frozenset({False}),
                "return_lse": frozenset({False, True}),
                "topk_layout": frozenset({"global_slots"}),
            },
        )(_function)
