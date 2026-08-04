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

"""Deterministic DSA decode indexer top-k via flashinfer.

The trtllm ``indexer_topk_decode`` kernel breaks ties (equal logits competing for
the last selected slot) non-deterministically: repeated runs select *different*
index sets, which makes long-context greedy decode irreproducible and breaks
eager-vs-CUDA-graph parity. flashinfer's radix top-k exposes a stable,
index-ordered tie-break plus a graph-safe path, so the selection is identical
across eager, repeated runs, and CUDA-graph replay -- with zero accuracy loss
(it still selects the mathematically-correct top-k set, only the tie-break and
output order become deterministic).
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.mla.flashinfer import (
    trtllm_batch_decode_with_kv_cache_mla,
)
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

platform = current_platform()

_sparse_workspace_buffers: dict[torch.device, torch.Tensor] = {}
_SPARSE_WORKSPACE_BYTES = 384 * 1024 * 1024

top_k = None
TopKTieBreak = None

if platform.is_nvidia:
    try:
        from flashinfer import TopKTieBreak, top_k
    except ImportError:
        pass


def has_deterministic_decode_topk() -> bool:
    """Whether the flashinfer deterministic top-k fallback is importable."""
    return top_k is not None and TopKTieBreak is not None


def deterministic_decode_topk(
    logits: torch.Tensor,
    out: torch.Tensor,
    topk: int,
) -> None:
    """Select per-row top-``topk`` local offsets deterministically via flashinfer.

    ``logits`` rows must already be pre-masked with ``-inf`` beyond each request's
    valid length; the fallback uses a stable ``tie_break=SMALL`` plus
    ``deterministic`` + ``dsa_graph_safe``. For the length-aware (ragged) path see
    :func:`tokenspeed_kernel.ops.attention.dsa.cuda.ragged_decode_topk`.
    """
    if top_k is None or TopKTieBreak is None:
        raise RuntimeError("flashinfer deterministic top_k is unavailable.")
    _values, indices = top_k(
        logits.contiguous(),
        int(topk),
        deterministic=True,
        tie_break=TopKTieBreak.SMALL,
        dsa_graph_safe=True,
    )
    out.copy_(indices.to(torch.int32))


def _get_sparse_workspace(device: torch.device | str) -> torch.Tensor:
    device = torch.device(device)
    workspace = _sparse_workspace_buffers.get(device)
    if workspace is None:
        workspace = torch.zeros(
            _SPARSE_WORKSPACE_BYTES, dtype=torch.uint8, device=device
        )
        _sparse_workspace_buffers[device] = workspace
    return workspace


def _trtllm_mla_kv_cache(
    kv_cache: torch.Tensor, page_size: int, dtype: torch.dtype
) -> torch.Tensor:
    if kv_cache.dtype != dtype:
        kv_cache = kv_cache.to(dtype)
    if kv_cache.dim() == 2:
        return kv_cache.view(-1, int(page_size), kv_cache.shape[-1]).unsqueeze(1)
    if kv_cache.dim() == 3 and kv_cache.shape[1] == 1:
        return (
            kv_cache.squeeze(1)
            .view(-1, int(page_size), kv_cache.shape[-1])
            .unsqueeze(1)
        )
    if kv_cache.dim() == 4:
        if kv_cache.shape[1] == int(page_size) and kv_cache.shape[2] == 1:
            return kv_cache.permute(0, 2, 1, 3).contiguous()
        if kv_cache.shape[1] == 1 and kv_cache.shape[2] == int(page_size):
            return kv_cache.contiguous()
    raise ValueError(
        "kv_cache must be flat [slots, dim], flat [slots, 1, dim], or paged "
        "[pages, page_size, 1, dim] for FlashInfer/TRTLLM sparse MLA, got "
        f"{tuple(kv_cache.shape)}"
    )


def _topk_lens_or_count(
    topk_slots: torch.Tensor, topk_lens: torch.Tensor | None
) -> torch.Tensor:
    if topk_lens is not None:
        return topk_lens.to(device=topk_slots.device, dtype=torch.int32).contiguous()
    return (topk_slots >= 0).sum(dim=-1, dtype=torch.int32).contiguous()


if platform.is_nvidia and platform.is_hopper_plus:

    @register_kernel(
        "attention",
        "dsa_decode",
        name="flashinfer_trtllm_dsa_decode",
        solution="flashinfer_trtllm",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0), vendors=frozenset({"nvidia"})
        ),
        signatures=frozenset(
            {
                format_signature(q=dense_tensor_format(torch.bfloat16)),
                format_signature(q=dense_tensor_format(torch.float8_e4m3fn)),
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "page_size": frozenset({64}),
            "q_len_per_req": frozenset({1, 2, 3, 4, 5, 6}),
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
    )
    def flashinfer_trtllm_dsa_decode(
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
            raise RuntimeError("FlashInfer/TRTLLM sparse MLA requires kv_cache")
        if return_lse:
            raise RuntimeError(
                "FlashInfer/TRTLLM sparse MLA does not support return_lse"
            )
        if logit_cap != 0.0:
            raise RuntimeError(
                "FlashInfer/TRTLLM sparse MLA does not support logit_cap"
            )
        if q.dim() == 3:
            num_tokens = q.shape[0]
            q_kernel = q.view(num_tokens, 1, q.shape[1], q.shape[2])
        elif q.dim() == 4:
            num_tokens = q.shape[0] * q.shape[1]
            q_kernel = q.reshape(num_tokens, 1, q.shape[2], q.shape[3])
        else:
            raise ValueError(f"unsupported q shape {tuple(q.shape)}")
        kv_dtype = q.dtype if q.dtype == torch.float8_e4m3fn else kv_cache.dtype
        result = trtllm_batch_decode_with_kv_cache_mla(
            query=q_kernel,
            kv_cache=_trtllm_mla_kv_cache(kv_cache, page_size, kv_dtype),
            workspace_buffer=_get_sparse_workspace(q.device),
            qk_nope_head_dim=int(qk_nope_head_dim),
            kv_lora_rank=int(kv_lora_rank),
            qk_rope_head_dim=int(qk_rope_head_dim),
            block_tables=topk_slots.view(num_tokens, 1, -1),
            seq_lens=_topk_lens_or_count(topk_slots, topk_lens),
            max_seq_len=int(max_seqlen_k),
            sparse_mla_top_k=topk_slots.shape[-1],
            bmm1_scale=float(k_scale) * float(softmax_scale),
            backend="trtllm-gen",
        )
        result = result.reshape(num_tokens, q_kernel.shape[2], int(kv_lora_rank))
        if out is not None:
            out.reshape_as(result).copy_(result)
            return out
        return result

    @register_kernel(
        "attention",
        "dsa_prefill",
        name="flashinfer_trtllm_dsa_prefill",
        solution="flashinfer_trtllm",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0), vendors=frozenset({"nvidia"})
        ),
        signatures=frozenset(
            {
                format_signature(q=dense_tensor_format(torch.bfloat16)),
                format_signature(q=dense_tensor_format(torch.float8_e4m3fn)),
            }
        ),
        priority=Priority.SPECIALIZED,
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
    )
    def flashinfer_trtllm_dsa_prefill(**kwargs) -> torch.Tensor:
        return flashinfer_trtllm_dsa_decode(**kwargs)
