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

"""DeepGEMM registration for KPool prefill selection."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.kpool._triton.expand import (
    expand_kpool_to_flat_kv,
)
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


def _kpool_cache_views(
    cache: torch.Tensor, page_size: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    groups = head_dim // 128
    page_stride = cache.stride(0)
    values = torch.as_strided(
        cache,
        (cache.shape[0], page_size, head_dim),
        (page_stride, head_dim, 1),
    )
    scales = torch.as_strided(
        cache,
        (cache.shape[0], page_size, groups * 4),
        (page_stride, groups * 4, 1),
        cache.storage_offset() + page_size * head_dim,
    )
    return values.view(torch.float8_e4m3fn), scales.view(torch.float32)


if current_platform().is_nvidia:
    from tokenspeed_kernel.ops.attention.dsa.deep_gemm import (
        deep_gemm_dsa_prefill_topk,
    )

    @register_kernel(
        "attention",
        "kpool_prefill_topk",
        name="deep_gemm_kpool_prefill_topk",
        solution="deep_gemm",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=frozenset({format_signature(q=dense_tensor_format(torch.bfloat16))}),
        traits={
            "head_dim": frozenset({128}),
            "pool_size": frozenset({4}),
            "page_size": frozenset({16, 64}),
            "index_k_format": frozenset({"fp8_scaled"}),
            "score_activation": frozenset({"relu"}),
            "topk_layout": frozenset({"global_slots"}),
            "topk_pools": frozenset({512, 1024, 2048}),
            "prefill_plan": frozenset({True}),
        },
        priority=Priority.PERFORMANT,
        tags={"deep_gemm", "kpool", "ragged-prefill"},
    )
    def deep_gemm_kpool_prefill_topk(
        q: torch.Tensor,
        pooled_k_cache: torch.Tensor,
        weights: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        index_block_table: torch.Tensor,
        kv_block_table: torch.Tensor,
        *,
        pool_size: int,
        page_size: int,
        kv_page_size: int,
        topk_pools: int,
        softmax_scale: float,
        apply_relu: bool = True,
        append_tail: bool = True,
        chunk_pools: int = 8192,
        req_ids: torch.Tensor | None = None,
        causal_lens: torch.Tensor | None = None,
        pool_workspace_slots: torch.Tensor | None = None,
        row_starts: torch.Tensor | None = None,
        row_ends: torch.Tensor | None = None,
        max_num_pools: int | None = None,
        max_logits_bytes: int | None = None,
        out: torch.Tensor | None = None,
        lens_out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run planned KPool prefill with DeepGEMM scoring and TRT-LLM top-k."""
        del positions, query_start_loc, index_block_table, chunk_pools
        if not apply_relu:
            raise ValueError("DeepGEMM KPool prefill requires apply_relu=True")
        planned = (
            req_ids,
            causal_lens,
            pool_workspace_slots,
            row_starts,
            row_ends,
            max_num_pools,
        )
        if any(value is None for value in planned):
            raise RuntimeError(
                "DeepGEMM KPool prefill requires a complete prefill plan"
            )
        if not q.is_cuda:
            raise RuntimeError("DeepGEMM KPool prefill requires CUDA tensors")

        device = q.device
        num_tokens = q.shape[0]
        req_ids = req_ids.to(device=device, dtype=torch.int32).contiguous()
        causal_lens = causal_lens.to(device=device, dtype=torch.int32).contiguous()
        pool_workspace_slots = pool_workspace_slots.to(
            device=device, dtype=torch.int64
        ).contiguous()
        row_starts = row_starts.to(device=device, dtype=torch.int32).contiguous()
        row_ends = row_ends.to(device=device, dtype=torch.int32).contiguous()
        kv_block_table = kv_block_table.to(
            device=device, dtype=torch.int32
        ).contiguous()
        for name, tensor in (
            ("req_ids", req_ids),
            ("causal_lens", causal_lens),
            ("row_starts", row_starts),
            ("row_ends", row_ends),
        ):
            if tensor.numel() != num_tokens:
                raise ValueError(
                    f"{name} must have {num_tokens} entries, got {tensor.numel()}"
                )

        max_num_pools = int(max_num_pools)
        if max_num_pools <= int(topk_pools):
            pool_indices = torch.empty(
                (num_tokens, int(topk_pools)),
                dtype=torch.int32,
                device=device,
            )
            return expand_kpool_to_flat_kv(
                pool_indices,
                causal_lens,
                req_ids,
                kv_block_table,
                pool_size=int(pool_size),
                kv_page_size=int(kv_page_size),
                append_tail=append_tail,
                out=out,
                lens_out=lens_out,
            )

        values, scales = _kpool_cache_views(
            pooled_k_cache,
            int(page_size),
            q.shape[-1],
        )
        pages = torch.div(pool_workspace_slots, int(page_size), rounding_mode="floor")
        rows = torch.remainder(pool_workspace_slots, int(page_size))
        index_k_fp8 = values[pages, rows]
        index_k_scale = scales[pages, rows]

        workspace_indices, _ = deep_gemm_dsa_prefill_topk(
            q,
            weights,
            pool_workspace_slots,
            row_starts,
            row_ends,
            topk=int(topk_pools),
            softmax_scale=softmax_scale,
            index_k_fp8=index_k_fp8,
            index_k_scale=index_k_scale,
            max_logits_bytes=max_logits_bytes,
            max_seqlen_k=max(max_num_pools, 1),
        )
        valid = workspace_indices >= 0
        pool_indices = torch.where(
            valid,
            workspace_indices - row_starts.unsqueeze(1),
            workspace_indices,
        ).to(torch.int32)
        return expand_kpool_to_flat_kv(
            pool_indices.contiguous(),
            causal_lens,
            req_ids,
            kv_block_table,
            pool_size=int(pool_size),
            kv_page_size=int(kv_page_size),
            append_tail=append_tail,
            out=out,
            lens_out=lens_out,
        )
