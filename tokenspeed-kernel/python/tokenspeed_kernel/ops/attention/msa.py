# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation

"""MiniMax MSA CuTe-DSL block-sparse prefill attend (SM100).

Registers the vendored MiniMax-AI/MSA kernels (``thirdparty/msa``,
the upstream package name) as the SPECIALIZED ``msa`` solution for
``msa_extend_with_kvcache``.  Block selection reuses the
Triton ``minimax_indexer``; only the attend over the selected blocks runs on
the CuTe kernel.  Decode is intentionally not registered and keeps selecting
the Triton solution (the CuTe decode entry point is fp8-only and unused by
vLLM as well).
"""

from __future__ import annotations

import functools
import importlib.util
from collections.abc import Sequence

import torch
from tokenspeed_kernel.ops.attention.triton.minimax_indexer import (
    SPARSE_BLOCK_SIZE,
    minimax_indexer,
)
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    format_signatures,
)

platform = current_platform()


def _fmha_sm100_importable() -> bool:
    return (
        importlib.util.find_spec("cutlass") is not None
        and importlib.util.find_spec("quack") is not None
    )


@functools.lru_cache(maxsize=1)
def _load_sparse_api():
    # Deferred: importing the vendored package pulls the cutlass-dsl stack and
    # performs its sys.path bootstrap of the cute/ sources.
    from tokenspeed_kernel.thirdparty.msa.sparse import (
        build_k2q_csr,
        sparse_atten_func,
    )

    return build_k2q_csr, sparse_atten_func


def _is_identity_scale(scale: float | torch.Tensor | None) -> bool:
    return scale is None or float(scale) == 1.0


if platform.is_nvidia and platform.is_blackwell and _fmha_sm100_importable():
    _MINIMAX_MSA_CUTE_TRAITS = {
        "head_dim": frozenset({128}),
        "index_head_dim": frozenset({128}),
        "page_size": frozenset({128}),
        "topk": frozenset({16}),
    }
    _MINIMAX_MSA_CUTE_SIGNATURES = format_signatures(
        ("q", "index_q", "index_k", "k_cache", "v_cache", "index_k_cache"),
        "dense",
        {torch.bfloat16},
    ) | frozenset(
        {
            # FP8-E4M3 main K/V cache: the CuTe kernel stages FP8 K/V to BF16
            # in-kernel (identity scale); queries and index stay BF16.
            format_signature(
                q=dense_tensor_format(torch.bfloat16),
                index_q=dense_tensor_format(torch.bfloat16),
                index_k=dense_tensor_format(torch.bfloat16),
                k_cache=dense_tensor_format(torch.float8_e4m3fn),
                v_cache=dense_tensor_format(torch.float8_e4m3fn),
                index_k_cache=dense_tensor_format(torch.bfloat16),
            )
        }
    )

    @register_kernel(
        "attention",
        "msa_extend_with_kvcache",
        name="msa_minimax_extend_with_kvcache",
        solution="msa",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=_MINIMAX_MSA_CUTE_SIGNATURES,
        traits=_MINIMAX_MSA_CUTE_TRAITS,
        priority=Priority.SPECIALIZED,
    )
    def msa_minimax_extend_with_kvcache(
        q: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        index_k_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        prefix_lens: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        *,
        topk: int,
        page_size: int,
        index_scale: float,
        attention_scale: float,
        init_blocks: int,
        local_blocks: int,
        k_scale: float | torch.Tensor | None = None,
        v_scale: float | torch.Tensor | None = None,
        query_lens_cpu: Sequence[int] | None = None,
        seq_lens_cpu: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Run MiniMax sparse-attention extend with the CuTe attend kernel."""

        if not (_is_identity_scale(k_scale) and _is_identity_scale(v_scale)):
            # The CuTe kernel has no descale parameters (FP8 K/V stage to BF16
            # at identity scale); descaled caches keep the Triton attend.
            from tokenspeed_kernel.ops.attention.triton.minimax_sparse_attention import (
                triton_minimax_msa_extend_with_kvcache,
            )

            return triton_minimax_msa_extend_with_kvcache(
                q,
                index_q,
                index_k,
                k_cache,
                v_cache,
                index_k_cache,
                slot_mapping,
                page_table,
                cache_seqlens,
                cu_seqlens_q,
                prefix_lens,
                max_seqlen_q,
                max_seqlen_k,
                topk=topk,
                page_size=page_size,
                index_scale=index_scale,
                attention_scale=attention_scale,
                init_blocks=init_blocks,
                local_blocks=local_blocks,
                k_scale=k_scale,
                v_scale=v_scale,
                query_lens_cpu=query_lens_cpu,
                seq_lens_cpu=seq_lens_cpu,
            )
        if q.shape[0] == 0:
            return torch.empty_like(q)

        build_k2q_csr, sparse_atten_func = _load_sparse_api()

        max_blocks = min(
            page_table.shape[1],
            (max_seqlen_k + page_size - 1) // page_size,
        )
        selected_blocks = minimax_indexer(
            index_q,
            index_k,
            index_k_cache,
            slot_mapping,
            page_table,
            cache_seqlens,
            topk=topk,
            scale=index_scale,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            cu_seqlens_q=cu_seqlens_q,
            prefix_lens=prefix_lens,
            max_query_len=max_seqlen_q,
            max_blocks=max_blocks,
            query_lens_cpu=query_lens_cpu,
            seq_lens_cpu=seq_lens_cpu,
        )

        q = q.contiguous()
        device = q.device
        cu_seqlens_q = cu_seqlens_q.to(device=device, dtype=torch.int32).contiguous()
        seqused_k = cache_seqlens.to(device=device, dtype=torch.int32).contiguous()
        page_table = page_table.to(device=device, dtype=torch.int32).contiguous()
        cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(seqused_k, dim=0, dtype=torch.int32), (1, 0)
        )
        # Exact packed KV-block row count for the CSR builder; the host read is
        # a sync, acceptable on the eager extend path (vLLM computes the same
        # value host-side in its metadata builder).
        total_rows = int(torch.sum((seqused_k + page_size - 1) // page_size).item())

        k2q_row_ptr, k2q_q_indices, schedule = build_k2q_csr(
            selected_blocks.transpose(0, 1),
            cu_seqlens_q,
            cu_seqlens_k,
            SPARSE_BLOCK_SIZE,
            total_k=0,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            total_rows=total_rows,
            qhead_per_kv=q.shape[1] // k_cache.shape[1],
            return_schedule=True,
        )
        out = torch.empty_like(q)
        sparse_atten_func(
            q,
            k_cache,
            v_cache,
            k2q_row_ptr,
            k2q_q_indices,
            topK=topk,
            blk_kv=SPARSE_BLOCK_SIZE,
            causal=True,
            softmax_scale=attention_scale,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            page_table=page_table,
            seqused_k=seqused_k,
            schedule=schedule,
            out=out,
        )
        return out
