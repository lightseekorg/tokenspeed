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

"""Ascend solutions for the common DSA kernel interfaces.

Communication and context-parallel scheduling deliberately do not live here.
This module has the same role as :mod:`dsa.cuda`: it adapts device operators to
the vendor-neutral tokenspeed-kernel boundary.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = ["ascend_dsa_kernels", "ascend_dsa_cp_metadata"]

platform = current_platform()
_kernels = None


def ascend_dsa_cp_metadata(table, lengths, **kwargs):
    """Write DSA query/group metadata and slots into caller-provided buffers.

    INT32 ``table[B,P]`` and ``lengths[B]`` expand to R=B*query_width causal
    query rows. ``out[R,P]`` and the per-query vectors receive local metadata;
    ``index_*`` buffers have R/query_group_size grouped rows. Optional global
    ``slots`` map to matching INT64 ``local_slots``, with -1 for skipped writes.
    Returns None; buffers are updated in place and retained through graph use.
    See :func:`tokenspeed_kernel_npu.ops.dsa.dsa_cp_metadata` for all keyword
    arguments, output shapes and ownership semantics.
    """
    from tokenspeed_kernel_npu.ops.dsa import dsa_cp_metadata

    dsa_cp_metadata(table, lengths, **kwargs)


def ascend_dsa_kernels():
    """Return the process-local Ascend DSA operator adapter."""
    global _kernels
    if not platform.is_npu:
        raise RuntimeError("Ascend DSA kernels require an NPU platform")
    if _kernels is None:
        from tokenspeed_kernel_npu.ops.dsa import AscendDSAKernels

        _kernels = AscendDSAKernels()
    return _kernels


if platform.is_npu:

    @register_kernel(
        "attention",
        "dsa_interleave_rope",
        name="ascend_dsa_interleave_rope",
        solution="ascend",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=frozenset(
            {
                format_signature(tensor=dense_tensor_format(torch.bfloat16)),
                format_signature(tensor=dense_tensor_format(torch.float32)),
            }
        ),
        traits={"rope_dim": frozenset({64})},
        priority=Priority.PERFORMANT,
    )
    def ascend_dsa_interleave_rope(
        tensor: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        rope_dim: int,
    ) -> torch.Tensor:
        return ascend_dsa_kernels().interleave_rope(
            tensor,
            positions,
            cos_sin_cache,
            rope_dim=rope_dim,
        )

    @register_kernel(
        "attention",
        "dsa_decode_topk",
        name="ascend_lightning_indexer_dsa_decode_topk",
        solution="ascend",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    weights=dense_tensor_format(torch.bfloat16),
                ),
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    weights=dense_tensor_format(torch.float32),
                ),
            }
        ),
        traits={
            "index_heads": frozenset({16}),
            "head_dim": frozenset({128}),
            "topk": frozenset({2048}),
            "page_size": frozenset({128}),
            "q_len_per_req": frozenset({1}),
            "index_k_layout": frozenset({"page_planar"}),
        },
        features={"logical_offsets", "forced_initial_local"},
        priority=Priority.PERFORMANT,
    )
    def ascend_lightning_indexer_dsa_decode_topk(
        q: torch.Tensor,
        weights: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        page_size: int,
        topk: int,
        softmax_scale: float,
        q_len_per_req: int,
        index_k_cache: torch.Tensor | None,
        seq_lens_2d: torch.Tensor | None,
        plan: object | None,
        out: torch.Tensor | None,
        lens_out: torch.Tensor | None,
        topk_layout: str,
        block_table_base_offsets: torch.Tensor | None,
        initial_tokens: int,
        local_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del softmax_scale, seq_lens_2d, plan, block_table_base_offsets
        if index_k_cache is None:
            raise ValueError("Ascend LightningIndexer requires index_k_cache")
        if q_len_per_req != 1:
            raise ValueError(
                "Ascend LightningIndexer decode currently requires q_len=1"
            )
        if topk_layout != "logical_offsets":
            raise ValueError("Ascend LightningIndexer returns logical offsets")
        q_ends = torch.arange(1, q.shape[0] + 1, dtype=torch.int32, device=q.device)
        indices, _ = ascend_dsa_kernels().index(
            q,
            index_k_cache,
            weights,
            q_ends,
            seq_lens,
            block_table,
            topk,
            initial_tokens,
            local_tokens,
        )
        if out is not None:
            out.copy_(indices)
            indices = out
        lengths = seq_lens.to(torch.int32).clamp(max=topk)
        if lens_out is not None:
            lens_out.copy_(lengths)
            lengths = lens_out
        return indices, lengths
