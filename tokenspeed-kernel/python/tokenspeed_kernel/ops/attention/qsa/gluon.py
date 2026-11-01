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

"""Registrations of the AMD GFX950 Gluon QSA kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

if current_platform().is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.attention.qsa import (
        launch_gluon_qsa_block_scores_gfx950,
        launch_gluon_qsa_block_topk_select_gfx950,
        launch_gluon_qsa_sparse_attention_gfx950,
    )

    @register_kernel(
        "attention",
        "qsa_sparse_attention",
        name="gluon_qsa_sparse_attention_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    k_cache=dense_tensor_format(cache_dtype),
                    v_cache=dense_tensor_format(cache_dtype),
                )
                for cache_dtype in (torch.bfloat16, torch.float8_e4m3fn)
            }
        ),
        traits={
            "head_dim": frozenset({128, 256}),
            "value_head_dim": frozenset({128, 256}),
        },
        priority=Priority.SPECIALIZED,
    )
    def gluon_qsa_sparse_attention_gfx950(*args, **kwargs):
        return launch_gluon_qsa_sparse_attention_gfx950(*args, **kwargs)

    @register_kernel(
        "attention",
        "qsa_block_scores",
        name="gluon_qsa_block_scores_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    query=dense_tensor_format(torch.bfloat16),
                    key_cache=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        priority=Priority.SPECIALIZED,
        features=frozenset({"logits"}),
    )
    def gluon_qsa_block_scores_gfx950(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        page_table: torch.Tensor,
        request_indices: torch.Tensor,
        complete_blocks: torch.Tensor,
        *,
        page_size: int,
        queries_per_request: int | None,
        enable_pdl: bool,
    ) -> torch.Tensor:
        # The kernel finds each tile's request runs itself, and GFX950 has no
        # programmatic dependent launch.
        del queries_per_request, enable_pdl
        return launch_gluon_qsa_block_scores_gfx950(
            query,
            key_cache,
            page_table,
            request_indices,
            complete_blocks,
            page_size=page_size,
        )

    @register_kernel(
        "attention",
        "qsa_block_topk_select",
        name="gluon_qsa_block_topk_select_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {format_signature(logits=dense_tensor_format(torch.float32))}
        ),
        priority=Priority.SPECIALIZED,
        features=frozenset({"logits"}),
    )
    def gluon_qsa_block_topk_select_gfx950(
        logits: torch.Tensor,
        topk: int,
        *,
        lengths: torch.Tensor,
        enable_pdl: bool,
    ) -> torch.Tensor:
        # GFX950 selection uses per-CTA scratch and has no PDL dependency wait.
        del enable_pdl
        return launch_gluon_qsa_block_topk_select_gfx950(logits, topk, lengths=lengths)

    __all__ = [
        "gluon_qsa_block_scores_gfx950",
        "gluon_qsa_block_topk_select_gfx950",
        "gluon_qsa_sparse_attention_gfx950",
    ]
else:
    __all__ = []
