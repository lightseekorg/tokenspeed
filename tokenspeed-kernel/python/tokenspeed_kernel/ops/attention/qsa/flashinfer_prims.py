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

"""Blackwell FlashInfer PrimTS registration for grouped QSA prefill."""

import torch
from tokenspeed_kernel.ops.attention.qsa.metadata import QSAPrefillMetadata
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


def flashinfer_prims_qsa_sparse_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    *,
    scale: float,
    max_seqlen_q: int,
    metadata_capacity_rows: int | None,
    k_scale: float | torch.Tensor | None,
    v_scale: float | torch.Tensor | None,
    prefill_metadata: QSAPrefillMetadata,
) -> torch.Tensor:
    """Use logical candidates to reuse KV within request-local query groups.

    Args:
        q: BF16 queries [tokens, query_heads, head_dim].
        k_cache: Flattened BF16 keys [slots, kv_heads, head_dim].
        v_cache: Flattened BF16 values with the same geometry as the keys.
        selected_slots: Equivalent physical candidates used by fallback kernels.
        scale: QK softmax scale.
        max_seqlen_q: Facade-normalized query width; logical metadata describes
            the actual ragged query lengths.
        metadata_capacity_rows: Fallback reservation; PrimTS sizes its shared
            workspace from the logical metadata's actual geometry.
        k_scale: Optional Python scalar key descale.
        v_scale: Optional Python scalar value descale.
        prefill_metadata: Logical candidates, positions and request boundaries.

    Returns:
        BF16 output with the query shape.
    """
    from tokenspeed_kernel.thirdparty.flashinfer.qsa_prims import qsa_prefill

    if prefill_metadata is None:
        raise ValueError("PrimTS QSA requires logical prefill metadata")
    if isinstance(k_scale, torch.Tensor) or isinstance(v_scale, torch.Tensor):
        raise TypeError("PrimTS QSA requires Python scalar K/V scales")
    # Logical metadata supplies the candidates and actual ragged row geometry.
    del selected_slots, max_seqlen_q, metadata_capacity_rows
    return qsa_prefill(
        q,
        k_cache,
        v_cache,
        prefill_metadata,
        scale=scale * (1.0 if k_scale is None else k_scale),
        v_scale=1.0 if v_scale is None else v_scale,
    )


if current_platform().is_nvidia:
    register_kernel(
        "attention",
        "qsa_sparse_attention",
        name="flashinfer_prims_qsa_sparse_attention",
        solution="flashinfer_prims",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    k_cache=dense_tensor_format(torch.bfloat16),
                    v_cache=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        features=frozenset({"qsa_prefill_metadata"}),
        traits={
            "is_decode": frozenset({False}),
            "has_prefill_metadata": frozenset({True}),
            "scalar_scales": frozenset({True}),
            "head_dim": frozenset({64, 128, 256}),
            "value_head_dim": frozenset({64, 128, 256}),
            "equal_head_dims": frozenset({True}),
            "block_size": frozenset({4, 8, 16, 32, 64, 128}),
            "block_topk": frozenset(range(1, 513)),
            "page_size": frozenset({4, 8, 16, 32, 64, 128, 256, 512, 1024}),
            "q_heads_per_kv": frozenset(range(1, 129)),
        },
        priority=Priority.SPECIALIZED,
    )(flashinfer_prims_qsa_sparse_attention)
