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

"""CuTe DSL registration for the B200/B300 QSA sparse-attention specialization."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
    pdl_enabled,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

_PLATFORM = current_platform()
_IS_NVIDIA_BLACKWELL = _PLATFORM.is_nvidia and _PLATFORM.is_blackwell

if _IS_NVIDIA_BLACKWELL:
    from tokenspeed_kernel.thirdparty.cute_dsl.qsa_sparse import (
        kernel as _cute_dsl_qsa_sparse_attention,
    )

_HEAD_DIM = 256
_SELECTED_WIDTH = 2051


def cute_dsl_blackwell_qsa_sparse_attention(
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
) -> torch.Tensor:
    """Run Blackwell QSA with exact query-prefix and KV operand reuse.

    Args:
        q: BF16 query tensor shaped [tokens, query_heads, 256], with 6,
            12 or 24 query heads, divisible by the KV-head count.
        k_cache: BF16 or FP8 E4M3 key cache shaped [slots, kv_heads, 256],
            with 1, 2 or 4 KV heads.
        v_cache: Value cache with the same shape and dtype as k_cache.
        selected_slots: Int32 physical slots shaped [tokens, 2051];
            non-positive entries are masked.
        scale: Softmax scale applied to query-key scores.
        max_seqlen_q: Uniform query-token count per request, including
            speculative decode. Query groups never cross request boundaries.
        metadata_capacity_rows: Capacity for stateful fallback plans, unused
            by this implementation's call-owned prefix flags.
        k_scale: Scalar key-cache descale, folded into scale.
        v_scale: Scalar value-cache descale, applied to the output.

    Returns:
        BF16 attention output with q's shape.

    Twelve BF16 query heads share one sixteen-head tile. Larger batches group
    consecutive queries only when their complete ordered 2048-entry prefixes
    match; each query keeps separate softmax state and three private tail slots.
    A distinct prefix retains one query per CTA. The measured head geometry
    selects query-group size, operand stages and warp/register allocation.

    Two-query groups and six-head geometries detect equality inside attention.
    Larger twelve-head groups refresh prefix flags in one preparation kernel
    before attention. Both launches are included in full-call benchmarks. KV
    data is gathered directly from the cache with 64-bit offsets; preparation
    neither copies KV nor caches decisions across calls or graph replays.
    FP8 retains its gather/conversion and split-cluster path.
    """

    del metadata_capacity_rows  # This backend does not use fallback plan storage.
    return _cute_dsl_qsa_sparse_attention(
        q,
        k_cache,
        v_cache,
        selected_slots,
        scale=scale,
        max_seqlen_q=max_seqlen_q,
        k_scale=k_scale,
        v_scale=v_scale,
        enable_pdl=pdl_enabled(),
    )


if _IS_NVIDIA_BLACKWELL:
    register_kernel(
        "attention",
        "qsa_sparse_attention",
        name="cute_dsl_blackwell_qsa_sparse_attention",
        solution="cute_dsl",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 7),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    k_cache=dense_tensor_format(torch.float8_e4m3fn),
                    v_cache=dense_tensor_format(torch.float8_e4m3fn),
                ),
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    k_cache=dense_tensor_format(torch.bfloat16),
                    v_cache=dense_tensor_format(torch.bfloat16),
                ),
            }
        ),
        traits={
            "num_q_heads": frozenset({6, 12, 24}),
            "num_kv_heads": frozenset({1, 2, 4}),
            "head_dim": frozenset({_HEAD_DIM}),
            "value_head_dim": frozenset({_HEAD_DIM}),
            "selected_width": frozenset({_SELECTED_WIDTH}),
            "is_decode": frozenset({True}),
        },
        priority=Priority.SPECIALIZED + 2,
    )(cute_dsl_blackwell_qsa_sparse_attention)
    __all__ = ["cute_dsl_blackwell_qsa_sparse_attention"]
else:
    __all__ = []
