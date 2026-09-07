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

"""CuTe DSL registration for the B200 QSA sparse-attention specialization."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from tokenspeed_kernel.thirdparty.cute_dsl.qsa_sparse import (
    kernel as _cute_dsl_qsa_sparse_attention,
)

_HEAD_DIM = 256
_SELECTED_WIDTH = 2051


@register_kernel(
    "attention",
    "qsa_sparse_attention",
    name="cute_dsl_blackwell_qsa_sparse_attention",
    solution="cute_dsl",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 0),
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
        "head_dim": frozenset({_HEAD_DIM}),
        "value_head_dim": frozenset({_HEAD_DIM}),
        "num_q_heads": frozenset({6, 12, 24}),
        "num_kv_heads": frozenset({1, 2, 4}),
        "selected_width": frozenset({_SELECTED_WIDTH}),
    },
    priority=Priority.SPECIALIZED + 2,
    tags={"latency", "blackwell", "sparse", "cluster"},
)
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
    """Run the adaptive workspace-free B200 QSA specialization.

    Args:
        q: BF16 query tensor shaped ``[tokens, query_heads, 256]``, with 6,
            12 or 24 query heads. The count must be divisible by ``kv_heads``.
        k_cache: BF16 or FP8 E4M3 key cache shaped
            ``[cache_slots, kv_heads, 256]``, with 1, 2 or 4 KV heads.
        v_cache: BF16 or FP8 E4M3 value cache shaped
            ``[cache_slots, kv_heads, 256]``. Its dtype must match ``k_cache``.
        selected_slots: Physical cache slots shaped ``[tokens, 2051]``;
            non-positive values are ignored.
        scale: Softmax scale applied to query-key scores.
        max_seqlen_q: Uniform query-token count per request; 1 for decode and
            ``spec_num_tokens`` for compact speculative decode.
        metadata_capacity_rows: Ignored because this implementation is
            workspace-free.
        k_scale: Scalar key-cache descale, folded into ``scale``.
        v_scale: Scalar value-cache descale, applied to the output.

    Returns:
        BF16 attention output shaped ``[tokens, query_heads, 256]``.

    Each KV head owns a contiguous group of query heads, divided into tiles
    of up to eight heads. One CTA cluster handles each (row, KV head, head
    tile), sharing that row's selected slots. Head counts and cache strides
    are compile-time parameters; a six-query-head, one-KV-head launch retains
    its original tile geometry. This also covers the twelve- and twenty-four-
    query-head shapes used with smaller tensor-parallel sizes.

    Small launches use sixteen sequence-split CTAs when the device occupancy
    probe allows all clusters to fit in one wave, otherwise eight CTAs for up
    to eight clusters and four for larger launches. All split counts use the
    same asynchronous K/V ring and head-owning DSM softmax combine. The final
    three selected entries are computed in that combine, avoiding a mostly
    empty tensor-core tile. BF16 V staging uses transposed matrix loads.
    """

    del metadata_capacity_rows  # The workspace-free specialization has no metadata.
    return _cute_dsl_qsa_sparse_attention(
        q,
        k_cache,
        v_cache,
        selected_slots,
        scale=scale,
        max_seqlen_q=max_seqlen_q,
        k_scale=k_scale,
        v_scale=v_scale,
    )


__all__ = ["cute_dsl_blackwell_qsa_sparse_attention"]
