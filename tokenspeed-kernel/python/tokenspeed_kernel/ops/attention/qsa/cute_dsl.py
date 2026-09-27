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
    """Run the adaptive workspace-free B200/B300 QSA specialization.

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
    of up to eight heads. A CTA or split cluster handles consecutive query
    rows for one KV/head tile. Every row retains its own selected slots and
    softmax state; query grouping reuses the CTA workspace and TMEM allocation.
    It does not widen the per-query MMA atom.

    The launch policy counts independent query/head outputs and queries the
    device SM count before jointly choosing query rows per CTA, KV splits and
    operand stages. Large BF16 launches use one split and two asynchronous
    K-or-V slots; sufficiently large outputs group two or four query rows per
    CTA. Intermediate BF16 launches use two or four splits with two slots.
    Smaller launches retain the sixteen/eight/four split policy, subject to
    the device's wide-cluster capacity. Actual compiled-kernel occupancy must
    be queried separately from these measured scheduling thresholds.

    Each BF16 operand slot holds 128x256 values in shared memory. Asynchronous
    barrier arrivals publish copy completion while the producer can fill the
    next free slot. An unsplit CTA merges each head and its three tail slots
    within one warp; split clusters use the existing head-owning DSM combine.
    FP8 retains the gather/conversion pipeline with operands in TMEM.
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
