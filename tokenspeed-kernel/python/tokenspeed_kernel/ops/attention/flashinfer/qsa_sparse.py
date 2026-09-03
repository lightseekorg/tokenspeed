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

"""FlashInfer FA2 sparse-attention registration for the QSA decode shape."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.triton.qsa_sparse import (
    prepare_qsa_sparse_indices,
)
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement, pdl_enabled
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from tokenspeed_kernel.thirdparty.flashinfer.qsa_sparse import (
    get_flashinfer_qsa_sparse_runner,
)


@register_kernel(
    "attention",
    "qsa_sparse_attention",
    name="flashinfer_fa2_qsa_sparse_attention",
    solution="flashinfer",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=frozenset(
        {
            format_signature(
                q=dense_tensor_format(torch.bfloat16),
                k_cache=dense_tensor_format(torch.float8_e4m3fn),
                v_cache=dense_tensor_format(torch.float8_e4m3fn),
            )
        }
    ),
    traits={
        "head_dim": frozenset({256}),
        "value_head_dim": frozenset({256}),
        "num_q_heads": frozenset({6}),
        "num_kv_heads": frozenset({1}),
        "selected_width": frozenset({2051}),
    },
    priority=Priority.SPECIALIZED,
    tags={"latency", "blackwell", "sparse"},
)
def flashinfer_fa2_qsa_sparse_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    *,
    scale: float,
    max_seqlen_q: int = 1,
    k_scale: float | torch.Tensor | None = None,
    v_scale: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Run QSA with a cached FlashInfer FA2 plan and dynamic packed mask."""

    del max_seqlen_q  # FA2 currently represents each packed query as one row.

    runner = get_flashinfer_qsa_sparse_runner(q.device)
    plan = runner.plan(
        q,
        k_cache,
        v_cache,
        selected_slots.shape[1],
        softmax_scale=scale,
    )
    use_pdl = pdl_enabled()
    prepare_qsa_sparse_indices(
        selected_slots,
        plan.indices,
        plan.packed_mask,
        enable_pdl=use_pdl,
    )
    return runner.run(
        plan,
        q,
        k_cache,
        v_cache,
        k_scale=k_scale,
        v_scale=v_scale,
        enable_pdl=use_pdl,
    )


__all__ = ["flashinfer_fa2_qsa_sparse_attention"]
