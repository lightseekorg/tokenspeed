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

"""FlashInfer persistent KDA prefill on SM100/SM103."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.kda import KdaPrefillResult
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.thirdparty.flashinfer.kda import persistent_kda_prefill

_SUPPORTED_ARCHES = frozenset({ArchVersion(10, 0), ArchVersion(10, 3)})


def flashinfer_kda_supported() -> bool:
    """Whether this device runs FlashInfer's persistent KDA prefill kernel."""
    platform = current_platform()
    return platform.is_nvidia and platform.arch_version in _SUPPORTED_ARCHES


@register_kernel(
    "attention",
    "kda_paged_prefill",
    name="flashinfer_kda_paged_prefill",
    solution="flashinfer",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=format_signatures(("q", "k", "v"), "dense", {torch.bfloat16}),
    priority=Priority.SPECIALIZED,
    traits={"recurrent_layout": frozenset({"v_major"})},
)
def flashinfer_kda_paged_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor,
    lower_bound: float | None,
) -> KdaPrefillResult:
    """Run the shared prefill contract with native V-major FP32 state.

    Arguments follow kda_paged_prefill. The launch is planned from the host
    mirror ``cu_seqlens_cpu``; the device boundaries are not read. The kernel
    advances ``initial_state`` (the prefill state gather's private FP32 copy)
    in place and returns it as the final state.
    """
    return KdaPrefillResult(
        *persistent_kda_prefill(
            q,
            k,
            v,
            g_raw,
            beta_logits,
            A_log,
            dt_bias,
            initial_state=initial_state,
            cu_seqlens_cpu=cu_seqlens_cpu,
            lower_bound=lower_bound,
        )
    )
