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

"""CUDA DeepSeek V4 attention kernels."""

import importlib.util

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

platform = current_platform()

if platform.is_hopper_plus:
    from tokenspeed_kernel.ops.attention.mla.cuda import flash_mla_sparse_fwd

try:
    from tokenspeed_kernel.thirdparty.cuda.dsv4_attention import (
        has_persistent_topk,
        persistent_topk,
    )
except ImportError:

    def has_persistent_topk() -> bool:
        return False

    persistent_topk = error_fn


if (
    platform.is_nvidia
    and platform.is_hopper_plus
    and importlib.util.find_spec("flash_mla") is not None
):

    @register_kernel(
        "attention",
        "dsv4_prefill",
        name="flashmla_dsv4_prefill",
        solution="flashmla",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    kv=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        traits={
            "num_q_heads": frozenset({64, 128}),
            "head_dim": frozenset({512}),
            "cache_layout": frozenset({"dense_workspace"}),
            "metadata_dtypes": frozenset({torch.int32}),
            "sinks": frozenset({True}),
        },
        priority=Priority.PERFORMANT,
    )
    def flashmla_dsv4_prefill(
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        lens: torch.Tensor,
        attn_sink: torch.Tensor,
        softmax_scale: float,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        result, _, _ = flash_mla_sparse_fwd(
            q=q,
            kv=kv.reshape(-1, 1, q.shape[-1]),
            indices=indices.unsqueeze(1),
            sm_scale=float(softmax_scale),
            attn_sink=attn_sink,
            topk_length=lens,
        )
        if out is not None:
            out.copy_(result)
            return out
        return result


__all__ = [
    "has_persistent_topk",
    "persistent_topk",
]
